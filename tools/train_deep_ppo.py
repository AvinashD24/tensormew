"""High-throughput visual PPO training using SpatialVisionPolicy (11.6M parameters) on CUDA.

Uses a vectorized worker architecture where 12 parallel PyBoy processes step the game
and stream raw screen buffers (144x160x3) + 17-dim RAM states to an RTX 4090 GPU for
batched tensor-core neural network inference and PPO updates.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.battle import tactical_battle_step
from src.deep_policy import SpatialVisionPolicy
from src.emulator import Emulator
from src.macros import Macros
from src.mover import walkable
from src.ram_parser import RamParser, dialogue_open
from src.rl_env import ACTIONS, StepInfo, observe, progress_reward, intrinsic_curiosity_reward
from src.states import StateRegistry
from src.step import TileStepper
from src.terminal_monitor import TerminalMonitor

CURRICULUM_STATES = [
    "route29_entry",
    "agent_map_24_7",
    "agent_map_26_3",
    "agent_map_26_1",
    "mystery_egg_received",
]

_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _worker_fn(pipe, seed: int, start_states: list[str], max_steps: int):
    """Child process running one headless PyBoy instance with curriculum resets and reflexive escape."""
    emu = None
    try:
        current_state_idx = seed % len(start_states)
        current_state_name = start_states[current_state_idx]
        data = StateRegistry().load_bytes(current_state_name)
        emu = Emulator(headless=True, speed=0, rtc_lock=True)
        emu.load_state(data)

        def equip_balls():
            try:
                b1, a1 = emu.symbol("wNumBalls")
                b2, a2 = emu.symbol("wBalls")
                if emu.read(b1, a1, 1)[0] == 0:
                    emu.write(b1, a1, bytes([1]))
                    emu.write(b2, a2, bytes([5, 10, 255]))
            except Exception:
                pass

        equip_balls()
        reader = RamParser(emu)
        stepper = TileStepper(emu, reader)
        macros = Macros(emu, reader)

        visited: set[tuple[int, int, int, int]] = set()
        seen_tiles: set[tuple[int, int, int, int]] = set()
        seen_maps: set[tuple[int, int]] = set()
        seen_events: set[int] = set()
        seen_waypoints: set[tuple[int, int]] = set()
        tile_visit_counts: dict[tuple[int, int, int, int], int] = {}
        caught_counts: dict[int, int] = {}
        blocked_count = 0
        consecutive_bumps = 0
        steps = 0
        last_dir = None

        def get_obs():
            st = reader.parse(emu.frame)
            tile = (st.map_group, st.map_number, st.x, st.y)
            visited.add(tile)
            seen_tiles.add(tile)
            seen_maps.add((st.map_group, st.map_number))
            tile_visit_counts[tile] = tile_visit_counts.get(tile, 0) + 1
            screen = emu._pb.screen.ndarray[:, :, :3].copy()
            ram_vec = observe(st, visited, include_goals=True)
            return screen, ram_vec, st

        def reset_environment():
            nonlocal current_state_idx, blocked_count, consecutive_bumps, steps, last_dir
            current_state_idx = (current_state_idx + 1) % len(start_states)
            state_to_load = start_states[current_state_idx]
            data_bytes = StateRegistry().load_bytes(state_to_load)
            emu.load_state(data_bytes)
            equip_balls()
            visited.clear()
            seen_tiles.clear()
            seen_maps.clear()
            seen_events.clear()
            seen_waypoints.clear()
            tile_visit_counts.clear()
            caught_counts.clear()
            blocked_count = 0
            consecutive_bumps = 0
            steps = 0
            last_dir = None

        while True:
            cmd, data_arg = pipe.recv()
            if cmd == "RESET":
                reset_environment()
                screen, ram_vec, _ = get_obs()
                pipe.send((screen, ram_vec))

            elif cmd == "STEP":
                act_idx = int(data_arg)
                before = reader.parse(emu.frame)
                action_name = ACTIONS[act_idx]
                moved = False

                # Handle in-game events
                if before.battle_mode != 0:
                    tactical_battle_step(before, macros)
                    blocked_count = 0
                elif dialogue_open(before):
                    macros.mash_text(max_frames=80)
                    blocked_count = 0
                elif action_name == "a":
                    emu.press("a", frames=8)
                    emu.tick(12, render=False)
                    moved = False
                else:
                    step_res = stepper.step(action_name)
                    moved = bool(step_res.moved)
                    if not moved:
                        blocked_count += 1
                        if blocked_count >= 4:
                            st_cur = reader.parse(emu.frame)
                            tiles = getattr(st_cur, "tiles", {})
                            candidates = []
                            for d in ("left", "right", "up", "down"):
                                if walkable(tiles.get(d, 0xFF), d):
                                    dx, dy = _DELTA[d]
                                    target = (st_cur.map_group, st_cur.map_number, st_cur.x + dx, st_cur.y + dy)
                                    candidates.append((tile_visit_counts.get(target, 0), d))
                            if candidates:
                                candidates.sort(key=lambda x: x[0])
                                escape_dir = candidates[0][1]
                                ok = stepper.step(escape_dir)
                                if not ok:
                                    dx, dy = _DELTA[escape_dir]
                                    target = (st_cur.map_group, st_cur.map_number, st_cur.x + dx, st_cur.y + dy)
                                    tile_visit_counts[target] = tile_visit_counts.get(target, 0) + 50
                            else:
                                emu.press("b", frames=8)
                                emu.tick(12, render=False)
                            blocked_count = 0
                    else:
                        blocked_count = 0

                if not moved and before.battle_mode == 0 and not dialogue_open(before):
                    consecutive_bumps += 1
                else:
                    consecutive_bumps = 0

                after = reader.parse(emu.frame)
                unvisited_neighbor = False
                if 0 <= act_idx < 4:
                    d_name = ACTIONS[act_idx]
                    dx, dy = _DELTA[d_name]
                    neighbor = (before.map_group, before.map_number, before.x + dx, before.y + dy)
                    unvisited_neighbor = neighbor not in seen_tiles

                info = StepInfo(consecutive_bumps=consecutive_bumps)
                step_rew, disc_tile, disc_map = progress_reward(
                    before,
                    after,
                    moved,
                    seen_tiles,
                    seen_maps,
                    unvisited_neighbor=unvisited_neighbor,
                    caught_counts=caught_counts,
                    seen_events=seen_events,
                    seen_waypoints=seen_waypoints,
                    info=info,
                )
                # 1. Intrinsic curiosity novelty reward: 1/sqrt(N)
                tile = (after.map_group, after.map_number, after.x, after.y)
                v_count = tile_visit_counts.get(tile, 0)
                step_rew += intrinsic_curiosity_reward(v_count, scale=0.08)

                # 2. Anti-looping pacing penalty if oscillating in a loop
                if v_count > 12:
                    step_rew -= 0.02 * min(4.0, (v_count - 10) / 5.0)

                # 3. Action smoothing & directional momentum bonus
                if action_name == last_dir and moved:
                    step_rew += 0.02
                last_dir = action_name if moved else None

                has_event = getattr(after, "has_event", None)
                gave_egg = callable(has_event) and has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
                quest_complete = gave_egg and after.map_id == (24, 5)
                if quest_complete:
                    step_rew += 50.0

                steps += 1
                done = steps >= max_steps or quest_complete
                if done:
                    reset_environment()

                screen, ram_vec, _ = get_obs()
                pipe.send(
                    (
                        screen,
                        ram_vec,
                        float(step_rew),
                        bool(done),
                        len(seen_maps),
                        after.badges,
                    )
                )

            elif cmd == "CLOSE":
                break
    finally:
        if emu is not None:
            emu.close()


class VectorEnv:
    """Manages parallel PyBoy worker processes with multi-state curriculum distribution."""

    def __init__(self, num_envs: int, start_states: list[str] | str, max_steps: int):
        self.num_envs = num_envs
        if isinstance(start_states, str):
            states_list = [s.strip() for s in start_states.split(",") if s.strip()]
        else:
            states_list = list(start_states)
        if not states_list:
            states_list = CURRICULUM_STATES

        self.pipes = []
        self.procs = []
        for i in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_worker_fn,
                args=(child_conn, i, states_list, max_steps),
                daemon=True,
            )
            p.start()
            self.pipes.append(parent_conn)
            self.procs.append(p)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        for pipe in self.pipes:
            pipe.send(("RESET", None))
        screens, rams = [], []
        for pipe in self.pipes:
            s, r = pipe.recv()
            screens.append(s)
            rams.append(r)
        return np.array(screens, dtype=np.uint8), np.array(rams, dtype=np.float64)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        for pipe, act in zip(self.pipes, actions):
            pipe.send(("STEP", act))
        screens, rams, rews, dones, maps, badges = [], [], [], [], [], []
        for pipe in self.pipes:
            s, r, rew, d, m, b = pipe.recv()
            screens.append(s)
            rams.append(r)
            rews.append(rew)
            dones.append(d)
            maps.append(m)
            badges.append(b)
        return (
            np.array(screens, dtype=np.uint8),
            np.array(rams, dtype=np.float64),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=bool),
            maps,
            badges,
        )

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("CLOSE", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2.0)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation on GPU."""
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        next_non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns, advantages


def train(
    updates: int = 50,
    rollout_steps: int = 128,
    num_envs: int = 12,
    batch_size: int = 128,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.02,
    start_state: str = "new_bark_outside",
    save_path: str = "logs/deep_policy.pt",
    resume: bool = True,
    use_gru: bool = True,
    curriculum: bool = False,
    states: list[str] | None = None,
    push_weights: bool = True,
    max_steps: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing SpatialVisionPolicy training on {device} (recurrent={use_gru})...")

    policy = SpatialVisionPolicy(obs_dim=21, num_actions=5, use_gru=use_gru, gru_hidden_dim=512)
    if resume and os.path.exists(save_path):
        try:
            policy = SpatialVisionPolicy.load(save_path, device=device, override_use_gru=use_gru, override_obs_dim=21)
            print(f"Loaded existing checkpoint from {save_path} (use_gru={policy.use_gru}, obs_dim={policy.obs_dim})")
        except Exception as e:
            print(f"Warning: could not load checkpoint ({e}), initializing fresh policy")

    policy.to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    print(f"Policy ready with {policy.param_count:,} trainable parameters.")

    if states is not None and len(states) > 0:
        states_to_use = states
    elif curriculum:
        states_to_use = CURRICULUM_STATES
    else:
        states_to_use = [start_state]

    ep_max_steps = max_steps if max_steps is not None else max(1024, rollout_steps * 8)
    print(f"Spawning {num_envs} vectorized PyBoy workers across states: {states_to_use} (max_steps={ep_max_steps})...")
    vec_env = VectorEnv(num_envs=num_envs, start_states=states_to_use, max_steps=ep_max_steps)
    screens_cur, rams_cur = vec_env.reset()

    best_reward = -float("inf")
    hx = torch.zeros(1, num_envs, 512, device=device) if policy.use_gru else None
    monitor = TerminalMonitor(total_updates=updates, jsonl_path="logs/training_metrics.jsonl")

    try:
        for update in range(updates):
            t0 = time.time()
            buf_screens, buf_rams, buf_actions = [], [], []
            buf_logps, buf_rewards, buf_dones, buf_values = [], [], [], []

            initial_hx = hx.detach().clone() if (policy.use_gru and hx is not None) else None

            # 1. Collect Rollouts with Batched GPU Inference
            policy.eval()
            for step in range(rollout_steps):
                # Format batch: (B, 144, 160, 3) -> (B, 3, 144, 160)
                screens_t = (
                    torch.from_numpy(screens_cur.transpose(0, 3, 1, 2))
                    .float()
                    .to(device)
                    .div_(255.0)
                )
                rams_t = torch.from_numpy(rams_cur).float().to(device)

                with torch.no_grad():
                    if policy.use_gru:
                        logits, val, next_hx = policy.forward_recurrent(screens_t, rams_t, hx)
                    else:
                        logits, val = policy(screens_t, rams_t)
                        next_hx = None

                    dist = torch.distributions.Categorical(logits=logits)
                    actions_t = dist.sample()
                    logp_t = dist.log_prob(actions_t)

                actions_np = actions_t.cpu().numpy()
                screens_next, rams_next, rews, dones, map_counts, badges = vec_env.step(actions_np)

                if policy.use_gru and next_hx is not None:
                    if any(dones):
                        done_mask = torch.from_numpy(dones).to(device)
                        next_hx[:, done_mask, :] = 0.0
                    hx = next_hx

                buf_screens.append(screens_t.cpu())
                buf_rams.append(rams_t.cpu())
                buf_actions.append(actions_t.cpu())
                buf_logps.append(logp_t.cpu())
                buf_rewards.append(torch.from_numpy(rews))
                buf_dones.append(torch.from_numpy(dones))
                buf_values.append(val.cpu())

                screens_cur, rams_cur = screens_next, rams_next

            # 2. GAE computation
            rewards_mat = torch.stack(buf_rewards).to(device)  # (T, B)
            values_mat = torch.stack(buf_values).to(device)    # (T, B)
            dones_mat = torch.stack(buf_dones).to(device)      # (T, B)

            returns, advantages = compute_gae(rewards_mat, values_mat, dones_mat)
            mean_rew = float(rewards_mat.sum(dim=0).mean().item())
            total_samples = len(buf_screens) * num_envs

            # 3. PPO Epochs
            policy.train()
            ep_pol_losses: list[float] = []
            ep_val_losses: list[float] = []
            ep_entropies: list[float] = []
            ep_kls: list[float] = []
            ep_clip_fracs: list[float] = []

            if policy.use_gru:
                # Recurrent Sequence Updates across (B, T)
                screens_seq = torch.stack(buf_screens, dim=1).to(device)   # (B, T, 3, 144, 160)
                rams_seq = torch.stack(buf_rams, dim=1).to(device)         # (B, T, 17)
                actions_seq = torch.stack(buf_actions, dim=1).to(device)   # (B, T)
                old_logps_seq = torch.stack(buf_logps, dim=1).to(device)   # (B, T)
                returns_seq = returns.transpose(0, 1).to(device)          # (B, T)
                adv_seq = advantages.transpose(0, 1).to(device)           # (B, T)
                adv_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)

                for epoch in range(4):
                    new_logp, entropy, val_pred = policy.evaluate_actions_sequence(
                        screens_seq, rams_seq, actions_seq, initial_hx
                    )
                    ratio = torch.exp(new_logp - old_logps_seq)
                    surr1 = ratio * adv_seq
                    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_seq
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = 0.5 * F.mse_loss(val_pred, returns_seq)
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + value_loss + entropy_coef * entropy_loss

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()
                        clip_frac = (torch.abs(ratio - 1.0) > clip_ratio).float().mean()
                        ep_pol_losses.append(policy_loss.item())
                        ep_val_losses.append(value_loss.item())
                        ep_entropies.append(entropy.mean().item())
                        ep_kls.append(approx_kl.item())
                        ep_clip_fracs.append(clip_frac.item())

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                    optimizer.step()

                if hx is not None:
                    hx = hx.detach()
            else:
                # Feedforward Shuffle Updates across flattened (T*B)
                b_screens = torch.cat(buf_screens, dim=0)  # (T*B, 3, 144, 160)
                b_rams = torch.cat(buf_rams, dim=0)        # (T*B, 17)
                b_actions = torch.cat(buf_actions, dim=0)  # (T*B,)
                b_logps = torch.cat(buf_logps, dim=0).to(device)  # (T*B,)
                b_returns = returns.view(-1)
                b_advantages = advantages.view(-1)
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                indices = np.arange(total_samples)
                for epoch in range(4):
                    np.random.shuffle(indices)
                    for start_idx in range(0, total_samples, batch_size):
                        batch_idx = indices[start_idx : start_idx + batch_size]
                        s_batch = b_screens[batch_idx].to(device)
                        r_batch = b_rams[batch_idx].to(device)
                        a_batch = b_actions[batch_idx].to(device)
                        old_logp = b_logps[batch_idx]
                        adv_batch = b_advantages[batch_idx]
                        ret_batch = b_returns[batch_idx]

                        new_logp, entropy, val_pred = policy.evaluate_actions(s_batch, r_batch, a_batch)

                        ratio = torch.exp(new_logp - old_logp)
                        surr1 = ratio * adv_batch
                        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_batch
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = 0.5 * F.mse_loss(val_pred, ret_batch)
                        entropy_loss = -entropy.mean()

                        loss = policy_loss + value_loss + entropy_coef * entropy_loss

                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()
                            clip_frac = (torch.abs(ratio - 1.0) > clip_ratio).float().mean()
                            ep_pol_losses.append(policy_loss.item())
                            ep_val_losses.append(value_loss.item())
                            ep_entropies.append(entropy.mean().item())
                            ep_kls.append(approx_kl.item())
                            ep_clip_fracs.append(clip_frac.item())

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                        optimizer.step()

            dt = time.time() - t0
            max_maps = max(map_counts)
            max_badges = max(badges)
            cur_lr = optimizer.param_groups[0]["lr"]

            mean_entropy = float(np.mean(ep_entropies)) if ep_entropies else 0.0
            mean_val_loss = float(np.mean(ep_val_losses)) if ep_val_losses else 0.0
            mean_pol_loss = float(np.mean(ep_pol_losses)) if ep_pol_losses else 0.0
            mean_kl = float(np.mean(ep_kls)) if ep_kls else 0.0
            mean_clip = float(np.mean(ep_clip_fracs)) if ep_clip_fracs else 0.0

            monitor.log_update(
                update=update,
                mean_reward=mean_rew,
                entropy=mean_entropy,
                value_loss=mean_val_loss,
                policy_loss=mean_pol_loss,
                approx_kl=mean_kl,
                clip_frac=mean_clip,
                lr=cur_lr,
                maps_discovered=max_maps,
                badges=max_badges,
                duration_sec=dt,
                samples=total_samples,
            )

            if mean_rew > best_reward or (update + 1) % 10 == 0:
                best_reward = max(best_reward, mean_rew)
                policy.save(save_path)

        print(f"\nTraining completed! Saved best model to {save_path}", flush=True)
        if push_weights:
            try:
                from tools.sync_weights import sync_and_push
                sync_and_push(push=True, message=f"feat(weights): deep_policy.pt retrained (updates={updates}, rew={best_reward:.2f})")
            except Exception as e:
                print(f"[Warning] Could not auto-push weights: {e}", flush=True)
    finally:
        vec_env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deep Visual PPO Training on CUDA")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1024)
    default_envs = 16 if torch.cuda.is_available() else 8
    default_batch_size = 256 if torch.cuda.is_available() else 128
    parser.add_argument("--envs", type=int, default=default_envs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--state", default="new_bark_outside")
    parser.add_argument("--states", default="", help="comma-separated list of curriculum states")
    parser.add_argument("--curriculum", action="store_true", default=False, help="use multi-state curriculum resets")
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum")
    parser.add_argument("--recurrent", action="store_true", default=True, help="use recurrent GRU temporal memory (13.2M model)")
    parser.add_argument("--no-recurrent", action="store_false", dest="recurrent")
    parser.add_argument("--save", default="logs/deep_policy.pt")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--push-weights", action="store_true", default=True, help="auto-push updated weights to GitHub upon completion")
    parser.add_argument("--no-push-weights", action="store_false", dest="push_weights")
    args = parser.parse_args(argv)

    states_list = [s.strip() for s in args.states.split(",") if s.strip()] if args.states else None

    train(
        updates=args.updates,
        rollout_steps=args.rollout,
        num_envs=args.envs,
        batch_size=args.batch_size,
        lr=args.lr,
        start_state=args.state,
        save_path=args.save,
        resume=args.resume,
        use_gru=args.recurrent,
        curriculum=args.curriculum,
        states=states_list,
        push_weights=args.push_weights,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
