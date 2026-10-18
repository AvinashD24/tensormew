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
from src.rl_env import ACTIONS, observe, progress_reward
from src.states import StateRegistry
from src.step import TileStepper


def _worker_fn(pipe, seed: int, start_state_name: str, max_steps: int):
    """Child process running one headless PyBoy instance."""
    emu = None
    try:
        data = StateRegistry().load_bytes(start_state_name)
        emu = Emulator(headless=True, speed=0, rtc_lock=True)
        emu.load_state(data)

        # Equip 10 Pokéballs if empty so player can catch Pokémon
        try:
            b1, a1 = emu.symbol("wNumBalls")
            b2, a2 = emu.symbol("wBalls")
            if emu.read(b1, a1, 1)[0] == 0:
                emu.write(b1, a1, bytes([1]))
                emu.write(b2, a2, bytes([5, 10, 255]))
        except Exception:
            pass

        reader = RamParser(emu)
        stepper = TileStepper(emu, reader)
        macros = Macros(emu, reader)

        visited: set[tuple[int, int, int, int]] = set()
        seen_tiles: set[tuple[int, int, int, int]] = set()
        seen_maps: set[tuple[int, int]] = set()
        seen_events: set[int] = set()
        seen_waypoints: set[tuple[int, int]] = set()
        caught_counts: dict[int, int] = {}
        blocked_count = 0

        def get_obs():
            st = reader.parse(emu.frame)
            tile = (st.map_group, st.map_number, st.x, st.y)
            visited.add(tile)
            seen_tiles.add(tile)
            seen_maps.add((st.map_group, st.map_number))
            screen = emu._pb.screen.ndarray[:, :, :3].copy()
            ram_vec = observe(st, visited)
            return screen, ram_vec, st

        steps = 0

        while True:
            cmd, data_arg = pipe.recv()
            if cmd == "RESET":
                data_bytes = StateRegistry().load_bytes(start_state_name)
                emu.load_state(data_bytes)
                visited.clear()
                seen_tiles.clear()
                seen_maps.clear()
                seen_events.clear()
                seen_waypoints.clear()
                caught_counts.clear()
                blocked_count = 0
                steps = 0
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
                    moved = stepper.step(action_name)
                    if not moved:
                        blocked_count += 1
                        if blocked_count >= 4:
                            st_cur = reader.parse(emu.frame)
                            tiles = getattr(st_cur, "tiles", {})
                            walkables = [d for d in ("left", "right", "up", "down") if walkable(tiles.get(d, 0xFF), d)]
                            if walkables:
                                stepper.step(walkables[seed % len(walkables)])
                            blocked_count = 0
                    else:
                        blocked_count = 0

                after = reader.parse(emu.frame)
                step_rew, disc_tile, disc_map = progress_reward(
                    before,
                    after,
                    moved,
                    seen_tiles,
                    seen_maps,
                    caught_counts=caught_counts,
                    seen_events=seen_events,
                    seen_waypoints=seen_waypoints,
                )
                steps += 1
                done = steps >= max_steps

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
    """Manages parallel PyBoy worker processes."""

    def __init__(self, num_envs: int, start_state: str, max_steps: int):
        self.num_envs = num_envs
        self.pipes = []
        self.procs = []
        for i in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_worker_fn,
                args=(child_conn, i, start_state, max_steps),
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
    rollout_steps: int = 64,
    num_envs: int = 12,
    batch_size: int = 128,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.02,
    start_state: str = "route29_entry",
    save_path: str = "logs/deep_policy.pt",
    resume: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing SpatialVisionPolicy training on {device}...")

    policy = SpatialVisionPolicy(obs_dim=17, num_actions=5)
    if resume and os.path.exists(save_path):
        try:
            policy = SpatialVisionPolicy.load(save_path, device=device)
            print(f"Loaded existing checkpoint from {save_path}")
        except Exception as e:
            print(f"Warning: could not load checkpoint ({e}), initializing fresh policy")

    policy.to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    print(f"Policy ready with {policy.param_count:,} trainable parameters.")

    print(f"Spawning {num_envs} vectorized PyBoy workers from '{start_state}'...")
    vec_env = VectorEnv(num_envs=num_envs, start_state=start_state, max_steps=rollout_steps)
    screens_cur, rams_cur = vec_env.reset()

    best_reward = -float("inf")

    try:
        for update in range(updates):
            t0 = time.time()
            buf_screens, buf_rams, buf_actions = [], [], []
            buf_logps, buf_rewards, buf_dones, buf_values = [], [], [], []

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
                    logits, val = policy(screens_t, rams_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions_t = dist.sample()
                    logp_t = dist.log_prob(actions_t)

                actions_np = actions_t.cpu().numpy()
                screens_next, rams_next, rews, dones, map_counts, badges = vec_env.step(actions_np)

                buf_screens.append(screens_t.cpu())
                buf_rams.append(rams_t.cpu())
                buf_actions.append(actions_t.cpu())
                buf_logps.append(logp_t.cpu())
                buf_rewards.append(torch.from_numpy(rews))
                buf_dones.append(torch.from_numpy(dones))
                buf_values.append(val.cpu())

                screens_cur, rams_cur = screens_next, rams_next

            # 2. GAE computation
            b_screens = torch.cat(buf_screens, dim=0)  # (T*B, 3, 144, 160)
            b_rams = torch.cat(buf_rams, dim=0)  # (T*B, 17)
            b_actions = torch.cat(buf_actions, dim=0)  # (T*B,)
            b_logps = torch.cat(buf_logps, dim=0).to(device)  # (T*B,)

            rewards_mat = torch.stack(buf_rewards).to(device)  # (T, B)
            values_mat = torch.stack(buf_values).to(device)  # (T, B)
            dones_mat = torch.stack(buf_dones).to(device)  # (T, B)

            returns, advantages = compute_gae(rewards_mat, values_mat, dones_mat)
            b_returns = returns.view(-1)
            b_advantages = advantages.view(-1)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            mean_rew = float(rewards_mat.sum(dim=0).mean().item())
            total_samples = len(b_actions)

            # 3. PPO Epochs
            policy.train()
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

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                    optimizer.step()

            dt = time.time() - t0
            max_maps = max(map_counts)
            max_badges = max(badges)

            print(
                f"update={update:3d} | samples={total_samples} | "
                f"mean_rew={mean_rew:6.2f} | maps_disc={max_maps} | "
                f"badges={max_badges} | time={dt:4.1f}s | "
                f"loss={loss.item():.4f}",
                flush=True,
            )

            if mean_rew > best_reward or (update + 1) % 10 == 0:
                best_reward = max(best_reward, mean_rew)
                policy.save(save_path)

        print(f"\nTraining completed! Saved best model to {save_path}", flush=True)
    finally:
        vec_env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deep Visual PPO Training on CUDA")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--envs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--state", default="route29_entry")
    parser.add_argument("--save", default="logs/deep_policy.pt")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args(argv)

    train(
        updates=args.updates,
        rollout_steps=args.rollout,
        num_envs=args.envs,
        batch_size=args.batch_size,
        lr=args.lr,
        start_state=args.state,
        save_path=args.save,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
