"""Live interactive evaluation: runs indefinitely until user closes the PyBoy window."""

from __future__ import annotations

import argparse
import collections
import os
import sys
import numpy as np

from src.battle import tactical_battle_step
from src.emulator import Emulator
from src.macros import Macros
from src.mover import Mover, walkable
from src.ppo import load_policy
from src.ram_parser import RamParser, dialogue_open
from src.rl_env import ACTIONS, observe
from src.states import StateRegistry
from src.step import TileStepper
from src.triggers import evaluate

_DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def equip_balls(emu: Emulator, count: int = 10) -> None:
    """Give player Pokéballs if currently holding none."""
    try:
        b1, a1 = emu.symbol("wNumBalls")
        b2, a2 = emu.symbol("wBalls")
        current = emu.read(b1, a1, 1)[0]
        if current == 0:
            emu.write(b1, a1, bytes([1]))
            emu.write(b2, a2, bytes([5, count, 255]))
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Live continuous PyBoy game player")
    parser.add_argument("--state", default="route29_entry", help="save state name")
    parser.add_argument("--policy", default="logs/deep_policy.pt", help="path to policy (.pt or .npz)")
    parser.add_argument(
        "--speed",
        type=int,
        default=int(os.environ.get("PYBOY_SPEED", "4")),
        help="emulation speed multiplier (default 4)",
    )
    parser.add_argument(
        "--balls",
        type=int,
        default=10,
        help="number of Pokéballs to equip if empty (default 10)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="use greedy argmax action selection instead of stochastic sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="action sampling temperature (default 0.5)",
    )
    args = parser.parse_args(argv)

    print(f"Starting live play from '{args.state}' at {args.speed}x speed.")
    print("Game will run continuously until you close the PyBoy window.\n")

    emu = None
    try:
        data = StateRegistry().load_bytes(args.state)
        emu = Emulator(headless=False, speed=args.speed, rtc_lock=True)
        emu.load_state(data)

        if args.balls > 0:
            equip_balls(emu, args.balls)

        reader = RamParser(emu)
        macros = Macros(emu, reader)
        mover = Mover(emu, reader)
        stepper = TileStepper(emu, reader)

        policy = None
        is_deep_policy = False
        if args.policy and os.path.exists(args.policy):
            if args.policy.endswith(".pt"):
                from src.deep_policy import SpatialVisionPolicy
                policy = SpatialVisionPolicy.load(args.policy, device="cpu")
                is_deep_policy = True
                print(
                    f"Loaded {policy.param_count:,} parameter SpatialVisionPolicy "
                    f"(recurrent={getattr(policy, 'use_gru', False)}) from {args.policy}"
                )
            else:
                policy = load_policy(args.policy)
                if policy is not None:
                    print(f"Loaded policy from {args.policy} ({policy.__class__.__name__})")

        visited: set[tuple[int, int, int, int]] = set()
        seen_tiles: set[tuple[int, int, int, int]] = set()
        seen_maps: set[tuple[int, int]] = set()
        tile_visit_counts: dict[tuple[int, int, int, int], int] = {}

        blocked_count = 0
        last_action = None
        step_idx = 0
        recurrent_hx = None
        consecutive_a_count = 0
        recent_positions: collections.deque[tuple[int, int]] = collections.deque(maxlen=8)

        while True:
            # Check if PyBoy window was closed
            if not emu.tick(1, render=True):
                print("\nPyBoy window closed by user. Exiting cleanly.")
                break

            state = reader.parse(emu.frame)
            tile = (state.map_group, state.map_number, state.x, state.y)
            visited.add(tile)
            seen_tiles.add(tile)
            seen_maps.add((state.map_group, state.map_number))

            # 1. Battle Handling
            if state.battle_mode != 0:
                blocked_count = 0
                recurrent_hx = None
                res = tactical_battle_step(state, macros)
                reason = getattr(res, "reason", "acted")
                print(
                    f"[Step {step_idx}] Battle (mode={state.battle_mode}, "
                    f"enemy_species={state.enemy_species}, hp={state.enemy_hp}/{state.enemy_max_hp}): {reason}",
                    flush=True,
                )
                for _ in range(10):
                    emu.tick(20, render=True)
                    emu.press("b", frames=6)
                step_idx += 1
                continue


            # 2. Dialogue Handling
            if dialogue_open(state):
                blocked_count = 0
                recurrent_hx = None
                macros.mash_text(max_frames=120)
                step_idx += 1
                continue

            # 3. Check Triggers (Cut, Stuck, etc.)
            trigger = evaluate(state, stuck_count=blocked_count)
            if trigger is not None or blocked_count >= 4:
                recurrent_hx = None
                kind_str = getattr(getattr(trigger, "kind", None), "value", str(getattr(trigger, "kind", "")))
                if kind_str == "CUT_BLOCKED" and state.party_can_cut:
                    print(f"[Step {step_idx}] Trigger {trigger.kind}: Using Cut on tree...", flush=True)
                    macros.use_cut()
                    step_idx += 1
                    continue
                elif kind_str == "STUCK" or blocked_count >= 4:
                    tiles = getattr(state, "tiles", {})
                    # Find all walkable directions and sort by fewest visits
                    candidates = []
                    for d in ("left", "right", "up", "down"):
                        if walkable(tiles.get(d, 0xFF), d):
                            dx, dy = _DELTA[d]
                            target = (state.map_group, state.map_number, state.x + dx, state.y + dy)
                            candidates.append((tile_visit_counts.get(target, 0), d))
                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        escape_dir = candidates[0][1]
                        print(
                            f"[Step {step_idx}] Stuck at {state.pos}! Escaping via '{escape_dir}' into open frontier...",
                            flush=True,
                        )
                        res = mover.run_path([escape_dir])
                        if not res.ok:
                            dx, dy = _DELTA[escape_dir]
                            target = (state.map_group, state.map_number, state.x + dx, state.y + dy)
                            tile_visit_counts[target] = tile_visit_counts.get(target, 0) + 50
                    else:
                        emu.press("b", frames=8)
                        emu.tick(16, render=True)
                    blocked_count = 0
                    step_idx += 1
                    continue

            # 4. Overworld Movement via Policy
            obs = observe(state, visited)
            probs = None
            if policy is not None:
                if is_deep_policy:
                    screen = emu._pb.screen.ndarray[:, :, :3].copy()
                    if getattr(policy, "use_gru", False):
                        probs, recurrent_hx = policy.probs_recurrent(screen, obs, hx=recurrent_hx)
                        probs = probs.copy()
                    else:
                        probs = policy.probs(screen, obs).copy()
                elif hasattr(policy, "probs"):
                    probs = policy.probs(obs).copy()

            if probs is not None:
                tiles = getattr(state, "tiles", {})
                OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
                dir_probs = probs[:4].copy()

                # A. Always mask blocked / non-walkable directions (trees, water, fences)
                for act_idx, dir_name in enumerate(ACTIONS[:4]):
                    t = tiles.get(dir_name, 0xFF)
                    if not walkable(t, dir_name):
                        dir_probs[act_idx] = 0.0

                # B. Novelty-biased exploration: penalize revisiting already-explored tiles
                cur_tile = (state.map_group, state.map_number, state.x, state.y)
                tile_visit_counts[cur_tile] = tile_visit_counts.get(cur_tile, 0) + 1
                for act_idx, dir_name in enumerate(ACTIONS[:4]):
                    dx, dy = _DELTA[dir_name]
                    target = (state.map_group, state.map_number, state.x + dx, state.y + dy)
                    visits = tile_visit_counts.get(target, 0)
                    if visits > 0:
                        dir_probs[act_idx] *= (1.0 / (1.0 + 0.4 * visits))

                # C. Anti-backtracking: reduce immediate 180-degree turnarounds
                if last_action and last_action in OPPOSITE:
                    opp_dir = OPPOSITE[last_action]
                    if opp_dir in ACTIONS[:4]:
                        opp_idx = ACTIONS.index(opp_dir)
                        dir_probs[opp_idx] *= 0.4

                # D. Goal-Directed Navigation Compass: prioritize forward progress towards Falkner/Gym
                from src.rl_env import navigation_potential
                cur_phi = navigation_potential(state)
                for act_idx, dir_name in enumerate(ACTIONS[:4]):
                    if dir_probs[act_idx] > 0:
                        dx, dy = _DELTA[dir_name]
                        try:
                            target_state = type(state)(**{**state.__dict__, "x": state.x + dx, "y": state.y + dy})
                            target_phi = navigation_potential(target_state)
                            delta_phi = target_phi - cur_phi
                            if delta_phi > 0:
                                dir_probs[act_idx] *= (1.0 + 4.0 * delta_phi)
                            elif delta_phi < 0:
                                cur_visits = tile_visit_counts.get(cur_tile, 0)
                                penalty_floor = 0.5 if cur_visits >= 3 else 0.05
                                dir_probs[act_idx] *= max(penalty_floor, 1.0 + 2.0 * delta_phi)
                        except Exception:
                            pass

                total_p = dir_probs.sum()
                if total_p > 1e-6:
                    dir_probs /= total_p
                    if args.greedy:
                        action_idx = int(np.argmax(dir_probs))
                    else:
                        logits = np.log(dir_probs + 1e-10) / max(0.01, args.temperature)
                        logits -= logits.max()
                        p_scaled = np.exp(logits)
                        p_scaled /= p_scaled.sum()
                        action_idx = int(np.random.choice(len(p_scaled), p=p_scaled))
                else:
                    action_idx = int(np.random.choice(4))
            else:
                action_idx = int(np.random.choice(4))

            direction = ACTIONS[action_idx]

            # Execute step
            before_pos = (state.x, state.y)
            if direction == "a":
                emu.press("a", frames=8)
                emu.tick(12, render=True)
                moved = False
                consecutive_a_count += 1
            else:
                res = mover.run_path([direction])
                moved = bool(res.ok)
                consecutive_a_count = 0

            after_state = reader.parse(emu.frame)
            after_pos = (after_state.x, after_state.y)

            recent_positions.append(after_pos)
            is_oscillating = (len(recent_positions) == 8 and len(set(recent_positions)) <= 2)

            # Check if player moved or if dialogue/battle engaged
            if direction == "a":
                if after_state.battle_mode != 0 or dialogue_open(after_state):
                    blocked_count = 0
                else:
                    blocked_count += 1
            elif after_pos == before_pos:
                blocked_count += 1
            elif is_oscillating:
                blocked_count = 4  # Break 2-tile oscillation cycle by triggering escape reflex!
            else:
                blocked_count = 0

            step_idx += 1
            if step_idx % 25 == 0:
                print(
                    f"[Step {step_idx:4d}] Map: {after_state.map_group},{after_state.map_number} | "
                    f"Pos: ({after_state.x:2d}, {after_state.y:2d}) | "
                    f"Party: {after_state.party_count}/6 | Balls: {after_state.num_balls} | "
                    f"Action: {direction}",
                    flush=True,
                )

        return 0
    except KeyboardInterrupt:
        print("\nSession stopped by user interrupt.")
        return 0
    except Exception as exc:
        print(f"Error during playback: {exc}")
        return 1
    finally:
        if emu is not None:
            emu.close()


if __name__ == "__main__":
    sys.exit(main())
