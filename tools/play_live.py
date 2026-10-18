"""Live interactive evaluation: runs indefinitely until user closes the PyBoy window."""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

from src.battle import tactical_battle_step
from src.emulator import Emulator
from src.llm_supervisor import Supervisor
from src.macros import Macros
from src.mover import Mover, walkable
from src.ppo import load_policy
from src.ram_parser import RamParser, dialogue_open
from src.rl_env import ACTIONS, observe
from src.states import StateRegistry
from src.step import TileStepper
from src.tool_dispatch import execution_registry
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
        supervisor = Supervisor(client=None, registry=execution_registry(emu, reader, mover=mover, macros=macros))

        policy = None
        is_deep_policy = False
        if args.policy and os.path.exists(args.policy):
            if args.policy.endswith(".pt"):
                from src.deep_policy import SpatialVisionPolicy
                policy = SpatialVisionPolicy.load(args.policy, device="cpu")
                is_deep_policy = True
                print(f"Loaded 11.6M Spatial Vision Policy from {args.policy}")
            else:
                policy = load_policy(args.policy)
                if policy is not None:
                    print(f"Loaded policy from {args.policy} ({policy.__class__.__name__})")

        visited: set[tuple[int, int, int, int]] = set()
        seen_tiles: set[tuple[int, int, int, int]] = set()
        seen_maps: set[tuple[int, int]] = set()

        blocked_count = 0
        last_action = None
        step_idx = 0

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
                res = tactical_battle_step(state, macros)
                reason = getattr(res, "reason", "acted")
                print(f"[Step {step_idx}] Battle (mode={state.battle_mode}, enemy_species={state.enemy_species}, hp={state.enemy_hp}/{state.enemy_max_hp}): {reason}")
                for _ in range(10):
                    emu.tick(20, render=True)
                    emu.press("b", frames=6)
                step_idx += 1
                continue


            # 2. Dialogue Handling
            if dialogue_open(state):
                blocked_count = 0
                macros.mash_text(max_frames=120)
                step_idx += 1
                continue

            # 3. Check Triggers (Cut, etc.)
            trigger = evaluate(state, stuck_count=blocked_count)
            if trigger is not None and getattr(trigger.kind, "value", trigger.kind) == "CUT_BLOCKED" and state.party_can_cut:
                print(f"[Step {step_idx}] Trigger {trigger.kind}: Using Cut on tree...")
                macros.use_cut()
                step_idx += 1
                continue

            # 4. Overworld Movement via Policy
            obs = observe(state, visited)
            probs = None
            if policy is not None:
                if is_deep_policy:
                    screen = emu._pb.screen.ndarray[:, :, :3].copy()
                    probs = policy.probs(screen, obs).copy()
                elif hasattr(policy, "probs"):
                    probs = policy.probs(obs).copy()

            # Obstacle avoidance: if bumping into obstacle, mask blocked directions
            if blocked_count >= 2 and probs is not None:
                tiles = getattr(state, "tiles", {})
                for act_idx, dir_name in enumerate(ACTIONS[:4]):
                    t = tiles.get(dir_name, 0xFF)
                    if not walkable(t, dir_name):
                        probs[act_idx] = 0.0
                total_p = probs.sum()
                if total_p > 1e-6:
                    probs /= total_p
                    action_idx = int(np.random.choice(len(probs), p=probs))
                else:
                    action_idx = int(np.random.choice(4))
            elif probs is not None:
                action_idx = int(np.random.choice(len(probs), p=probs))
            else:
                action_idx = int(np.random.choice(4))

            direction = ACTIONS[action_idx]

            # Execute step
            before_pos = (state.x, state.y)
            if direction == "a":
                emu.press("a", frames=8)
                emu.tick(12, render=True)
                moved = False
            else:
                res = mover.run_path([direction])
                moved = bool(res.ok)

            after_state = reader.parse(emu.frame)
            after_pos = (after_state.x, after_state.y)

            if after_pos == before_pos and direction != "a":
                blocked_count += 1
            else:
                blocked_count = 0

            step_idx += 1
            if step_idx % 25 == 0:
                print(
                    f"[Step {step_idx:4d}] Map: {after_state.map_group},{after_state.map_number} | "
                    f"Pos: ({after_state.x:2d}, {after_state.y:2d}) | "
                    f"Party: {after_state.party_count}/6 | Balls: {after_state.num_balls} | "
                    f"Action: {direction}"
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
