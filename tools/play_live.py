"""Live interactive evaluation: runs indefinitely until user closes the PyBoy window.

The policy plays everything it controls in training: movement, A / B, and the WHOLE battle (menus, moves, items,
switching). Only overworld dialogue is auto-advanced (``mash_text``), exactly as in training. There is no route
compass, no scripted navigator and no scripted battle any more; the only assists left are the reflexive unstick /
Cut / LLM-director helpers for when the agent is physically wedged.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import numpy as np

from src.collision import is_ledge
from src.emulator import Emulator
from src.macros import Macros
from src.unstick import Unsticker
from src.mover import Mover, walkable
from src.ppo import load_policy
from src.ram_parser import RamParser, dialogue_open
from src.local_nav import LocalNav
from src.battle_obs import equip_supplies
from src.rl_env import ACTIONS, BUTTON_ACTIONS, StepInfo, observe, progress_reward
from src.states import StateRegistry
from src.step import TileStepper
from src.live_status import (
    MilestoneTracker,
    checklist_line,
    milestone_line,
    seed_seen_events,
    story_snapshot,
    write_status,
)
from src.game_data import get_map_name, get_party_moveset, get_species_name
from src.tool_dispatch import execution_registry
from src.triggers import evaluate

_DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def equip_balls(emu: Emulator, count: int = 10) -> None:
    """Give the player Poke Balls and 5 Potions (the same supplies as training) if the bag is short."""
    equip_supplies(emu, balls=count)


def _write_gif(frames, path, durations=None) -> None:
    if frames:
        frames[0].save(
            path, save_all=True, append_images=frames[1:], loop=0,
            duration=list(durations) if durations else 90,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Live continuous PyBoy game player")
    parser.add_argument("--state", default="route29_entry", help="save state name")
    parser.add_argument("--policy", default="logs/deep_policy.pt", help="path to policy (.pt or .npz)")
    parser.add_argument(
        "--record-dir",
        default=None,
        help="save a run.gif (and a .state per story milestone) here; needs a window, headless frames are blank",
    )
    parser.add_argument("--record-every", type=int, default=6, help="steps between recorded GIF frames")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="draw every frame and pause on battle text so viewers can follow (use with --speed 2)",
    )
    parser.add_argument(
        "--record-frames-every",
        type=int,
        default=0,
        help="with --watch and --record-dir: also record a smooth GIF frame every N game frames in Violet City/the gym",
    )
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run headless without opening an SDL window for maximum throughput (~5,000 FPS)",
    )
    parser.add_argument(
        "--tag",
        default=f"p{os.getpid()}",
        help="name for this run in logs/live_status (used by tools.live_status)",
    )
    args = parser.parse_args(argv)

    print(f"Starting live play from '{args.state}' at {args.speed}x speed (headless={args.headless}).")
    if not args.headless:
        print("Game will run continuously until you close the PyBoy window.\n")
    else:
        print("Running in headless high-throughput mode. Press Ctrl+C to exit.\n")

    emu = None
    try:
        if args.state.endswith(".state") and os.path.isfile(args.state):
            with open(args.state, "rb") as fh:  # resume from a recorded milestone file
                data = fh.read()
        else:
            data = StateRegistry().load_bytes(args.state)
        emu = Emulator(headless=args.headless, speed=args.speed, rtc_lock=True)
        emu.load_state(data)

        if args.balls > 0:
            equip_balls(emu, args.balls)

        reader = RamParser(emu)
        macros = Macros(emu, reader)
        mover = Mover(emu, reader)
        stepper = TileStepper(emu, reader)

        supervisor = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from src.llm_supervisor import Supervisor
                registry = execution_registry(emu, reader, mover=mover, macros=macros)
                supervisor = Supervisor(registry=registry)
                print(
                    f"LLM Supervisor ACTIVE ({supervisor.model}) — will provide strategic steering on bottlenecks."
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM Supervisor: {e}")

        policy = None
        is_deep_policy = False
        policy_path = args.policy
        if policy_path and not os.path.exists(policy_path):
            fallback = os.path.join("models", os.path.basename(policy_path))
            if os.path.exists(fallback):
                policy_path = fallback

        if policy_path and os.path.exists(policy_path):
            if policy_path.endswith(".pt"):
                from src.deep_policy import SpatialVisionPolicy
                policy = SpatialVisionPolicy.load(policy_path, device="cpu", check_schema=True)
                is_deep_policy = True
                print(
                    f"Loaded {policy.param_count:,} parameter SpatialVisionPolicy "
                    f"(recurrent={getattr(policy, 'use_gru', False)}) from {policy_path}"
                )
            else:
                policy = load_policy(policy_path)
                if policy is not None:
                    print(f"Loaded policy from {policy_path} ({policy.__class__.__name__})")

        active_policy = policy

        visited: set[tuple[int, int, int, int]] = set()
        seen_tiles: set[tuple[int, int, int, int]] = set()
        seen_maps: set[tuple[int, int]] = set()
        seen_events: set[int] = set()
        seen_waypoints: set[tuple[int, int]] = set()
        caught_counts: dict[int, int] = {}
        tile_visit_counts: dict[tuple[int, int, int, int], int] = {}
        cumulative_reward: float = 0.0

        blocked_count = 0
        nav = LocalNav()
        last_action = None
        last_heading = None
        step_idx = 0
        recurrent_hx = None
        consecutive_a_count = 0
        recent_positions: collections.deque[tuple[int, int]] = collections.deque(maxlen=8)
        render_screen = not args.headless

        last_story_flags = None
        milestones = MilestoneTracker()
        unsticker = Unsticker()
        last_status_write = 0.0
        record_dir = None
        rec_frames: list = []
        rec_durations: list = []
        last_rec_step = -args.record_every
        rec_ctl = {"detail": False, "n": 0}   # detail: smooth capture while in Violet City / the gym
        if args.watch:
            emu.always_render = True
        if args.record_dir:
            from pathlib import Path
            record_dir = Path(args.record_dir)
            record_dir.mkdir(parents=True, exist_ok=True)
            print(f"Recording to {record_dir} (GIF frame every {args.record_every} steps, state saved per milestone)")
            if args.watch and args.record_frames_every > 0:
                frame_ms = max(1, round(args.record_frames_every * 1000 / 60))

                def _record_game_frame():
                    if not rec_ctl["detail"]:
                        return
                    rec_ctl["n"] += 1
                    if rec_ctl["n"] % args.record_frames_every == 0:
                        rec_frames.append(emu._pb.screen.image.convert("RGB").quantize(64))
                        rec_durations.append(frame_ms)
                        if len(rec_frames) % 500 == 0:
                            _write_gif(rec_frames, record_dir / "run.gif", rec_durations)

                emu.frame_hook = _record_game_frame
                print(f"Smooth recording: a frame every {args.record_frames_every} game frames in Violet City / the gym")
        seed_seen_events(reader.parse(emu.frame), seen_events)
        while True:
            # Check if PyBoy window was closed
            if not emu.tick(1, render=render_screen):
                print("\nPyBoy window closed by user. Exiting cleanly.")
                break

            state = reader.parse(emu.frame)
            rec_ctl["detail"] = state.map_group == 10   # Violet City, gym and Pokémon Center
            if record_dir is not None and not (emu.frame_hook and rec_ctl["detail"]) and step_idx - last_rec_step >= args.record_every:
                last_rec_step = step_idx
                rec_frames.append(emu._pb.screen.image.convert("RGB").quantize(64))
                rec_durations.append(90)
                if len(rec_frames) % 100 == 0:
                    _write_gif(rec_frames, record_dir / "run.gif", rec_durations)
            if time.time() - last_status_write > 3.0:
                write_status(args.tag, state, story_snapshot(state, milestones), step_idx, cumulative_reward)
                last_status_write = time.time()
            for _key in milestones.update(state, step_idx):
                print(milestone_line(_key, step_idx, milestones), flush=True)
                write_status(args.tag, state, story_snapshot(state, milestones), step_idx, cumulative_reward)
                if record_dir is not None:
                    emu.save_state(record_dir / f"{_key}.state")
                    if _key == "falkner_beaten":
                        _write_gif(rec_frames, record_dir / "run.gif", rec_durations)
                        print(f"Falkner beaten: recording saved to {record_dir / 'run.gif'}", flush=True)
            tile = (state.map_group, state.map_number, state.x, state.y)
            visited.add(tile)
            seen_tiles.add(tile)
            seen_maps.add((state.map_group, state.map_number))

            # 1. Battle: the POLICY plays it, button by button (menus, moves, items, switching). Nothing is scripted.
            if state.battle_mode != 0:
                blocked_count = 0
                if is_deep_policy and active_policy is not None:
                    from tools.evaluate_policy import obs_vector
                    obs = obs_vector(state, visited, int(getattr(active_policy, "obs_dim", 0)), seen_maps)
                    screen = emu._pb.screen.ndarray[:, :, :3].copy()
                    if getattr(active_policy, "use_gru", False):
                        probs, recurrent_hx = active_policy.probs_recurrent(screen, obs, hx=recurrent_hx)
                    else:
                        probs = active_policy.probs(screen, obs)
                    probs = np.asarray(probs, dtype=np.float64)
                    if args.greedy:
                        b_idx = int(np.argmax(probs))
                    else:
                        lg = np.log(probs + 1e-10) / max(0.01, args.temperature)
                        lg -= lg.max()
                        pp = np.exp(lg)
                        b_idx = int(np.random.choice(len(pp), p=pp / pp.sum()))
                else:  # no deep policy loaded: random buttons, so the script never plays for it either
                    b_idx = int(np.random.choice(len(ACTIONS)))
                button = ACTIONS[b_idx]
                emu.press(button, frames=8)
                emu.tick(12 if not args.watch else 40, render=render_screen)
                print(
                    f"[Step {step_idx}] Battle (mode={state.battle_mode}, active HP {state.battle_mon_hp}/{state.battle_mon_max_hp}, "
                    f"enemy species={state.enemy_mon_species} Lv{state.enemy_level} hp={state.enemy_hp}/{state.enemy_max_hp}): "
                    f"policy pressed {button}",
                    flush=True,
                )
                step_idx += 1
                continue


            # 2. Dialogue Handling
            if dialogue_open(state):
                blocked_count = 0
                recurrent_hx = None
                hp_before = sum(p[0] for p in getattr(state, "party_hp", ()) or ())
                macros.mash_text(max_frames=300)
                state_after_text = reader.parse(emu.frame)
                hp_after = sum(p[0] for p in getattr(state_after_text, "party_hp", ()) or ())
                if hp_after > hp_before:
                    print(f"[Step {step_idx}] [Pokémon Center] Party healed to full health! ({hp_after} HP)", flush=True)
                step_idx += 1
                continue

            # 2b. Same tile for many turns (NPC in a corridor, wall): talk, wait, walk around, back off
            unsticker.note(state)
            if unsticker.count >= 6 and last_heading in ("up", "down", "left", "right"):
                ures = unsticker.act(state, last_heading, emu, macros, stepper, tile_visit_counts)
                if ures is not None:
                    print(f"[Step {step_idx}] Unstick@{state.pos} {ures.kind}: {ures.detail}", flush=True)
                    recurrent_hx = None
                    blocked_count = 0
                    recent_positions.clear()
                    step_idx += 1
                    continue

            # 3. Check Triggers (Cut, NPC Interaction, Lateral Obstacle Bypass)
            if blocked_count == 2:
                # Try interacting with whatever entity is directly ahead (talk to NPC / inspect sign)
                emu.press("a", frames=8)
                emu.tick(20, render=render_screen)
                after_a = reader.parse(emu.frame)
                if dialogue_open(after_a):
                    print(f"[Step {step_idx}] Talked to NPC / interacted with object at {state.pos} facing {state.facing}! Mashing text...", flush=True)
                    macros.mash_text(max_frames=300)
                    blocked_count = 0
                    step_idx += 1
                    continue

            trigger = evaluate(state, stuck_count=blocked_count)
            if trigger is not None or blocked_count >= 3:
                recurrent_hx = None
                kind_str = getattr(getattr(trigger, "kind", None), "value", str(getattr(trigger, "kind", "")))
                if kind_str == "CUT_BLOCKED" and state.party_can_cut:
                    print(f"[Step {step_idx}] Trigger {trigger.kind}: Using Cut on tree...", flush=True)
                    macros.use_cut()
                    step_idx += 1
                    continue
                elif kind_str == "STUCK" or blocked_count >= 3:
                    # Penalize the forward blocked tile so compass/policy avoids walking into the blocking entity
                    forward_dir = last_action if last_action in _DIRS else state.facing
                    if forward_dir in _DELTA:
                        fdx, fdy = _DELTA[forward_dir]
                        blocked_target = (state.map_group, state.map_number, state.x + fdx, state.y + fdy)
                        tile_visit_counts[blocked_target] = tile_visit_counts.get(blocked_target, 0) + 150

                    if supervisor is not None and blocked_count >= 5:
                        trigger_arg = trigger or "STUCK: agent blocked or oscillating near dead-end"
                        print(f"[Step {step_idx}] Asking LLM Director ({supervisor.model}) to resolve bottleneck at {state.pos}...", flush=True)
                        try:
                            decision = supervisor.decide(state.to_text(), trigger_arg)
                            # The offline fallback only mashes text: it cannot fix a block, so fall through to the reflex
                            if decision and getattr(decision, 'name', '') != 'mash_text':
                                if decision.name == "use_cut" and not state.party_can_cut:
                                    print(f"[Step {step_idx}] [LLM Director] use_cut vetoed (no HM Cut before Gym 1); bypassing east...", flush=True)
                                    res = mover.run_path(["right"])
                                else:
                                    exec_args = dict(decision.arguments)
                                    if "reason" not in exec_args and decision.reason:
                                        exec_args["reason"] = decision.reason
                                    res = supervisor.run_tool(decision.name, exec_args)
                                print(f"[Step {step_idx}] [LLM Director] Execution result: ok={getattr(res, 'ok', True)}", flush=True)
                                blocked_count = 0
                                recent_positions.clear()
                                step_idx += 1
                                continue
                        except Exception as exc:
                            print(f"[Step {step_idx}] LLM Director error ({exc}), falling back to reflexive escape...", flush=True)

                    tiles = getattr(state, "tiles", {})
                    # Prioritize perpendicular lateral detour directions around the blocking entity
                    laterals = ["up", "down"] if forward_dir in ("left", "right") else ["left", "right"]
                    lateral_candidates = [
                        (tile_visit_counts.get((state.map_group, state.map_number, state.x + _DELTA[d][0], state.y + _DELTA[d][1]), 0), d)
                        for d in laterals if walkable(tiles.get(d, 0xFF), d)
                    ]
                    if lateral_candidates:
                        lateral_candidates.sort(key=lambda x: x[0])
                        escape_dir = lateral_candidates[0][1]
                    else:
                        candidates = []
                        for d in _DIRS:
                            if walkable(tiles.get(d, 0xFF), d):
                                dx, dy = _DELTA[d]
                                target = (state.map_group, state.map_number, state.x + dx, state.y + dy)
                                candidates.append((tile_visit_counts.get(target, 0), d))
                        candidates.sort(key=lambda x: x[0])
                        escape_dir = candidates[0][1] if candidates else None

                    if escape_dir:
                        print(
                            f"[Step {step_idx}] Stuck facing {forward_dir} at {state.pos}! Detouring around obstacle via '{escape_dir}'...",
                            flush=True,
                        )
                        res = mover.run_path([escape_dir])
                        if not res.ok:
                            dx, dy = _DELTA[escape_dir]
                            target = (state.map_group, state.map_number, state.x + dx, state.y + dy)
                            tile_visit_counts[target] = tile_visit_counts.get(target, 0) + 50
                    else:
                        emu.press("b", frames=8)
                        emu.tick(16, render=render_screen)
                    blocked_count = 0
                    step_idx += 1
                    continue

            # 4. Overworld Movement via Policy
            if is_deep_policy:
                # Same layout the checkpoint was trained with, as the evaluator builds it.
                from tools.evaluate_policy import obs_vector
                obs = obs_vector(state, visited, int(getattr(active_policy, "obs_dim", 0)), seen_maps)
            else:
                obs = observe(state, visited, nav, seen_maps=seen_maps)
            probs = None
            if active_policy is not None:
                if is_deep_policy:
                    screen = emu._pb.screen.ndarray[:, :, :3].copy()
                    if getattr(active_policy, "use_gru", False):
                        probs, recurrent_hx = active_policy.probs_recurrent(screen, obs, hx=recurrent_hx)
                        probs = probs.copy()
                    else:
                        probs = active_policy.probs(screen, obs).copy()
                elif hasattr(active_policy, "probs"):
                    probs = active_policy.probs(obs).copy()

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

                # D. Pokecenter / dialogue are handled by the policy itself (A on Nurse Joy); no scripted overrides.
                full = probs.astype(np.float64).copy()
                full[:4] = dir_probs
                total_p = full.sum()
                if total_p > 1e-6:
                    full /= total_p
                    if args.greedy:
                        action_idx = int(np.argmax(full))
                    else:
                        logits = np.log(full + 1e-10) / max(0.01, args.temperature)
                        logits -= logits.max()
                        p_scaled = np.exp(logits)
                        p_scaled /= p_scaled.sum()
                        action_idx = int(np.random.choice(len(p_scaled), p=p_scaled))
                else:
                    action_idx = int(np.random.choice(4))
            else:
                action_idx = int(np.random.choice(4))

            direction = ACTIONS[action_idx]
            if direction in ("up", "down", "left", "right"):
                last_heading = direction

            # Execute step
            before_pos = (state.x, state.y)
            if direction in BUTTON_ACTIONS:
                emu.press(direction, frames=8)
                emu.tick(12, render=render_screen)
                moved = False
                consecutive_a_count += 1
            else:
                res = mover.run_path([direction])
                moved = bool(res.ok)
                consecutive_a_count = 0

            after_state = reader.parse(emu.frame)
            after_pos = (after_state.x, after_state.y)
            nav.note_action(state, action_idx, moved)

            recent_positions.append(after_pos)
            is_oscillating = (len(recent_positions) == 8 and len(set(recent_positions)) <= 2)

            # Check if player moved or if dialogue/battle engaged
            if direction in BUTTON_ACTIONS:
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

            # Calculate live active reward
            unvisited_neighbor = False
            if 0 <= action_idx < 4:
                d_name = ACTIONS[action_idx]
                dx, dy = _DELTA[d_name]
                neighbor = (state.map_group, state.map_number, state.x + dx, state.y + dy)
                unvisited_neighbor = neighbor not in seen_tiles

            info = StepInfo(consecutive_bumps=blocked_count)
            step_rew, disc_tile, disc_map = progress_reward(
                before=state,
                after=after_state,
                moved=moved,
                seen_tiles=seen_tiles,
                seen_maps=seen_maps,
                unvisited_neighbor=unvisited_neighbor,
                caught_counts=caught_counts,
                seen_events=seen_events,
                seen_waypoints=seen_waypoints,
                info=info,
            )
            cumulative_reward += step_rew

            notes = []
            if disc_map:
                notes.append("NEW MAP!")
            elif disc_tile:
                notes.append("NEW TILE")
            if info.caught_pokemon:
                notes.append("CAUGHT POKEMON!")
            if not moved and after_pos == before_pos and direction not in BUTTON_ACTIONS:
                notes.append(f"BUMP x{blocked_count}")
            notes_str = f" [{' '.join(notes)}]" if notes else ""

            step_idx += 1
            map_name = get_map_name(after_state.map_group, after_state.map_number)
            print(
                f"[Step {step_idx:4d}] {map_name:18s} ({after_state.x:2d}, {after_state.y:2d}) | "
                f"Act: {direction:5s} | Rew: {step_rew:+.2f} | "
                f"Return: {cumulative_reward:+.2f} | "
                f"Tiles: {len(seen_tiles):3d} | Party: {after_state.party_count}/6{notes_str}",
                flush=True,
            )

            # Story progress: announce every milestone change, restate the checklist with the telemetry report
            snap = story_snapshot(after_state, milestones)
            if snap["known_flags"] != last_story_flags:
                new = sorted(set(snap["known_flags"]) - set(last_story_flags or ()))
                if last_story_flags is not None and new:
                    print(f"[Story] +{', '.join(new)}", flush=True)
                if last_story_flags is None or new:
                    print(f"[Story] {checklist_line(snap)}", flush=True)
                last_story_flags = snap["known_flags"]
            if step_idx == 1 or step_idx % 25 == 0:
                print(f"[Story] {checklist_line(snap)}", flush=True)
                write_status(args.tag, after_state, snap, step_idx, cumulative_reward)

            # Every 25 moves, print dedicated telemetry report with reward, moveset, and location
            if step_idx == 1 or step_idx % 25 == 0:
                lead_name = get_species_name(after_state.party_species[0]) if after_state.party_species else "None"
                lead_lvl = after_state.party_levels[0] if getattr(after_state, "party_levels", None) else "?"
                lead_hp = (
                    f"{after_state.party_hp[0][0]}/{after_state.party_hp[0][1]}"
                    if getattr(after_state, "party_hp", None) and len(after_state.party_hp) > 0
                    else "?/?"
                )
                moves = get_party_moveset(getattr(after_state, "party_moves", ()))
                moves_str = ", ".join(moves) if moves else "None"

                print(
                    f"\n{'='*75}\n"
                    f"🎮 [Move {step_idx:4d}] LOCATION : {map_name} (Map {after_state.map_group},{after_state.map_number}) at ({after_state.x:2d}, {after_state.y:2d}) | Facing: {after_state.facing}\n"
                    f"🏆 [Move {step_idx:4d}] REWARD   : Step={step_rew:+.2f} | Cumulative Return={cumulative_reward:+.2f} | Explored Tiles={len(seen_tiles):3d}\n"
                    f"🐾 [Move {step_idx:4d}] LEAD MON : {lead_name} Lv.{lead_lvl} (HP: {lead_hp}) | Party: {after_state.party_count}/6\n"
                    f"⚔️  [Move {step_idx:4d}] MOVESET  : [{moves_str}]\n"
                    f"{'='*75}\n",
                    flush=True,
                )

                try:
                    from PIL import Image
                    scr = emu._pb.screen.ndarray[:, :, :3]
                    img = Image.fromarray(scr).resize((320, 288), Image.Resampling.NEAREST)
                    img.save(f"logs/live_snapshot_{args.tag}.png")
                    img.save("logs/live_snapshot.png")
                except Exception:
                    pass

            # Auto-save frontier map states for curriculum expansion
            if disc_map:
                map_file = f"states/agent_map_{after_state.map_group}_{after_state.map_number}.state"
                if not os.path.exists(map_file):
                    try:
                        emu.save_state(map_file)
                        print(f"[Frontier] Saved new map checkpoint: {map_file} ({map_name})!", flush=True)
                    except Exception:
                        pass

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
