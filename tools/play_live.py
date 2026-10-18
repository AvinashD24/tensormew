"""Live interactive evaluation: runs indefinitely until user closes the PyBoy window."""

from __future__ import annotations

import argparse
import collections
import os
import sys
from dotenv import load_dotenv
import numpy as np

load_dotenv()

from src.battle import tactical_battle_step
from src.collision import is_ledge
from src.emulator import Emulator
from src.llm_supervisor import Supervisor
from src.macros import Macros
from src.mover import Mover, walkable
from src.ppo import load_policy
from src.ram_parser import RamParser, dialogue_open
from src.local_nav import LocalNav
from src.rl_env import ACTIONS, StepInfo, navigation_potential, observe, progress_reward
from src.states import StateRegistry
from src.step import TileStepper
from src.live_status import checklist_line, story_snapshot, write_status
from src.game_data import get_map_name, get_party_moveset, get_species_name
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
                policy = SpatialVisionPolicy.load(policy_path, device="cpu")
                is_deep_policy = True
                print(
                    f"Loaded {policy.param_count:,} parameter SpatialVisionPolicy "
                    f"(recurrent={getattr(policy, 'use_gru', False)}) from {policy_path}"
                )
            else:
                policy = load_policy(policy_path)
                if policy is not None:
                    print(f"Loaded policy from {policy_path} ({policy.__class__.__name__})")

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
        step_idx = 0
        recurrent_hx = None
        consecutive_a_count = 0
        recent_positions: collections.deque[tuple[int, int]] = collections.deque(maxlen=8)
        render_screen = not args.headless

        last_story_flags = None
        while True:
            # Check if PyBoy window was closed
            if not emu.tick(1, render=render_screen):
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
                act_idx = getattr(state, "cur_battle_mon", 0)
                act_idx = min(act_idx, len(state.party_hp) - 1) if state.party_hp else 0
                b_cur = getattr(state, "battle_mon_hp", 0)
                b_max = getattr(state, "battle_mon_max_hp", 0)
                if b_max > 0:
                    p_cur, p_max = b_cur, b_max
                else:
                    p_cur = state.party_hp[act_idx][0] if state.party_hp else 0
                    p_max = state.party_hp[act_idx][1] if state.party_hp else 1
                res = tactical_battle_step(state, macros)
                reason = getattr(res, "message", getattr(res, "reason", "acted"))
                mon_name = get_species_name(state.party_species[act_idx]) if state.party_species and act_idx < len(state.party_species) else "Mon"
                print(
                    f"[Step {step_idx}] Battle (mode={state.battle_mode}, "
                    f"active={mon_name} {p_cur}/{p_max}, "
                    f"enemy_species={state.enemy_species}, hp={state.enemy_hp}/{state.enemy_max_hp}): {reason}",
                    flush=True,
                )
                if not reason.startswith("switched"):
                    for _ in range(8):
                        emu.tick(20, render=render_screen)
                        emu.press("b", frames=6)
                else:
                    for _ in range(4):
                        emu.tick(20, render=render_screen)
                try:
                    from PIL import Image
                    scr = emu._pb.screen.ndarray[:, :, :3]
                    img = Image.fromarray(scr).resize((320, 288), Image.Resampling.NEAREST)
                    img.save(f"logs/live_snapshot_{args.tag}.png")
                    img.save("logs/live_snapshot.png")
                except Exception:
                    pass
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
                    if supervisor is not None:
                        trigger_arg = trigger or "STUCK: agent blocked or oscillating near dead-end"
                        print(f"[Step {step_idx}] Asking LLM Director ({supervisor.model}) to resolve bottleneck at {state.pos}...", flush=True)
                        try:
                            decision = supervisor.decide(state.to_text(), trigger_arg)
                            if decision:
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
                        emu.tick(16, render=render_screen)
                    blocked_count = 0
                    step_idx += 1
                    continue

            # 4. Overworld Movement via Policy
            use_goals = is_deep_policy and getattr(policy, "obs_dim", 17) == 21
            obs = observe(state, visited, None if is_deep_policy else nav, include_goals=use_goals)
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

                # E. Pokémon Center Nurse Joy Interaction when low HP
                is_pc = state.map_id in ((26, 5), (10, 10), (10, 13))
                party_hp = getattr(state, "party_hp", ()) or ()
                cur_hp = sum(p[0] for p in party_hp) if party_hp else 0
                max_hp = sum(p[1] for p in party_hp) if party_hp else 0
                needs_heal = max_hp > 0 and cur_hp < max_hp

                has_ev = getattr(state, "has_event", None)
                has_egg = callable(has_ev) and has_ev("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
                gave_egg = callable(has_ev) and has_ev("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
                beat_rival = getattr(state, "beat_rival", False) or (callable(has_ev) and has_ev("EVENT_RIVAL_CHERRYGROVE_CITY"))
                beat_falkner = callable(has_ev) and has_ev("EVENT_BEAT_FALKNER")

                total_p = dir_probs.sum()
                if is_pc and needs_heal and state.x == 3 and state.y == 3:
                    if state.facing == "up":
                        # Standing at Nurse Joy counter facing up with damaged party -> interact to heal!
                        action_idx = 4
                    else:
                        # Turn to face Nurse Joy directly across the counter
                        action_idx = 0
                elif state.map_id == (26, 10) and has_egg and not gave_egg and state.y == 7 and state.x in (2, 3):
                    # Standing on exit carpet in Mr. Pokémon's House with Mystery Egg -> step out!
                    action_idx = 1
                elif state.map_id == (26, 3) and has_egg and not beat_rival and state.x in (31, 32) and state.y in (6, 7):
                    # Approaching Rival trigger at (33, 6..7) in Cherrygrove City -> step right!
                    action_idx = 3
                elif state.map_id == (26, 3) and has_egg and beat_rival and not gave_egg and state.x == 39 and state.y in (6, 7):
                    # Defeated rival, standing at Cherrygrove east gate -> step right into Route 29!
                    action_idx = 3
                elif state.map_id == (24, 3) and has_egg and beat_rival and not gave_egg:
                    # Route 29 Eastbound to Elm: stay on upper northern highway to avoid one-way ledge traps
                    if state.x == 59 and state.y in (8, 9):
                        action_idx = 3
                    elif state.y >= 7 and state.x <= 38:
                        dir_probs[1] = 0.0  # Mask 'down' to strictly prevent falling off ledges
                        if dir_probs[3] > 0:
                            dir_probs[3] *= 3.0  # Boost 'right'
                        if dir_probs[0] > 0:
                            dir_probs[0] *= 2.0  # Boost 'up'
                        total_p = dir_probs.sum()
                        if total_p > 0:
                            dir_probs /= total_p
                elif state.map_id == (24, 4) and has_egg and beat_rival and not gave_egg and state.x == 6 and state.y == 4:
                    # Outside Elm's Lab in New Bark Town with Mystery Egg -> enter lab!
                    action_idx = 0
                elif state.map_id == (24, 5) and has_egg and beat_rival and not gave_egg and state.x in (4, 5) and state.y == 3:
                    if state.facing == "up":
                        # Standing at Professor Elm's desk facing up with Mystery Egg -> deliver it!
                        action_idx = 4
                    else:
                        action_idx = 0
                elif state.map_id == (24, 5) and gave_egg and state.y == 11 and state.x in (4, 5):
                    # Standing on exit carpet in Elm's Lab after delivering egg -> step out!
                    action_idx = 1
                elif state.map_id == (26, 2) and state.x == 4 and state.y in (6, 7):
                    # Route 31 west entrance doorway into Violet Gate -> step left!
                    action_idx = 2
                elif state.map_id == (26, 11) and state.x == 0 and state.y in (4, 5):
                    # Route 31 Violet Gate west exit doorway into Violet City -> step left!
                    action_idx = 2
                elif state.map_id == (10, 5) and not beat_falkner and state.x == 18 and state.y == 18:
                    # Outside Violet Gym in Violet City facing entrance -> step up!
                    action_idx = 0
                elif state.map_id == (10, 7) and not beat_falkner and state.x in (4, 5) and state.y in (2, 3):
                    if state.facing == "up":
                        # Standing in front of Leader Falkner facing up -> challenge Falkner!
                        action_idx = 4
                    else:
                        action_idx = 0
                elif state.map_id in ((26, 4), (26, 6), (26, 7), (26, 8), (26, 9), (24, 6), (24, 7), (24, 8), (24, 9), (10, 6), (10, 8), (10, 9), (10, 11)) and state.y == 7 and state.x in (2, 3, 4):
                    # Incidental building exit mat -> step out!
                    action_idx = 1
                elif is_pc and not needs_heal and state.y == 7 and state.x in (3, 4):
                    # Fully healed in Pokémon Center on exit mat -> step out!
                    action_idx = 1
                elif total_p > 1e-6:
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
            phi_now = navigation_potential(after_state)

            notes = []
            if disc_map:
                notes.append("NEW MAP!")
            elif disc_tile:
                notes.append("NEW TILE")
            if info.caught_pokemon:
                notes.append("CAUGHT POKEMON!")
            if not moved and after_pos == before_pos and direction != "a":
                notes.append(f"BUMP x{blocked_count}")
            notes_str = f" [{' '.join(notes)}]" if notes else ""

            step_idx += 1
            map_name = get_map_name(after_state.map_group, after_state.map_number)
            print(
                f"[Step {step_idx:4d}] {map_name:18s} ({after_state.x:2d}, {after_state.y:2d}) | "
                f"Act: {direction:5s} | Rew: {step_rew:+.2f} | "
                f"Return: {cumulative_reward:+.2f} | Phi: {phi_now:5.1f} | "
                f"Tiles: {len(seen_tiles):3d} | Party: {after_state.party_count}/6{notes_str}",
                flush=True,
            )

            # Story progress: announce every milestone change, restate the checklist with the telemetry report
            snap = story_snapshot(after_state)
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
                    f"🏆 [Move {step_idx:4d}] REWARD   : Step={step_rew:+.2f} | Cumulative Return={cumulative_reward:+.2f} | Potential Φ={phi_now:5.2f} | Explored Tiles={len(seen_tiles):3d}\n"
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

            # Auto-save checkpoints on major story milestones
            if has_egg and not os.path.exists("states/mystery_egg_received.state"):
                try:
                    emu.save_state("states/mystery_egg_received.state")
                    print("[Checkpoint] Saved states/mystery_egg_received.state!", flush=True)
                except Exception:
                    pass
            if gave_egg and not os.path.exists("states/mystery_egg_delivered.state"):
                try:
                    emu.save_state("states/mystery_egg_delivered.state")
                    print("[Checkpoint] Saved states/mystery_egg_delivered.state!", flush=True)
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
