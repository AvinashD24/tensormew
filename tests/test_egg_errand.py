"""long_v8 ``egg_errand_route``: the phased route (Mr. Pokemon's egg -> Elm's lab -> Falkner), its 11 milestones, dims, rewards.

The 309-step route is the no-egg route and is unreachable at game start (Route 30 is cut at the one-tile choke (5, 25) until Elm
has the egg). These tests pin the phase lengths, the continuity of the combined progress, the blocked choke, the milestone
table / dims, the presets' isolation and (with the ROM and states) the real flag reads and transitions.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from src import path_progress as pp
from src import route_graph as rg
from src import story_path as sp
from src.config import ROM_PATH
from src.story_flags import EVENT_FLAGS
from src.train_config import PRESETS, ConfigError, from_dict, load_config, preset

ROOT = Path(__file__).resolve().parents[1]
HAVE_ROM = Path(ROM_PATH).exists()
SCALE = 0.3
BASE_FLAGS = {30: "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON", 31: "EVENT_GAVE_MYSTERY_EGG_TO_ELM"}


def fake(node, flags=(), **kw):
    """A parsed-state stand-in: position plus a ``has_event`` over a set of flag ids (names resolve through EVENT_FLAGS)."""
    mid, x, y = node
    fl = set(flags)

    def has_event(f):
        return (EVENT_FLAGS.get(f) if isinstance(f, str) else int(f)) in fl

    kw.setdefault("script_mode", 0)
    kw.setdefault("script_running", 0)
    return SimpleNamespace(map_group=mid[0], map_number=mid[1], x=x, y=y, fainted=False, has_event=has_event, **kw)


def phase_flags(phase):
    return {0: (), 1: (30,), 2: (30, 31, 1812)}[phase]


class EggProfile(unittest.TestCase):
    def setUp(self):
        self.addCleanup(sp.set_profile, False)   # the profile is process-global: never leak it into other tests


class StoryProfileTests(EggProfile):
    def test_base_profile_is_unchanged_and_is_the_default(self):
        self.assertIs(sp.profile(), sp.BASE_PROFILE)
        self.assertEqual((sp.MILESTONE_COUNT, sp.STORY_FEATURE_DIM, sp.TOTAL_STORY_REWARD), (8, 10, 140.0))
        self.assertEqual([m.name for m in sp.MILESTONES][-1], "beat_falkner")
        self.assertEqual(sp.STORY_PATH_STEPS, 309)

    def test_egg_milestone_table_rewards_and_total(self):
        p = sp.EGG_PROFILE
        self.assertEqual(p.names, ("route29", "cherrygrove", "route30", "mr_pokemon_house", "egg_received", "egg_delivered",
                                   "route31", "violet_gate", "violet_city", "violet_gym", "beat_falkner"))
        rewards = {m.name: m.reward for m in p.milestones}
        self.assertEqual(rewards["mr_pokemon_house"], 10.0)
        self.assertEqual(rewards["egg_received"], 15.0)
        self.assertEqual(rewards["egg_delivered"], 30.0)
        for m in sp.MILESTONES:   # the common milestones keep their values
            self.assertEqual(rewards[m.name], m.reward, m.name)
        self.assertEqual(p.count, 11)
        self.assertEqual(p.feature_dim, 13)
        self.assertEqual(p.total_reward, 195.0)   # 125 milestones + 60 badge + 10 EVENT_BEAT_FALKNER
        self.assertEqual(p.key_event_rewards, {"EVENT_BEAT_FALKNER": 10.0, sp.EGG_RECEIVED_EVENT: 15.0, sp.EGG_DELIVERED_EVENT: 30.0})
        self.assertEqual(p.map_milestones[(26, 10)].name, "mr_pokemon_house")
        self.assertEqual(sp.EGG_PATH_STEPS, 695)

    def test_set_profile_and_context_manager(self):
        with sp.use_profile(True) as p:
            self.assertIs(p, sp.EGG_PROFILE)
            self.assertIs(sp.profile(), sp.EGG_PROFILE)
        self.assertIs(sp.profile(), sp.BASE_PROFILE)
        self.assertIs(sp.set_profile("egg"), sp.EGG_PROFILE)
        self.assertIs(sp.set_profile(False), sp.BASE_PROFILE)

    def test_obs_dims_follow_the_profile_only(self):
        from src.rl_env import BASE_OBS_DIM, DEEP_OBS_DIM, deep_obs_dim, observe
        self.assertEqual(DEEP_OBS_DIM, 77)
        self.assertEqual(deep_obs_dim(), 77)
        self.assertEqual(deep_obs_dim(True), 80)
        s = fake(((24, 4), 3, 9), tiles={}, badges=0, battle_mode=0, party_count=1, party_levels=(5,), pokedex_caught_count=1)
        s.tiles = {d: 0 for d in ("up", "down", "left", "right")}
        base = observe(s, set(), deep=True, seen_maps={(24, 4)})
        self.assertEqual(base.shape, (77,))
        with sp.use_profile(True):
            egg = observe(s, set(), deep=True, seen_maps={(24, 4)})
            self.assertEqual(egg.shape, (80,))
            self.assertEqual(sp.story_features(s, {(24, 4)}).shape[0], 13)
        # the story block holds only done-bits and the map id: no direction, no distance
        self.assertEqual(list(egg[BASE_OBS_DIM:BASE_OBS_DIM + 11]), [0.0] * 11)

    def test_milestones_done_by_flags_and_implications(self):
        with sp.use_profile(True):
            s = fake(((24, 4), 3, 9))
            self.assertEqual(sum(sp.milestones_done(s)), 0)
            s = fake(((26, 10), 3, 6), flags=(30,))
            done = sp.milestones_done(s)
            # the egg in hand implies Route 29 .. Mr. Pokemon's house behind it, but not the delivery
            self.assertEqual(done, [True] * 5 + [False] * 6)
            s = fake(((24, 5), 5, 3), flags=(30, 31, 1812))
            self.assertEqual(sp.milestones_done(s), [True] * 6 + [False] * 5)
            self.assertEqual(sp.milestones_done(fake(((10, 7), 5, 2), flags=(30, 31)), {(10, 7)})[:7], [True] * 6 + [False])
            self.assertEqual(sp.milestones_done(fake(((10, 7), 5, 2), badges=1)), [True] * 11)

    def test_current_stage_and_implied_maps(self):
        with sp.use_profile(True):
            self.assertEqual(sp.current_stage(fake(((24, 4), 3, 9))), 0)
            self.assertEqual(sp.current_stage(fake(((26, 10), 3, 6))), 4)              # in Mr. Pokemon's house, no egg yet
            self.assertEqual(sp.current_stage(fake(((26, 10), 3, 6), flags=(30,))), 5)
            self.assertEqual(sp.current_stage(fake(((24, 4), 6, 7), flags=(30, 31))), 6)   # delivered, standing in New Bark
            self.assertEqual(sp.current_stage(fake(((26, 2), 7, 7), flags=(30, 31))), 7)   # Route 31
            self.assertEqual(sp.current_stage(fake(((10, 7), 5, 2), badges=1)), 11)
            self.assertEqual(sp.implied_maps(fake(((24, 4), 3, 9))), set())
            self.assertEqual(sp.implied_maps(fake(((26, 10), 3, 6), flags=(30,))), {(24, 3), (26, 3), (26, 1), (26, 10)})
            self.assertEqual(sp.implied_maps(fake(((24, 5), 5, 3), flags=(30, 31))), {(24, 3), (26, 3), (26, 1), (26, 10)})
        self.assertEqual(sp.implied_maps(fake(((26, 10), 3, 6), flags=(30,))), set())   # base: no event milestones

    def test_progress_reward_pays_the_egg_milestones_once(self):
        from src.rl_env import progress_reward
        with sp.use_profile(True):
            seen_maps, seen_events, comps = set(), set(), {}
            before = fake(((26, 1), 17, 6), party_hp=(), tiles={}, event_flags=bytes(256))
            after = fake(((26, 10), 2, 7), party_hp=(), tiles={}, event_flags=bytes(256))
            progress_reward(before, after, True, set(), seen_maps, seen_events=seen_events, components=comps)
            self.assertEqual(comps["milestone"], 10.0)    # Mr. Pokemon's house
            ev = bytearray(256)
            ev[30 >> 3] |= 1 << (30 & 7)
            got = fake(((26, 10), 3, 6), party_hp=(), tiles={}, event_flags=bytes(ev))
            comps = {}
            progress_reward(after, got, True, set(), seen_maps, seen_events=seen_events, components=comps)
            self.assertEqual(comps["event"], 15.0)
            ev[31 >> 3] |= 1 << (31 & 7)
            ev[1812 >> 3] |= 1 << (1812 & 7)
            given = fake(((24, 5), 5, 3), party_hp=(), tiles={}, event_flags=bytes(ev))
            comps = {}
            progress_reward(got, given, True, set(), seen_maps, seen_events=seen_events, components=comps)
            self.assertGreaterEqual(comps["event"], 30.0)
            self.assertLessEqual(comps["event"], 30.0 + rg_unnamed() + 1e-9)   # + the unnamed hide flag 1812 (0.05, explore budget)
            comps = {}
            progress_reward(given, given, False, set(), seen_maps, seen_events=seen_events, components=comps)
            self.assertNotIn("event", comps)   # nothing is paid twice


def rg_unnamed():
    from src.rl_env import UNNAMED_EVENT_REWARD
    return UNNAMED_EVENT_REWARD


class MilestoneBookkeepingTests(EggProfile):
    def test_stats_records_and_split_follow_the_milestone_count(self):
        from src.milestone_stats import MilestoneRecords, MilestoneStats, SourceSplitStats, milestone_labels
        self.assertEqual(MilestoneStats().n, 8)
        with sp.use_profile(True):
            self.assertEqual(milestone_labels()[3:6], ("Mr. Pokemon's house", "Egg received", "Egg delivered"))
            ms = MilestoneStats()
            self.assertEqual(ms.n, 11)
            ms.add_episode(0, 6)    # reached stage 6 = the egg delivered
            out = ms.end_update([6, 2, None])
            self.assertEqual(len(out["ms_reach"]), 11)
            self.assertEqual(out["ms_reach"][:7], [1.0] * 6 + [0.0])
            self.assertEqual(len(out["ms_envs_at"]), 11)
            rec = MilestoneRecords()
            ev = rec.observe(1, 5, 480, 3, from_start=True)
            self.assertEqual(ev[0]["name"], "Egg delivered")
            snap = rec.snapshot()
            self.assertEqual(len(snap["start"]), 11)
            self.assertEqual(snap["names"][4], "Egg received")
            sp_stats = SourceSplitStats()
            sp_stats.add_episode("base", 0, 11 - 1, best_path=600.0, start_path=0.0)
            self.assertEqual(len(sp_stats.end_update()["src_split_cum"]["base"]["reach"]), 11)
        self.assertEqual(len(MilestoneRecords().snapshot()["start"]), 8)


class ScriptedDialogueTests(EggProfile):
    """The long scripted Mr. Pokemon / Elm dialogue (about 33 text boxes at Elm) is never penalised."""

    def test_first_time_long_dialogue_pays_no_dialogue_loop_or_idle_press_cost(self):
        from src.dialogue_loop import DialogueLoopTracker
        from src.idle_press import IdlePressTracker
        node = ((24, 5), 5, 3)
        closed = fake(node, (30,), event_flags=(0,) * 256, facing=1)
        text = fake(node, (30,), event_flags=(0,) * 256, facing=1, script_mode=1)
        flags = [0] * 256
        flags[3] |= 0x80    # flag 31
        done = fake(node, (30, 31), event_flags=tuple(flags), facing=1, script_mode=0)
        tr = DialogueLoopTracker()
        tr.reset()
        total = tr.update(closed, text, True)                  # A at Elm: the first talk to this tile and facing
        for _ in range(150):                                    # the auto-skip keeps the text open for many steps
            total += tr.update(text, text, False)
        total += tr.update(text, done, False)                   # the script ends with the egg given (flag 31 flips)
        self.assertEqual(total, 0.0)
        self.assertEqual(tr.charged, 0)
        idle = IdlePressTracker()
        idle.reset()
        self.assertEqual(idle.update(closed, text), 0.0)        # the press opened a script: an effect, not an idle press


class DwellTargetsTests(EggProfile):
    def test_targets_table_and_validation(self):
        from src import dwell
        self.assertEqual(len(dwell.DWELL_LEVEL_TARGETS), 9)
        self.assertEqual(len(dwell.DWELL_LEVEL_TARGETS_EGG), 12)
        self.assertEqual(dwell.validate_params(192, 16, 0.005, 0.03, 384, 15.0, 512, dwell.DWELL_LEVEL_TARGETS_EGG, 11), [])
        self.assertTrue(dwell.validate_params(192, 16, 0.005, 0.03, 384, 15.0, 512, dwell.DWELL_LEVEL_TARGETS, 11))
        self.assertTrue(dwell.validate_params(192, 16, 0.005, 0.03, 384, 15.0, 512, dwell.DWELL_LEVEL_TARGETS_EGG))   # base count
        with sp.use_profile(True):
            t = dwell.DwellTracker()
            self.assertEqual(t.level_targets, dwell.DWELL_LEVEL_TARGETS_EGG)
            t.reset(fake(((24, 4), 3, 9), party_levels=(5,)))
            self.assertEqual(t.target_level(), 8)
            self.assertEqual(sp.current_stage(fake(((26, 10), 3, 6), flags=(30,))), 5)
            t.furthest = 5
            self.assertEqual(t.target_level(), 10)
        self.assertEqual(dwell.DwellTracker().level_targets, dwell.DWELL_LEVEL_TARGETS)


class PresetTests(unittest.TestCase):
    def test_only_long_v8_has_the_egg_route(self):
        for name in PRESETS:
            cfg = preset(name)
            self.assertEqual(cfg.egg_errand_route, name == "long_v8", name)
            self.assertEqual(cfg.story_profile(), sp.EGG_PROFILE if name == "long_v8" else sp.BASE_PROFILE)
        for name in ("long_v3", "long_v4", "long_v5", "long_v6", "long_v7"):
            self.assertEqual(len(preset(name).dwell_level_targets), 9, name)

    def test_long_v8_differs_from_long_v7_only_in_the_v8_fields(self):
        a, b = dataclasses.asdict(preset("long_v7")), dataclasses.asdict(preset("long_v8"))
        changed = {k for k in a if a[k] != b[k]}
        self.assertEqual(changed, {"name", "battle_action_mask", "periodic_eval", "egg_errand_route", "dwell_level_targets"})
        self.assertEqual(b["dwell_level_targets"], list(__import__("src.dwell", fromlist=["x"]).DWELL_LEVEL_TARGETS_EGG))
        cfg = preset("long_v8").validate()
        self.assertTrue(cfg.path_progress and cfg.retreat_penalty and cfg.heal_trip_exempt and cfg.terminate_on_whiteout)
        self.assertEqual((cfg.path_progress_scale, cfg.route_ckpt_every, cfg.route_ckpt_max), (0.3, 12, 60))

    def test_wrong_dwell_table_length_is_rejected_with_a_clear_message(self):
        with self.assertRaises(ConfigError) as cm:
            from_dict({"egg_errand_route": True}, preset("long_v7"))
        self.assertIn("dwell_level_targets needs 12 entries", str(cm.exception))
        with self.assertRaises(ConfigError):
            from_dict({"dwell_level_targets": [8] * 12}, preset("long_v7"))
        self.assertTrue(from_dict({"egg_errand_route": True, "dwell_level_targets": [8] * 12}, preset("long_v7")).egg_errand_route)

    def test_json_roundtrip_keeps_the_flag(self):
        cfg = preset("long_v8")
        again = from_dict(json.loads(cfg.to_json()), preset("blueprint_v1"))
        self.assertTrue(again.egg_errand_route)
        self.assertEqual(load_config("long_v8").egg_errand_route, True)

    def test_cloud_train_accepts_long_v8_and_the_launcher_stays_numpy_free(self):
        from tools import cloud_train_deep as c
        self.assertEqual(c.resolve_config(argparse.Namespace(config="long_v8")), ("long_v8", False))
        code = ("import sys; sys.modules['numpy'] = None; sys.modules['torch'] = None; "
                "from src.train_config import PRESETS, load_config; import src.story_path, src.milestone_stats; "
                "assert load_config('long_v8').egg_errand_route and load_config('long_v8').story_profile().count == 11; print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)

    def test_worker_env_cfg_carries_the_flag(self):
        from tools import train_deep_ppo as t
        self.assertTrue(t.worker_env_cfg(preset("long_v8"), None)["egg_errand_route"])
        self.assertFalse(t.worker_env_cfg(preset("long_v7"), None)["egg_errand_route"])

    def test_old_checkpoints_raise_the_clear_error(self):
        import torch   # noqa: F401
        from src.deep_policy import CheckpointIncompatibleError, SpatialVisionPolicy
        from src.rl_env import N_ACTIONS, deep_obs_dim
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "old.pt")
            SpatialVisionPolicy(obs_dim=deep_obs_dim(False), num_actions=N_ACTIONS, use_gru=False).save(path)
            with self.assertRaises(CheckpointIncompatibleError):
                SpatialVisionPolicy.load(path, override_obs_dim=deep_obs_dim(True), check_schema=True)
            SpatialVisionPolicy.load(path, override_obs_dim=deep_obs_dim(False), check_schema=True)   # same layout: fine


class RouteCkptPhaseTests(unittest.TestCase):
    def test_phase_in_the_file_name_and_old_names_still_parse(self):
        from src import route_ckpt as rc
        self.assertEqual(rc.key_info("rc_b007_2_s3"), (7, 2, 3))
        self.assertIsNone(rc.key_phase("rc_b007_2_s3"))
        self.assertEqual(rc.key_info("rc_b012_1_s4_p1"), (12, 1, 4))
        self.assertEqual(rc.key_phase("rc_b012_1_s4_p1"), 1)
        with tempfile.TemporaryDirectory() as tmp:
            a = rc.RouteCheckpointArchive(tmp, per_bucket=2, max_files=10)
            self.assertTrue(a.save(15, 4, b"x", phase=1))
            self.assertTrue(a.save(16, 4, b"y"))
            keys = a.keys()
            self.assertEqual(keys, ["rc_b015_0_s4_p1", "rc_b016_0_s4"])
            self.assertEqual(a.load("rc_b015_0_s4_p1"), (b"x", 4))
            self.assertEqual(a.counts(), {15: 1, 16: 1})

    def test_stage_helpers_follow_the_profile(self):
        from src import route_ckpt as rc
        self.addCleanup(sp.set_profile, False)
        self.assertEqual(rc.milestone_stage({(24, 3), (26, 3), (26, 1), (26, 10)}), 3)       # base: Route 30 is milestone 3
        with sp.use_profile(True):
            self.assertEqual(rc.milestone_stage({(24, 3), (26, 3), (26, 1), (26, 10)}), 4)   # Mr. Pokemon's house is milestone 4
            self.assertEqual(rc.stage_maps(4), {(24, 3), (26, 3), (26, 1), (26, 10)})
            self.assertEqual(rc.stage_maps(7), {(24, 3), (26, 3), (26, 1), (26, 10), (26, 2)})


@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class PhasedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = rg._graph()
        if cls.g is None:
            raise unittest.SkipTest("ROM not available")
        cls.info = rg.egg_phase_info()
        cls.paths = [cls.walk(cls.g.phase_dist[i], s) for i, s in enumerate((rg.PATH_START, rg.EGG_RECEIVE_GOAL, rg.EGG_DELIVER_GOAL))]

    @classmethod
    def walk(cls, dist, start):
        n, out = start, [start]
        while dist[n] > 0:
            n = next(o for o in cls.g.fwd[n] if o is not None and dist.get(o) == dist[n] - 1)
            out.append(n)
        return out

    def test_phase_lengths_and_total(self):
        self.assertEqual(tuple(self.info["len"]), sp.EGG_PHASE_STEPS)
        self.assertEqual(tuple(self.info["len"]), (185, 183, 327))
        self.assertEqual(self.info["off"], (0, 185, 368))
        self.assertEqual(self.info["total"], sp.EGG_PATH_STEPS)
        self.assertEqual(rg.path_total(egg=True), 695.0)
        self.assertEqual(rg.path_total(), 309.0)      # the flag-off measure is untouched
        self.assertEqual([len(p) - 1 for p in self.paths], [185, 183, 327])
        from tools import dashboard_events
        self.assertEqual(dashboard_events.EGG_PATH_TOTAL, sp.EGG_PATH_STEPS)
        self.assertEqual(len(dashboard_events.EGG_MILESTONE_NAMES), sp.EGG_PROFILE.count)

    def test_each_phase_path_is_monotone_one_step_at_a_time(self):
        for i, path in enumerate(self.paths):
            dist = self.g.phase_dist[i]
            for k, (a, b) in enumerate(zip(path, path[1:])):
                self.assertIn(b, self.g.fwd[a], (i, k))                   # a legal single action
                self.assertEqual(dist[a] - dist[b], 1, (i, k, a, b))      # exactly one step closer to the phase goal
            self.assertEqual(dist[path[-1]], 0)

    def test_the_path_passes_the_expected_places(self):
        def maps(p):
            out = []
            for n in p:
                if not out or out[-1] != n[0]:
                    out.append(n[0])
            return out
        self.assertEqual(maps(self.paths[0]), [(24, 4), (24, 3), (26, 3), (26, 1), (26, 10)])
        self.assertEqual(maps(self.paths[1]), [(26, 10), (26, 1), (26, 3), (24, 3), (24, 4), (24, 5)])
        self.assertEqual(maps(self.paths[2])[:3], [(24, 5), (24, 4), (24, 3)])
        self.assertEqual(maps(self.paths[2])[-3:], [(26, 11), (10, 5), (10, 7)])
        self.assertEqual(self.paths[2][-1], rg.GOALS[0])

    def test_phase_goals_are_where_the_game_scripts_put_the_player(self):
        # emulator-verified: entering Mr. Pokemon's house walks the player to (3, 6) facing Mr. Pokemon at (3, 5), and giving
        # Elm the egg is done from (5, 3) facing Elm at (5, 2)
        self.assertEqual(rg.EGG_RECEIVE_GOAL, ((26, 10), 3, 6))
        self.assertEqual(rg.EGG_DELIVER_GOAL, ((24, 5), 5, 3))
        self.assertEqual(rg.path_progress(fake(rg.PATH_START), egg=True), 0.0)

    def test_phase_from_flags(self):
        node = rg.PATH_START
        self.assertEqual([rg.egg_phase(fake(node, f)) for f in ((), (30,), (30, 31), (30, 31, 1812), (1812,))], [0, 1, 2, 2, 2])
        self.assertEqual(rg.egg_phase(SimpleNamespace(map_group=24, map_number=4, x=1, y=1)), 0)   # no has_event: phase 0

    def test_choke_is_blocked_until_the_egg_is_given(self):
        g = self.g
        d0, d1, d2 = g.phase_dist

        def forward_reach(start, blocked):
            seen, q = {start}, deque([start])
            while q:
                n = q.popleft()
                for o in g.fwd.get(n, ()):
                    if o is not None and o not in seen and o not in blocked:
                        seen.add(o)
                        q.append(o)
            return seen
        entry = ((26, 1), 7, 53)
        blocked = set(rg.NPC_BLOCKERS)
        self.assertEqual(blocked, {((26, 1), 5, y) for y in (23, 24, 25, 26)})
        closed = forward_reach(entry, blocked)
        opened = forward_reach(entry, set())
        route31 = {n for n in opened if n[0] == (26, 2)}
        self.assertTrue(route31)
        self.assertFalse(closed & route31)                                       # 1812 clear: the left half is cut off
        self.assertFalse({n for n in closed if n[0] == (26, 11) or n[0][0] == 10 and n[0][1] in (5, 7)})
        self.assertIn(((26, 10), 3, 6), closed)                                  # Mr. Pokemon's house stays reachable
        self.assertTrue(blocked.isdisjoint(d0) and blocked.isdisjoint(d1))      # phases 0 / 1 never route through them
        self.assertTrue(blocked & set(d2) == blocked)                            # phase 2: the opened graph

    def test_progress_is_continuous_across_the_phase_borders_and_never_jumps(self):
        p0, p1, p2 = self.paths
        prog = lambda node, ph: rg.path_progress(fake(node, phase_flags(ph)), egg=True)   # noqa: E731
        self.assertEqual(prog(p0[0], 0), 0.0)
        self.assertEqual(prog(p0[-1], 0), 185.0)
        self.assertEqual(prog(p1[0], 1), 185.0)           # the same tile one phase later: no free jump
        self.assertEqual(prog(p1[-1], 1), 368.0)
        self.assertEqual(prog(p2[0], 2), 368.0)
        self.assertEqual(prog(p2[-1], 2), 695.0)
        seq = [prog(n, 0) for n in p0] + [prog(n, 1) for n in p1[1:]] + [prog(n, 2) for n in p2[1:]]
        self.assertEqual(seq, [float(i) for i in range(696)])   # +1 per step over the whole errand
        # the flag-off measure of an unflagged state is the old one
        self.assertEqual(rg.path_progress(fake(rg.PATH_START)), 0.0)

    def test_off_graph_positions_have_no_progress(self):
        self.assertIsNone(rg.path_progress(fake(((1, 1), 4, 4)), egg=True))
        self.assertIsNone(rg.path_progress(fake(((24, 4), 500, 500)), egg=True))

    def test_best_ever_reward_telescopes_over_the_three_phases(self):
        p0, p1, p2 = self.paths
        states = [fake(n) for n in p0] + [fake(n, (30,)) for n in p1[1:]] + [fake(n, phase_flags(2)) for n in p2[1:]]
        tr = pp.PathProgress(SCALE, egg=True)
        tr.reset(states[0])
        paid = [tr.update(a, b) for a, b in zip(states, states[1:])]
        self.assertAlmostEqual(sum(paid), SCALE * 695, places=6)
        self.assertEqual(tr.best, 695.0)
        self.assertEqual(sum(1 for v in paid if v == 0.0), 0)
        # a round trip pays nothing; retreating is charged in combined tiles; a phase-2 start keeps its offset free
        tr = pp.PathProgress(SCALE, retreat_frac=0.35, egg=True)
        tr.reset(fake(p1[40], (30,)))
        self.assertEqual(tr.best, 185.0 + 40)
        self.assertEqual(tr.update(fake(p1[40], (30,)), fake(p1[41], (30,))), SCALE)
        self.assertEqual(tr.update(fake(p1[41], (30,)), fake(p1[40], (30,))), 0.0)
        self.assertAlmostEqual(tr.retreat_step, -0.35 * SCALE, places=9)
        self.assertEqual(tr.update(fake(p1[40], (30,)), fake(p1[41], (30,))), 0.0)
        self.assertEqual(tr.update(fake(p1[41], (30,)), fake(p1[42], (30,))), SCALE)
        # white-out neither pays nor moves the best
        w = fake(p1[43], (30,))
        w.fainted = True
        self.assertEqual(tr.update(fake(p1[42], (30,)), w), 0.0)
        self.assertEqual(tr.best, 185.0 + 42)

    def test_flag_off_tracker_is_untouched(self):
        tr = pp.PathProgress(0.08)
        self.assertFalse(tr.egg)
        tr.reset(fake(rg.PATH_START))
        self.assertEqual(tr.best, 0.0)
        self.assertEqual(pp.total(), 309.0)
        self.assertEqual(pp.total(egg=True), 695.0)
        self.assertEqual(pp.progress(fake(rg.PATH_START)), 0.0)
        self.assertIsNone(pp.PathProgress.from_scale(None, egg=True))
        self.assertTrue(pp.PathProgress.from_scale(0.3, egg=True).egg)

    def test_progress_function_is_fast(self):
        import time
        node = self.paths[0][100]
        s = fake(node)
        t0 = time.perf_counter()
        for _ in range(20000):
            rg.path_progress(s, egg=True)
        self.assertLess(time.perf_counter() - t0, 2.5)


@unittest.skipUnless(HAVE_ROM and (ROOT / "states" / "mystery_egg_received.state").exists(), "needs the ROM and the saved states")
class RealEmulatorTests(unittest.TestCase):
    """Flag reads and phase detection on real saved states, and the worker's reward / milestone bookkeeping."""

    @classmethod
    def setUpClass(cls):
        from src.emulator import Emulator
        from src.ram_parser import RamParser
        cls.emu = Emulator(headless=True, rtc_lock=True)
        cls.parser = RamParser(cls.emu)

    def setUp(self):
        self.addCleanup(sp.set_profile, False)

    def load(self, name):
        self.emu.load_state(ROOT / "states" / f"{name}.state")
        self.emu.tick(5, render=False)
        return self.parser.parse(self.emu.frame)

    def set_flags(self, on=(), off=()):
        b = bytearray(self.emu.read(1, 0xDA72, 256))
        for f in on:
            b[f >> 3] |= 1 << (f & 7)
        for f in off:
            b[f >> 3] &= ~(1 << (f & 7))
        self.emu.write(1, 0xDA72, bytes(b))
        self.emu.tick(2, render=False)
        return self.parser.parse(self.emu.frame)

    def test_phase_detection_on_real_states(self):
        s0 = self.load("new_bark_outside")
        self.assertEqual((rg.egg_phase(s0), s0.has_event(30), s0.has_event(31), s0.has_event(1812)), (0, False, False, False))
        self.assertEqual(rg.egg_phase(self.load("agent_map_26_1")), 0)
        s1 = self.load("mystery_egg_received")                    # in Mr. Pokemon's house with the egg
        self.assertEqual(((s1.map_group, s1.map_number), rg.egg_phase(s1), s1.has_event(30), s1.has_event(31)), ((26, 10), 1, True, False))
        self.assertEqual(rg.path_progress(s1, egg=True), float(185 + max(0, 183 - self.g_dist(1, s1))))
        s2 = self.set_flags(on=(31, 1812), off=(1813,))           # what the Elm script does (ElmsLab.asm:345)
        self.assertEqual((rg.egg_phase(s2), s2.has_event(31), s2.has_event(1812), s2.has_event(1813)), (2, True, True, False))

    def g_dist(self, phase, s):
        return rg._graph().phase_dist[phase].get(((s.map_group, s.map_number), s.x, s.y), 10 ** 6)

    def test_milestone_detection_with_forced_flags_on_a_real_state(self):
        with sp.use_profile(True):
            s = self.load("agent_map_26_1")
            self.assertEqual(sum(sp.milestones_done(s, {(24, 3), (26, 3)})), 3)   # Route 29, Cherrygrove, Route 30 (current map)
            s = self.load("mystery_egg_received")
            self.assertEqual(sp.milestones_done(s), [True] * 5 + [False] * 6)
            self.assertEqual(sp.current_stage(s), 5)
            s = self.set_flags(on=(31, 1812), off=(1813,))
            self.assertEqual(sp.milestones_done(s), [True] * 6 + [False] * 5)
            self.assertEqual(sp.current_stage(s), 6)
            self.assertEqual(sp.implied_maps(s), {(24, 3), (26, 3), (26, 1), (26, 10)})

    def test_worker_restores_the_egg_state_and_pays_the_delivery_once(self):
        from src.rl_env import ACTIONS
        from tools import train_deep_ppo as t
        w = t.EnvWorker(0, ["mystery_egg_received"], max_steps=50, cfg={"egg_errand_route": True, "path_progress": SCALE,
                                                                       "retreat": {"frac": 0.35, "budget": 15.0}})
        self.addCleanup(w.close)
        self.addCleanup(sp.set_profile, False)
        screen, ram = w.reset()
        self.assertEqual(ram.shape, (80,))
        self.assertEqual(list(ram[17:28]), [1.0] * 5 + [0.0] * 6)   # story bits: the milestones already passed
        self.assertTrue({(24, 3), (26, 3), (26, 1), (26, 10)} <= w.seen_maps)   # nothing is repaid for them
        self.assertEqual(w._ep_start_stage, 5)
        start_best = w.path_progress.best
        phase1 = rg._graph().phase_off[1] + rg._graph().phase_len[1] - self.g_dist(1, w.reader.parse(w.emu.frame))
        self.assertEqual(start_best, float(phase1))
        *_, info = w.step(ACTIONS.index("b"))
        self.assertEqual(info["reward_components"]["path"], 0.0)
        self.assertEqual(info["reward_components"].get("milestone", 0.0), 0.0)
        self.assertEqual(info["reward_components"].get("event", 0.0), 0.0)
        # the Elm script's flags appear: +30 for the delivery, milestone 5 reached, the observation shows it
        b = bytearray(w.emu.read(1, 0xDA72, 256))
        for f in (31, 1812):
            b[f >> 3] |= 1 << (f & 7)
        b[1813 >> 3] &= ~(1 << (1813 & 7))
        w.emu.write(1, 0xDA72, bytes(b))
        *_, info = w.step(ACTIONS.index("b"))
        self.assertGreaterEqual(info["reward_components"]["event"], 30.0)
        self.assertLessEqual(info["reward_components"]["event"], 30.0 + rg_unnamed() + 1e-9)
        self.assertEqual(info["cur_furthest"], 6)
        self.assertEqual([m[0] for m in info["ms_new"]], [5])
        *_, info = w.step(ACTIONS.index("b"))
        self.assertEqual(info["reward_components"].get("event", 0.0), 0.0)   # never again


@unittest.skipUnless(HAVE_ROM and (ROOT / "states" / "mystery_egg_received.state").exists(), "needs the ROM and the saved states")
class EggRouteCheckpointTests(unittest.TestCase):
    """A route checkpoint taken in phase 1 (egg in hand) restores the egg flags, the phase and the passed milestones."""

    def test_phase_one_checkpoint_restores_flags_and_does_not_repay(self):
        from src import route_ckpt as rc
        from src.rl_env import ACTIONS
        from tools import train_deep_ppo as t
        self.addCleanup(sp.set_profile, False)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"egg_errand_route": True, "path_progress": SCALE, "route_dir": str(Path(tmp) / "route"), "route_prob": 1.0,
                   "route_every": 12}
            w = t.EnvWorker(0, ["new_bark_outside"], max_steps=50, cfg=cfg)
            self.addCleanup(w.close)
            w.reset()                                                        # the first episode is always the start state
            self.assertEqual(w.state_name, "new_bark_outside")
            self.assertEqual(rg.egg_phase(w.reader.parse(w.emu.frame)), 0)
            data = (ROOT / "states" / "mystery_egg_received.state").read_bytes()
            w.emu.load_state(data)
            w.emu.tick(5, render=False)
            st = w.reader.parse(w.emu.frame)
            phase, progress = rg.egg_phase(st), pp.progress(st, True)
            self.assertEqual(phase, 1)
            stage = rc.milestone_stage({(24, 3), (26, 3), (26, 1), (26, 10)})
            self.assertTrue(w.route_archive.save(rc.bucket_of(progress, 12), stage, data, phase=phase))
            self.assertEqual(rc.key_phase(w.route_archive.keys()[0]), 1)
            w.reset()                                                        # route_prob 1: loads the checkpoint
            self.assertTrue(w.state_name.startswith("route:"), w.state_name)
            now = w.reader.parse(w.emu.frame)
            self.assertEqual((rg.egg_phase(now), now.has_event(30), now.has_event(31)), (1, True, False))
            self.assertEqual(w.path_progress.best, progress)                  # the state's own progress is free
            self.assertEqual(w._route_saved_bucket, rc.bucket_of(progress, 12))
            self.assertTrue({(24, 3), (26, 3), (26, 1), (26, 10)} <= w.seen_maps)
            *_, info = w.step(ACTIONS.index("b"))
            for key in ("path", "milestone", "event"):
                self.assertEqual(info["reward_components"].get(key, 0.0), 0.0, key)


if __name__ == "__main__":
    unittest.main()
