"""long_v6: route-checkpoint archive (bucketing, saving, eviction, sampling, reset order, baseline / milestone
consistency), dialogue-loop penalty (detection and negative cases), reward-scaler warm-up, and preset isolation
(long_v3 / v4 / v5 must stay exactly as they were)."""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import dialogue_loop as dl
from src import route_ckpt as rc
from src.config import ROM_PATH
from src.dialogue_loop import DialogueLoopTracker
from src.ppo_update import RunningReturnScaler
from dataclasses import replace as dataclasses_replace
from types import SimpleNamespace
from unittest import mock

from src import path_progress as pp
from src.stagnation import StagnationTracker
from src.ram_parser import GameState
from tests.test_path_progress import PathFixture
from src.rl_env import ACTIONS, progress_reward
from src.train_config import PRESETS, ConfigError, apply_overrides, from_dict, load_config
from tools import cloud_train_deep as c
from tools import train_deep_ppo as t

ROOT = Path(__file__).resolve().parents[1]
HAVE_ROM = Path(ROM_PATH).exists()
A = ACTIONS.index("a")

V6_FIELDS = ("route_ckpt", "dialogue_loop_penalty", "retreat_penalty", "stagnation_penalty", "terminate_on_whiteout")
V6_ALL = ("route_ckpt", "route_ckpt_every", "route_ckpt_prob", "route_ckpt_per_bucket", "route_ckpt_max",
          "route_ckpt_weight_floor", "dialogue_loop_penalty", "dialogue_loop_window", "dialogue_loop_max_steps",
          "dialogue_loop_max_talks", "dialogue_loop_cost", "dialogue_loop_budget", "warmup_rollout")


def _gs(**kw):
    base = {"map_group": 24, "map_number": 3, "x": 3, "y": 9, "tiles": {}, "event_flags": [0] * 300,
            "party_count": 1, "party_species": [155], "party_hp": [[19, 19]], "party_levels": [5],
            "party_moves": [[33, 43, 0, 0]], "items": [], "item_counts": []}
    base.update(kw)
    return GameState.from_dict(base)


def _set_age(path: Path, age: int) -> None:
    """Make ``path`` look ``age`` seconds old (a larger age = older); mtime resolution differs per platform."""
    ts = 1_700_000_000 - age
    os.utime(path, (ts, ts))


# ------------------------------------------------------------------------------------------------ presets / config
class PresetTests(unittest.TestCase):
    def test_long_v6_values(self):
        self.assertIn("long_v6", PRESETS)
        cfg = load_config("long_v6")
        self.assertEqual(cfg.name, "long_v6")
        self.assertTrue(cfg.route_ckpt)
        self.assertEqual((cfg.route_ckpt_every, cfg.route_ckpt_per_bucket, cfg.route_ckpt_max), (12, 4, 60))
        self.assertAlmostEqual(cfg.route_ckpt_prob, 0.4)
        self.assertAlmostEqual(cfg.route_ckpt_weight_floor, 0.2)
        self.assertAlmostEqual(cfg.swarm_prob, 0.25)
        self.assertAlmostEqual(cfg.battle_archive_prob, 0.15)
        self.assertLessEqual(cfg.swarm_prob + cfg.battle_archive_prob + cfg.route_ckpt_prob, 1.0)
        self.assertTrue(cfg.dialogue_loop_penalty)
        self.assertEqual((cfg.dialogue_loop_window, cfg.dialogue_loop_max_steps, cfg.dialogue_loop_max_talks),
                         (200, 60, 6))
        self.assertAlmostEqual(cfg.dialogue_loop_cost, 0.02)
        self.assertAlmostEqual(cfg.dialogue_loop_budget, 10.0)
        self.assertEqual(cfg.warmup_rollout, 1)
        self.assertTrue(cfg.retreat_penalty)
        self.assertEqual((cfg.retreat_cost_frac, cfg.retreat_budget), (0.35, 15.0))
        self.assertTrue(cfg.stagnation_penalty)
        self.assertEqual((cfg.stagnation_after, cfg.stagnation_min, cfg.stagnation_max, cfg.stagnation_ramp,
                          cfg.stagnation_budget), (250, 0.004, 0.02, 300, 10.0))
        self.assertTrue(cfg.terminate_on_whiteout)

    def test_long_v6_only_differs_from_long_v5_in_the_new_fields(self):
        a, b = load_config("long_v5").to_dict(), load_config("long_v6").to_dict()
        diff = {k for k in a if a[k] != b[k]}
        # the other v6 fields keep their dataclass defaults, which long_v6 sets explicitly to the same values
        self.assertEqual(diff, {"name", "swarm_prob", "route_ckpt", "route_ckpt_prob", "dialogue_loop_penalty",
                                "warmup_rollout", "retreat_penalty", "stagnation_penalty", "terminate_on_whiteout"})

    def test_long_v6_keeps_the_v5_anti_collapse_settings(self):
        cfg = load_config("long_v6")
        self.assertTrue(cfg.disable_bump_penalty and cfg.disable_loop_penalty and cfg.idle_press_penalty)
        self.assertAlmostEqual(cfg.path_progress_scale, 0.3)
        self.assertEqual((cfg.entropy_start, cfg.entropy_end, cfg.entropy_anneal_updates), (0.02, 0.005, 1200))
        self.assertTrue(cfg.entropy_boost)

    def test_older_presets_are_unchanged_and_have_every_v6_flag_off(self):
        for name in ("blueprint_v1", "legacy", "long_v1", "long_v2", "long_v3", "long_v4", "long_v5"):
            cfg = load_config(name)
            for f in V6_FIELDS:
                self.assertFalse(getattr(cfg, f), f"{name}.{f}")
            self.assertEqual(cfg.warmup_rollout, 0, name)
            self.assertIsNone(cfg.dialogue_loop_params(), name)
            env = t.worker_env_cfg(cfg, "swarm", "battle", "route")
            self.assertIsNone(env["route_dir"], name)
            self.assertEqual(env["route_prob"], 0.0, name)
            self.assertIsNone(env["dialogue_loop"], name)
            self.assertIsNone(env["retreat"], name)
            self.assertIsNone(env["stagnation"], name)
            self.assertFalse(env["terminate_on_whiteout"], name)
        v5, v3 = load_config("long_v5"), load_config("long_v3")
        self.assertAlmostEqual(v5.swarm_prob, 0.5)
        self.assertAlmostEqual(v5.battle_archive_prob, 0.15)
        self.assertEqual((v3.swarm_prob, v3.battle_archive_prob), (0.5, 0.15))
        self.assertAlmostEqual(load_config("long_v4").path_progress_scale, 0.08)

    def test_validation(self):
        for bad in ({"route_ckpt_every": 0}, {"route_ckpt_prob": 1.5}, {"route_ckpt_per_bucket": 0},
                    {"route_ckpt_max": 0}, {"route_ckpt_weight_floor": 0.0}, {"dialogue_loop_window": 0},
                    {"dialogue_loop_max_steps": 0}, {"dialogue_loop_max_talks": -1}, {"dialogue_loop_cost": -0.1},
                    {"dialogue_loop_budget": -1.0}, {"warmup_rollout": -1},
                    {"retreat_cost_frac": -0.1}, {"retreat_budget": -1.0}, {"stagnation_after": 0},
                    {"stagnation_min": -0.1}, {"stagnation_max": 0.001}, {"stagnation_ramp": 0},
                    {"stagnation_budget": -1.0}, {"dwell_penalty": False},   # stagnation needs the dwell penalty
                    {"route_ckpt_prob": 0.7},                    # 0.25 + 0.15 + 0.7 > 1
                    {"path_progress": False}):                   # route_ckpt needs path_progress
            with self.assertRaises(ConfigError, msg=str(bad)):
                apply_overrides(load_config("long_v6"), bad)
        # an unused archive does not count towards the share; swarm off frees its share
        apply_overrides(load_config("long_v6"), {"route_ckpt_prob": 0.7, "swarm": False})
        with self.assertRaises(ConfigError):
            from_dict({"route_ckpt": True})   # blueprint_v1 has no path_progress

    def test_each_feature_can_be_ablated(self):
        cfg = apply_overrides(load_config("long_v6"), {"route_ckpt": False, "dialogue_loop_penalty": False,
                                                       "warmup_rollout": 0})
        env = t.worker_env_cfg(cfg, "s", "b", "r")
        self.assertIsNone(env["route_dir"])
        self.assertIsNone(env["dialogue_loop"])

    def test_worker_env_cfg_for_long_v6(self):
        env = t.worker_env_cfg(load_config("long_v6"), "s", "b", "r")
        self.assertEqual(env["route_dir"], "r")
        self.assertAlmostEqual(env["route_prob"], 0.4)
        self.assertEqual(env["dialogue_loop"], {"window": 200, "max_steps": 60, "max_talks": 6, "cost": 0.02,
                                                "budget": 10.0})

    def test_config_is_accepted_by_the_trainer_cli_and_the_cloud_launcher(self):
        args = t.build_parser().parse_args(["--config", "long_v6", "--envs", "2", "--updates", "3", "--rollout", "64"])
        cfg = t.build_config(args)
        self.assertEqual((cfg.name, cfg.envs, cfg.updates, cfg.rollout_steps), ("long_v6", 2, 3, 64))
        self.assertEqual(c.resolve_config(argparse.Namespace(config="long_v6")), ("long_v6", False))
        args = argparse.Namespace(updates=1600, rollout=128, max_steps=1024, batch_size=256, lr=3e-4,
                                  state="new_bark_outside", save="s.pt", envs=24, curriculum=False, recurrent=True,
                                  states="", config="long_v6", explicit={"updates", "envs", "state", "curriculum"},
                                  resume=False, run_id="v6-1")
        self.assertIn("--config long_v6", c.build_remote_train_cmd(args, 24, "long_v6"))

    def test_config_chain_imports_without_numpy_or_torch(self):
        code = ("import sys; sys.modules['numpy'] = None; sys.modules['torch'] = None; "
                "from src.train_config import PRESETS, load_config; "
                "assert load_config('long_v6').route_ckpt; print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)


# ------------------------------------------------------------------------------------------------ route archive
class BucketTests(unittest.TestCase):
    def test_bucket_of(self):
        self.assertEqual([rc.bucket_of(p, 12) for p in (None, -3, 0, 11.9, 12, 23, 24, 97)], [0, 0, 0, 0, 1, 1, 2, 8])

    def test_bucket_weight_favours_the_furthest_with_a_floor(self):
        self.assertAlmostEqual(rc.bucket_weight(8, 8, 0.2), 1.0)
        self.assertAlmostEqual(rc.bucket_weight(0, 8, 0.2), 0.2)
        self.assertAlmostEqual(rc.bucket_weight(4, 8, 0.2), 0.6)
        self.assertAlmostEqual(rc.bucket_weight(3, 0, 0.2), 1.0)   # a single bucket: nothing to weigh

    def test_key_info_and_stage_helpers(self):
        self.assertEqual(rc.key_info("rc_b007_2_s3"), (7, 2, 3))
        self.assertIsNone(rc.key_info("stage1_0"))
        self.assertEqual(rc.milestone_stage({(24, 3)}), 1)
        self.assertEqual(rc.milestone_stage({(24, 3), (26, 3), (99, 99)}), 2)
        self.assertEqual(rc.milestone_stage({(24, 4)}), 0)
        self.assertEqual(rc.stage_maps(2), {(24, 3), (26, 3)})
        self.assertEqual(rc.stage_maps(0), set())


class RouteArchiveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "route"

    def make(self, **kw):
        return rc.RouteCheckpointArchive(self.dir, **kw)

    def test_empty(self):
        a = self.make()
        self.assertEqual((a.keys(), a.counts(), a.max_bucket()), ([], {}, 0))
        self.assertIsNone(a.sample(np.random.default_rng(0)))

    def test_save_fills_slots_then_replaces_the_oldest_in_the_bucket(self):
        a = self.make(per_bucket=3)
        for i in range(3):
            self.assertTrue(a.save(2, 1, f"s{i}".encode()))
        self.assertEqual(a.counts(), {2: 3})
        for p in sorted(self.dir.glob("rc_b002_*.state")):   # slot n is n * 10 s old, so slot 2 is the oldest
            _set_age(p, 10 * int(p.stem.split("_")[2]))
        self.assertTrue(a.save(2, 1, b"new"))
        self.assertEqual(a.counts(), {2: 3})                 # bounded per bucket
        self.assertEqual(sorted(p.name for p in self.dir.glob("*.state")),
                         ["rc_b002_0_s1.state", "rc_b002_1_s1.state", "rc_b002_2_s1.state"])
        self.assertEqual((self.dir / "rc_b002_2_s1.state").read_bytes(), b"new")   # the oldest slot was replaced
        self.assertEqual((self.dir / "rc_b002_0_s1.state").read_bytes(), b"s0")

    def test_replacing_a_slot_with_a_different_stage_leaves_one_file(self):
        a = self.make(per_bucket=1)
        a.save(1, 0, b"old")
        a.save(1, 2, b"new")
        self.assertEqual(a.keys(), ["rc_b001_0_s2"])
        self.assertEqual(a.load("rc_b001_0_s2"), (b"new", 2))

    def test_global_bound_evicts_from_the_fullest_bucket_oldest_first(self):
        a = self.make(per_bucket=4, max_files=6)
        for b in (1, 2, 3):
            a.save(b, 0, b"x")
        for _ in range(3):
            a.save(2, 0, b"y")             # bucket 2 now holds 4 files, total 6
        self.assertEqual(a.counts(), {1: 1, 2: 4, 3: 1})
        files = sorted(self.dir.glob("rc_b002_*.state"), key=lambda p: p.name)
        for i, p in enumerate(files):
            _set_age(p, 100 - i)           # slot 0 is the oldest
        a.save(4, 0, b"z")                 # 7 > 6 -> the oldest file of the fullest bucket (2) goes
        self.assertEqual(a.counts(), {1: 1, 2: 3, 3: 1, 4: 1})
        self.assertFalse((self.dir / "rc_b002_0_s0.state").exists())
        self.assertEqual(sum(a.counts().values()), 6)

    def test_never_exceeds_max_files_and_keeps_the_furthest_bucket(self):
        a = self.make(per_bucket=4, max_files=10)
        for rnd in range(5):
            for b in range(1, 9):
                a.save(b, 0, b"x")
        self.assertLessEqual(sum(a.counts().values()), 10)
        self.assertEqual(a.max_bucket(), 8)
        self.assertTrue(all(n >= 1 for n in a.counts().values()))    # every bucket keeps a state

    def test_load_returns_bytes_and_stage_and_none_when_evicted(self):
        a = self.make()
        a.save(3, 2, b"data")
        self.assertEqual(a.load("rc_b003_0_s2"), (b"data", 2))
        self.assertIsNone(a.load("rc_b003_1_s2"))
        self.assertIsNone(a.load("garbage"))

    def test_sampling_weights_favour_far_buckets_with_a_floor(self):
        a = self.make(floor=0.2)
        for b in (1, 5, 9):
            a.save(b, 0, b"x")
        rng = np.random.default_rng(0)
        n = 6000
        hits = {1: 0, 5: 0, 9: 0}
        for _ in range(n):
            hits[rc.key_info(a.sample(rng))[0]] += 1
        w = {b: rc.bucket_weight(b, 9, 0.2) for b in hits}
        total = sum(w.values())
        for b in hits:
            self.assertAlmostEqual(hits[b] / n, w[b] / total, delta=0.03, msg=str(b))
        self.assertLess(hits[1], hits[5])
        self.assertLess(hits[5], hits[9])
        self.assertGreater(hits[1], n * 0.05)   # the floor: early buckets are still drawn

    def test_sample_is_uniform_inside_a_bucket(self):
        a = self.make()
        for i in range(4):
            a.save(3, 0, f"{i}".encode())
        rng = np.random.default_rng(1)
        seen = {a.sample(rng) for _ in range(200)}
        self.assertEqual(len(seen), 4)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            self.make(per_bucket=0)


# ------------------------------------------------------------------------------------------------ reset order
class FixedRng:
    def __init__(self, *values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0) if self.values else 0.5

    def integers(self, lo, hi):
        return lo


class SelectorOrderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.swarm = t.SwarmArchive(root / "swarm")
        self.swarm.save("stage1_0", b"s")
        self.battle = t.SwarmArchive(root / "battle")
        self.battle.save("m1_1-1_s1", b"b")
        self.route = rc.RouteCheckpointArchive(root / "route")
        self.route.save(2, 1, b"r")

    def selector(self, rng, route=True, swarm=True, battle=True):
        return t.StartStateSelector(
            ["start_a"], 0, None, self.swarm if swarm else None, 0.25, rng,
            battle_archive=self.battle if battle else None, battle_prob=0.15 if battle else 0.0,
            route_archive=self.route if route else None, route_prob=0.4)

    def test_order_is_battle_then_swarm_then_route_then_start_state(self):
        sel = self.selector(FixedRng(0.5))
        self.assertEqual(sel.pick(), "start_a")                       # first episode: always a start state
        picks = []
        for u in (0.10, 0.20, 0.39, 0.40, 0.79, 0.80, 0.99):
            sel.rng = FixedRng(u, 0.0, 0.0)
            picks.append(sel.pick().split(":")[0])
        self.assertEqual(picks, ["battle", "swarm", "swarm", "route", "route", "start_a", "start_a"])

    def test_route_pick_names_a_checkpoint_key(self):
        sel = self.selector(FixedRng(0.5))
        sel.pick()
        sel.rng = FixedRng(0.6, 0.0, 0.0)
        self.assertEqual(sel.pick(), "route:rc_b002_0_s1")

    def test_empty_route_archive_falls_through_to_a_start_state(self):
        for f in self.route.dir.glob("*.state"):
            f.unlink()
        sel = self.selector(FixedRng(0.5))
        sel.pick()
        sel.rng = FixedRng(0.6, 0.0, 0.0)
        self.assertEqual(sel.pick(), "start_a")

    def test_route_share_is_prob_when_battle_and_swarm_are_off(self):
        sel = self.selector(np.random.default_rng(3), swarm=False, battle=False)
        sel.pick()
        n = 3000
        route = sum(sel.pick().startswith("route:") for _ in range(n))
        self.assertAlmostEqual(route / n, 0.4, delta=0.04)

    def test_without_a_route_archive_the_draws_are_exactly_the_old_ones(self):
        a = self.selector(np.random.default_rng(5), route=False)
        b = t.StartStateSelector(["start_a"], 0, None, self.swarm, 0.25, np.random.default_rng(5),
                                 battle_archive=self.battle, battle_prob=0.15)
        self.assertEqual([a.pick() for _ in range(300)], [b.pick() for _ in range(300)])


# ------------------------------------------------------------------------------------------------ dialogue loop
def _talk(tr, key_x=4, steps=2, facing="up", a_start=True, **extra):
    """One talk from the plain overworld: the A press that opens text, ``steps - 1`` auto-advance steps, last one closes."""
    total = 0.0
    before = _gs(x=key_x, facing=facing, **extra)
    opened = _gs(x=key_x, facing=facing, script_running=1, **extra)
    total += tr.update(before, opened, a_start)
    for i in range(steps - 1):
        nxt = opened if i < steps - 2 else before
        total += tr.update(opened, nxt, False)
    return total


class DialogueLoopTests(unittest.TestCase):
    def tr(self, **kw):
        return DialogueLoopTracker(**{"window": 200, "max_steps": 60, "max_talks": 6, "cost": 0.02, "budget": 10.0, **kw})

    def test_first_talk_and_the_first_repeats_are_free(self):
        tr = self.tr()
        for _ in range(7):    # 1 first-time talk + 6 repeats: rule B needs MORE than 6 effect-less earlier talks
            self.assertEqual(_talk(tr), 0.0)

    def test_same_tile_talked_to_more_than_m_times_without_effect_is_charged(self):
        tr = self.tr()
        for _ in range(7):
            _talk(tr)
        pen = _talk(tr, steps=3)      # the 8th talk: 7 effect-less talks before it (> 6)
        self.assertAlmostEqual(pen, -0.02 * 3)
        self.assertEqual(tr.charged, 3)

    def test_talks_with_an_effect_reset_the_same_tile_count(self):
        tr = self.tr()
        for _ in range(7):
            _talk(tr)
        # this talk changes the bag: not charged here (progress), and the tile count starts again afterwards
        before, opened = _gs(x=4, facing="up"), _gs(x=4, facing="up", script_running=1)
        got = _gs(x=4, facing="up", items=[4], item_counts=[1])
        total = tr.update(before, opened, True) + tr.update(opened, got, False)
        self.assertLessEqual(total, 0.0)                               # the already-looping start may be charged once
        self.assertEqual(_talk(tr), 0.0)                               # counter reset by the productive talk

    def test_a_different_tile_is_a_different_interaction(self):
        tr = self.tr()
        for i in range(30):
            self.assertEqual(_talk(tr, key_x=4 + (i % 10)), 0.0, i)   # 10 tiles x 3 talks: nothing exceeds 6

    def test_window_rule_charges_after_more_than_k_countable_steps(self):
        tr = self.tr(max_talks=10_000, window=200, max_steps=10)   # isolate rule A
        pens = [_talk(tr, steps=2) for _ in range(12)]
        # talk 1 is first-time (not counted); every later talk adds 2 countable steps -> > 10 after talk 7
        self.assertTrue(all(p == 0.0 for p in pens[:6]), pens)
        self.assertLess(sum(pens[6:]), 0.0)

    def test_progress_signals_clear_the_window(self):
        tr = self.tr(max_talks=10_000, max_steps=10)
        for _ in range(6):
            _talk(tr)
        self.assertGreater(tr.window_count, 0)
        tr.update(_gs(), _gs(map_number=4), False)                    # a map change
        self.assertEqual(tr.window_count, 0)
        for change in ({"event_flags": [1] + [0] * 299}, {"items": [4], "item_counts": [1]},
                       {"party_species": [155, 16], "party_count": 2}, {"money": 5}):
            for _ in range(6):
                _talk(tr)
            self.assertGreater(tr.window_count, 0)
            tr.update(_gs(), _gs(**change), False)
            self.assertEqual(tr.window_count, 0, change)

    def test_budget_caps_the_total_and_reset_starts_fresh(self):
        tr = self.tr(cost=0.5, budget=1.0, max_talks=0)
        total = sum(_talk(tr) for _ in range(9))   # talks 2.. are charged 0.5 per step, the budget stops it at 1.0
        self.assertAlmostEqual(total, -1.0)
        self.assertAlmostEqual(tr.spent, 1.0)
        tr.reset()
        self.assertEqual((tr.spent, tr.charged, tr.window_count), (0.0, 0, 0))
        self.assertEqual(_talk(tr), 0.0)

    def test_window_slides(self):
        tr = self.tr(window=10, max_steps=4, max_talks=10_000)
        for _ in range(8):
            _talk(tr)
        for _ in range(12):
            tr.update(_gs(), _gs(x=9), False)                          # walking: pushes False
        self.assertEqual(tr.window_count, 0)

    # -- negative cases
    def test_battle_text_is_never_counted_or_charged(self):
        tr = self.tr(max_talks=0, max_steps=1)
        for _ in range(100):
            self.assertEqual(tr.update(_gs(battle_mode=1, script_running=1), _gs(battle_mode=1, script_running=1), True),
                             0.0)
        self.assertEqual(tr.window_count, 0)

    def test_pokemon_center_is_exempt(self):
        tr = self.tr(max_talks=0, max_steps=1)
        for _ in range(30):
            self.assertEqual(_talk(tr, map_group=26, map_number=5), 0.0)   # Cherrygrove Pokemon Center
        self.assertEqual(tr.charged, 0)

    def test_text_not_opened_by_a_press_is_untracked(self):
        tr = self.tr(max_talks=0, max_steps=1)
        txt = _gs(script_running=1)
        for _ in range(100):
            self.assertEqual(tr.update(txt, txt, False), 0.0)               # a cutscene / scripted text

    def test_pressing_a_without_dialogue_is_not_this_penalty(self):
        tr = self.tr(max_talks=0, max_steps=1)
        for _ in range(100):
            self.assertEqual(tr.update(_gs(), _gs(), True), 0.0)

    def test_params_and_factory(self):
        self.assertIsNone(DialogueLoopTracker.from_params(None))
        self.assertIsInstance(DialogueLoopTracker.from_params(load_config("long_v6").dialogue_loop_params()),
                              DialogueLoopTracker)
        for bad in ({"window": 0}, {"max_steps": 0}, {"max_talks": -1}, {"cost": -1.0}, {"budget": -1.0}):
            with self.assertRaises(ValueError):
                self.tr(**bad)
        self.assertEqual(dl.validate_params(200, 60, 6, 0.02, 10.0), [])


# ------------------------------------------------------------------------------------------------ warm-up
class _StubPolicy:
    use_gru = False

    def __init__(self):
        self.calls = 0

    def eval(self):
        pass

    def __call__(self, screens, rams):
        self.calls += 1
        n = screens.shape[0]
        return torch.zeros(n, 8), torch.zeros(n)


class _StubVecEnv:
    def __init__(self, n=3, rewards=(10.0, -2.0)):
        self.n, self.steps, self.rewards = n, 0, rewards

    def step(self, actions):
        assert len(actions) == self.n
        self.steps += 1
        rew = np.full(self.n, self.rewards[self.steps % len(self.rewards)], dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)
        dones[0] = self.steps % 5 == 0
        return (np.zeros((self.n, 144, 160, 3), np.uint8), np.zeros((self.n, 4)), rew, dones, [1] * self.n, [0] * self.n)


class WarmupTests(unittest.TestCase):
    def run_warmup(self, rollouts=2, steps=8):
        scaler = RunningReturnScaler(3, 0.999)
        env = _StubVecEnv()
        policy = _StubPolicy()
        screens, rams = np.zeros((3, 144, 160, 3), np.uint8), np.zeros((3, 4))
        out = t.warmup_scaler(policy, env, scaler, screens, rams, None, rollouts, steps, torch.device("cpu"))
        return scaler, env, policy, out

    def test_feeds_only_the_scaler(self):
        scaler, env, policy, (screens, rams, hx, info) = self.run_warmup(2, 8)
        self.assertEqual(env.steps, 16)
        self.assertEqual(policy.calls, 16)
        self.assertEqual(info["steps"], 16)
        self.assertGreater(scaler.count, 10)                       # the statistics were updated ...
        self.assertNotAlmostEqual(info["std_before"], info["std_after"])
        self.assertEqual(screens.shape, (3, 144, 160, 3))          # ... and the envs' latest observations are returned
        self.assertIsNone(hx)

    def test_raw_mean_reward_is_reported_per_env_per_rollout(self):
        info = self.run_warmup(1, 4)[3][3]
        self.assertAlmostEqual(info["mean_reward"], -2.0 + 10.0 - 2.0 + 10.0)

    def test_without_a_scaler_it_is_a_harmless_rollout(self):
        env = _StubVecEnv()
        screens, rams = np.zeros((3, 144, 160, 3), np.uint8), np.zeros((3, 4))
        *_, info = t.warmup_scaler(_StubPolicy(), env, None, screens, rams, None, 1, 4, torch.device("cpu"))
        self.assertEqual((env.steps, info["std_before"], info["std_after"]), (4, 1.0, 1.0))

    def test_scaler_std_after_warmup_is_not_the_prior(self):
        scaler, *_ = self.run_warmup(1, 32)
        self.assertNotAlmostEqual(scaler.std, 1.0, places=2)


@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class TrainSmokeTests(unittest.TestCase):
    def test_long_v6_trains_with_a_warmup_that_does_not_count_as_an_update(self):
        with tempfile.TemporaryDirectory() as d:
            metrics = Path(d) / "m.jsonl"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                t.main(["--config", "long_v6", "--envs", "2", "--updates", "2", "--rollout", "16", "--max-steps", "64",
                        "--no-curriculum", "--state", "new_bark_outside", "--no-live", "--run-id", "smoke",
                        "--runs-dir", str(Path(d) / "runs"), "--metrics-path", str(metrics),
                        "--save", str(Path(d) / "smoke.pt")])
            text = out.getvalue()
            self.assertIn("[warmup] 1 rollout(s) x 16 steps fed the reward scaler only", text)
            self.assertIn("[long_v6]", text)
            rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([r["update"] for r in rows], [0, 1])          # the warm-up is not an update
            self.assertIn("route_ckpt_size", rows[0])
            self.assertIn("whiteout_episodes", rows[0])
            self.assertIn("route_ckpt_reset_share", rows[0])
            self.assertEqual(rows[0]["samples"], 2 * 16)                   # only the real rollout is a sample batch


# ------------------------------------------------------------------------------------------------ real emulator
@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class EnvWorkerRouteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "route"

    def make(self, state="new_bark_outside", **cfg):
        base = {"route_dir": str(self.dir), "route_prob": 1.0, "path_progress": 0.3}
        base.update(cfg)
        w = t.EnvWorker(0, [state], max_steps=500, cfg=base)
        self.addCleanup(w.close)
        w.reset()
        return w

    def test_off_without_a_route_dir(self):
        w = t.EnvWorker(0, ["new_bark_outside"], max_steps=50, cfg={"path_progress": 0.3})
        self.addCleanup(w.close)
        w.reset()
        self.assertIsNone(w.route_archive)
        w._maybe_save_route(w.reader.parse(w.emu.frame))   # a no-op
        self.assertFalse(self.dir.exists())

    def test_saves_once_per_bucket_per_episode_and_files_by_current_bucket(self):
        w = self.make()
        st = w.reader.parse(w.emu.frame)
        w.path_progress.current = 5.0
        w._maybe_save_route(st)
        self.assertEqual(w.route_archive.counts(), {})             # bucket 0 is the base start state
        w.path_progress.current = 30.0                              # bucket 2
        w._total_steps += 100
        w._maybe_save_route(st)
        self.assertEqual(w.route_archive.counts(), {2: 1})
        w._total_steps += 100
        w._maybe_save_route(st)                                     # same bucket again in this episode
        self.assertEqual(w.route_archive.counts(), {2: 1})
        w.path_progress.current = 13.0                              # stepped back: bucket 1 < saved bucket 2
        w._total_steps += 100
        w._maybe_save_route(st)
        self.assertEqual(w.route_archive.counts(), {2: 1})
        w.path_progress.current = 37.0                              # bucket 3
        w._total_steps += 100
        w._maybe_save_route(st)
        self.assertEqual(w.route_archive.counts(), {2: 1, 3: 1})

    def test_rate_limit_calm_state_and_whiteout_guard(self):
        w = self.make()
        st = w.reader.parse(w.emu.frame)
        w.path_progress.current = 30.0
        w._total_steps = 10                                         # right after the previous save
        w._last_route_save = 0
        w._maybe_save_route(st)
        self.assertEqual(w.route_archive.counts(), {})              # rate limited, retried later
        w._total_steps = 100
        w._maybe_save_route(_gs(battle_mode=1))
        w._maybe_save_route(_gs(script_running=1))
        w._maybe_save_route(_gs(party_hp=[[0, 19]]))   # all fainted: the white-out teleport
        self.assertEqual(w.route_archive.counts(), {})
        w.path_progress.current = None
        w._maybe_save_route(st)
        self.assertEqual(w.route_archive.counts(), {})
        w.path_progress.current = 30.0
        w._maybe_save_route(st)
        self.assertEqual(w.route_archive.counts(), {2: 1})

    def test_checkpoint_reset_starts_the_baseline_at_the_state_progress_and_pays_nothing_for_it(self):
        w = self.make("route29_entry", path_progress=0.3)
        from src import path_progress as pp
        actual = pp.progress(w.reader.parse(w.emu.frame))
        self.assertIsNotNone(actual)
        w.route_archive.save(rc.bucket_of(actual, 12), 1, w.emu.save_state())
        w.reset()                                                   # k > 0 and route_prob 1 -> the checkpoint
        self.assertTrue(w.state_name.startswith("route:"), w.state_name)
        st = w.reader.parse(w.emu.frame)
        self.assertAlmostEqual(w.path_progress.best, pp.progress(st))   # no free reward for the state's own progress
        self.assertEqual(w._route_saved_bucket, rc.bucket_of(w.path_progress.best, 12))
        self.assertEqual(w.budget.get("cap:explore", 30.0), 30.0)  # budgets are the normal fresh ones
        *_, info = w.step(ACTIONS.index("b"))
        self.assertEqual(info["reward_components"]["path"], 0.0)

    def test_checkpoint_reset_does_not_repay_passed_milestones(self):
        w = self.make("agent_map_26_3")                              # Cherrygrove
        w.route_archive.save(5, 2, w.emu.save_state())              # saved by an episode that had passed Route 29 + Cherrygrove
        w.reset()
        self.assertTrue(w.state_name.startswith("route:"))
        self.assertTrue({(24, 3), (26, 3)} <= w.seen_maps)
        comps = {}
        progress_reward(_gs(map_group=26, map_number=3), _gs(map_group=24, map_number=3), True, set(), w.seen_maps,
                        components=comps)                           # walking back onto Route 29
        self.assertNotIn("milestone", comps)
        control = {}
        progress_reward(_gs(map_group=26, map_number=3), _gs(map_group=24, map_number=3), True, set(), {(26, 3)},
                        components=control)                         # without the seeding it would pay
        self.assertGreater(control.get("milestone", 0.0), 0.0)

    def test_a_vanished_checkpoint_falls_back_to_the_start_state(self):
        w = self.make()
        w.route_archive.save(2, 1, w.emu.save_state())
        w.selector.pick = lambda: "route:rc_b009_0_s0"              # evicted by another worker meanwhile
        w.reset()
        self.assertEqual(w.state_name, "new_bark_outside")
        self.assertEqual(w._route_seed_stage, 0)

    def test_dialogue_loop_component_in_the_real_env(self):
        w = self.make("ilex_src", dialogue_loop={"window": 200, "max_steps": 60, "max_talks": 6, "cost": 0.02,
                                                 "budget": 10.0}, route_dir=None)
        *_, info = w.step(A)
        self.assertIn("dialogue_loop", info["reward_components"])
        self.assertEqual(info["reward_components"]["dialogue_loop"], 0.0)   # the first talk is free
        w2 = t.EnvWorker(0, ["ilex_src"], max_steps=50, cfg={})
        self.addCleanup(w2.close)
        w2.reset()
        *_, info2 = w2.step(A)
        self.assertNotIn("dialogue_loop", info2["reward_components"])


# ------------------------------------------------------------------------------------------------ retreat cost
SCALE = 0.3
FRAC = 0.35


def st2(node, **kw):
    mid, x, y = node
    base = dict(map_group=mid[0], map_number=mid[1], x=x, y=y, fainted=False, battle_mode=0, script_mode=0,
                script_running=0)
    base.update(kw)
    return SimpleNamespace(**base)


class RetreatRealRouteTests(PathFixture):
    """The real 309-step route: the retreat cost never triggers on it, but does in the Route 29 west corridor."""

    def tracker(self, **kw):
        return pp.PathProgress(SCALE, retreat_frac=kw.pop("frac", FRAC), retreat_budget=kw.pop("budget", 15.0))

    def walk(self, nodes, tr=None):
        tr = tr or self.tracker()
        tr.reset(st2(nodes[0]))
        paid, cost = [], []
        for a, b in zip(nodes, nodes[1:]):
            paid.append(tr.update(st2(a), st2(b)))
            cost.append(tr.retreat_step)
        return tr, paid, cost

    def corridor(self):
        """(junction index on the path, tiles from the junction down to the west end of the Route 29 corridor)."""
        e = ((24, 3), 14, 15)   # the camping spot of the live run (x 14-16, y 14-15)
        onpath = set(self.path)
        n = e
        while n not in onpath:
            n = [o for o in self.g.fwd[n] if o is not None and self.d.get(o, 1e9) == self.d[n] - 1][0]
        return self.path.index(n), self.bfs(n, e)

    def test_zero_retreat_cost_walking_the_full_309_step_path(self):
        tr, paid, cost = self.walk(self.path)
        self.assertEqual(len(cost), 309)
        self.assertTrue(all(c == 0.0 for c in cost))
        self.assertEqual(tr.retreat_spent, 0.0)
        self.assertAlmostEqual(sum(paid), SCALE * 309)

    def test_walking_into_the_route_29_west_corridor_and_back_costs_only_the_way_out(self):
        i, walk_in = self.corridor()
        depth = len(walk_in) - 1
        self.assertGreaterEqual(depth, 15)
        walk = self.path[:i + 1] + walk_in[1:] + walk_in[-2::-1]
        tr, paid, cost = self.walk(walk)
        out, back = cost[i:i + depth], cost[i + depth:]
        self.assertTrue(all(c < 0.0 for c in out))
        self.assertAlmostEqual(sum(out), -FRAC * SCALE * depth)       # 0.105 per tile lost
        self.assertTrue(all(c == 0.0 for c in back))                    # re-approach is free
        self.assertTrue(all(c == 0.0 for c in cost[:i]))                # the route up to the junction is free
        self.assertAlmostEqual(sum(paid), SCALE * i)                    # the path reward is unchanged by the cost
        self.assertAlmostEqual(tr.retreat_spent, FRAC * SCALE * depth)

    def test_twenty_tiles_back_along_the_route_and_forward_again(self):
        walk = self.path[:41] + self.path[39:19:-1] + self.path[20:60]
        tr, paid, cost = self.walk(walk)
        self.assertAlmostEqual(sum(cost), -FRAC * SCALE * 20)          # 20 tiles lost: 40 -> 20
        self.assertAlmostEqual(sum(paid), SCALE * 59)                   # new bests still pay exactly once

    def test_oscillating_in_the_corridor_cannot_farm_and_is_capped(self):
        i, walk_in = self.corridor()
        a, b = walk_in[-2], walk_in[-1]                                 # the two innermost tiles
        walk = self.path[:i + 1] + walk_in[1:] + [a, b] * 400
        tr, paid, cost = self.walk(walk)
        self.assertAlmostEqual(tr.retreat_spent, 15.0)                  # the per-episode cap
        self.assertAlmostEqual(sum(cost), -15.0)
        self.assertLess(sum(paid) + sum(cost), SCALE * i)               # strictly worse than not going in
        _, _, free = self.walk(walk, self.tracker(frac=0.0))
        self.assertEqual(sum(free), 0.0)                                # off by default

    def test_a_new_best_after_the_detour_pays_full_and_costs_nothing_extra(self):
        p = self.path
        walk = p[:11] + p[10:6:-1] + p[7:16]
        tr, paid, cost = self.walk(walk)
        self.assertAlmostEqual(sum(cost), -FRAC * SCALE * 3)
        self.assertAlmostEqual(sum(paid), SCALE * 15)

    def test_off_graph_positions_and_whiteout_are_free(self):
        p = self.path
        tr = self.tracker()
        tr.reset(st2(p[50]))
        off = ((1, 1), 3, 3)
        tr.update(st2(p[50]), st2(off))
        self.assertEqual(tr.retreat_step, 0.0)                          # progress None
        tr.update(st2(off), st2(p[40]))
        self.assertEqual(tr.retreat_step, 0.0)                          # previous position unknown
        tr.update(st2(p[40]), st2(p[30], fainted=True))
        self.assertEqual((tr.retreat_step, tr.current), (0.0, None))    # white-out teleport
        tr.update(st2(p[30], fainted=True), st2(p[20]))
        self.assertEqual(tr.retreat_step, 0.0)
        self.assertEqual(tr.retreat_spent, 0.0)


class RetreatExemptionTests(unittest.TestCase):
    """Exemptions on synthetic progress (progress = x): battle, dialogue, Pokemon Center, map change, budget."""

    def setUp(self):
        patcher = mock.patch.object(pp, "progress", lambda s: None if s.x < 0 else float(s.x))
        patcher.start()
        self.addCleanup(patcher.stop)

    def step(self, tr, x0, x1, **kw):
        b = kw.pop("before", {})
        tr.update(st2(((24, 3), x0, 0), **b), st2(((24, 3), x1, 0), **kw))
        return tr.retreat_step

    def tr(self, **kw):
        t = pp.PathProgress(SCALE, retreat_frac=FRAC, retreat_budget=kw.pop("budget", 15.0))
        t.reset(st2(((24, 3), 10, 0)))
        return t

    def test_charges_per_tile_lost_and_nothing_for_gaining(self):
        t = self.tr()
        self.assertAlmostEqual(self.step(t, 10, 9), -FRAC * SCALE)
        self.assertAlmostEqual(self.step(t, 9, 7), -FRAC * SCALE * 2)
        self.assertEqual(self.step(t, 7, 8), 0.0)
        self.assertEqual(self.step(t, 8, 9), 0.0)

    def test_never_in_battle_or_dialogue(self):
        for kw in ({"battle_mode": 1}, {"script_running": 1}, {"script_mode": 1}):
            t = self.tr()
            self.assertEqual(self.step(t, 10, 9, **kw), 0.0, kw)
            t = self.tr()
            self.assertEqual(self.step(t, 10, 9, before=kw), 0.0, kw)

    def test_never_in_a_pokemon_center_or_on_a_map_change(self):
        t = self.tr()
        t.update(st2(((26, 5), 10, 0)), st2(((26, 5), 9, 0)))
        self.assertEqual(t.retreat_step, 0.0)
        t = self.tr()
        t.update(st2(((24, 3), 10, 0)), st2(((26, 3), 9, 0)))        # warp / door
        self.assertEqual(t.retreat_step, 0.0)

    def test_budget_caps_and_reset_clears(self):
        t = self.tr(budget=0.2)
        self.step(t, 10, 5)
        self.assertAlmostEqual(t.retreat_step, -0.2)
        self.assertEqual(self.step(t, 5, 3), 0.0)
        t.reset(st2(((24, 3), 10, 0)))
        self.assertEqual((t.retreat_spent, t.retreat_step), (0.0, 0.0))

    def test_off_by_default(self):
        t = pp.PathProgress(SCALE)
        t.reset(st2(((24, 3), 10, 0)))
        self.assertEqual(self.step(t, 10, 1), 0.0)
        self.assertIsNone(pp.PathProgress.from_scale(None, {"frac": 0.35}))
        self.assertEqual(pp.PathProgress.from_scale(0.3, {"frac": 0.35, "budget": 7.0}).retreat_budget, 7.0)
        with self.assertRaises(ValueError):
            pp.PathProgress(0.3, retreat_frac=-1.0)


# ------------------------------------------------------------------------------------------------ stagnation
class StagnationTests(unittest.TestCase):
    def tr(self, **kw):
        return StagnationTracker(**{"after": 250, "min_penalty": 0.004, "max_penalty": 0.02, "ramp_steps": 300,
                                    "budget": 10.0, **kw})

    def run_steps(self, tr, n, **kw):
        a, b = st2(((24, 3), 3, 3)), st2(((24, 3), 4, 3))
        return [tr.update(a, b, new_best=False, **kw) for _ in range(n)]

    def test_free_until_after_then_ramps_min_to_max(self):
        tr = self.tr()
        pens = self.run_steps(tr, 250 + 400)
        self.assertTrue(all(p == 0.0 for p in pens[:250]))
        self.assertAlmostEqual(pens[250], -0.004)
        self.assertAlmostEqual(pens[250 + 150], -(0.004 + 0.016 * 150 / 300))
        self.assertAlmostEqual(pens[250 + 300], -0.02)
        self.assertAlmostEqual(pens[-1], -0.02)

    def test_budget_caps_the_episode_total(self):
        tr = self.tr()
        total = sum(self.run_steps(tr, 5000))
        self.assertAlmostEqual(total, -10.0)
        self.assertAlmostEqual(tr.spent, 10.0)
        tr.reset()
        self.assertEqual((tr.since_best, tr.run, tr.spent), (0, 0, 0.0))

    def test_a_new_best_resets_the_counter_and_the_ramp(self):
        tr = self.tr()
        self.run_steps(tr, 500)
        self.assertEqual(tr.update(st2(((24, 3), 3, 3)), st2(((24, 3), 4, 3)), new_best=True), 0.0)
        self.assertEqual((tr.since_best, tr.run), (0, 0))
        pens = self.run_steps(tr, 252)
        self.assertEqual(pens[249], 0.0)
        self.assertAlmostEqual(pens[250], -0.004)

    def test_battle_text_and_pokemon_center_steps_are_not_counted(self):
        tr = self.tr(after=3)
        a = st2(((24, 3), 3, 3))
        for kw in ({"battle_mode": 1}, {"script_running": 1}):
            for _ in range(20):
                self.assertEqual(tr.update(a, st2(((24, 3), 4, 3), **kw), new_best=False), 0.0)
        for _ in range(20):
            self.assertEqual(tr.update(st2(((26, 5), 3, 3)), st2(((26, 5), 4, 3)), new_best=False), 0.0)
        self.assertEqual(tr.since_best, 0)

    def test_level_exemption_is_free_and_does_not_bank_ramp_time(self):
        tr = self.tr(after=5)
        self.assertTrue(all(p == 0.0 for p in self.run_steps(tr, 400, level_exempt=True)))
        self.assertEqual(tr.run, 0)
        self.assertAlmostEqual(self.run_steps(tr, 1)[0], -0.004)       # the ramp starts from the minimum

    def test_composition_with_dwell_is_the_max_not_the_sum(self):
        a, b = st2(((24, 3), 3, 3)), st2(((24, 3), 4, 3))
        for dwell, expect_total in ((-0.03, 0.03), (-0.01, 0.02), (0.0, 0.02), (-0.02, 0.02)):
            tr = self.tr(after=1, min_penalty=0.02, max_penalty=0.02)
            tr.update(a, b, new_best=False)                              # since_best 1: still below the threshold
            stag = tr.update(a, b, new_best=False, dwell_charge=dwell)
            self.assertAlmostEqual(-dwell - stag, expect_total, msg=str(dwell))
            self.assertLessEqual(stag, 0.0)
        # only the excess is charged to the stagnation budget
        tr = self.tr(after=1, min_penalty=0.02, max_penalty=0.02)
        tr.update(a, b, new_best=False)
        tr.update(a, b, new_best=False, dwell_charge=-0.015)
        self.assertAlmostEqual(tr.spent, 0.005)

    def test_params_and_factory(self):
        self.assertIsNone(StagnationTracker.from_params(None))
        self.assertIsInstance(StagnationTracker.from_params(load_config("long_v6").stagnation_params()),
                              StagnationTracker)
        for bad in ({"after": 0}, {"min_penalty": -1.0}, {"max_penalty": 0.001}, {"ramp_steps": 0}, {"budget": -1.0}):
            with self.assertRaises(ValueError):
                self.tr(**bad)


# ------------------------------------------------------------------------------------------------ white-out termination
class TerminalTargetTests(unittest.TestCase):
    def test_a_termination_does_not_bootstrap_but_a_truncation_does(self):
        from src.ppo_update import compute_gae
        r = torch.tensor([[1.0], [-15.0], [0.5]])
        v = torch.tensor([[2.0], [3.0], [4.0]])
        done = torch.tensor([[False], [True], [False]])
        last = torch.tensor([9.0])
        ret_term, _ = compute_gae(r, v, done, last, 0.99, 0.95, truncated=torch.zeros_like(done),
                                  bootstrap_values=torch.zeros(3, 1))
        self.assertAlmostEqual(float(ret_term[1]), -15.0)                # value target is the terminal reward alone
        trunc = torch.tensor([[False], [True], [False]])
        ret_trunc, _ = compute_gae(r, v, done, last, 0.99, 0.95, truncated=trunc,
                                   bootstrap_values=torch.tensor([[0.0], [7.0], [0.0]]))
        self.assertAlmostEqual(float(ret_trunc[1]), -15.0 + 0.99 * 7.0, places=4)


@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class WhiteoutTerminationTests(unittest.TestCase):
    CFG = {"path_progress": 0.3, "dwell": {"window": 192, "max_tiles": 16, "min_penalty": 0.005, "max_penalty": 0.03,
                                          "ramp_steps": 384, "budget": 15.0, "grind_grace": 512,
                                          "level_targets": [8, 8, 10, 10, 10, 11, 12, 12, 0]}}

    def make(self, state="new_bark_outside", **cfg):
        w = t.EnvWorker(0, [state], max_steps=500, cfg={**self.CFG, **cfg})
        self.addCleanup(w.close)
        w.reset()
        return w

    @staticmethod
    def faint_after(w):
        """Make the worker's step() see a whole-party faint in its ``after`` state (its 2nd parse call)."""
        real = w.reader.parse
        calls = {"n": 0}

        def parse(frame):
            st = real(frame)
            f = sys._getframe(1)
            if f.f_code.co_name == "step" and f.f_code.co_filename.endswith("train_deep_ppo.py"):
                calls["n"] += 1
                if calls["n"] == 2:
                    return dataclasses_replace(st, party_hp=[[0, max(1, h[1])] for h in st.party_hp])
            return st

        w.reader.parse = parse

    def test_whiteout_ends_the_episode_as_a_termination_with_a_clean_reset(self):
        w = self.make(terminate_on_whiteout=True)
        for _ in range(3):
            w.step(ACTIONS.index("b"))
        self.assertEqual(w.steps, 3)
        self.faint_after(w)
        _s, _r, rew, done, _m, _b, info = w.step(ACTIONS.index("b"))
        self.assertTrue(done)
        self.assertTrue(info["terminated"])
        self.assertFalse(info["truncated"])
        self.assertEqual(info["end_reason"], "whiteout")
        self.assertNotIn("final_obs", info)                           # nothing to bootstrap from
        self.assertAlmostEqual(info["reward_components"]["whiteout"], -15.0)   # the penalty is the terminal reward
        self.assertLess(rew, -14.0)
        # the next episode is a normal fresh one
        self.assertEqual(w.steps, 0)
        self.assertFalse(w._partial)
        self.assertEqual(w.path_progress.gained, 0.0)
        self.assertEqual(w.path_progress.retreat_spent, 0.0)
        self.assertEqual(w.dwell.steps, 0)
        self.assertEqual(w.seen_maps, {(24, 4)})
        self.assertEqual(info["ep_steps"], 0)

    def test_without_the_flag_a_whiteout_does_not_end_the_episode(self):
        w = self.make()
        self.faint_after(w)
        _s, _r, _rew, done, _m, _b, info = w.step(ACTIONS.index("b"))
        self.assertFalse(done)
        self.assertFalse(info["terminated"])
        self.assertIsNone(info["end_reason"])
        self.assertEqual(w.steps, 1)

    def test_time_limit_is_still_a_truncation_with_its_own_end_reason(self):
        w = self.make(terminate_on_whiteout=True)
        w.ep_len = 2
        w.step(ACTIONS.index("b"))
        _s, _r, _rew, done, _m, _b, info = w.step(ACTIONS.index("b"))
        self.assertTrue(done and info["truncated"] and not info["terminated"])
        self.assertEqual(info["end_reason"], "time_limit")

    def test_whiteout_on_the_time_limit_step_is_a_termination(self):
        w = self.make(terminate_on_whiteout=True)
        w.ep_len = 1
        self.faint_after(w)
        _s, _r, _rew, done, _m, _b, info = w.step(ACTIONS.index("b"))
        self.assertTrue(done and info["terminated"] and not info["truncated"])
        self.assertNotIn("final_obs", info)

    def test_a_staggered_partial_first_episode_ended_by_a_whiteout_is_a_real_end(self):
        w = t.EnvWorker(3, ["new_bark_outside"], max_steps=500,
                        cfg={**self.CFG, "terminate_on_whiteout": True, "stagger_episodes": True, "num_envs": 4})
        self.addCleanup(w.close)
        w.reset()
        self.assertTrue(w._partial)
        self.faint_after(w)
        _s, _r, _rew, done, _m, _b, info = w.step(ACTIONS.index("b"))
        self.assertTrue(done)
        self.assertFalse(info["partial_episode"])
        self.assertEqual(w.steps, 0)
        self.assertFalse(w._partial)

    def test_retreat_and_stagnation_components_in_the_real_env(self):
        w = self.make("route29_entry", retreat={"frac": 0.35, "budget": 15.0},
                      stagnation={"after": 1, "min_penalty": 0.02, "max_penalty": 0.02, "ramp_steps": 10, "budget": 10.0})
        w.dwell.last_grind = -10 ** 6      # a level-5 party is otherwise exempt for the first 512 steps (grind grace)
        w.step(ACTIONS.index("b"))
        _s, _r, _rew, _d, _m, _b, info = w.step(ACTIONS.index("b"))
        c = info["reward_components"]
        self.assertIn("retreat", c)
        self.assertIn("stagnation", c)
        # max(dwell, stagnation), never the sum: the two terms together cost exactly the stagnation rate
        self.assertAlmostEqual(-c["stagnation"] - c["dwell"], 0.02)
        w2 = self.make("route29_entry")
        _s, _r, _rew, _d, _m, _b, info2 = w2.step(ACTIONS.index("b"))
        self.assertNotIn("retreat", info2["reward_components"])
        self.assertNotIn("stagnation", info2["reward_components"])


if __name__ == "__main__":
    unittest.main()
