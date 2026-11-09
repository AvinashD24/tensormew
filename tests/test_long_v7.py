"""long_v7: source-split (learned vs head start) metrics, healing-care reward terms, preset isolation, offline sampled
evaluation helpers.  Older presets (long_v3..v6) must stay exactly as they were."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from src import heal_care as hc
from src.heal_care import HealCareTracker, is_heal_trip
from src.milestone_stats import SourceSplitStats, source_of
from src.ram_parser import GameState
from src.train_config import PRESETS, ConfigError, apply_overrides, load_config
from tools import evaluate_policy as ev
from tools import train_deep_ppo as t

from src.config import ROM_PATH

ROOT = Path(__file__).resolve().parents[1]
HAVE_ROM = Path(ROM_PATH).exists()
POTION = 0x12

V7_FLAGS = ("source_split_metrics", "lowhp_exposure_penalty", "battle_survive_bonus", "potion_use_reward", "heal_trip_exempt")


def gs(hp=(19, 19), map_id=(24, 3), battle=0, script=0, potions=0, extra_hp=(), enemy_hp=1, enemy_max=10):
    party = [list(hp)] + [list(h) for h in extra_hp]
    base = {"map_group": map_id[0], "map_number": map_id[1], "x": 3, "y": 9, "tiles": {}, "event_flags": [0] * 300,
            "party_count": len(party), "party_species": [155] * len(party), "party_hp": party,
            "party_levels": [5] * len(party), "party_moves": [[33, 43, 0, 0]] * len(party),
            "items": [POTION] if potions else [], "item_counts": [potions] if potions else [],
            "battle_mode": battle, "script_mode": script, "enemy_hp": enemy_hp, "enemy_max_hp": enemy_max}
    return GameState.from_dict(base)


class SourceSplitTests(unittest.TestCase):
    def test_source_of(self):
        self.assertEqual(source_of("new_bark_outside"), "base")
        self.assertEqual(source_of("states/new_bark_outside.state"), "base")
        self.assertEqual(source_of("swarm:stage1_0"), "swarm")
        self.assertEqual(source_of("battle:wild_x"), "battle")
        self.assertEqual(source_of("route:b3_0"), "route")
        self.assertEqual(source_of("some_other_state"), "other")
        self.assertEqual(source_of(None), "other")

    def test_gain_and_reach_count_only_what_was_not_passed_at_the_start(self):
        s = SourceSplitStats()
        # base start: gained 40 tiles, reached Route 29 (stage 1) only
        s.add_episode("base", 0, 1, best_path=40.0, start_path=0.0)
        # route checkpoint that already passed Route 29 and Cherrygrove (start stage 2): gained 10, reached Route 30 (stage 3)
        s.add_episode("route", 2, 3, best_path=130.0, start_path=120.0)
        # a checkpoint that gained nothing (best == start) and a negative gap is clipped to 0
        s.add_episode("route", 2, 2, best_path=120.0, start_path=121.0)
        out = s.end_update()
        base, route = out["src_split"]["base"], out["src_split"]["route"]
        self.assertEqual(base["n"], 1)
        self.assertAlmostEqual(base["gain"], 40.0)
        self.assertEqual(base["reach"][:3], [1.0, 0.0, 0.0])
        self.assertEqual(route["n"], 2)
        self.assertAlmostEqual(route["gain"], 5.0)   # (10 + 0) / 2
        # milestones 0 and 1 were already passed: nobody is eligible for them
        self.assertEqual(route["elig"][:3], [0, 0, 2])
        self.assertEqual(route["reach"][:3], [None, None, 0.5])
        self.assertEqual(out["src_split_cum"]["route"]["n"], 2)

    def test_window_resets_but_cumulative_keeps_growing(self):
        s = SourceSplitStats()
        s.add_episode("base", 0, 0, 5.0, 0.0)
        s.end_update()
        s.add_episode("base", 0, 1, 15.0, 0.0)
        out = s.end_update()
        self.assertEqual(out["src_split"]["base"]["n"], 1)
        self.assertEqual(out["src_split_cum"]["base"]["n"], 2)
        self.assertAlmostEqual(out["src_split_cum"]["base"]["gain"], 10.0)
        self.assertEqual(s.end_update()["src_split"], {})

    def test_partial_and_missing_data_are_skipped_and_missing_progress_only_drops_the_gain(self):
        s = SourceSplitStats()
        self.assertFalse(s.add_episode("base", 0, 1, 5.0, 0.0, partial=True))
        self.assertFalse(s.add_episode("base", None, 1))
        self.assertTrue(s.add_episode("base", 0, 1, best_path=None, start_path=None))
        out = s.end_update()["src_split"]["base"]
        self.assertEqual(out["n"], 1)
        self.assertIsNone(out["gain"])
        self.assertEqual(out["reach"][0], 1.0)

    def test_payload_is_json_serialisable(self):
        import json
        s = SourceSplitStats()
        s.add_episode("battle", 1, 2, 60.0, 50.0)
        json.dumps(s.end_update())


class HealCareTests(unittest.TestCase):
    def tracker(self, **kw):
        return HealCareTracker(**kw)

    def test_exposure_ramps_is_capped_and_stops_at_a_center_a_heal_or_a_battle(self):
        h = self.tracker(survive=False, potion=False, ramp=4, e_min=0.002, e_max=0.01, e_budget=0.02)
        low = gs(hp=(5, 30))
        c1 = h.update(low, low)["lowhp"]
        c2 = h.update(low, low)["lowhp"]
        self.assertLess(c1, 0)
        self.assertLess(c2, c1)                             # ramps: larger cost
        self.assertAlmostEqual(c1, -(0.002 + 0.008 * 0.25))
        out = h.update(low, low), h.update(low, low), h.update(low, low)
        total = c1 + c2 + sum(o.get("lowhp", 0.0) for o in out)
        self.assertAlmostEqual(total, -0.02)                # budget cap
        h.reset()
        self.assertAlmostEqual(h.update(low, low)["lowhp"], c1)   # a fresh episode starts a fresh run
        # not charged: Pokemon Center, Mart, battle, text, fainted, healthy
        h.reset()
        self.assertEqual(h.update(gs(hp=(5, 30), map_id=(26, 5)), gs(hp=(5, 30), map_id=(26, 5))), {})
        self.assertEqual(h.update(gs(hp=(5, 30), map_id=(26, 4)), gs(hp=(5, 30), map_id=(26, 4))), {})
        self.assertEqual(h.update(low, gs(hp=(5, 30), battle=1)), {})
        self.assertEqual(h.update(low, gs(hp=(5, 30), script=1)), {})
        self.assertEqual(h.update(gs(hp=(1, 30)), gs(hp=(0, 30))), {})   # whole party fainted = white-out
        self.assertEqual(h.update(gs(hp=(25, 30)), gs(hp=(25, 30))), {})
        # a heal step is not charged and restarts the ramp
        h.reset()
        h.update(low, low)
        self.assertEqual(h.update(low, gs(hp=(9, 30))), {})
        self.assertAlmostEqual(h.run, 0)

    def test_exposure_is_not_charged_inside_battle_but_the_run_survives_it(self):
        h = self.tracker(survive=False, potion=False)
        low = gs(hp=(5, 30))
        h.update(low, low)
        h.update(low, gs(hp=(5, 30), battle=1))
        self.assertEqual(h.run, 1)

    def test_survive_bonus_only_for_a_won_battle_with_a_healthy_lead(self):
        h = self.tracker(exposure=False, potion=False, s_bonus=0.5, s_budget=1.0)
        fight = gs(hp=(20, 30), battle=1)
        won = gs(hp=(20, 30), enemy_hp=0)
        self.assertEqual(h.update(fight, won), {"survive": 0.5})
        # fled: the wild enemy still has HP -> nothing
        self.assertEqual(h.update(fight, gs(hp=(20, 30), enemy_hp=5)), {})
        # won but the lead is weak (< 50 %)
        self.assertEqual(h.update(gs(hp=(10, 30), battle=1), gs(hp=(10, 30), enemy_hp=0)), {})
        # trainer battle that ended with the lead alive
        self.assertEqual(h.update(gs(hp=(20, 30), battle=2), gs(hp=(20, 30))), {"survive": 0.5})
        # budget
        self.assertEqual(h.update(fight, won), {})
        # white-out ends a battle too
        h.reset()
        self.assertEqual(h.update(gs(hp=(1, 30), battle=1), gs(hp=(0, 30), enemy_hp=0)), {})

    def test_potion_bonus_needs_item_used_hp_restored_and_a_low_lead(self):
        h = self.tracker(exposure=False, survive=False, p_bonus=1.0, p_budget=1.5)
        before = gs(hp=(5, 30), potions=3)
        used = gs(hp=(25, 30), potions=2)
        self.assertEqual(h.update(before, used), {"potion": 1.0})
        self.assertEqual(h.update(before, used), {"potion": 0.5})   # budget cap
        self.assertEqual(h.update(before, used), {})
        h.reset()
        # sold / tossed: the count drops but HP does not rise
        self.assertEqual(h.update(before, gs(hp=(5, 30), potions=2)), {})
        # a Center heal: HP rises but no item is used
        self.assertEqual(h.update(before, gs(hp=(30, 30), potions=3)), {})
        # lead healthy before: no bonus for topping up
        self.assertEqual(h.update(gs(hp=(25, 30), potions=3), gs(hp=(30, 30), potions=2)), {})

    def test_heal_trip_helper(self):
        self.assertTrue(is_heal_trip(gs(hp=(5, 30))))
        self.assertFalse(is_heal_trip(gs(hp=(25, 30))))
        self.assertTrue(is_heal_trip(gs(hp=(30, 30), map_id=(26, 5))))      # inside a Center
        self.assertTrue(is_heal_trip(gs(hp=(30, 30), map_id=(26, 4))))      # inside a Mart
        self.assertFalse(is_heal_trip(gs(hp=(0, 30))))                      # white-out teleport is not a trip
        self.assertFalse(is_heal_trip(gs(hp=(5, 30)), threshold=0.1))

    def test_params_and_validation(self):
        self.assertIsNone(HealCareTracker.from_params(None))
        self.assertIsNotNone(HealCareTracker.from_params({"survive": False}))
        for bad in ({"threshold": 0.0}, {"e_min": -1.0}, {"e_max": 0.0001}, {"ramp": 0}, {"e_budget": -1.0},
                    {"s_threshold": 1.5}, {"p_budget": -1.0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                HealCareTracker(**bad)


class ConfigTests(unittest.TestCase):
    def test_long_v7_values(self):
        self.assertIn("long_v7", PRESETS)
        cfg = load_config("long_v7")
        self.assertEqual(cfg.name, "long_v7")
        for f in V7_FLAGS:
            self.assertTrue(getattr(cfg, f), f)
        self.assertEqual((cfg.heal_reward_scale, cfg.heal_budget), (6.0, 10.0))
        self.assertEqual((cfg.battle_lowhp_weight, cfg.battle_lowhp_threshold), (6.0, 0.5))
        self.assertEqual((cfg.lowhp_exposure_threshold, cfg.lowhp_exposure_min, cfg.lowhp_exposure_max,
                          cfg.lowhp_exposure_ramp, cfg.lowhp_exposure_budget), (0.3, 0.002, 0.01, 128, 6.0))
        self.assertEqual((cfg.battle_survive_amount, cfg.battle_survive_threshold, cfg.battle_survive_budget), (0.5, 0.5, 3.0))
        self.assertEqual((cfg.potion_use_amount, cfg.potion_use_threshold, cfg.potion_use_budget), (1.0, 0.5, 3.0))

    def test_long_v7_only_differs_from_long_v6_in_the_new_settings(self):
        a, b = load_config("long_v6").to_dict(), load_config("long_v7").to_dict()
        diff = {k for k in a if a[k] != b[k]}
        self.assertEqual(diff, {"name", "source_split_metrics", "heal_reward_scale", "heal_budget", "lowhp_exposure_penalty",
                                "battle_survive_bonus", "potion_use_reward", "heal_trip_exempt", "battle_lowhp_weight",
                                "battle_lowhp_threshold"})

    def test_older_presets_have_every_v7_flag_off_and_unchanged_healing(self):
        for name in ("blueprint_v1", "legacy", "long_v1", "long_v2", "long_v3", "long_v4", "long_v5", "long_v6"):
            cfg = load_config(name)
            for f in V7_FLAGS:
                self.assertFalse(getattr(cfg, f), f"{name}.{f}")
            self.assertIsNone(cfg.heal_care_params(), name)
            env = t.worker_env_cfg(cfg, "swarm", "battle", "route")
            self.assertIsNone(env["heal_care"], name)
            self.assertFalse(env["heal_trip_exempt"], name)
        v6 = load_config("long_v6")
        self.assertEqual((v6.heal_reward_scale, v6.heal_budget, v6.battle_lowhp_weight, v6.battle_lowhp_threshold),
                         (4.0, 8.0, 3.0, 0.4))

    def test_each_feature_can_be_ablated(self):
        cfg = apply_overrides(load_config("long_v7"), {"lowhp_exposure_penalty": False, "battle_survive_bonus": False})
        self.assertEqual(cfg.heal_care_params()["potion"], True)
        self.assertFalse(cfg.heal_care_params()["exposure"])
        cfg = apply_overrides(cfg, {"potion_use_reward": False})
        self.assertIsNone(cfg.heal_care_params())

    def test_validation(self):
        for bad in ({"lowhp_exposure_threshold": 0.0}, {"lowhp_exposure_ramp": 0}, {"lowhp_exposure_budget": -1.0},
                    {"battle_survive_threshold": 2.0}, {"potion_use_budget": -1.0}, {"lowhp_exposure_max": 0.0001}):
            with self.assertRaises(ConfigError, msg=str(bad)):
                apply_overrides(load_config("long_v7"), bad)

    def test_worker_env_cfg_for_long_v7(self):
        env = t.worker_env_cfg(load_config("long_v7"), "s", "b", "r")
        self.assertTrue(env["heal_trip_exempt"])
        self.assertEqual(env["heal_trip_threshold"], 0.3)
        self.assertEqual(env["heal_scale"], 6.0)
        self.assertEqual(env["heal_care"]["ramp"], 128)

    def test_trainer_cli_accepts_the_preset(self):
        args = t.build_parser().parse_args(["--config", "long_v7", "--envs", "2", "--updates", "3", "--rollout", "64"])
        self.assertEqual(t.build_config(args).name, "long_v7")

    def test_config_chain_imports_without_numpy_or_torch(self):
        code = ("import sys; sys.modules['numpy'] = None; sys.modules['torch'] = None; "
                "from src.train_config import PRESETS; from src.train_config import load_config; "
                "assert load_config('long_v7').heal_trip_exempt; print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)


class WorkerTripExemptionTests(unittest.TestCase):
    """The retreat / stagnation / dwell costs are zeroed while the lead is low (checked against the real EnvWorker code path
    without an emulator: the helper decides, the worker applies it)."""

    def test_helper_covers_the_exempt_cases_the_worker_relies_on(self):
        self.assertTrue(is_heal_trip(gs(hp=(2, 30)), 0.3))
        self.assertFalse(is_heal_trip(gs(hp=(20, 30)), 0.3))
        self.assertIn((26, 5), hc.HEAL_EXEMPT_MAPS)
        self.assertIn((10, 9), hc.HEAL_EXEMPT_MAPS)


@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class EnvWorkerV7Tests(unittest.TestCase):
    def make(self, **cfg):
        base = {"path_progress": 0.3, "retreat": {"frac": 0.35, "budget": 15.0},
                "stagnation": {"after": 2, "min_penalty": 0.1, "max_penalty": 0.1, "ramp_steps": 1, "budget": 10.0},
                "dwell": None}
        base.update(cfg)
        w = t.EnvWorker(0, ["new_bark_outside"], max_steps=6, cfg=base)
        self.addCleanup(w.close)
        w.reset()
        return w

    def test_finished_episode_reports_its_start_progress_and_state(self):
        w = self.make(heal_care={"survive": False, "potion": False})
        self.assertIsNotNone(w._ep_start_path)
        info = {}
        for _ in range(6):
            *_, info = w.step(0)
        self.assertEqual(info["ended_state"], "new_bark_outside")
        self.assertIsNotNone(info["ended_best_path"])
        self.assertGreaterEqual(info["ended_best_path"], info["ended_start_path"])
        self.assertEqual(source_of(info["ended_state"]), "base")

    def test_trip_exemption_zeroes_the_three_costs_and_off_leaves_them(self):
        from unittest import mock
        for exempt in (True, False):
            w = self.make(heal_trip_exempt=exempt)
            with mock.patch.object(t, "is_heal_trip", return_value=True):
                comps = {}
                for i in range(5):
                    *_, info = w.step(5)   # B presses: no new best path progress, so stagnation accrues
                    for k in ("retreat", "stagnation", "dwell"):
                        comps[k] = comps.get(k, 0.0) + info["reward_components"].get(k, 0.0)
            if exempt:
                self.assertEqual((comps.get("retreat", 0.0), comps.get("stagnation", 0.0), comps.get("dwell", 0.0)), (0.0, 0.0, 0.0))
            else:
                self.assertLess(comps.get("stagnation", 0.0), 0.0)   # stagnation (0.1/step after 2 steps) is charged


class SampledEvalSummaryTests(unittest.TestCase):
    def test_summary_rates_medians_and_steps(self):
        eps = [
            {"best_path": 10.0, "gain": 10.0, "ms_steps": {"0": 40}, "whiteout": True},
            {"best_path": 50.0, "gain": 50.0, "ms_steps": {"0": 20, "1": 300}, "whiteout": False},
            {"best_path": 0.0, "gain": 0.0, "ms_steps": {}, "whiteout": True},
            {"best_path": 30.0, "gain": 30.0, "ms_steps": {"0": 60}, "whiteout": False},
        ]
        s = ev.summarize_sampled(eps)
        self.assertEqual(s["episodes"], 4)
        self.assertAlmostEqual(s["best_path_mean"], 22.5)
        self.assertAlmostEqual(s["best_path_median"], 20.0)
        self.assertEqual(s["whiteouts"], 2)
        r29 = s["milestones"]["Route 29"]
        self.assertEqual((r29["reached"], r29["rate"]), (3, 0.75))
        self.assertAlmostEqual(r29["steps_mean"], 40.0)
        self.assertAlmostEqual(r29["steps_median"], 40.0)
        self.assertEqual(s["milestones"]["Cherrygrove"]["reached"], 1)
        self.assertIsNone(s["milestones"]["Route 30"]["steps_mean"])

    def test_empty(self):
        s = ev.summarize_sampled([])
        self.assertEqual(s["episodes"], 0)
        self.assertIsNone(s["best_path_mean"])

    def test_cli_requires_a_policy_source(self):
        with self.assertRaises(SystemExit):
            ev.main(["--sampled"])
        with self.assertRaises(SystemExit):
            ev.main([])


if __name__ == "__main__":
    unittest.main()
