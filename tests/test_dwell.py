"""long_v3 dwell: the dwell penalty (staying in a small area without progress) with the level-target exemption."""

import argparse
import unittest
from pathlib import Path

from src.config import ROM_PATH
from src.dwell import DWELL_LEVEL_TARGETS, DwellTracker
from src.ram_parser import GameState
from src.rl_env import ACTIONS
from src.story_path import MILESTONE_COUNT
from src.train_config import ConfigError, PRESETS, apply_overrides, load_config
from tools import cloud_train_deep as c
from tools import train_deep_ppo as t

HAVE_ROM = Path(ROM_PATH).exists()
ROUTE29, CHERRYGROVE, POKECENTER = (24, 3), (26, 3), (26, 5)


def gs(map_id=ROUTE29, x=5, y=5, levels=(5,), battle=0, **kw):
    base = {"map_group": map_id[0], "map_number": map_id[1], "x": x, "y": y, "tiles": {}, "event_flags": [0] * 300,
            "party_count": len(levels), "party_species": [155] * len(levels),
            "party_hp": [[19, 19]] * len(levels), "party_levels": list(levels), "battle_mode": battle}
    base.update(kw)
    return GameState.from_dict(base)


def tracker(**kw):
    p = dict(window=16, max_tiles=4, min_penalty=0.005, max_penalty=0.03, ramp_steps=20, budget=15.0, grind_grace=50,
             level_targets=DWELL_LEVEL_TARGETS)
    p.update(kw)
    return DwellTracker(**p)


def shuffle_tiles(tr, n, levels=(12,), map_id=ROUTE29, **flags):
    """n overworld steps bouncing between 3 tiles; returns the list of penalties."""
    out, cur = [], gs(map_id, 5, 5, levels)
    for i in range(n):
        nxt = gs(map_id, 5 + (i % 3), 5, levels)
        out.append(tr.update(cur, nxt, **flags)[0])
        cur = nxt
    return out


class DwellTrackerTests(unittest.TestCase):
    def test_triggers_in_a_small_region_and_ramps(self):
        tr = tracker()
        tr.reset(gs(levels=(12,)))
        pens = shuffle_tiles(tr, 16 + 30)
        self.assertTrue(all(p == 0.0 for p in pens[:15]))  # window not yet full
        self.assertAlmostEqual(pens[15], -0.005)          # first stuck step: minimum
        self.assertLess(pens[20], pens[16])                # ramping up (more negative)
        self.assertAlmostEqual(pens[-1], -0.03)            # ramp complete: maximum
        self.assertTrue(all(p <= 0 for p in pens))

    def test_resets_on_leaving_the_area(self):
        tr = tracker()
        tr.reset(gs(levels=(12,)))
        shuffle_tiles(tr, 30)
        self.assertGreater(tr.run, 0)
        cur = gs(levels=(12,), x=5, y=5)
        pens = []
        for i in range(8):  # walk away along a line: more than max_tiles distinct tiles in the window
            nxt = gs(levels=(12,), x=10 + i, y=5)
            pens.append(tr.update(cur, nxt, new_tile=True)[0])
            cur = nxt
        self.assertEqual(pens[-1], 0.0)
        self.assertEqual(tr.run, 0)

    def test_new_map_clears_the_window(self):
        tr = tracker()
        tr.reset(gs(levels=(12,)))
        shuffle_tiles(tr, 30)
        pen, _ = tr.update(gs(levels=(12,)), gs(CHERRYGROVE, 3, 3, (12,)), new_map=True)
        self.assertEqual(pen, 0.0)
        self.assertEqual(tr.run, 0)
        self.assertEqual(shuffle_tiles(tr, 10, map_id=CHERRYGROVE), [0.0] * 10)  # fresh window, not yet full

    def test_progress_events_postpone_the_penalty(self):
        tr = tracker()
        tr.reset(gs(levels=(12,)))
        shuffle_tiles(tr, 30)
        self.assertGreater(tr.run, 0)
        a, b = gs(levels=(12,)), gs(levels=(12,), x=6)
        self.assertEqual(tr.update(a, b, progress=True)[0], 0.0)  # e.g. a new species was caught
        self.assertEqual(shuffle_tiles(tr, 10), [0.0] * 10)       # a full window of nothing new is needed again
        pens = shuffle_tiles(tr, 20)
        self.assertLess(pens[-1], 0.0)

    def test_exempt_below_the_stage_target_and_active_above_it(self):
        target = DWELL_LEVEL_TARGETS[1]  # Route 29
        self.assertEqual(target, 8)
        below = tracker(grind_grace=10 ** 6)
        below.reset(gs(levels=(target - 1,)))
        self.assertEqual(shuffle_tiles(below, 60, levels=(target - 1,)), [0.0] * 60)  # grinding is legitimate
        self.assertEqual(below.run, 0)
        above = tracker()
        above.reset(gs(levels=(target,)))
        self.assertLess(shuffle_tiles(above, 60, levels=(target,))[-1], 0.0)

    def test_exemption_needs_actual_fighting_and_target_follows_the_furthest_stage(self):
        tr = tracker(grind_grace=10)
        tr.reset(gs(levels=(5,)))
        pens = shuffle_tiles(tr, 40, levels=(5,))
        self.assertEqual(pens[5], 0.0)       # fresh episode: inside the grace period
        self.assertLess(pens[-1], 0.0)       # no fight for > grace steps: standing on a bush does not count
        walking, fight = gs(levels=(5,)), gs(levels=(5,), battle=1)
        tr.update(walking, fight)
        tr.update(fight, walking)            # a fight just ended -> grinding again
        self.assertEqual(shuffle_tiles(tr, 5, levels=(5,))[-1], 0.0)
        # the target grows with the story: level 9 is enough on Route 29 (8) but not once Route 31 was reached (10)
        tr2 = tracker(grind_grace=10 ** 6)
        tr2.reset(gs(levels=(9,)))
        self.assertFalse(tr2.level_exempt(gs(levels=(9,))))
        tr2.update(gs(levels=(9,)), gs((26, 2), 4, 4, (9,)))  # Route 31
        self.assertEqual(tr2.target_level(), 10)
        self.assertTrue(tr2.level_exempt(gs((26, 2), 4, 4, (9,))))
        tr2.update(gs((26, 2), 4, 4, (9,)), gs((26, 2), 4, 4, (10,)))
        self.assertFalse(tr2.level_exempt(gs((26, 2), 4, 4, (10,))))

    def test_level_exemption_also_reports_loop_exempt(self):
        tr = tracker()
        tr.reset(gs(levels=(5,)))
        _, loop_exempt = tr.update(gs(levels=(5,)), gs(levels=(5,), x=6))
        self.assertTrue(loop_exempt)
        tr.reset(gs(levels=(12,)))
        self.assertFalse(tr.update(gs(levels=(12,)), gs(levels=(12,), x=6))[1])

    def test_battle_and_text_steps_do_not_count_and_do_not_reset(self):
        tr = tracker()
        tr.reset(gs(levels=(12,)))
        shuffle_tiles(tr, 30)
        run, overworld = tr.run, tr.overworld
        walking, fight = gs(levels=(12,)), gs(levels=(12,), battle=1)
        self.assertEqual(tr.update(walking, fight)[0], 0.0)   # battle starts
        for _ in range(10):
            self.assertEqual(tr.update(fight, fight)[0], 0.0)  # battle turns
        self.assertEqual(tr.update(fight, walking)[0], 0.0)   # battle ends
        talking = gs(levels=(12,), script_mode=1)
        self.assertEqual(tr.update(talking, talking)[0], 0.0)
        self.assertEqual((tr.run, tr.overworld), (run, overworld))  # nothing counted, nothing reset
        self.assertLess(shuffle_tiles(tr, 1)[0], 0.0)               # walk-battle-walk keeps the counter

    def test_pokecenter_is_exempt(self):
        tr = tracker()
        tr.reset(gs(POKECENTER, 3, 3, (12,)))
        self.assertEqual(shuffle_tiles(tr, 60, map_id=POKECENTER), [0.0] * 60)

    def test_per_episode_budget_cap_and_reset(self):
        tr = tracker(budget=0.1)
        tr.reset(gs(levels=(12,)))
        pens = shuffle_tiles(tr, 200)
        self.assertAlmostEqual(sum(pens), -0.1, places=9)
        self.assertEqual(pens[-1], 0.0)
        tr.reset(gs(levels=(12,)))
        self.assertEqual(tr.spent, 0.0)
        self.assertAlmostEqual(sum(shuffle_tiles(tr, 200)), -0.1, places=9)

    def test_default_budget_bounds_the_whole_episode(self):
        tr = DwellTracker()
        tr.reset(gs(levels=(12,)))
        self.assertGreaterEqual(sum(shuffle_tiles(tr, 4000)), -15.0 - 1e-9)

    def test_falkner_beaten_has_no_level_target(self):
        self.assertEqual(len(DWELL_LEVEL_TARGETS), MILESTONE_COUNT + 1)
        self.assertEqual(DWELL_LEVEL_TARGETS[-1], 0)
        self.assertEqual(list(DWELL_LEVEL_TARGETS), sorted(DWELL_LEVEL_TARGETS[:-1]) + [0])  # non-decreasing until the end


class ConfigTests(unittest.TestCase):
    def test_dwell_is_on_in_long_v3_only(self):
        self.assertNotIn("long_v4", PRESETS)
        v2, v3 = load_config("long_v2"), load_config("long_v3")
        self.assertTrue(v3.dwell_penalty)
        self.assertFalse(v2.dwell_penalty)
        self.assertIsNone(v2.dwell_params())
        self.assertEqual(v3.dwell_params()["level_targets"], list(DWELL_LEVEL_TARGETS))
        self.assertEqual(list(DWELL_LEVEL_TARGETS), [8, 8, 10, 10, 10, 11, 12, 12, 0])

    def test_worker_env_cfg_carries_the_dwell_params(self):
        env = t.worker_env_cfg(load_config("long_v3"), "logs/swarm/r", "logs/swarm/r/battle")
        self.assertEqual(env["dwell"]["window"], 192)
        self.assertTrue(env["dwell_exempt_loop"])
        self.assertIsNone(t.worker_env_cfg(load_config("long_v2"), None)["dwell"])

    def test_can_be_ablated(self):
        cfg = apply_overrides(load_config("long_v3"), {"dwell_penalty": False})
        self.assertIsNone(t.worker_env_cfg(cfg, None)["dwell"])

    def test_validation(self):
        for bad in ({"dwell_window": 4}, {"dwell_max_tiles": 0}, {"dwell_max_tiles": 192},
                    {"dwell_min_penalty": 0.05}, {"dwell_ramp_steps": 0}, {"dwell_budget": -1.0},
                    {"dwell_grind_grace": -1}, {"dwell_level_targets": [8, 10]},
                    {"dwell_level_targets": [8, 8, 10, 10, 10, 11, 12, 12, -1]}):
            with self.assertRaises(ConfigError, msg=str(bad)):
                apply_overrides(load_config("long_v3"), bad)
        ok = apply_overrides(load_config("long_v3"), {"dwell_level_targets": [6, 6, 8, 8, 8, 9, 10, 10, 0]})
        self.assertEqual(ok.dwell_level_targets[0], 6)

    def test_cloud_train_accepts_long_v3_with_dwell(self):
        self.assertEqual(c.resolve_config(argparse.Namespace(config="long_v3")), ("long_v3", False))
        args = argparse.Namespace(updates=1600, rollout=128, max_steps=1024, batch_size=256, lr=3e-4,
                                  state="new_bark_outside", save="s.pt", envs=24, curriculum=False, recurrent=True,
                                  states="", config="long_v3", explicit={"updates", "envs", "state", "curriculum"},
                                  resume=False, run_id="v3-1")
        cmd = c.build_remote_train_cmd(args, 24, "long_v3")
        self.assertIn("--config long_v3", cmd)


@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class EnvWorkerDwellTests(unittest.TestCase):
    STATE = "new_bark_outside"
    B = ACTIONS.index("b")

    def make(self, **cfg):
        w = t.EnvWorker(0, [self.STATE], max_steps=200, cfg=cfg)
        self.addCleanup(w.close)
        w.reset()
        return w

    def test_real_env_pays_the_dwell_component_when_standing_still(self):
        dwell = {"window": 8, "max_tiles": 2, "min_penalty": 0.01, "max_penalty": 0.02, "ramp_steps": 4,
                 "budget": 15.0, "grind_grace": 0, "level_targets": [0] * (MILESTONE_COUNT + 1)}
        w = self.make(dwell=dwell, dwell_exempt_loop=True)
        comps = []
        for _ in range(14):
            *_, info = w.step(self.B)
            comps.append(info["reward_components"]["dwell"])
        self.assertTrue(all(v == 0.0 for v in comps[:7]))
        self.assertTrue(all(v < 0.0 for v in comps[9:]), comps)
        w.reset()  # a new episode starts from zero
        *_, info = w.step(self.B)
        self.assertEqual(info["reward_components"]["dwell"], 0.0)

    def test_without_the_flag_there_is_no_dwell_component(self):
        w = self.make()
        *_, info = w.step(self.B)
        self.assertNotIn("dwell", info["reward_components"])


if __name__ == "__main__":
    unittest.main()
