import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import evaluate_policy as ev

ROM = Path(__file__).resolve().parent.parent / "rom" / "pokemon_crystal.gbc"


class FakeState:
    def __init__(self, flags=(), map_id=(24, 4), badges=0):
        self.flags = set(flags)
        self.map_group, self.map_number = map_id
        self.badges = badges

    def has_event(self, name):
        return name in self.flags


class FakeEnv:
    """Scripted story: the map (and finally the badge) changes at fixed steps so leg times are known."""

    def __init__(self, schedule):
        self.schedule = schedule  # {step: (group, number) map to move to, or "badge"}

    def reset(self, seed):
        self.t = 0
        self.map_id, self.badges, self.flags = (24, 4), 0, set()
        return np.zeros((144, 160, 3), np.uint8), np.zeros(4), FakeState()

    def step(self, act):
        self.t += 1
        move = self.schedule.get(self.t)
        if move == "badge":
            self.badges, self.flags = 1, {"EVENT_BEAT_FALKNER"}
        elif move is not None:
            self.map_id = move
        st = FakeState(self.flags, self.map_id, self.badges)
        return np.zeros((144, 160, 3), np.uint8), np.zeros(4), st, 0.5, {"step": -0.01, "milestone": 0.51}, False


class FakePolicy:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def act(self, screen, obs):
        return 0


SCHEDULE = {10: (24, 3), 25: (26, 3), 60: "badge"}


class EpisodeTests(unittest.TestCase):
    def test_leg_times_events_and_success(self):
        env = FakeEnv(SCHEDULE)
        pol = FakePolicy()
        ep = ev.run_episode(env, pol, seed=3, max_steps=100)
        self.assertTrue(ep["success"])
        self.assertEqual(ep["final_stage"], ev.SUCCESS_STAGE)
        self.assertEqual(ep["steps"], 60)  # stops once Falkner is beaten
        self.assertEqual(ep["leg_steps"]["0"], 10)   # left New Bark: Route 29 reached
        self.assertEqual(ep["leg_steps"]["1"], 25)   # Cherrygrove reached
        self.assertEqual(ep["leg_steps"]["7"], 60)
        self.assertEqual(ep["events"]["EVENT_BEAT_FALKNER"], 60)
        self.assertEqual(ep["maps_visited"], 3)
        self.assertAlmostEqual(ep["total_reward"], 30.0)
        self.assertAlmostEqual(ep["components"]["milestone"], 0.51 * 60, places=2)
        self.assertEqual(pol.resets, 1)

    def test_timeout_records_partial_progress(self):
        ep = ev.run_episode(FakeEnv(SCHEDULE), FakePolicy(), seed=0, max_steps=30)
        self.assertFalse(ep["success"])
        self.assertEqual(ep["final_stage"], 2)
        self.assertEqual(ep["final_stage_name"], "cherrygrove")
        self.assertEqual(sorted(ep["leg_steps"]), ["0", "1"])
        self.assertEqual(ep["steps"], 30)

    def test_evaluate_aggregates_and_report(self):
        class Varying(FakeEnv):
            def reset(self, seed):
                self.schedule = {10 + seed: (24, 3), 25 + seed: (26, 3), 60 + seed: "badge"}
                return super().reset(seed)

        seen = []
        res = ev.evaluate(Varying({}), FakePolicy(), episodes=4, max_steps=70, seed=0,
                          meta={"checkpoint": "x.pt", "start_state": "new_bark_outside"},
                          on_episode=lambda i, e: seen.append(i))
        self.assertEqual(seen, [0, 1, 2, 3])
        agg = res["aggregate"]
        self.assertEqual(agg["episodes"], 4)
        self.assertAlmostEqual(agg["legs"]["0"]["mean"], 11.5)  # seeds 0..3 -> 10..13
        self.assertEqual(agg["legs"]["2"]["reached"], 4)
        self.assertLessEqual(agg["legs"]["0"]["ci_low"], agg["legs"]["0"]["mean"])
        self.assertGreaterEqual(agg["legs"]["0"]["ci_high"], agg["legs"]["0"]["mean"])
        self.assertEqual(agg["success_rate"]["mean"], 1.0)
        text = ev.format_report(res)
        self.assertIn("success rate", text)
        self.assertIn("cherrygrove", text)
        json.dumps(res)  # serialisable


class StatsTests(unittest.TestCase):
    def test_bootstrap_ci(self):
        mean, lo, hi = ev.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(mean, 3.0)
        self.assertTrue(1.0 <= lo < 3.0 < hi <= 5.0)
        self.assertEqual(ev.bootstrap_ci([7.0]), (7.0, 7.0, 7.0))
        self.assertTrue(np.isnan(ev.bootstrap_ci([])[0]))

    def test_stage_follows_map_and_badge(self):
        self.assertEqual(ev.stage_of(FakeState()), 0)
        self.assertEqual(ev.stage_of(FakeState(map_id=(26, 3))), 2)
        self.assertEqual(ev.stage_of(FakeState(["EVENT_BEAT_FALKNER"], badges=1)), ev.SUCCESS_STAGE)

    def test_obs_vector_rejects_old_layouts(self):
        with self.assertRaises(ValueError):
            ev.obs_vector(FakeState(), set(), 30)


@unittest.skipUnless(ROM.is_file(), "ROM not available")
class RomTests(unittest.TestCase):
    def test_random_policy_plays_a_few_greedy_steps(self):
        from src.rl_env import DEEP_OBS_DIM

        class Cycle:
            obs_dim = DEEP_OBS_DIM

            def __init__(self):
                self.i = 0

            def reset(self):
                self.i = 0

            def act(self, screen, obs):
                self.i += 1
                assert obs.shape == (DEEP_OBS_DIM,)
                return self.i % 6

        env = ev.GameEnv("new_bark_outside", DEEP_OBS_DIM, max_steps=6, jitter=2)
        try:
            res = ev.evaluate(env, Cycle(), episodes=2, max_steps=6, seed=1, meta={})
        finally:
            env.close()
        self.assertEqual([e["steps"] for e in res["episodes"]], [6, 6])
        self.assertGreaterEqual(res["episodes"][0]["maps_visited"], 1)
        self.assertEqual(res["episodes"][0]["final_stage"], 0)


if __name__ == "__main__":
    unittest.main()
