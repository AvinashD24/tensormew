"""The route graph is now only the OPTIONAL goal potential (off by default); it is no longer an observation."""

import threading
import time
import unittest
from types import SimpleNamespace

from src import route_graph as rg

START = ((24, 4), 3, 9)  # states/new_bark_outside.state


def _state(mid, x, y):
    return SimpleNamespace(map_group=mid[0], map_number=mid[1], x=x, y=y)


class OracleRemovedTests(unittest.TestCase):
    def test_no_direction_or_distance_features_exist(self):
        for name in ("route_features", "ROUTE_FEATURE_DIM", "current_stage", "next_direction", "DIST_SCALE"):
            self.assertFalse(hasattr(rg, name), name)
        self.assertFalse(hasattr(rg._Graph, "next_direction"))
        self.assertFalse(hasattr(rg._Graph, "path_to_goal"))

    def test_single_goal_in_front_of_falkner(self):
        self.assertEqual(rg.STAGE_COUNT, 1)
        self.assertEqual(rg.GOALS, (((10, 7), 5, 2),))


class RouteGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = rg._graph()
        if cls.g is None:
            raise unittest.SkipTest("ROM not available")

    def test_covers_route_maps_and_not_others(self):
        for mid in rg.ROUTE_MAPS:
            self.assertTrue(rg.covers_map(mid), mid)
        for mid in ((1, 1), (3, 1), (9, 9), (24, 99), (0, 0)):
            self.assertFalse(rg.covers_map(mid), mid)

    def test_minimal_story_path_is_about_320_steps_without_any_errand(self):
        # New Bark start tile -> in front of Falkner, no egg / Elm / rival detour: the audit's ~319 steps.
        d = rg.goal_remaining(_state(*START))
        self.assertIsNotNone(d)
        self.assertEqual(d, 319.0)

    def test_phi_rises_one_per_step_toward_the_goal(self):
        g = self.g
        node = START
        phi = rg.route_potential(_state(*node))
        self.assertEqual(phi, -319.0)
        for _ in range(40):
            dist = g.dist[0]
            nxt = min((n for n in g.fwd[node] if n is not None and n in dist), key=lambda n: dist[n])
            node = nxt
            new = rg.route_potential(_state(*node))
            self.assertEqual(new - phi, 1.0)
            phi = new

    def test_off_graph_state_is_neutral(self):
        s = _state((1, 1), 4, 4)
        self.assertEqual(rg.route_potential(s), 0.0)
        self.assertIsNone(rg.goal_remaining(s))

    def test_out_of_grid_position_on_a_covered_map_is_safe(self):
        s = _state((24, 4), 500, 500)
        self.assertEqual(rg.route_potential(s), -(self.g.max_dist[0] + rg.UNREACHABLE_EXTRA))
        self.assertIsNone(rg.goal_remaining(s))

    def test_graph_builds_once_and_is_thread_safe(self):
        out = []
        threads = [threading.Thread(target=lambda: out.append(rg._graph())) for _ in range(4)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len({id(g) for g in out}), 1)
        self.assertLess(time.time() - t0, 30)


if __name__ == "__main__":
    unittest.main()
