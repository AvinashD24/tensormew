import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from src import route_graph as rg

EGG = "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"
GAVE = "EVENT_GAVE_MYSTERY_EGG_TO_ELM"
FALKNER = "EVENT_BEAT_FALKNER"
TOGEPI = "EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE"
STAGE_FLAGS = (
    frozenset(),
    frozenset({EGG}),
    frozenset({EGG, GAVE}),
    frozenset({EGG, GAVE, FALKNER}),
)


def _state(mid, x, y, stage=0, flags=None):
    flags = STAGE_FLAGS[stage] if flags is None else frozenset(flags)
    return SimpleNamespace(
        map_group=mid[0], map_number=mid[1], x=x, y=y, party_hp=((20, 20),), has_event=lambda n: n in flags
    )


def _at(node, stage):
    return _state(node[0], node[1], node[2], stage)


class StageTests(unittest.TestCase):
    def test_stage_follows_event_flags(self):
        self.assertEqual(rg.current_stage(_state((24, 4), 1, 1, flags=())), 0)
        self.assertEqual(rg.current_stage(_state((24, 4), 1, 1, flags=(EGG,))), 1)
        self.assertEqual(rg.current_stage(_state((24, 4), 1, 1, flags=(EGG, GAVE))), 2)
        self.assertEqual(rg.current_stage(_state((24, 4), 1, 1, flags=(EGG, GAVE, FALKNER))), 3)
        self.assertEqual(rg.current_stage(_state((24, 4), 1, 1, flags=(EGG, GAVE, FALKNER, TOGEPI))), 3)

    def test_state_without_has_event_is_stage_zero(self):
        self.assertEqual(rg.current_stage(SimpleNamespace(map_group=24, map_number=4, x=0, y=0)), 0)

    def test_constants(self):
        self.assertEqual(rg.STAGE_COUNT, 4)
        self.assertEqual(rg.ROUTE_FEATURE_DIM, 9)


class RouteGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = rg._graph()
        if cls.g is None:
            raise unittest.SkipTest("ROM not available")

    # --- helpers -------------------------------------------------------------------------------
    def full_walk(self, start=((24, 4), 6, 9)):
        """(stage, node) after every step of the scripted route start -> egg -> Elm -> Falkner -> aide.

        The stage flips on the step that stands on the goal tile, as the event flag does in the game.
        """
        g = self.g
        walk = [(0, start)]
        for stage in range(rg.STAGE_COUNT):
            path = g.path_to_goal(stage, walk[-1][1])
            self.assertTrue(path, f"no path to the stage {stage} goal")
            walk += [(stage, n) for n in path[1:]]
            if stage + 1 < rg.STAGE_COUNT:
                walk.append((stage + 1, walk[-1][1]))   # goal reached: flag flips, the player has not moved
        return walk

    def phis(self, walk):
        return [rg.route_potential(_at(node, stage)) for stage, node in walk]

    # --- coverage ------------------------------------------------------------------------------
    def test_covers_route_maps_and_not_others(self):
        for mid in rg.ROUTE_MAPS:
            self.assertTrue(rg.covers_map(mid), mid)
        for mid in ((1, 1), (3, 1), (9, 9), (24, 99), (0, 0)):
            self.assertFalse(rg.covers_map(mid), mid)

    def test_goals_are_reachable_from_each_other_and_from_the_start(self):
        g = self.g
        for stage in range(rg.STAGE_COUNT):
            self.assertIsNotNone(g.goal_distance(stage, ((24, 4), 6, 9)))
            for goal in g.goals:
                self.assertIsNotNone(g.goal_distance(stage, goal))

    def test_off_graph_state_is_neutral(self):
        for stage in range(rg.STAGE_COUNT):
            s = _state((1, 1), 4, 4, stage)
            self.assertEqual(rg.route_potential(s), 0.0)
            self.assertIsNone(rg.goal_remaining(s))
            f = rg.route_features(s)
            self.assertEqual(f.shape, (rg.ROUTE_FEATURE_DIM,))
            self.assertEqual(f.dtype, np.float32)
            expect = np.zeros(rg.ROUTE_FEATURE_DIM, dtype=np.float32)
            expect[stage] = 1.0
            np.testing.assert_array_equal(f, expect)

    def test_out_of_grid_position_on_a_covered_map_is_safe(self):
        s = _state((24, 4), 500, 500, 1)
        self.assertEqual(rg.route_potential(s), -(self.g.max_dist[1] + rg.UNREACHABLE_EXTRA + self.g.tail[1]))
        self.assertIsNone(rg.goal_remaining(s))
        f = rg.route_features(s)
        self.assertEqual(float(f[4:8].sum()), 0.0)
        self.assertEqual(float(f[8]), 1.0)

    # --- potential -----------------------------------------------------------------------------
    def test_phi_strictly_rises_one_step_at_a_time_across_every_goal(self):
        walk = self.full_walk()
        phis = self.phis(walk)
        for i in range(1, len(walk)):
            if walk[i][0] != walk[i - 1][0]:
                # goal handoff: same tile, the stage flips, Phi must not move
                self.assertEqual(walk[i][1], walk[i - 1][1])
                self.assertEqual(phis[i], phis[i - 1], f"Phi jumped at the handoff to stage {walk[i][0]}")
            else:
                self.assertEqual(phis[i] - phis[i - 1], 1.0, f"step {i}: {walk[i - 1]} -> {walk[i]}")

    def test_phi_is_zero_at_the_final_goal_and_minus_route_length_at_the_start(self):
        walk = self.full_walk()
        phis = self.phis(walk)
        self.assertEqual(phis[-1], 0.0)
        moves = sum(1 for i in range(1, len(walk)) if walk[i][0] == walk[i - 1][0])
        self.assertEqual(phis[0], -float(moves))

    def test_full_circle_telescopes(self):
        walk = self.full_walk()
        phis = self.phis(walk)
        self.assertAlmostEqual(sum(b - a for a, b in zip(phis, phis[1:])), phis[-1] - phis[0])
        # a reward of w * dPhi summed over the whole route is w * (route length), whatever the leg order
        w = 0.08
        total = sum(w * (b - a) for a, b in zip(phis, phis[1:]))
        self.assertAlmostEqual(total, w * -phis[0])
        # out and back over any single step returns to the same Phi (no hysteresis in a stage)
        for stage in range(rg.STAGE_COUNT):
            node = ((26, 1), 10, 30)
            fwd = self.g.fwd[node]
            nxt = next(n for n in fwd if n is not None and n[0] == node[0] and node in self.g.fwd[n])
            there = rg.route_potential(_at(nxt, stage))
            back = rg.route_potential(_at(node, stage))
            self.assertAlmostEqual((there - back) + (back - there), 0.0)

    def test_phi_is_continuous_at_each_handoff_from_nearby_tiles(self):
        # one step off the goal tile when the flag flips: Phi moves by at most that one step
        g = self.g
        for stage in range(rg.STAGE_COUNT - 1):
            goal = g.goals[stage]
            for nxt in g.fwd[goal]:
                if nxt is None:
                    continue
                before = rg.route_potential(_at(nxt, stage))
                after = rg.route_potential(_at(nxt, stage + 1))
                self.assertLessEqual(abs(after - before), 2.0, (stage, nxt))

    def test_phi_matches_return_nav_distances(self):
        """The merged graph reproduces the return_nav BFS (it has no extra step to enter the lab door)."""
        from src.return_nav import CHAIN, _nav
        nav = _nav()
        if nav is None:
            self.skipTest("return_nav unavailable")
        landing = self.g.warp_at[nav.goal]
        offset = self.g.dist[rg.STAGE_RETURN_EGG][landing]
        checked = 0
        for node, d in nav.dist.items():
            if node[0] not in CHAIN or node == nav.goal:
                continue
            ours = self.g.dist[rg.STAGE_RETURN_EGG].get(node)
            self.assertIsNotNone(ours, node)
            self.assertEqual(ours - offset, d, node)
            checked += 1
        self.assertGreater(checked, 1000)

    def test_spur_buildings_cost_the_walk_back_out(self):
        # Cherrygrove Pokecenter (a spur): entering it must lower Phi, not jump it to 0
        for stage in (0, 1, 2):
            outside = rg.route_potential(_state((26, 3), 29, 4, stage))
            door_in = self.g.warp_at[((26, 3), 29, 3)]
            inside = rg.route_potential(_at(door_in, stage))
            self.assertTrue(rg.covers_map((26, 5)))
            self.assertLess(inside, outside)
            self.assertLess(-inside, -outside + 6)

    # --- features ------------------------------------------------------------------------------
    def test_same_tile_gives_different_features_at_stage_1_and_3(self):
        s1 = rg.route_features(_state((26, 1), 10, 30, 1))
        s3 = rg.route_features(_state((26, 1), 10, 30, 3))
        self.assertFalse(np.array_equal(s1, s3))
        self.assertEqual(int(np.argmax(s1[:4])), 1)
        self.assertEqual(int(np.argmax(s3[:4])), 3)
        # Route 30 is walked north to Mr. Pokemon at stage 0 and south to Cherrygrove at stage 1
        north = rg.route_features(_state((26, 1), 10, 30, 0))
        south = rg.route_features(_state((26, 1), 10, 30, 1))
        self.assertEqual(float(north[4:8].sum()), 1.0)
        self.assertEqual(float(south[4:8].sum()), 1.0)
        self.assertFalse(np.array_equal(north[4:8], south[4:8]))

    def test_next_step_follows_the_shortest_path_to_every_goal(self):
        g = self.g
        starts = [((24, 4), 6, 9), ((26, 3), 20, 8), ((26, 1), 10, 30), ((10, 5), 20, 20), ((24, 3), 40, 5)]
        for stage in range(rg.STAGE_COUNT):
            for node in starts:
                want = g.goal_distance(stage, node)
                self.assertIsNotNone(want, (stage, node))
                steps = 0
                while True:
                    s = _at(node, stage)
                    f = rg.route_features(s)
                    rem = rg.goal_remaining(s)
                    self.assertAlmostEqual(float(f[8]), min(1.0, rem / rg.DIST_SCALE), places=6)
                    if rem == 0:
                        self.assertEqual(float(f[4:8].sum()), 0.0)
                        break
                    self.assertEqual(float(f[4:8].sum()), 1.0)
                    node = g.fwd[node][int(np.argmax(f[4:8]))]
                    steps += 1
                    self.assertLessEqual(steps, want)
                self.assertEqual(steps, want)

    def test_features_layout_at_a_known_tile(self):
        # Mr. Pokemon's door tile on Route 30 at stage 0: stage one-hot, one step (down, then in), distance < 100
        f = rg.route_features(_state((26, 1), 17, 6, 0))
        self.assertEqual(f.dtype, np.float32)
        np.testing.assert_array_equal(f[:4], [1, 0, 0, 0])
        self.assertEqual(float(f[4:8].sum()), 1.0)
        self.assertAlmostEqual(float(f[8]), rg.goal_remaining(_state((26, 1), 17, 6, 0)) / rg.DIST_SCALE, places=6)
        # far away (> 100 steps) the distance no longer saturates: scale is 400
        self.assertEqual(rg.DIST_SCALE, 400.0)
        far_state = _state((10, 5), 20, 20, 0)
        far = rg.route_features(far_state)
        d = rg.goal_remaining(far_state)
        self.assertGreater(d, 100.0)
        self.assertAlmostEqual(float(far[8]), min(1.0, d / 400.0), places=6)
        self.assertLess(float(far[8]), 1.0)

    def test_goal_tile_has_zero_remaining_and_no_step(self):
        for stage, goal in enumerate(self.g.goals):
            s = _at(goal, stage)
            self.assertEqual(rg.goal_remaining(s), 0.0)
            f = rg.route_features(s)
            self.assertEqual(float(f[4:9].sum()), 0.0)

    def test_unreachable_tile_has_no_next_step(self):
        g = self.g
        stuck = next(n for n in g.fwd if n[0] == (10, 5) and n not in g.dist[0])
        s = _at(stuck, 0)
        f = rg.route_features(s)
        self.assertEqual(float(f[4:8].sum()), 0.0)
        self.assertEqual(float(f[8]), 1.0)
        self.assertIsNone(rg.goal_remaining(s))
        self.assertLess(rg.route_potential(s), 0.0)

    # --- speed and safety ----------------------------------------------------------------------
    def test_calls_are_cheap(self):
        states = [_state((26, 1), 10, y, y % 4) for y in range(10, 40)]
        for fn in (rg.route_potential, rg.route_features, rg.goal_remaining):
            for s in states:
                fn(s)   # warm-up fills the per-tile caches
            n = 3000
            t = time.perf_counter()
            for i in range(n):
                fn(states[i % len(states)])
            per_call = (time.perf_counter() - t) / n
            self.assertLess(per_call, 0.001, f"{fn.__name__} took {per_call * 1e6:.0f} us per call")

    def test_graph_is_built_once_across_threads(self):
        seen = []
        threads = [threading.Thread(target=lambda: seen.append(rg._graph())) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(all(x is self.g for x in seen))


if __name__ == "__main__":
    unittest.main()
