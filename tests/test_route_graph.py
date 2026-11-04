"""The route graph is now only the OPTIONAL goal potential (off by default); it is no longer an observation."""

import threading
import time
import unittest
from types import SimpleNamespace

from src import route_graph as rg
from src.collision import COLL_HOP_DOWN, COLL_HOP_LEFT, is_ledge
from src.mover import walkable
from src.story_path import STORY_PATH_STEPS

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

    def test_minimal_story_path_is_309_steps_without_any_errand(self):
        # New Bark start tile -> in front of Falkner, no egg / Elm / rival detour: 309 actions (319 before the ledge-jump fix).
        d = rg.goal_remaining(_state(*START))
        self.assertIsNotNone(d)
        self.assertEqual(d, 309.0)

    def test_phi_rises_one_per_step_toward_the_goal(self):
        g = self.g
        node = START
        phi = rg.route_potential(_state(*node))
        self.assertEqual(phi, -309.0)
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


def _route(g):
    """The BFS shortest path as a node list (first neighbour that is one step closer, the order the dashboard uses)."""
    node = rg.PATH_START
    path = [node]
    while g.dist[0][node] > 0:
        d0 = g.dist[0][node]
        node = next(n for n in g.fwd[node] if n is not None and g.dist[0].get(n) == d0 - 1)
        path.append(node)
    return path


_STEP = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


class LedgeTests(unittest.TestCase):
    """Ledges (Crystal): walk onto a hop tile from above or the side, press its direction to jump TWO tiles, never climb."""

    @classmethod
    def setUpClass(cls):
        cls.g = rg._graph()
        if cls.g is None:
            raise unittest.SkipTest("ROM not available")
        cls.hops = [(mid, x, y, d) for mid, m in cls.g.maps.items() for (x, y), tile in m.cells.items()
                    for d in rg._HOP_DIRS.get(tile, ())]

    def test_route_maps_have_ledges_and_every_one_has_a_cliff_wall_behind_its_hop_direction(self):
        self.assertGreater(len(self.hops), 40)
        for mid, x, y, d in self.hops:
            dx, dy = _STEP[d]
            beyond = self.g.maps[mid].cells.get((x + dx, y + dy))
            # the tile right after a ledge tile in its hop direction is the cliff face: a wall, so nothing can be walked
            # up (or across) the ledge from the far side, whatever the graph rules say
            self.assertIsNotNone(beyond, (mid, x, y, d))
            self.assertFalse(walkable(beyond), (mid, x, y, d, hex(beyond)))

    def test_moving_against_a_ledge_is_impossible(self):
        for mid, x, y, d in self.hops:
            dx, dy = _STEP[d]
            below = (mid, x + dx, y + dy)   # the cliff tile next to the ledge, in its hop direction
            self.assertIsNone(self.g._step(below, _OPP[d]), (mid, x, y, d))

    def test_a_ledge_tile_is_entered_from_above_and_from_the_side(self):
        r29 = (24, 3)
        self.assertEqual(self.g.maps[r29].cells[(12, 8)], COLL_HOP_DOWN)
        self.assertEqual(self.g._step((r29, 12, 7), "down"), (r29, 12, 8))     # from above
        self.assertEqual(self.g._step((r29, 11, 8), "right"), (r29, 12, 8))    # from the side (a gap tile), as in the game
        self.assertEqual(self.g._step((r29, 12, 8), "right"), (r29, 13, 8))    # and along the ledge row
        self.assertEqual(self.g._step((r29, 12, 8), "up"), (r29, 12, 7))       # standing on the edge tile you can step back

    def test_a_jump_lands_two_tiles_away_in_the_hop_direction_only(self):
        r29, r31 = (24, 3), (26, 2)
        self.assertEqual(self.g._step((r29, 12, 8), "down"), (r29, 12, 10))    # over the wall at (12, 9)
        self.assertEqual(self.g._step((r31, 21, 10), "down"), (r31, 21, 12))
        self.assertEqual(self.g.maps[r29].cells[(9, 6)], COLL_HOP_LEFT)
        self.assertEqual(self.g._step((r29, 9, 6), "left"), (r29, 7, 6))
        # (12, 8) is a hop-DOWN tile: it cannot be jumped sideways
        self.assertNotEqual(self.g._step((r29, 12, 8), "left"), (r29, 10, 8))
        for mid, x, y, d in self.hops:
            dx, dy = _STEP[d]
            out = self.g._step((mid, x, y), d)
            land = self.g.maps[mid].cells.get((x + 2 * dx, y + 2 * dy))
            if land is not None and walkable(land) and (mid, x + 2 * dx, y + 2 * dy) not in self.g.warp_at:
                self.assertEqual(out, (mid, x + 2 * dx, y + 2 * dy), (mid, x, y, d))

    def test_the_route_jumps_two_ledges_and_never_walks_up_one(self):
        route = _route(self.g)
        jumps = [(a, b) for a, b in zip(route, route[1:]) if a[0] == b[0] and abs(a[1] - b[1]) + abs(a[2] - b[2]) == 2]
        self.assertEqual(len(jumps), 2)
        self.assertIn((((24, 3), 9, 6), ((24, 3), 7, 6)), jumps)               # Route 29, hop left over the wall at (8, 6)
        self.assertIn((((26, 2), 21, 10), ((26, 2), 21, 12)), jumps)           # Route 31, hop down over the wall at (21, 11)


class CorrectedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = rg._graph()
        if cls.g is None:
            raise unittest.SkipTest("ROM not available")
        cls.route = _route(cls.g)

    def test_length_constants_agree_everywhere(self):
        from tools import dashboard_events
        self.assertEqual(rg.path_total(), float(STORY_PATH_STEPS))
        self.assertEqual(len(self.route) - 1, STORY_PATH_STEPS)
        self.assertEqual(dashboard_events.PATH_TOTAL, STORY_PATH_STEPS)
        self.assertEqual(STORY_PATH_STEPS, 309)

    def test_no_illegal_ledge_crossing_on_the_route(self):
        g = self.g
        for i, (a, b) in enumerate(zip(self.route, self.route[1:]), start=1):
            self.assertEqual(g.dist[0][a] - g.dist[0][b], 1, (i, a, b))      # monotone: +1 progress per step
            if a[0] != b[0]:
                continue                                                      # warp / connection
            m = g.maps[a[0]]
            dx, dy = b[1] - a[1], b[2] - a[2]
            ta, tb = m.cells[(a[1], a[2])], m.cells[(b[1], b[2])]
            self.assertTrue(walkable(tb), (i, b, hex(tb)))                    # no wall, water, cut tree on the way
            if abs(dx) + abs(dy) == 2:                                        # a jump: from a hop tile, in its direction
                d = next(k for k, (sx, sy) in _STEP.items() if (2 * sx, 2 * sy) == (dx, dy))
                self.assertIn(d, rg._HOP_DIRS.get(ta, ()), (i, a, b))
            else:
                self.assertEqual(abs(dx) + abs(dy), 1, (i, a, b))
                d = next(k for k, v in _STEP.items() if v == (dx, dy))
                if is_ledge(tb):                                              # entering a ledge tile: never against it
                    self.assertNotIn(_OPP[d], rg._HOP_DIRS.get(tb, ()), (i, a, b))

    def test_every_graph_edge_is_a_legal_ledge_move(self):
        g = self.g
        for src, outs in g.fwd.items():
            for d, dst in zip(rg.DIR_NAMES, outs):
                if dst is None or src[0] != dst[0]:
                    continue
                dx, dy = dst[1] - src[1], dst[2] - src[2]
                ts, td = g.maps[src[0]].cells[(src[1], src[2])], g.maps[dst[0]].cells[(dst[1], dst[2])]
                if abs(dx) + abs(dy) == 2:
                    self.assertIn(d, rg._HOP_DIRS.get(ts, ()), (src, d, dst))
                elif td in rg._HOP_DIRS:
                    self.assertNotIn(_OPP[d], rg._HOP_DIRS[td], (src, d, dst))

    def test_violet_gym_entrance_and_the_tile_in_front_of_falkner(self):
        """Verified in the emulator (stage2_gym_entered.state, trainers' beaten flags set): the Violet City door at (18, 17) is
        the Gym (10, 7) and not the Mart / school / Pokecenter, the 20 gym steps from the entrance mat (4, 15) all move, and
        (5, 2) is where Falkner (5, 1) can be challenged (face up, A).  Bird Keepers Abe (2, 10) and Rod (7, 6) face right /
        left with sight 3, so the path crosses their lines on (4, 10) / (5, 10) and (4, 6) whatever the route."""
        g = self.g
        city = g.maps[(10, 5)]
        gym_doors = [(x, y) for (y, x, _wid, grp, num) in city.warps if (grp, num) == (10, 7)]
        self.assertEqual(gym_doors, [(18, 17)])
        self.assertEqual(g.warp_at[((10, 5), 18, 17)], ((10, 7), 4, 15))
        self.assertEqual(g.fwd[((10, 5), 18, 18)][0], ((10, 7), 4, 15))       # up from the tile below the door
        self.assertEqual(g.goal_distance(0, ((10, 7), 4, 15)), 20)
        self.assertTrue(walkable(g.maps[(10, 7)].cells[(5, 2)]))
        for x, y in ((2, 2), (3, 2), (6, 2), (7, 2)):                          # the wall stubs beside the approach tile
            self.assertFalse(walkable(g.maps[(10, 7)].cells[(x, y)]), (x, y))


    def test_dashboard_layout_matches_the_route_and_marks_the_ledges(self):
        import json
        from tools.dashboard import STATIC_DIR
        data = json.loads((STATIC_DIR / "world_layout.json").read_text(encoding="utf-8"))
        self.assertEqual(data["legs"][0]["steps"], STORY_PATH_STEPS)
        maps = {m["id"]: m for m in data["maps"]}
        for mid, m in self.g.maps.items():
            key = f"{mid[0]},{mid[1]}"
            if key not in maps:
                continue
            for (x, y), tile in m.cells.items():
                if tile in rg._HOP_DIRS and y < len(maps[key]["rows"]) and x < len(maps[key]["rows"][y]):
                    self.assertEqual(maps[key]["rows"][y][x], "l", (key, x, y))
        route_by_map = {}
        for (mid, x, y) in self.route:
            route_by_map.setdefault(f"{mid[0]}_{mid[1]}", []).append([x, y])
        for seg in data["legs"][0]["segments"]:
            ours = route_by_map[seg["map"]]
            self.assertEqual(seg["pts"][0], ours[0], seg["map"])
            self.assertEqual(seg["pts"][-1], ours[-1], seg["map"])
            self.assertTrue(all(c in ours for c in seg["pts"]), seg["map"])


if __name__ == "__main__":
    unittest.main()
