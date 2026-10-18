import unittest
from types import SimpleNamespace

from src.return_nav import CHAIN, RIVAL_X, _nav, egg_return_potential


def _state(mid, x, y, hp=((20, 20),), egg=True, gave=False):
    flags = {"EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON": egg, "EVENT_GAVE_MYSTERY_EGG_TO_ELM": gave}
    return SimpleNamespace(
        map_group=mid[0], map_number=mid[1], x=x, y=y, party_hp=hp, has_event=lambda n: flags.get(n, False)
    )


class ReturnRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nav = _nav()
        if cls.nav is None:
            raise unittest.SkipTest("ROM not available")

    def test_every_cell_with_a_route_has_a_better_neighbour(self):
        """No dead-end maxima: following the potential uphill always reaches Elm's Lab."""
        nav = self.nav
        stuck = []
        for node, d in nav.dist.items():
            if node == nav.goal:
                continue
            # rev[m] lists cells that can step INTO m, so a successor of `node` is any m with node in rev[m]
            succ = [m for m, nexts in nav.rev.items() if node in nexts and nav.dist.get(m, 1e9) < d]
            if not succ:
                stuck.append(node)
        self.assertEqual(stuck, [])

    def test_potential_rises_by_one_step_along_the_whole_route(self):
        nav = self.nav
        node = ((26, 1), 17, 6)
        steps = 0
        while node != nav.goal:
            nxt = next(m for m, nexts in nav.rev.items() if node in nexts and nav.dist[m] == nav.dist[node] - 1)
            self.assertAlmostEqual(nav.phi_node(nxt) - nav.phi_node(node), nav.scale)
            node, steps = nxt, steps + 1
        self.assertEqual(steps, nav.dist[((26, 1), 17, 6)])

    def test_route_crosses_the_rival_trigger(self):
        nav = self.nav
        node, seen = ((26, 1), 17, 6), set()
        while node != nav.goal:
            node = next(m for m, nexts in nav.rev.items() if node in nexts and nav.dist[m] == nav.dist[node] - 1)
            seen.add(node)
        self.assertTrue({((26, 3), RIVAL_X, 6), ((26, 3), RIVAL_X, 7)} & seen)

    def test_building_doors_do_not_pay(self):
        outside = egg_return_potential(_state((26, 3), 23, 4))
        door = egg_return_potential(_state((26, 3), 23, 3))  # Cherrygrove Mart door
        inside = egg_return_potential(_state((26, 4), 3, 6))
        self.assertLess(door, outside)
        self.assertLess(inside, outside)

    def test_leaving_mr_pokemons_house_is_progress(self):
        inside = egg_return_potential(_state((26, 10), 3, 5))
        mat = egg_return_potential(_state((26, 10), 3, 7))
        outside = egg_return_potential(_state((26, 1), 17, 6))
        self.assertGreater(mat, inside)
        self.assertGreater(outside, mat)

    def test_low_hp_before_the_rival_heads_to_the_pokecenter_then_resumes(self):
        hurt = ((2, 20),)
        west_of_rival = (_state((26, 3), 27, 5, hp=hurt), _state((26, 3), 28, 5, hp=hurt))
        self.assertGreater(egg_return_potential(west_of_rival[1]), egg_return_potential(west_of_rival[0]))
        # past the rival trigger the detour is off: the normal route east is preferred
        self.assertGreater(
            egg_return_potential(_state((26, 3), RIVAL_X + 2, 6, hp=hurt)),
            egg_return_potential(_state((26, 3), RIVAL_X, 6, hp=hurt)),
        )

    def test_uncovered_maps_fall_through(self):
        self.assertIsNone(egg_return_potential(_state((10, 5), 5, 5)))

    def test_goal_interior_is_above_the_door(self):
        door = egg_return_potential(_state((24, 4), 6, 3))
        lab = egg_return_potential(_state((24, 5), 4, 6))
        self.assertGreater(lab, door)




class DirectionTests(unittest.TestCase):
    def test_following_the_direction_always_reaches_a_lower_distance(self):
        from src.return_nav import _DIRS, _nav, egg_return_direction

        nav = _nav()
        for (mid, x, y), dist in list(nav.dist.items())[:400]:
            if dist == 0:
                continue
            d = egg_return_direction(_state(mid, x, y))
            self.assertIsNotNone(d, (mid, x, y))
            dx, dy = _DIRS[d]
            nxt = nav._step(mid, x + dx, y + dy, d) or (mid, x + dx, y + dy)
            self.assertLess(nav.dist.get(nxt, 0), dist)

    def test_uncovered_map_has_no_direction(self):
        from src.return_nav import egg_return_direction

        self.assertIsNone(egg_return_direction(_state((26, 10), 3, 3)))
