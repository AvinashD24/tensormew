import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src import rl_env
from src.rl_env import (
    COVERED_NEW_TILE_REWARD,
    ROUTE_FEATURE_DIM,
    ROUTE_PROGRESS_WEIGHT,
    behind_map_penalty_applies,
    initial_seen_events,
    observe,
    progress_reward,
)
from src.story_flags import EVENT_FLAGS

COMPONENT_KEYS = {
    "step", "bump", "route", "tile", "neighbor", "map", "warp", "interaction", "event", "badge",
    "catch", "level", "battle", "damage", "heal", "whiteout", "behind_map", "other",
}


class _S:
    """Minimal parsed state."""

    def __init__(self, group=24, number=3, x=5, y=5, events=(), flags=None):
        self.map_group, self.map_number, self.x, self.y = group, number, x, y
        self.events = set(events)
        self.event_flags = tuple(flags) if flags is not None else None
        self.badges = 0
        self.battle_mode = 0
        self.party_hp = ((20, 20),)
        self.party_levels = (5,)
        self.party_count = 1
        self.tiles = {"up": 0, "down": 0, "left": 0, "right": 0}
        self.script_running = 0
        self.script_mode = 0

    def has_event(self, name):
        return name in self.events


class _FakeRoute:
    """Route covering maps in COVERED; Phi = -(distance to x = 20)."""

    ROUTE_FEATURE_DIM = 9
    COVERED = {(24, 3), (26, 3)}

    @staticmethod
    def covers_map(map_id):
        return tuple(map_id) in _FakeRoute.COVERED

    @staticmethod
    def route_potential(state):
        return -float(abs(20 - state.x))

    @staticmethod
    def route_features(state):
        v = np.zeros(9, dtype=np.float32)
        v[0] = 1.0
        v[8] = 0.5
        return v


def _flags_with(name):
    flags = [0] * 256
    fid = EVENT_FLAGS[name]
    flags[fid // 8] |= 1 << (fid % 8)
    return flags


class SeedEventTests(unittest.TestCase):
    def test_seeded_flags_pay_nothing_unseeded_pay(self):
        flags = _flags_with("EVENT_GOT_A_POKEMON_FROM_ELM")
        s = _S(flags=flags)
        seeded = initial_seen_events(s)
        self.assertIn(EVENT_FLAGS["EVENT_GOT_A_POKEMON_FROM_ELM"], seeded)
        comp_a, comp_b = {}, {}
        progress_reward(s, s, False, set(), {(24, 3)}, seen_events=set(seeded), components=comp_a)
        progress_reward(s, s, False, set(), {(24, 3)}, seen_events=set(), components=comp_b)
        self.assertNotIn("event", comp_a)
        self.assertAlmostEqual(comp_b["event"], rl_env.KEY_EVENT_REWARDS["EVENT_GOT_A_POKEMON_FROM_ELM"])

    def test_no_flags_gives_empty_set(self):
        self.assertEqual(initial_seen_events(_S()), set())


class RouteRewardTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(rl_env, "route_graph", _FakeRoute)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_route_progress_and_exploration_shrink(self):
        before, after = _S(x=5), _S(x=6)
        comp = {}
        r, new_tile, _ = progress_reward(
            before, after, True, set(), {(24, 3)}, unvisited_neighbor=True, seen_waypoints=set(), components=comp
        )
        self.assertTrue(new_tile)
        self.assertAlmostEqual(comp["route"], ROUTE_PROGRESS_WEIGHT)
        self.assertAlmostEqual(comp["tile"], COVERED_NEW_TILE_REWARD)
        self.assertAlmostEqual(comp["neighbor"], 0.0)
        self.assertNotIn("other", comp)
        self.assertAlmostEqual(r, -0.01 + ROUTE_PROGRESS_WEIGHT + COVERED_NEW_TILE_REWARD)

    def test_route_telescopes_around_a_loop(self):
        total = 0.0
        xs = [5, 6, 7, 6, 5, 4, 5]
        for a, b in zip(xs, xs[1:]):
            comp = {}
            progress_reward(_S(x=a), _S(x=b), True, set(), {(24, 3)}, components=comp)
            total += comp["route"]
        self.assertAlmostEqual(total, 0.0)

    def test_waypoints_removed_when_covered_kept_otherwise(self):
        wp = set()
        comp = {}
        progress_reward(_S(26, 3, 5, 5), _S(26, 3, 6, 5), True, set(), {(26, 3)}, seen_waypoints=wp, components=comp)
        self.assertNotIn("other", comp)
        # Violet City (10, 5) is not covered by the fake route: waypoint (12) and full tile bonus return.
        comp = {}
        progress_reward(_S(10, 5, 5, 5), _S(10, 5, 6, 5), True, set(), {(10, 5)}, unvisited_neighbor=True,
                        seen_waypoints=wp, components=comp)
        self.assertAlmostEqual(comp["other"], rl_env.GYM_PATH_WAYPOINTS[(10, 5)])
        self.assertAlmostEqual(comp["tile"], 1.0)
        self.assertAlmostEqual(comp["neighbor"], rl_env.UNVISITED_NEIGHBOR_BONUS)
        self.assertNotIn("route", comp)

    def test_egg_return_waypoint_removed_when_covered(self):
        ev = {"EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}
        comp = {}
        progress_reward(_S(24, 3, 5, 5, ev), _S(24, 3, 6, 5, ev), True, set(), {(24, 3)}, seen_waypoints=set(), components=comp)
        self.assertNotIn("other", comp)

    def test_use_route_false_restores_legacy(self):
        comp = {}
        progress_reward(_S(24, 3, 5, 5), _S(24, 3, 6, 5), True, set(), {(24, 3)}, legacy_potential=True,
                        components=comp, use_route=False)
        self.assertNotIn("route", comp)
        self.assertIn("other", comp)

    def test_route_features_in_observation(self):
        s = _S(24, 3, 5, 5)
        obs = observe(s, set(), include_goals=True, include_route=True)
        self.assertEqual(len(obs), rl_env.DEEP_OBS_DIM)
        self.assertEqual(obs[-ROUTE_FEATURE_DIM], 1.0)
        self.assertEqual(obs[-1], 0.5)


class NoRouteTests(unittest.TestCase):
    def test_degrades_to_zero_route_and_full_exploration(self):
        with mock.patch.object(rl_env, "route_graph", None):
            comp = {}
            progress_reward(_S(x=5), _S(x=6), True, set(), {(24, 3)}, unvisited_neighbor=True, components=comp)
            self.assertNotIn("route", comp)
            self.assertAlmostEqual(comp["neighbor"], rl_env.UNVISITED_NEIGHBOR_BONUS)
            self.assertTrue(np.all(rl_env.route_features(_S()) == 0))
            self.assertEqual(rl_env.route_potential(_S()), 0.0)


class ComponentSumTests(unittest.TestCase):
    def _check(self, *args, **kwargs):
        comp = {}
        r, _, _ = progress_reward(*args, components=comp, **kwargs)
        self.assertTrue(set(comp) <= COMPONENT_KEYS, set(comp) - COMPONENT_KEYS)
        self.assertEqual(sum(comp.values()), r)  # exact: the reward is accumulated from the same terms
        plain, _, _ = progress_reward(*args, **kwargs)
        return comp, r

    def test_sum_equals_reward_across_scenarios(self):
        for route in (None, _FakeRoute):
            with mock.patch.object(rl_env, "route_graph", route):
                # movement, new map, waypoint, event, badge, level, damage, low hp, whiteout
                b = _S(10, 5, 5, 5)
                a = _S(10, 7, 5, 5, flags=_flags_with("EVENT_BEAT_FALKNER"))
                a.badges = 1
                a.party_levels = (8,)
                a.party_hp = ((3, 20),)
                a.fainted = True
                self._check(b, a, True, set(), set(), unvisited_neighbor=True, seen_events=set(), seen_waypoints=set(),
                            shaping=0.3)
                # bump
                self._check(_S(), _S(), False, set(), {(24, 3)})
                # heal in a Pokecenter, then catch
                h0, h1 = _S(26, 5, 3, 3), _S(26, 5, 3, 3)
                h0.party_hp, h1.party_hp = ((5, 20),), ((20, 20),)
                self._check(h0, h1, False, set(), {(26, 5)}, seen_waypoints=set())
                c1 = _S(24, 3, 5, 5)
                c1.party_count, c1.party_species = 2, (155, 16)
                self._check(_S(24, 3, 5, 5), c1, False, set(), {(24, 3)}, caught_counts={})
                # battle win
                w0 = _S()
                w0.battle_mode = 2
                self._check(w0, _S(), False, set(), {(24, 3)})

    def test_positional_signature_unchanged(self):
        r, t, m = progress_reward(_S(x=5), _S(x=6), True, set(), set())
        self.assertTrue(t and m)


class BehindMapTests(unittest.TestCase):
    def test_plain_step_back_is_penalised(self):
        self.assertTrue(behind_map_penalty_applies(_S(24, 3), _S(24, 4), {(24, 4)}, _S(24, 4)))
        self.assertTrue(behind_map_penalty_applies((24, 3), (24, 4), {(24, 4)}, None))

    def test_not_penalised_if_already_inside_or_not_finished(self):
        self.assertFalse(behind_map_penalty_applies(_S(24, 4), _S(24, 4), {(24, 4)}, _S(24, 4)))
        self.assertFalse(behind_map_penalty_applies(_S(24, 3), _S(26, 3), {(24, 4)}, _S(26, 3)))

    def test_egg_return_allows_going_back(self):
        carrying = _S(24, 4, events={"EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"})
        self.assertFalse(behind_map_penalty_applies(_S(24, 3), carrying, {(24, 4)}, carrying))
        delivered = _S(24, 4, events={"EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON", "EVENT_GAVE_MYSTERY_EGG_TO_ELM"})
        self.assertTrue(behind_map_penalty_applies(_S(24, 3), delivered, {(24, 4)}, delivered))


if __name__ == "__main__":
    unittest.main()
