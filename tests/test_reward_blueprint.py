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

    def test_waypoints_and_legacy_potentials_off_in_route_mode(self):
        wp = set()
        comp = {}
        progress_reward(_S(26, 3, 5, 5), _S(26, 3, 6, 5), True, set(), {(26, 3)}, seen_waypoints=wp, components=comp)
        self.assertNotIn("other", comp)
        # Violet City (10, 5) is not covered by the fake route: no waypoint and no legacy potential either
        # (route mode); the full exploration terms come back instead.
        comp = {}
        progress_reward(_S(10, 5, 5, 5), _S(10, 5, 6, 5), True, set(), {(10, 5)}, unvisited_neighbor=True,
                        seen_waypoints=wp, legacy_potential=True, components=comp)
        self.assertNotIn("other", comp)
        self.assertAlmostEqual(comp["tile"], 1.0)
        self.assertAlmostEqual(comp["neighbor"], rl_env.UNVISITED_NEIGHBOR_BONUS)
        self.assertNotIn("route", comp)
        # use_route=False keeps the waypoint
        comp = {}
        progress_reward(_S(10, 5, 5, 5), _S(10, 5, 6, 5), True, set(), {(10, 5)}, seen_waypoints=set(),
                        use_route=False, components=comp)
        self.assertAlmostEqual(comp["other"], rl_env.GYM_PATH_WAYPOINTS[(10, 5)])

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


def _sn(**kw):
    s = _S(kw.pop("group", 24), kw.pop("number", 3), kw.pop("x", 5), kw.pop("y", 5))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class BudgetTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(rl_env, "route_graph", _FakeRoute)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_budget_pay_helper(self):
        b = {}
        self.assertEqual(rl_env.budget_pay(b, "battle", 6.0), 6.0)
        self.assertEqual(rl_env.budget_pay(b, "battle", 6.0), rl_env.BATTLE_BUDGET - 6.0)
        self.assertEqual(rl_env.budget_pay(b, "battle", 1.0), 0.0)
        self.assertEqual(rl_env.budget_pay(None, "battle", 6.0), 6.0)
        self.assertEqual(rl_env.budget_pay({}, "battle", -1.0), -1.0)  # penalties are never capped

    def test_level_ups_capped(self):
        b, total = {}, 0.0
        for lv in range(5, 40):
            comp = {}
            progress_reward(_sn(party_levels=(lv,)), _sn(party_levels=(lv + 1,)), False, set(), {(24, 3)},
                            components=comp, budget=b)
            total += comp.get("level", 0.0)
        self.assertAlmostEqual(total, rl_env.LEVEL_BUDGET)
        comp = {}  # the first payment is full size
        progress_reward(_sn(party_levels=(5,)), _sn(party_levels=(6,)), False, set(), {(24, 3)}, components=comp, budget={})
        self.assertAlmostEqual(comp["level"], rl_env.LEVEL_REWARD_SCALE)

    def test_catch_does_not_pay_level_and_is_capped(self):
        b, total_catch, total_level = {}, 0.0, 0.0
        for i in range(10):
            before = _sn(party_count=1, party_levels=(5,), party_species=(155,))
            after = _sn(party_count=2, party_levels=(5, 12), party_species=(155, 16 + i))
            comp = {}
            progress_reward(before, after, False, set(), {(24, 3)}, caught_counts={}, components=comp, budget=b)
            total_catch += comp.get("catch", 0.0)
            total_level += comp.get("level", 0.0)
        self.assertEqual(total_level, 0.0)
        self.assertAlmostEqual(total_catch, rl_env.CATCH_BUDGET)

    def test_wild_win_only_for_genuine_win_and_key_changes(self):
        def battle_end(enemy_hp, **after_kw):
            return _sn(battle_mode=1, enemy_hp=10, enemy_max_hp=20), _sn(enemy_hp=enemy_hp, enemy_max_hp=20, **after_kw)
        budget = {}
        comp = {}
        progress_reward(*battle_end(10), False, set(), {(24, 3)}, components=comp, budget=budget)  # fled
        self.assertEqual(comp.get("battle", 0.0), 0.0)
        comp = {}
        b, a = battle_end(0, party_count=2, party_levels=(5, 3), party_species=(155, 16))
        progress_reward(b, a, False, set(), {(24, 3)}, caught_counts={}, components=comp, budget=budget)  # caught
        self.assertEqual(comp.get("battle", 0.0), 0.0)
        paid = []
        for _ in range(5):
            comp = {}
            progress_reward(*battle_end(0), False, set(), {(24, 3)}, components=comp, budget=budget)
            paid.append(comp["battle"])
        self.assertEqual(paid[:3], [rl_env.WILD_WIN_FULL] * 3)  # the party level never changed, the win count did
        self.assertEqual(paid[3], rl_env.WILD_WIN_LATE)
        self.assertLessEqual(sum(paid), rl_env.BATTLE_BUDGET)

    def test_heal_budget_and_proportional(self):
        def heal(cur):
            h0 = _sn(group=26, number=5, party_hp=((cur, 20),))
            h1 = _sn(group=26, number=5, party_hp=((20, 20),))
            comp = {}
            progress_reward(h0, h1, False, set(), {(26, 5)}, seen_waypoints=set(), components=comp, budget={})
            return comp.get("heal", 0.0)
        self.assertLess(heal(19), 1.5)                        # one missing HP: almost nothing (was +15.2)
        self.assertAlmostEqual(heal(1), rl_env.HEAL_BUDGET)   # near-full heal pays ~24, capped to the budget
        b, total = {}, 0.0
        for _ in range(6):
            comp = {}
            h0, h1 = _sn(group=26, number=5, party_hp=((2, 20),)), _sn(group=26, number=5, party_hp=((20, 20),))
            progress_reward(h0, h1, False, set(), {(26, 5)}, components=comp, budget=b)
            total += comp.get("heal", 0.0)
        self.assertLessEqual(total, rl_env.HEAL_BUDGET + 1e-9)

    def test_whiteout_nets_strongly_negative(self):
        faint_b = _sn(party_hp=((5, 20),))
        faint_a = _sn(party_hp=((0, 20),), fainted=True)
        c1, c2 = {}, {}
        r1, _, _ = progress_reward(faint_b, faint_a, False, set(), {(24, 3)}, components=c1, budget={})
        # next step: teleported to a covered map, party restored, the route potential jumps up
        tele = _sn(group=26, number=3, x=0, y=5, party_hp=((20, 20),))
        r2, _, _ = progress_reward(faint_a, tele, False, set(), {(24, 3), (26, 3)}, components=c2, budget={},
                                   seen_waypoints=set())
        self.assertNotIn("heal", c2)
        self.assertLessEqual(c2.get("route", 0.0), 0.0)
        self.assertLess(r1 + r2, -20.0)
        pc = _sn(group=26, number=5, party_hp=((20, 20),))  # a teleport into a Pokecenter map pays no heal either
        c3 = {}
        progress_reward(faint_a, pc, False, set(), {(24, 3), (26, 5)}, components=c3, budget={}, seen_waypoints=set())
        self.assertNotIn("heal", c3)

    def test_uncovered_long_walk_stops_paying_after_cap(self):
        b, seen_tiles, seen_maps, total = {}, set(), {(10, 5)}, 0.0
        first = None
        for x in range(2, 202):
            comp = {}
            progress_reward(_sn(group=10, number=5, x=x), _sn(group=10, number=5, x=x + 1), True, seen_tiles,
                            seen_maps, unvisited_neighbor=True, components=comp, budget=b)
            paid = comp.get("tile", 0.0) + comp.get("neighbor", 0.0) + comp.get("map", 0.0)
            first = paid if first is None else first
            total += paid
        self.assertAlmostEqual(first, 1.0 + rl_env.UNVISITED_NEIGHBOR_BONUS)  # first payments are full size
        self.assertAlmostEqual(total, rl_env.EXPLORE_BUDGET)
        seen, t = set(), 0.0  # and without a budget the same walk keeps paying
        for x in range(2, 202):
            c = {}
            progress_reward(_sn(group=10, number=5, x=x), _sn(group=10, number=5, x=x + 1), True, seen, {(10, 5)},
                            unvisited_neighbor=True, components=c)
            t += c.get("tile", 0.0)
        self.assertGreater(t, 100.0)

    def test_map_bonus_zero_on_covered_maps_with_budget(self):
        comp = {}
        progress_reward(_sn(group=24, number=3), _sn(group=26, number=3), True, set(), {(24, 3)}, components=comp, budget={})
        self.assertEqual(comp["map"], 0.0)
        comp = {}  # without a budget: as before
        progress_reward(_sn(group=24, number=3), _sn(group=26, number=3), True, set(), {(24, 3)}, components=comp)
        self.assertEqual(comp["map"], 10.0)
        comp = {}  # uncovered map with a budget: paid, within the exploration budget
        progress_reward(_sn(group=24, number=3), _sn(group=10, number=5), True, set(), {(24, 3)}, components=comp, budget={})
        self.assertEqual(comp["map"], 10.0)

    def test_enter_and_leave_a_covered_map_nets_no_map_reward(self):
        comp_in, comp_out, b, seen_maps = {}, {}, {}, {(24, 3)}
        progress_reward(_sn(), _sn(group=26, number=3), True, set(), seen_maps, components=comp_in, budget=b)
        progress_reward(_sn(group=26, number=3), _sn(), True, set(), seen_maps, components=comp_out, budget=b)
        self.assertEqual(comp_in["map"] + comp_out.get("map", 0.0), 0.0)

    def test_legacy_potential_never_fires_in_route_mode(self):
        # Violet PC -> PC 2F (neither on the fake route); hybrid callers pass legacy_potential=True or shaping=None
        for kw in ({}, {"legacy_potential": True}):
            comp = {}
            progress_reward(_sn(group=10, number=10, x=4, y=4), _sn(group=20, number=1, x=0, y=7), True, set(),
                            {(10, 10)}, components=comp, **kw)
            self.assertNotIn("other", comp)
        comp = {}  # use_route=False keeps it
        progress_reward(_sn(group=10, number=10, x=4, y=4), _sn(group=20, number=1, x=0, y=7), True, set(),
                        {(10, 10)}, components=comp, use_route=False)
        self.assertIn("other", comp)

    def test_named_nonkey_flag_pays_two_key_flag_keeps_value(self):
        name = next(n for n in EVENT_FLAGS if n not in rl_env.KEY_EVENT_REWARDS and n != "EVENT_RIVAL_CHERRYGROVE_CITY")
        comp = {}
        progress_reward(_sn(event_flags=tuple([0] * 256)), _sn(event_flags=tuple(_flags_with(name))), False, set(),
                        {(24, 3)}, seen_events=set(), components=comp)
        self.assertAlmostEqual(comp["event"], 2.0)
        key = "EVENT_BEAT_FALKNER"
        comp = {}
        progress_reward(_sn(event_flags=tuple([0] * 256)), _sn(event_flags=tuple(_flags_with(key))), False, set(),
                        {(24, 3)}, seen_events=set(), components=comp)
        self.assertGreaterEqual(comp["event"], rl_env.KEY_EVENT_REWARDS[key])

    def test_a_press_is_not_a_bump(self):
        comp = {}
        progress_reward(_S(), _S(), False, set(), {(24, 3)}, info=rl_env.StepInfo(consecutive_bumps=0), components=comp)
        self.assertNotIn("bump", comp)
        comp = {}
        progress_reward(_S(), _S(), False, set(), {(24, 3)}, info=rl_env.StepInfo(consecutive_bumps=2), components=comp)
        self.assertAlmostEqual(comp["bump"], -0.1)


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
