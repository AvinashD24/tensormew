"""Reward tests for the no-oracle design: story milestones, generic exploration, learned-battle terms, budgets."""

import unittest
from unittest import mock

from src import rl_env
from src.rl_env import behind_map_penalty_applies, initial_seen_events, progress_reward
from src.story_flags import EVENT_FLAGS
from src.story_path import BADGE_REWARD, KEY_EVENT_REWARDS, MAP_MILESTONES, MILESTONES

COMPONENT_KEYS = {
    "step", "bump", "route", "tile", "neighbor", "map", "milestone", "warp", "interaction", "event", "badge",
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


def _sn(**kw):
    s = _S(kw.pop("group", 24), kw.pop("number", 3), kw.pop("x", 5), kw.pop("y", 5))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _flags_with(name):
    flags = [0] * 256
    fid = EVENT_FLAGS[name]
    flags[fid // 8] |= 1 << (fid % 8)
    return flags


class SeedEventTests(unittest.TestCase):
    def test_seeded_flags_pay_nothing_unseeded_pay(self):
        flags = _flags_with("EVENT_BEAT_FALKNER")
        s = _S(flags=flags)
        seeded = initial_seen_events(s)
        self.assertIn(EVENT_FLAGS["EVENT_BEAT_FALKNER"], seeded)
        comp_a, comp_b = {}, {}
        progress_reward(s, s, False, set(), {(24, 3)}, seen_events=set(seeded), components=comp_a)
        progress_reward(s, s, False, set(), {(24, 3)}, seen_events=set(), components=comp_b)
        self.assertNotIn("event", comp_a)
        self.assertAlmostEqual(comp_b["event"], KEY_EVENT_REWARDS["EVENT_BEAT_FALKNER"])

    def test_no_flags_gives_empty_set(self):
        self.assertEqual(initial_seen_events(_S()), set())


class MilestoneTests(unittest.TestCase):
    def test_first_arrival_on_each_story_map_pays_once(self):
        seen = {(24, 4)}
        total = 0.0
        for m in MILESTONES:
            if m.map_id is None:
                continue
            comp = {}
            progress_reward(_sn(group=24, number=4), _sn(group=m.map_id[0], number=m.map_id[1]), True, set(), seen,
                            components=comp)
            self.assertAlmostEqual(comp["milestone"], m.reward)
            self.assertNotIn("map", comp)
            total += comp["milestone"]
            comp = {}  # second arrival pays nothing
            progress_reward(_sn(group=24, number=4), _sn(group=m.map_id[0], number=m.map_id[1]), True, set(), seen,
                            components=comp)
            self.assertNotIn("milestone", comp)
        self.assertAlmostEqual(total, sum(m.reward for m in MILESTONES))

    def test_start_map_and_off_path_maps_do_not_pay_milestones(self):
        self.assertNotIn((24, 4), MAP_MILESTONES)
        comp = {}
        progress_reward(_sn(group=24, number=4), _sn(group=24, number=5), True, set(), {(24, 4)}, components=comp,
                        budget={})
        self.assertNotIn("milestone", comp)
        self.assertAlmostEqual(comp["map"], rl_env.NEW_MAP_REWARD)  # Elm's lab: ordinary exploration, no goal

    def test_egg_elm_and_rival_are_not_goals(self):
        for name in ("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON", "EVENT_GAVE_MYSTERY_EGG_TO_ELM",
                     "EVENT_RIVAL_CHERRYGROVE_CITY", "EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST"):
            self.assertNotIn(name, KEY_EVENT_REWARDS)
            comp = {}
            progress_reward(_sn(event_flags=tuple([0] * 256)), _sn(event_flags=tuple(_flags_with(name))), False, set(),
                            {(24, 3)}, seen_events=set(), components=comp)
            self.assertLessEqual(comp.get("event", 0.0), 0.1)

    def test_badge_and_falkner_event(self):
        comp = {}
        after = _sn(badges=1, event_flags=tuple(_flags_with("EVENT_BEAT_FALKNER")))
        progress_reward(_sn(), after, False, set(), {(24, 3)}, seen_events=set(), components=comp)
        self.assertAlmostEqual(comp["badge"], BADGE_REWARD)
        self.assertAlmostEqual(comp["event"], KEY_EVENT_REWARDS["EVENT_BEAT_FALKNER"])

    def test_named_nonkey_flag_pays_nothing_unnamed_pays_a_token(self):
        name = next(n for n in EVENT_FLAGS if n not in KEY_EVENT_REWARDS)
        comp = {}
        progress_reward(_sn(event_flags=tuple([0] * 256)), _sn(event_flags=tuple(_flags_with(name))), False, set(),
                        {(24, 3)}, seen_events=set(), components=comp)
        self.assertEqual(comp["event"], 0.0)
        unnamed_id = next(i for i in range(2000) if i not in rl_env.REVERSE_EVENT_FLAGS)
        flags = [0] * 256
        flags[unnamed_id // 8] |= 1 << (unnamed_id % 8)
        comp = {}
        progress_reward(_sn(event_flags=tuple([0] * 256)), _sn(event_flags=tuple(flags)), False, set(), {(24, 3)},
                        seen_events=set(), components=comp)
        self.assertLessEqual(comp["event"], 0.1)
        self.assertGreater(comp["event"], 0.0)


class _Route:
    """Fake route graph covering two maps; Phi = -(distance to x = 20)."""

    @staticmethod
    def covers_map(map_id):
        return tuple(map_id) in {(24, 3), (26, 3)}

    @staticmethod
    def route_potential(state):
        return -float(abs(20 - state.x))


class GoalPotentialTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(rl_env, "route_graph", _Route)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_off_by_default_no_route_term(self):
        comp = {}
        progress_reward(_S(x=5), _S(x=6), True, set(), {(24, 3)}, components=comp)
        self.assertNotIn("route", comp)

    def test_optional_flag_enables_potential_and_it_telescopes(self):
        total = 0.0
        xs = [5, 6, 7, 6, 5, 4, 5]
        for a, b in zip(xs, xs[1:]):
            comp = {}
            progress_reward(_S(x=a), _S(x=b), True, set(), {(24, 3)}, components=comp, goal_potential=True)
            total += comp["route"]
        self.assertAlmostEqual(total, 0.0)
        comp = {}
        progress_reward(_S(x=5), _S(x=6), True, set(), {(24, 3)}, components=comp, goal_potential=True)
        self.assertAlmostEqual(comp["route"], rl_env.ROUTE_PROGRESS_WEIGHT)

    def test_trainconfig_default_is_off(self):
        from src.train_config import preset, PRESETS

        for name in PRESETS:
            self.assertFalse(preset(name).goal_potential, name)


class ExplorationTests(unittest.TestCase):
    def test_new_tile_is_small_and_budgeted(self):
        b, seen_tiles, seen_maps, total = {}, set(), {(10, 5)}, 0.0
        for x in range(2, 600):
            comp = {}
            progress_reward(_sn(group=10, number=5, x=x), _sn(group=10, number=5, x=x + 1), True, seen_tiles, seen_maps,
                            unvisited_neighbor=True, components=comp, budget=b)
            total += comp.get("tile", 0.0) + comp.get("neighbor", 0.0) + comp.get("map", 0.0)
        self.assertAlmostEqual(total, rl_env.EXPLORE_BUDGET)
        comp = {}
        progress_reward(_sn(group=10, number=5, x=2), _sn(group=10, number=5, x=3), True, set(), {(10, 5)},
                        components=comp, budget={})
        self.assertAlmostEqual(comp["tile"], rl_env.NEW_TILE_REWARD)

    def test_a_press_is_not_a_bump(self):
        comp = {}
        progress_reward(_S(), _S(), False, set(), {(24, 3)}, info=rl_env.StepInfo(consecutive_bumps=0), components=comp)
        self.assertNotIn("bump", comp)
        comp = {}
        progress_reward(_S(), _S(), False, set(), {(24, 3)}, info=rl_env.StepInfo(consecutive_bumps=2), components=comp)
        self.assertAlmostEqual(comp["bump"], -0.1)


class BudgetTests(unittest.TestCase):
    def test_budget_pay_helper(self):
        b = {}
        self.assertEqual(rl_env.budget_pay(b, "battle", 6.0), 6.0)
        self.assertEqual(rl_env.budget_pay(b, "battle", 6.0), 6.0)
        self.assertEqual(rl_env.budget_pay(b, "battle", 10.0), rl_env.BATTLE_BUDGET - 12.0)
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

    def test_catch_is_small_does_not_pay_level_and_is_capped(self):
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
        self.assertLess(rl_env.catch_reward_for_species(1), 5.0)

    def test_heal_budget_and_proportional(self):
        def heal(cur):
            h0 = _sn(group=26, number=5, party_hp=((cur, 20),))
            h1 = _sn(group=26, number=5, party_hp=((20, 20),))
            comp = {}
            progress_reward(h0, h1, False, set(), {(26, 5)}, seen_waypoints=set(), components=comp, budget={})
            return comp.get("heal", 0.0)
        self.assertLess(heal(19), 1.5)
        self.assertAlmostEqual(heal(1), rl_env.HEAL_BUDGET)
        b, total = {}, 0.0
        for _ in range(6):
            comp = {}
            h0, h1 = _sn(group=26, number=5, party_hp=((2, 20),)), _sn(group=26, number=5, party_hp=((20, 20),))
            progress_reward(h0, h1, False, set(), {(26, 5)}, components=comp, budget=b)
            total += comp.get("heal", 0.0)
        self.assertLessEqual(total, rl_env.HEAL_BUDGET + 1e-9)

    def test_potion_heal_outside_a_pokecenter_pays_a_little(self):
        h0, h1 = _sn(party_hp=((5, 20),)), _sn(party_hp=((20, 20),))
        comp = {}
        progress_reward(h0, h1, False, set(), {(24, 3)}, components=comp, budget={})
        self.assertGreater(comp["heal"], 0.0)
        self.assertLess(comp["heal"], 4.0)

    def test_level_up_hp_growth_is_not_a_heal(self):
        b4, b5 = _sn(party_hp=((10, 20),), party_levels=(5,)), _sn(party_hp=((24, 25),), party_levels=(6,))
        comp = {}
        progress_reward(b4, b5, False, set(), {(24, 3)}, components=comp, budget={})
        self.assertNotIn("heal", comp)


class BattleRewardTests(unittest.TestCase):
    def _fight(self, e0, e1, **kw):
        b = _sn(battle_mode=1, enemy_hp=e0, enemy_max_hp=20, enemy_mon_species=16, **kw)
        a = _sn(battle_mode=1, enemy_hp=e1, enemy_max_hp=20, enemy_mon_species=16, **kw)
        comp = {}
        progress_reward(b, a, False, set(), {(24, 3)}, components=comp, budget={})
        return comp

    def test_damage_dealt_and_ko_pay_generically(self):
        self.assertAlmostEqual(self._fight(20, 10)["battle"], rl_env.DAMAGE_DEALT_WEIGHT * 0.5)
        ko = self._fight(5, 0)["battle"]
        self.assertAlmostEqual(ko, rl_env.DAMAGE_DEALT_WEIGHT * 0.25 + rl_env.KO_REWARD)
        self.assertNotIn("battle", self._fight(10, 10))
        self.assertNotIn("battle", self._fight(10, 20))  # enemy healing is not a payment

    def test_new_enemy_species_is_not_damage(self):
        b = _sn(battle_mode=2, enemy_hp=20, enemy_max_hp=20, enemy_mon_species=16)
        a = _sn(battle_mode=2, enemy_hp=5, enemy_max_hp=20, enemy_mon_species=17)
        comp = {}
        progress_reward(b, a, False, set(), {(24, 3)}, components=comp, budget={})
        self.assertNotIn("battle", comp)

    def test_damage_taken_ally_faint_and_whiteout(self):
        b = _sn(party_hp=((10, 20), (10, 20)), party_count=2, party_levels=(5, 5))
        a = _sn(party_hp=((0, 20), (10, 20)), party_count=2, party_levels=(5, 5))
        comp = {}
        progress_reward(b, a, False, set(), {(24, 3)}, components=comp, budget={})
        self.assertLess(comp["damage"], -rl_env.ALLY_FAINT_PENALTY)
        faint_b = _sn(party_hp=((5, 20),))
        faint_a = _sn(party_hp=((0, 20),), fainted=True)
        comp = {}
        progress_reward(faint_b, faint_a, False, set(), {(24, 3)}, components=comp, budget={})
        self.assertAlmostEqual(comp["whiteout"], -rl_env.WHITEOUT_PENALTY)

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
        self.assertEqual(paid[:3], [rl_env.WILD_WIN_FULL] * 3)
        self.assertEqual(paid[3], rl_env.WILD_WIN_LATE)
        self.assertLessEqual(sum(paid), rl_env.BATTLE_BUDGET)

    def test_trainer_win(self):
        w0 = _sn(battle_mode=2)
        comp = {}
        progress_reward(w0, _S(), False, set(), {(24, 3)}, components=comp, budget={})
        self.assertAlmostEqual(comp["battle"], rl_env.TRAINER_WIN)

    def test_whiteout_nets_strongly_negative_and_teleport_pays_no_heal(self):
        faint_b = _sn(party_hp=((5, 20),))
        faint_a = _sn(party_hp=((0, 20),), fainted=True)
        r1, _, _ = progress_reward(faint_b, faint_a, False, set(), {(24, 3)}, components={}, budget={})
        pc = _sn(group=26, number=5, party_hp=((20, 20),))
        c3 = {}
        r2, _, _ = progress_reward(faint_a, pc, False, set(), {(24, 3), (26, 5)}, components=c3, budget={},
                                   seen_waypoints=set())
        self.assertNotIn("heal", c3)
        self.assertLess(r1 + r2, -10.0)


class ComponentSumTests(unittest.TestCase):
    def _check(self, *args, **kwargs):
        comp = {}
        r, _, _ = progress_reward(*args, components=comp, **kwargs)
        self.assertTrue(set(comp) <= COMPONENT_KEYS, set(comp) - COMPONENT_KEYS)
        self.assertEqual(sum(comp.values()), r)  # exact: the reward is accumulated from the same terms
        return comp, r

    def test_sum_equals_reward_across_scenarios(self):
        b = _S(10, 5, 5, 5)
        a = _S(10, 7, 5, 5, flags=_flags_with("EVENT_BEAT_FALKNER"))
        a.badges = 1
        a.party_levels = (8,)
        a.party_hp = ((3, 20),)
        a.fainted = True
        self._check(b, a, True, set(), set(), unvisited_neighbor=True, seen_events=set(), seen_waypoints=set(),
                    shaping=0.3, budget={})
        self._check(_S(), _S(), False, set(), {(24, 3)})
        h0, h1 = _S(26, 5, 3, 3), _S(26, 5, 3, 3)
        h0.party_hp, h1.party_hp = ((5, 20),), ((20, 20),)
        self._check(h0, h1, False, set(), {(26, 5)}, seen_waypoints=set())
        c1 = _S(24, 3, 5, 5)
        c1.party_count, c1.party_species = 2, (155, 16)
        self._check(_S(24, 3, 5, 5), c1, False, set(), {(24, 3)}, caught_counts={})
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


if __name__ == "__main__":
    unittest.main()
