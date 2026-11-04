"""No-oracle observation, minimal story route, battle features, six-button action space, supplies and long_v2."""

import unittest
from types import SimpleNamespace

import numpy as np

from src import rl_env
from src.battle_obs import (
    BATTLE_FEATURE_DIM,
    BATTLE_FEATURE_NAMES,
    ball_count,
    battle_features,
    equip_supplies,
    heal_item_count,
)
from src.ram_parser import GameState
from src.rl_env import ACTIONS, DEEP_OBS_DIM, N_ACTIONS, OBS_DIM, observe
from src.story_path import (
    MILESTONE_COUNT,
    MILESTONES,
    STORY_FEATURE_DIM,
    current_stage,
    milestones_done,
    story_features,
)


def _gs(**kw):
    base = {"map_group": 24, "map_number": 4, "x": 3, "y": 9, "tiles": {}, "event_flags": [0] * 300,
            "party_count": 1, "party_species": [155], "party_hp": [[19, 19]], "party_levels": [5],
            "party_moves": [[33, 43, 0, 0]]}
    base.update(kw)
    return GameState.from_dict(base)


class NoOracleObservationTests(unittest.TestCase):
    def test_dims(self):
        self.assertEqual(rl_env.BASE_OBS_DIM, 17)
        self.assertEqual(DEEP_OBS_DIM, 17 + STORY_FEATURE_DIM + BATTLE_FEATURE_DIM)
        self.assertEqual(DEEP_OBS_DIM, 77)
        self.assertEqual(len(observe(_gs(), set(), deep=True)), DEEP_OBS_DIM)
        self.assertEqual(N_ACTIONS, 6)
        self.assertEqual(OBS_DIM, len(observe(_gs(), set(), rl_env.LocalNav())))

    def test_direction_and_distance_features_are_gone(self):
        for name in ("route_features", "ROUTE_FEATURE_DIM", "quest_goal_vector", "navigation_potential",
                     "story_objective_maps", "GYM_PATH_WAYPOINTS", "_STAGE_OBJECTIVE_MAPS"):
            self.assertFalse(hasattr(rl_env, name), name)
        names = " ".join(BATTLE_FEATURE_NAMES)
        for banned in ("next", "dist", "direction", "goal"):
            self.assertNotIn(banned, names)

    def test_observation_is_identical_across_positions_with_the_same_state_except_position(self):
        # Two states that differ only in where the goal lies must give the same non-position features: there is no
        # goal-dependent column. (Same map, different tile: only x / y, neighbour-visited and the map bits may change.)
        a = observe(_gs(x=3, y=9), set(), deep=True)
        b = observe(_gs(x=40, y=2), set(), deep=True)
        diff = np.nonzero(a != b)[0].tolist()
        self.assertEqual(diff, [4, 5])  # x / y only

    def test_story_block_is_milestone_bits_and_map_id_only(self):
        f = story_features(_gs(map_group=26, map_number=3), seen_maps={(24, 3), (26, 3)})
        self.assertEqual(len(f), MILESTONE_COUNT + 2)
        self.assertEqual(list(f[:MILESTONE_COUNT]), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(f[-2], 26 / 32)


class StoryRouteTests(unittest.TestCase):
    def test_minimal_stage_list(self):
        names = [m.name for m in MILESTONES]
        self.assertEqual(names, ["route29", "cherrygrove", "route30", "route31", "violet_gate", "violet_city",
                                 "violet_gym", "beat_falkner"])
        maps = [m.map_id for m in MILESTONES if m.map_id]
        self.assertEqual(maps, [(24, 3), (26, 3), (26, 1), (26, 2), (26, 11), (10, 5), (10, 7)])
        joined = " ".join(names)
        for banned in ("egg", "elm", "rival", "mr_pokemon", "togepi"):
            self.assertNotIn(banned, joined)

    def test_stage_follows_the_map_and_the_badge(self):
        self.assertEqual(current_stage(_gs()), 0)
        self.assertEqual(current_stage(_gs(map_group=24, map_number=5)), 0)  # Elm's lab: off the path
        self.assertEqual(current_stage(_gs(map_group=26, map_number=3)), 2)
        self.assertEqual(current_stage(_gs(map_group=10, map_number=7)), 7)
        self.assertEqual(current_stage(_gs(badge_bits=1)), MILESTONE_COUNT)

    def test_milestones_survive_leaving_the_map_and_a_badge_completes_all(self):
        done = milestones_done(_gs(map_group=24, map_number=4), {(24, 3), (26, 3), (26, 1)})
        self.assertEqual(done, [True, True, True, False, False, False, False, False])
        self.assertTrue(all(milestones_done(_gs(badge_bits=1))))

    def test_total_reward_scale(self):
        from src.story_path import TOTAL_STORY_REWARD
        self.assertGreater(TOTAL_STORY_REWARD, 100.0)
        self.assertLess(TOTAL_STORY_REWARD, 200.0)


class BattleFeatureTests(unittest.TestCase):
    def _battle(self, **kw):
        base = dict(battle_mode=2, enemy_mon_species=16, enemy_level=4, enemy_hp=8, enemy_max_hp=16,
                    battle_mon_species=155, battle_mon_level=6, battle_mon_hp=10, battle_mon_max_hp=20,
                    battle_mon_moves=[33, 43, 0, 0], battle_mon_pp=[35, 30 | 0x40, 0, 0], battle_mon_status=0x08,
                    enemy_status=0x40, battle_menu_cursor=2, cur_move_num=1, menu_cursor_y=2, menu_cursor_x=1,
                    party_count=2, party_hp=[[10, 20], [0, 18]], party_levels=[6, 4], cur_battle_mon=0)
        base.update(kw)
        return _gs(**base)

    def test_width_names_and_range(self):
        f = battle_features(self._battle())
        self.assertEqual(f.shape, (BATTLE_FEATURE_DIM,))
        self.assertEqual(len(BATTLE_FEATURE_NAMES), BATTLE_FEATURE_DIM)
        self.assertTrue(np.all(f >= 0) and np.all(f <= 1.5))

    def test_fields_are_populated_in_battle(self):
        idx = {n: i for i, n in enumerate(BATTLE_FEATURE_NAMES)}
        f = battle_features(self._battle())
        self.assertEqual(f[idx["in_battle"]], 1.0)
        self.assertEqual(f[idx["is_trainer"]], 1.0)
        self.assertEqual(f[idx["is_wild"]], 0.0)
        self.assertAlmostEqual(f[idx["enemy_hp"]], 0.5)
        self.assertAlmostEqual(f[idx["active_hp"]], 0.5)
        self.assertAlmostEqual(f[idx["enemy_level"]], 0.04)
        self.assertEqual(f[idx["enemy_paralysis"]], 1.0)
        self.assertEqual(f[idx["active_poison"]], 1.0)
        self.assertAlmostEqual(f[idx["party0_hp"]], 0.5)
        self.assertEqual(f[idx["party1_hp"]], 0.0)
        self.assertAlmostEqual(f[idx["move0_id"]], 33 / 251)
        self.assertAlmostEqual(f[idx["move1_pp"]], 30 / 40)  # PP Up bits masked off
        self.assertEqual(f[idx["move0_usable"]], 1.0)
        self.assertEqual(f[idx["move2_usable"]], 0.0)
        self.assertAlmostEqual(f[idx["menu_cursor"]], 0.5)
        self.assertAlmostEqual(f[idx["move_cursor"]], 0.25)

    def test_outside_battle_zeroes_battle_wram_junk_but_keeps_party_and_bag(self):
        junk = self._battle(battle_mode=0, battle_mon_species=68, battle_mon_hp=17476)
        f = battle_features(junk)
        idx = {n: i for i, n in enumerate(BATTLE_FEATURE_NAMES)}
        self.assertEqual(f[idx["in_battle"]], 0.0)
        for name in ("enemy_hp", "active_hp", "active_species", "menu_cursor", "move0_pp", "move0_usable"):
            self.assertEqual(f[idx[name]], 0.0, name)
        self.assertAlmostEqual(f[idx["party0_hp"]], 0.5)
        self.assertAlmostEqual(f[idx["move0_id"]], 33 / 251)  # the lead's moves come from the party

    def test_bag_counts(self):
        s = SimpleNamespace(items=(0x12, 0x01, 0x11), item_counts=(5, 1, 2), balls=(5, 10, 4, 3, 255, 0))
        self.assertEqual(heal_item_count(s), 7)
        self.assertEqual(ball_count(s), 13)

    def test_missing_fields_never_raise(self):
        f = battle_features(SimpleNamespace(battle_mode=1))
        self.assertEqual(f.shape, (BATTLE_FEATURE_DIM,))


class ActionSpaceTests(unittest.TestCase):
    def test_six_buttons_with_b_and_no_start(self):
        self.assertEqual(ACTIONS, ("up", "down", "left", "right", "a", "b"))
        self.assertNotIn("start", ACTIONS)
        self.assertNotIn("select", ACTIONS)

    def test_no_sticky_actions_in_battle(self):
        from tools.train_deep_ppo import pick_executed_action

        rng = np.random.default_rng(0)
        for _ in range(200):
            self.assertEqual(pick_executed_action(4, 0, 0.99, rng, in_battle=True), 4)
        self.assertEqual(pick_executed_action(4, 0, 1.0 - 1e-12, rng, in_battle=False), 0)

    def test_policy_head_matches_the_action_space(self):
        from src.deep_policy import SpatialVisionPolicy

        pol = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM)
        self.assertEqual(pol.num_actions, N_ACTIONS)

    def test_trainer_does_not_import_the_battle_script(self):
        import inspect
        import tools.train_deep_ppo as t
        import tools.evaluate_policy as e
        import tools.play_live as p

        for mod in (t, e, p):
            self.assertNotIn("tactical_battle_step(", inspect.getsource(mod).replace("No tactical script", ""), mod.__name__)


class _FakeBus:
    """Byte-addressable fake for equip_supplies."""

    SYMS = {"wNumBalls": (1, 0xD8D7), "wBalls": (1, 0xD8D8), "wNumItems": (1, 0xD892), "wItems": (1, 0xD893)}

    def __init__(self):
        self.mem = {}

    def symbol(self, name):
        return self.SYMS[name]

    def read(self, bank, addr, n=1):
        return bytes(self.mem.get(addr + i, 0) for i in range(n))

    def write(self, bank, addr, data):
        for i, v in enumerate(data):
            self.mem[addr + i] = v


class SuppliesTests(unittest.TestCase):
    def test_gives_balls_and_potions_without_removing_items(self):
        bus = _FakeBus()
        bus.write(1, 0xD892, bytes([1]))
        bus.write(1, 0xD893, bytes([0x11, 2, 255]))  # one Super Potion
        equip_supplies(bus)
        self.assertEqual(bus.read(1, 0xD8D7)[0], 1)
        self.assertEqual(bus.read(1, 0xD8D8, 3), bytes([5, 10, 255]))
        self.assertEqual(bus.read(1, 0xD892)[0], 2)
        self.assertEqual(bus.read(1, 0xD893, 5), bytes([0x11, 2, 0x12, 5, 255]))
        equip_supplies(bus)  # idempotent
        self.assertEqual(bus.read(1, 0xD893, 5), bytes([0x11, 2, 0x12, 5, 255]))

    def test_tops_up_an_existing_potion_stack(self):
        bus = _FakeBus()
        bus.write(1, 0xD892, bytes([1]))
        bus.write(1, 0xD893, bytes([0x12, 1, 255]))
        equip_supplies(bus)
        self.assertEqual(bus.read(1, 0xD893, 3), bytes([0x12, 5, 255]))


class LongV2ConfigTests(unittest.TestCase):
    def test_preset(self):
        from src.train_config import PRESETS, load_config

        self.assertIn("long_v2", PRESETS)
        cfg = load_config("long_v2")
        self.assertEqual(cfg.name, "long_v2")
        self.assertEqual(cfg.start_states, ["new_bark_outside"])
        self.assertFalse(cfg.goal_potential)
        self.assertTrue(cfg.recurrent)

    def test_legacy_presets_still_load(self):
        from src.train_config import load_config

        for name in ("blueprint_v1", "long_v1", "legacy"):
            self.assertFalse(load_config(name).goal_potential)

    def test_goal_potential_is_an_explicit_opt_in(self):
        from src.train_config import apply_overrides, load_config

        cfg = apply_overrides(load_config("long_v2"), {"goal_potential": True})
        self.assertTrue(cfg.goal_potential)


if __name__ == "__main__":
    unittest.main()
