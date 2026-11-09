"""long_v9: progress-first rewards (diminishing wild-battle returns, level gate by stage target, new-species catches, milestone
scale and speed bonus), the time-in-episode observation scalar, per-term maxima, a farming simulation and preset isolation.
The older presets (long_v3..long_v8) must stay exactly as they were."""

import argparse
import dataclasses
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import anti_farm as af
from src import heal_care as hc
from src import rl_env
from src import story_path as sp
from src.config import ROM_PATH
from src.heal_reward import heal_payment
from src.rl_env import BUDGET_CAPS, progress_reward
from src.train_config import PRESETS, ConfigError, from_dict, load_config, preset

ROOT = Path(__file__).resolve().parents[1]
HAVE_ROM = Path(ROM_PATH).exists()
OLD = tuple(n for n in PRESETS if n != "long_v9")


class _S:
    """Minimal parsed state."""

    def __init__(self, **kw):
        self.map_group, self.map_number, self.x, self.y = 24, 3, 5, 5
        self.badges, self.battle_mode, self.fainted = 0, 0, False
        self.party_hp, self.party_levels, self.party_count = ((20, 20),), (5,), 1
        self.party_species = (155,)
        self.enemy_hp = self.enemy_max_hp = self.enemy_mon_species = 0
        self.event_flags = None
        self.script_running = self.script_mode = 0
        for k, v in kw.items():
            setattr(self, k, v)


def wild_battle(budget, farm, comps_sum, hp=20, species=16):
    """Play one wild battle (start, 20-HP enemy KO'd in one hit, end) through progress_reward; returns the battle payment."""
    paid = 0.0
    steps = [(_S(), _S(battle_mode=1, enemy_hp=hp, enemy_max_hp=hp, enemy_mon_species=species)),
             (_S(battle_mode=1, enemy_hp=hp, enemy_max_hp=hp, enemy_mon_species=species),
              _S(battle_mode=1, enemy_hp=0, enemy_max_hp=hp, enemy_mon_species=species)),
             (_S(battle_mode=1, enemy_hp=0, enemy_max_hp=hp, enemy_mon_species=species),
              _S(enemy_hp=0, enemy_max_hp=hp, enemy_mon_species=species))]
    for b, a in steps:
        comp = {}
        progress_reward(b, a, False, set(), {(24, 3)}, components=comp, budget=budget, farm=farm)
        paid += comp.get("battle", 0.0)
    return paid


def v9_farm():
    return preset("long_v9").anti_farm_params()


def v9_budget():
    return dict(preset("long_v9").budget_cap_overrides())


class DiminishingReturnsMath(unittest.TestCase):
    def test_multiplier_values_and_floor(self):
        self.assertEqual([af.battle_multiplier(n) for n in (1, 2)], [1.0, 1.0])
        self.assertAlmostEqual(af.battle_multiplier(3), 0.6)
        self.assertAlmostEqual(af.battle_multiplier(4), 0.36)
        self.assertAlmostEqual(af.battle_multiplier(5), 0.216)
        self.assertAlmostEqual(af.battle_multiplier(6), 0.1296)
        self.assertAlmostEqual(af.battle_multiplier(7), 0.1)    # 0.6**5 = 0.078 is below the floor
        self.assertAlmostEqual(af.battle_multiplier(500), 0.1)
        self.assertEqual(af.battle_multiplier(0), 1.0)
        values = [af.battle_multiplier(n) for n in range(1, 40)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_speed_multiplier_bounds(self):
        self.assertAlmostEqual(af.speed_multiplier(0, 2048, 1.5), 1.5)
        self.assertAlmostEqual(af.speed_multiplier(1024, 2048, 1.5), 1.25)
        self.assertAlmostEqual(af.speed_multiplier(2048, 2048, 1.5), 1.0)
        self.assertAlmostEqual(af.speed_multiplier(5000, 2048, 1.5), 1.0)   # past the limit: clamped
        self.assertAlmostEqual(af.speed_multiplier(-3, 2048, 1.5), 1.5)
        self.assertEqual(af.speed_multiplier(10, 2048, 1.0), 1.0)           # off
        self.assertEqual(af.speed_multiplier(10, 0, 1.5), 1.0)
        for e in range(0, 2100, 37):
            self.assertTrue(1.0 <= af.speed_multiplier(e, 2048, 1.5) <= 1.5)

    def test_level_pays(self):
        self.assertTrue(af.level_pays((5,), None))
        self.assertTrue(af.level_pays((5, 7), 8))
        self.assertFalse(af.level_pays((5, 8), 8))     # the highest level is already at the target
        self.assertFalse(af.level_pays((12,), 8))
        self.assertFalse(af.level_pays((5,), 0))       # target 0 = nothing more to level for
        self.assertTrue(af.level_pays((), 8))

    def test_validation(self):
        for bad in ({"battle_decay": 0.0}, {"battle_decay": 1.5}, {"battle_floor": -0.1}, {"milestone_speed_bonus": 0.5},
                    {"milestone_reward_scale": 0.0}, {"wild_battle_budget": -1.0}, {"catch_budget": -1.0}):
            with self.assertRaises(ConfigError, msg=str(bad)):
                from_dict(bad, preset("long_v9"))
        with self.assertRaises(ConfigError):   # the level gate needs the dwell table
            from_dict({"dwell_penalty": False, "stagnation_penalty": False}, preset("long_v9"))


class FarmingSimulation(unittest.TestCase):
    def test_wild_battle_farming_never_exceeds_the_budget(self):
        farm, budget = v9_farm(), v9_budget()
        paid = [wild_battle(budget, farm, None) for _ in range(300)]
        cap = preset("long_v9").wild_battle_budget
        self.assertLessEqual(sum(paid), cap + 1e-9)
        self.assertAlmostEqual(sum(paid), cap)               # a farmer does reach the cap, and nothing more
        self.assertEqual(budget["wild_battles"], 300)
        self.assertAlmostEqual(paid[0], 3.0 * 0.4 + 1.0 * 0.4 + 1.5 * 0.4)   # full wild battle = 2.2
        self.assertAlmostEqual(paid[1], paid[0])
        self.assertAlmostEqual(paid[2], paid[0] * 0.6)
        self.assertAlmostEqual(paid[3], cap - sum(paid[:3]))  # 2.2 + 2.2 + 1.32 = 5.72: the 4th battle only gets the 0.28 left
        self.assertEqual(sum(paid[4:]), 0.0)                  # budget exhausted after four battles

    def test_diminishing_returns_without_a_cap(self):
        farm, budget = v9_farm(), {"cap:battle": 1e9}
        paid = [wild_battle(budget, farm, None) for _ in range(9)]
        full = paid[0]
        for n, p in enumerate(paid, start=1):
            self.assertAlmostEqual(p, full * af.battle_multiplier(n), msg=f"battle {n}")
        self.assertAlmostEqual(paid[-1], full * 0.1)

    def test_old_presets_keep_the_shared_budget_of_20(self):
        budget = {}
        paid = [wild_battle(budget, None, None) for _ in range(40)]
        self.assertLessEqual(sum(paid), rl_env.BATTLE_BUDGET + 1e-9)
        self.assertGreater(sum(paid), 10.0)
        self.assertNotIn("wild_battles", budget)

    def test_trainer_battles_are_undiminished_under_their_own_budget(self):
        farm, budget = v9_farm(), v9_budget()
        for _ in range(50):
            wild_battle(budget, farm, None)       # exhaust the wild budget
        comp = {}
        progress_reward(_S(battle_mode=2), _S(), False, set(), {(24, 3)}, components=comp, budget=budget, farm=farm)
        self.assertAlmostEqual(comp["battle"], rl_env.TRAINER_WIN)          # win 5, not scaled, not decayed
        comp = {}
        b = _S(battle_mode=2, enemy_hp=40, enemy_max_hp=40, enemy_mon_species=17)
        a = _S(battle_mode=2, enemy_hp=0, enemy_max_hp=40, enemy_mon_species=17)
        progress_reward(b, a, False, set(), {(24, 3)}, components=comp, budget=budget, farm=farm)
        self.assertAlmostEqual(comp["battle"], rl_env.DAMAGE_DEALT_WEIGHT + rl_env.KO_REWARD)
        # the trainer cap is separate and larger
        total = 0.0
        for _ in range(100):
            comp = {}
            progress_reward(_S(battle_mode=2), _S(), False, set(), {(24, 3)}, components=comp, budget=budget, farm=farm)
            total += comp.get("battle", 0.0)
        self.assertLessEqual(total, preset("long_v9").trainer_battle_budget)
        self.assertGreater(preset("long_v9").trainer_battle_budget, preset("long_v9").wild_battle_budget)

    def test_a_battle_archive_start_mid_battle_counts_as_the_first_battle(self):
        farm, budget = v9_farm(), v9_budget()
        b = _S(battle_mode=1, enemy_hp=20, enemy_max_hp=20, enemy_mon_species=16)
        a = _S(battle_mode=1, enemy_hp=0, enemy_max_hp=20, enemy_mon_species=16)
        comp = {}
        progress_reward(b, a, False, set(), {(24, 3)}, components=comp, budget=budget, farm=farm)
        self.assertAlmostEqual(comp["battle"], (1.5 + 1.0) * 0.4)

    def test_catch_only_new_species(self):
        farm, budget, counts = v9_farm(), v9_budget(), {155: 1}
        before = _S()
        after = _S(party_count=2, party_species=(155, 16), party_levels=(5, 3), party_hp=((20, 20), (10, 10)))
        comp = {}
        progress_reward(before, after, False, set(), {(24, 3)}, caught_counts=counts, components=comp, budget=budget, farm=farm)
        self.assertAlmostEqual(comp["catch"], 2.0)
        comp = {}   # the same species again: nothing
        again = _S(party_count=3, party_species=(155, 16, 16), party_levels=(5, 3, 3), party_hp=((20, 20),) * 3)
        progress_reward(after, again, False, set(), {(24, 3)}, caught_counts=counts, components=comp, budget=budget, farm=farm)
        self.assertEqual(comp.get("catch", 0.0), 0.0)
        # the old rule: a duplicate still pays 1.5
        counts = {155: 1, 16: 1}
        comp = {}
        progress_reward(after, again, False, set(), {(24, 3)}, caught_counts=counts, components=comp, budget={})
        self.assertAlmostEqual(comp["catch"], 1.5)
        # budget 4: at most two new species
        budget, total = v9_budget(), 0.0
        for sp_id in range(30, 40):
            comp = {}
            c = _S(party_count=2, party_species=(155, sp_id))
            progress_reward(_S(), c, False, set(), {(24, 3)}, caught_counts={}, components=comp, budget=budget, farm=farm)
            total += comp.get("catch", 0.0)
        self.assertAlmostEqual(total, preset("long_v9").catch_budget)


class LevelGate(unittest.TestCase):
    def level_up(self, levels_before, target, farm=True, gain=1):
        b = _S(party_levels=levels_before, party_hp=((20, 20),) * len(levels_before), party_count=len(levels_before))
        a_levels = (levels_before[0] + gain,) + tuple(levels_before[1:])
        a = _S(party_levels=a_levels, party_hp=((24, 24),) + ((20, 20),) * (len(levels_before) - 1),
               party_count=len(levels_before))
        comp = {}
        progress_reward(b, a, False, set(), {(24, 3)}, components=comp, budget=v9_budget(),
                        farm=v9_farm() if farm else None, level_target=target)
        return comp.get("level", 0.0)

    def test_pays_below_target_and_nothing_at_or_above_it(self):
        self.assertGreater(self.level_up((5,), 8), 0.0)          # 5 -> 6 with target 8: needed levelling stays rewarded
        self.assertGreater(self.level_up((7,), 8), 0.0)          # 7 -> 8 reaches the target: still paid
        self.assertEqual(self.level_up((8,), 8), 0.0)            # grinding above the target pays nothing
        self.assertEqual(self.level_up((11,), 8), 0.0)
        self.assertEqual(self.level_up((5,), 0), 0.0)            # target 0 = Falkner beaten: nothing more to level for
        self.assertGreater(self.level_up((5,), None), 0.0)       # no target known: ungated

    def test_old_presets_are_ungated(self):
        self.assertGreater(self.level_up((11,), 8, farm=False), 0.0)

    def test_gate_follows_the_stage_table(self):
        cfg = preset("long_v9")
        self.assertEqual(len(cfg.dwell_level_targets), cfg.story_profile().count + 1)
        self.assertEqual(max(cfg.dwell_level_targets), 12)

    def test_level_income_is_bounded_per_episode(self):
        farm, budget, total = v9_farm(), v9_budget(), 0.0
        lv = 5
        for _ in range(40):
            b = _S(party_levels=(lv,), party_hp=((20, 20),))
            a = _S(party_levels=(lv + 1,), party_hp=((24, 24),))
            comp = {}
            progress_reward(b, a, False, set(), {(24, 3)}, components=comp, budget=budget, farm=farm, level_target=10)
            total += comp.get("level", 0.0)
            lv += 1
        self.assertLessEqual(total, rl_env.LEVEL_BUDGET + 1e-9)
        self.assertGreater(total, 0.0)


class MilestoneScaling(unittest.TestCase):
    def test_milestone_event_and_badge_follow_the_multiplier(self):
        mult = 1.5 * 1.25
        comp, comp1 = {}, {}
        b, a = _S(map_group=24, map_number=4), _S(map_group=24, map_number=3)
        progress_reward(b, a, False, set(), {(24, 4)}, components=comp, budget={}, milestone_mult=mult)
        progress_reward(b, a, False, set(), {(24, 4)}, components=comp1, budget={})
        self.assertAlmostEqual(comp["milestone"], comp1["milestone"] * mult)
        badge_b, badge_a = _S(), _S(badges=1)
        comp, comp1 = {}, {}
        progress_reward(badge_b, badge_a, False, set(), {(24, 3)}, components=comp, budget={}, milestone_mult=mult)
        progress_reward(badge_b, badge_a, False, set(), {(24, 3)}, components=comp1, budget={})
        self.assertAlmostEqual(comp["badge"], comp1["badge"] * mult)
        self.assertAlmostEqual(comp1["badge"], sp.BADGE_REWARD)

    def test_unnamed_events_and_novelty_are_not_scaled(self):
        flags = [0] * 256
        flags[200 // 8] |= 1 << (200 % 8)    # some unnamed flag
        comp = {}
        progress_reward(_S(), _S(event_flags=tuple(flags)), False, set(), {(24, 3)}, seen_events=set(), components=comp,
                        budget={}, milestone_mult=2.0)
        self.assertAlmostEqual(comp["event"], rl_env.UNNAMED_EVENT_REWARD)


class Arithmetic(unittest.TestCase):
    """Per-episode maxima of v8 vs v9 (the table in research/analysis/design_hardcoded_vs_learned.md)."""

    @staticmethod
    def farm_max(cfg):
        explore = cfg.explore_budget
        heal = cfg.heal_budget if cfg.heal_reward else rl_env.HEAL_BUDGET
        extras = (cfg.battle_survive_budget if cfg.battle_survive_bonus else 0.0) + (cfg.potion_use_budget if cfg.potion_use_reward else 0.0)
        if cfg.anti_farm:
            battle, trainer, catch = cfg.wild_battle_budget, 0.0, cfg.catch_budget
        else:
            battle, trainer, catch = rl_env.BATTLE_BUDGET, 0.0, rl_env.CATCH_BUDGET
        return {"battle": battle, "level": rl_env.LEVEL_BUDGET, "catch": catch, "heal": heal, "explore": explore,
                "extras": extras, "trainer": trainer}

    @staticmethod
    def route_pay(cfg):
        p = cfg.story_profile()
        mult = cfg.milestone_reward_scale
        return {"path": cfg.path_progress_scale * p.path_steps, "milestones": p.total_reward * mult,
                "milestones_fast": p.total_reward * mult * cfg.milestone_speed_bonus}

    def test_a_walk_along_the_route_pays_far_more_than_the_best_farm(self):
        v9, v8 = preset("long_v9"), preset("long_v8")
        farm9, farm8 = sum(self.farm_max(v9).values()), sum(self.farm_max(v8).values())
        route9, route8 = self.route_pay(v9), self.route_pay(v8)
        self.assertLess(farm9, farm8)
        self.assertGreaterEqual(route9["path"], 3.0 * farm9)                       # the requirement: >= 2-3x
        self.assertGreaterEqual(route9["path"] + route9["milestones"], 10.0 * farm9)
        self.assertGreater(route9["path"], 3.0 * route8["path"])
        self.assertGreater(route9["milestones"], route8["milestones"])
        self.assertGreater(v9.path_progress_scale, v8.path_progress_scale)

    @unittest.skipUnless(HAVE_ROM, "needs the ROM for the real route graph")
    def test_the_real_route_total_pays_more_than_farming(self):
        from src import route_graph
        total = route_graph.path_total(egg=True)
        self.assertIsNotNone(total)
        v9 = preset("long_v9")
        self.assertGreater(v9.path_progress_scale * total, 3.0 * sum(self.farm_max(v9).values()))
        self.assertEqual(float(total), float(v9.story_profile().path_steps))

    def test_a_clean_wild_battle_is_worth_less_than_the_smallest_milestone(self):
        v9 = preset("long_v9")
        one_battle = 3.0 * v9.wild_battle_scale + 1.0 * v9.wild_battle_scale + 1.5 * v9.wild_battle_scale
        smallest = min(m.reward for m in v9.story_profile().milestones if m.reward > 0) * v9.milestone_reward_scale
        self.assertGreater(smallest, 3.0 * one_battle)
        # one tile of progress beats the per-step farm rate of a battle (a battle takes dozens of steps)
        self.assertGreater(v9.path_progress_scale, one_battle / 40.0)

    def test_fast_arrival_pays_more_but_never_more_than_the_cap(self):
        v9 = preset("long_v9")
        base = 10.0 * v9.milestone_reward_scale
        fast = base * af.speed_multiplier(100, 2048, v9.milestone_speed_bonus)
        slow = base * af.speed_multiplier(2000, 2048, v9.milestone_speed_bonus)
        self.assertGreater(fast, slow)
        self.assertLessEqual(fast, 10.0 * v9.milestone_reward_scale * v9.milestone_speed_bonus)
        self.assertGreaterEqual(slow, base)

    def test_heal_trips_stay_net_positive_when_hp_is_low(self):
        v9, v8 = preset("long_v9"), preset("long_v8")
        self.assertEqual((v9.heal_reward_scale, v9.heal_budget), (v8.heal_reward_scale, v8.heal_budget))
        low, full = _S(party_hp=((2, 20),), map_group=26, map_number=5), _S(party_hp=((20, 20),), map_group=26, map_number=5)
        pay = heal_payment(low, full, v9.heal_reward_scale)
        # a 100-step walk back at low HP: step cost + the full exposure cost (retreat / stagnation / dwell are exempt on a heal trip)
        walk = 100 * (0.01 + hc.EXPOSURE_MAX)
        self.assertGreater(pay, 2.0 * walk)
        self.assertTrue(v9.heal_trip_exempt and v9.lowhp_exposure_penalty and v9.terminate_on_whiteout)
        self.assertEqual(rl_env.WHITEOUT_PENALTY, 15.0)    # kept


class TimeFeature(unittest.TestCase):
    def setUp(self):
        self.addCleanup(sp.set_profile, False)

    def test_dims(self):
        base = rl_env.deep_obs_dim(False)
        self.assertEqual(rl_env.deep_obs_dim(False, time_aware=True), base + 1)
        self.assertEqual(rl_env.deep_obs_dim(True, time_aware=True), rl_env.deep_obs_dim(True) + 1)
        self.assertEqual(rl_env.deep_obs_dim(True, time_aware=True), 81)
        self.assertEqual(rl_env.DEEP_OBS_DIM, 77)

    def test_observe_appends_one_clipped_scalar_at_the_end(self):
        from src.ram_parser import GameState
        st = GameState.from_dict({"map_group": 24, "map_number": 3, "x": 3, "y": 9, "tiles": {}, "event_flags": [0] * 300,
                                  "party_count": 1, "party_species": [155], "party_hp": [[19, 19]], "party_levels": [5],
                                  "party_moves": [[33, 43, 0, 0]], "items": [], "item_counts": []})
        plain = rl_env.observe(st, set(), deep=True)
        self.assertEqual(plain.shape, (rl_env.deep_obs_dim(),))
        for f in (0.0, 0.25, 1.0):
            v = rl_env.observe(st, set(), deep=True, time_frac=f)
            self.assertEqual(v.shape, (plain.shape[0] + 1,))
            self.assertTrue(np.array_equal(v[:-1], plain))        # nothing else moves
            self.assertAlmostEqual(float(v[-1]), f)
        self.assertEqual(float(rl_env.observe(st, set(), deep=True, time_frac=7.0)[-1]), 1.0)
        self.assertEqual(float(rl_env.observe(st, set(), deep=True, time_frac=-1.0)[-1]), 0.0)

    def test_checkpoint_dims_are_not_interchangeable(self):
        import torch   # noqa: F401
        from src.deep_policy import CheckpointIncompatibleError, SpatialVisionPolicy
        from src.rl_env import N_ACTIONS, deep_obs_dim
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "v8.pt")
            SpatialVisionPolicy(obs_dim=deep_obs_dim(True), num_actions=N_ACTIONS, use_gru=False).save(path)
            with self.assertRaises(CheckpointIncompatibleError):
                SpatialVisionPolicy.load(path, override_obs_dim=deep_obs_dim(True, time_aware=True), check_schema=True)
            path9 = str(Path(tmp) / "v9.pt")
            SpatialVisionPolicy(obs_dim=deep_obs_dim(True, time_aware=True), num_actions=N_ACTIONS, use_gru=False).save(path9)
            with self.assertRaises(CheckpointIncompatibleError):
                SpatialVisionPolicy.load(path9, override_obs_dim=deep_obs_dim(True), check_schema=True)
            SpatialVisionPolicy.load(path9, override_obs_dim=deep_obs_dim(True, time_aware=True), check_schema=True)

    def test_evaluator_obs_vector_follows_the_checkpoint_dim(self):
        from src.ram_parser import GameState
        from tools import evaluate_policy as ev
        st = GameState.from_dict({"map_group": 24, "map_number": 3, "x": 3, "y": 9, "tiles": {}, "event_flags": [0] * 300,
                                  "party_count": 1, "party_species": [155], "party_hp": [[19, 19]], "party_levels": [5],
                                  "party_moves": [[33, 43, 0, 0]], "items": [], "item_counts": []})
        self.assertEqual(ev.obs_vector(st, set(), 81, time_frac=0.5).shape, (81,))
        self.assertAlmostEqual(float(ev.obs_vector(st, set(), 81, time_frac=0.5)[-1]), 0.5)
        self.assertEqual(ev.obs_vector(st, set(), 78).shape, (78,))
        self.assertEqual(ev.obs_vector(st, set(), 80).shape, (80,))
        self.assertEqual(ev.obs_vector(st, set(), 77).shape, (77,))
        with self.assertRaises(ValueError):
            ev.obs_vector(st, set(), 79)


class PresetTests(unittest.TestCase):
    def test_long_v9_differs_from_long_v8_only_in_the_v9_fields(self):
        a, b = dataclasses.asdict(preset("long_v8")), dataclasses.asdict(preset("long_v9"))
        changed = {k for k in a if a[k] != b[k]}
        self.assertEqual(changed, {"name", "path_progress_scale", "milestone_reward_scale", "milestone_speed_bonus", "explore_budget",
                                   "retreat_budget", "anti_farm", "time_feature", "entropy_boost_threshold"})
        self.assertEqual(a["ppo"], b["ppo"])        # gamma 0.999 etc. untouched
        cfg = preset("long_v9").validate()
        self.assertEqual((cfg.path_progress_scale, cfg.milestone_reward_scale, cfg.milestone_speed_bonus), (1.0, 1.5, 1.5))
        self.assertTrue(cfg.egg_errand_route and cfg.battle_action_mask and cfg.periodic_eval and cfg.terminate_on_whiteout)
        self.assertTrue(cfg.anti_farm and cfg.time_feature and cfg.level_gate_by_target and cfg.catch_new_species_only)
        self.assertEqual((cfg.wild_battle_budget, cfg.trainer_battle_budget, cfg.catch_budget, cfg.catch_reward), (6.0, 20.0, 4.0, 2.0))
        self.assertEqual((cfg.battle_free_count, cfg.battle_decay, cfg.battle_floor, cfg.wild_battle_scale), (2, 0.6, 0.1, 0.4))
        self.assertEqual(cfg.entropy_boost_threshold, 1.0)
        self.assertEqual(cfg.ppo.gamma, 0.999)

    def test_older_presets_are_untouched(self):
        for name in OLD:
            cfg = preset(name)
            self.assertFalse(cfg.anti_farm, name)
            self.assertFalse(cfg.time_feature, name)
            self.assertEqual((cfg.milestone_reward_scale, cfg.milestone_speed_bonus), (1.0, 1.0), name)
            env = __import__("tools.train_deep_ppo", fromlist=["x"]).worker_env_cfg(cfg, None)
            self.assertIsNone(env["farm"], name)
            self.assertEqual(env["budget_caps"], {}, name)
            self.assertFalse(env["time_feature"], name)
        self.assertEqual(preset("long_v8").path_progress_scale, 0.3)
        self.assertEqual(preset("long_v8").entropy_boost_threshold, 0.8)
        self.assertEqual(preset("long_v8").explore_budget, 12.0)
        self.assertEqual(rl_env.deep_obs_dim(True), 80)

    def test_json_round_trip_and_worker_cfg(self):
        from tools import train_deep_ppo as t
        cfg = preset("long_v9")
        again = from_dict(__import__("json").loads(cfg.to_json()), preset("blueprint_v1"))
        self.assertEqual(again.to_dict(), cfg.to_dict())
        env = t.worker_env_cfg(cfg, None)
        self.assertEqual(env["farm"]["wild_scale"], 0.4)
        self.assertEqual(env["budget_caps"], {"cap:battle": 6.0, "cap:trainer": 20.0, "cap:catch": 4.0})
        self.assertEqual((env["milestone_scale"], env["milestone_speed_bonus"], env["time_feature"]), (1.5, 1.5, True))
        self.assertEqual(env["path_progress"], 1.0)

    def test_budget_caps_table_gained_only_the_trainer_key(self):
        self.assertEqual(set(BUDGET_CAPS), {"battle", "trainer", "level", "catch", "heal", "explore"})
        self.assertEqual(BUDGET_CAPS["battle"], 20.0)

    def test_cloud_train_accepts_long_v9_and_the_launcher_stays_numpy_free(self):
        from tools import cloud_train_deep as c
        self.assertEqual(c.resolve_config(argparse.Namespace(config="long_v9")), ("long_v9", False))
        code = ("import sys; sys.modules['numpy'] = None; sys.modules['torch'] = None; "
                "from src.train_config import PRESETS, load_config; import src.story_path, src.milestone_stats, src.anti_farm; "
                "c = load_config('long_v9'); assert c.anti_farm and c.time_feature and c.story_profile().count == 11; print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)


@unittest.skipUnless(HAVE_ROM and (ROOT / "states" / "mystery_egg_received.state").exists(), "needs the ROM and the saved states")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(sp.set_profile, False)

    def worker(self, max_steps=100, **extra):
        from tools import train_deep_ppo as t
        cfg = t.worker_env_cfg(preset("long_v9"), None)
        cfg.update({"stagger_episodes": False, "episode_len_jitter": 0.0, "jitter_max": 0, "swarm_dir": None, "seed": 3, **extra})
        w = t.EnvWorker(0, ["mystery_egg_received"], max_steps=max_steps, cfg=cfg)
        self.addCleanup(w.close)
        return w

    def test_time_scalar_follows_the_episode_clock(self):
        from src.rl_env import ACTIONS
        w = self.worker(max_steps=40)
        _screen, ram = w.reset()
        self.assertEqual(ram.shape, (81,))
        self.assertEqual(float(ram[-1]), 0.0)
        seen = []
        for _ in range(10):
            _s, ram, _r, _d, _m, _b, info = w.step(ACTIONS.index("b"))
            seen.append(float(ram[-1]))
        self.assertTrue(all(abs(v - (i + 1) / 40.0) < 1e-9 for i, v in enumerate(seen)), seen)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in seen))

    def test_delivery_event_pays_scale_times_speed(self):
        from src.rl_env import ACTIONS
        w = self.worker(max_steps=100)
        w.reset()
        w.step(ACTIONS.index("b"))
        b = bytearray(w.emu.read(1, 0xDA72, 256))
        for f in (31, 1812):
            b[f >> 3] |= 1 << (f & 7)
        b[1813 >> 3] &= ~(1 << (1813 & 7))
        w.emu.write(1, 0xDA72, bytes(b))
        *_, info = w.step(ACTIONS.index("b"))     # step 2 of 100
        pay = info["reward_components"]["event"]
        speed = af.speed_multiplier(2, 100, 1.5)
        self.assertGreaterEqual(pay, 30.0 * 1.5 * speed - 1e-6)
        self.assertLessEqual(pay, 30.0 * 1.5 * speed + 0.05 + 1e-6)   # + at most a token unnamed-flag payment

    def test_budget_overrides_and_level_target_reach_the_episode(self):
        w = self.worker()
        w.reset()
        self.assertEqual((w.budget["cap:battle"], w.budget["cap:trainer"], w.budget["cap:catch"]), (6.0, 20.0, 4.0))
        self.assertEqual(w.budget["cap:heal"], preset("long_v9").heal_budget)
        self.assertEqual(w.budget["cap:explore"], 8.0)
        self.assertIsNotNone(w.farm)
        self.assertEqual(w.dwell.target_level(), preset("long_v9").dwell_level_targets[w.dwell.furthest])


if __name__ == "__main__":
    unittest.main()
