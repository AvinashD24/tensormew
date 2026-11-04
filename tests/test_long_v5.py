"""long_v5 anti-collapse changes: config flags, bump / loop switches, the no-effect press penalty (pure and on the real
emulator), the entropy anneal floor and boost, and a reward comparison of 'press A forever' vs 'walk the story path'.
long_v4 / long_v3 must stay exactly as they were."""

import argparse
import dataclasses
import subprocess
import sys
import unittest
from pathlib import Path

from src import entropy_guard as eg
from src import idle_press as ip
from src import route_graph as rg
from src.config import ROM_PATH
from src.entropy_guard import EntropyGuard, anneal_fraction
from src.idle_press import IdlePressTracker, press_had_effect
from src.ppo_update import EntropySchedule
from src.ram_parser import GameState, dialogue_open
from src.rl_env import ACTIONS, progress_reward
from src.train_config import PRESETS, ConfigError, TrainConfig, apply_overrides, from_dict, load_config
from tools import cloud_train_deep as c
from tools import train_deep_ppo as t

ROOT = Path(__file__).resolve().parents[1]
HAVE_ROM = Path(ROM_PATH).exists()
A, B = ACTIONS.index("a"), ACTIONS.index("b")

V5_FIELDS = ("disable_bump_penalty", "disable_loop_penalty", "idle_press_penalty", "entropy_boost")


def _gs(**kw):
    base = {"map_group": 24, "map_number": 4, "x": 3, "y": 9, "tiles": {}, "event_flags": [0] * 300,
            "party_count": 1, "party_species": [155], "party_hp": [[19, 19]], "party_levels": [5],
            "party_moves": [[33, 43, 0, 0]], "items": [], "item_counts": []}
    base.update(kw)
    return GameState.from_dict(base)


# ------------------------------------------------------------------------------------------------ presets / config
class PresetTests(unittest.TestCase):
    def test_long_v5_values(self):
        self.assertIn("long_v5", PRESETS)
        cfg = load_config("long_v5")
        self.assertEqual(cfg.name, "long_v5")
        self.assertTrue(cfg.path_progress)
        self.assertAlmostEqual(cfg.path_progress_scale, 0.3)
        self.assertTrue(cfg.disable_bump_penalty and cfg.disable_loop_penalty)
        self.assertTrue(cfg.idle_press_penalty)
        self.assertAlmostEqual(cfg.idle_press_cost, 0.015)
        self.assertAlmostEqual(cfg.idle_press_budget, 10.0)
        self.assertAlmostEqual(cfg.entropy_start, 0.02)
        self.assertAlmostEqual(cfg.entropy_end, 0.005)
        self.assertEqual(cfg.entropy_anneal_updates, 1200)
        self.assertTrue(cfg.entropy_boost)
        self.assertEqual((cfg.entropy_boost_threshold, cfg.entropy_boost_until, cfg.entropy_boost_coef),
                         (0.8, 300, 0.05))

    def test_long_v5_only_differs_from_long_v4_in_the_anti_collapse_fields(self):
        a, b = load_config("long_v4").to_dict(), load_config("long_v5").to_dict()
        diff = {k for k in a if a[k] != b[k]}
        self.assertEqual(diff, {"name", "entropy_start", "entropy_end", "path_progress_scale", "disable_bump_penalty",
                                "disable_loop_penalty", "idle_press_penalty", "entropy_anneal_updates",
                                "entropy_boost"})

    def test_older_presets_have_every_new_flag_off(self):
        for name in ("blueprint_v1", "legacy", "long_v1", "long_v2", "long_v3", "long_v4"):
            cfg = load_config(name)
            for f in V5_FIELDS:
                self.assertFalse(getattr(cfg, f), f"{name}.{f}")
            self.assertEqual(cfg.entropy_anneal_updates, 0, name)
            self.assertIsNone(cfg.idle_press_params(), name)
            self.assertIsNone(cfg.entropy_guard_params(), name)

    def test_long_v4_and_v3_values_are_unchanged(self):
        v4, v3 = load_config("long_v4"), load_config("long_v3")
        self.assertAlmostEqual(v4.path_progress_scale, 0.08)
        self.assertEqual((v4.entropy_start, v4.entropy_end, v4.ppo.gamma), (0.01, 0.001, 0.999))
        self.assertEqual((v3.entropy_start, v3.entropy_end), (0.04, 0.01))
        self.assertFalse(v3.path_progress)
        env4 = t.worker_env_cfg(v4, None, None)
        self.assertIsNone(env4["idle_press"])
        self.assertFalse(env4["disable_bump_penalty"] or env4["disable_loop_penalty"])

    def test_worker_env_cfg_for_long_v5(self):
        env = t.worker_env_cfg(load_config("long_v5"), None, None)
        self.assertEqual(env["idle_press"], {"cost": 0.015, "budget": 10.0})
        self.assertTrue(env["disable_bump_penalty"] and env["disable_loop_penalty"])
        self.assertAlmostEqual(env["path_progress"], 0.3)

    def test_each_flag_can_be_ablated(self):
        cfg = apply_overrides(load_config("long_v5"), {
            "disable_bump_penalty": False, "disable_loop_penalty": False, "idle_press_penalty": False,
            "entropy_boost": False, "entropy_anneal_updates": 0})
        env = t.worker_env_cfg(cfg, None, None)
        self.assertIsNone(env["idle_press"])
        self.assertFalse(env["disable_bump_penalty"] or env["disable_loop_penalty"])
        self.assertIsNone(cfg.entropy_guard_params())

    def test_validation(self):
        for bad in ({"idle_press_cost": -0.1}, {"idle_press_budget": -1.0}, {"entropy_anneal_updates": -1},
                    {"entropy_boost_threshold": 0.0}, {"entropy_boost_coef": -0.1}, {"entropy_boost_until": -1},
                    {"path_progress": False}):   # bump / loop switches need the path reward
            with self.assertRaises(ConfigError, msg=str(bad)):
                apply_overrides(load_config("long_v5"), bad)
        with self.assertRaises(ConfigError):
            from_dict({"disable_bump_penalty": True})   # blueprint_v1 has no path_progress

    def test_cloud_train_accepts_long_v5(self):
        self.assertEqual(c.resolve_config(argparse.Namespace(config="long_v5")), ("long_v5", False))
        args = argparse.Namespace(updates=1600, rollout=128, max_steps=1024, batch_size=256, lr=3e-4,
                                  state="new_bark_outside", save="s.pt", envs=24, curriculum=False, recurrent=True,
                                  states="", config="long_v5", explicit={"updates", "envs", "state", "curriculum"},
                                  resume=False, run_id="v5-1")
        self.assertIn("--config long_v5", c.build_remote_train_cmd(args, 24, "long_v5"))

    def test_config_chain_imports_without_numpy_or_torch(self):
        code = ("import sys; sys.modules['numpy'] = None; sys.modules['torch'] = None; "
                "from src.train_config import PRESETS, load_config; "
                "assert load_config('long_v5').idle_press_penalty; print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)


# ------------------------------------------------------------------------------------------------ bump switch
class BumpSwitchTests(unittest.TestCase):
    def test_bump_is_paid_by_default_and_dropped_with_pay_bump_false(self):
        s = _gs()
        comps = {}
        r1, _, _ = progress_reward(s, s, False, set(), {(24, 4)}, components=comps)
        self.assertLess(comps["bump"], 0.0)
        comps2 = {}
        r2, _, _ = progress_reward(s, s, False, set(), {(24, 4)}, components=comps2, pay_bump=False)
        self.assertNotIn("bump", comps2)
        self.assertAlmostEqual(r2 - r1, -comps["bump"])
        self.assertAlmostEqual(comps2["step"], -0.01)   # the step cost stays


# ------------------------------------------------------------------------------------------------ idle press (pure)
class IdlePressPureTests(unittest.TestCase):
    def test_nothing_changed_is_idle(self):
        self.assertFalse(press_had_effect(_gs(), _gs()))

    def test_any_observable_change_counts_as_an_effect(self):
        before = _gs()
        for kw in ({"script_running": 1}, {"script_mode": 1}, {"battle_mode": 1}, {"x": 4}, {"map_number": 5},
                   {"facing_raw": 4}, {"party_hp": [[18, 19]]}, {"party_levels": [6]}, {"party_count": 2},
                   {"items": [17], "item_counts": [3]}, {"balls": [4, 2]}, {"money": 5}, {"event_flags": [1] + [0] * 299}):
            self.assertTrue(press_had_effect(before, _gs(**kw)), kw)

    def test_tracker_charges_idle_presses_up_to_the_budget_and_resets(self):
        tr = IdlePressTracker(cost=1.0, budget=2.5)
        s = _gs()
        self.assertEqual([tr.update(s, s) for _ in range(4)], [-1.0, -1.0, -0.5, -0.0])
        self.assertEqual(tr.count, 4)
        self.assertEqual(tr.update(s, _gs(script_running=1)), 0.0)   # opened a dialogue: free
        tr.reset()
        self.assertEqual(tr.update(s, s), -1.0)

    def test_from_params(self):
        self.assertIsNone(IdlePressTracker.from_params(None))
        self.assertIsNone(IdlePressTracker.from_params({}))
        self.assertEqual(IdlePressTracker.from_params({"cost": 0.02, "budget": 3.0}).cost, 0.02)
        with self.assertRaises(ValueError):
            IdlePressTracker(cost=-1.0)
        self.assertEqual(ip.validate_params(0.01, 5.0), [])


# ------------------------------------------------------------------------------------------------ entropy
class EntropyTests(unittest.TestCase):
    def test_anneal_fraction_is_progress_only_when_off_and_time_capped_when_on(self):
        self.assertEqual(anneal_fraction(0.5, 10, 0), 0.5)
        self.assertEqual(anneal_fraction(2.0, 10, 0), 1.0)
        self.assertEqual(anneal_fraction(0.9, 120, 1200), 0.1)     # progress is fast, the time cap is not
        self.assertEqual(anneal_fraction(0.05, 600, 1200), 0.05)   # progress is slow: it still gates
        self.assertEqual(anneal_fraction(1.0, 5000, 1200), 1.0)

    def test_schedule_with_anneal_floor_decays_no_faster_than_the_cap(self):
        sched = EntropySchedule(0.02, 0.005)
        self.assertAlmostEqual(sched.value(anneal_fraction(1.0, 0, 1200)), 0.02)
        self.assertAlmostEqual(sched.value(anneal_fraction(1.0, 600, 1200)), 0.0125)
        self.assertAlmostEqual(sched.value(anneal_fraction(1.0, 1200, 1200)), 0.005)

    def test_guard_boosts_below_the_threshold_and_releases_with_hysteresis(self):
        g = EntropyGuard(threshold=0.8, until_update=300, boost_coef=0.05, margin=0.2)
        self.assertEqual(g.coef(0, 0.02, None), (0.02, False))          # first update: no entropy yet
        self.assertEqual(g.coef(10, 0.02, 1.5), (0.02, False))
        self.assertEqual(g.coef(11, 0.02, 0.79), (0.05, True))          # collapse detected
        self.assertEqual(g.coef(12, 0.02, 0.9), (0.05, True))           # still inside the hysteresis band
        self.assertEqual(g.coef(13, 0.02, 1.0), (0.02, False))          # recovered to threshold + margin
        self.assertEqual(g.triggers, 1)
        self.assertEqual(g.coef(14, 0.02, 0.5), (0.05, True))           # collapses again
        self.assertEqual(g.triggers, 2)

    def test_guard_never_lowers_the_scheduled_value_and_stops_after_until_update(self):
        g = EntropyGuard(threshold=0.8, until_update=300, boost_coef=0.05)
        self.assertEqual(g.coef(5, 0.08, 0.1), (0.08, True))            # scheduled value already above the boost
        self.assertEqual(g.coef(300, 0.01, 0.1), (0.01, False))         # the window is over
        self.assertFalse(g.active)

    def test_guard_params_follow_the_config(self):
        self.assertEqual(load_config("long_v5").entropy_guard_params(),
                         {"threshold": 0.8, "until_update": 300, "boost_coef": 0.05, "margin": 0.2})
        self.assertEqual(eg.validate_params(0.8, 300, 0.05, 0.2), [])
        with self.assertRaises(ValueError):
            EntropyGuard(threshold=-1.0)


# ------------------------------------------------------------------------------------------------ real emulator
@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class EnvWorkerIdlePressTests(unittest.TestCase):
    IDLE = {"cost": 0.015, "budget": 10.0}

    def make(self, state="new_bark_outside", **cfg):
        w = t.EnvWorker(0, [state], max_steps=500, cfg=cfg)
        self.addCleanup(w.close)
        w.reset()
        return w

    def test_a_and_b_in_the_open_overworld_with_nothing_happening_are_penalized(self):
        w = self.make(idle_press=self.IDLE)
        for action in (B, A, B):
            *_, rew, _, _, _, info = w.step(action)
            self.assertAlmostEqual(info["reward_components"]["idle_press"], -0.015, msg=ACTIONS[action])
        self.assertEqual(w.idle_press.count, 3)

    def test_budget_caps_the_total_and_a_new_episode_starts_fresh(self):
        w = self.make(idle_press={"cost": 0.5, "budget": 1.0})
        paid = [w.step(B)[-1]["reward_components"]["idle_press"] for _ in range(4)]
        self.assertEqual(paid, [-0.5, -0.5, 0.0, 0.0])
        w.reset()
        self.assertEqual(w.step(B)[-1]["reward_components"]["idle_press"], -0.5)

    def test_movement_is_never_an_idle_press(self):
        w = self.make(idle_press=self.IDLE)
        for _ in range(6):
            *_, info = w.step(ACTIONS.index("down"))
            self.assertEqual(info["reward_components"]["idle_press"], 0.0)

    def test_a_that_opens_a_dialogue_and_the_dialogue_steps_are_free(self):
        w = self.make("ilex_src", idle_press=self.IDLE)
        *_, info = w.step(A)    # faces an NPC / sign: opens text
        self.assertTrue(dialogue_open(w.reader.parse(w.emu.frame)))
        self.assertEqual(info["reward_components"]["idle_press"], 0.0)
        *_, info = w.step(A)    # the open text is auto-advanced, not a plain press
        self.assertEqual(info["reward_components"]["idle_press"], 0.0)

    def test_presses_in_battle_are_never_charged(self):
        w = self.make(idle_press=self.IDLE)
        real_parse = w.reader.parse
        calls = {"n": 0}

        def parse(frame):
            st = real_parse(frame)
            calls["n"] += 1
            return dataclasses.replace(st, battle_mode=1) if calls["n"] == 1 else st   # only the step's 'before'

        w.reader.parse = parse
        *_, info = w.step(A)
        self.assertEqual(info["reward_components"]["idle_press"], 0.0)
        self.assertEqual(w.idle_press.count, 0)

    def test_flag_off_has_no_component_and_no_cost(self):
        w = self.make()
        *_, info = w.step(B)
        self.assertNotIn("idle_press", info["reward_components"])
        self.assertIsNone(w.idle_press)

    def test_loop_and_bump_switches(self):
        for flags, loop_expected, bump_expected in (
            ({}, True, True),
            ({"path_progress": 0.3}, True, True),
            ({"path_progress": 0.3, "disable_loop_penalty": True, "disable_bump_penalty": True}, False, False),
        ):
            w = self.make(**flags)
            st = w.reader.parse(w.emu.frame)
            w.tile_visit_counts[(st.map_group, st.map_number, st.x, st.y)] = 60   # a heavily revisited tile
            *_, info = w.step(B)
            self.assertEqual(info["reward_components"]["loop"] < 0.0, loop_expected, flags)
            # a wall bump: keep walking into the same direction until the position stops changing
            bumped = False
            for name in ("up", "left", "down", "right"):
                for _ in range(6):
                    *_, info = w.step(ACTIONS.index(name))
                    if info["reward_components"].get("bump", 0.0) < 0.0:
                        bumped = True
                if bumped:
                    break
            if bump_expected:
                self.assertTrue(bumped, flags)
            else:
                self.assertFalse(any("bump" in str(k) for k in info["reward_components"]), flags)
                self.assertFalse(bumped, flags)


@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class SpamVersusWalkTests(unittest.TestCase):
    """Forward walking must clearly beat pressing A in place under long_v5 (the v4-1 collapse)."""
    N = 150

    def total(self, policy):
        import numpy as np  # noqa: F401  (the worker needs it anyway)
        g = rg._graph()
        d = g.dist[0]
        cfg = load_config("long_v5")
        env_cfg = t.worker_env_cfg(cfg, None, None)
        env_cfg.update(sticky_prob=0.0, jitter_max=0, stagger_episodes=False, episode_len_jitter=0.0,
                       battle_dir=None, battle_prob=0.0, swarm_dir=None)
        w = t.EnvWorker(0, ["new_bark_outside"], max_steps=10_000, cfg=env_cfg)
        self.addCleanup(w.close)
        w.reset()
        total = 0.0
        for _ in range(self.N):
            action = A
            if policy == "walk":
                s = w.reader.parse(w.emu.frame)
                n = ((s.map_group, s.map_number), s.x, s.y)
                if not s.battle_mode and n in d and g.fwd.get(n):
                    for i, o in enumerate(g.fwd[n]):
                        if o is not None and d.get(o, 1e9) == d[n] - 1:
                            action = i
                            break
            total += w.step(action)[2]
        return total

    def test_walking_beats_spamming_a(self):
        spam, walk = self.total("spam"), self.total("walk")
        self.assertLess(spam, 0.0)
        self.assertGreater(walk, spam + 20.0, (spam, walk))


if __name__ == "__main__":
    unittest.main()
