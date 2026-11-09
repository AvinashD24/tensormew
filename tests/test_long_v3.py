"""long_v3: staggered episodes, battle archive + early A-bias in battle menus, furthest-map weighted archive resets."""

import argparse
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.config import ROM_PATH
from src.deep_policy import ACTION_A_INDEX, BATTLE_FLAG_INDEX, SpatialVisionPolicy
from src.ram_parser import GameState
from src.rl_env import ACTIONS, DEEP_OBS_DIM, N_ACTIONS, observe
from src.story_path import MILESTONE_COUNT
from src.train_config import ConfigError, PRESETS, apply_overrides, load_config
from tools import cloud_train_deep as c
from tools import train_deep_ppo as t

HAVE_ROM = Path(ROM_PATH).exists()


def _gs(**kw):
    base = {"map_group": 24, "map_number": 4, "x": 3, "y": 9, "tiles": {}, "event_flags": [0] * 300,
            "party_count": 1, "party_species": [155], "party_hp": [[19, 19]], "party_levels": [5],
            "party_moves": [[33, 43, 0, 0]]}
    base.update(kw)
    return GameState.from_dict(base)


def small_policy(gru=True):
    torch.manual_seed(0)
    p = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=N_ACTIONS, ram_hidden_dim=16, fusion_dim=32,
                            use_gru=gru, gru_hidden_dim=16)
    for m in p.actor.parameters():  # a non-trivial actor so the bias effect is visible
        torch.nn.init.normal_(m, std=0.3)
    return p.eval()


# ------------------------------------------------------------------------------------------------ presets
class PresetTests(unittest.TestCase):
    def test_long_v3_loads_with_all_three_features(self):
        self.assertIn("long_v3", PRESETS)
        cfg = load_config("long_v3")
        self.assertEqual(cfg.name, "long_v3")
        self.assertTrue(cfg.stagger_episodes)
        self.assertAlmostEqual(cfg.episode_len_jitter, 0.25)
        self.assertTrue(cfg.battle_archive)
        self.assertAlmostEqual(cfg.battle_archive_prob, 0.15)
        self.assertEqual(cfg.battle_archive_max, 32)
        self.assertAlmostEqual(cfg.battle_a_bias, 0.5)
        self.assertEqual(cfg.battle_a_bias_updates, 150)
        self.assertAlmostEqual(cfg.swarm_prob, 0.5)
        self.assertTrue(cfg.swarm_furthest_weighting)
        self.assertEqual(cfg.start_states, ["new_bark_outside"])
        self.assertFalse(cfg.goal_potential)

    def test_long_v2_is_unchanged(self):
        v2, v3 = load_config("long_v2"), load_config("long_v3")
        self.assertFalse(v2.stagger_episodes or v2.battle_archive or v2.swarm_furthest_weighting)
        self.assertEqual((v2.battle_a_bias, v2.episode_len_jitter, v2.swarm_prob), (0.0, 0.0, 0.3))
        for key in ("envs", "rollout_steps", "updates", "max_steps", "lr", "lr_end_frac", "entropy_start",
                    "entropy_end", "sticky_prob", "jitter_max"):
            self.assertEqual(getattr(v2, key), getattr(v3, key), key)
        self.assertEqual(v2.ppo, v3.ppo)

    def test_each_feature_can_be_ablated(self):
        cfg = apply_overrides(load_config("long_v3"), {"stagger_episodes": False, "battle_archive": False,
                                                       "battle_a_bias": 0.0, "swarm_furthest_weighting": False,
                                                       "swarm_prob": 0.3})
        env = t.worker_env_cfg(cfg, "logs/swarm/r", "logs/swarm/r/battle")
        self.assertFalse(env["stagger_episodes"])
        self.assertIsNone(env["battle_dir"])
        self.assertEqual(env["battle_prob"], 0.0)
        self.assertIsNone(env["swarm_furthest_floor"])
        self.assertEqual(env["swarm_prob"], 0.3)

    def test_worker_env_cfg_for_long_v3(self):
        cfg = load_config("long_v3")
        env = t.worker_env_cfg(cfg, "logs/swarm/r", "logs/swarm/r/battle")
        self.assertTrue(env["stagger_episodes"])
        self.assertEqual(env["num_envs"], cfg.envs)
        self.assertEqual(env["battle_dir"], "logs/swarm/r/battle")
        self.assertAlmostEqual(env["battle_prob"], 0.15)
        self.assertAlmostEqual(env["swarm_furthest_floor"], 0.15)
        self.assertAlmostEqual(env["swarm_prob"], 0.5)

    def test_validation(self):
        for bad in ({"swarm_prob": 1.5}, {"episode_len_jitter": 1.0}, {"battle_archive_prob": -0.1},
                    {"battle_a_bias": -1.0}, {"battle_archive_max": 0}, {"swarm_weight_floor": 0.0},
                    {"swarm_prob": 0.9, "battle_archive_prob": 0.3}):
            with self.assertRaises(ConfigError, msg=str(bad)):
                apply_overrides(load_config("long_v3"), bad)

    def test_cloud_train_accepts_long_v3(self):
        self.assertEqual(c.resolve_config(argparse.Namespace(config="long_v3")), ("long_v3", False))
        args = argparse.Namespace(updates=1600, rollout=128, max_steps=1024, batch_size=256, lr=3e-4,
                                  state="new_bark_outside", save="s.pt", envs=24, curriculum=False, recurrent=True,
                                  states="", config="long_v3", explicit={"updates", "envs", "state", "curriculum"},
                                  resume=False, run_id="v3")
        cmd = c.build_remote_train_cmd(args, 24, "long_v3")
        self.assertIn("--config long_v3", cmd)
        self.assertNotIn("--resume", cmd)


# ------------------------------------------------------------------------------------------------ A-bias
class BattleABiasTests(unittest.TestCase):
    def test_constants_match_the_observation_and_action_layout(self):
        self.assertEqual(ACTION_A_INDEX, ACTIONS.index("a"))
        outside = observe(_gs(), set(), deep=True)
        in_battle = observe(_gs(battle_mode=1), set(), deep=True)
        self.assertEqual(outside[BATTLE_FLAG_INDEX], 0.0)
        self.assertEqual(in_battle[BATTLE_FLAG_INDEX], 1.0)
        self.assertEqual(np.nonzero(outside != in_battle)[0].tolist()[0], BATTLE_FLAG_INDEX)

    def test_schedule_decays_linearly_to_zero(self):
        f = t.battle_a_bias_value
        self.assertAlmostEqual(f(0, 0.5, 150), 0.5)
        self.assertAlmostEqual(f(75, 0.5, 150), 0.25)
        self.assertEqual(f(150, 0.5, 150), 0.0)
        self.assertEqual(f(1000, 0.5, 150), 0.0)
        self.assertEqual(f(0, 0.0, 150), 0.0)
        self.assertEqual(f(0, 0.5, 0), 0.0)

    def test_bias_only_moves_the_a_logit_inside_battle(self):
        policy = small_policy()
        screen = torch.rand(2, 3, 32, 32)
        rams = torch.zeros(2, DEEP_OBS_DIM)
        rams[1, BATTLE_FLAG_INDEX] = 1.0  # sample 1 is in a battle
        base, _ = policy(screen, rams)
        policy.set_battle_a_bias(0.7)
        biased, _ = policy(screen, rams)
        torch.testing.assert_close(biased[0], base[0])  # overworld: untouched
        delta = (biased[1] - base[1]).detach()
        self.assertAlmostEqual(float(delta[ACTION_A_INDEX]), 0.7, places=5)
        delta[ACTION_A_INDEX] = 0
        self.assertEqual(float(delta.abs().max()), 0.0)  # every other logit unchanged

    def test_act_and_evaluate_log_probs_match_with_bias_on(self):
        for gru in (False, True):
            policy = small_policy(gru)
            policy.set_battle_a_bias(0.9)
            g = torch.Generator().manual_seed(3)
            screen = torch.rand(144, 160, 3, generator=g).mul(255).to(torch.uint8).numpy()
            ram = np.zeros(DEEP_OBS_DIM, dtype=np.float32)
            ram[BATTLE_FLAG_INDEX] = 1.0
            ram[10:20] = np.random.default_rng(0).random(10)
            for _ in range(5):
                action, logp, _ = policy.act(screen, ram)
                s_t = torch.from_numpy(screen).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                r_t = torch.from_numpy(ram).unsqueeze(0)
                lp, _, _ = policy.evaluate_actions(s_t, r_t, torch.tensor([action]))
                self.assertAlmostEqual(float(lp.detach()), logp, places=5)
            # and the probabilities act() samples from are the biased ones
            probs = policy.probs(screen, ram)
            policy.set_battle_a_bias(0.0)
            plain = policy.probs(screen, ram)
            self.assertGreater(probs[ACTION_A_INDEX], plain[ACTION_A_INDEX])

    def test_rollout_forward_and_sequence_evaluation_agree(self):
        """The trainer samples with forward_recurrent and PPO re-evaluates with evaluate_actions_sequence."""
        policy = small_policy(True)
        policy.set_battle_a_bias(0.6)
        g = torch.Generator().manual_seed(1)
        B, T = 2, 5
        screens = torch.rand(B, T, 3, 32, 32, generator=g)
        rams = torch.zeros(B, T, DEEP_OBS_DIM)
        rams[:, :, BATTLE_FLAG_INDEX] = (torch.rand(B, T, generator=g) > 0.4).float()
        hx = torch.zeros(1, B, 16)
        cur, old_lp, acts = hx, [], []
        for step in range(T):
            logits, _, cur = policy.forward_recurrent(screens[:, step], rams[:, step], cur)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            acts.append(a)
            old_lp.append(dist.log_prob(a))
        new_lp, _, _ = policy.evaluate_actions_sequence(screens, rams, torch.stack(acts, dim=1), hx)
        torch.testing.assert_close(new_lp, torch.stack(old_lp, dim=1), atol=1e-5, rtol=1e-4)
        # the bias is not a no-op: dropping it changes the log-probs of in-battle A presses
        policy.set_battle_a_bias(0.0)
        plain_lp, _, _ = policy.evaluate_actions_sequence(screens, rams, torch.stack(acts, dim=1), hx)
        self.assertFalse(torch.allclose(plain_lp, new_lp, atol=1e-4))

    def test_bias_is_not_in_the_state_dict_and_evaluation_loads_with_zero(self):
        torch.manual_seed(0)
        policy = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=N_ACTIONS, use_gru=True)
        policy.set_battle_a_bias(0.4)
        self.assertNotIn("battle_a_bias", policy.state_dict())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.pt"
            policy.save(path)
            self.assertNotIn("battle_a_bias", torch.load(str(path)))  # checkpoint format untouched
            self.assertEqual(SpatialVisionPolicy.load(path).battle_a_bias, 0.0)  # evaluation: off
            optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
            t.save_checkpoint(policy, optimizer, path, 3, extra={"battle_a_bias": policy.battle_a_bias})
            self.assertAlmostEqual(torch.load(str(t.checkpoint_sidecar(path)))["battle_a_bias"], 0.4)


# ------------------------------------------------------------------------------------------------ archives
class BattleArchiveTests(unittest.TestCase):
    def test_save_fifo_never_overwrites_and_stays_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            a = t.SwarmArchive(Path(d) / "battle", max_files=3)
            self.assertTrue(a.save_fifo("m1_1-1_s1", b"1"))
            self.assertFalse(a.save_fifo("m1_1-1_s1", b"other"))  # same key: kept as is
            self.assertEqual(a.load("m1_1-1_s1"), b"1")
            import os, time
            base = time.time() - 1000

            def age(i):  # give state i a strictly increasing mtime in the past, so new saves are always the newest
                os.utime(a.dir / f"m1_1-1_s{i}.state", (base + i, base + i))

            age(1)
            for i in range(2, 7):
                self.assertTrue(a.save_fifo(f"m1_1-1_s{i}", bytes([i])))
                age(i)
            self.assertEqual(len(a.keys()), 3)
            self.assertEqual(a.keys(), ["m1_1-1_s4", "m1_1-1_s5", "m1_1-1_s6"])  # the oldest were evicted
            self.assertEqual([p.name for p in a.dir.glob(".*tmp")], [])

    def test_key_is_kind_map_species(self):
        self.assertEqual(t.battle_archive_key(_gs(battle_mode=1, map_group=24, map_number=3, enemy_mon_species=16)),
                         "m1_24-3_s16")
        self.assertEqual(t.battle_archive_key(_gs(battle_mode=2, map_group=10, map_number=7, enemy_mon_species=0)),
                         "m2_10-7_s0")

    def _stub(self, directory, max_files=32):
        emu = mock.Mock()
        emu.save_state.return_value = b"state"
        return types.SimpleNamespace(battle_archive=t.SwarmArchive(directory, max_files), _battle_wait=None,
                                     _total_steps=0, _last_battle_save=-t.BATTLE_SAVE_GAP, emu=emu)

    def test_saves_when_the_battle_flag_goes_from_zero_to_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            w = self._stub(d)
            save = lambda before, after: t.EnvWorker._maybe_archive_battle(w, before, after)  # noqa: E731
            walking = _gs(battle_mode=0)
            fight = _gs(battle_mode=1, map_group=24, map_number=3, enemy_mon_species=16)
            w._total_steps = 100
            save(walking, walking)
            self.assertEqual(w.battle_archive.keys(), [])  # no battle: nothing
            save(walking, fight)  # flag 0 -> 1
            self.assertEqual(w.battle_archive.keys(), ["m1_24-3_s16"])
            w._total_steps = 101
            save(fight, fight)  # still the same battle: no second save
            self.assertEqual(w.emu.save_state.call_count, 1)

    def test_waits_for_the_species_and_drops_the_save_if_the_battle_ends(self):
        with tempfile.TemporaryDirectory() as d:
            w = self._stub(d)
            save = lambda before, after: t.EnvWorker._maybe_archive_battle(w, before, after)  # noqa: E731
            w._total_steps = 100
            walking, no_species = _gs(), _gs(battle_mode=1, enemy_mon_species=0)
            save(walking, no_species)
            self.assertEqual(w.battle_archive.keys(), [])  # species unknown yet
            save(no_species, _gs(battle_mode=1, enemy_mon_species=19))
            self.assertEqual(w.battle_archive.keys(), ["m1_24-4_s19"])
            w2 = self._stub(d + "/b2")
            w2._total_steps = 100
            t.EnvWorker._maybe_archive_battle(w2, walking, no_species)
            t.EnvWorker._maybe_archive_battle(w2, no_species, walking)  # fled before the species appeared
            self.assertIsNone(w2._battle_wait)
            self.assertEqual(w2.battle_archive.keys(), [])

    def test_rate_limit_and_diversity(self):
        with tempfile.TemporaryDirectory() as d:
            w = self._stub(d)
            walking = _gs()
            w._total_steps = 100
            t.EnvWorker._maybe_archive_battle(w, walking, _gs(battle_mode=1, enemy_mon_species=16))
            w._total_steps = 100 + t.BATTLE_SAVE_GAP - 1  # too soon after the last save
            t.EnvWorker._maybe_archive_battle(w, walking, _gs(battle_mode=1, enemy_mon_species=17))
            self.assertEqual(w.battle_archive.keys(), ["m1_24-4_s16"])
            w._total_steps = 100 + 2 * t.BATTLE_SAVE_GAP  # later: a new species is stored ...
            t.EnvWorker._maybe_archive_battle(w, walking, _gs(battle_mode=1, enemy_mon_species=17))
            self.assertEqual(len(w.battle_archive.keys()), 2)
            calls = w.emu.save_state.call_count
            w._total_steps += 2 * t.BATTLE_SAVE_GAP  # ... a known combination is not saved again
            t.EnvWorker._maybe_archive_battle(w, walking, _gs(battle_mode=1, enemy_mon_species=17))
            self.assertEqual(w.emu.save_state.call_count, calls)

    def test_disabled_without_an_archive(self):
        w = types.SimpleNamespace(battle_archive=None)
        t.EnvWorker._maybe_archive_battle(w, _gs(), _gs(battle_mode=1))  # must not raise


class WeightedSamplingTests(unittest.TestCase):
    def test_weights_increase_with_stage_and_keep_a_floor(self):
        w = t.furthest_weights([1, 3, 5, 7], 0.15)
        self.assertEqual(w[-1], 1.0)
        self.assertEqual(w, sorted(w))
        self.assertGreaterEqual(min(w), 0.15)
        self.assertEqual(t.furthest_weights([0, 0], 0.15), [1.0, 1.0])
        self.assertEqual(t.furthest_weights([], 0.15), [])

    def test_key_stage(self):
        self.assertEqual(t.archive_key_stage("stage4_2"), 4)
        self.assertEqual(t.archive_key_stage("event1234"), MILESTONE_COUNT)
        self.assertEqual(t.archive_key_stage("weird"), 0)

    def test_sampling_favors_the_furthest_stage_but_covers_all(self):
        with tempfile.TemporaryDirectory() as d:
            a = t.SwarmArchive(d)
            for s in (1, 3, 7):
                for slot in range(2):
                    a.save(f"stage{s}_{slot}", b"x")
            rng = np.random.default_rng(0)
            n = 6000
            counts = {1: 0, 3: 0, 7: 0}
            for _ in range(n):
                counts[t.archive_key_stage(a.sample(rng, 0.15))] += 1
            w = t.furthest_weights([1, 3, 7], 0.15)
            for stage, wt in zip((1, 3, 7), w):
                self.assertAlmostEqual(counts[stage] / n, wt / sum(w), delta=0.03)
            self.assertGreater(counts[7], counts[3])
            self.assertGreater(counts[3], counts[1])
            self.assertGreater(counts[1], 0)  # the floor keeps early stages in the mix
            uniform = [t.archive_key_stage(a.sample(np.random.default_rng(i))) for i in range(300)]
            self.assertGreater(uniform.count(1), 60)  # without weighting the stages are (roughly) even

    def test_selector_shares(self):
        with tempfile.TemporaryDirectory() as d:
            swarm, battle = t.SwarmArchive(Path(d) / "s"), t.SwarmArchive(Path(d) / "b")
            swarm.save("stage2_0", b"x")
            battle.save_fifo("m1_24-3_s16", b"y")
            sel = t.StartStateSelector(["a"], 0, archive=swarm, swarm_prob=0.5, rng=np.random.default_rng(1),
                                       battle_archive=battle, battle_prob=0.15, furthest_floor=0.15)
            sel.pick()
            n = 4000
            picks = [sel.pick() for _ in range(n)]
            share = lambda prefix: sum(p.startswith(prefix) for p in picks) / n  # noqa: E731
            self.assertAlmostEqual(share("battle:"), 0.15, delta=0.03)
            self.assertAlmostEqual(share("swarm:"), 0.5, delta=0.03)
            self.assertAlmostEqual(share("a"), 0.35, delta=0.03)

    def test_empty_battle_archive_falls_back_to_start_state(self):
        with tempfile.TemporaryDirectory() as d:
            sel = t.StartStateSelector(["a"], 0, battle_archive=t.SwarmArchive(d), battle_prob=1.0)
            self.assertEqual([sel.pick() for _ in range(4)], ["a"] * 4)

    def test_legacy_selector_draws_are_unchanged(self):
        """Without a battle archive the selector consumes exactly one draw per reset, like before."""
        with tempfile.TemporaryDirectory() as d:
            arch = t.SwarmArchive(d)
            arch.save("stage2_0", b"x")
            rng = np.random.default_rng(5)
            sel = t.StartStateSelector(["a"], 0, archive=arch, swarm_prob=0.3, rng=rng)
            sel.pick()
            got = [sel.pick() for _ in range(20)]
            ref = np.random.default_rng(5)
            expect = []
            for _ in range(20):
                if ref.random() < 0.3:
                    ref.integers(0, 1)  # group
                    ref.integers(0, 1)  # inside the group
                    expect.append("swarm:stage2_0")
                else:
                    expect.append("a")
            self.assertEqual(got, expect)


# ------------------------------------------------------------------------------------------------ staggering
class StaggerLogicTests(unittest.TestCase):
    """Pure-Python simulation of the reset schedule using the real offset / length helpers (no emulator)."""

    def _worker(self, index, num_envs, max_steps, jitter, seed=0):
        w = types.SimpleNamespace(index=index, num_envs=num_envs, max_steps=max_steps, len_jitter=jitter,
                                  rng=np.random.default_rng([seed, index]))
        w.ep_len = t.EnvWorker._draw_ep_len(w)
        w.offset = t.EnvWorker._initial_offset(w)
        return w

    def _resets_per_update(self, workers, updates, rollout):
        counts = np.zeros(updates, dtype=int)
        for w in workers:
            steps, ep_len = w.offset, w.ep_len
            for step in range(updates * rollout):
                steps += 1
                if steps >= ep_len:
                    counts[step // rollout] += 1
                    steps, ep_len = 0, t.EnvWorker._draw_ep_len(w)
        return counts

    def test_without_staggering_all_resets_land_in_the_same_update(self):
        ws = [self._worker(i, 24, 2048, 0.0) for i in range(24)]
        for w in ws:
            w.offset = 0
        counts = self._resets_per_update(ws, 16, 256)
        self.assertEqual(np.nonzero(counts)[0].tolist(), [7, 15])
        self.assertEqual(counts[7], 24)

    def test_staggered_resets_spread_over_every_update(self):
        ws = [self._worker(i, 24, 2048, 0.25) for i in range(24)]
        counts = self._resets_per_update(ws, 24, 256)
        self.assertGreaterEqual(int((counts > 0).sum()), 20)  # almost every update has resets
        self.assertLessEqual(int(counts.max()), 8)  # never the old "all 24 at once"
        first = self._resets_per_update([self._worker(i, 24, 2048, 0.0) for i in range(24)], 8, 256)
        self.assertEqual(int((first > 0).sum()), 8)  # even without jitter the first cycle covers all 8 updates
        self.assertEqual(int(first.max()), 3)

    def test_offsets_use_evenly_spaced_slots_and_stay_in_range(self):
        for i in range(24):
            w = self._worker(i, 24, 2048, 0.25)
            self.assertTrue(0 <= w.offset < w.ep_len)
            self.assertTrue(i * w.ep_len // 24 <= w.offset <= (i + 1) * w.ep_len // 24)
            self.assertTrue(int(2048 * 0.75) <= w.ep_len <= int(2048 * 1.25) + 1)

    def test_phase_spread_metric(self):
        self.assertEqual(t.phase_spread([100] * 24, [2048] * 24), 0.0)  # lockstep
        spread = t.phase_spread([i * 2048 // 24 for i in range(24)], [2048] * 24)
        self.assertAlmostEqual(spread, 0.289, delta=0.01)  # uniformly spread phases
        self.assertEqual(t.phase_spread([], []), 0.0)


@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class StaggerEmulatorTests(unittest.TestCase):
    STATE = "new_bark_outside"

    def make(self, index=0, max_steps=12, **cfg):
        w = t.EnvWorker(index, [self.STATE], max_steps=max_steps, cfg=cfg)
        self.addCleanup(w.close)
        return w

    def test_default_has_no_offset_and_exact_length(self):
        w = self.make(max_steps=5)
        w.reset()
        self.assertEqual((w.steps, w.ep_len), (0, 5))
        *_, info = w.step(4)
        self.assertEqual((info["ep_steps"], info["ep_len"], info["partial_episode"]), (1, 5, False))

    def test_first_episode_starts_at_an_offset_then_full_length_episodes(self):
        w = self.make(index=2, max_steps=12, stagger_episodes=True, num_envs=4)
        w.reset()
        self.assertTrue(6 <= w.steps < 9)  # slot 2 of 4 over 12 steps
        remaining = w.ep_len - w.steps
        ends = []
        for n in range(remaining + 12 + 1):
            screen, ram, rew, done, _, _, info = w.step(4)
            if done:
                ends.append((n + 1, info))
        self.assertEqual(ends[0][0], remaining)
        first = ends[0][1]
        self.assertTrue(first["truncated"] and first["partial_episode"])
        self.assertIn("final_obs", first)  # truncation bootstrapping still gets the final observation
        self.assertEqual(ends[1][0] - ends[0][0], 12)  # the next episode is a full one
        self.assertFalse(ends[1][1]["partial_episode"])

    def test_first_episode_is_not_otherwise_changed(self):
        """Budgets, seen sets and the start state are fresh: only the step counter is advanced."""
        w = self.make(index=1, max_steps=12, stagger_episodes=True, num_envs=2)
        w.reset()
        st = w.reader.parse(w.emu.frame)
        self.assertEqual(w.budget, {})
        self.assertEqual(w.seen_maps, {(st.map_group, st.map_number)})
        self.assertEqual(len(w.seen_tiles), 1)

    def test_episode_length_jitter(self):
        w = self.make(max_steps=100, episode_len_jitter=0.25)
        lens = set()
        for _ in range(6):
            w.reset()
            lens.add(w.ep_len)
            self.assertTrue(75 <= w.ep_len <= 125)
            self.assertEqual(w.steps, 0)  # jitter alone never offsets the counter
        self.assertGreater(len(lens), 1)

    def test_battle_state_is_loaded_on_reset(self):
        with tempfile.TemporaryDirectory() as d:
            battle = Path(d) / "battle"
            w = self.make(battle_dir=str(battle), battle_prob=1.0, battle_max=4)
            w.reset()  # first episode: the normal start state
            self.assertTrue(w.battle_archive.save_fifo("m1_24-4_s16", w.emu.save_state()))
            w.reset()
            self.assertEqual(w.state_name, "battle:m1_24-4_s16")
            *_, info = w.step(4)
            self.assertTrue(info["state"].startswith("battle:"))


if __name__ == "__main__":
    unittest.main()
