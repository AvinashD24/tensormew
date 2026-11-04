"""Trainer integration: sticky actions, jitter, start-state sampling, event seeding, truncation plumbing, config
handling, the ``dones`` fix in the recurrent sequence evaluation, and optimizer-state sidecars."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.config import ROM_PATH
from src.deep_policy import SpatialVisionPolicy
from src.ppo_update import PPOHParams, RolloutBatch, compute_gae, ppo_update
from src.rl_env import DEEP_OBS_DIM, initial_seen_events
from tools import train_deep_ppo as t

HAVE_ROM = Path(ROM_PATH).exists()


class FixedRng:
    """rng stub: ``random()`` returns the queued values in turn."""

    def __init__(self, *values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0) if self.values else 0.5


class StickyActionTests(unittest.TestCase):
    def test_off_never_repeats(self):
        rng = np.random.default_rng(0)
        for last in (None, 0, 3):
            self.assertEqual(t.pick_executed_action(2, last, 0.0, rng), 2)

    def test_repeats_last_move_when_drawn(self):
        self.assertEqual(t.pick_executed_action(1, 3, 0.1, FixedRng(0.05)), 3)  # 0.05 < 0.1 -> sticky
        self.assertEqual(t.pick_executed_action(1, 3, 0.1, FixedRng(0.5)), 1)  # not drawn

    def test_no_previous_move_means_no_repeat(self):
        self.assertEqual(t.pick_executed_action(4, None, 0.99, FixedRng(0.0)), 4)

    def test_rate_matches_probability(self):
        rng = np.random.default_rng(1)
        n = 4000
        sticky = sum(t.pick_executed_action(1, 0, 0.1, rng) == 0 for _ in range(n))
        self.assertAlmostEqual(sticky / n, 0.1, delta=0.03)


class SwarmArchiveTests(unittest.TestCase):
    def test_save_load_sample_no_overwrite_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            a = t.SwarmArchive(Path(d) / "run", max_files=2)
            self.assertEqual(a.keys(), [])
            self.assertIsNone(a.sample(np.random.default_rng(0)))
            self.assertTrue(a.save("stage1", b"one"))
            self.assertFalse(a.save("stage1", b"other"))  # never overwritten
            self.assertEqual(a.load("stage1"), b"one")
            self.assertTrue(a.save("stage2", b"two"))
            self.assertFalse(a.save("stage3", b"three"))  # full
            self.assertEqual(a.keys(), ["stage1", "stage2"])
            self.assertIn(a.sample(np.random.default_rng(0)), ("stage1", "stage2"))
            self.assertIsNone(a.load("missing"))
            self.assertEqual([p.name for p in Path(d, "run").glob(".*tmp")], [])  # no temp files left behind


class StartStateSelectorTests(unittest.TestCase):
    def test_plain_cycle_is_offset_by_env_index(self):
        a = t.StartStateSelector(["a", "b", "c"], env_index=0)
        b = t.StartStateSelector(["a", "b", "c"], env_index=1)
        self.assertEqual([a.pick() for _ in range(4)], ["a", "b", "c", "a"])
        self.assertEqual([b.pick() for _ in range(4)], ["b", "c", "a", "b"])

    def test_stage_balanced_samples_stages_evenly(self):
        # three states in stage 0 and one in stage 1: a plain cycle would give stage 1 a quarter of the episodes
        stages = {"s0a": 0, "s0b": 0, "s0c": 0, "s1": 1}
        sel = t.StartStateSelector(list(stages), env_index=0, stages=stages)
        picks = [sel.pick() for _ in range(40)]
        per_stage = [sum(stages[p] == s for p in picks) for s in (0, 1)]
        self.assertEqual(per_stage, [20, 20])
        self.assertEqual({p for p in picks if stages[p] == 0}, {"s0a", "s0b", "s0c"})

    def test_stage_balanced_envs_start_on_different_stages(self):
        stages = {"x": 0, "y": 1, "z": 2, "w": 3}
        first = [t.StartStateSelector(list(stages), env_index=i, stages=stages).pick() for i in range(8)]
        self.assertEqual({stages[s] for s in first}, {0, 1, 2, 3})

    def test_swarm_archive_used_only_on_resets_with_probability(self):
        with tempfile.TemporaryDirectory() as d:
            arch = t.SwarmArchive(d)
            arch.save("stage2", b"x")
            sel = t.StartStateSelector(["a", "b"], 0, archive=arch, swarm_prob=1.0)
            self.assertEqual(sel.pick(), "a")  # the very first pick never comes from the archive
            self.assertEqual(sel.pick(), "swarm:stage2")
            never = t.StartStateSelector(["a", "b"], 0, archive=arch, swarm_prob=0.0)
            self.assertEqual({never.pick() for _ in range(6)}, {"a", "b"})
            share = t.StartStateSelector(["a"], 0, archive=arch, swarm_prob=0.3, rng=np.random.default_rng(3))
            share.pick()
            picks = [share.pick() for _ in range(2000)]
            self.assertAlmostEqual(sum(p.startswith("swarm:") for p in picks) / 2000, 0.3, delta=0.05)

    def test_empty_archive_falls_back_to_start_states(self):
        with tempfile.TemporaryDirectory() as d:
            sel = t.StartStateSelector(["a"], 0, archive=t.SwarmArchive(d), swarm_prob=1.0)
            self.assertEqual([sel.pick() for _ in range(3)], ["a", "a", "a"])


class RouteProgressTests(unittest.TestCase):
    def test_fraction_of_remaining_route(self):
        tr = t.RouteProgressTracker()
        self.assertEqual(tr.value, 0.0)
        tr.add(-100.0, -50.0)  # half the route
        tr.add(-100.0, 0.0)  # all of it
        self.assertAlmostEqual(tr.value, 0.75)

    def test_clamped_and_skips_off_graph(self):
        tr = t.RouteProgressTracker()
        tr.add(-10.0, -30.0)  # went backwards -> 0
        tr.add(None, -5.0)
        tr.add(-10.0, None)
        tr.add(0.0, 0.0)
        self.assertEqual(list(tr.fracs), [0.0])

    def test_entropy_follows_progress(self):
        from src.ppo_update import EntropySchedule

        sched, tr = EntropySchedule(0.03, 0.005), t.RouteProgressTracker()
        self.assertAlmostEqual(sched.value(tr.value), 0.03)
        tr.add(-100.0, 0.0)
        self.assertAlmostEqual(sched.value(tr.value), 0.005)


class ConfigCliTests(unittest.TestCase):
    def cfg(self, *argv):
        return t.build_config(t.build_parser().parse_args(list(argv)))

    def test_default_is_blueprint_and_untyped_flags_do_not_override(self):
        cfg = self.cfg()
        self.assertEqual(cfg.name, "blueprint_v1")
        self.assertEqual((cfg.envs, cfg.rollout_steps, cfg.updates, cfg.lr, cfg.sticky_prob), (16, 128, 50, 3e-4, 0.1))
        self.assertEqual((cfg.ppo.gamma, cfg.ppo.num_minibatches, cfg.ppo.epochs), (0.997, 4, 3))

    def test_legacy_reproduces_old_trainer(self):
        cfg = self.cfg("--config", "legacy")
        self.assertEqual((cfg.ppo.num_minibatches, cfg.ppo.epochs, cfg.ppo.gamma), (1, 4, 0.99))
        self.assertEqual((cfg.entropy_start, cfg.entropy_end, cfg.sticky_prob, cfg.jitter_max), (0.02, 0.02, 0.0, 0))
        self.assertFalse(cfg.stage_balanced or cfg.swarm)

    def test_typed_flags_override(self):
        cfg = self.cfg("--config", "legacy", "--updates", "3", "--envs", "4", "--rollout", "16", "--max-steps", "48",
                       "--lr", "0.001", "--no-recurrent", "--save", "x.pt", "--run-id", "r", "--sticky-prob", "0.2")
        self.assertEqual((cfg.updates, cfg.envs, cfg.rollout_steps, cfg.max_steps, cfg.lr), (3, 4, 16, 48, 0.001))
        self.assertFalse(cfg.recurrent)
        self.assertEqual((cfg.save_path, cfg.run_id, cfg.sticky_prob), ("x.pt", "r", 0.2))
        self.assertEqual(cfg.ppo.num_minibatches, 1)  # legacy preset untouched

    def test_minibatches_adapt_to_env_count_unless_typed(self):
        self.assertEqual(self.cfg("--envs", "6").ppo.num_minibatches, 4)  # uneven groups (2/2/1/1) are allowed
        self.assertEqual(self.cfg("--envs", "5").ppo.num_minibatches, 4)
        self.assertEqual(self.cfg("--envs", "3").ppo.num_minibatches, 3)  # capped at the env count
        self.assertEqual(self.cfg("--envs", "6", "--minibatches", "2").ppo.num_minibatches, 2)
        self.assertEqual(t.fit_minibatches(16, 4), 4)

    def test_start_state_resolution(self):
        cfg = self.cfg()
        self.assertEqual(t.resolve_start_states(cfg, None, ["a", "b"], None), ["a", "b"])
        self.assertEqual(t.resolve_start_states(cfg, "new_bark_outside", None, False), ["new_bark_outside"])
        self.assertEqual(t.resolve_start_states(cfg, None, None, False), ["route29_entry"])
        self.assertEqual(t.resolve_start_states(cfg, None, None, None), t.CURRICULUM_STATES)
        cfg.start_states = ["x"]
        self.assertEqual(t.resolve_start_states(cfg, None, None, None), ["x"])

    def test_worker_cfg_follows_config(self):
        legacy = t.worker_env_cfg(self.cfg("--config", "legacy"), "logs/swarm/r")
        self.assertEqual((legacy["sticky_prob"], legacy["jitter_max"], legacy["swarm_dir"], legacy["stage_balanced"]),
                         (0.0, 0, None, False))
        bp = t.worker_env_cfg(self.cfg(), "logs/swarm/r")
        self.assertEqual((bp["sticky_prob"], bp["jitter_max"], bp["swarm_dir"]), (0.1, 6, "logs/swarm/r"))
        self.assertEqual(bp["swarm_prob"], 0.3)

    def test_compute_gae_is_the_ppo_update_one(self):
        self.assertIs(t.compute_gae, compute_gae)


def tiny_policy(obs_dim=5, gru=16):
    return SpatialVisionPolicy(obs_dim=obs_dim, num_actions=5, ram_hidden_dim=16, fusion_dim=32,
                               use_gru=True, gru_hidden_dim=gru)


class DonesInSequenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = tiny_policy().eval()  # eval: BatchNorm statistics fixed, results comparable across batch shapes
        g = torch.Generator().manual_seed(1)
        self.B, self.T = 3, 8
        self.screens = torch.rand(self.B, self.T, 3, 32, 32, generator=g)
        self.rams = torch.randn(self.B, self.T, 5, generator=g)
        self.actions = torch.randint(0, 5, (self.B, self.T), generator=g)
        self.hx = torch.randn(1, self.B, 16, generator=g)

    def run_seq(self, sl, hx, dones=None):
        with torch.no_grad():
            return self.policy.evaluate_actions_sequence(
                self.screens[:, sl], self.rams[:, sl], self.actions[:, sl], hx,
                **({} if dones is None else {"dones": dones}),
            )

    def test_default_none_is_unchanged(self):
        a = self.run_seq(slice(None), self.hx)
        b = self.run_seq(slice(None), self.hx, dones=torch.zeros(self.B, self.T, dtype=torch.bool))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)

    def test_mid_sequence_done_equals_running_episodes_separately(self):
        dones = torch.zeros(self.B, self.T, dtype=torch.bool)
        dones[1, 3] = True  # env 1's episode ends after step 3; step 4 starts a fresh one
        lp, ent, val = self.run_seq(slice(None), self.hx, dones)
        # env 1 run as two episodes: steps 0..3 with the carried state, steps 4..7 from a zero state
        first = self.run_seq(slice(0, 4), self.hx)
        second = self.run_seq(slice(4, 8), torch.zeros_like(self.hx))
        for full, a, b in zip((lp, ent, val), first, second):
            torch.testing.assert_close(full[1], torch.cat([a[1], b[1]]), atol=1e-5, rtol=1e-5)
        # envs without a done keep the carried state across the whole sequence
        plain = self.run_seq(slice(None), self.hx)
        for full, ref in zip((lp, ent, val), plain):
            torch.testing.assert_close(full[0], ref[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(full[2], ref[2], atol=1e-5, rtol=1e-5)
        # and the done really matters for env 1 after the boundary
        self.assertFalse(torch.allclose(lp[1, 4:], plain[0][1, 4:], atol=1e-6))
        torch.testing.assert_close(lp[1, :4], plain[0][1, :4], atol=1e-5, rtol=1e-5)

    def test_done_on_last_step_changes_nothing_inside_the_sequence(self):
        dones = torch.zeros(self.B, self.T, dtype=torch.bool)
        dones[:, -1] = True
        a = self.run_seq(slice(None), self.hx, dones)
        b = self.run_seq(slice(None), self.hx)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)

    def test_multiple_dones_and_adjacent_dones(self):
        dones = torch.zeros(self.B, self.T, dtype=torch.bool)
        dones[0, 1] = dones[0, 2] = dones[2, 5] = True
        lp, _, _ = self.run_seq(slice(None), self.hx, dones)
        z = torch.zeros_like(self.hx)
        a = self.run_seq(slice(0, 2), self.hx)[0][0]
        b = self.run_seq(slice(2, 3), z)[0][0]
        c = self.run_seq(slice(3, 8), z)[0][0]
        torch.testing.assert_close(lp[0], torch.cat([a, b, c]), atol=1e-5, rtol=1e-5)

    def test_ppo_update_passes_dones_through(self):
        policy = tiny_policy().train()
        B, T = 4, 6
        g = torch.Generator().manual_seed(5)
        dones = torch.zeros(B, T, dtype=torch.bool)
        dones[:, 2] = True
        seen = []
        orig = policy.evaluate_actions_sequence

        def spy(*a, **kw):
            seen.append(kw.get("dones"))
            return orig(*a, **kw)

        policy.evaluate_actions_sequence = spy
        batch = RolloutBatch(
            screens=torch.randint(0, 255, (B, T, 3, 32, 32), dtype=torch.uint8, generator=g),
            rams=torch.randn(B, T, 5, generator=g), actions=torch.randint(0, 5, (B, T), generator=g),
            old_logps=torch.full((B, T), -1.6), returns=torch.randn(B, T, generator=g),
            advantages=torch.randn(B, T, generator=g), initial_hx=torch.zeros(1, B, 16), dones=dones,
        )
        hp = PPOHParams(epochs=1, num_minibatches=2, target_kl=None)
        stats = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), batch, hp)
        self.assertEqual(stats["n_optimizer_steps"], 2)
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(d is not None and d.shape == (2, T) and bool(d[:, 2].all()) for d in seen))


class TruncationPlumbingTests(unittest.TestCase):
    def test_truncated_steps_bootstrap_with_final_observation_value(self):
        T, B = 3, 2
        rewards = torch.ones(T, B)
        values = torch.zeros(T, B)
        dones = torch.zeros(T, B, dtype=torch.bool)
        dones[1, 0] = dones[1, 1] = True
        trunc = torch.zeros(T, B, dtype=torch.bool)
        trunc[1, 0] = True  # env 0 hit the time limit, env 1 truly ended
        boot = torch.zeros(T, B)
        boot[1, 0] = 10.0
        _, adv = compute_gae(rewards, values, dones, torch.zeros(B), 1.0, 1.0, truncated=trunc, bootstrap_values=boot)
        self.assertAlmostEqual(float(adv[1, 0]), 11.0)  # reward + V(final obs)
        self.assertAlmostEqual(float(adv[1, 1]), 1.0)  # terminal: no bootstrap
        self.assertAlmostEqual(float(adv[0, 0]), 12.0)  # the chain before the truncation sees the bootstrapped value

    def test_policy_values_batches_final_observations(self):
        torch.manual_seed(0)
        policy = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=5, ram_hidden_dim=16, fusion_dim=32,
                                     use_gru=True, gru_hidden_dim=16).eval()
        rng = np.random.default_rng(0)
        scr = rng.integers(0, 255, (3, 144, 160, 3), dtype=np.uint8)
        ram = rng.standard_normal((3, DEEP_OBS_DIM))
        hx = torch.randn(1, 3, 16)
        v = t.policy_values(policy, scr, ram, hx, torch.device("cpu"))
        self.assertEqual(v.shape, (3,))
        one = t.policy_values(policy, scr[1:2], ram[1:2], hx[:, 1:2], torch.device("cpu"))
        torch.testing.assert_close(v[1:2], one, atol=1e-4, rtol=1e-4)  # subset == batch of one

    def test_vector_env_pops_final_obs_and_flags(self):
        class Pipe:
            def __init__(self, msg):
                self.msg = msg

            def send(self, m):
                pass

            def recv(self):
                return self.msg

        scr, ram = np.zeros((144, 160, 3), np.uint8), np.zeros(DEEP_OBS_DIM)
        trunc_info = {"truncated": True, "final_obs": (scr + 1, ram + 1)}
        ven = t.VectorEnv.__new__(t.VectorEnv)
        ven.num_envs = 2
        ven.last_infos = [None, None]
        ven.last_truncated = np.zeros(2, dtype=bool)
        ven.last_final = {}
        ven.pipes = [Pipe((scr, ram, 1.0, True, 1, 0, trunc_info)), Pipe((scr, ram, 1.0, True, 1, 0, {"truncated": False}))]
        ven.step(np.zeros(2, dtype=int))
        self.assertEqual(list(ven.last_truncated), [True, False])
        self.assertEqual(list(ven.last_final), [0])
        self.assertNotIn("final_obs", ven.last_infos[0])
        self.assertEqual(int(ven.last_final[0][0].max()), 1)


class OptimizerSidecarTests(unittest.TestCase):
    def test_roundtrip_and_missing_message(self):
        torch.manual_seed(0)
        policy = tiny_policy()
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        policy(torch.rand(2, 3, 32, 32), torch.randn(2, 5))[1].sum().backward()
        opt.step()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            t.save_checkpoint(policy, opt, path, update=7, route_progress=0.4)
            self.assertTrue(path.exists())
            side = t.checkpoint_sidecar(path)
            self.assertEqual(side.name, "m.optim.pt")
            self.assertTrue(side.exists())
            opt2 = torch.optim.AdamW(tiny_policy().parameters(), lr=1e-3)
            self.assertTrue(t.load_optimizer_state(opt2, path, torch.device("cpu")))
            self.assertEqual(len(opt2.state_dict()["state"]), len(opt.state_dict()["state"]))
            side.unlink()
            self.assertFalse(t.load_optimizer_state(opt2, path, torch.device("cpu")))  # prints, does not raise

    def test_policy_save_format_untouched(self):
        policy = tiny_policy()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            t.save_checkpoint(policy, torch.optim.SGD(policy.parameters(), lr=0.1), path, 0)
            blob = torch.load(str(path), map_location="cpu")
            self.assertEqual(set(blob), {"state_dict", "obs_dim", "num_actions", "param_count", "use_gru",
                                         "gru_hidden_dim", "model_type", "obs_schema"})


@unittest.skipUnless(HAVE_ROM, "needs the ROM")
class EnvWorkerEmulatorTests(unittest.TestCase):
    STATE = "new_bark_outside"

    def make(self, **cfg):
        w = t.EnvWorker(0, [self.STATE], max_steps=cfg.pop("max_steps", 6), cfg=cfg)
        self.addCleanup(w.close)
        return w

    def test_reset_seeds_events_so_no_episode_start_bonus(self):
        w = self.make()
        w.reset()
        st = w.reader.parse(w.emu.frame)
        self.assertGreater(len(w.seen_events), 50)  # the start state has ~129 flags already set
        self.assertEqual(w.seen_events, initial_seen_events(st))
        *_, rew, done, _, _, info = w.step(4)  # press A: harmless
        self.assertLess(abs(rew), 5.0)  # was ~+300 from re-paying the already-set flags
        self.assertEqual(info["reward_components"].get("event", 0.0), 0.0)

    def test_components_sum_to_reward_and_info_fields(self):
        w = self.make()
        w.reset()
        for a in (2, 2, 4, 3):
            _, _, rew, _, _, _, info = w.step(a)
            comps = info["reward_components"]
            self.assertAlmostEqual(sum(comps.values()), rew, places=5)
            for key in ("curiosity", "loop", "momentum"):
                self.assertIn(key, comps)
            self.assertNotIn("quest", comps)
            self.assertIn(info["stage"], (0, 1, 2, 3))
            self.assertIn("route_phi", info)
            self.assertIn("ep_start_phi", info)

    def test_truncation_returns_final_observation_and_resets(self):
        w = self.make(max_steps=3)
        w.reset()
        out = [w.step(4) for _ in range(3)]
        for _, _, _, done, _, _, info in out[:2]:
            self.assertFalse(done)
            self.assertFalse(info["truncated"])
            self.assertNotIn("final_obs", info)
        screen, ram, _, done, _, _, info = out[2]
        self.assertTrue(done)
        self.assertTrue(info["truncated"])  # time limit, not a real terminal
        fscreen, fram = info["final_obs"]
        self.assertEqual(fscreen.shape, (144, 160, 3))
        self.assertEqual(fram.shape, (DEEP_OBS_DIM,))
        self.assertEqual(w.steps, 0)  # the worker already reset; the returned obs is the new episode's first
        self.assertEqual(ram.shape, (DEEP_OBS_DIM,))

    def test_sticky_repeats_previous_move_but_reports_executed_action(self):
        w = self.make(sticky_prob=1.0 - 1e-9, max_steps=50)
        w.reset()
        *_, info = w.step(2)  # first move: nothing to repeat
        self.assertEqual(info["executed_action"], 2)
        for policy_action in (3, 1, 0):
            *_, info = w.step(policy_action)
            self.assertEqual(info["executed_action"], 2)  # sticky: still the first move
        off = self.make(sticky_prob=0.0, max_steps=50)
        off.reset()
        off.step(2)
        *_, info = off.step(3)
        self.assertEqual(info["executed_action"], 3)

    def test_jitter_moves_without_polluting_seen_sets(self):
        for seed in range(6):
            w = t.EnvWorker(0, [self.STATE], 10, {"jitter_max": 6, "seed": seed})
            try:
                w.reset()
                st = w.reader.parse(w.emu.frame)
                tile = (st.map_group, st.map_number, st.x, st.y)
                self.assertEqual(w.seen_tiles, {tile})
                self.assertEqual(w.seen_maps, {(st.map_group, st.map_number)})
                self.assertEqual(w.seen_events, initial_seen_events(st))
                self.assertEqual(w.steps, 0)
            finally:
                w.close()

    def test_jitter_changes_start_positions(self):
        starts = set()
        for seed in range(8):
            w = t.EnvWorker(0, [self.STATE], 10, {"jitter_max": 6, "seed": seed})
            try:
                w.reset()
                st = w.reader.parse(w.emu.frame)
                starts.add((st.map_group, st.map_number, st.x, st.y))
            finally:
                w.close()
        self.assertGreater(len(starts), 1)

    def test_stage_balanced_computes_stage_per_state(self):
        w = t.EnvWorker(0, [self.STATE, "agent_map_26_3"], 10, {"stage_balanced": True})
        try:
            self.assertEqual(sorted(w.selector.by_stage), [0, 2])  # New Bark / Cherrygrove (a story map)
            names = [w.selector.pick() for _ in range(4)]
            self.assertEqual(sorted(names), sorted([self.STATE, "agent_map_26_3"] * 2))
        finally:
            w.close()

    def test_swarm_archive_written_on_new_event_and_loaded_on_reset(self):
        with tempfile.TemporaryDirectory() as d:
            w = t.EnvWorker(0, [self.STATE], 10, {"swarm_dir": d, "swarm_prob": 1.0})
            try:
                w.reset()
                self.assertEqual(w.archive.keys(), [])
                # pretend a key flag flipped: the worker saves its state the next time it checks
                key_id = sorted(t.KEY_EVENT_IDS)[0]
                w.seen_events.add(key_id)
                st = w.reader.parse(w.emu.frame)
                w._maybe_archive(st)
                self.assertIn(f"event{key_id}", w.archive.keys())
                w.reset()  # swarm_prob 1.0 -> loads from the archive
                self.assertTrue(w.state_name.startswith("swarm:"))
                *_, info = w.step(4)
                self.assertTrue(info["state"].startswith("swarm:"))
            finally:
                w.close()


def _exit_worker(pipe, *args):
    """Worker that answers nothing and dies hard, like an emulator segfault / OOM kill."""
    import os
    pipe.recv()
    os._exit(1)


def _raising_worker(pipe, *args):
    from tools import train_deep_ppo as tt
    try:
        pipe.recv()
        raise ValueError("boom in worker")
    except ValueError:
        import traceback
        pipe.send(tt.WorkerFailure(traceback.format_exc()))


def _stuck_worker(pipe, *args):
    import time
    while True:
        time.sleep(0.2)  # ignores every command, including CLOSE


class WorkerCrashTests(unittest.TestCase):
    def make(self, target, n=2):
        ven = t.VectorEnv(n, ["a"], 4, worker_target=target)
        self.addCleanup(ven.close)
        return ven

    def test_hard_exit_raises_instead_of_hanging(self):
        import time
        ven = self.make(_exit_worker)
        t0 = time.time()
        with self.assertRaises(t.WorkerCrashed) as ctx:
            ven.reset()
        self.assertLess(time.time() - t0, 20)
        self.assertIn("exit code 1", str(ctx.exception))
        ven.close()
        ven.close()  # idempotent
        self.assertFalse(any(p.is_alive() for p in ven.procs))

    def test_worker_exception_reports_traceback(self):
        ven = self.make(_raising_worker, n=1)
        with self.assertRaises(t.WorkerCrashed) as ctx:
            ven.reset()
        self.assertIn("boom in worker", str(ctx.exception))

    def test_close_terminates_stuck_workers(self):
        ven = t.VectorEnv(2, ["a"], 4, worker_target=_stuck_worker)
        self.assertTrue(all(p.is_alive() for p in ven.procs))
        ven.close()
        self.assertFalse(any(p.is_alive() for p in ven.procs))
        ven.close()

    def test_parent_does_not_hold_child_end(self):
        ven = self.make(_exit_worker, n=1)
        # the child end is closed in the parent: a dead worker must surface as EOF, not block forever
        ven.pipes[0].send(("RESET", None))
        ven.procs[0].join(10)
        with self.assertRaises((EOFError, BrokenPipeError)):  # Windows raises BrokenPipeError from poll
            ven.pipes[0].poll(5)
            ven.pipes[0].recv()


class SwarmSnapshotTests(unittest.TestCase):
    def test_rotating_slots_fill_then_replace_oldest(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            a = t.SwarmArchive(d)
            for k in range(3):
                self.assertTrue(a.save_rotating("stage2", bytes([k]), slots=3))
                os.utime(a.dir / f"stage2_{k}.state", (100 + k, 100 + k))
            self.assertEqual(a.keys(), ["stage2_0", "stage2_1", "stage2_2"])
            self.assertTrue(a.save_rotating("stage2", b"new", slots=3))
            self.assertEqual(a.keys(), ["stage2_0", "stage2_1", "stage2_2"])
            self.assertEqual(a.load("stage2_0"), b"new")  # the oldest slot was replaced

    def test_sample_is_uniform_over_groups_then_slots(self):
        with tempfile.TemporaryDirectory() as d:
            a = t.SwarmArchive(d)
            for k in range(4):
                a.save(f"stage2_{k}", b"x")
            a.save("event5", b"x")
            rng = np.random.default_rng(0)
            picks = [a.sample(rng) for _ in range(400)]
            share = sum(p == "event5" for p in picks) / len(picks)
            self.assertGreater(share, 0.35)  # one group of two, not 1 of 5 files
            self.assertEqual({p for p in picks if p != "event5"}, {f"stage2_{k}" for k in range(4)})


class WorkerInteractionRewardTests(unittest.TestCase):
    class S:
        def __init__(self, map_id=(26, 3), x=3, y=3, facing=0, dialogue=False, battle=0):
            self.map_group, self.map_number = map_id
            self.x, self.y, self.facing, self.battle_mode = x, y, facing, battle
            self.party_hp = ()
            self.dlg = dialogue

    def run_one(self, before, after, action, ints, warps):
        from unittest import mock
        with mock.patch.object(t, "dialogue_open", lambda s: s.dlg):
            comps = {}
            return t.worker_interaction_reward(before, after, action, ints, warps, comps), comps

    def test_first_a_press_opening_text_pays_once(self):
        ints, warps = set(), set()
        r, c = self.run_one(self.S(), self.S(dialogue=True), 4, ints, warps)
        self.assertEqual(r, 0.5)
        self.assertEqual(c["interaction"], 0.5)
        r2, _ = self.run_one(self.S(), self.S(dialogue=True), 4, ints, warps)
        self.assertEqual(r2, 0.0)
        r3, _ = self.run_one(self.S(), self.S(dialogue=True), 2, set(), set())  # not an A press
        self.assertEqual(r3, 0.0)

    def test_new_warp_pays_once_per_edge(self):
        ints, warps = set(), set()
        r, c = self.run_one(self.S((26, 3)), self.S((26, 4)), 0, ints, warps)
        self.assertEqual(r, 1.0)
        self.assertEqual(c["warp"], 1.0)
        r, _ = self.run_one(self.S((26, 3)), self.S((26, 4)), 0, ints, warps)
        self.assertEqual(r, 0.0)  # same edge again

    def test_interaction_and_warp_are_explore_budgeted(self):
        ints, warps, budget = set(), set(), {"explore": 29.9}
        from unittest import mock
        with mock.patch.object(t, "dialogue_open", lambda s: s.dlg):
            r = t.worker_interaction_reward(self.S((26, 3)), self.S((26, 4), dialogue=False), 0, ints, warps, {}, budget)
        self.assertAlmostEqual(r, 0.1)


class RestoredStageStatesTests(unittest.TestCase):
    # story stage = position along the minimal path: Violet City 6, Violet Gym 7, Falkner beaten 8 (= MILESTONE_COUNT)
    EXPECTED = {"stage2_violet": 6, "stage2_gym_entered": 7, "stage3_falkner_beaten": 8}

    def test_files_exist_and_curriculum_lists_them(self):
        from src.states import StateRegistry
        reg = StateRegistry()
        for name in self.EXPECTED:
            self.assertGreater(len(reg.load_bytes(name)), 1000)  # loadable by name without a manifest entry
            self.assertIn(name, t.CURRICULUM_STATES)

    @unittest.skipUnless(HAVE_ROM, "needs the ROM")
    def test_story_path_reports_the_stage_of_each_state(self):
        w = t.EnvWorker(0, list(self.EXPECTED), 10, {"stage_balanced": True})
        try:
            self.assertEqual(w._stage_of_states(sorted(self.EXPECTED)), self.EXPECTED)
            self.assertEqual(sorted(w.selector.by_stage), [6, 7, 8])
        finally:
            w.close()


@unittest.skipUnless(HAVE_ROM, "needs the ROM")
class EnvWorkerQuestAndBumpTests(unittest.TestCase):
    def test_no_quest_bonus_or_termination_after_delivery(self):
        w = t.EnvWorker(0, ["stage2_violet"], 50, {})
        try:
            w.reset()
            self.assertGreaterEqual(t.current_stage(w.reader.parse(w.emu.frame)), 6)
            for a in (0, 0):
                *_, rew, done, _, _, info = w.step(a)
                self.assertFalse(done)
                self.assertNotIn("quest", info["reward_components"])
                self.assertLess(rew, 5.0)  # was +49.99 / +59.91 with the quest block
        finally:
            w.close()

    def test_a_press_is_not_a_bump(self):
        w = t.EnvWorker(0, ["new_bark_outside"], 50, {})
        try:
            w.reset()
            for _ in range(3):
                w.step(4)
                self.assertEqual(w.consecutive_bumps, 0)
        finally:
            w.close()

    def test_stage_snapshot_at_entry_not_mid_cutscene(self):
        with tempfile.TemporaryDirectory() as d:
            w = t.EnvWorker(0, ["new_bark_outside"], 50, {"swarm_dir": d})
            try:
                w.reset()
                st = w.reader.parse(w.emu.frame)
                w._last_stage = 0
                from unittest import mock
                with mock.patch.object(t, "current_stage", lambda s: 2), \
                        mock.patch.object(t, "dialogue_open", lambda s: True):
                    w._maybe_archive(st)
                    self.assertEqual(w.archive.keys(), [])  # deferred while the dialogue is open
                self.assertEqual(w._pending_stage, 2)
                with mock.patch.object(t, "current_stage", lambda s: 2):
                    w._maybe_archive(st)
                    w._maybe_archive(st)  # staying in stage 2 does not save again
                self.assertEqual(w.archive.keys(), ["stage2_0"])
            finally:
                w.close()


if __name__ == "__main__":
    unittest.main()
