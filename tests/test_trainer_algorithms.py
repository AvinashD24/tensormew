"""Trainer-side algorithm plumbing: memory-capped minibatches, atomic checkpoints, resume, argument parsing."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.deep_policy import SpatialVisionPolicy
from src.ppo_update import PPOHParams, RolloutBatch, ppo_update
from tools import train_deep_ppo as t

GIB = 1024 ** 3


def tiny_policy():
    return SpatialVisionPolicy(obs_dim=5, num_actions=5, ram_hidden_dim=16, fusion_dim=32, use_gru=True, gru_hidden_dim=16)


def cfg(*argv):
    return t.build_config(t.build_parser().parse_args(list(argv)))


class PlanMinibatchesTests(unittest.TestCase):
    def test_cpu_has_no_cap(self):
        n, msg = t.plan_minibatches(16, 128, 4, None, False)
        self.assertEqual(n, 4)
        self.assertIn("no CUDA", msg)

    def test_10gb_card_16_envs_is_capped(self):
        # 45% of 10 GiB / 13 MiB ~ 354 frames -> 2 envs (256 frames) per minibatch -> 8 minibatches
        n, msg = t.plan_minibatches(16, 128, 4, 10 * GIB, False)
        self.assertEqual(n, 8)
        self.assertIn("raised 4 -> 8", msg)

    def test_big_card_keeps_requested(self):
        n, _ = t.plan_minibatches(16, 128, 4, 80 * GIB, False)
        self.assertEqual(n, 4)

    def test_typed_value_is_overridable_but_warned(self):
        n, msg = t.plan_minibatches(16, 128, 4, 10 * GIB, True)
        self.assertEqual(n, 4)
        self.assertIn("[warn]", msg)

    def test_never_more_minibatches_than_units(self):
        n, _ = t.plan_minibatches(3, 1024, 2, 2 * GIB, False)  # one env alone exceeds the budget
        self.assertEqual(n, 3)

    def test_fit_minibatches_allows_uneven_groups(self):
        self.assertEqual(t.fit_minibatches(7, 4), 4)  # prime env count no longer degrades to 1
        self.assertEqual(t.fit_minibatches(3, 4), 3)
        self.assertEqual(t.fit_minibatches(16, 4), 4)


class UnevenGroupsTests(unittest.TestCase):
    def test_ppo_update_with_prime_env_count(self):
        torch.manual_seed(0)
        policy = tiny_policy()
        B, T = 7, 4
        g = torch.Generator().manual_seed(1)
        batch = RolloutBatch(
            screens=(torch.rand(B, T, 3, 32, 32, generator=g) * 255).to(torch.uint8), rams=torch.randn(B, T, 5, generator=g),
            actions=torch.randint(0, 5, (B, T), generator=g), old_logps=torch.zeros(B, T),
            returns=torch.randn(B, T, generator=g), advantages=torch.randn(B, T, generator=g), initial_hx=torch.zeros(1, B, 16),
        )
        hp = PPOHParams(epochs=2, num_minibatches=4, target_kl=None)
        stats = ppo_update(policy, torch.optim.SGD(policy.parameters(), lr=1e-3), batch, hp)
        self.assertEqual(stats["n_optimizer_steps"], 8)  # 4 uneven groups x 2 epochs, none dropped


class CheckpointTests(unittest.TestCase):
    def test_atomic_save_leaves_no_temp_and_extra_roundtrips(self):
        policy = tiny_policy()
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.pt"
            t.save_checkpoint(policy, opt, path, update=4, route_progress=0.3,
                              extra={"progress": {"fracs": [0.1, 0.2]}, "reward_scaler": {"var": 2.0}})
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), ["run.optim.pt", "run.pt"])
            blob = t.load_sidecar(path, "cpu")
            self.assertEqual((blob["update"], blob["route_progress"]), (4, 0.3))
            self.assertEqual(blob["progress"]["fracs"], [0.1, 0.2])
            self.assertEqual(blob["reward_scaler"]["var"], 2.0)

    def test_failed_write_keeps_the_previous_checkpoint(self):
        policy = tiny_policy()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.pt"
            policy.save(path)
            before = path.read_bytes()

            def boom(tmp):
                Path(tmp).write_bytes(b"partial")
                raise OSError("disk full")

            with self.assertRaises(OSError):
                t._atomic(boom, path)
            self.assertEqual(path.read_bytes(), before)

    def test_best_is_a_separate_file(self):
        policy = tiny_policy()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "logs" / "run.pt"
            self.assertEqual(t.best_path(path).name, "run.best.pt")
            out = t.save_best(policy, path)
            self.assertTrue(out.exists())
            self.assertFalse(path.exists())

    def test_missing_sidecar_warns_and_returns_none(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print") as pr:
            self.assertIsNone(t.load_sidecar(Path(d) / "nope.pt", "cpu"))
            self.assertIn("WARNING", pr.call_args[0][0])

    def test_tracker_state_roundtrip(self):
        a = t.RouteProgressTracker()
        a.add(-10.0, -5.0)
        a.add(-10.0, -8.0)
        b = t.RouteProgressTracker()
        b.load_state_dict(a.state_dict())
        self.assertAlmostEqual(b.value, a.value)


class ResumeAndParserTests(unittest.TestCase):
    def test_resume_without_checkpoint_raises_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as cm:
                t.check_resume(Path(d) / "missing.pt")
            self.assertIn("--resume", str(cm.exception))
            # train() fails before spawning a single worker
            c = cfg("--save", str(Path(d) / "missing.pt"), "--resume")
            with mock.patch.object(t, "VectorEnv") as ve:
                with self.assertRaises(FileNotFoundError):
                    t.train(c, states=["x"], resume=True)
                ve.assert_not_called()

    def test_main_exits_nonzero_on_missing_resume(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                t.main(["--resume", "--save", str(Path(d) / "missing.pt")])
            self.assertNotEqual(cm.exception.code, 0)

    def test_existing_checkpoint_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pt"
            tiny_policy().save(p)
            self.assertEqual(t.check_resume(p), p)

    def test_abbreviated_flags_rejected(self):
        parser = t.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--minibatch", "2"])  # prefix of --minibatches

    def test_minibatches_more_than_envs_refused(self):
        with self.assertRaises(SystemExit):
            t.main(["--envs", "2", "--minibatches", "4", "--save", "x.pt"])
        with self.assertRaises(t.ConfigError):
            cfg("--envs", "2", "--minibatches", "4")

    def test_config_flags_for_scaling_and_bn(self):
        c = cfg("--config", "legacy", "--reward-scale", "running_std", "--bn-mode", "eval_refresh", "--vf-clip")
        self.assertEqual((c.reward_scale, c.ppo.bn_mode, c.ppo.vf_clip), ("running_std", "eval_refresh", True))
        c = cfg()
        self.assertEqual((c.reward_scale, c.ppo.bn_mode, c.ppo.vf_clip), ("running_std", "eval_refresh", False))


if __name__ == "__main__":
    unittest.main()
