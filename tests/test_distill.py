"""Distillation: sequence logits, loss, shards / sampler, learnability, and a real collector run."""

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.distill import ChunkSampler, distill_loss, load_shard, save_shard, uniform_baseline
from src.grid_policy import SpatialGridPolicy
from src.spatial_grid import GRID_CHANNELS, GRID_SIZE

OBS = 80


def fake_shard(n=200, seed=0, learnable=False):
    rng = np.random.default_rng(seed)
    grids = rng.integers(0, 3, (n, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    rams = rng.normal(size=(n, OBS)).astype(np.float32)
    masks = np.ones((n, 6), dtype=bool)
    masks[::7, 5] = False
    if learnable:   # the teacher's action is a function of one grid cell
        best = grids[:, 0, 7, 8] % 4
    else:
        best = rng.integers(0, 6, n)
    probs = np.full((n, 6), 0.02, dtype=np.float32)
    probs[np.arange(n), best] = 0.9
    probs[~masks] = 0.0
    probs /= probs.sum(-1, keepdims=True)
    dones = np.zeros(n, dtype=bool)
    dones[40::50] = True
    return dict(grids=grids, rams=rams, probs=probs, masks=masks, dones=dones)


class SequenceOutputsTests(unittest.TestCase):
    def test_matches_evaluate_actions_sequence(self):
        torch.manual_seed(0)
        policy = SpatialGridPolicy(obs_dim=OBS, use_gru=True).eval()
        B, T = 3, 12
        grids = torch.randint(0, 3, (B, T, GRID_CHANNELS, GRID_SIZE, GRID_SIZE)).float()
        rams = torch.randn(B, T, OBS)
        actions = torch.randint(0, 5, (B, T))   # never the masked action 5
        dones = torch.zeros(B, T, dtype=torch.bool)
        dones[1, 5] = True
        masks = torch.ones(B, T, 6, dtype=torch.bool)
        masks[:, ::3, 5] = False
        with torch.no_grad():
            logits, values = policy.sequence_outputs(grids, rams, None, dones, masks)
            logp, _ent, vals2 = policy.evaluate_actions_sequence(grids, rams, actions, dones=dones, action_mask=masks)
        got = torch.log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(got, logp, atol=1e-5))
        self.assertTrue(torch.allclose(values, vals2, atol=1e-5))
        self.assertEqual(tuple(logits.shape), (B, T, 6))
        self.assertTrue((logits[:, ::3, 5] < -1e8).all())   # the mask is applied


class LossAndDataTests(unittest.TestCase):
    def test_kl_zero_when_student_equals_teacher(self):
        p = torch.softmax(torch.randn(2, 10, 6), -1)
        kl, agree = distill_loss(torch.log(p), p, burn_in=2)
        self.assertAlmostEqual(float(kl), 0.0, places=5)
        self.assertAlmostEqual(float(agree), 1.0)

    def test_kl_positive_and_masked_actions_are_free(self):
        p = torch.zeros(1, 4, 3)
        p[..., 0], p[..., 1] = 0.7, 0.3                    # the teacher never picks action 2 (masked)
        logits = torch.tensor([0.0, 0.0, -1e9]).expand(1, 4, 3)
        kl, _ = distill_loss(logits, p, burn_in=0)
        self.assertGreater(float(kl), 0.0)
        self.assertTrue(np.isfinite(float(kl)))

    def test_shard_roundtrip_and_sampler(self):
        with tempfile.TemporaryDirectory() as d:
            s = fake_shard()
            save_shard(Path(d) / "shard_0.npz", **s)
            back = load_shard(Path(d) / "shard_0.npz")
            self.assertTrue((back["grids"] == s["grids"]).all())
            sampler = ChunkSampler([back], length=32, burn_in=8, seed=1)
            batch = sampler.sample(5)
            self.assertEqual(batch["grids"].shape, (5, 32, GRID_CHANNELS, GRID_SIZE, GRID_SIZE))
            self.assertEqual(batch["masks"].shape, (5, 32, 6))
            self.assertEqual(batch["dones"].shape, (5, 32))
            self.assertIn("kl", uniform_baseline([back]))

    def test_sampler_rejects_short_shards(self):
        with self.assertRaises(ValueError):
            ChunkSampler([fake_shard(n=10)], length=32)


class LearnabilityTests(unittest.TestCase):
    def test_student_fits_a_learnable_teacher(self):
        torch.manual_seed(0)
        shard = fake_shard(n=600, learnable=True)
        sampler = ChunkSampler([shard], length=40, burn_in=8, seed=0)
        student = SpatialGridPolicy(obs_dim=OBS, use_gru=True)
        opt = torch.optim.AdamW(student.parameters(), lr=2e-3)

        def batch_loss():
            b = sampler.sample(16)
            t = {k: torch.from_numpy(v) for k, v in b.items()}
            logits, _ = student.sequence_outputs(t["grids"].float(), t["rams"], None, t["dones"], t["masks"])
            return distill_loss(logits, t["probs"], 8)

        first = float(batch_loss()[0])
        for _ in range(120):
            kl, _ = batch_loss()
            opt.zero_grad()
            kl.backward()
            opt.step()
        last, agree = batch_loss()
        self.assertLess(float(last), 0.6 * first)
        self.assertGreater(float(agree), 0.5)


class CollectorTests(unittest.TestCase):
    def test_collects_aligned_grids_with_a_random_teacher(self):
        try:
            from src.deep_policy import SpatialVisionPolicy
            from tools.distill_collect import collect
        except Exception as exc:
            raise unittest.SkipTest(f"collector unavailable: {exc}")
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "teacher.pt"
            SpatialVisionPolicy(obs_dim=80, use_gru=True).save(ckpt)
            args = argparse.Namespace(teacher=str(ckpt), config="long_v8", out=str(Path(d) / "out"), procs=1, steps=40,
                                      seed=0, archive_root=str(Path(d) / "swarm"), cpu=True)
            try:
                collect(0, args)
            except Exception as exc:   # no pyboy / ROM
                raise unittest.SkipTest(f"emulator unavailable: {exc}")
            shard = load_shard(Path(d) / "out" / "shard_0.npz")
            self.assertEqual(shard["grids"].shape, (40, GRID_CHANNELS, GRID_SIZE, GRID_SIZE))
            self.assertEqual(shard["rams"].shape, (40, 80))
            self.assertTrue(np.allclose(shard["probs"].sum(-1), 1.0, atol=1e-4))
            self.assertTrue((shard["grids"][:, 1, 7, 7] <= 1).all())   # centre is the player (class 1) or empty while a menu is open


if __name__ == "__main__":
    unittest.main()
