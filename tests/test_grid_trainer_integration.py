"""The grid policy wired into the trainer: config, class dispatch, tensor helpers, a real grid-mode env worker."""

import unittest

import numpy as np
import torch

from src.deep_policy import SpatialVisionPolicy
from src.grid_policy import SpatialGridPolicy, policy_class
from src.spatial_grid import GRID_CHANNELS, GRID_SIZE
from src.train_config import ConfigError, apply_overrides, from_dict, load_config


class ConfigAndDispatchTests(unittest.TestCase):
    def test_visual_defaults_to_pixels_and_validates(self):
        cfg = load_config("long_v9")
        self.assertEqual(cfg.visual, "pixels")
        self.assertEqual(apply_overrides(cfg, {"visual": "grid"}).visual, "grid")
        with self.assertRaises(ConfigError):
            apply_overrides(cfg, {"visual": "hologram"})
        self.assertEqual(from_dict({"visual": "grid"}).visual, "grid")

    def test_policy_class(self):
        self.assertIs(policy_class("grid"), SpatialGridPolicy)
        self.assertIs(policy_class("pixels"), SpatialVisionPolicy)


class VisualHelperTests(unittest.TestCase):
    def test_pixel_helpers_scale_and_permute(self):
        arr = np.full((2, 144, 160, 3), 255, dtype=np.uint8)
        batch = SpatialVisionPolicy.visual_batch(arr, "cpu")
        self.assertEqual(tuple(batch.shape), (2, 3, 144, 160))
        self.assertAlmostEqual(float(batch.max()), 1.0)
        stored = SpatialVisionPolicy.visual_store(arr)
        self.assertEqual(stored.dtype, torch.uint8)
        self.assertEqual(tuple(stored.shape), (2, 3, 144, 160))

    def test_grid_helpers_keep_class_ids(self):
        arr = np.zeros((3, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        arr[:, 0] = 7
        batch = SpatialGridPolicy.visual_batch(arr, "cpu")
        self.assertEqual(tuple(batch.shape), (3, GRID_CHANNELS, GRID_SIZE, GRID_SIZE))
        self.assertEqual(float(batch[0, 0, 0, 0]), 7.0)              # not divided by 255
        stored = SpatialGridPolicy.visual_store(arr)
        self.assertEqual(stored.dtype, SpatialGridPolicy.VISUAL_STORE_DTYPE)
        self.assertEqual(tuple(stored.shape[1:]), SpatialGridPolicy.VISUAL_SHAPE)

    def test_trainer_value_pass_accepts_grids(self):
        from tools.train_deep_ppo import policy_values
        policy = SpatialGridPolicy(obs_dim=81, use_gru=True)
        arr = np.zeros((2, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        hx = torch.zeros(1, 2, policy.gru_hidden_dim)
        values = policy_values(policy, arr, np.zeros((2, 81)), hx, "cpu")
        self.assertEqual(tuple(values.shape), (2,))

    def test_ppo_update_runs_on_grid_buffers(self):
        """One recurrent PPO update with a float grid buffer: no shape / dtype / rescaling errors."""
        from src.ppo_update import PPOHParams, RolloutBatch, ppo_update
        B, T = 2, 8
        policy = SpatialGridPolicy(obs_dim=81, use_gru=True)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        grids = torch.randint(0, 3, (B, T, *SpatialGridPolicy.VISUAL_SHAPE)).to(SpatialGridPolicy.VISUAL_STORE_DTYPE)
        batch = RolloutBatch(
            screens=grids, rams=torch.zeros(B, T, 81), actions=torch.randint(0, 6, (B, T)),
            old_logps=torch.full((B, T), -1.79), returns=torch.randn(B, T), advantages=torch.randn(B, T),
            initial_hx=torch.zeros(1, B, policy.gru_hidden_dim), dones=torch.zeros(B, T, dtype=torch.bool), masks=None,
        )
        stats = ppo_update(policy, optimizer, batch, PPOHParams(num_minibatches=2, epochs=1))
        self.assertTrue(np.isfinite(stats["value_loss"]))


class GridWorkerTests(unittest.TestCase):
    def test_env_worker_emits_grids(self):
        try:
            from tools.train_deep_ppo import EnvWorker
            worker = EnvWorker(0, ["route29_entry"], 32, {"visual": "grid", "time_feature": True})
        except Exception as exc:   # no pyboy / ROM
            raise unittest.SkipTest(f"env worker unavailable: {exc}")
        try:
            grid, ram = worker.reset()
            self.assertEqual(grid.shape, (GRID_CHANNELS, GRID_SIZE, GRID_SIZE))
            self.assertEqual(grid.dtype, np.uint8)
            self.assertEqual(int(grid[1, 7, 7]), 1)   # the player is class 1 at the centre
            for act in (0, 2, 1, 3):
                grid, ram, _rew, _done, _maps, _badges, _info = worker.step(act)
                self.assertEqual(grid.shape, (GRID_CHANNELS, GRID_SIZE, GRID_SIZE))
            self.assertGreater(int(grid[6].sum()), 0)   # the visited plane fills in as the agent walks
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
