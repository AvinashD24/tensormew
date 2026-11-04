import tempfile
import unittest
import numpy as np
import torch

from src.deep_policy import SpatialVisionPolicy
from src.ram_parser import GameState
from src.deep_policy import CheckpointIncompatibleError
from src.rl_env import DEEP_OBS_DIM, intrinsic_curiosity_reward, observe


class ImprovedTrainingTests(unittest.TestCase):
    def test_intrinsic_curiosity_decay(self):
        r1 = intrinsic_curiosity_reward(1, scale=0.08)
        r4 = intrinsic_curiosity_reward(4, scale=0.08)
        r9 = intrinsic_curiosity_reward(9, scale=0.08)
        r16 = intrinsic_curiosity_reward(16, scale=0.08)

        self.assertAlmostEqual(r1, 0.08, places=4)
        self.assertAlmostEqual(r4, 0.04, places=4)
        self.assertAlmostEqual(r9, 0.08 / 3.0, places=4)
        self.assertAlmostEqual(r16, 0.02, places=4)
        self.assertGreater(r1, r4)
        self.assertGreater(r4, r9)

    def test_deep_observation_has_no_oracle_and_the_right_width(self):
        s = GameState.from_dict({
            "map_group": 24, "map_number": 4, "x": 5, "y": 5, "tiles": {}, "event_flags": [0] * 300
        })
        obs17 = observe(s, set())
        self.assertEqual(len(obs17), 17)
        deep = observe(s, set(), deep=True)
        self.assertEqual(len(deep), DEEP_OBS_DIM)
        self.assertEqual(DEEP_OBS_DIM, 77)
        np.testing.assert_array_equal(deep[:16], obs17[:16])

    def test_old_dim_checkpoint_is_refused_with_a_clear_error(self):
        old_policy = SpatialVisionPolicy(obs_dim=17, num_actions=5, use_gru=True)
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/old.pt"
            old_policy.save(path)
            with self.assertRaises(CheckpointIncompatibleError) as ctx:
                SpatialVisionPolicy.load(path, device="cpu", override_obs_dim=DEEP_OBS_DIM, check_schema=True)
            self.assertIn("Retrain from scratch", str(ctx.exception))
            with self.assertRaises(CheckpointIncompatibleError):
                SpatialVisionPolicy.load(path, device="cpu", override_obs_dim=DEEP_OBS_DIM)  # no silent zero-padding

    def test_v1_30_dim_five_action_checkpoint_is_refused(self):
        v1 = SpatialVisionPolicy(obs_dim=30, num_actions=5, use_gru=True)
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/v1.pt"
            blob = {"state_dict": v1.state_dict(), "obs_dim": 30, "num_actions": 5, "use_gru": True,
                    "gru_hidden_dim": 512, "model_type": "SpatialVisionPolicy"}  # a v1 file: no obs_schema key
            torch.save(blob, path)
            with self.assertRaises(CheckpointIncompatibleError):
                SpatialVisionPolicy.load(path, device="cpu", check_schema=True)

    def test_current_checkpoint_round_trips_with_schema_check(self):
        pol = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=6, use_gru=True)
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/new.pt"
            pol.save(path)
            loaded = SpatialVisionPolicy.load(path, device="cpu", override_obs_dim=DEEP_OBS_DIM, check_schema=True)
            self.assertEqual(loaded.obs_dim, DEEP_OBS_DIM)
            self.assertEqual(loaded.num_actions, 6)


if __name__ == "__main__":
    unittest.main()
