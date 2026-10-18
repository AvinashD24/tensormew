import tempfile
import unittest
import numpy as np
import torch

from src.deep_policy import SpatialVisionPolicy
from src.ram_parser import GameState
from src.rl_env import intrinsic_curiosity_reward, observe, quest_goal_vector


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

    def test_quest_goal_vector_progression(self):
        # Phase 1: New Bark Town / Route 29 -> Goal is reach Cherrygrove [1, 0, 0, 0]
        s1 = GameState.from_dict({
            "map_group": 24, "map_number": 4, "x": 5, "y": 5, "event_flags": [0] * 300
        })
        v1 = quest_goal_vector(s1)
        np.testing.assert_array_equal(v1, [1.0, 0.0, 0.0, 0.0])

        # Phase 2: Route 30 -> Goal is get Mystery Egg [0, 1, 0, 0]
        s2 = GameState.from_dict({
            "map_group": 26, "map_number": 1, "x": 10, "y": 10, "event_flags": [0] * 300
        })
        v2 = quest_goal_vector(s2)
        np.testing.assert_array_equal(v2, [0.0, 1.0, 0.0, 0.0])

        # Phase 3: Got egg from Mr. Pokemon, rival not beaten -> Goal is beat Rival [0, 0, 1, 0]
        # Flag 30: EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON (byte 3, bit 6)
        flags = [0] * 300
        flags[30 // 8] |= (1 << (30 % 8))
        s3 = GameState.from_dict({
            "map_group": 26, "map_number": 3, "x": 33, "y": 6, "event_flags": flags, "cherrygrove_scene": 1
        })
        v3 = quest_goal_vector(s3)
        np.testing.assert_array_equal(v3, [0.0, 0.0, 1.0, 0.0])

        # Phase 4: Beat rival, egg not delivered -> Goal is deliver egg to Elm [0, 0, 0, 1]
        s4 = GameState.from_dict({
            "map_group": 26, "map_number": 3, "x": 35, "y": 6, "event_flags": flags, "cherrygrove_scene": 0
        })
        v4 = quest_goal_vector(s4)
        np.testing.assert_array_equal(v4, [0.0, 0.0, 0.0, 1.0])

    def test_observe_goal_conditioning_dimensions(self):
        s = GameState.from_dict({
            "map_group": 24, "map_number": 4, "x": 5, "y": 5, "tiles": {}, "event_flags": [0] * 300
        })
        # Standard legacy base observe: 17 dims
        obs17 = observe(s, set(), include_goals=False)
        self.assertEqual(len(obs17), 17)

        # Goal-conditioned observe: 21 dims (17 base + 4 goals)
        obs21 = observe(s, set(), include_goals=True)
        self.assertEqual(len(obs21), 21)
        np.testing.assert_array_equal(obs21[:17], obs17)
        np.testing.assert_array_equal(obs21[17:], [1.0, 0.0, 0.0, 0.0])

    def test_load_17_dim_checkpoint_into_21_dim_policy(self):
        # Create and save a 17-dim model checkpoint
        old_policy = SpatialVisionPolicy(obs_dim=17, num_actions=5, use_gru=True)
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            old_policy.save(tmp.name)

            # Load into 21-dim policy with override_obs_dim=21
            new_policy = SpatialVisionPolicy.load(tmp.name, device="cpu", override_obs_dim=21)
            self.assertEqual(new_policy.obs_dim, 21)
            self.assertTrue(new_policy.use_gru)

            # Verify that original weights for features 0..16 were preserved exactly
            old_w = old_policy.ram_trunk[0].weight.data
            new_w = new_policy.ram_trunk[0].weight.data
            torch.testing.assert_close(new_w[:, :17], old_w)

            # Verify forward pass works on 21-dim RAM vector
            screen = torch.zeros(2, 3, 144, 160)
            ram21 = torch.zeros(2, 21)
            logits, val = new_policy(screen, ram21)
            self.assertEqual(logits.shape, (2, 5))
            self.assertEqual(val.shape, (2,))


if __name__ == "__main__":
    unittest.main()
