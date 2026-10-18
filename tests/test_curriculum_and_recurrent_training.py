import unittest
import torch
import numpy as np

from src.deep_policy import SpatialVisionPolicy
from tools.train_deep_ppo import CURRICULUM_STATES, VectorEnv, compute_gae


class CurriculumAndRecurrentTrainingTests(unittest.TestCase):
    def test_curriculum_states_exist_in_manifest(self):
        from src.states import StateRegistry
        reg = StateRegistry()
        for state_name in CURRICULUM_STATES:
            bytes_data = reg.load_bytes(state_name)
            self.assertIsNotNone(bytes_data)
            self.assertGreater(len(bytes_data), 1000)

    def test_recurrent_gae_and_sequence_shapes(self):
        # 4 envs, 8 steps rollout
        B, T = 4, 8
        rewards = torch.ones(T, B)
        values = torch.zeros(T, B)
        dones = torch.zeros(T, B, dtype=torch.bool)
        dones[4, 1] = True  # Env 1 terminates at step 4

        returns, advantages = compute_gae(rewards, values, dones)
        self.assertEqual(returns.shape, (T, B))
        self.assertEqual(advantages.shape, (T, B))

        # Check sequence evaluation
        policy = SpatialVisionPolicy(obs_dim=17, num_actions=5, use_gru=True, gru_hidden_dim=512)
        screens = torch.zeros(B, T, 3, 144, 160)
        rams = torch.zeros(B, T, 17)
        actions = torch.zeros(B, T, dtype=torch.long)
        hx = torch.zeros(1, B, 512)

        log_probs, entropy, values_pred = policy.evaluate_actions_sequence(screens, rams, actions, hx)
        self.assertEqual(log_probs.shape, (B, T))
        self.assertEqual(entropy.shape, (B, T))
        self.assertEqual(values_pred.shape, (B, T))

    def test_reflexive_escape_candidates_calculation(self):
        from src.mover import walkable
        _DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        tiles = {"up": 0, "down": 122, "left": 0, "right": 0}  # 0 is walkable path
        tile_visit_counts = {(24, 3, 52, 8): 10, (24, 3, 51, 9): 2, (24, 3, 53, 9): 0}
        pos = (24, 3, 52, 9)

        candidates = []
        for d in ("left", "right", "up", "down"):
            if walkable(tiles.get(d, 0xFF), d):
                dx, dy = _DELTA[d]
                target = (pos[0], pos[1], pos[2] + dx, pos[3] + dy)
                candidates.append((tile_visit_counts.get(target, 0), d))

        candidates.sort(key=lambda x: x[0])
        # The least-visited open frontier tile should be chosen
        self.assertEqual(candidates[0][1], "right")
        self.assertEqual(candidates[0][0], 0)
