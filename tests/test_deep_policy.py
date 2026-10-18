import tempfile
import unittest
import torch
import numpy as np

from src.deep_policy import SpatialVisionPolicy


class DeepPolicyTests(unittest.TestCase):
    def test_feedforward_policy_forward_and_params(self):
        policy = SpatialVisionPolicy(obs_dim=17, num_actions=5, use_gru=False)
        self.assertGreater(policy.param_count, 11_000_000)
        self.assertFalse(policy.use_gru)

        screen = torch.zeros(2, 3, 144, 160)
        ram = torch.zeros(2, 17)
        logits, val = policy(screen, ram)
        self.assertEqual(logits.shape, (2, 5))
        self.assertEqual(val.shape, (2,))

    def test_recurrent_policy_forward_and_hidden_state(self):
        policy = SpatialVisionPolicy(obs_dim=17, num_actions=5, use_gru=True, gru_hidden_dim=512)
        self.assertTrue(policy.use_gru)
        # Recurrent model has ~13.2M params (11.6M feedforward + 1.57M GRU)
        self.assertGreater(policy.param_count, 13_000_000)

        screen = torch.zeros(2, 3, 144, 160)
        ram = torch.zeros(2, 17)

        # Standard forward
        logits, val = policy(screen, ram)
        self.assertEqual(logits.shape, (2, 5))
        self.assertEqual(val.shape, (2,))

        # Recurrent forward with hx
        hx = torch.zeros(1, 2, 512)
        logits_rec, val_rec, new_hx = policy.forward_recurrent(screen, ram, hx)
        self.assertEqual(logits_rec.shape, (2, 5))
        self.assertEqual(val_rec.shape, (2,))
        self.assertIsNotNone(new_hx)
        self.assertEqual(new_hx.shape, (1, 2, 512))

    def test_save_and_load_recurrent_policy(self):
        policy = SpatialVisionPolicy(obs_dim=17, num_actions=5, use_gru=True)
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            policy.save(tmp.name)
            loaded = SpatialVisionPolicy.load(tmp.name, device="cpu")
            self.assertTrue(loaded.use_gru)
            self.assertEqual(loaded.param_count, policy.param_count)

            screen = torch.zeros(1, 3, 144, 160)
            ram = torch.zeros(1, 17)
            l1, v1 = policy(screen, ram)
            l2, v2 = loaded(screen, ram)
            torch.testing.assert_close(l1, l2)
            torch.testing.assert_close(v1, v2)
