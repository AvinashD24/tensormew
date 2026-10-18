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

    def test_probs_recurrent_and_sequence_evaluation(self):
        policy = SpatialVisionPolicy(obs_dim=17, num_actions=5, use_gru=True, gru_hidden_dim=512)
        screen_np = np.zeros((144, 160, 3), dtype=np.uint8)
        ram_np = np.zeros(17, dtype=np.float64)

        # Step 1: initial frame with hx=None
        probs1, hx1 = policy.probs_recurrent(screen_np, ram_np, hx=None)
        self.assertEqual(probs1.shape, (5,))
        self.assertAlmostEqual(float(probs1.sum()), 1.0, places=4)
        self.assertIsNotNone(hx1)
        self.assertEqual(hx1.shape, (1, 1, 512))

        # Step 2: subsequent frame passing hx1
        probs2, hx2 = policy.probs_recurrent(screen_np, ram_np, hx=hx1)
        self.assertEqual(probs2.shape, (5,))
        self.assertIsNotNone(hx2)

        # Sequence evaluation (B=2 envs, T=4 timesteps)
        screens_seq = torch.zeros(2, 4, 3, 144, 160)
        ram_seq = torch.zeros(2, 4, 17)
        actions_seq = torch.zeros(2, 4, dtype=torch.long)
        log_probs, entropy, values = policy.evaluate_actions(screens_seq, ram_seq, actions_seq)
        self.assertEqual(log_probs.shape, (2, 4))
        self.assertEqual(entropy.shape, (2, 4))
        self.assertEqual(values.shape, (2, 4))

    def test_load_feedforward_checkpoint_into_recurrent(self):
        ff_policy = SpatialVisionPolicy(obs_dim=17, num_actions=5, use_gru=False)
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            ff_policy.save(tmp.name)
            # Load feedforward checkpoint with override_use_gru=True
            rec_loaded = SpatialVisionPolicy.load(tmp.name, device="cpu", override_use_gru=True)
            self.assertTrue(rec_loaded.use_gru)
            self.assertIsNotNone(rec_loaded.gru)
            self.assertGreater(rec_loaded.param_count, 13_000_000)

