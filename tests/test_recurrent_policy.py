import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.recurrent_policy import RecurrentPolicy, gru_step


class RecurrentPolicyTests(unittest.TestCase):
    def test_hidden_changes_after_act(self):
        policy = RecurrentPolicy(obs_dim=4, rng=np.random.default_rng(1))
        before = policy.hidden.copy()
        action, logp = policy.act(np.array([1.0, -1.0, 0.5, 0.25]))
        self.assertEqual(before.shape, (16,))
        self.assertFalse(np.allclose(before, policy.hidden))
        self.assertIn(action, range(5))
        self.assertTrue(np.isfinite(logp))

    def test_reset_zeros_hidden(self):
        policy = RecurrentPolicy(obs_dim=4, rng=np.random.default_rng(2))
        policy.act(np.ones(4))
        policy.reset()
        self.assertTrue(np.all(policy.hidden == 0.0))

    def test_save_load_roundtrip(self):
        policy = RecurrentPolicy(obs_dim=4, hidden_dim=16, n_actions=5, rng=np.random.default_rng(3))
        policy.act(np.ones(4))
        self.assertFalse(np.allclose(policy.hidden, 0.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "policy.npz")
            policy.save(path)
            other = RecurrentPolicy(obs_dim=2, hidden_dim=4, n_actions=3, rng=np.random.default_rng(9))
            other.load(path)
        for key in policy.w:
            np.testing.assert_allclose(other.w[key], policy.w[key])
        np.testing.assert_allclose(other.w_a, policy.w_a)
        np.testing.assert_allclose(other.b_a, policy.b_a)
        np.testing.assert_allclose(other.hidden, policy.hidden)
        self.assertEqual(other.obs_dim, 4)
        self.assertEqual(other.hidden_dim, 16)
        self.assertEqual(other.n_actions, 5)

    def test_two_observations_yield_actions_in_range(self):
        policy = RecurrentPolicy(obs_dim=3, rng=np.random.default_rng(4))
        first, first_logp = policy.act(np.array([0.0, 1.0, 0.0]))
        policy.reset()
        second, second_logp = policy.act(np.array([1.0, 0.0, -1.0]))
        self.assertIn(first, range(5))
        self.assertIn(second, range(5))
        self.assertTrue(np.isfinite(first_logp))
        self.assertTrue(np.isfinite(second_logp))

    def test_gru_step_is_a_forward(self):
        rng = np.random.default_rng(5)
        policy = RecurrentPolicy(obs_dim=3, hidden_dim=16, rng=rng)
        hidden = np.zeros(16)
        nxt = gru_step(hidden, np.array([0.2, -0.4, 0.6]), policy.w)
        again = gru_step(hidden, np.array([0.2, -0.4, 0.6]), policy.w)
        self.assertEqual(nxt.shape, (16,))
        self.assertFalse(np.allclose(nxt, hidden))
        np.testing.assert_allclose(nxt, again)
        np.testing.assert_allclose(hidden, 0.0)


if __name__ == "__main__":
    unittest.main()
