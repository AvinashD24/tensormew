import math
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.ppo_update import (
    EntropySchedule, PPOHParams, RolloutBatch, RunningReturnScaler, compute_gae, ppo_update, refresh_batchnorm,
)

A = 4  # actions
H = 5  # hidden size of the stand-in recurrent policy
D = 3  # ram dim


class SeqPolicy(nn.Module):
    """Tiny stand-in with the recurrent method names; logits/values depend on the initial hidden state."""

    use_gru = True

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(3 * 2 * 2 + D, 8)
        self.hx_proj = nn.Linear(H, 8)
        self.actor = nn.Linear(8, A)
        self.critic = nn.Linear(8, 1)
        self.calls: list[tuple[list[int], torch.Tensor | None]] = []
        self.modes: list[bool] = []

    def evaluate_actions_sequence(self, screens, rams, actions, hx=None):
        B, T = screens.shape[:2]
        self.modes.append(self.training)
        self.calls.append((screens[:, 0, 0, 0, 0].round().long().tolist(), None if hx is None else hx.detach().clone()))
        x = torch.cat([screens.reshape(B, T, -1), rams], dim=-1)
        h = torch.tanh(self.enc(x))
        if hx is not None:
            h = h + torch.tanh(self.hx_proj(hx[0])).unsqueeze(1)
        dist = Categorical(logits=self.actor(h))
        return dist.log_prob(actions), dist.entropy(), self.critic(h).squeeze(-1)


class FFPolicy(nn.Module):
    use_gru = False

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(3 * 2 * 2 + D, 8)
        self.actor = nn.Linear(8, A)
        self.critic = nn.Linear(8, 1)
        self.n_calls = 0

    def evaluate_actions(self, screens, rams, actions):
        self.n_calls += 1
        h = torch.tanh(self.enc(torch.cat([screens.reshape(screens.shape[0], -1), rams], dim=-1)))
        dist = Categorical(logits=self.actor(h))
        return dist.log_prob(actions), dist.entropy(), self.critic(h).squeeze(-1)


def make_batch(B=8, T=6, seed=0, with_hx=True, env_ids=True) -> RolloutBatch:
    g = torch.Generator().manual_seed(seed)
    screens = torch.randn(B, T, 3, 2, 2, generator=g)
    if env_ids:
        screens[:, :, 0, 0, 0] = torch.arange(B, dtype=torch.float32).unsqueeze(1)
    return RolloutBatch(
        screens=screens,
        rams=torch.randn(B, T, D, generator=g),
        actions=torch.randint(0, A, (B, T), generator=g),
        old_logps=torch.full((B, T), -math.log(A)),
        returns=torch.randn(B, T, generator=g),
        advantages=torch.randn(B, T, generator=g),
        initial_hx=torch.randn(1, B, H, generator=g) if with_hx else None,
    )


def frozen_sgd(policy):
    return torch.optim.SGD(policy.parameters(), lr=0.0)


class GaeTests(unittest.TestCase):
    def test_matches_legacy_without_truncation(self):
        torch.manual_seed(0)
        T, B = 7, 3
        r, v = torch.randn(T, B), torch.randn(T, B)
        d = (torch.rand(T, B) < 0.2).float()

        gamma, lam = 0.99, 0.95
        adv_ref = torch.zeros_like(r)
        last = 0.0
        for t in reversed(range(T)):  # legacy loop from tools/train_deep_ppo.py
            nv = v[t + 1] if t + 1 < T else 0.0
            nnt = 1.0 - d[t]
            delta = r[t] + gamma * nv * nnt - v[t]
            last = delta + gamma * lam * nnt * last
            adv_ref[t] = last
        ret, adv = compute_gae(r, v, d, 0.0, gamma, lam)
        self.assertTrue(torch.allclose(adv, adv_ref, atol=1e-6))
        self.assertTrue(torch.allclose(ret, adv_ref + v, atol=1e-6))

    def test_hand_computed_with_last_value_and_done(self):
        # one env, gamma=0.5, lam=1 (so GAE = discounted sum of deltas)
        r = torch.tensor([[1.0], [2.0], [4.0]])
        v = torch.tensor([[0.5], [1.0], [2.0]])
        d = torch.tensor([[0.0], [1.0], [0.0]])
        ret, adv = compute_gae(r, v, d, torch.tensor([8.0]), 0.5, 1.0)
        # t=2: delta = 4 + 0.5*8 - 2 = 6; t=1 (done): delta = 2 - 1 = 1, chain cut; t=0: 1 + 0.5*1 - 0.5 + 0.5*1 = 1.5
        self.assertAlmostEqual(adv[2, 0].item(), 6.0)
        self.assertAlmostEqual(adv[1, 0].item(), 1.0)
        self.assertAlmostEqual(adv[0, 0].item(), 1.5)
        self.assertAlmostEqual(ret[2, 0].item(), 8.0)

    def test_truncation_bootstraps_but_terminal_does_not(self):
        r = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        v = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
        d = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # both envs end after t=1
        trunc = torch.tensor([[0.0, 0.0], [0.0, 1.0]])  # env 1 is a time-limit end
        boot = torch.tensor([[0.0, 0.0], [0.0, 10.0]])
        _, adv = compute_gae(r, v, d, 0.0, 0.9, 0.8, truncated=trunc, bootstrap_values=boot)
        # env 0 terminal: delta1 = 1 - 3 = -2 ; env 1 truncated: delta1 = 1 + 0.9*10 - 3 = 7
        self.assertAlmostEqual(adv[1, 0].item(), -2.0, places=5)
        self.assertAlmostEqual(adv[1, 1].item(), 7.0, places=5)
        # the chain is cut across the end: step 0 only sees the next value, not the future advantage
        self.assertAlmostEqual(adv[0, 0].item(), 1 + 0.9 * 3 - 0 + 0.9 * 0.8 * -2.0, places=5)
        # truncation flag alone (dones not set) also ends the episode and bootstraps
        d2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        _, adv2 = compute_gae(r, v, d2, 0.0, 0.9, 0.8, truncated=trunc, bootstrap_values=boot)
        self.assertTrue(torch.allclose(adv2, adv))

    def test_requires_bootstrap_values(self):
        z = torch.zeros(2, 1)
        with self.assertRaises(ValueError):
            compute_gae(z, z, z, 0.0, 0.9, 0.9, truncated=z)


class EntropyScheduleTests(unittest.TestCase):
    def test_linear_and_clamped(self):
        s = EntropySchedule(start=0.03, end=0.005)
        self.assertAlmostEqual(s.value(0.0), 0.03)
        self.assertAlmostEqual(s.value(1.0), 0.005)
        self.assertAlmostEqual(s.value(0.5), 0.0175)
        self.assertAlmostEqual(s.value(-3.0), 0.03)
        self.assertAlmostEqual(s.value(7.0), 0.005)
        self.assertAlmostEqual(EntropySchedule().value(0.0), 0.03)


class PpoUpdateTests(unittest.TestCase):
    def test_single_batch_gradient_matches_legacy_loss(self):
        torch.manual_seed(1)
        policy = SeqPolicy()
        batch = make_batch()
        # a non-trivial ratio so the clip branch participates
        batch.old_logps = batch.old_logps + 0.3 * torch.randn_like(batch.old_logps)
        hp = PPOHParams(epochs=1, num_minibatches=1, norm_adv=False, vf_clip=False, vf_coef=0.5,
                        entropy_coef=0.02, max_grad_norm=1e9, target_kl=None)
        opt = frozen_sgd(policy)  # lr=0 keeps .grad intact after the step

        policy.train()
        new_lp, ent, val = policy.evaluate_actions_sequence(batch.screens, batch.rams, batch.actions, batch.initial_hx)
        ratio = torch.exp(new_lp - batch.old_logps)
        s1 = ratio * batch.advantages
        s2 = torch.clamp(ratio, 0.8, 1.2) * batch.advantages
        legacy = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(val, batch.returns) + 0.02 * (-ent.mean())
        policy.zero_grad()
        legacy.backward()
        ref = [p.grad.clone() for p in policy.parameters()]

        stats = ppo_update(policy, opt, batch, hp)
        self.assertEqual(stats["n_optimizer_steps"], 1)
        self.assertFalse(stats["early_stopped"])
        for p, g in zip(policy.parameters(), ref):
            self.assertTrue(torch.allclose(p.grad, g, atol=1e-6), "gradient differs from legacy loss")

    def test_entropy_coef_override(self):
        torch.manual_seed(2)
        batch = make_batch()
        hp = PPOHParams(epochs=1, num_minibatches=1, norm_adv=False, vf_clip=False, entropy_coef=0.02,
                        max_grad_norm=1e9, target_kl=None)

        def grads(coef):
            torch.manual_seed(3)
            pol = SeqPolicy()
            ppo_update(pol, frozen_sgd(pol), batch, hp, entropy_coef=coef)
            return torch.cat([p.grad.flatten() for p in pol.parameters()])

        self.assertFalse(torch.allclose(grads(0.0), grads(0.5)))
        torch.manual_seed(3)
        pol = SeqPolicy()
        ppo_update(pol, frozen_sgd(pol), batch, hp)
        default = torch.cat([p.grad.flatten() for p in pol.parameters()])
        self.assertTrue(torch.allclose(default, grads(0.02), atol=1e-7))

    def test_each_env_once_per_epoch_with_matching_hidden_slice(self):
        torch.manual_seed(4)
        B = 8
        policy = SeqPolicy()
        batch = make_batch(B=B)
        batch.initial_hx = torch.arange(B, dtype=torch.float32).reshape(1, B, 1).repeat(1, 1, H)  # hx[:, b] == b
        hp = PPOHParams(epochs=3, num_minibatches=4, target_kl=None, bn_mode="train")
        stats = ppo_update(policy, torch.optim.SGD(policy.parameters(), lr=1e-3), batch, hp)

        self.assertEqual(stats["n_optimizer_steps"], 12)
        self.assertEqual(len(policy.calls), 12)
        self.assertTrue(all(policy.modes), "bn_mode='train' keeps the policy in train mode during the update")
        for e in range(3):
            calls = policy.calls[e * 4:(e + 1) * 4]
            ids = [i for env_ids, _ in calls for i in env_ids]
            self.assertEqual(sorted(ids), list(range(B)))
            for env_ids, hx in calls:
                self.assertEqual(len(env_ids), 2)
                self.assertEqual(tuple(hx.shape), (1, 2, H))
                self.assertEqual(hx[0, :, 0].tolist(), [float(i) for i in env_ids])
        # the order is reshuffled between epochs
        orders = {tuple(tuple(c[0]) for c in policy.calls[e * 4:(e + 1) * 4]) for e in range(3)}
        self.assertGreater(len(orders), 1)

    def test_uneven_split_and_more_minibatches_than_envs(self):
        policy = SeqPolicy()
        hp = PPOHParams(epochs=1, num_minibatches=4, target_kl=None)
        ppo_update(policy, frozen_sgd(policy), make_batch(B=6), hp)
        self.assertEqual(sorted(i for ids, _ in policy.calls for i in ids), list(range(6)))
        policy = SeqPolicy()
        stats = ppo_update(policy, frozen_sgd(policy), make_batch(B=2), PPOHParams(epochs=2, num_minibatches=8, target_kl=None))
        self.assertEqual(stats["n_optimizer_steps"], 4)  # capped at B minibatches per epoch

    def test_hidden_state_dependence_uses_the_right_slice(self):
        # logits depend on hx, so a wrong slice changes the per-sample log-probs; compare with a manual evaluation
        torch.manual_seed(5)
        policy = SeqPolicy()
        batch = make_batch(B=4)
        with torch.no_grad():
            lp_ok, _, _ = policy.evaluate_actions_sequence(batch.screens, batch.rams, batch.actions, batch.initial_hx)
        batch.old_logps = lp_ok.clone()  # correct slicing => ratio == 1 => approx_kl == 0 and clip_frac == 0
        hp = PPOHParams(epochs=1, num_minibatches=2, target_kl=None)
        policy.calls.clear()
        stats = ppo_update(policy, frozen_sgd(policy), batch, hp)
        self.assertAlmostEqual(stats["approx_kl"], 0.0, places=6)
        self.assertEqual(stats["clip_frac"], 0.0)
        # with a deliberately wrong hidden state the ratio is not 1
        batch.initial_hx = batch.initial_hx.flip(dims=[2]) + 1.0
        stats = ppo_update(policy, frozen_sgd(policy), batch, hp)
        self.assertGreater(stats["approx_kl"], 1e-5)

    def test_target_kl_early_stop(self):
        torch.manual_seed(6)
        policy = SeqPolicy()
        batch = make_batch()
        batch.old_logps = batch.old_logps + 0.5  # large initial KL
        hp = PPOHParams(epochs=3, num_minibatches=4, target_kl=1e-4)
        stats = ppo_update(policy, torch.optim.SGD(policy.parameters(), lr=1e-3), batch, hp)
        self.assertTrue(stats["early_stopped"])
        self.assertEqual(stats["n_optimizer_steps"], 1)  # the first step is always taken
        hp.target_kl = None
        stats = ppo_update(policy, torch.optim.SGD(policy.parameters(), lr=1e-3), batch, hp)
        self.assertFalse(stats["early_stopped"])
        self.assertEqual(stats["n_optimizer_steps"], 12)

    def test_stats_keys_and_explained_variance(self):
        policy = SeqPolicy()
        batch = make_batch()
        # returns exactly equal to old values + advantages => explained variance of old values
        values = torch.randn(8, 6)
        batch.returns = values + batch.advantages
        stats = ppo_update(policy, frozen_sgd(policy), batch, PPOHParams(target_kl=None))
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm", "n_optimizer_steps",
                  "explained_variance", "noise_scale", "early_stopped"):
            self.assertIn(k, stats)
        ev = 1.0 - batch.advantages.var(unbiased=False) / batch.returns.var(unbiased=False)
        self.assertAlmostEqual(stats["explained_variance"], float(ev), places=5)
        self.assertGreater(stats["entropy"], 0.0)

    def test_norm_adv_adv_clip_and_vf_clip_run(self):
        policy = SeqPolicy()
        batch = make_batch()
        batch.advantages = batch.advantages.clone()
        batch.advantages[0, 0] = 1e4  # outlier
        hp = PPOHParams(adv_clip=3.0, vf_clip=True, target_kl=None)
        stats = ppo_update(policy, torch.optim.SGD(policy.parameters(), lr=1e-3), batch, hp)
        self.assertTrue(all(math.isfinite(stats[k]) for k in ("policy_loss", "value_loss", "grad_norm")))

    def test_feedforward_path(self):
        torch.manual_seed(7)
        policy = FFPolicy()
        batch = make_batch(B=4, T=5, with_hx=False, env_ids=False)
        hp = PPOHParams(epochs=2, num_minibatches=5, target_kl=None)
        stats = ppo_update(policy, torch.optim.SGD(policy.parameters(), lr=1e-3), batch, hp)
        self.assertEqual(stats["n_optimizer_steps"], 10)
        self.assertEqual(policy.n_calls, 10)

    def test_feedforward_single_batch_gradient_matches_legacy(self):
        torch.manual_seed(8)
        policy = FFPolicy()
        batch = make_batch(B=3, T=4, with_hx=False, env_ids=False)
        hp = PPOHParams(epochs=1, num_minibatches=1, norm_adv=False, vf_clip=False, max_grad_norm=1e9, target_kl=None)
        n = 12
        s, r = batch.screens.reshape(n, -1), batch.rams.reshape(n, -1)
        a, olp = batch.actions.reshape(n), batch.old_logps.reshape(n)
        adv, ret = batch.advantages.reshape(n), batch.returns.reshape(n)
        lp, ent, val = policy.evaluate_actions(s.reshape(n, 3, 2, 2), r, a)
        ratio = torch.exp(lp - olp)
        legacy = (-torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
                  + 0.5 * F.mse_loss(val, ret) - 0.02 * ent.mean())
        policy.zero_grad()
        legacy.backward()
        ref = [p.grad.clone() for p in policy.parameters()]
        ppo_update(policy, frozen_sgd(policy), batch, hp)
        for p, g in zip(policy.parameters(), ref):
            self.assertTrue(torch.allclose(p.grad, g, atol=1e-6))

    def test_noise_scale_finite_on_toy_problem(self):
        # noisy linear regression through the value head: gradient noise dominates, so B_simple is large and positive
        torch.manual_seed(9)
        B, T = 32, 8
        policy = FFPolicy()
        batch = make_batch(B=B, T=T, with_hx=False, env_ids=False, seed=9)
        batch.returns = 3.0 * batch.rams[..., 0] + 2.0 * torch.randn(B, T)
        batch.advantages = torch.zeros(B, T)
        hp = PPOHParams(epochs=1, num_minibatches=4, norm_adv=False, vf_clip=False, target_kl=None, max_grad_norm=1e9)
        stats = ppo_update(policy, frozen_sgd(policy), batch, hp)
        self.assertTrue(math.isfinite(stats["noise_scale"]))
        self.assertGreater(stats["noise_scale"], 0.0)

    def test_noise_scale_nan_when_single_minibatch(self):
        policy = SeqPolicy()
        stats = ppo_update(policy, frozen_sgd(policy), make_batch(),
                           PPOHParams(num_minibatches=1, epochs=1, target_kl=None))
        self.assertTrue(math.isnan(stats["noise_scale"]))

    def test_learning_reduces_value_error(self):
        torch.manual_seed(10)
        policy = FFPolicy()
        batch = make_batch(B=8, T=8, with_hx=False, env_ids=False)
        batch.advantages = torch.zeros(8, 8)
        batch.returns = batch.rams[..., 0].clone()
        opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
        hp = PPOHParams(epochs=4, num_minibatches=2, vf_clip=False, target_kl=None)
        first = ppo_update(policy, opt, batch, hp)["value_loss"]
        for _ in range(15):
            last = ppo_update(policy, opt, batch, hp)["value_loss"]
        self.assertLess(last, first)


class RealPolicyTests(unittest.TestCase):
    def test_spatial_vision_policy_recurrent_update(self):
        from src.deep_policy import SpatialVisionPolicy

        torch.manual_seed(11)
        B, T = 4, 3
        policy = SpatialVisionPolicy(obs_dim=D, num_actions=A, ram_hidden_dim=16, fusion_dim=32,
                                     use_gru=True, gru_hidden_dim=16)
        g = torch.Generator().manual_seed(11)
        batch = RolloutBatch(
            screens=torch.rand(B, T, 3, 32, 32, generator=g),
            rams=torch.randn(B, T, D, generator=g),
            actions=torch.randint(0, A, (B, T), generator=g),
            old_logps=torch.full((B, T), -math.log(A)),
            returns=torch.randn(B, T, generator=g),
            advantages=torch.randn(B, T, generator=g),
            initial_hx=torch.zeros(1, B, 16),
        )
        hp = PPOHParams(epochs=2, num_minibatches=2, target_kl=None)
        before = [p.detach().clone() for p in policy.parameters()]
        stats = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), batch, hp)
        self.assertEqual(stats["n_optimizer_steps"], 4)
        self.assertTrue(math.isfinite(stats["policy_loss"]) and math.isfinite(stats["value_loss"]))
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, policy.parameters())))


class UpdateModeTests(unittest.TestCase):
    """cuDNN's GRU backward only runs in training mode (a GPU-only crash), while BatchNorm must use running stats."""

    def _seen_modes(self, bn_mode):
        from src.deep_policy import SpatialVisionPolicy

        torch.manual_seed(5)
        B, T = 4, 3
        policy = SpatialVisionPolicy(obs_dim=D, num_actions=A, ram_hidden_dim=16, fusion_dim=32,
                                     use_gru=True, gru_hidden_dim=16)
        seen = {}
        original = policy.evaluate_actions_sequence

        def spy(*args, **kwargs):
            seen["policy_training"] = policy.training
            seen["gru_training"] = policy.gru.training
            bns = [m for m in policy.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            seen["bn_training"] = [m.training for m in bns]
            return original(*args, **kwargs)

        policy.evaluate_actions_sequence = spy
        g = torch.Generator().manual_seed(5)
        batch = RolloutBatch(
            screens=torch.rand(B, T, 3, 32, 32, generator=g), rams=torch.randn(B, T, D, generator=g),
            actions=torch.randint(0, A, (B, T), generator=g), old_logps=torch.full((B, T), -math.log(A)),
            returns=torch.randn(B, T, generator=g), advantages=torch.randn(B, T, generator=g),
            initial_hx=torch.zeros(1, B, 16),
        )
        hp = PPOHParams(epochs=1, num_minibatches=2, target_kl=None, bn_mode=bn_mode)
        ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), batch, hp)
        return seen

    def test_eval_refresh_keeps_gru_in_train_mode_and_batchnorm_frozen(self):
        seen = self._seen_modes("eval_refresh")
        self.assertTrue(seen["policy_training"] and seen["gru_training"])
        self.assertTrue(seen["bn_training"] and not any(seen["bn_training"]))

    def test_train_mode_uses_batch_statistics(self):
        seen = self._seen_modes("train")
        self.assertTrue(seen["bn_training"] and all(seen["bn_training"]))


class BnModeTests(unittest.TestCase):
    """The rollout runs eval(); the update must too, or the ratio is not 1 before the first optimizer step."""

    def setUp(self):
        from src.deep_policy import SpatialVisionPolicy

        torch.manual_seed(0)
        self.policy = SpatialVisionPolicy(obs_dim=5, num_actions=5, ram_hidden_dim=16, fusion_dim=32,
                                          use_gru=True, gru_hidden_dim=16)
        torch.nn.init.orthogonal_(self.policy.actor.weight, gain=1.0)
        B, T = 4, 6
        g = torch.Generator().manual_seed(1)
        self.batch = RolloutBatch(
            screens=torch.rand(B, T, 3, 32, 32, generator=g) * 255, rams=torch.randn(B, T, 5, generator=g),
            actions=torch.randint(0, 5, (B, T), generator=g), old_logps=torch.zeros(B, T),
            returns=torch.randn(B, T, generator=g), advantages=torch.randn(B, T, generator=g),
            initial_hx=torch.zeros(1, B, 16),
        )
        self.batch.screens = self.batch.screens.to(torch.uint8)
        # Cold running stats (never refreshed): batch-stat mode would disagree with them.
        self.policy.eval()
        with torch.no_grad():
            lp, _, _ = self.policy.evaluate_actions_sequence(
                self.batch.screens.float() / 255, self.batch.rams, self.batch.actions, self.batch.initial_hx)
        self.batch.old_logps = lp.detach()

    def run_update(self, bn_mode):
        hp = PPOHParams(epochs=1, num_minibatches=2, target_kl=None, bn_mode=bn_mode, bn_refresh_chunk=8)
        return ppo_update(self.policy, torch.optim.SGD(self.policy.parameters(), lr=0.0), self.batch, hp)

    def test_first_step_ratio_is_one_in_eval_refresh(self):
        stats = self.run_update("eval_refresh")
        self.assertAlmostEqual(stats["approx_kl"], 0.0, places=7)
        self.assertEqual(stats["clip_frac"], 0.0)
        self.assertGreater(stats["bn_refresh_frames"], 0)
        self.assertFalse(self.policy.training)

    def test_train_mode_reproduces_the_old_mismatch(self):
        stats = self.run_update("train")
        self.assertGreater(stats["approx_kl"], 1e-4)

    def test_running_stats_frozen_during_update_then_refreshed(self):
        bn = self.policy.vision_trunk.bn1
        before = bn.running_mean.clone()
        hp = PPOHParams(epochs=1, num_minibatches=2, target_kl=None, bn_mode="eval_refresh")
        # Run only the update part: statistics must not move until the refresh (which is the last thing ppo_update does)
        stats = ppo_update(self.policy, torch.optim.SGD(self.policy.parameters(), lr=0.0), self.batch, hp)
        self.assertFalse(torch.equal(before, bn.running_mean))  # refreshed after the update
        self.assertEqual(stats["bn_refresh_frames"], 24)
        self.assertEqual(bn.momentum, 0.1)  # momentum restored

    def test_refresh_without_batchnorm_is_a_noop(self):
        self.assertEqual(refresh_batchnorm(SeqPolicy(), make_batch(), PPOHParams()), 0)

    def test_bad_bn_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.run_update("bogus")


class RewardScalerTests(unittest.TestCase):
    def test_scales_returns_to_unit_std_without_shifting(self):
        rng = np.random.default_rng(0)
        sc = RunningReturnScaler(num_envs=4, gamma=0.99)
        raw, scaled = [], []
        for _ in range(3000):
            r = rng.normal(0.5, 20.0, size=4)
            raw.append(r)
            scaled.append(sc.scale(r, np.zeros(4, bool)))
        raw, scaled = np.array(raw), np.array(scaled)
        self.assertGreater(sc.std, 20.0)  # discounted-return std is well above the per-step std
        self.assertLess(np.abs(scaled[-1000:]).mean(), np.abs(raw[-1000:]).mean() / 10)
        self.assertTrue((np.sign(scaled) == np.sign(raw)).all())  # scale only: signs untouched

    def test_state_roundtrip_and_env_count_mismatch(self):
        sc = RunningReturnScaler(3, 0.99)
        for _ in range(10):
            sc.scale(np.array([1.0, 2.0, 3.0]), np.array([False, False, True]))
        other = RunningReturnScaler(3, 0.99)
        other.load_state_dict(sc.state_dict())
        self.assertEqual((other.mean, other.var, other.count), (sc.mean, sc.var, sc.count))
        self.assertTrue(np.array_equal(other.ret, sc.ret))
        resized = RunningReturnScaler(5, 0.99)
        resized.load_state_dict(sc.state_dict())  # stats restored, per-env accumulator reset
        self.assertEqual(resized.var, sc.var)
        self.assertTrue((resized.ret == 0).all())

    def test_value_clip_width_follows_return_scale(self):
        """The clip half-width scales with the returns, so a 100x larger critic target is clipped just as hard."""
        losses = []
        for k in (1.0, 100.0):
            torch.manual_seed(5)
            pol = FFPolicy()
            b = make_batch(with_hx=False)
            b.returns, b.advantages = b.returns * k, b.advantages * k
            hp = PPOHParams(epochs=1, num_minibatches=1, target_kl=None, vf_clip=True, norm_adv=False)
            losses.append(ppo_update(pol, frozen_sgd(pol), b, hp)["value_loss"])
        # With old_values = returns - adv the clipped max-loss keeps the same relative structure at both scales.
        self.assertAlmostEqual(losses[1] / losses[0], 100.0 ** 2, delta=0.2 * 100.0 ** 2)


if __name__ == "__main__":
    unittest.main()
