"""PPO update step: GAE with truncation bootstrapping, env-wise minibatching, entropy schedule.

Replaces the inline loop in ``tools/train_deep_ppo.py`` (4 full-batch optimizer steps per update).  For a recurrent
policy each minibatch is a subset of *environments* with their full T-step sequences and their own slice of the
initial hidden state, so peak activation memory scales with the minibatch rather than the whole rollout, and the
number of optimizer steps per update is ``epochs * num_minibatches`` instead of 4.  Feed-forward policies shuffle the
flattened ``T * B`` samples instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PPOHParams:
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5  # loss = policy + vf_coef * MSE(value, return) - entropy_coef * entropy
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    epochs: int = 3
    num_minibatches: int = 4
    target_kl: float | None = 0.05  # stop the update once a minibatch's approx KL exceeds this (None disables)
    norm_adv: bool = True
    vf_clip: bool = True
    adv_clip: float | None = None  # clamp normalized advantages to [-adv_clip, adv_clip]
    vf_clip_range: float | None = None  # value clip half-width; None -> clip_ratio


@dataclass
class RolloutBatch:
    screens: torch.Tensor  # (B, T, 3, H, W)  float, or uint8 (scaled by 1/255)
    rams: torch.Tensor  # (B, T, D)
    actions: torch.Tensor  # (B, T)
    old_logps: torch.Tensor  # (B, T)
    returns: torch.Tensor  # (B, T)
    advantages: torch.Tensor  # (B, T)  raw (un-normalized) GAE advantages
    initial_hx: torch.Tensor | None = None  # (1, B, H) hidden state at the start of the rollout


class EntropySchedule:
    """Linear entropy-coefficient schedule over training progress in [0, 1] (clamped)."""

    def __init__(self, start: float = 0.03, end: float = 0.005):
        self.start = float(start)
        self.end = float(end)

    def value(self, progress: float) -> float:
        p = min(1.0, max(0.0, float(progress)))
        return self.start + (self.end - self.start) * p


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor | float,
    gamma: float,
    lam: float,
    truncated: torch.Tensor | None = None,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. All tensors are (T, B); returns ``(returns, advantages)``.

    ``dones[t, b]`` means the episode ended after step t.  ``last_value`` (scalar or (B,)) is V of the observation
    after the final step and bootstraps an unfinished last step.  ``truncated[t, b]`` marks time-limit ends: the
    episode ends (the GAE chain is cut) but the target bootstraps with ``bootstrap_values[t, b]``, the value of the
    final observation, instead of treating the end as terminal.  Truncation takes precedence if both flags are set,
    so ``dones`` may or may not already include the truncated steps.
    """
    T = rewards.shape[0]
    values = values.to(rewards.dtype)
    ended = dones.bool()
    trunc = None
    if truncated is not None:
        trunc = truncated.bool()
        ended = ended | trunc
        if bootstrap_values is None:
            raise ValueError("bootstrap_values is required when truncated is given")
        bootstrap_values = bootstrap_values.to(rewards.dtype)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    if not torch.is_tensor(last_value):
        last_value = torch.full_like(rewards[0], float(last_value))
    last_value = last_value.to(rewards.dtype)
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else last_value
        non_terminal = (~ended[t]).to(rewards.dtype)
        boot = next_val * non_terminal
        if trunc is not None:
            boot = boot + bootstrap_values[t] * trunc[t].to(rewards.dtype)
        delta = rewards[t] + gamma * boot - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    return advantages + values, advantages


def _noise_scale(sum_g: torch.Tensor, sum_sq: float, n_total: int, n_small: float) -> float:
    """Simple gradient noise scale B_simple = tr(Sigma) / |G|^2 (McCandlish et al. 2018, appendix A).

    Compares the mean squared norm of minibatch gradients (batch ``n_small``) with the squared norm of their average
    (batch ``n_total``).  ``sum_g`` is the sample-weighted mean gradient, ``sum_sq`` the weighted mean of |g_i|^2.
    """
    if n_total <= n_small:
        return float("nan")
    g_small = sum_sq
    g_big = float(sum_g.pow(2).sum())
    true_sq = (n_total * g_big - n_small * g_small) / (n_total - n_small)
    trace = (g_small - g_big) / (1.0 / n_small - 1.0 / n_total)
    if not (math.isfinite(true_sq) and math.isfinite(trace)) or true_sq <= 0.0 or trace < 0.0:
        return float("nan")
    return trace / true_sq


def _flat_grad(params) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1) for p in params])


def ppo_update(
    policy,
    optimizer,
    batch: RolloutBatch,
    hp: PPOHParams,
    entropy_coef: float | None = None,
    *,
    generator: torch.Generator | None = None,
) -> dict:
    """Run PPO epochs over ``batch`` and return per-update statistics.

    Recurrent policies (``policy.use_gru``) are minibatched by environment: every epoch the envs are shuffled and
    split into ``num_minibatches`` groups; each group keeps its full T-step sequences and its own
    ``initial_hx[:, idx, :]``.  Feed-forward policies shuffle the flattened T*B samples.  One optimizer step per
    minibatch, so ``n_optimizer_steps = epochs * num_minibatches`` (fewer if ``target_kl`` stops the update).

    Advantages are normalized over the whole batch before minibatching.  Gradient norms are pre-clip.  The
    ``value_loss`` stat is the plain MSE (the loss uses ``vf_coef * MSE``).  Note that the policy runs in train mode
    here (BatchNorm batch statistics) while rollouts use eval mode, so the first minibatch's ratio is not exactly 1.
    """
    ent_coef = hp.entropy_coef if entropy_coef is None else float(entropy_coef)
    params = [p for p in policy.parameters() if p.requires_grad]
    device = params[0].device
    recurrent = bool(getattr(policy, "use_gru", False))

    B, T = batch.actions.shape[:2]
    adv_raw = batch.advantages.detach()
    returns = batch.returns.detach()
    old_values = returns - adv_raw  # returns = advantages + values in GAE
    var_ret = returns.var(unbiased=False)
    explained_var = (
        float(1.0 - (returns - old_values).var(unbiased=False) / var_ret) if float(var_ret) > 1e-12 else float("nan")
    )

    adv = adv_raw
    if hp.norm_adv:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    if hp.adv_clip is not None:
        adv = adv.clamp(-hp.adv_clip, hp.adv_clip)
    vf_range = hp.clip_ratio if hp.vf_clip_range is None else hp.vf_clip_range

    def to_dev(x: torch.Tensor, is_screen: bool = False) -> torch.Tensor:
        x = x.to(device)
        if is_screen and x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        return x

    if recurrent:
        n_units = B
        samples_per_unit = T
    else:
        n_units = B * T
        samples_per_unit = 1
    n_mb = max(1, min(int(hp.num_minibatches), n_units))
    n_total = B * T

    if not recurrent:
        flat = lambda x: x.reshape(n_total, *x.shape[2:])  # noqa: E731
        f_screens, f_rams = flat(batch.screens), flat(batch.rams)
        f_actions, f_logps = flat(batch.actions), flat(batch.old_logps)
        f_returns, f_adv, f_old_v = flat(returns), flat(adv), flat(old_values)

    policy.train()
    pol_l, val_l, ent_l, kl_l, clip_l, gn_l = [], [], [], [], [], []
    n_steps = 0
    early_stopped = False
    # noise scale accumulators (epoch 0 only)
    g_sum = None
    g_sq = 0.0
    g_n = 0
    mb_sizes: list[int] = []

    for epoch in range(int(hp.epochs)):
        perm = torch.randperm(n_units, generator=generator)
        groups = torch.tensor_split(perm, n_mb)
        for idx in groups:
            if idx.numel() == 0:
                continue
            if recurrent:
                s = to_dev(batch.screens[idx], True)
                r = to_dev(batch.rams[idx])
                a = to_dev(batch.actions[idx])
                old_lp = to_dev(batch.old_logps[idx])
                mb_adv, mb_ret, mb_old_v = to_dev(adv[idx]), to_dev(returns[idx]), to_dev(old_values[idx])
                hx = None if batch.initial_hx is None else to_dev(batch.initial_hx[:, idx, :])
                new_lp, entropy, v_pred = policy.evaluate_actions_sequence(s, r, a, hx)
            else:
                s = to_dev(f_screens[idx], True)
                r = to_dev(f_rams[idx])
                a = to_dev(f_actions[idx])
                old_lp = to_dev(f_logps[idx])
                mb_adv, mb_ret, mb_old_v = to_dev(f_adv[idx]), to_dev(f_returns[idx]), to_dev(f_old_v[idx])
                new_lp, entropy, v_pred = policy.evaluate_actions(s, r, a)

            log_ratio = new_lp - old_lp
            ratio = torch.exp(log_ratio)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - hp.clip_ratio, 1.0 + hp.clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            if hp.vf_clip:
                v_clipped = mb_old_v + torch.clamp(v_pred - mb_old_v, -vf_range, vf_range)
                value_loss = torch.max((v_pred - mb_ret) ** 2, (v_clipped - mb_ret) ** 2).mean()
            else:
                value_loss = F.mse_loss(v_pred, mb_ret)
            ent_mean = entropy.mean()
            loss = policy_loss + hp.vf_coef * value_loss - ent_coef * ent_mean

            with torch.no_grad():
                approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                clip_frac = float((torch.abs(ratio - 1.0) > hp.clip_ratio).float().mean())

            # KL guard: judge this minibatch before stepping; always take the very first step.
            if hp.target_kl is not None and n_steps > 0 and approx_kl > hp.target_kl:
                early_stopped = True
                kl_l.append(approx_kl)
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 0 and n_mb > 1:
                gflat = _flat_grad(params)
                n_i = idx.numel() * samples_per_unit
                g_sum = gflat * n_i if g_sum is None else g_sum + gflat * n_i
                g_sq += float(gflat.pow(2).sum()) * n_i
                g_n += n_i
                mb_sizes.append(n_i)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, hp.max_grad_norm)
            optimizer.step()
            n_steps += 1

            pol_l.append(policy_loss.item())
            val_l.append(value_loss.item())
            ent_l.append(ent_mean.item())
            kl_l.append(approx_kl)
            clip_l.append(clip_frac)
            gn_l.append(float(grad_norm.detach()))
        if early_stopped:
            break

    noise = float("nan")
    if g_sum is not None and g_n > 0 and len(mb_sizes) > 1:
        noise = _noise_scale(g_sum / g_n, g_sq / g_n, g_n, g_n / len(mb_sizes))

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "policy_loss": mean(pol_l),
        "value_loss": mean(val_l),
        "entropy": mean(ent_l),
        "approx_kl": mean(kl_l),
        "clip_frac": mean(clip_l),
        "grad_norm": mean(gn_l),
        "n_optimizer_steps": n_steps,
        "explained_variance": explained_var,
        "noise_scale": noise,
        "early_stopped": early_stopped,
    }
