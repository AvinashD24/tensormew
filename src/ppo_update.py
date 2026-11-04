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

import numpy as np
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
    vf_clip_range: float | None = None  # value clip half-width (in units of the return scale); None -> clip_ratio
    # BatchNorm handling: rollouts run in eval mode (running statistics), so the update must too or the ratio is not 1
    # before the first step ("eval_refresh": update in eval mode, then refresh running stats with a no-grad train-mode
    # pass over a frame subsample).  "train" reproduces the old behaviour (batch statistics during the update).
    bn_mode: str = "eval_refresh"
    bn_refresh_frames: int = 256  # frames (subsample of the rollout) used to refresh the running statistics
    bn_refresh_chunk: int = 64  # frames per refresh forward pass (bounds memory)
    bn_refresh_momentum: float = 0.3  # running-stat momentum per refresh chunk
    # Auxiliary prediction loss (src.aux_tasks), recurrent policies built with aux_heads only; 0 = off (default).
    aux_coef: float = 0.0
    aux_horizon: int = 8


@dataclass
class RolloutBatch:
    screens: torch.Tensor  # (B, T, 3, H, W)  float, or uint8 (scaled by 1/255)
    rams: torch.Tensor  # (B, T, D)
    actions: torch.Tensor  # (B, T)
    old_logps: torch.Tensor  # (B, T)
    returns: torch.Tensor  # (B, T)
    advantages: torch.Tensor  # (B, T)  raw (un-normalized) GAE advantages
    initial_hx: torch.Tensor | None = None  # (1, B, H) hidden state at the start of the rollout
    dones: torch.Tensor | None = None  # (B, T) bool: episode ended after step t -> hidden state reset before t + 1
    masks: torch.Tensor | None = None  # (B, T, A) bool valid-action mask stored at rollout time (True = allowed); None = no masking


class EntropySchedule:
    """Linear entropy-coefficient schedule over training progress in [0, 1] (clamped)."""

    def __init__(self, start: float = 0.03, end: float = 0.005):
        self.start = float(start)
        self.end = float(end)

    def value(self, progress: float) -> float:
        p = min(1.0, max(0.0, float(progress)))
        return self.start + (self.end - self.start) * p


class RunningReturnScaler:
    """VecNormalize-style reward scaling: divide rewards by the running std of the per-env discounted return.

    Scale only (no mean shift), so the sign and zero point of the reward are untouched.  ``scale(rewards, dones)``
    updates the statistics with this step's raw rewards and returns ``rewards / std``.  ``state_dict`` /
    ``load_state_dict`` make it resumable; the per-env return accumulator is saved only if the env count matches.
    """

    def __init__(self, num_envs: int, gamma: float, epsilon: float = 1e-8):
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.ret = np.zeros(num_envs, dtype=np.float64)
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    @property
    def std(self) -> float:
        return math.sqrt(self.var + self.epsilon)

    def update(self, rewards, dones) -> None:
        rewards = np.asarray(rewards, dtype=np.float64)
        self.ret = self.ret * self.gamma + rewards
        batch_mean, batch_var, n = float(self.ret.mean()), float(self.ret.var()), self.ret.size
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m2 = self.var * self.count + batch_var * n + delta * delta * self.count * n / total
        self.var, self.count = m2 / total, total
        self.ret[np.asarray(dones, dtype=bool)] = 0.0

    def scale(self, rewards, dones) -> np.ndarray:
        """Update with the raw ``rewards`` and return them divided by the running return std."""
        self.update(rewards, dones)
        return (np.asarray(rewards, dtype=np.float64) / self.std).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count, "ret": self.ret.tolist(), "gamma": self.gamma}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.var, self.count = float(state["mean"]), float(state["var"]), float(state["count"])
        ret = np.asarray(state.get("ret", []), dtype=np.float64)
        if ret.shape == self.ret.shape:
            self.ret = ret


def refresh_batchnorm(policy, batch: RolloutBatch, hp: PPOHParams, generator: torch.Generator | None = None) -> int:
    """Refresh BatchNorm running statistics with a no-grad train-mode pass over a subsample of the rollout frames.

    Chunked so peak memory is ``bn_refresh_chunk`` frames.  Leaves the policy in eval mode.  Returns the number of
    frames used (0 when the policy has no BatchNorm layers).
    """
    bns = [m for m in policy.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    policy.eval()
    if not bns:
        return 0
    B, T = batch.actions.shape[:2]
    n_total = B * T
    n_use = min(n_total, max(1, int(hp.bn_refresh_frames)))
    pick = torch.randperm(n_total, generator=generator)[:n_use]
    screens = batch.screens.reshape(n_total, *batch.screens.shape[2:])
    device = next(policy.parameters()).device
    trunk = getattr(policy, "vision_trunk", None)
    rams = batch.rams.reshape(n_total, *batch.rams.shape[2:])
    old_momentum = [m.momentum for m in bns]
    for m in bns:
        m.train()
        m.momentum = float(hp.bn_refresh_momentum)
    try:
        with torch.no_grad():
            for chunk in torch.split(pick, max(1, int(hp.bn_refresh_chunk))):
                if chunk.numel() < 2:
                    continue  # a single frame would give degenerate batch statistics
                x = screens[chunk].to(device)
                x = x.float().div_(255.0) if x.dtype == torch.uint8 else x.float()
                if trunk is not None:
                    trunk(x)
                else:
                    policy.evaluate_actions(x, rams[chunk].to(device), batch.actions.reshape(n_total)[chunk].to(device))
    finally:
        for m, mom in zip(bns, old_momentum):
            m.momentum = mom
        policy.eval()
    return n_use


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
    value_scale: float | None = None,
) -> dict:
    """Run PPO epochs over ``batch`` and return per-update statistics.

    Recurrent policies (``policy.use_gru``) are minibatched by environment: every epoch the envs are shuffled and
    split into ``num_minibatches`` groups; each group keeps its full T-step sequences and its own
    ``initial_hx[:, idx, :]``.  Feed-forward policies shuffle the flattened T*B samples.  One optimizer step per
    minibatch, so ``n_optimizer_steps = epochs * num_minibatches`` (fewer if ``target_kl`` stops the update).

    Advantages are normalized over the whole batch before minibatching.  Gradient norms are pre-clip.  The
    ``value_loss`` stat is the plain MSE (the loss uses ``vf_coef * MSE``).

    BatchNorm: with ``hp.bn_mode == "eval_refresh"`` (default) the update runs in eval mode like the rollouts, so the
    first minibatch's ratio is exactly 1, and the running statistics are refreshed afterwards by ``refresh_batchnorm``.
    ``"train"`` keeps the old behaviour (batch statistics, ratio != 1 before the first step).

    Value clipping is scale-aware: the half-width is ``vf_clip_range`` (default ``clip_ratio``) times ``value_scale``,
    the typical magnitude of the returns (default: their std over the batch, floored at 1e-3).  With reward scaling
    the returns have std ~1, so this matches the old absolute 0.2; with raw rewards it widens with the return scale.
    """
    if hp.bn_mode not in ("eval_refresh", "train"):
        raise ValueError(f"bn_mode must be 'eval_refresh' or 'train' (got {hp.bn_mode!r})")
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
    if value_scale is None:
        value_scale = max(float(returns.std(unbiased=False)), 1e-3)
    vf_range = (hp.clip_ratio if hp.vf_clip_range is None else hp.vf_clip_range) * float(value_scale)

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
        f_masks = flat(batch.masks) if batch.masks is not None else None

    # cuDNN's GRU only runs its backward pass in training mode (a GPU-only error: "cudnn RNN backward can only be
    # called in training mode"), so keep the module in train mode and freeze just BatchNorm for eval_refresh. There
    # is no dropout, so this matches eval-mode numerics exactly.
    policy.train()
    if hp.bn_mode != "train":
        for m in policy.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    pol_l, val_l, ent_l, kl_l, clip_l, gn_l = [], [], [], [], [], []
    use_aux = bool(recurrent and hp.aux_coef > 0 and getattr(policy, "aux_enabled", False))
    aux_l: dict[str, list[float]] = {"map": [], "hp": [], "battle": []}
    if use_aux:
        from src import aux_tasks  # lazy: pulls in the observation layout
    n_steps = 0
    first_kl = None  # approx KL of the very first minibatch, before any optimizer step (0 when rollout and update agree)
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
                mb_dones = None if batch.dones is None else to_dev(batch.dones[idx])
                aux_raw = None
                # The rollout's valid-action mask (src.battle_mask) enters the re-evaluation too; only passed when present.
                mkw = {} if batch.masks is None else {"action_mask": to_dev(batch.masks[idx])}
                if use_aux:
                    new_lp, entropy, v_pred, aux_raw = policy.evaluate_actions_sequence(
                        s, r, a, hx, dones=mb_dones, return_aux=True, **mkw
                    )
                elif mb_dones is None:
                    new_lp, entropy, v_pred = policy.evaluate_actions_sequence(s, r, a, hx, **mkw)
                else:
                    new_lp, entropy, v_pred = policy.evaluate_actions_sequence(s, r, a, hx, dones=mb_dones, **mkw)
            else:
                s = to_dev(f_screens[idx], True)
                r = to_dev(f_rams[idx])
                a = to_dev(f_actions[idx])
                old_lp = to_dev(f_logps[idx])
                mb_adv, mb_ret, mb_old_v = to_dev(f_adv[idx]), to_dev(f_returns[idx]), to_dev(f_old_v[idx])
                if f_masks is None:
                    new_lp, entropy, v_pred = policy.evaluate_actions(s, r, a)
                else:
                    new_lp, entropy, v_pred = policy.evaluate_actions(s, r, a, action_mask=to_dev(f_masks[idx]))

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
            aux_parts = None
            if use_aux and aux_raw is not None:
                # Auxiliary prediction loss on the shared recurrent features (UNREAL-style). The targets are read from
                # the stored RAM observations k steps ahead; the gradient flows into the whole trunk, not into the
                # action / value heads.
                aux_total, aux_parts = aux_tasks.aux_loss(
                    aux_tasks.split_outputs(aux_raw), aux_tasks.make_targets(r, mb_dones, hp.aux_horizon)
                )
                loss = loss + hp.aux_coef * aux_total

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

            if first_kl is None:
                first_kl = approx_kl
            pol_l.append(policy_loss.item())
            val_l.append(value_loss.item())
            ent_l.append(ent_mean.item())
            if aux_parts is not None:
                for k_, v_ in aux_parts.items():
                    aux_l[k_].append(v_)
            kl_l.append(approx_kl)
            clip_l.append(clip_frac)
            gn_l.append(float(grad_norm.detach()))
        if early_stopped:
            break

    bn_frames = 0
    if hp.bn_mode == "eval_refresh":
        bn_frames = refresh_batchnorm(policy, batch, hp, generator)

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
        "bn_refresh_frames": bn_frames,
        "approx_kl_first": float("nan") if first_kl is None else first_kl,
        "aux_map_loss": mean(aux_l["map"]),
        "aux_hp_loss": mean(aux_l["hp"]),
        "aux_battle_loss": mean(aux_l["battle"]),
    }
