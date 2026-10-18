"""PPO policy models (Linear and 2-layer MLP). Weights start random and update only by playing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os

import numpy as np

from src.rl_env import MASK_START, N_ACTIONS, OBS_DIM


@dataclass
class PpoUpdateStats:
    """Diagnostics from one ``ppo_update`` call."""

    weight_change: float
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    mean_entropy: float = 0.0
    value_mse: float = 0.0

    def __float__(self) -> float:
        return self.weight_change

    def __gt__(self, other: object) -> bool:
        return self.weight_change > float(other)

    def __ge__(self, other: object) -> bool:
        return self.weight_change >= float(other)


def _batch_policy_stats(
    policy: Policy | MlpPolicy,
    batch: dict,
    *,
    clip: float,
) -> tuple[float, float, float, float]:
    obs = np.stack(batch["obs"])
    actions = np.asarray(batch["actions"])
    old_logp = np.asarray(batch["logp"], dtype=np.float64)
    returns = np.asarray(batch["ret"], dtype=np.float64)
    kls: list[float] = []
    clips: list[float] = []
    ents: list[float] = []
    value_sq: list[float] = []
    for i in range(len(actions)):
        probabilities = policy.probs(obs[i])
        logp = float(np.log(probabilities[actions[i]] + 1e-8))
        ratio = float(np.exp(logp - old_logp[i]))
        kls.append(old_logp[i] - logp)
        clips.append(float(ratio < 1.0 - clip or ratio > 1.0 + clip))
        ents.append(-float(np.sum(probabilities * np.log(probabilities + 1e-8))))
        err = returns[i] - policy.value(obs[i])
        value_sq.append(float(err * err))
    return (
        float(np.mean(kls)),
        float(np.mean(clips)),
        float(np.mean(ents)),
        float(np.mean(value_sq)),
    )


def mask_logits(logits: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Remove moves the observation flags as blocked. Never masks every move."""
    if obs.shape[-1] < MASK_START + 4:
        return logits
    blocked = obs[MASK_START : MASK_START + 4] > 0.5
    if not blocked.any() or blocked.all():
        return logits
    masked = np.array(logits, dtype=np.float64, copy=True)
    masked[:4][blocked] = -1e9
    return masked


@dataclass
class Policy:
    """Linear Softmax policy and value function."""

    rng: np.random.Generator
    obs_dim: int = OBS_DIM
    n_actions: int = N_ACTIONS
    w: np.ndarray | None = None
    b: np.ndarray | None = None
    vw: np.ndarray | None = None
    vb: float = 0.0

    def __post_init__(self):
        if self.w is None:
            self.w = self.rng.normal(0.0, 0.01, size=(self.n_actions, self.obs_dim))
        if self.b is None:
            self.b = np.zeros(self.n_actions)
        if self.vw is None:
            self.vw = self.rng.normal(0.0, 0.01, size=self.obs_dim)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        logits = mask_logits(self.w @ obs + self.b, obs)
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def value(self, obs: np.ndarray) -> float:
        return float(self.vw @ obs + self.vb)

    def act(self, obs: np.ndarray) -> tuple[int, float]:
        probabilities = self.probs(obs)
        action = int(self.rng.choice(self.n_actions, p=probabilities))
        return action, float(np.log(probabilities[action] + 1e-8))

    def entropy(self, obs: np.ndarray) -> float:
        probabilities = self.probs(obs)
        return -float(np.sum(probabilities * np.log(probabilities + 1e-8)))

    def save(self, path: str) -> None:
        np.savez(path, w=self.w, b=self.b, vw=self.vw, vb=np.asarray(self.vb), model_type="linear")

    def load(self, path: str) -> None:
        data = np.load(path)
        w = data["w"]
        vw = data["vw"]
        self.b = data["b"]
        self.vb = float(data["vb"])
        # Support loading checkpoints with different obs_dim (e.g. legacy 13-dim to 17-dim)
        if w.shape[1] < self.obs_dim:
            pad = self.obs_dim - w.shape[1]
            self.w = np.pad(w, ((0, 0), (0, pad)))
            self.vw = np.pad(vw, (0, pad))
        elif w.shape[1] > self.obs_dim:
            self.w = w[:, : self.obs_dim]
            self.vw = vw[: self.obs_dim]
        else:
            self.w = w
            self.vw = vw
        self.n_actions = self.w.shape[0]

    def get_weights(self) -> dict[str, Any]:
        return {
            "w": np.array(self.w, copy=True),
            "b": np.array(self.b, copy=True),
            "vw": np.array(self.vw, copy=True),
            "vb": float(self.vb),
            "model_type": "linear",
        }

    def set_weights(self, weights: dict[str, Any]) -> None:
        self.w = np.array(weights["w"], copy=True)
        self.b = np.array(weights["b"], copy=True)
        self.vw = np.array(weights["vw"], copy=True)
        self.vb = float(weights["vb"])


@dataclass
class MlpPolicy:
    """Two-layer non-linear MLP policy with tanh activations."""

    rng: np.random.Generator
    obs_dim: int = OBS_DIM
    hidden_dim: int = 64
    n_actions: int = N_ACTIONS
    w1: np.ndarray | None = None
    b1: np.ndarray | None = None
    w2: np.ndarray | None = None
    b2: np.ndarray | None = None
    vw: np.ndarray | None = None
    vb: float = 0.0

    def __post_init__(self):
        scale1 = np.sqrt(2.0 / (self.obs_dim + self.hidden_dim))
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.n_actions))
        if self.w1 is None:
            self.w1 = self.rng.normal(0.0, scale1, size=(self.hidden_dim, self.obs_dim))
        if self.b1 is None:
            self.b1 = np.zeros(self.hidden_dim)
        if self.w2 is None:
            self.w2 = self.rng.normal(0.0, scale2, size=(self.n_actions, self.hidden_dim))
        if self.b2 is None:
            self.b2 = np.zeros(self.n_actions)
        if self.vw is None:
            self.vw = self.rng.normal(0.0, scale1, size=self.hidden_dim)

    def forward_hidden(self, obs: np.ndarray) -> np.ndarray:
        return np.tanh(self.w1 @ obs + self.b1)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        hidden = self.forward_hidden(obs)
        logits = mask_logits(self.w2 @ hidden + self.b2, obs)
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def value(self, obs: np.ndarray) -> float:
        hidden = self.forward_hidden(obs)
        return float(self.vw @ hidden + self.vb)

    def act(self, obs: np.ndarray) -> tuple[int, float]:
        probabilities = self.probs(obs)
        action = int(self.rng.choice(self.n_actions, p=probabilities))
        return action, float(np.log(probabilities[action] + 1e-8))

    def entropy(self, obs: np.ndarray) -> float:
        probabilities = self.probs(obs)
        return -float(np.sum(probabilities * np.log(probabilities + 1e-8)))

    def save(self, path: str) -> None:
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            vw=self.vw,
            vb=np.asarray(self.vb),
            model_type="mlp",
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        if "w1" in data:
            w1 = data["w1"]
            if w1.shape[1] < self.obs_dim:
                pad = self.obs_dim - w1.shape[1]
                self.w1 = np.pad(w1, ((0, 0), (0, pad)))
            else:
                self.w1 = w1[:, : self.obs_dim]
            self.b1 = data["b1"]
            self.w2 = data["w2"]
            self.b2 = data["b2"]
            self.vw = data["vw"]
            self.vb = float(data["vb"])
        else:
            # Fallback initialization if a linear checkpoint was provided
            linear = Policy(rng=self.rng, obs_dim=self.obs_dim)
            linear.load(path)
            self.b2 = linear.b.copy()

    def get_weights(self) -> dict[str, Any]:
        return {
            "w1": np.array(self.w1, copy=True),
            "b1": np.array(self.b1, copy=True),
            "w2": np.array(self.w2, copy=True),
            "b2": np.array(self.b2, copy=True),
            "vw": np.array(self.vw, copy=True),
            "vb": float(self.vb),
            "model_type": "mlp",
        }

    def set_weights(self, weights: dict[str, Any]) -> None:
        self.w1 = np.array(weights["w1"], copy=True)
        self.b1 = np.array(weights["b1"], copy=True)
        self.w2 = np.array(weights["w2"], copy=True)
        self.b2 = np.array(weights["b2"], copy=True)
        self.vw = np.array(weights["vw"], copy=True)
        self.vb = float(weights["vb"])


def _mlp_ppo_update(
    policy: MlpPolicy,
    batch: dict,
    *,
    clip: float = 0.2,
    epochs: int = 4,
    lr: float = 0.05,
    entropy_coef: float = 0.01,
    batch_size: int | None = None,
) -> PpoUpdateStats:
    obs = np.stack(batch["obs"])
    actions = np.asarray(batch["actions"])
    old_logp = np.asarray(batch["logp"])
    adv = np.asarray(batch["adv"], dtype=np.float64)
    returns = np.asarray(batch["ret"], dtype=np.float64)
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    before_w2 = policy.w2.copy()
    order = np.arange(len(actions))
    size = len(actions) if batch_size is None else max(1, batch_size)
    for _ in range(epochs):
        if batch_size is not None:
            policy.rng.shuffle(order)
        for start in range(0, len(actions), size):
            chosen = order[start : start + size]
            dw1 = np.zeros_like(policy.w1)
            db1 = np.zeros_like(policy.b1)
            dw2 = np.zeros_like(policy.w2)
            db2 = np.zeros_like(policy.b2)
            dvw = np.zeros_like(policy.vw)
            dvb = 0.0
            for i in chosen:
                h = policy.forward_hidden(obs[i])
                logits = mask_logits(policy.w2 @ h + policy.b2, obs[i])
                logits = logits - logits.max()
                exp_logits = np.exp(logits)
                probabilities = exp_logits / exp_logits.sum()

                logp = np.log(probabilities[actions[i]] + 1e-8)
                ratio = np.exp(logp - old_logp[i])
                clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip)
                weight = adv[i]

                # Policy objective gradient
                grad_logits = np.zeros_like(probabilities)
                if ratio * weight <= clipped * weight:
                    g = -probabilities
                    g[actions[i]] += 1.0
                    grad_logits += (weight * ratio) * g

                # Entropy bonus gradient
                log_probs = np.log(probabilities + 1e-8)
                entropy = -float(np.sum(probabilities * log_probs))
                grad_entropy = -probabilities * (log_probs + entropy)
                grad_logits += entropy_coef * grad_entropy

                # Layer 2 gradients
                dw2 += np.outer(grad_logits, h)
                db2 += grad_logits
                dh_actor = policy.w2.T @ grad_logits

                # Critic value gradient
                value_pred = float(policy.vw @ h + policy.vb)
                value_error = returns[i] - value_pred
                dvw += value_error * h
                dvb += value_error
                dh_critic = value_error * policy.vw

                # Backprop to Layer 1 through tanh activation
                dh = dh_actor + dh_critic
                dh_pre = dh * (1.0 - h**2)
                dw1 += np.outer(dh_pre, obs[i])
                db1 += dh_pre

            scale = lr / len(chosen)
            policy.w1 += scale * dw1
            policy.b1 += scale * db1
            policy.w2 += scale * dw2
            policy.b2 += scale * db2
            policy.vw += scale * dvw
            policy.vb += scale * dvb

    weight_change = float(np.mean(np.abs(policy.w2 - before_w2)))
    approx_kl, clip_frac, mean_entropy, value_mse = _batch_policy_stats(
        policy, batch, clip=clip
    )
    return PpoUpdateStats(
        weight_change=weight_change,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        mean_entropy=mean_entropy,
        value_mse=value_mse,
    )


def ppo_update(
    policy: Policy | MlpPolicy,
    batch: dict,
    *,
    clip: float = 0.2,
    epochs: int = 4,
    lr: float = 0.05,
    entropy_coef: float = 0.01,
    batch_size: int | None = None,
) -> PpoUpdateStats:
    """One clipped policy update plus an entropy bonus."""
    if isinstance(policy, MlpPolicy):
        return _mlp_ppo_update(
            policy,
            batch,
            clip=clip,
            epochs=epochs,
            lr=lr,
            entropy_coef=entropy_coef,
            batch_size=batch_size,
        )

    obs = np.stack(batch["obs"])
    actions = np.asarray(batch["actions"])
    old_logp = np.asarray(batch["logp"])
    adv = np.asarray(batch["adv"], dtype=np.float64)
    returns = np.asarray(batch["ret"], dtype=np.float64)
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    before = policy.w.copy()
    order = np.arange(len(actions))
    size = len(actions) if batch_size is None else max(1, batch_size)
    for _ in range(epochs):
        if batch_size is not None:
            policy.rng.shuffle(order)
        for start in range(0, len(actions), size):
            chosen = order[start : start + size]
            dw = np.zeros_like(policy.w)
            db = np.zeros_like(policy.b)
            dvw = np.zeros_like(policy.vw)
            dvb = 0.0
            for i in chosen:
                probabilities = policy.probs(obs[i])
                logp = np.log(probabilities[actions[i]] + 1e-8)
                ratio = np.exp(logp - old_logp[i])
                clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip)
                weight = adv[i]
                if ratio * weight <= clipped * weight:
                    grad_logits = -probabilities
                    grad_logits[actions[i]] += 1.0
                    scale = weight * ratio
                    dw += scale * np.outer(grad_logits, obs[i])
                    db += scale * grad_logits
                log_probs = np.log(probabilities + 1e-8)
                entropy = -float(np.sum(probabilities * log_probs))
                grad = -probabilities * (log_probs + entropy)
                dw += entropy_coef * np.outer(grad, obs[i])
                db += entropy_coef * grad
                value_error = returns[i] - policy.value(obs[i])
                dvw += value_error * obs[i]
                dvb += value_error
            scale = lr / len(chosen)
            policy.w += scale * dw
            policy.b += scale * db
            policy.vw += scale * dvw
            policy.vb += scale * dvb
    weight_change = float(np.mean(np.abs(policy.w - before)))
    approx_kl, clip_frac, mean_entropy, value_mse = _batch_policy_stats(
        policy, batch, clip=clip
    )
    return PpoUpdateStats(
        weight_change=weight_change,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        mean_entropy=mean_entropy,
        value_mse=value_mse,
    )


def collect_rollout(
    env,
    policy: Policy | MlpPolicy,
    steps: int,
    gamma: float = 0.99,
    on_step: Callable[[dict], None] | None = None,
) -> dict:
    obs = env.reset()
    batch = {"obs": [], "actions": [], "logp": [], "rew": []}
    for step_idx in range(steps):
        probabilities = policy.probs(obs)
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-8)))
        val = policy.value(obs)
        action, logp = policy.act(obs)
        nxt, reward, done, info = env.step(action)
        batch["obs"].append(obs)
        batch["actions"].append(action)
        batch["logp"].append(logp)
        batch["rew"].append(reward)
        if on_step is not None:
            badges = 0
            map_id = None
            reader = getattr(env, "reader", None)
            machine = getattr(env, "machine", None)
            if reader is not None and machine is not None:
                state = reader.parse(machine.frame)
                badges = int(getattr(state, "badges", 0))
                map_id = (int(state.map_group), int(state.map_number))
            on_step(
                {
                    "step": step_idx,
                    "obs": obs,
                    "action": action,
                    "logp": logp,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "entropy": entropy,
                    "value": val,
                    "probabilities": probabilities,
                    "badges": badges,
                    "map_id": map_id,
                }
            )
        obs = env.reset() if done else nxt
    rewards = np.asarray(batch["rew"], dtype=np.float64)
    returns = np.zeros_like(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    values = np.asarray([policy.value(o) for o in batch["obs"]])
    batch["ret"] = returns
    batch["adv"] = returns - values
    return batch


def load_policy(path: str, rng: np.random.Generator | None = None) -> Policy | MlpPolicy:
    """Load a policy from npz, automatically selecting MlpPolicy or Policy."""
    generator = np.random.default_rng(0) if rng is None else rng
    p = path
    if not os.path.exists(p):
        fallback = os.path.join("models", os.path.basename(p))
        if os.path.exists(fallback):
            p = fallback
    data = np.load(p)
    if "w1" in data or str(data.get("model_type", "")) == "mlp":
        hidden_dim = int(data.get("hidden_dim", data["w1"].shape[0]))
        # Always build at the current width. load() zero-pads older, narrower checkpoints.
        policy: Policy | MlpPolicy = MlpPolicy(rng=generator, obs_dim=OBS_DIM, hidden_dim=hidden_dim)
    else:
        policy = Policy(rng=generator, obs_dim=OBS_DIM)
    policy.load(p)
    return policy

