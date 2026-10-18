"""Small PPO policy. Weights start random and change only by playing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.rl_env import N_ACTIONS, OBS_DIM


@dataclass
class Policy:
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
        logits = self.w @ obs + self.b
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def value(self, obs: np.ndarray) -> float:
        return float(self.vw @ obs + self.vb)

    def act(self, obs: np.ndarray) -> tuple[int, float]:
        probabilities = self.probs(obs)
        action = int(self.rng.choice(self.n_actions, p=probabilities))
        return action, float(np.log(probabilities[action] + 1e-8))

    def save(self, path: str) -> None:
        np.savez(path, w=self.w, b=self.b, vw=self.vw, vb=np.asarray(self.vb))

    def load(self, path: str) -> None:
        data = np.load(path)
        self.w = data["w"]
        self.b = data["b"]
        self.vw = data["vw"]
        self.vb = float(data["vb"])


def ppo_update(
    policy: Policy,
    batch: dict,
    *,
    clip: float = 0.2,
    epochs: int = 4,
    lr: float = 0.05,
    entropy_coef: float = 0.01,
    batch_size: int | None = None,
) -> float:
    """One clipped policy update plus an entropy bonus. Returns mean absolute weight change."""
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
    return float(np.mean(np.abs(policy.w - before)))


def collect_rollout(env, policy: Policy, steps: int, gamma: float = 0.99) -> dict:
    obs = env.reset()
    batch = {"obs": [], "actions": [], "logp": [], "rew": []}
    for _ in range(steps):
        action, logp = policy.act(obs)
        nxt, reward, done, _info = env.step(action)
        batch["obs"].append(obs)
        batch["actions"].append(action)
        batch["logp"].append(logp)
        batch["rew"].append(reward)
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
