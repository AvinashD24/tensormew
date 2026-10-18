"""Numpy GRU policy. Weights start small and random; the step is a forward only."""

from __future__ import annotations

import numpy as np

_SCALE = 0.01
_GRU_KEYS = ("w_z", "u_z", "b_z", "w_r", "u_r", "b_r", "w_n", "u_n", "b_n")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def gru_step(h: np.ndarray, x: np.ndarray, w: dict) -> np.ndarray:
    """One GRU forward step. No gradients."""
    hidden = np.asarray(h, dtype=np.float64).reshape(-1)
    inputs = np.asarray(x, dtype=np.float64).reshape(-1)
    update = _sigmoid(w["w_z"] @ inputs + w["u_z"] @ hidden + w["b_z"])
    reset = _sigmoid(w["w_r"] @ inputs + w["u_r"] @ hidden + w["b_r"])
    candidate = np.tanh(w["w_n"] @ inputs + w["u_n"] @ (reset * hidden) + w["b_n"])
    return (1.0 - update) * hidden + update * candidate


def _init_gru(obs_dim: int, hidden_dim: int, rng: np.random.Generator) -> dict:
    def matrix(rows: int, cols: int) -> np.ndarray:
        return rng.normal(0.0, _SCALE, size=(rows, cols))

    return {
        "w_z": matrix(hidden_dim, obs_dim),
        "u_z": matrix(hidden_dim, hidden_dim),
        "b_z": np.zeros(hidden_dim),
        "w_r": matrix(hidden_dim, obs_dim),
        "u_r": matrix(hidden_dim, hidden_dim),
        "b_r": np.zeros(hidden_dim),
        "w_n": matrix(hidden_dim, obs_dim),
        "u_n": matrix(hidden_dim, hidden_dim),
        "b_n": np.zeros(hidden_dim),
    }


class RecurrentPolicy:
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 16,
        n_actions: int = 5,
        rng: np.random.Generator | None = None,
    ):
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng() if rng is None else rng
        self.w = _init_gru(self.obs_dim, self.hidden_dim, self.rng)
        self.w_a = self.rng.normal(0.0, _SCALE, size=(self.n_actions, self.hidden_dim))
        self.b_a = np.zeros(self.n_actions)
        self.reset()

    def reset(self) -> None:
        self.hidden = np.zeros(self.hidden_dim, dtype=np.float64)

    def act(self, obs: np.ndarray) -> tuple[int, float]:
        self.hidden = gru_step(self.hidden, obs, self.w)
        logits = self.w_a @ self.hidden + self.b_a
        logits = logits - logits.max()
        probabilities = np.exp(logits)
        probabilities = probabilities / probabilities.sum()
        action = int(self.rng.choice(self.n_actions, p=probabilities))
        return action, float(np.log(probabilities[action] + 1e-8))

    def save(self, path: str) -> None:
        arrays = {key: self.w[key] for key in _GRU_KEYS}
        arrays["w_a"] = self.w_a
        arrays["b_a"] = self.b_a
        arrays["hidden"] = self.hidden
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.w = {key: np.array(data[key], copy=True) for key in _GRU_KEYS}
        self.w_a = np.array(data["w_a"], copy=True)
        self.b_a = np.array(data["b_a"], copy=True)
        self.hidden = np.array(data["hidden"], copy=True)
        self.hidden_dim = int(self.hidden.shape[0])
        self.obs_dim = int(self.w["w_z"].shape[1])
        self.n_actions = int(self.w_a.shape[0])
