"""Policy distillation: teach the grid policy to act like a trained pixel policy.

A *teacher* (a ``SpatialVisionPolicy`` checkpoint) plays while we record, step by step, what a grid student would have seen
(the semantic grid + the same RAM vector), the teacher's full action distribution and the valid-action mask. The student is
then trained by supervised learning to match those distributions (KL), on contiguous chunks so its GRU gets real context.
After this the student is fine-tuned with ordinary PPO (``tools.train_deep_ppo --visual grid --resume``).

A *shard* is one worker's uninterrupted stream (``.npz``), in time order:

``grids`` (N, 7, 15, 15) uint8 | ``rams`` (N, D) float32 | ``probs`` (N, A) float32 teacher distribution (after masking) |
``masks`` (N, A) bool | ``dones`` (N,) bool: the episode ended after this step (the GRU state resets before the next one).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHARD_KEYS = ("grids", "rams", "probs", "masks", "dones")


def save_shard(path: str | Path, grids, rams, probs, masks, dones) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, grids=np.asarray(grids, np.uint8), rams=np.asarray(rams, np.float32),
                        probs=np.asarray(probs, np.float32), masks=np.asarray(masks, bool), dones=np.asarray(dones, bool))


def load_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        shard = {k: z[k] for k in SHARD_KEYS}
    n = len(shard["grids"])
    for k in SHARD_KEYS:
        if len(shard[k]) != n:
            raise ValueError(f"{path}: '{k}' has {len(shard[k])} rows, expected {n}")
    return shard


@dataclass
class ChunkSampler:
    """Random contiguous chunks of ``length`` steps from a list of shards. The first ``burn_in`` steps of a chunk only warm the
    GRU up (no loss), since a chunk starts from a zero hidden state that the real policy would not have there."""

    shards: list[dict]
    length: int = 80
    burn_in: int = 16
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.shards = [s for s in self.shards if len(s["grids"]) >= self.length]
        if not self.shards:
            raise ValueError(f"no shard has {self.length} steps")
        sizes = np.array([len(s["grids"]) - self.length + 1 for s in self.shards], dtype=np.float64)
        self.weights = sizes / sizes.sum()

    @property
    def steps(self) -> int:
        return int(sum(len(s["grids"]) for s in self.shards))

    def sample(self, batch: int) -> dict[str, np.ndarray]:
        picks = self.rng.choice(len(self.shards), size=batch, p=self.weights)
        out = {k: [] for k in SHARD_KEYS}
        for i in picks:
            s = self.shards[i]
            start = int(self.rng.integers(0, len(s["grids"]) - self.length + 1))
            for k in SHARD_KEYS:
                out[k].append(s[k][start:start + self.length])
        return {k: np.stack(v) for k, v in out.items()}


def distill_loss(student_logits, teacher_probs, burn_in: int):
    """Mean KL(teacher || student) over the non-burn-in steps, plus top-1 agreement with the teacher's argmax.

    ``student_logits`` (B, T, A) already carry the same action mask the teacher acted under (masked logits are ~-1e9 and the
    teacher gives them probability 0, so they contribute nothing)."""
    import torch

    logp = torch.log_softmax(student_logits[:, burn_in:].float(), dim=-1)
    p = teacher_probs[:, burn_in:].float()
    kl = (torch.special.xlogy(p, p) - p * logp).sum(-1).mean()
    agree = (logp.argmax(-1) == p.argmax(-1)).float().mean()
    return kl, agree


def uniform_baseline(shards: list[dict]) -> dict[str, float]:
    """What an untrained (uniform over valid actions) student scores, to put the distillation numbers in context."""
    kls, agree = [], []
    for s in shards:
        m = s["masks"].astype(np.float64)
        u = m / np.maximum(m.sum(-1, keepdims=True), 1.0)
        p = s["probs"].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            kls.append(np.nansum(np.where(p > 0, p * (np.log(np.where(p > 0, p, 1)) - np.log(np.where(u > 0, u, 1))), 0), -1))
        agree.append((u.argmax(-1) == p.argmax(-1)).astype(np.float64))
    return {"kl": float(np.concatenate(kls).mean()), "agree": float(np.concatenate(agree).mean())}
