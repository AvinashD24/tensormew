"""Periodic from-start evaluation (flag ``periodic_eval``, OFF everywhere except ``long_v8``): pure helpers.

Every ``eval_every`` updates the trainer plays ``eval_episodes`` episodes of ``eval_max_steps`` steps with the CURRENT policy
from the BASE start state (``new_bark_outside``) on one dedicated, in-process environment: no archive / checkpoint / swarm
resets, no stagger offset, no start jitter, stochastic sampling like training (the policy's own distribution, with the
valid-action mask when masking is on, and the early battle A-bias off because that is a training aid). It is the honest
"learned from scratch" curve: the training episodes mix in route checkpoints, swarm states and battle states, so their
progress partly measures the head start. The evaluation reads and writes none of the training state: separate emulator,
separate sampling generator, separate GRU state, no archive writes, no reward-scaler / counter update. The runner lives in
``tools/train_deep_ppo.py`` (it needs ``EnvWorker``); this module is the dependency-free bookkeeping.
"""

from __future__ import annotations

EVAL_STATE = "new_bark_outside"   # the base start state: the only state evaluation episodes start from


def is_eval_update(update: int, every: int, enabled: bool = True) -> bool:
    """True after updates ``every - 1``, ``2 * every - 1``, ... (so ``every`` training updates precede each evaluation)."""
    return bool(enabled) and int(every) >= 1 and (int(update) + 1) % int(every) == 0


def episode_record(ret: float, steps: int, furthest: int, best_path, end_reason, ms_steps: dict) -> dict:
    """One finished evaluation episode. ``ms_steps`` maps milestone index -> steps since the episode start (first reach)."""
    return {"return": float(ret), "steps": int(steps), "furthest": int(furthest or 0),
            "best_path": float(best_path) if best_path is not None else 0.0,
            "whiteout": end_reason == "whiteout", "ms_steps": {int(k): int(v) for k, v in dict(ms_steps).items()}}


def aggregate(episodes: list[dict], n_milestones: int, update: int, seconds: float = 0.0) -> dict:
    """The ``eval_*`` payload (flat, JSON-able) of one evaluation. ``eval_ms_reach[i]`` is the fraction of episodes that reached
    milestone ``i``; ``eval_ms_steps[i]`` the mean steps to it over the episodes that did (None when none did)."""
    n = len(episodes)
    if n == 0:
        raise ValueError("no evaluation episodes")
    reach, steps = [], []
    for m in range(int(n_milestones)):
        hit = [e["ms_steps"][m] for e in episodes if m in e["ms_steps"] or e["furthest"] >= m + 1]
        got = [e["ms_steps"][m] for e in episodes if m in e["ms_steps"]]
        reach.append(round(len(hit) / n, 4))
        steps.append(round(sum(got) / len(got), 1) if got else None)
    best = [e["best_path"] for e in episodes]
    return {
        "eval_update": int(update),
        "eval_episodes": n,
        "eval_steps": int(sum(e["steps"] for e in episodes)),
        "eval_best_progress_mean": round(sum(best) / n, 3),
        "eval_best_progress_max": round(max(best), 3),
        "eval_ms_reach": reach,
        "eval_ms_steps": steps,
        "eval_whiteout_frac": round(sum(1 for e in episodes if e["whiteout"]) / n, 4),
        "eval_return_mean": round(sum(e["return"] for e in episodes) / n, 3),
        "eval_seconds": round(float(seconds), 2),
    }
