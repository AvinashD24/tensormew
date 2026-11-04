"""Progress-keyed stagnation penalty (``long_v6``): no new best path progress for a long time.

Why: in the live ``long_v5`` run 36-76 % of the Route 29 visits sat in x 14-16, y 14-15, the west dead-end corridor.
``src.dwell`` only fires for <= 16 tiles in a 192-step window with nothing new; wandering around a larger region without
ever improving the best path progress costs nothing but the step cost.  This penalty is keyed to the thing that matters:
steps since the episode's best path progress last improved (``src.path_progress``).

Rule: outside battle, text and Pokemon Center maps every step counts; once ``after`` (250) such steps passed without a new
best, each further step costs ``min_penalty`` (0.004) rising linearly to ``max_penalty`` (0.02) over ``ramp_steps`` (300)
steps, at most ``budget`` (10) per episode.  A new best resets the counter and the ramp.  While the dwell tracker's level
exemption holds (party below the stage's level target AND a battle or level-up in the last 512 steps, ``src.dwell``) the
step is free and the ramp does not bank time.

Composition with the dwell penalty (no double counting): both measure "stuck", so per step the agent pays
``max(dwell, stagnation)``, never the sum.  ``update`` takes the dwell charge of the same step and returns only the
EXCESS ``max(0, stagnation - dwell)``; only that excess is charged to this tracker's budget.  Reported as the
``stagnation`` reward component (<= 0) next to ``dwell``.  The thresholds are guesses (no literature value); time-since-
last-improvement penalties are a plain engineering shaping term.

Pure Python on already-parsed ``GameState`` fields (no numpy), so the cloud launcher's config import chain stays light.
"""

from __future__ import annotations

from src.dwell import POKECENTER_MAPS
from src.ram_parser import dialogue_open

STAGNATION_AFTER = 250
STAGNATION_MIN = 0.004
STAGNATION_MAX = 0.02
STAGNATION_RAMP = 300
STAGNATION_BUDGET = 10.0


def validate_params(after, min_penalty, max_penalty, ramp_steps, budget) -> list[str]:
    errs: list[str] = []
    if after < 1:
        errs.append(f"stagnation_after must be >= 1 (got {after})")
    if min_penalty < 0:
        errs.append(f"stagnation_min must be >= 0 (got {min_penalty})")
    if max_penalty < min_penalty:
        errs.append(f"stagnation_max ({max_penalty}) must not be below stagnation_min ({min_penalty})")
    if ramp_steps < 1:
        errs.append(f"stagnation_ramp must be >= 1 (got {ramp_steps})")
    if budget < 0:
        errs.append(f"stagnation_budget must be >= 0 (got {budget})")
    return errs


class StagnationTracker:
    """Per-env, per-episode tracker. ``reset()`` at every episode start, ``update(...)`` per step."""

    def __init__(self, after: int = STAGNATION_AFTER, min_penalty: float = STAGNATION_MIN,
                 max_penalty: float = STAGNATION_MAX, ramp_steps: int = STAGNATION_RAMP,
                 budget: float = STAGNATION_BUDGET):
        errs = validate_params(after, min_penalty, max_penalty, ramp_steps, budget)
        if errs:
            raise ValueError("; ".join(errs))
        self.after, self.ramp_steps = int(after), int(ramp_steps)
        self.min_penalty, self.max_penalty, self.budget = float(min_penalty), float(max_penalty), float(budget)
        self.reset()

    @classmethod
    def from_params(cls, params: dict | None) -> "StagnationTracker | None":
        return None if not params else cls(**params)

    def reset(self) -> None:
        self.since_best = 0   # counted steps since the best path progress last improved
        self.run = 0          # counted steps past the threshold (drives the ramp)
        self.spent = 0.0      # cost charged this episode (positive)

    def raw_penalty(self) -> float:
        """The stagnation charge for the current ``run`` before composing with dwell (0 below the threshold)."""
        if self.run <= 0:
            return 0.0
        frac = min(1.0, (self.run - 1) / float(self.ramp_steps))
        return self.min_penalty + (self.max_penalty - self.min_penalty) * frac

    def update(self, before, after, *, new_best: bool, level_exempt: bool = False, dwell_charge: float = 0.0) -> float:
        """Reward term (<= 0) for one step.  ``new_best``: the best path progress improved this step; ``level_exempt``:
        the dwell tracker's leveling exemption holds; ``dwell_charge``: the (<= 0) dwell term of this same step."""
        g = getattr
        if new_best:
            self.since_best = 0
            self.run = 0
            return 0.0
        busy = (g(before, "battle_mode", 0) != 0 or g(after, "battle_mode", 0) != 0
                or dialogue_open(before) or dialogue_open(after))
        if busy or (g(after, "map_group", 0), g(after, "map_number", 0)) in POKECENTER_MAPS:
            return 0.0   # battle / text / healing: not stagnation time, and it does not reset anything either
        self.since_best += 1
        if self.since_best <= self.after:
            return 0.0
        if level_exempt:   # needed leveling: no cost, and do not bank ramp time for the moment the target is reached
            self.run = 0
            return 0.0
        self.run += 1
        excess = max(0.0, self.raw_penalty() - (-float(dwell_charge)))
        pen = max(0.0, min(excess, self.budget - self.spent))
        self.spent += pen
        return -pen
