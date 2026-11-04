"""Milestone success-rate bookkeeping: how often do episodes reach each story milestone?

A pure diagnostic (no effect on training).  The mean return is a poor "is it learning?" signal when many reward
terms move at once; the fraction of episodes that get past Route 29, Cherrygrove, ..., Violet Gym, Falkner is not.

Definitions (``MILESTONES`` order, milestone ``i`` has story stage ``i + 1``, see ``src.story_path.current_stage``):

* an episode is ELIGIBLE for milestone ``i`` when it started before it (``start_stage <= i``): an archive reset that
  starts in Violet City cannot "reach" Route 29;
* it REACHED milestone ``i`` when its furthest stage was ``>= i + 1``;
* ``reach[i] = reached / eligible`` over the finished episodes of the window (None when nothing was eligible);
* ``envs_at[i]`` = share of ALL envs whose running episode is currently at or above stage ``i + 1``;
* ``best_path`` = best path progress (tiles of the 319-tile story path, ``src.path_progress``) of each finished episode.

Staggered first episodes that were cut short (``partial``) are skipped, like the route-progress signal does.  No numpy.
"""

from __future__ import annotations

from src.story_path import MILESTONES, MILESTONE_COUNT

MILESTONE_NAMES = tuple(m.name for m in MILESTONES)


def _rates(reached: list[int], eligible: list[int]) -> list[float | None]:
    return [round(r / e, 3) if e > 0 else None for r, e in zip(reached, eligible)]


class MilestoneStats:
    def __init__(self, n: int = MILESTONE_COUNT):
        self.n = int(n)
        self._reset_window()
        self.cum_reached = [0] * self.n
        self.cum_eligible = [0] * self.n
        self.cum_eps = 0
        self.cum_best_max = 0.0

    def _reset_window(self) -> None:
        self.win_reached = [0] * self.n
        self.win_eligible = [0] * self.n
        self.win_eps = 0
        self.win_best: list[float] = []

    def add_episode(self, start_stage, furthest, best_path=None, partial: bool = False) -> bool:
        """Record one finished episode; returns False when it was skipped (partial or missing data)."""
        if partial or start_stage is None or furthest is None:
            return False
        start_stage, furthest = int(start_stage), int(furthest)
        for i in range(self.n):
            if start_stage <= i:
                self.win_eligible[i] += 1
                self.cum_eligible[i] += 1
                if furthest >= i + 1:
                    self.win_reached[i] += 1
                    self.cum_reached[i] += 1
        self.win_eps += 1
        self.cum_eps += 1
        if best_path is not None:
            self.win_best.append(float(best_path))
            self.cum_best_max = max(self.cum_best_max, float(best_path))
        return True

    def end_update(self, env_furthest=()) -> dict:
        """Compact payload for the metrics / history / state.json of one update; starts a new window."""
        fur = [int(f) for f in env_furthest if f is not None]
        at = [round(sum(1 for f in fur if f >= i + 1) / len(fur), 3) if fur else None for i in range(self.n)]
        out = {
            "ms_reach": _rates(self.win_reached, self.win_eligible),
            "ms_elig": list(self.win_eligible),
            "ms_reach_cum": _rates(self.cum_reached, self.cum_eligible),
            "ms_envs_at": at,
            "ms_eps": self.win_eps,
            "ms_eps_cum": self.cum_eps,
            "path_best_mean": round(sum(self.win_best) / len(self.win_best), 2) if self.win_best else None,
            "path_best_max": round(max(self.win_best), 1) if self.win_best else None,
            "path_best_max_cum": round(self.cum_best_max, 1),
        }
        self._reset_window()
        return out
