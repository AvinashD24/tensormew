"""Dense "best-ever" path-progress reward (``long_v4``): reward shaping, NOT an observation.

The minimal story path (New Bark -> Route 29 -> Cherrygrove -> Route 30 -> Route 31 -> Violet gate -> Violet City ->
Violet Gym -> the tile in front of Falkner) is 319 steps long. ``progress(position)`` is how many of those steps are
already behind the agent: ``319 - BFS steps still to go`` over the ROM map graph (``src.route_graph``, warps and
doors included, ledges respected). Off-path tiles take the progress of the path tile they branch from minus the
walk back, so dead ends and buildings never score above their junction.

Per step the reward is ``scale * max(0, progress - best_progress)`` and ``best_progress`` only ever rises:

* a new best pays the improvement once; stepping back pays nothing and costs nothing, stepping forward again pays
  nothing until the old best is passed. Over an episode the payments telescope to ``scale * (final best - start
  progress)``, at most ``scale * 319`` (about 25.5 at the default 0.08), so it cannot be farmed by oscillating,
  circling or walking in and out of buildings;
* ``best_progress`` starts at the progress of the actual start state (after jitter, from the archive too), so a
  mid-route start pays nothing for the progress it already has;
* a white-out teleport (``fainted``) neither pays nor moves the best;
* milestone bonuses, the badge and everything else are unchanged; nothing is added to the observation.

The route graph is imported lazily: ``src.train_config`` (and so the bare-python cloud launcher) must stay importable
without numpy, and the graph needs the ROM; without it the reward is simply 0.
"""

from __future__ import annotations

DEFAULT_SCALE = 0.08   # reward per path tile gained; the whole path pays about 25.5 vs ~140 of milestone bonuses


def progress(state) -> float | None:
    """Path progress of ``state`` in tiles (0 at the start, ~319 in front of Falkner), or None when unknown."""
    try:
        from src import route_graph
        return route_graph.path_progress(state)
    except Exception:  # no ROM / unreadable state: the term is simply absent
        return None


def total() -> float | None:
    try:
        from src import route_graph
        return route_graph.path_total()
    except Exception:
        return None


class PathProgress:
    """Per-env, per-episode tracker. ``reset(state)`` at every episode start, ``update(before, after)`` per step."""

    def __init__(self, scale: float = DEFAULT_SCALE):
        if scale < 0:
            raise ValueError(f"path_progress scale must be >= 0 (got {scale})")
        self.scale = float(scale)
        self.best: float | None = 0.0   # None = the start was off the route graph; the first known tile sets it, unpaid
        self.gained = 0.0   # tiles of progress paid this episode (for tests / logging)

    @classmethod
    def from_scale(cls, scale: float | None) -> "PathProgress | None":
        return None if scale is None else cls(scale)

    def reset(self, state) -> None:
        p = progress(state)
        self.best = p
        self.gained = 0.0

    def update(self, before, after) -> float:
        """Reward (>= 0) for this step."""
        if getattr(after, "fainted", False) or getattr(before, "fainted", False):
            return 0.0   # the white-out teleport is not progress
        p = progress(after)
        if p is None:
            return 0.0
        if self.best is None:   # first position on the graph after an off-graph start: nothing to pay for
            self.best = p
            return 0.0
        if p <= self.best:
            return 0.0
        gain = p - self.best
        self.best = p
        self.gained += gain
        return self.scale * gain
