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

from src.dwell import POKECENTER_MAPS
from src.ram_parser import dialogue_open

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


RETREAT_FRAC = 0.35     # long_v6: fraction of the per-tile path reward charged per tile of progress lost
RETREAT_BUDGET = 15.0   # long_v6: most that can be charged for retreating in one episode


def validate_retreat(frac, budget) -> list[str]:
    errs: list[str] = []
    if frac < 0:
        errs.append(f"retreat_cost_frac must be >= 0 (got {frac})")
    if budget < 0:
        errs.append(f"retreat_budget must be >= 0 (got {budget})")
    return errs


class PathProgress:
    """Per-env, per-episode tracker. ``reset(state)`` at every episode start, ``update(before, after)`` per step.

    Optional retreat cost (``retreat_frac`` > 0, ``long_v6``): a soft ASYMMETRIC potential.  Let ``p`` be the BFS progress
    of the current position and ``best`` the episode's best.  A step that lowers ``p`` (further from the goal than the
    previous position) is charged ``retreat_frac * scale`` per tile of progress lost (``retreat_step``, <= 0, read by the
    caller as the ``retreat`` reward component); a step that raises ``p`` pays nothing unless it is a new best (full
    ``scale``, as before).  So a round trip off the route of ``k`` tiles each way costs ``retreat_frac * scale * k`` and
    can never be farmed: every re-approach is free, every new best pays once.  The true route is monotone in BFS distance
    (progress rises by exactly 1 per step), so it never triggers on the real path.  Never charged: with a white-out or
    battle / text open, in Pokemon Center maps, on a step whose map changes (warp, door, teleport), or when either
    position is off the route graph (progress None); the cost is capped at ``retreat_budget`` per episode.

    This is NOT potential-based shaping in the sense of Ng, Harada & Russell (1999): a potential ``F = gamma * Phi(s') -
    Phi(s)`` (with ``Phi = progress``) pays for both directions and so leaves the optimal policy unchanged; here the
    gain side of every non-best step is dropped, so the shaping is not policy-invariant.  It deliberately prefers
    routes that never retreat.
    """

    def __init__(self, scale: float = DEFAULT_SCALE, retreat_frac: float = 0.0, retreat_budget: float = RETREAT_BUDGET):
        if scale < 0:
            raise ValueError(f"path_progress scale must be >= 0 (got {scale})")
        errs = validate_retreat(retreat_frac, retreat_budget)
        if errs:
            raise ValueError("; ".join(errs))
        self.scale = float(scale)
        self.retreat_frac, self.retreat_budget = float(retreat_frac), float(retreat_budget)
        self.best: float | None = 0.0   # None = the start was off the route graph; the first known tile sets it, unpaid
        self.gained = 0.0   # tiles of progress paid this episode (for tests / logging)
        self.current: float | None = None   # progress of the latest position (None = unknown / white-out); route checkpoints
        self.retreat_step = 0.0    # this step's retreat cost (<= 0)
        self.retreat_spent = 0.0   # retreat cost charged this episode (positive)

    @classmethod
    def from_scale(cls, scale: float | None, retreat: dict | None = None) -> "PathProgress | None":
        if scale is None:
            return None
        r = retreat or {}
        return cls(scale, retreat_frac=r.get("frac", 0.0), retreat_budget=r.get("budget", RETREAT_BUDGET))

    def reset(self, state) -> None:
        p = progress(state)
        self.best = p
        self.gained = 0.0
        self.current = p
        self.retreat_step = 0.0
        self.retreat_spent = 0.0

    def _retreat(self, before, after, prev, p) -> float:
        """Retreat cost (<= 0) of the step from progress ``prev`` to ``p``; see the class docstring for the exemptions."""
        if self.retreat_frac <= 0 or prev is None or p is None or p >= prev:
            return 0.0
        g = getattr
        if (g(before, "map_group", 0), g(before, "map_number", 0)) != (g(after, "map_group", 0), g(after, "map_number", 0)):
            return 0.0   # warp / door / teleport: not a walked step
        if g(before, "battle_mode", 0) != 0 or g(after, "battle_mode", 0) != 0 or dialogue_open(before) or dialogue_open(after):
            return 0.0
        if (g(after, "map_group", 0), g(after, "map_number", 0)) in POKECENTER_MAPS:
            return 0.0   # healing visits are not a detour to punish
        cost = self.retreat_frac * self.scale * (prev - p)
        cost = max(0.0, min(cost, self.retreat_budget - self.retreat_spent))
        self.retreat_spent += cost
        return -cost

    def update(self, before, after) -> float:
        """Reward (>= 0) for this step.  The retreat cost (if enabled) is left in ``retreat_step`` (<= 0)."""
        self.retreat_step = 0.0
        if getattr(after, "fainted", False) or getattr(before, "fainted", False):
            self.current = None
            return 0.0   # the white-out teleport is not progress
        prev = self.current
        p = progress(after)
        self.current = p
        if p is None:
            return 0.0
        self.retreat_step = self._retreat(before, after, prev, p)
        if self.best is None:   # first position on the graph after an off-graph start: nothing to pay for
            self.best = p
            return 0.0
        if p <= self.best:
            return 0.0
        gain = p - self.best
        self.best = p
        self.gained += gain
        return self.scale * gain
