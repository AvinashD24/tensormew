"""Dwell penalty: a small cost for staying in one small area without making progress (``long_v4``).

Why: in the live run most visits are on Route 29 and thousands of wild battles are fought in the same grass.  The
existing per-tile ``loop`` term (``tools.train_deep_ppo.EnvWorker.step``) counts revisits of single tiles over the whole
episode (including battle steps) and never looks at what the party needs.  This tracker is different:

* it looks at a sliding window of the last ``window`` NON-battle, non-dialogue overworld steps (battle and text steps
  neither enter the window nor reset it, so a walk -> battle -> walk farming loop keeps its counter);
* "stuck" = the window is full, it covers at most ``max_tiles`` distinct tiles, and nothing new happened in it (no
  never-seen tile, no new map, no story milestone / event, no new species caught);
* the cost ramps from ``min_penalty`` to ``max_penalty`` per step over ``ramp_steps`` stuck steps, resets as soon as
  the agent is not stuck (the window sees more than ``max_tiles`` tiles, or the map changes) and is capped by a
  per-episode ``budget`` so it can never dominate the reward;
* LEVEL EXEMPTION: leveling that is actually needed is not punished.  ``level_targets[stage]`` is the party level
  wanted for the furthest story stage reached this episode.  While the party's highest level is below it, and the
  agent has fought or leveled within the last ``grind_grace`` steps (so standing still on a bush with no encounters
  does not count as grinding), the dwell cost is zero and the old ``loop`` cost is switched off as well
  (``loop_exempt``).  Pokecenter maps are always exempt (healing).

Pure Python on already-parsed ``GameState`` fields; it never reads the emulator.
"""

from __future__ import annotations

from collections import deque

from src.ram_parser import dialogue_open
from src.story_path import MILESTONE_COUNT, current_stage

# Same maps as ``src.rl_env.POKECENTER_MAPS`` (not imported: rl_env pulls in numpy-heavy observation code).
POKECENTER_MAPS = ((26, 5), (10, 10), (10, 13))

DWELL_WINDOW = 192
DWELL_MAX_TILES = 16
DWELL_MIN_PENALTY = 0.005
DWELL_MAX_PENALTY = 0.03
DWELL_RAMP_STEPS = 384
DWELL_BUDGET = 15.0
DWELL_GRIND_GRACE = 512
# Target party level (highest level in the party) by furthest story stage reached in the episode:
# index = ``src.story_path.current_stage`` value: 0 New Bark / unknown, 1 Route 29, 2 Cherrygrove, 3 Route 30,
# 4 Route 31, 5 Violet gatehouse, 6 Violet City, 7 Violet Gym, 8 Falkner beaten (0 = nothing more to level for).
# Falkner's team is Pidgey L7 + Pidgeotto L9 (Flying: Cyndaquil's Ember is neutral, Quilava at 14 is comfortable);
# 8 before Cherrygrove, 10 on Routes 30 / 31, 11 at the Violet gate, 12 for Violet City and the Gym.
DWELL_LEVEL_TARGETS = (8, 8, 10, 10, 10, 11, 12, 12, 0)


def validate_params(window, max_tiles, min_penalty, max_penalty, ramp_steps, budget, grind_grace, level_targets) -> list[str]:
    errs: list[str] = []
    if window < 8:
        errs.append(f"dwell_window must be >= 8 (got {window})")
    if max_tiles < 1:
        errs.append(f"dwell_max_tiles must be >= 1 (got {max_tiles})")
    elif max_tiles >= window:
        errs.append(f"dwell_max_tiles ({max_tiles}) must be below dwell_window ({window})")
    if min_penalty < 0:
        errs.append(f"dwell_min_penalty must be >= 0 (got {min_penalty})")
    if max_penalty < min_penalty:
        errs.append(f"dwell_max_penalty ({max_penalty}) must not be below dwell_min_penalty ({min_penalty})")
    if ramp_steps < 1:
        errs.append(f"dwell_ramp_steps must be >= 1 (got {ramp_steps})")
    if budget < 0:
        errs.append(f"dwell_budget must be >= 0 (got {budget})")
    if grind_grace < 0:
        errs.append(f"dwell_grind_grace must be >= 0 (got {grind_grace})")
    targets = list(level_targets)
    if len(targets) != MILESTONE_COUNT + 1:
        errs.append(f"dwell_level_targets needs {MILESTONE_COUNT + 1} entries (one per story stage 0..{MILESTONE_COUNT}), got {len(targets)}")
    elif any((not isinstance(v, int)) or isinstance(v, bool) or v < 0 or v > 100 for v in targets):
        errs.append("dwell_level_targets entries must be integers in [0, 100]")
    return errs


class DwellTracker:
    """Per-env, per-episode dwell state.  ``reset(state)`` at every episode start, then ``update(...)`` per step."""

    def __init__(self, window: int = DWELL_WINDOW, max_tiles: int = DWELL_MAX_TILES,
                 min_penalty: float = DWELL_MIN_PENALTY, max_penalty: float = DWELL_MAX_PENALTY,
                 ramp_steps: int = DWELL_RAMP_STEPS, budget: float = DWELL_BUDGET,
                 grind_grace: int = DWELL_GRIND_GRACE, level_targets=DWELL_LEVEL_TARGETS):
        errs = validate_params(window, max_tiles, min_penalty, max_penalty, ramp_steps, budget, grind_grace, level_targets)
        if errs:
            raise ValueError("; ".join(errs))
        self.window, self.max_tiles = int(window), int(max_tiles)
        self.min_penalty, self.max_penalty = float(min_penalty), float(max_penalty)
        self.ramp_steps, self.budget = int(ramp_steps), float(budget)
        self.grind_grace = int(grind_grace)
        self.level_targets = tuple(int(v) for v in level_targets)
        self.reset(None)

    @classmethod
    def from_params(cls, params: dict | None) -> "DwellTracker | None":
        return None if not params else cls(**params)

    def reset(self, state=None) -> None:
        self._tiles: deque = deque()
        self._counts: dict = {}
        self._map = None
        self.steps = 0          # every step of the episode (battle and text included)
        self.overworld = 0      # steps that entered the window
        self.last_progress = 0  # value of ``overworld`` at the last progress event
        self.last_grind = 0     # value of ``steps`` at the last battle end / level-up
        self.run = 0            # consecutive stuck steps (drives the ramp)
        self.spent = 0.0        # dwell cost paid this episode (positive number)
        self.furthest = current_stage(state) if state is not None else 0
        if state is not None:
            self._map = (state.map_group, state.map_number)

    # -- helpers
    def target_level(self) -> int:
        return self.level_targets[min(self.furthest, len(self.level_targets) - 1)]

    def level_exempt(self, state) -> bool:
        """True while more levels are wanted for the current stage and the agent is actually grinding."""
        target = self.target_level()
        levels = tuple(getattr(state, "party_levels", ()) or ())
        if target <= 0 or not levels:
            return False
        return max(levels) < target and (self.steps - self.last_grind) <= self.grind_grace

    def _clear_window(self) -> None:
        self._tiles.clear()
        self._counts.clear()
        self.run = 0

    def _push(self, tile) -> None:
        self._tiles.append(tile)
        self._counts[tile] = self._counts.get(tile, 0) + 1
        if len(self._tiles) > self.window:
            old = self._tiles.popleft()
            if self._counts[old] <= 1:
                del self._counts[old]
            else:
                self._counts[old] -= 1

    @property
    def stuck(self) -> bool:
        return (len(self._tiles) >= self.window and len(self._counts) <= self.max_tiles
                and self.overworld - self.last_progress >= self.window)

    # -- per step
    def update(self, before, after, *, new_tile: bool = False, new_map: bool = False, progress: bool = False) -> tuple[float, bool]:
        """Returns ``(penalty, loop_exempt)``: the dwell reward term (<= 0) and whether the ``loop`` term is switched off.

        ``new_tile`` / ``new_map``: this step discovered a tile / map for the episode; ``progress``: any other progress
        event (story milestone or event, new species caught).
        """
        self.steps += 1
        self.furthest = max(self.furthest, current_stage(after))
        if getattr(before, "battle_mode", 0) != 0 and getattr(after, "battle_mode", 0) == 0:
            self.last_grind = self.steps  # a fight just ended
        lb, la = tuple(getattr(before, "party_levels", ()) or ()), tuple(getattr(after, "party_levels", ()) or ())
        if len(lb) == len(la) and sum(la) > sum(lb):
            self.last_grind = self.steps  # a level-up
        loop_exempt = self.level_exempt(after)

        map_id = (after.map_group, after.map_number)
        if map_id != self._map:  # warp, door, white-out teleport: a fresh window
            self._map = map_id
            self._clear_window()
            self.last_progress = self.overworld
        busy = (getattr(before, "battle_mode", 0) != 0 or getattr(after, "battle_mode", 0) != 0
                or dialogue_open(before) or dialogue_open(after))
        if busy:
            return 0.0, loop_exempt  # battle / text: not dwell time, and it does not reset anything either
        if map_id in POKECENTER_MAPS:
            self.run = 0
            return 0.0, loop_exempt

        self.overworld += 1
        self._push((after.x, after.y))
        if new_tile or new_map or progress:
            self.last_progress = self.overworld
        if not self.stuck:
            self.run = 0
            return 0.0, loop_exempt
        if loop_exempt:  # needed leveling: no cost, and do not bank ramp time for the moment the target is reached
            self.run = 0
            return 0.0, loop_exempt
        self.run += 1
        frac = min(1.0, (self.run - 1) / float(self.ramp_steps))
        pen = self.min_penalty + (self.max_penalty - self.min_penalty) * frac
        pen = max(0.0, min(pen, self.budget - self.spent))
        self.spent += pen
        return -pen, loop_exempt
