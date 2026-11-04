"""Dialogue-loop penalty (``long_v6``): repeated A-driven dialogue that leads nowhere.

Why: in run v5-1 (``long_v5``) one tile of the Route 29 gate building drew ~170k visits and the recent actions were 49 %
A.  ``idle_press`` exempts dialogue (a press that opens text had an effect), and ``src.dwell`` is only a mild ramp, so
talking to the same NPC again and again was nearly free.  This tracker charges a small cost per dialogue step while the
loop holds.

Definitions (a *talk* = an A press from the plain overworld that opens text, through the step that closes it):

* a dialogue step is a step where text is open before or after it;
* rule A (window): more than ``max_steps`` (60) countable dialogue steps in the last ``window`` (200) steps without a
  progress signal;
* rule B (same tile): the talk's (map, x, y, facing) was already talked to more than ``max_talks`` (6) times in this
  episode WITHOUT any observable effect (a talk that changes nothing adds one; a talk with an effect resets the count);
* a *progress signal* is any change of map, event flags, bag (items / counts / balls), party (count, species, levels,
  HP), money or Pokedex between the two states of a step; it clears the window and that step is never charged.

Never counted or charged: the first talk to a (tile, facing) in the episode (first-time dialogue), dialogue not started by
an A press of the agent (cutscenes, scripted text: origin unknown), any step with a battle (battle text) and every
Pokemon Center map (the healing flow).  Charge = ``cost`` (0.02) per charged step, at most ``budget`` (10) per episode.
Reported as the ``dialogue_loop`` reward component (<= 0).  The thresholds are guesses.

Pure Python on already-parsed ``GameState`` fields (no numpy), so the cloud launcher's config import chain stays light.
"""

from __future__ import annotations

from src.dwell import POKECENTER_MAPS
from src.ram_parser import dialogue_open

DIALOGUE_LOOP_WINDOW = 200
DIALOGUE_LOOP_MAX_STEPS = 60
DIALOGUE_LOOP_MAX_TALKS = 6
DIALOGUE_LOOP_COST = 0.02
DIALOGUE_LOOP_BUDGET = 10.0


def validate_params(window, max_steps, max_talks, cost, budget) -> list[str]:
    errs: list[str] = []
    if window < 1:
        errs.append(f"dialogue_loop_window must be >= 1 (got {window})")
    if max_steps < 1:
        errs.append(f"dialogue_loop_max_steps must be >= 1 (got {max_steps})")
    if max_talks < 0:
        errs.append(f"dialogue_loop_max_talks must be >= 0 (got {max_talks})")
    if cost < 0:
        errs.append(f"dialogue_loop_cost must be >= 0 (got {cost})")
    if budget < 0:
        errs.append(f"dialogue_loop_budget must be >= 0 (got {budget})")
    return errs


def progress_signature(s) -> tuple:
    """What counts as 'something happened' besides moving: map, flags, bag, party, money, Pokedex."""
    g = getattr
    return (
        g(s, "map_group", 0), g(s, "map_number", 0),
        g(s, "party_count", 0), tuple(g(s, "party_species", ()) or ()),
        tuple(tuple(h) for h in (g(s, "party_hp", ()) or ())), tuple(g(s, "party_levels", ()) or ()),
        tuple(g(s, "items", ()) or ()), tuple(g(s, "item_counts", ()) or ()), tuple(g(s, "balls", ()) or ()),
        g(s, "num_items", 0), g(s, "num_balls", 0), g(s, "money", 0), tuple(g(s, "event_flags", ()) or ()),
        bytes(g(s, "pokedex_caught", b"") or b""), bytes(g(s, "pokedex_seen", b"") or b""),
    )


class DialogueLoopTracker:
    """Per-env, per-episode tracker. ``reset()`` at every episode start, ``update(before, after, pressed_a)`` per step."""

    def __init__(self, window: int = DIALOGUE_LOOP_WINDOW, max_steps: int = DIALOGUE_LOOP_MAX_STEPS,
                 max_talks: int = DIALOGUE_LOOP_MAX_TALKS, cost: float = DIALOGUE_LOOP_COST,
                 budget: float = DIALOGUE_LOOP_BUDGET):
        errs = validate_params(window, max_steps, max_talks, cost, budget)
        if errs:
            raise ValueError("; ".join(errs))
        self.window, self.max_steps, self.max_talks = int(window), int(max_steps), int(max_talks)
        self.cost, self.budget = float(cost), float(budget)
        self.reset()

    @classmethod
    def from_params(cls, params: dict | None) -> "DialogueLoopTracker | None":
        return None if not params else cls(**params)

    def reset(self) -> None:
        self.spent = 0.0       # cost charged this episode (positive)
        self.charged = 0       # charged steps this episode
        self._win: list[bool] = []   # last ``window`` steps: True = a countable dialogue step
        self._win_count = 0
        self._talks: dict = {}       # (map, x, y, facing) -> talks started this episode
        self._idle: dict = {}        # same key -> consecutive talks that changed nothing
        self._talk: dict | None = None

    # -- window helpers
    def _push(self, countable: bool) -> None:
        self._win.append(bool(countable))
        self._win_count += 1 if countable else 0
        if len(self._win) > self.window:
            self._win_count -= 1 if self._win.pop(0) else 0

    def _clear(self) -> None:
        self._win, self._win_count = [], 0

    @property
    def window_count(self) -> int:
        return self._win_count

    def update(self, before, after, pressed_a: bool) -> float:
        """Reward term (<= 0) for one step from ``before`` to ``after``; ``pressed_a`` = the executed action was A."""
        g = getattr
        in_center = ((g(before, "map_group", 0), g(before, "map_number", 0)) in POKECENTER_MAPS
                     or (g(after, "map_group", 0), g(after, "map_number", 0)) in POKECENTER_MAPS)
        if in_center or g(before, "battle_mode", 0) != 0 or g(after, "battle_mode", 0) != 0:
            self._talk = None
            self._push(False)
            return 0.0
        d_before, d_after = bool(dialogue_open(before)), bool(dialogue_open(after))
        dialogue_step = d_before or d_after
        if not dialogue_step and self._win_count == 0 and self._talk is None:
            self._push(False)   # nothing to track and nothing to clear: skip the signature work
            return 0.0
        sig_b = progress_signature(before)
        progressed = sig_b != progress_signature(after)
        if progressed:
            self._clear()
        if not dialogue_step:
            self._push(False)
            self._talk = None
            return 0.0

        if self._talk is None:   # a talk starts here, or the text was not opened by an A press (cutscene: untracked)
            if pressed_a and not d_before:
                key = (g(before, "map_group", 0), g(before, "map_number", 0), g(before, "x", 0), g(before, "y", 0),
                       g(before, "facing", g(before, "facing_raw", 0)))
                self._talks[key] = self._talks.get(key, 0) + 1
                self._talk = {"key": key, "repeat": self._talks[key] >= 2,
                              "same_tile": self._idle.get(key, 0) > self.max_talks, "sig": sig_b}
            else:
                self._talk = {"key": None, "repeat": False, "same_tile": False, "sig": sig_b}
        talk = self._talk
        countable = talk["repeat"] and not progressed
        self._push(countable)
        pen = 0.0
        if countable and (talk["same_tile"] or self._win_count > self.max_steps):
            pen = max(0.0, min(self.cost, self.budget - self.spent))
            if pen > 0:
                self.spent += pen
                self.charged += 1
        if not d_after:   # the text closed: did the talk change anything?
            if talk["key"] is not None:
                changed = progress_signature(after) != talk["sig"]
                self._idle[talk["key"]] = 0 if changed else self._idle.get(talk["key"], 0) + 1
            self._talk = None
        return -pen
