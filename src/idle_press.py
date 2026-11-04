"""No-effect button press penalty (``long_v5``).

Why: in the live ``long_v4`` run (v4-1) the policy collapsed to pressing A (58 % of the actions). In the overworld an
A / B press that talks to nobody does not move the character, so it dodges every movement-linked cost (``bump``,
``loop``) while the dense path reward is tiny: standing still was the cheapest thing to do and PPO locked it in.

``press_had_effect(before, after)`` says whether a single A / B press made anything observable happen; the tracker
charges a small cost for each press that did not, up to a per-episode budget, so spamming buttons can never cost more
than ``budget`` (and so cannot dominate the reward). It only looks at presses made in the plain overworld: battle,
menu and dialogue steps are never charged (the caller passes only those), so mashing A through a text box, a battle
menu or a Pokecenter conversation is free.

"Something happened" = any of: the map / position / facing changed, a dialogue or script started, a battle started,
the party (count, species, HP, levels), the bag (items, counts, balls), the money or the event flags changed.

Pure Python on already-parsed ``GameState`` fields (no emulator reads, no numpy), so the cloud launcher's config import
chain stays light.
"""

from __future__ import annotations

from src.ram_parser import dialogue_open

IDLE_PRESS_COST = 0.015     # per no-effect A / B press in the overworld (about 1.5x the per-step cost)
IDLE_PRESS_BUDGET = 10.0    # most that can be charged in one episode


def validate_params(cost, budget) -> list[str]:
    errs: list[str] = []
    if cost < 0:
        errs.append(f"idle_press_cost must be >= 0 (got {cost})")
    if budget < 0:
        errs.append(f"idle_press_budget must be >= 0 (got {budget})")
    return errs


def _snapshot(s) -> tuple:
    g = getattr
    return (
        g(s, "map_group", 0), g(s, "map_number", 0), g(s, "x", 0), g(s, "y", 0), g(s, "facing_raw", 0),
        g(s, "battle_mode", 0), bool(dialogue_open(s)),
        g(s, "party_count", 0), tuple(g(s, "party_species", ()) or ()),
        tuple(tuple(h) for h in (g(s, "party_hp", ()) or ())), tuple(g(s, "party_levels", ()) or ()),
        tuple(g(s, "items", ()) or ()), tuple(g(s, "item_counts", ()) or ()), tuple(g(s, "balls", ()) or ()),
        g(s, "num_items", 0), g(s, "num_balls", 0), g(s, "money", 0), tuple(g(s, "event_flags", ()) or ()),
    )


def press_had_effect(before, after) -> bool:
    """True when the press changed anything observable (see the module docstring)."""
    return _snapshot(before) != _snapshot(after)


class IdlePressTracker:
    """Per-env, per-episode tracker. ``reset()`` at every episode start, ``update(before, after)`` per overworld press."""

    def __init__(self, cost: float = IDLE_PRESS_COST, budget: float = IDLE_PRESS_BUDGET):
        errs = validate_params(cost, budget)
        if errs:
            raise ValueError("; ".join(errs))
        self.cost, self.budget = float(cost), float(budget)
        self.reset()

    @classmethod
    def from_params(cls, params: dict | None) -> "IdlePressTracker | None":
        return None if not params else cls(**params)

    def reset(self) -> None:
        self.spent = 0.0   # cost charged this episode (positive number)
        self.count = 0     # no-effect presses this episode

    def update(self, before, after) -> float:
        """Reward term (<= 0) for ONE overworld A / B press from ``before`` to ``after`` (not for any other action)."""
        if press_had_effect(before, after):
            return 0.0
        self.count += 1
        pen = max(0.0, min(self.cost, self.budget - self.spent))
        self.spent += pen
        return -pen
