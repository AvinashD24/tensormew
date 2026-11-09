"""Healing reward for HP actually restored, weighted toward low HP (``long_v4``).

Replaces the old flat ``heal`` terms of ``src.rl_env.progress_reward`` (a one-off Pokecenter bonus and 3x the healed
fraction for items) when ``TrainConfig.heal_reward`` is on.

* Per party slot the HP fraction ``f`` is valued by a concave ``u(f) = 1 - (1 - f)^2`` (marginal value ``2 (1 - f)``:
  restoring HP from 0 % pays twice the average, topping up from 90 % pays almost nothing).  The payment is
  ``scale * mean over party slots of (u(f_after) - u(f_before))`` for slots that gained HP, so a full heal of a
  nearly dead party pays about ``scale`` and one HP of a healthy mon pays ~0.
* Only genuine heals pay: nothing during a white-out (the ``fainted`` teleport), when the map changed, after a level-up
  or any max-HP change (level-ups and evolution raise current HP too), or when the party size changed.  Natural
  regeneration does not exist in this game; a Potion in or out of battle and a Pokecenter heal both show up as a
  current-HP gain with unchanged max HP on the same map and are paid identically.
* The caller routes the payment through the per-episode ``heal`` budget (``budget_pay``), so a take-damage / heal loop
  can earn at most that budget.  Pleines et al. 2025 ("Pokemon Red via RL", arXiv 2502.19920) report exactly this
  failure with an uncapped heal reward (agents heal at the mother / Pokecenter repeatedly and stall), which is why the
  total is small and capped here.
"""

from __future__ import annotations

DEFAULT_SCALE = 4.0
DEFAULT_BUDGET = 8.0


def hp_value(frac: float) -> float:
    """Concave value of an HP fraction in [0, 1]: 0 at 0 HP, 1 at full HP, steepest at low HP."""
    f = min(1.0, max(0.0, float(frac)))
    return 1.0 - (1.0 - f) ** 2


def heal_gain(before, after) -> float:
    """Unscaled ``mean(u(f_after) - u(f_before))`` over party slots that gained HP; 0 when the change is not a heal."""
    if getattr(before, "fainted", False) or getattr(after, "fainted", False):
        return 0.0
    if (getattr(before, "map_group", None), getattr(before, "map_number", None)) != (
            getattr(after, "map_group", None), getattr(after, "map_number", None)):
        return 0.0
    hp_b = tuple(getattr(before, "party_hp", ()) or ())
    hp_a = tuple(getattr(after, "party_hp", ()) or ())
    if not hp_b or len(hp_b) != len(hp_a):
        return 0.0
    if all(c == 0 for c, _ in hp_b):
        return 0.0   # whole party fainted: the recovery after a white-out is not a heal to pay for
    lv_b = tuple(getattr(before, "party_levels", ()) or ())
    lv_a = tuple(getattr(after, "party_levels", ()) or ())
    if lv_b != lv_a:
        return 0.0   # level-up / evolution raise current HP together with the level
    total = 0.0
    for (cb, mb), (ca, ma) in zip(hp_b, hp_a):
        if mb != ma or mb <= 0:
            return 0.0   # max HP changed: a level-up, not a heal
        if ca > cb:
            total += hp_value(ca / ma) - hp_value(cb / mb)
    return total / len(hp_b)


def heal_payment(before, after, scale: float = DEFAULT_SCALE) -> float:
    """Reward (>= 0) for HP restored between two parsed states, before the per-episode budget."""
    return float(scale) * heal_gain(before, after)
