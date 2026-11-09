"""Healing-care reward terms (``long_v7``): make looking after the lead Pokemon worth the walk.  Reward shaping only.

Diagnosis of run v6-1 (24 envs, ~400 updates): 43-63 % of episodes ended in a white-out (-15, terminal), the lead was below
30 % HP in about half of the showcase snapshots, yet the heal reward paid only ~3 per update in total.  Nothing made
*walking to a Pokemon Center / using a Potion* better than walking on with a nearly dead lead.  Four small terms, each
behind its own switch (``HealCareTracker.from_params``), each capped per episode:

* ``lowhp_exposure``: a cost per step while the lead is below ``exposure_threshold`` HP in the plain overworld (no battle,
  no text, not fainted), ramping ``exposure_min`` -> ``exposure_max`` over ``exposure_ramp`` consecutive exposed steps, at most
  ``exposure_budget`` per episode.  Free inside Pokemon Center / Mart maps; the run (ramp) restarts after any heal.
* ``survive_bonus``: a one-off bonus when a battle is WON (trainer battle ended, or the wild enemy fainted) with the lead
  alive and above ``survive_threshold`` HP, at most ``survive_budget`` per episode.  Fleeing and losing pay nothing, so it
  rewards winning cleanly rather than escaping.
* ``potion_bonus``: a bonus when a healing item is USED (the bag count of Potion-class items drops) on a step that really
  restores HP while the lead was below ``potion_threshold``, at most ``potion_budget`` per episode.  Selling an item or a
  Pokemon Center heal never qualifies (no item used / no HP gain through the bag).  Bag navigation by buttons is a long
  sequence; this gives that sequence a payoff of its own on top of the HP-restored term.
* ``trip_exempt``: while the lead is below ``exposure_threshold`` HP (or in a Pokemon Center / Mart map) the retreat cost,
  stagnation penalty and dwell penalty are switched off by the caller (``is_heal_trip``), so a walk back to a Pokemon Center
  is never punished as a retreat or a stall.

Not scripted: no action is chosen for the agent.  Literature: Pleines et al. 2025 (Pokemon Red RL, arXiv 2502.19920) report
that an uncapped heal reward is exploited, hence the budgets; OpenAI Five (2019) shaped health explicitly.  Both are cited
from memory (not re-checked here).  No numpy; no heavy imports.
"""

from __future__ import annotations

from src.heal_reward import heal_gain   # pure python; battle_obs / ram_parser are imported lazily (src.train_config must stay numpy-free)

POKECENTER_MAPS = ((26, 5), (10, 10), (10, 13), (21, 6), (23, 7))   # Cherrygrove, Violet, Route 32, Azalea, Goldenrod
MART_MAPS = ((26, 4), (10, 9), (21, 7))                              # Cherrygrove, Violet, Azalea
HEAL_EXEMPT_MAPS = POKECENTER_MAPS + MART_MAPS

EXPOSURE_THRESHOLD = 0.3
EXPOSURE_MIN = 0.002
EXPOSURE_MAX = 0.01
EXPOSURE_RAMP = 128
EXPOSURE_BUDGET = 6.0
SURVIVE_BONUS = 0.5
SURVIVE_THRESHOLD = 0.5
SURVIVE_BUDGET = 3.0
POTION_BONUS = 1.0
POTION_THRESHOLD = 0.5
POTION_BUDGET = 3.0


def lead_fraction(state) -> float | None:
    """HP fraction of the first party slot, or None when unknown."""
    hp = getattr(state, "party_hp", ()) or ()
    if not hp:
        return None
    cur, mx = hp[0]
    return (cur / mx) if mx > 0 else None


def in_exempt_map(state) -> bool:
    return (getattr(state, "map_group", None), getattr(state, "map_number", None)) in HEAL_EXEMPT_MAPS


def is_heal_trip(state, threshold: float = EXPOSURE_THRESHOLD) -> bool:
    """True while walking to heal is the right thing: the lead is below ``threshold`` HP (and not mid white-out), or the
    agent is inside a Pokemon Center / Mart.  The caller switches the retreat / stagnation / dwell costs off then."""
    if getattr(state, "fainted", False):
        return False
    if in_exempt_map(state):
        return True
    f = lead_fraction(state)
    return f is not None and f < threshold


def validate_params(threshold, e_min, e_max, ramp, budget, s_bonus, s_thr, s_budget, p_bonus, p_thr, p_budget) -> list[str]:
    errs: list[str] = []
    if not 0 < threshold <= 1:
        errs.append(f"lowhp_exposure_threshold must be in (0, 1] (got {threshold})")
    if e_min < 0 or e_max < e_min:
        errs.append(f"lowhp_exposure_min/max must satisfy 0 <= min <= max (got {e_min}, {e_max})")
    if ramp < 1:
        errs.append(f"lowhp_exposure_ramp must be >= 1 (got {ramp})")
    for name, v in (("lowhp_exposure_budget", budget), ("survive_bonus", s_bonus), ("survive_budget", s_budget),
                    ("potion_bonus", p_bonus), ("potion_budget", p_budget)):
        if v < 0:
            errs.append(f"{name} must be >= 0 (got {v})")
    for name, v in (("survive_threshold", s_thr), ("potion_threshold", p_thr)):
        if not 0 < v <= 1:
            errs.append(f"{name} must be in (0, 1] (got {v})")
    return errs


class HealCareTracker:
    """Per-env, per-episode tracker.  ``reset()`` at every episode start; ``update(before, after)`` per step returns the
    step's component dict (keys ``lowhp``, ``survive``, ``potion``; absent when 0)."""

    def __init__(self, *, exposure: bool = True, threshold: float = EXPOSURE_THRESHOLD, e_min: float = EXPOSURE_MIN,
                 e_max: float = EXPOSURE_MAX, ramp: int = EXPOSURE_RAMP, e_budget: float = EXPOSURE_BUDGET,
                 survive: bool = True, s_bonus: float = SURVIVE_BONUS, s_threshold: float = SURVIVE_THRESHOLD,
                 s_budget: float = SURVIVE_BUDGET, potion: bool = True, p_bonus: float = POTION_BONUS,
                 p_threshold: float = POTION_THRESHOLD, p_budget: float = POTION_BUDGET, trip_exempt: bool = True):
        errs = validate_params(threshold, e_min, e_max, ramp, e_budget, s_bonus, s_threshold, s_budget,
                               p_bonus, p_threshold, p_budget)
        if errs:
            raise ValueError("; ".join(errs))
        self.exposure, self.threshold, self.e_min, self.e_max, self.ramp, self.e_budget = exposure, threshold, e_min, e_max, ramp, e_budget
        self.survive, self.s_bonus, self.s_threshold, self.s_budget = survive, s_bonus, s_threshold, s_budget
        self.potion, self.p_bonus, self.p_threshold, self.p_budget = potion, p_bonus, p_threshold, p_budget
        self.trip_exempt = bool(trip_exempt)
        self.reset()

    @classmethod
    def from_params(cls, params: dict | None) -> "HealCareTracker | None":
        return None if not params else cls(**params)

    def reset(self) -> None:
        self.run = 0            # consecutive exposed steps
        self.exposure_spent = 0.0
        self.survive_paid = 0.0
        self.potion_paid = 0.0

    def _pay(self, want: float, paid: float, budget: float) -> float:
        return max(0.0, min(want, budget - paid))

    def update(self, before, after) -> dict:
        from src.battle_obs import heal_item_count
        from src.ram_parser import dialogue_open

        out: dict[str, float] = {}
        g = getattr
        f_a, f_b = lead_fraction(after), lead_fraction(before)
        # -- exposure: lead low, plain overworld, not at a Center / Mart, not healing this step
        if self.exposure:
            busy = g(after, "battle_mode", 0) != 0 or dialogue_open(after)
            if g(after, "fainted", False) or in_exempt_map(after) or f_a is None or f_a >= self.threshold \
                    or (f_b is not None and f_a > f_b):
                self.run = 0
            elif not busy:
                self.run += 1
                frac = min(1.0, self.run / self.ramp)
                cost = self._pay(self.e_min + (self.e_max - self.e_min) * frac, self.exposure_spent, self.e_budget)
                self.exposure_spent += cost
                if cost:
                    out["lowhp"] = -cost
        # -- survive: a battle won with the lead healthy
        if self.survive and g(before, "battle_mode", 0) != 0 and g(after, "battle_mode", 0) == 0 \
                and not g(after, "fainted", False) and f_a is not None and f_a >= self.s_threshold:
            e_max = max(g(after, "enemy_max_hp", 0), g(before, "enemy_max_hp", 0))
            won = g(before, "battle_mode", 0) == 2 or (e_max > 0 and g(after, "enemy_hp", 1) == 0)
            if won:
                pay = self._pay(self.s_bonus, self.survive_paid, self.s_budget)
                self.survive_paid += pay
                if pay:
                    out["survive"] = pay
        # -- potion: a healing item used, HP really restored, lead was low
        if self.potion and f_b is not None and f_b < self.p_threshold \
                and heal_item_count(after) < heal_item_count(before) and heal_gain(before, after) > 0:
            pay = self._pay(self.p_bonus, self.potion_paid, self.p_budget)
            self.potion_paid += pay
            if pay:
                out["potion"] = pay
        return out
