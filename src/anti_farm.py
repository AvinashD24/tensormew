"""Progress-first reward helpers (``long_v9``): diminishing battle returns, level-reward gating and the milestone speed bonus.

Why (run v8-1, ``long_v8``, 107 updates, 12 envs): the positive reward came mostly from battles (+2208, 846 wild battles, 0
trainer battles) and curiosity (+1842), the dense path reward paid only +1176 and the agent's best path progress was ~51 of
695 tiles, 92 % of its visits on Route 29, 131 white-outs.  Battling in the same grass paid more than walking the route.
Three reward-only changes (nothing is scripted, nothing is added to the observation about where to go):

* ``battle_multiplier``: the wild-battle win / KO / damage payments decay per wild battle within an episode.  The first
  ``free`` battles pay in full, battle ``n`` pays ``decay ** (n - free)`` of it, never below ``floor``.  Together with a small
  per-episode wild-battle budget (the existing capped-budget mechanism, ``src.rl_env.budget_pay``) the most a farmer can take
  is bounded, and a trainer battle (forced, story-gating: Falkner) keeps its own larger undiminished budget.  Diminishing
  returns for repeated rewards are the standard fix for a farmable shaping term (Ng et al. 1999 on why unbounded shaping
  changes the optimal policy; Pleines et al. 2025, arXiv 2502.19920, on capped / decaying Pokemon rewards).
* level-up reward only while the party's highest level is below the current story stage's target (the same table the dwell
  penalty uses, ``dwell_level_targets``): needed levelling for Falkner stays rewarded, grinding above the target pays nothing.
* ``speed_multiplier``: each milestone / key-event / badge payment is multiplied by a factor from ``max_mult`` (reached at
  step 0 of the episode) falling linearly to 1.0 at the end of the episode, so reaching a milestone sooner pays more.  An
  episode that starts mid-route (route checkpoint, archive) counts its own steps from its own start.

Pure Python (the cloud launcher imports this through ``src.train_config`` under a bare python: no numpy, no torch).
"""

from __future__ import annotations

FREE_BATTLES = 2          # wild battles per episode that pay in full
DECAY = 0.6               # per-battle factor after the free ones: 0.6, 0.36, 0.216, ...
FLOOR = 0.10              # a wild battle never pays less than this fraction of the full amount
WILD_SCALE = 0.4          # wild win / KO / damage scale (full wild battle: 3 * .4 + 1 * .4 + 1.5 * .4 = 2.2)
WILD_BUDGET = 6.0         # per-episode cap on all wild-battle payments (long_v8: 20 for every battle)
TRAINER_BUDGET = 20.0     # per-episode cap on trainer-battle payments (undiminished: win 5, KO 1, damage 1.5 per HP fraction)
CATCH_REWARD = 2.0        # first catch of a species in an episode (long_v8: 3, then 1.5, 0.75 ...)
CATCH_BUDGET = 4.0        # per-episode cap on catch payments (long_v8: 6)
SPEED_MAX = 1.5           # milestone speed bonus multiplier at the start of an episode (1.0 = off)
MILESTONE_SCALE = 1.5     # long_v9: multiplier on every milestone / key-event / badge payment


def battle_multiplier(n: int, free: int = FREE_BATTLES, decay: float = DECAY, floor: float = FLOOR) -> float:
    """Fraction of a wild battle's full reward paid for the ``n``-th (1-based) wild battle of an episode."""
    n = max(1, int(n))
    if n <= free:
        return 1.0
    return max(float(floor), float(decay) ** (n - free))


def speed_multiplier(elapsed: float, limit: float, max_mult: float = SPEED_MAX) -> float:
    """Milestone multiplier in [1, ``max_mult``]: ``max_mult`` at ``elapsed`` 0, linearly down to 1.0 at ``limit`` steps.

    ``max_mult`` <= 1 (or no usable limit) returns 1.0, i.e. the speed bonus is off."""
    if max_mult <= 1.0 or limit <= 0:
        return 1.0
    frac = min(1.0, max(0.0, float(elapsed) / float(limit)))
    return 1.0 + (float(max_mult) - 1.0) * (1.0 - frac)


def level_pays(levels_before, target: int | None) -> bool:
    """Whether a level-up pays: ``target`` None = gate off (always pays); otherwise only while the party's highest level
    BEFORE the gain is below the stage target (0 = nothing more to level for: pays nothing)."""
    if target is None:
        return True
    if target <= 0:
        return False
    lv = tuple(levels_before or ())
    return (max(lv) if lv else 0) < target


def validate_params(free, decay, floor, wild_scale, wild_budget, trainer_budget, catch_reward, catch_budget,
                    speed_max, milestone_scale) -> list[str]:
    errs: list[str] = []
    if free < 0:
        errs.append(f"battle_free_count must be >= 0 (got {free})")
    if not 0 < decay <= 1:
        errs.append(f"battle_decay must be in (0, 1] (got {decay})")
    if not 0 <= floor <= 1:
        errs.append(f"battle_floor must be in [0, 1] (got {floor})")
    if wild_scale < 0 or wild_budget < 0 or trainer_budget < 0:
        errs.append("wild_battle_scale, wild_battle_budget and trainer_battle_budget must be >= 0")
    if catch_reward < 0 or catch_budget < 0:
        errs.append("catch_reward and catch_budget must be >= 0")
    if speed_max < 1.0:
        errs.append(f"milestone_speed_bonus must be >= 1.0 (1.0 = off; got {speed_max})")
    if milestone_scale <= 0:
        errs.append(f"milestone_reward_scale must be > 0 (got {milestone_scale})")
    return errs
