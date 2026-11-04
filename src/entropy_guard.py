"""Anti-collapse entropy controls (``long_v5``). Both only change the entropy COEFFICIENT used in the PPO loss, so
the rollout policy, the stored log-probs and the clipped surrogate are untouched (the update stays valid PPO).

* ``anneal_fraction``: the entropy coefficient used to anneal with route progress only (``EntropySchedule``); with
  ``anneal_updates`` > 0 the anneal can also not run faster than ``update / anneal_updates``, so a lucky early progress
  spike cannot drop the coefficient to its floor within a few hundred updates.
* ``EntropyGuard``: a safety net. While ``update < until_update``, when the mean policy entropy of the previous update
  falls below ``threshold`` the coefficient is raised to ``boost_coef`` (at least; never below the scheduled value)
  until the entropy has recovered to ``threshold + margin`` (hysteresis, so it does not flicker around the threshold).

Pure Python (no numpy / torch), importable from the cloud launcher's config chain.
"""

from __future__ import annotations


def anneal_fraction(route_progress: float, update: int, anneal_updates: int) -> float:
    """Fraction in [0, 1] fed to ``EntropySchedule.value``: route progress, capped by ``update / anneal_updates``
    when ``anneal_updates`` > 0 (0 = progress only, the old behaviour)."""
    p = min(1.0, max(0.0, float(route_progress)))
    if anneal_updates and anneal_updates > 0:
        p = min(p, min(1.0, max(0.0, float(update) / float(anneal_updates))))
    return p


def validate_params(threshold, until_update, boost_coef, margin) -> list[str]:
    errs: list[str] = []
    if threshold <= 0:
        errs.append(f"entropy_boost_threshold must be > 0 (got {threshold})")
    if until_update < 0:
        errs.append(f"entropy_boost_until must be >= 0 (got {until_update})")
    if boost_coef < 0:
        errs.append(f"entropy_boost_coef must be >= 0 (got {boost_coef})")
    if margin < 0:
        errs.append(f"entropy_boost_margin must be >= 0 (got {margin})")
    return errs


class EntropyGuard:
    """``coef(update, scheduled, last_entropy) -> (coefficient, boosted)``; call once per update before the rollout."""

    def __init__(self, threshold: float = 0.8, until_update: int = 300, boost_coef: float = 0.05, margin: float = 0.2):
        errs = validate_params(threshold, until_update, boost_coef, margin)
        if errs:
            raise ValueError("; ".join(errs))
        self.threshold, self.until_update = float(threshold), int(until_update)
        self.boost_coef, self.margin = float(boost_coef), float(margin)
        self.active = False
        self.triggers = 0   # number of times the boost switched on

    def coef(self, update: int, scheduled: float, last_entropy: float | None) -> tuple[float, bool]:
        if update >= self.until_update:
            self.active = False
            return float(scheduled), False
        if last_entropy is not None:
            if not self.active and last_entropy < self.threshold:
                self.active = True
                self.triggers += 1
            elif self.active and last_entropy >= self.threshold + self.margin:
                self.active = False
        if self.active:
            return max(float(scheduled), self.boost_coef), True
        return float(scheduled), False
