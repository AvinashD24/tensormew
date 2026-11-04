"""Single place for every training knob, with presets, JSON round trip and validation.

Presets: ``long_v6`` (``long_v5`` plus a route-checkpoint archive that restarts episodes along the agents' own best
route progress, a dialogue-loop penalty and a reward-scaler warm-up rollout, after v5-1 got stuck at the end of Route 29),
``long_v5`` (``long_v4`` plus anti-collapse changes after the v4-1 run collapsed to pressing A: wall-bump and
tile-revisit penalties off while the path reward is on, a small penalty for A / B presses that change nothing, a 0.3
path reward per tile, a higher and slower-annealed entropy coefficient and an entropy-boost safety net),
``long_v4`` (``long_v3`` plus the dense best-ever path-progress reward, ``src.path_progress``, a smaller and
decaying exploration budget, a heal-for-HP-restored reward, low-HP battle practice, auxiliary prediction heads, gamma
0.999 and a lower entropy start; recommended),
``long_v3`` (``long_v2`` plus staggered episodes, a self-generated battle archive with an early A-bias in
battle menus, furthest-map weighted archive resets, and a dwell penalty for staying in one small area without progress
that is exempt while the party is below the target level for the story stage), ``long_v2`` (fresh from-scratch run, no
oracle, learned battles), and the older
``blueprint_v1`` / ``long_v1`` / ``legacy`` (same environment and observation now, only the hyperparameters and start
states differ; their old curriculum start states carry egg / rival flags from the removed errand and are LEGACY).
The PPO fields mirror ``src.ppo_update.PPOHParams`` as plain values so this module has no heavy imports.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from src import dialogue_loop as _dialogue_loop
from src import dwell as _dwell
from src import entropy_guard as _entropy_guard
from src import idle_press as _idle_press
from src import path_progress as _path_progress
from src import route_ckpt as _route_ckpt
from src import stagnation as _stagnation

PRESETS = ("blueprint_v1", "legacy", "long_v1", "long_v2", "long_v3", "long_v4", "long_v5", "long_v6")


class ConfigError(ValueError):
    """Raised for unknown keys, bad values or unknown presets."""


@dataclass
class PPOConfig:
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.03  # nominal; the schedule below overrides it per update
    max_grad_norm: float = 0.5
    epochs: int = 3
    num_minibatches: int = 4
    target_kl: float | None = 0.05
    norm_adv: bool = True
    # Off by default: a fixed +/-0.2 value clip slows the critic when returns are large. With reward_scale
    # "running_std" returns have std ~1 and the clip (itself scaled by the return std) is mild, so it can be turned on.
    vf_clip: bool = False
    adv_clip: float | None = None
    bn_mode: str = "eval_refresh"  # "train" = old behaviour (BatchNorm batch stats in the update; legacy preset)
    bn_refresh_frames: int = 256


@dataclass
class TrainConfig:
    name: str = "blueprint_v1"
    run_id: str = ""
    seed: int = 0
    # scale
    envs: int = 16
    rollout_steps: int = 128
    updates: int = 50
    max_steps: int = 1024
    lr: float = 3e-4
    lr_end_frac: float = 1.0  # linear decay of lr to lr * lr_end_frac over `updates`; 1.0 = constant (old behaviour)
    recurrent: bool = True
    reward_scale: str = "running_std"  # divide rewards by the running std of the discounted return; "none" = raw
    # PPO
    ppo: PPOConfig = field(default_factory=PPOConfig)
    entropy_start: float = 0.03
    entropy_end: float = 0.005
    # environment robustness
    sticky_prob: float = 0.1
    jitter_max: int = 6
    # starts / sharing
    start_states: list[str] = field(default_factory=list)
    stage_balanced: bool = True
    swarm: bool = True
    swarm_prob: float = 0.3  # share of resets (after the first episode) that load a swarm-archive state
    # Furthest-map resets (OFF by default): weight archive states by story stage, the furthest stage reached so far
    # gets weight 1 and stage s gets floor + (1 - floor) * s / furthest, so earlier stages still get some weight.
    swarm_furthest_weighting: bool = False
    swarm_weight_floor: float = 0.15
    # Staggered episodes (OFF by default): each env's first episode starts with a random step offset (evenly spread
    # slots over the episode length) and every episode length is jittered by +-episode_len_jitter, so envs do not
    # reset in lockstep (all resets at the same update = an 8-update reward cycle with 2048 / 256 steps).
    stagger_episodes: bool = False
    episode_len_jitter: float = 0.0
    # Battle practice (OFF by default): a second self-generated archive of states saved when a battle starts (up to
    # battle_archive_max, keyed by battle kind / map / enemy species); battle_archive_prob of resets load one.
    battle_archive: bool = False
    battle_archive_prob: float = 0.0
    battle_archive_max: int = 32
    # Early-training bias added to the A-button logit while the observation's in-battle flag is set, applied inside the
    # policy (rollout and PPO update see the same distribution). Decays linearly to 0 over battle_a_bias_updates.
    battle_a_bias: float = 0.0
    battle_a_bias_updates: int = 0
    # OPTIONAL tile-level potential toward Falkner (BFS steps, src.route_graph). OFF by default and in every preset:
    # it is an oracle (a hand-computed gradient toward the goal) kept only for ablations / debugging.
    goal_potential: bool = False
    # Dense path-progress reward (OFF by default, ON in long_v4; see src.path_progress): path_progress_scale per tile of
    # NEW best progress along the 319-step story path (BFS over the ROM graph), once per tile, never negative. Reward
    # shaping only: nothing is added to the observation. Not combinable with the older goal_potential.
    path_progress: bool = False
    path_progress_scale: float = 0.08
    # Dwell penalty (OFF by default, see src.dwell): a cost for staying in <= dwell_max_tiles tiles for dwell_window
    # overworld steps with nothing new happening; ramps dwell_min_penalty -> dwell_max_penalty over dwell_ramp_steps,
    # at most dwell_budget per episode. Zero while the party's highest level is below dwell_level_targets[furthest story
    # stage] and the agent fought / leveled within dwell_grind_grace steps (that exemption also switches the old per-tile
    # `loop` term off when dwell_exempt_loop is set). Pokecenter maps are always exempt.
    dwell_penalty: bool = False
    dwell_window: int = _dwell.DWELL_WINDOW
    dwell_max_tiles: int = _dwell.DWELL_MAX_TILES
    dwell_min_penalty: float = _dwell.DWELL_MIN_PENALTY
    dwell_max_penalty: float = _dwell.DWELL_MAX_PENALTY
    dwell_ramp_steps: int = _dwell.DWELL_RAMP_STEPS
    dwell_budget: float = _dwell.DWELL_BUDGET
    dwell_grind_grace: int = _dwell.DWELL_GRIND_GRACE
    dwell_level_targets: list[int] = field(default_factory=lambda: list(_dwell.DWELL_LEVEL_TARGETS))
    dwell_exempt_loop: bool = True
    # Exploration budget (long_v4). ``explore_budget`` is the per-episode cap on the summed novelty payments (curiosity,
    # new tiles / neighbours / maps / warps / interactions, momentum); 30.0 = ``src.rl_env.EXPLORE_BUDGET`` (the
    # unchanged default). The intrinsic weight multiplies those payments and decays linearly from 1.0 to
    # ``intrinsic_weight_end`` over ``intrinsic_decay_updates`` updates (0 = no decay). Reward shaping only.
    explore_budget: float = 30.0
    intrinsic_weight_end: float = 1.0
    intrinsic_decay_updates: int = 0
    # Healing reward for HP actually restored, weighted toward low HP (OFF by default, see src.heal_reward). It replaces
    # the flat ``heal`` terms; ``heal_budget`` caps the total paid per episode.
    heal_reward: bool = False
    heal_reward_scale: float = 4.0
    heal_budget: float = 8.0
    # Low-HP battle practice (0 = off): in the battle archive, states whose lead Pokemon is at <= battle_lowhp_threshold
    # HP while the bag holds a heal item get sampling weight 1 + battle_lowhp_weight (others 1). Needs battle_archive.
    battle_lowhp_weight: float = 0.0
    battle_lowhp_threshold: float = 0.4
    # Auxiliary prediction heads (OFF by default, see src.aux_tasks): predict, from the recurrent features, the map id,
    # the lead-HP change and whether a battle starts within ``aux_horizon`` steps. Loss weight ``aux_coef``;
    # the targets come from the stored RAM observations, so they cost no emulator time. Needs ``recurrent``.
    aux_loss: bool = False
    aux_coef: float = 0.05
    aux_horizon: int = 8
    # --- long_v5 anti-collapse fields (all OFF / unchanged by default and in every older preset) ---
    # With the dense path reward on, the wall-bump (``bump``) and per-tile revisit (``loop``) penalties can be switched
    # off: dwell + path progress already cover idling and wall hugging, and both costs are movement-linked, so they
    # made standing still (pressing A) the cheapest policy in the v4-1 run. Step, damage and whiteout stay.
    disable_bump_penalty: bool = False
    disable_loop_penalty: bool = False
    # Small cost per A / B press in the plain overworld that changes nothing observable (src.idle_press), at most
    # idle_press_budget per episode. Reported as the ``idle_press`` reward component.
    idle_press_penalty: bool = False
    idle_press_cost: float = _idle_press.IDLE_PRESS_COST
    idle_press_budget: float = _idle_press.IDLE_PRESS_BUDGET
    # With entropy_anneal_updates > 0 the entropy coefficient cannot anneal faster than update / entropy_anneal_updates
    # (it is otherwise annealed by route progress only). 0 = progress only (the old behaviour).
    entropy_anneal_updates: int = 0
    # Entropy safety net (src.entropy_guard): while update < entropy_boost_until, when the previous update's mean policy
    # entropy is below entropy_boost_threshold the coefficient is raised to entropy_boost_coef until the entropy has
    # recovered to threshold + entropy_boost_margin. Logged as the ``entropy_boost`` metric (1 = boosted).
    entropy_boost: bool = False
    entropy_boost_threshold: float = 0.8
    entropy_boost_until: int = 300
    entropy_boost_coef: float = 0.05
    entropy_boost_margin: float = 0.2
    # --- long_v6 fields (all OFF / unchanged by default and in every older preset) ---
    # Route-checkpoint archive (src.route_ckpt; Go-Explore / reverse curriculum): workers save an emulator state whenever
    # the path progress reaches a new multiple of route_ckpt_every tiles (up to route_ckpt_per_bucket states per
    # bucket, route_ckpt_max files in all); route_ckpt_prob of the resets load one, buckets weighted
    # floor + (1 - floor) * bucket / max_bucket. Reset order: battle archive, swarm archive, route checkpoints, start
    # state; the three shares must sum to <= 1. Needs path_progress.
    route_ckpt: bool = False
    route_ckpt_every: int = _route_ckpt.ROUTE_CKPT_EVERY
    route_ckpt_prob: float = 0.0
    route_ckpt_per_bucket: int = _route_ckpt.ROUTE_PER_BUCKET
    route_ckpt_max: int = _route_ckpt.ROUTE_MAX_FILES
    route_ckpt_weight_floor: float = _route_ckpt.ROUTE_WEIGHT_FLOOR
    # Dialogue-loop penalty (src.dialogue_loop): small cost per step of repeated, effect-less A-driven dialogue.
    dialogue_loop_penalty: bool = False
    dialogue_loop_window: int = _dialogue_loop.DIALOGUE_LOOP_WINDOW
    dialogue_loop_max_steps: int = _dialogue_loop.DIALOGUE_LOOP_MAX_STEPS
    dialogue_loop_max_talks: int = _dialogue_loop.DIALOGUE_LOOP_MAX_TALKS
    dialogue_loop_cost: float = _dialogue_loop.DIALOGUE_LOOP_COST
    dialogue_loop_budget: float = _dialogue_loop.DIALOGUE_LOOP_BUDGET
    # Warm-up rollouts before update 0 (0 = none): their rewards only initialise the running reward scaler (RND,
    # Burda et al. 2018, does the same for its normalisation statistics). No gradient step, not logged as an update.
    warmup_rollout: int = 0
    # Retreat cost (src.path_progress, reward only): charge retreat_cost_frac * path_progress_scale per tile of path
    # progress lost on every walked step that moves further from the goal, at most retreat_budget per episode; re-approach
    # is free. A soft asymmetric potential (NOT policy-invariant, unlike Ng et al. 1999). Needs path_progress.
    retreat_penalty: bool = False
    retreat_cost_frac: float = _path_progress.RETREAT_FRAC
    retreat_budget: float = _path_progress.RETREAT_BUDGET
    # Stagnation penalty (src.stagnation): no new best path progress for stagnation_after counted steps costs
    # stagnation_min -> stagnation_max per step over stagnation_ramp steps (at most stagnation_budget per episode);
    # composed with the dwell penalty as max(), not a sum. Needs path_progress and dwell_penalty (level exemption).
    stagnation_penalty: bool = False
    stagnation_after: int = _stagnation.STAGNATION_AFTER
    stagnation_min: float = _stagnation.STAGNATION_MIN
    stagnation_max: float = _stagnation.STAGNATION_MAX
    stagnation_ramp: int = _stagnation.STAGNATION_RAMP
    stagnation_budget: float = _stagnation.STAGNATION_BUDGET
    # A white-out (the whole party fainted) ENDS the episode as a true termination (done, not a time-limit truncation:
    # no value bootstrap) instead of teleporting the agent and continuing. The -15 white-out penalty is the terminal reward.
    terminate_on_whiteout: bool = False
    # output
    save_path: str = "logs/deep_policy.pt"

    def validate(self) -> "TrainConfig":
        errs: list[str] = []
        for key in ("envs", "rollout_steps", "updates", "max_steps"):
            if getattr(self, key) < 1:
                errs.append(f"{key} must be >= 1 (got {getattr(self, key)})")
        if self.lr <= 0:
            errs.append(f"lr must be > 0 (got {self.lr})")
        if not 0 < self.lr_end_frac <= 1:
            errs.append(f"lr_end_frac must be in (0, 1] (got {self.lr_end_frac})")
        p = self.ppo
        if not 0 < p.gamma <= 1:
            errs.append(f"ppo.gamma must be in (0, 1] (got {p.gamma})")
        if not 0 <= p.gae_lambda <= 1:
            errs.append(f"ppo.gae_lambda must be in [0, 1] (got {p.gae_lambda})")
        if p.clip_ratio <= 0:
            errs.append(f"ppo.clip_ratio must be > 0 (got {p.clip_ratio})")
        if p.epochs < 1:
            errs.append(f"ppo.epochs must be >= 1 (got {p.epochs})")
        if p.num_minibatches < 1:
            errs.append(f"ppo.num_minibatches must be >= 1 (got {p.num_minibatches})")
        elif self.recurrent and p.num_minibatches > self.envs:
            errs.append(
                f"ppo.num_minibatches ({p.num_minibatches}) cannot exceed envs ({self.envs}); "
                "minibatches are groups of whole environments (uneven groups are fine)"
            )
        if p.bn_mode not in ("eval_refresh", "train"):
            errs.append(f"ppo.bn_mode must be 'eval_refresh' or 'train' (got {p.bn_mode!r})")
        if self.reward_scale not in ("running_std", "none"):
            errs.append(f"reward_scale must be 'running_std' or 'none' (got {self.reward_scale!r})")
        if p.target_kl is not None and p.target_kl <= 0:
            errs.append(f"ppo.target_kl must be > 0 or null (got {p.target_kl})")
        if self.entropy_start < 0 or self.entropy_end < 0:
            errs.append("entropy_start and entropy_end must be >= 0")
        if self.entropy_end > self.entropy_start:
            errs.append(f"entropy_end ({self.entropy_end}) must not exceed entropy_start ({self.entropy_start})")
        if not 0 <= self.sticky_prob < 1:
            errs.append(f"sticky_prob must be in [0, 1) (got {self.sticky_prob})")
        if self.jitter_max < 0:
            errs.append(f"jitter_max must be >= 0 (got {self.jitter_max})")
        if not 0 <= self.swarm_prob <= 1:
            errs.append(f"swarm_prob must be in [0, 1] (got {self.swarm_prob})")
        if not 0 <= self.battle_archive_prob <= 1:
            errs.append(f"battle_archive_prob must be in [0, 1] (got {self.battle_archive_prob})")
        if self.swarm_prob + (self.battle_archive_prob if self.battle_archive else 0.0) > 1:
            errs.append("swarm_prob + battle_archive_prob must not exceed 1")
        if not 0 < self.swarm_weight_floor <= 1:
            errs.append(f"swarm_weight_floor must be in (0, 1] (got {self.swarm_weight_floor})")
        if not 0 <= self.episode_len_jitter < 1:
            errs.append(f"episode_len_jitter must be in [0, 1) (got {self.episode_len_jitter})")
        if self.battle_archive_max < 1:
            errs.append(f"battle_archive_max must be >= 1 (got {self.battle_archive_max})")
        if self.battle_a_bias < 0:
            errs.append(f"battle_a_bias must be >= 0 (got {self.battle_a_bias})")
        if self.battle_a_bias_updates < 0:
            errs.append(f"battle_a_bias_updates must be >= 0 (got {self.battle_a_bias_updates})")
        if self.path_progress_scale < 0:
            errs.append(f"path_progress_scale must be >= 0 (got {self.path_progress_scale})")
        if self.path_progress and self.goal_potential:
            errs.append("path_progress and goal_potential both reward route progress; enable only one")
        if self.explore_budget < 0:
            errs.append(f"explore_budget must be >= 0 (got {self.explore_budget})")
        if not 0 <= self.intrinsic_weight_end <= 1:
            errs.append(f"intrinsic_weight_end must be in [0, 1] (got {self.intrinsic_weight_end})")
        if self.intrinsic_decay_updates < 0:
            errs.append(f"intrinsic_decay_updates must be >= 0 (got {self.intrinsic_decay_updates})")
        if self.heal_reward_scale < 0 or self.heal_budget < 0:
            errs.append("heal_reward_scale and heal_budget must be >= 0")
        if self.battle_lowhp_weight < 0:
            errs.append(f"battle_lowhp_weight must be >= 0 (got {self.battle_lowhp_weight})")
        if not 0 < self.battle_lowhp_threshold <= 1:
            errs.append(f"battle_lowhp_threshold must be in (0, 1] (got {self.battle_lowhp_threshold})")
        if self.battle_lowhp_weight > 0 and not self.battle_archive:
            errs.append("battle_lowhp_weight > 0 needs battle_archive")
        if self.aux_coef < 0:
            errs.append(f"aux_coef must be >= 0 (got {self.aux_coef})")
        if self.aux_horizon < 1:
            errs.append(f"aux_horizon must be >= 1 (got {self.aux_horizon})")
        if self.aux_loss and not self.recurrent:
            errs.append("aux_loss needs a recurrent policy (its targets are k steps ahead inside a rollout sequence)")
        if self.aux_loss and self.aux_horizon >= self.rollout_steps:
            errs.append(f"aux_horizon ({self.aux_horizon}) must be below rollout_steps ({self.rollout_steps})")
        if self.entropy_anneal_updates < 0:
            errs.append(f"entropy_anneal_updates must be >= 0 (got {self.entropy_anneal_updates})")
        if (self.disable_bump_penalty or self.disable_loop_penalty) and not self.path_progress:
            errs.append("disable_bump_penalty / disable_loop_penalty need path_progress (nothing else would reward moving)")
        errs.extend(_route_ckpt.validate_params(
            self.route_ckpt_every, self.route_ckpt_prob, self.route_ckpt_per_bucket, self.route_ckpt_max,
            self.route_ckpt_weight_floor))
        if self.route_ckpt:
            if not self.path_progress:
                errs.append("route_ckpt needs path_progress (buckets are path-progress tiles)")
            share = ((self.swarm_prob if self.swarm else 0.0) + (self.battle_archive_prob if self.battle_archive else 0.0)
                     + self.route_ckpt_prob)
            if share > 1.0 + 1e-9:
                errs.append(f"swarm_prob + battle_archive_prob + route_ckpt_prob must not exceed 1 (got {share:.3f})")
        errs.extend(_dialogue_loop.validate_params(
            self.dialogue_loop_window, self.dialogue_loop_max_steps, self.dialogue_loop_max_talks,
            self.dialogue_loop_cost, self.dialogue_loop_budget))
        errs.extend(_path_progress.validate_retreat(self.retreat_cost_frac, self.retreat_budget))
        if self.retreat_penalty and not self.path_progress:
            errs.append("retreat_penalty needs path_progress (it is measured in path-progress tiles)")
        errs.extend(_stagnation.validate_params(
            self.stagnation_after, self.stagnation_min, self.stagnation_max, self.stagnation_ramp, self.stagnation_budget))
        if self.stagnation_penalty and not (self.path_progress and self.dwell_penalty):
            errs.append("stagnation_penalty needs path_progress (new-best signal) and dwell_penalty (level exemption, no double count)")
        if self.warmup_rollout < 0:
            errs.append(f"warmup_rollout must be >= 0 (got {self.warmup_rollout})")
        errs.extend(_idle_press.validate_params(self.idle_press_cost, self.idle_press_budget))
        errs.extend(_entropy_guard.validate_params(
            self.entropy_boost_threshold, self.entropy_boost_until, self.entropy_boost_coef, self.entropy_boost_margin))
        errs.extend(_dwell.validate_params(
            self.dwell_window, self.dwell_max_tiles, self.dwell_min_penalty, self.dwell_max_penalty,
            self.dwell_ramp_steps, self.dwell_budget, self.dwell_grind_grace, self.dwell_level_targets))
        if not self.save_path:
            errs.append("save_path must not be empty")
        if errs:
            raise ConfigError("invalid training config:\n  - " + "\n  - ".join(errs))
        return self

    def dwell_params(self) -> dict | None:
        """Keyword arguments for ``src.dwell.DwellTracker``, or None when the penalty is off."""
        if not self.dwell_penalty:
            return None
        return {
            "window": self.dwell_window, "max_tiles": self.dwell_max_tiles, "min_penalty": self.dwell_min_penalty,
            "max_penalty": self.dwell_max_penalty, "ramp_steps": self.dwell_ramp_steps, "budget": self.dwell_budget,
            "grind_grace": self.dwell_grind_grace, "level_targets": list(self.dwell_level_targets),
        }

    def idle_press_params(self) -> dict | None:
        """Keyword arguments for ``src.idle_press.IdlePressTracker``, or None when the penalty is off."""
        if not self.idle_press_penalty:
            return None
        return {"cost": self.idle_press_cost, "budget": self.idle_press_budget}

    def retreat_params(self) -> dict | None:
        """``retreat`` argument of ``PathProgress.from_scale``, or None when the retreat cost is off."""
        if not self.retreat_penalty:
            return None
        return {"frac": self.retreat_cost_frac, "budget": self.retreat_budget}

    def stagnation_params(self) -> dict | None:
        """Keyword arguments for ``src.stagnation.StagnationTracker``, or None when the penalty is off."""
        if not self.stagnation_penalty:
            return None
        return {"after": self.stagnation_after, "min_penalty": self.stagnation_min,
                "max_penalty": self.stagnation_max, "ramp_steps": self.stagnation_ramp,
                "budget": self.stagnation_budget}

    def dialogue_loop_params(self) -> dict | None:
        """Keyword arguments for ``src.dialogue_loop.DialogueLoopTracker``, or None when the penalty is off."""
        if not self.dialogue_loop_penalty:
            return None
        return {"window": self.dialogue_loop_window, "max_steps": self.dialogue_loop_max_steps,
                "max_talks": self.dialogue_loop_max_talks, "cost": self.dialogue_loop_cost,
                "budget": self.dialogue_loop_budget}

    def entropy_guard_params(self) -> dict | None:
        """Keyword arguments for ``src.entropy_guard.EntropyGuard``, or None when the safety net is off."""
        if not self.entropy_boost:
            return None
        return {"threshold": self.entropy_boost_threshold, "until_update": self.entropy_boost_until,
                "boost_coef": self.entropy_boost_coef, "margin": self.entropy_boost_margin}

    def path_progress_scale_or_none(self) -> float | None:
        """Scale for ``src.path_progress.PathProgress``, or None when the reward is off."""
        return float(self.path_progress_scale) if self.path_progress else None

    def explore_cap_override(self) -> float | None:
        """The explore cap for the worker budget, or None when it is the unchanged default (30.0)."""
        return None if abs(float(self.explore_budget) - 30.0) < 1e-9 else float(self.explore_budget)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def legacy() -> TrainConfig:
    """The old trainer: gamma 0.99, 4 whole-batch epochs, entropy 0.02, no sticky actions or jitter."""
    return TrainConfig(
        name="legacy",
        ppo=PPOConfig(
            gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, entropy_coef=0.02, epochs=4, num_minibatches=1,
            target_kl=None, vf_clip=False, bn_mode="train",
        ),
        reward_scale="none",
        entropy_start=0.02,
        entropy_end=0.02,
        sticky_prob=0.0,
        jitter_max=0,
        stage_balanced=False,
        swarm=False,
    )


def blueprint_v1() -> TrainConfig:
    return TrainConfig()


def long_v1() -> TrainConfig:
    """LEGACY (same environment as long_v2 now; kept for its hyperparameters). Unattended ~6 hour cloud run from a fresh policy: New Bark start, no curriculum (the swarm archive supplies
    later stages), 32 envs x 256 steps = 8192 samples per update, lr decaying to a third, a longer episode cap so a
    full egg round trip fits, and about 800 updates (~26 s each on a 4090 pod, ~5.8 h). Override with --updates."""
    cfg = TrainConfig(
        name="long_v1",
        envs=32,
        rollout_steps=256,
        updates=800,
        max_steps=2048,
        lr=3e-4,
        lr_end_frac=0.33,
        entropy_start=0.03,
        entropy_end=0.01,
        start_states=["new_bark_outside"],
    )
    cfg.ppo.num_minibatches = 8  # 4 envs x 256 steps = 1024 sequences per minibatch
    return cfg


def long_v2() -> TrainConfig:
    """Fresh from-scratch run from ``states/new_bark_outside.state`` (Cyndaquil, outside New Bark Town).

    No route oracle, learned battles, minimal story path (Route 29 -> Cherrygrove -> Route 30 -> Route 31 ->
    Violet City -> Violet Gym -> Falkner). Compared to ``long_v1``: more entropy for longer (menu navigation in
    battle has to be discovered), a longer episode cap (the path is ~320 steps but battles cost steps), 32 envs x 256
    steps per update, lr decaying to 0.33x, ~800 updates (~6 h on a 4090 pod). The swarm archive (first env to reach a
    new stage saves a state, 30% of resets load one) supplies later-stage starts; stages are cut by map, not by flags.
    """
    cfg = TrainConfig(
        name="long_v2",
        envs=32,
        rollout_steps=256,
        updates=800,
        max_steps=2048,
        lr=3e-4,
        lr_end_frac=0.33,
        entropy_start=0.04,
        entropy_end=0.01,
        start_states=["new_bark_outside"],
        sticky_prob=0.1,
        jitter_max=6,
    )
    cfg.ppo.num_minibatches = 8  # 4 envs x 256 steps = 1024 sequences per minibatch
    return cfg


def long_v3() -> TrainConfig:
    """``long_v2`` plus three self-generated mechanisms (each behind its own flag, so they can be ablated):

    1. staggered episodes: random per-env initial step offset + +-25% episode length jitter, so resets spread over
       the updates instead of all 24-32 envs resetting in the same update (``phase_spread`` is logged);
    2. battle practice: a battle archive (states saved when a battle starts, up to 32, 15% of resets load one) and a
       0.5 -> 0 (over 150 updates) A-button logit bias while the in-battle flag is set, applied inside the policy;
    3. furthest-map resets: 50% of resets load a swarm-archive state (was 30%), weighted toward the furthest story
       stage reached so far (floor 0.15 for earlier stages).
    4. dwell penalty (``src.dwell``): staying in <= 16 tiles for 192 non-battle steps with nothing new (tile / map /
       milestone / species) costs 0.005 -> 0.03 per step (ramp over 384 steps, at most 15 per episode, reset by leaving
       the area). Zero while the party's highest level is below the target for the furthest story stage (8 before
       Cherrygrove, 10 on Cherrygrove / Routes 30-31, 11 at the Violet gate, 12 in Violet City / the Gym) and the agent
       is actually fighting, so leveling for Falkner is not punished; the old per-tile ``loop`` term is off then too.
    Everything else (start state, no oracle, hyperparameters) is ``long_v2``.
    """
    cfg = long_v2()
    cfg.name = "long_v3"
    cfg.dwell_penalty = True
    cfg.stagger_episodes = True
    cfg.episode_len_jitter = 0.25
    cfg.swarm_prob = 0.5
    cfg.swarm_furthest_weighting = True
    cfg.swarm_weight_floor = 0.15
    cfg.battle_archive = True
    cfg.battle_archive_prob = 0.15
    cfg.battle_archive_max = 32
    cfg.battle_a_bias = 0.5
    cfg.battle_a_bias_updates = 150
    return cfg


def long_v4() -> TrainConfig:
    """``long_v3`` plus the dense best-ever path-progress reward and the learning-speed changes below.

    ``path_progress`` pays ``path_progress_scale`` (0.08) per tile of NEW best progress along the 319-step minimal
    story path (``src.path_progress``): monotone per episode, nothing for stepping back, nothing for re-walking, the
    start state's own progress is free, a new best resets the dwell window. The whole path pays about 25.5 in total,
    next to ~140 of milestone bonuses. Reward-only: the observation is exactly ``long_v3``'s.

    Added on top (each behind its own flag/field; literature in ``research/analysis/design_hardcoded_vs_learned.md``):

    * horizon: ``gamma`` 0.997 -> 0.999 (horizon 1000 steps instead of 333: the story path is 319 steps but an episode
      is up to 2048 steps with battles; 0.997**1000 = 0.05 hides the last milestones from the start, 0.999**1000 =
      0.37). RND (Burda et al. 2018) used gamma_E = 0.999 and found it clearly better than 0.99. ``gae_lambda`` stays 0.95
      (OpenAI Five, Pleines et al. 2025). The recurrent state is carried across rollouts and the whole 256-step rollout
      is one BPTT sequence (unchanged);
    * exploration budget: ``explore_budget`` 30 -> 12 per episode and the intrinsic weight decays 1.0 -> 0.4 over 800
      updates (the live v3-1 run paid ~30 of exploration per episode vs 1.8 of milestones, 71 % of the positive reward);
    * ``heal_reward``: pays for HP actually restored, concave in the HP fraction (low-HP heals pay most), budget 8;
    * ``battle_lowhp_weight`` 3: the battle archive favours states with a low-HP lead and a heal item in the bag;
    * ``aux_loss``: auxiliary map / lead-HP-change / battle-start prediction heads, weight 0.05, horizon 8 steps;
    * entropy 0.01 -> 0.001 (was 0.04 -> 0.01; the live run sat at entropy 1.70 of 1.79).
    """
    cfg = long_v3()
    cfg.name = "long_v4"
    cfg.path_progress = True
    cfg.path_progress_scale = 0.08
    cfg.ppo.gamma = 0.999
    cfg.entropy_start = 0.01
    cfg.entropy_end = 0.001
    cfg.explore_budget = 12.0
    cfg.intrinsic_weight_end = 0.4
    cfg.intrinsic_decay_updates = 800
    cfg.heal_reward = True
    cfg.heal_reward_scale = 4.0
    cfg.heal_budget = 8.0
    cfg.battle_lowhp_weight = 3.0
    cfg.battle_lowhp_threshold = 0.4
    cfg.aux_loss = True
    cfg.aux_coef = 0.05
    cfg.aux_horizon = 8
    return cfg


def long_v5() -> TrainConfig:
    """``long_v4`` plus anti-collapse changes (each behind its own flag, so they can be ablated).

    Live evidence (run v4-1, ``long_v4``, 75 updates, 24 envs): the policy collapsed to pressing A (58 % of actions),
    entropy 1.78 -> 1.12 (update 38) -> 0.40 (update 75), path progress mean best only 8-22 of 319 tiles. An A press
    does not move the character, so it avoids the movement-linked ``bump`` (-37 per update) and ``loop`` (-54)
    penalties, while the path reward (0.08 / tile) was tiny next to them; with a low entropy coefficient (0.01 -> 0.001)
    and gamma 0.999 PPO locked the cheapest behaviour in. Changes:

    1. ``disable_bump_penalty`` / ``disable_loop_penalty``: both off while ``path_progress`` is on (dwell + path progress
       cover idling and wall hugging). Step, damage and whiteout stay;
    2. ``idle_press_penalty``: 0.015 per A / B press in the overworld that changes nothing observable (no dialogue,
       battle, position, party / bag / flag change), at most 10 per episode (``src.idle_press``);
    3. ``path_progress_scale`` 0.08 -> 0.3 (the whole 319-tile path pays ~96 instead of ~25.5, next to ~140 of
       milestone bonuses and an exploration cap of 12);
    4. entropy: start 0.02 (was 0.01), floor 0.005 (was 0.001), and the anneal cannot run faster than
       update / 1200 (``entropy_anneal_updates``); the safety net ``entropy_boost`` raises the coefficient to 0.05 while
       the mean policy entropy is below 0.8 before update 300, releasing at 1.0 (``src.entropy_guard``).
    """
    cfg = long_v4()
    cfg.name = "long_v5"
    cfg.disable_bump_penalty = True
    cfg.disable_loop_penalty = True
    cfg.idle_press_penalty = True
    cfg.idle_press_cost = 0.015
    cfg.idle_press_budget = 10.0
    cfg.path_progress_scale = 0.3
    cfg.entropy_start = 0.02
    cfg.entropy_end = 0.005
    cfg.entropy_anneal_updates = 1200
    cfg.entropy_boost = True
    cfg.entropy_boost_threshold = 0.8
    cfg.entropy_boost_until = 300
    cfg.entropy_boost_coef = 0.05
    cfg.entropy_boost_margin = 0.2
    return cfg


def long_v6() -> TrainConfig:
    """``long_v5`` plus three changes (each behind its own flag, so they can be ablated), after run v5-1 (``long_v5``,
    update 317, 24 envs) got one agent 97 of 99 steps of Route 29 but typically only ~30, never reached Cherrygrove, spent
    19 % of its visits on one tile of the Route 29 gate building (a dialogue loop) and had a net reward of ~0:

    1. ``route_ckpt`` (``src.route_ckpt``): self-generated reverse curriculum / Go-Explore archive. A state is saved at
       every new multiple of 12 tiles of path progress (4 per bucket, 60 files); 40 % of the resets load one,
       weighted toward the furthest bucket (0.2 + 0.8 * bucket / max_bucket). Reset shares: battle 0.15, swarm 0.25
       (was 0.5; the swarm archive only holds map-entry states), route checkpoints 0.40, start state 0.20.
    2. ``dialogue_loop_penalty`` (``src.dialogue_loop``): 0.02 per step of repeated effect-less dialogue (more than 60 such
       steps in 200, or one tile talked to more than 6 times without effect), at most 10 per episode.
    3. ``warmup_rollout`` 1: one rollout before update 0 that only initialises the reward scaler. Every tile is novel at
       the start of a run, so update 0 pays curiosity + tile novelty (mean reward 12.4 vs ~0 later); without the warm-up
       the first update is scaled by a std that does not describe the steady state.

    4. ``retreat_penalty`` (``src.path_progress``): 0.35 x the path reward per tile of progress lost on each step further
       from the goal (re-approach free, at most 15 per episode); reward only, not policy-invariant.
    5. ``stagnation_penalty`` (``src.stagnation``): 0.004 -> 0.02 per step after 250 steps without a new best path
       progress (ramp 300, at most 10 per episode), max() with the dwell penalty, off while leveling is needed.
    6. ``terminate_on_whiteout``: a white-out ends the episode (true termination, no bootstrap; the -15 stays).

    Everything else (no bump / loop penalty, idle_press, path scale 0.3, entropy 0.02 -> 0.005 with the boost guard) is
    ``long_v5``.
    """
    cfg = long_v5()
    cfg.name = "long_v6"
    cfg.swarm_prob = 0.25
    cfg.route_ckpt = True
    cfg.route_ckpt_every = 12
    cfg.route_ckpt_prob = 0.4
    cfg.route_ckpt_per_bucket = 4
    cfg.route_ckpt_max = 60
    cfg.route_ckpt_weight_floor = 0.2
    cfg.dialogue_loop_penalty = True
    cfg.dialogue_loop_window = 200
    cfg.dialogue_loop_max_steps = 60
    cfg.dialogue_loop_max_talks = 6
    cfg.dialogue_loop_cost = 0.02
    cfg.dialogue_loop_budget = 10.0
    cfg.warmup_rollout = 1
    cfg.retreat_penalty = True
    cfg.retreat_cost_frac = 0.35
    cfg.retreat_budget = 15.0
    cfg.stagnation_penalty = True
    cfg.stagnation_after = 250
    cfg.stagnation_min = 0.004
    cfg.stagnation_max = 0.02
    cfg.stagnation_ramp = 300
    cfg.stagnation_budget = 10.0
    cfg.terminate_on_whiteout = True
    return cfg


def preset(name: str) -> TrainConfig:
    if name == "blueprint_v1":
        return blueprint_v1()
    if name == "legacy":
        return legacy()
    if name == "long_v1":
        return long_v1()
    if name == "long_v2":
        return long_v2()
    if name == "long_v3":
        return long_v3()
    if name == "long_v4":
        return long_v4()
    if name == "long_v5":
        return long_v5()
    if name == "long_v6":
        return long_v6()
    raise ConfigError(f"unknown preset '{name}' (known: {', '.join(PRESETS)})")


def _build(data: dict, base: TrainConfig) -> TrainConfig:
    """Apply a (possibly partial, possibly nested ``ppo``) dict on top of ``base``; reject unknown keys."""
    top = {f.name for f in dataclasses.fields(TrainConfig)}
    ppo_keys = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = dataclasses.replace(base, ppo=dataclasses.replace(base.ppo))
    for key, value in data.items():
        if key == "ppo":
            if not isinstance(value, dict):
                raise ConfigError("'ppo' must be an object")
            bad = set(value) - ppo_keys
            if bad:
                raise ConfigError(f"unknown ppo keys: {sorted(bad)}")
            cfg.ppo = dataclasses.replace(cfg.ppo, **value)
        elif key in top:
            setattr(cfg, key, list(value) if key == "start_states" else value)
        elif key in ppo_keys:  # flat shorthand, e.g. {"gamma": 0.99}
            cfg.ppo = dataclasses.replace(cfg.ppo, **{key: value})
        else:
            raise ConfigError(f"unknown config key '{key}'")
    return cfg


def from_dict(data: dict, base: TrainConfig | None = None) -> TrainConfig:
    return _build(data, base or blueprint_v1()).validate()


def load_config(path_or_preset: str) -> TrainConfig:
    """Load a preset by name, or a JSON file (missing keys fall back to ``blueprint_v1``, or to the preset
    named by the file's optional ``"preset"`` key)."""
    if path_or_preset in PRESETS:
        return preset(path_or_preset).validate()
    path = Path(path_or_preset)
    if not path.is_file():
        raise ConfigError(f"'{path_or_preset}' is neither a preset ({', '.join(PRESETS)}) nor an existing JSON file")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be an object")
    base = preset(data.pop("preset")) if "preset" in data else blueprint_v1()
    return from_dict(data, base)


def apply_overrides(cfg: TrainConfig, overrides: dict) -> TrainConfig:
    """Return a new validated config with ``overrides`` applied. ``None`` values are skipped so argparse
    defaults of None mean "not given"."""
    given = {k: v for k, v in overrides.items() if v is not None}
    return _build(given, cfg).validate()


def dump_effective_config(cfg: TrainConfig, run_dir: str | Path) -> Path:
    """Write the effective config to ``<run_dir>/config.json`` and return the path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.json"
    out.write_text(cfg.to_json() + "\n", encoding="utf-8")
    return out
