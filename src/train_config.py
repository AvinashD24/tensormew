"""Single place for every training knob, with presets, JSON round trip and validation.

Presets: ``long_v3`` (``long_v2`` plus staggered episodes, a self-generated battle archive with an early A-bias in
battle menus, furthest-map weighted archive resets, and a dwell penalty for staying in one small area without progress
that is exempt while the party is below the target level for the story stage; the recommended one), ``long_v2`` (fresh from-scratch run, no
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

from src import dwell as _dwell

PRESETS = ("blueprint_v1", "legacy", "long_v1", "long_v2", "long_v3")


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
