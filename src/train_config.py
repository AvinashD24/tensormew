"""Single place for every training knob, with presets, JSON round trip and validation.

Presets: ``blueprint_v1`` (defaults from the long-horizon research) and ``legacy`` (the old trainer).
The PPO fields mirror ``src.ppo_update.PPOHParams`` as plain values so this module has no heavy imports.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

PRESETS = ("blueprint_v1", "legacy")


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
    # output
    save_path: str = "logs/deep_policy.pt"

    def validate(self) -> "TrainConfig":
        errs: list[str] = []
        for key in ("envs", "rollout_steps", "updates", "max_steps"):
            if getattr(self, key) < 1:
                errs.append(f"{key} must be >= 1 (got {getattr(self, key)})")
        if self.lr <= 0:
            errs.append(f"lr must be > 0 (got {self.lr})")
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
        if not self.save_path:
            errs.append("save_path must not be empty")
        if errs:
            raise ConfigError("invalid training config:\n  - " + "\n  - ".join(errs))
        return self

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


def preset(name: str) -> TrainConfig:
    if name == "blueprint_v1":
        return blueprint_v1()
    if name == "legacy":
        return legacy()
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
