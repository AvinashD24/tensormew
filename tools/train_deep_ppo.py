"""High-throughput visual PPO training using SpatialVisionPolicy on vectorized PyBoy workers.

Parallel PyBoy processes step the game and stream raw screen buffers (144x160x3) plus the RAM/route feature vector to
one policy process that does batched inference and the PPO update (``src.ppo_update``: env-wise minibatches, GAE with
truncation bootstrapping, entropy schedule).

Every knob lives in ``src.train_config.TrainConfig``.  ``--config`` picks a preset (``long_v2`` for a fresh run from
the New Bark start state; ``blueprint_v1``, ``long_v1``, ``legacy`` are older presets) or a JSON file; every other CLI
flag defaults to None and overrides the config only when it is typed.

What the policy controls (nothing here is scripted any more): movement, A / B, and the whole battle (menus, moves,
items, switching) with raw button presses.  Only overworld dialogue is auto-advanced with ``mash_text``.  There is no
route oracle in the observation (``src.rl_env.observe(deep=True)``: state, story milestones, battle features).

Training-only environment features (all off in ``--config legacy``):
- sticky actions: with probability ``sticky_prob`` the executed action repeats the previous executed move (never in
  battle, where a repeated direction would corrupt menu navigation); PPO stores the action the policy sampled;
- start jitter: 0..``jitter_max`` random safe moves after each reset (no reward, nothing added to the seen sets);
- stage-balanced start states: each story stage is sampled evenly;
- swarm archive (optional, file based): the first env to reach a new stage or key event saves its emulator state under
  ``<logs>/swarm/<run-id>/``; on reset an env loads from that archive with probability ``swarm_prob`` (default
  ``SWARM_SAMPLE_PROB``), optionally weighted toward the furthest story stage reached so far (``long_v3``);
- staggered episodes (``long_v3``): a random initial step offset per env plus episode-length jitter, so resets do not
  happen in lockstep (``phase_spread`` metric);
- battle archive (``long_v3``): states saved when a battle starts (bounded, keyed by kind / map / enemy species);
  ``battle_archive_prob`` of resets load one, and the trainer schedules a decaying A-logit bias inside battle menus
  (applied inside the policy, so PPO log-probs stay valid);
- ``long_v4`` additions: milestone success-rate diagnostics (``src.milestone_stats``), a smaller and decaying exploration
  budget, a heal-for-HP-restored reward (``src.heal_reward``), low-HP battle practice (weighted battle-archive resets),
  and auxiliary prediction heads trained from the same rollouts (``src.aux_tasks``);
- ``long_v5`` additions (anti-collapse after v4-1 degenerated to pressing A): wall-bump / tile-revisit penalties off, a
  cost for A / B presses that change nothing (``src.idle_press``), path reward 0.3 per tile, a higher entropy floor with
  a slower anneal and an entropy-boost safety net (``src.entropy_guard``, loss coefficient only);
- ``long_v6`` additions (after v5-1 stalled near the end of Route 29): a route-checkpoint archive (``src.route_ckpt``, a
  self-generated reverse curriculum: states saved every 12 tiles of path progress, ``route_ckpt_prob`` of the resets load
  one), a dialogue-loop penalty (``src.dialogue_loop``) and a warm-up rollout before update 0 that only initialises the
  running reward scaler.  Reset order: battle archive, swarm archive, route checkpoints, start state.  Also long_v6: a retreat cost
  and a stagnation penalty (reward only, ``src.path_progress`` / ``src.stagnation``) and white-out as a true episode end.
- ``long_v8`` additions: battle-menu action mask, periodic from-start evaluation and ``egg_errand_route`` (``src.story_path`` EGG
  profile: 11 milestones incl. the egg received / delivered events, 80 observation inputs; ``src.route_graph`` three-phase path
  measure, 695 steps).  The workers and the trainer switch the process-global story profile from the config at startup.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim

from src.battle_obs import equip_supplies, heal_item_count
from src.periodic_eval import EVAL_STATE, aggregate, episode_record, is_eval_update
from src.battle_mask import N_ACTIONS as MASK_N_ACTIONS
from src.battle_mask import all_true as mask_all_true, battle_action_mask, pack_mask, unpack_mask
from src.deep_policy import BATTLE_FLAG_INDEX, CheckpointIncompatibleError, SpatialVisionPolicy
from src.dialogue_loop import DialogueLoopTracker
from src.stagnation import StagnationTracker
from src.dwell import DwellTracker
from src.route_ckpt import (
    ROUTE_PREFIX, ROUTE_SAVE_GAP, RouteCheckpointArchive, bucket_of, milestone_stage, stage_maps,
)
from src.entropy_guard import EntropyGuard, anneal_fraction
from src.idle_press import IdlePressTracker
from src.path_progress import PathProgress
from src.route_graph import egg_phase as route_graph_phase
from src.emulator import Emulator
from src.heal_care import HealCareTracker, is_heal_trip
from src.heal_reward import heal_payment
from src.milestone_stats import MilestoneRecords, MilestoneStats, SourceSplitStats, is_base_start, milestone_labels, source_of
from src.live_feed import LiveFeed
from src.macros import Macros
from src.mover import walkable
from src.ppo_update import (  # noqa: F401
    EntropySchedule, PPOHParams, RolloutBatch, RunningReturnScaler, compute_gae, ppo_update,
)
from src.ram_parser import RamParser, dialogue_open
from src.rl_env import (
    ACTIONS,
    BUTTON_ACTIONS,
    DEEP_OBS_DIM,  # noqa: F401  (base-profile width; the trainer uses deep_obs_dim() of its story profile)
    N_ACTIONS,
    NEW_INTERACTION_REWARD,
    POKECENTER_MAPS,
    budget_pay,
    deep_obs_dim,
    NEW_WARP_REWARD,
    StepInfo,
    initial_seen_events,
    intrinsic_curiosity_reward,
    observe,
    progress_reward,
)
from src.states import StateRegistry
from src.step import TileStepper
from src.story_flags import REVERSE_EVENT_FLAGS
from src import story_path as _story
from src.story_path import KEY_EVENT_REWARDS, MILESTONE_COUNT, current_stage, implied_maps, progress_potential  # noqa: F401
from src.terminal_monitor import TerminalMonitor
from src.train_config import ConfigError, TrainConfig, apply_overrides, dump_effective_config, load_config

CURRICULUM_STATES = [
    # Tier 1: Retention (4 workers / 25%) - maintain Route 29 & Cherrygrove mastery without compute waste
    "route29_entry",
    "route29_entry",
    "agent_map_26_3",
    "agent_map_26_3",

    # Tier 2: Bottlenecks (6 workers / 37.5%) - Route 30 maze & Rival battle
    "agent_map_26_1",
    "agent_map_26_1",
    "agent_map_26_1",
    "mystery_egg_received",
    "mystery_egg_received",
    "mystery_egg_received",

    # Tier 3: Frontier Exploration (6 workers / 37.5%) - Route 29 Eastbound to Elm Lab delivery
    "post_rival_cherrygrove",
    "post_rival_cherrygrove",
    "post_rival_cherrygrove",
    "post_rival_cherrygrove",
    "post_rival_cherrygrove",
    "post_rival_cherrygrove",

    # Stages 2-3 (walk to Falkner, Togepi egg errand): with stage_balanced these get a quarter of the episodes
    "stage2_violet",
    "stage2_gym_entered",
    "stage3_falkner_beaten",
]

_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

SWARM_SAMPLE_PROB = 0.3  # chance that a reset loads a state from the swarm archive instead of a start state
SWARM_MAX_FILES = 64  # the archive never holds more states than this (they are ~100 KB each)
SWARM_PREFIX = "swarm:"
BATTLE_PREFIX = "battle:"
BATTLE_SAVE_GAP = 64  # a worker saves at most one battle state per this many of its own steps
BATTLE_SAVE_WAIT = 5  # steps to wait after a battle starts for the enemy species to be readable
SWARM_STAGE_SLOTS = 4  # rotating snapshots kept per stage entry


def key_event_ids() -> frozenset:
    """Flag ids of the key events of this process's story profile (Falkner; with the egg errand also the egg flags)."""
    return frozenset(fid for fid, name in REVERSE_EVENT_FLAGS.items() if name in _story.profile().key_event_rewards)


KEY_EVENT_IDS = key_event_ids()   # base profile (kept for importers); the worker re-evaluates it for its own profile
ROUTE_PROGRESS_WINDOW = 64  # recent episodes averaged for the entropy schedule's progress signal
REWARD_CYCLE_WINDOW = 8  # updates averaged into ``reward_8`` (the lockstep cycle was 8 updates long)
SAVE_EVERY = 10  # updates between "latest" checkpoints (the final update is always saved)
BEST_WINDOW = 5  # updates averaged into the score that decides the "best" checkpoint
SCREEN_SHAPE = (3, 144, 160)


# --------------------------------------------------------------------------------------------- pure helpers
def pick_executed_action(act_idx: int, last_move: int | None, sticky_prob: float, rng, in_battle: bool = False) -> int:
    """Sticky actions: with probability ``sticky_prob`` repeat the previous executed movement (index < 4).

    ``act_idx`` is what the policy sampled (and what PPO stores); the return value is what the game executes.
    Never sticky in battle: repeating a direction there would move a menu cursor the policy did not ask to move.
    """
    if not in_battle and sticky_prob > 0.0 and last_move is not None and rng.random() < sticky_prob:
        return int(last_move)
    return int(act_idx)


def archive_key_stage(key: str) -> int:
    """Story stage a swarm key stands for: ``stageN_k`` = N, ``event<id>`` (key events) = the last stage, else 0."""
    head = key.split("_")[0]
    if head.startswith("stage") and head[5:].isdigit():
        return int(head[5:])
    if head.startswith("event"):
        return _story.profile().count
    return 0


def furthest_weights(stages: list[int], floor: float) -> list[float]:
    """Sampling weights that grow with the stage: the furthest stage present gets 1.0, stage s gets
    ``floor + (1 - floor) * s / furthest`` (never below ``floor``).  All equal when nothing is past stage 0."""
    top = max(stages) if stages else 0
    if top <= 0:
        return [1.0] * len(stages)
    floor = min(1.0, max(1e-6, float(floor)))
    return [floor + (1.0 - floor) * max(0, s) / top for s in stages]


def weighted_index(weights: list[float], rng) -> int:
    """Index drawn with probability proportional to ``weights`` (one ``rng.random()`` draw)."""
    total = float(sum(weights))
    x = float(rng.random()) * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x < acc:
            return i
    return len(weights) - 1


LOWHP_TAG = "hLp1"   # key suffix of a battle state whose lead is at low HP while the bag holds a heal item


def battle_archive_key(state, with_hp: bool = False, low_threshold: float = 0.4) -> str:
    """Diversity key of a battle: kind (1 wild / 2 trainer), map and enemy species.

    ``with_hp`` (long_v4 low-HP practice) appends ``_h<L|M|H>p<0|1>``: the lead's HP band (L = at most ``low_threshold``,
    M = at most 0.8) and whether the bag holds a heal item; ``hLp1`` is the state worth practising healing in."""
    key = (f"m{int(getattr(state, 'battle_mode', 0))}_{int(state.map_group)}-{int(state.map_number)}"
           f"_s{int(getattr(state, 'enemy_mon_species', 0))}")
    if not with_hp:
        return key
    hp = list(getattr(state, "party_hp", ()) or ())
    frac = (hp[0][0] / hp[0][1]) if hp and hp[0][1] > 0 else 1.0
    band = "L" if frac <= low_threshold else ("M" if frac <= 0.8 else "H")
    return f"{key}_h{band}p{1 if heal_item_count(state) > 0 else 0}"


def lowhp_battle_weight(key: str, weight: float) -> float:
    """Sampling weight of a battle-archive key: ``1 + weight`` for low-HP-with-heal-item states, else 1."""
    return 1.0 + float(weight) if key.endswith("_" + LOWHP_TAG) else 1.0


def archive_files(directory) -> list[str]:  # state files currently in an archive dir
    try:
        return [p.name for p in Path(directory).glob("*.state")]
    except OSError:
        return []


def phase_spread(ep_steps, ep_lens) -> float:
    """Std over envs of the episode phase ``ep_steps / ep_len``.  0 when all envs reset in lockstep; about 0.29 when
    the phases are spread uniformly."""
    steps = np.asarray(ep_steps, dtype=np.float64)
    lens = np.maximum(np.asarray(ep_lens, dtype=np.float64), 1.0)
    return float(np.std(steps / lens)) if steps.size else 0.0


def intrinsic_weight_value(update: int, end: float, updates: int) -> float:
    """Weight of the exploration payments: linear 1.0 -> ``end`` over the first ``updates`` updates (absolute index),
    ``end`` afterwards; 1.0 when ``updates`` is 0 or ``end`` is 1 (long_v3)."""
    if updates <= 0 or end >= 1.0:
        return 1.0
    return 1.0 + (float(end) - 1.0) * min(1.0, max(0.0, float(update) / float(updates)))


def battle_a_bias_value(update: int, start: float, updates: int) -> float:
    """Linear schedule ``start`` -> 0 over the first ``updates`` updates (absolute update index; 0 afterwards)."""
    if start <= 0 or updates <= 0:
        return 0.0
    return float(start) * max(0.0, 1.0 - float(update) / float(updates))


class SwarmArchive:
    """Tiny file-based archive of emulator states shared by the worker processes of one run.

    One file per key (``stage2``, ``event123``), written to a temp name and renamed, never overwritten, at most
    ``max_files`` files.  Nothing but the directory is shared, so it is safe across processes.
    """

    def __init__(self, directory: str | Path, max_files: int = SWARM_MAX_FILES, protect_suffix: str | None = None):
        self.dir = Path(directory)
        self.max_files = int(max_files)
        self.protect_suffix = protect_suffix   # states whose key ends with it are evicted last (long_v4 low-HP states)
        self.dir.mkdir(parents=True, exist_ok=True)

    def keys(self) -> list[str]:
        try:
            return sorted(p.stem for p in self.dir.glob("*.state"))
        except OSError:
            return []

    def save(self, key: str, data: bytes) -> bool:
        """Store ``data`` under ``key``; returns False when the key exists or the archive is full."""
        target = self.dir / f"{key}.state"
        if target.exists() or len(self.keys()) >= self.max_files:
            return False
        tmp = self.dir / f".{key}.{os.getpid()}.tmp"
        try:
            tmp.write_bytes(data)
            os.replace(tmp, target)
        except OSError:
            return False
        return True

    def load(self, key: str) -> bytes | None:
        try:
            return (self.dir / f"{key}.state").read_bytes()
        except OSError:
            return None

    def save_rotating(self, prefix: str, data: bytes, slots: int = SWARM_STAGE_SLOTS) -> bool:
        """Keep up to ``slots`` snapshots under ``<prefix>_<n>``: fill free slots, then replace the oldest one."""
        have = {}
        for n in range(slots):
            p = self.dir / f"{prefix}_{n}.state"
            if p.exists():
                try:
                    have[n] = p.stat().st_mtime
                except OSError:
                    pass
        free = [n for n in range(slots) if n not in have]
        if free:
            return self.save(f"{prefix}_{free[0]}", data)
        oldest = min(have, key=have.get)
        tmp = self.dir / f".{prefix}_{oldest}.{os.getpid()}.tmp"
        try:
            tmp.write_bytes(data)
            os.replace(tmp, self.dir / f"{prefix}_{oldest}.state")
        except OSError:
            return False
        return True

    def has(self, key: str) -> bool:
        return (self.dir / f"{key}.state").exists()

    def save_fifo(self, key: str, data: bytes) -> bool:
        """Bounded archive that keeps accepting NEW keys: an existing key is never overwritten (diversity), and when
        more than ``max_files`` states exist the oldest ones are deleted.  Used by the battle archive."""
        target = self.dir / f"{key}.state"
        if target.exists():
            return False
        tmp = self.dir / f".{key}.{os.getpid()}.tmp"
        try:
            tmp.write_bytes(data)
            os.replace(tmp, target)
        except OSError:
            return False
        self.prune()
        return True

    def prune(self) -> None:
        """Delete the oldest states until at most ``max_files`` remain (several workers may overshoot briefly)."""
        suffix = f"{self.protect_suffix}.state" if self.protect_suffix else None
        try:
            files = sorted(self.dir.glob("*.state"),
                           key=lambda p: (bool(suffix and p.name.endswith(suffix)), p.stat().st_mtime))
        except OSError:
            return
        for p in files[: max(0, len(files) - self.max_files)]:
            try:
                p.unlink()
            except OSError:
                pass

    def sample_weighted(self, rng, weight_fn) -> str | None:
        """A key drawn with probability proportional to ``weight_fn(key)`` (no grouping)."""
        keys = self.keys()
        if not keys:
            return None
        return keys[weighted_index([float(weight_fn(k)) for k in keys], rng)]

    def sample(self, rng, furthest_floor: float | None = None) -> str | None:
        """Uniform over groups (``stage2_0..3`` is one group, ``event123`` another), then uniform inside the group.

        With ``furthest_floor`` the groups are weighted by story stage instead (``furthest_weights``): the furthest
        stage in the archive gets weight 1, earlier stages at least ``furthest_floor``."""
        keys = self.keys()
        if not keys:
            return None
        groups: dict[str, list[str]] = {}
        for k in keys:
            groups.setdefault(k.split("_")[0], []).append(k)
        names = sorted(groups)
        if furthest_floor is None:
            group = groups[names[int(rng.integers(0, len(names)))]]
        else:
            weights = furthest_weights([archive_key_stage(groups[n][0]) for n in names], furthest_floor)
            group = groups[names[weighted_index(weights, rng)]]
        return group[int(rng.integers(0, len(group)))]


class StartStateSelector:
    """Chooses the start state of each episode of one env.

    Without ``stages`` the states cycle (offset by the env index, so envs start on different states).  With
    ``stages`` (state name -> story stage) the stages are cycled instead, so every stage is sampled evenly, and the
    states inside a stage cycle too.  Resets (not the very first pick) load from ``archive`` with probability
    ``swarm_prob``; archived picks are returned as ``"swarm:<key>"``.
    """

    def __init__(self, start_states, env_index: int = 0, stages: dict[str, int] | None = None,
                 archive: SwarmArchive | None = None, swarm_prob: float = SWARM_SAMPLE_PROB, rng=None,
                 battle_archive: SwarmArchive | None = None, battle_prob: float = 0.0,
                 furthest_floor: float | None = None, battle_lowhp_weight: float = 0.0,
                 route_archive: RouteCheckpointArchive | None = None, route_prob: float = 0.0):
        self.states = list(start_states)
        self.route_archive = route_archive   # long_v6 route checkpoints (src.route_ckpt)
        self.route_prob = float(route_prob) if route_archive is not None else 0.0
        self.battle_lowhp_weight = float(battle_lowhp_weight)   # > 0: favour low-HP battle states (long_v4)
        self.battle_archive = battle_archive
        self.battle_prob = float(battle_prob) if battle_archive is not None else 0.0
        self.furthest_floor = furthest_floor  # None = uniform over archive stages, else furthest-weighted
        if not self.states:
            raise ValueError("start_states must not be empty")
        self.env_index = int(env_index)
        self.archive = archive
        self.swarm_prob = float(swarm_prob)
        self.rng = rng if rng is not None else np.random.default_rng(env_index)
        self.by_stage: dict[int, list[str]] = {}
        if stages:
            for name in self.states:
                self.by_stage.setdefault(int(stages[name]), []).append(name)
        self.stage_order = sorted(self.by_stage)
        self.count = 0

    def pick(self) -> str:
        k = self.count
        self.count += 1
        swarm_on = self.archive is not None and self.swarm_prob > 0
        route_on = self.route_prob > 0
        if k > 0 and (swarm_on or self.battle_prob > 0 or route_on):
            # One draw: [0, battle_prob) = battle archive, next swarm_prob = swarm archive, next route_prob = route
            # checkpoint (long_v6), the rest = start states. An empty archive falls through to a start state.
            u = float(self.rng.random())
            route_lo = self.battle_prob + (self.swarm_prob if swarm_on else 0.0)
            if u < self.battle_prob:
                if self.battle_lowhp_weight > 0:
                    key = self.battle_archive.sample_weighted(
                        self.rng, lambda k: lowhp_battle_weight(k, self.battle_lowhp_weight))
                else:
                    key = self.battle_archive.sample(self.rng)
                if key is not None:
                    return BATTLE_PREFIX + key
            elif swarm_on and u < self.battle_prob + self.swarm_prob:
                key = self.archive.sample(self.rng, self.furthest_floor)
                if key is not None:
                    return SWARM_PREFIX + key
            elif route_on and route_lo <= u < route_lo + self.route_prob:
                key = self.route_archive.sample(self.rng)
                if key is not None:
                    return ROUTE_PREFIX + key
        n = self.env_index + k
        if self.by_stage:
            group = self.by_stage[self.stage_order[n % len(self.stage_order)]]
            return group[(n // len(self.stage_order)) % len(group)]
        return self.states[n % len(self.states)]


class RouteProgressTracker:
    """Smoothed fraction of the remaining route covered per episode, in [0, 1] (drives the entropy schedule)."""

    def __init__(self, window: int = ROUTE_PROGRESS_WINDOW):
        self.fracs: deque[float] = deque(maxlen=window)

    def add(self, start_phi: float | None, end_phi: float | None) -> None:
        """``phi`` is the route potential (negative steps remaining); None means off the route graph."""
        if start_phi is None or end_phi is None or start_phi >= -1e-6:
            return
        self.fracs.append(min(1.0, max(0.0, (end_phi - start_phi) / -start_phi)))

    @property
    def value(self) -> float:
        return float(sum(self.fracs) / len(self.fracs)) if self.fracs else 0.0

    def state_dict(self) -> dict:
        return {"fracs": list(self.fracs)}

    def load_state_dict(self, state: dict) -> None:
        self.fracs.clear()
        self.fracs.extend(float(x) for x in state.get("fracs", []))


def _story_phi(state, seen_maps) -> float:
    """-(story milestones left): the progress signal of the entropy schedule (0 = the whole path is done)."""
    return float(progress_potential(state, seen_maps))


def _progress_reward_takes_budget() -> bool:
    """F1 adds a per-episode ``budget`` dict to ``progress_reward``; pass it only when the signature has it."""
    try:
        return "budget" in inspect.signature(progress_reward).parameters
    except (TypeError, ValueError):
        return False


def worker_interaction_reward(before, after, action_idx: int, seen_interactions: set, seen_warps: set,
                              components: dict | None = None, budget: dict | None = None) -> float:
    """The one-time exploration payments ``CrystalEnv`` makes outside ``progress_reward`` (explore-budgeted).

    +NEW_INTERACTION_REWARD for the first A press that opens a script from a (tile, facing); +NEW_WARP_REWARD for the
    first map-to-map edge.
    """
    bonus = 0.0
    busy = before.battle_mode != 0 or dialogue_open(before)
    if action_idx == ACTIONS.index("a") and not busy and dialogue_open(after):
        full_hp_center = False
        if (before.map_group, before.map_number) in POKECENTER_MAPS:  # no bonus when nothing to heal
            hp_b = getattr(before, "party_hp", ()) or ()
            full_hp_center = bool(hp_b) and sum(p[0] for p in hp_b) >= sum(p[1] for p in hp_b)
        key = (before.map_group, before.map_number, before.x, before.y, before.facing)
        if not full_hp_center and key not in seen_interactions:
            seen_interactions.add(key)
            pay = budget_pay(budget, "explore", NEW_INTERACTION_REWARD)
            bonus += pay
            if components is not None:
                components["interaction"] = components.get("interaction", 0.0) + pay
    before_map, after_map = (before.map_group, before.map_number), (after.map_group, after.map_number)
    if before_map != after_map:
        edge = (before_map, after_map)
        if edge not in seen_warps:
            seen_warps.add(edge)
            pay = budget_pay(budget, "explore", NEW_WARP_REWARD)
            bonus += pay
            if components is not None:
                components["warp"] = components.get("warp", 0.0) + pay
    return bonus


# --------------------------------------------------------------------------------------------- environment
class EnvWorker:
    """One headless PyBoy instance with the training reward, sticky actions, start jitter and swarm sharing.

    ``cfg`` keys (all optional): ``sticky_prob``, ``jitter_max``, ``stage_balanced``, ``swarm_dir``, ``swarm_prob``,
    ``seed``, ``goal_potential`` (optional tile potential toward Falkner, default off), ``stagger_episodes`` /
    ``episode_len_jitter`` / ``num_envs`` (desynchronized episodes), ``swarm_furthest_floor``, ``battle_dir`` /
    ``battle_prob`` / ``battle_max`` (battle archive).  Call ``reset()`` before the first ``step()``.
    """

    def __init__(self, index: int, start_states: list[str], max_steps: int, cfg: dict | None = None):
        cfg = dict(cfg or {})
        # The story profile (milestone list, observation width) of this run; a spawned worker process starts on the base one.
        self.egg_route = bool(cfg.get("egg_errand_route", False))
        _story.set_profile(self.egg_route)
        self.index = int(index)
        self.max_steps = int(max_steps)
        self.sticky_prob = float(cfg.get("sticky_prob", 0.0))
        self.jitter_max = int(cfg.get("jitter_max", 0))
        self.rng = np.random.default_rng([int(cfg.get("seed", 0)), self.index])
        self.registry = StateRegistry()
        self._bytes: dict[str, bytes] = {}
        self.emu = Emulator(headless=True, speed=0, rtc_lock=True)
        self.reader = RamParser(self.emu)
        self.stepper = TileStepper(self.emu, self.reader)
        self.macros = Macros(self.emu, self.reader)

        names = list(start_states)
        stages = self._stage_of_states(sorted(set(names))) if cfg.get("stage_balanced") else None
        swarm_dir = cfg.get("swarm_dir")
        self.archive = SwarmArchive(swarm_dir) if swarm_dir else None
        battle_dir = cfg.get("battle_dir")
        self.battle_lowhp_weight = float(cfg.get("battle_lowhp_weight", 0.0))
        self.battle_lowhp_threshold = float(cfg.get("battle_lowhp_threshold", 0.4))
        self.battle_archive = (SwarmArchive(battle_dir, int(cfg.get("battle_max", 32)),
                                            protect_suffix=LOWHP_TAG if self.battle_lowhp_weight > 0 else None)
                               if battle_dir else None)
        # long_v6 route checkpoints (OFF when cfg has no "route_dir"): src.route_ckpt.
        route_dir = cfg.get("route_dir")
        self.route_every = int(cfg.get("route_every", 12))
        self.route_archive = (RouteCheckpointArchive(
            route_dir, self.route_every, int(cfg.get("route_per_bucket", 4)), int(cfg.get("route_max", 60)),
            float(cfg.get("route_floor", 0.2))) if route_dir else None)
        self.selector = StartStateSelector(
            names, self.index, stages, self.archive, float(cfg.get("swarm_prob", SWARM_SAMPLE_PROB)), self.rng,
            battle_archive=self.battle_archive, battle_prob=float(cfg.get("battle_prob", 0.0)),
            furthest_floor=cfg.get("swarm_furthest_floor"), battle_lowhp_weight=self.battle_lowhp_weight,
            route_archive=self.route_archive, route_prob=float(cfg.get("route_prob", 0.0)),
        )
        self._route_seed_stage = 0        # milestones already passed by the loaded route checkpoint's saver
        self._route_saved_bucket = 0      # highest bucket saved (or started in) during this episode
        self._last_route_save = -ROUTE_SAVE_GAP
        # Staggered episodes: the first episode starts at a random step offset, every episode length is jittered.
        self.stagger = bool(cfg.get("stagger_episodes", False))
        self.len_jitter = float(cfg.get("episode_len_jitter", 0.0))
        self.num_envs = int(cfg.get("num_envs", 0))
        self.ep_len = self.max_steps  # step limit of the current episode
        self._first_episode = True
        self._partial = False  # the running episode is a first episode that started with a step offset
        self._battle_wait: int | None = None  # steps left to wait for the enemy species after a battle started
        self._total_steps = 0
        self._last_battle_save = -BATTLE_SAVE_GAP

        self.visited: set = set()
        self.seen_tiles: set = set()
        self.seen_maps: set = set()
        self.seen_events: set = set()
        self.seen_waypoints: set = set()
        self.tile_visit_counts: dict = {}
        self.caught_counts: dict = {}
        self.blocked_count = 0
        self.consecutive_bumps = 0
        self.steps = 0
        self.last_dir: str | None = None
        self.last_move: int | None = None
        self.state_name = names[0]
        self.goal_potential = bool(cfg.get("goal_potential", False))  # OPTIONAL oracle shaping, default OFF
        self.start_phi: float | None = None
        self._event_snapshot: set = set()
        self.seen_interactions: set = set()
        self.seen_warps: set = set()
        self.budget: dict = {}  # per-episode payment budget handed to progress_reward (cleared on reset)
        self._budget_ok = _progress_reward_takes_budget()
        # long_v4 budget overrides, re-applied to the fresh budget at every reset (empty = exactly the long_v3 behaviour).
        self._budget_init: dict = {}
        if cfg.get("explore_cap") is not None:
            self._budget_init["cap:explore"] = float(cfg["explore_cap"])
        self.heal_reward = bool(cfg.get("heal_reward", False))
        self.heal_scale = float(cfg.get("heal_scale", 4.0))
        if self.heal_reward:
            self._budget_init["cap:heal"] = float(cfg.get("heal_budget", 8.0))
        self._intrinsic_scale = 1.0
        self._ep_start_stage = 0   # milestone diagnostics (src.milestone_stats)
        self._ep_furthest = 0
        # Dwell penalty (long_v4, OFF when cfg has no "dwell" params): see src.dwell.
        self.dwell = DwellTracker.from_params(cfg.get("dwell"))
        self.dwell_exempt_loop = bool(cfg.get("dwell_exempt_loop", True))
        # Dense best-ever path-progress reward (long_v4, OFF when cfg has no "path_progress" scale): src.path_progress.
        self.path_progress = (PathProgress.from_scale(cfg.get("path_progress"), cfg.get("retreat"), egg=True) if self.egg_route
                              else PathProgress.from_scale(cfg.get("path_progress"), cfg.get("retreat")))
        # long_v6: stagnation penalty (src.stagnation, OFF without params), white-out as a true episode end.
        self.stagnation = StagnationTracker.from_params(cfg.get("stagnation"))
        self.terminate_on_whiteout = bool(cfg.get("terminate_on_whiteout", False))
        # long_v8: valid-action mask of the CURRENT observation (src.battle_mask), refreshed by get_obs; OFF = all True.
        self.action_mask = bool(cfg.get("action_mask", False))
        self.last_mask = mask_all_true()
        # long_v5 anti-collapse (all OFF when absent from cfg): no-effect A / B press cost (src.idle_press), and the
        # wall-bump / tile-revisit penalties switched off while the path reward is on.
        self.idle_press = IdlePressTracker.from_params(cfg.get("idle_press"))
        self.disable_bump = bool(cfg.get("disable_bump_penalty", False))
        self.disable_loop = bool(cfg.get("disable_loop_penalty", False))
        # long_v6: repeated effect-less dialogue cost (src.dialogue_loop), OFF when cfg has no "dialogue_loop" params.
        self.dialogue_loop = DialogueLoopTracker.from_params(cfg.get("dialogue_loop"))
        # long_v7 healing care (src.heal_care; OFF when cfg has no "heal_care" params / "heal_trip_exempt").
        self.heal_care = HealCareTracker.from_params(cfg.get("heal_care"))
        self.heal_trip_exempt = bool(cfg.get("heal_trip_exempt", False))
        self.heal_trip_threshold = float(cfg.get("heal_trip_threshold", 0.3))
        self._ep_start_path: float | None = None   # path progress at the start of the running episode (source-split metrics)
        self._last_stage = 0 # stage already archived / seen at the start of the episode
        self._pending_stage: int | None = None  # stage entered while mid-cutscene: snapshot once the screen is calm
        self._pending_events: set = set()

    # -- setup helpers
    def _state_bytes(self, name: str) -> bytes:
        if name not in self._bytes:
            self._bytes[name] = self.registry.load_bytes(name)
        return self._bytes[name]

    def _stage_of_states(self, names: list[str]) -> dict[str, int]:
        stages = {}
        for name in names:
            self.emu.load_state(self._state_bytes(name))
            stages[name] = int(current_stage(self.reader.parse(self.emu.frame)))
        return stages

    def _equip_balls(self) -> None:
        """Hard-coded start supplies (10 Poke Balls, 5 Potions) so catching and healing can be learned from episode one."""
        equip_supplies(self.emu)

    def _load(self, name: str) -> bytes:
        """Load a start state (or an archived ``swarm:<key>`` / ``battle:<key>`` state) into the emulator; returns its bytes."""
        data = None
        self._route_seed_stage = 0
        if name.startswith(SWARM_PREFIX) and self.archive is not None:
            data = self.archive.load(name[len(SWARM_PREFIX):])
        elif name.startswith(BATTLE_PREFIX) and self.battle_archive is not None:
            data = self.battle_archive.load(name[len(BATTLE_PREFIX):])
        elif name.startswith(ROUTE_PREFIX) and self.route_archive is not None:
            loaded = self.route_archive.load(name[len(ROUTE_PREFIX):])
            if loaded is not None:
                data, self._route_seed_stage = loaded
        if data is None:
            name = name if not name.startswith((SWARM_PREFIX, BATTLE_PREFIX, ROUTE_PREFIX)) else self.selector.states[0]
            data = self._state_bytes(name)
        self.state_name = name
        self.emu.load_state(data)
        self._equip_balls()
        return data

    # -- observation
    def _screen(self) -> np.ndarray:
        return self.emu._pb.screen.ndarray[:, :, :3].copy()

    def get_obs(self):
        st = self.reader.parse(self.emu.frame)
        tile = (st.map_group, st.map_number, st.x, st.y)
        self.visited.add(tile)
        self.seen_tiles.add(tile)
        self.seen_maps.add((st.map_group, st.map_number))
        self.tile_visit_counts[tile] = self.tile_visit_counts.get(tile, 0) + 1
        ram_vec = observe(st, self.visited, deep=True, seen_maps=self.seen_maps)
        if self.action_mask:
            self.last_mask = battle_action_mask(self.emu, st)   # only reads the tilemap while battle_mode != 0
        return self._screen(), ram_vec, st

    def _peek_obs(self, st):
        """Observation of ``st`` without touching the visit bookkeeping (the final obs of a truncated episode)."""
        return self._screen(), observe(st, self.visited, deep=True, seen_maps=self.seen_maps)

    # -- episode control
    def _jitter(self, data: bytes) -> None:
        """0..jitter_max random moves; no reward, and the start state is restored if one leaves the map or fights."""
        n = int(self.rng.integers(0, self.jitter_max + 1)) if self.jitter_max > 0 else 0
        if n == 0:
            return
        start = self.reader.parse(self.emu.frame)
        start_map = (start.map_group, start.map_number)
        for _ in range(n):
            cur = self.reader.parse(self.emu.frame)
            if cur.battle_mode != 0 or dialogue_open(cur):
                break
            self.stepper.step(ACTIONS[int(self.rng.integers(0, 4))])
            after = self.reader.parse(self.emu.frame)
            if (after.map_group, after.map_number) != start_map or after.battle_mode != 0:
                self.emu.load_state(data)
                self._equip_balls()
                break

    def _draw_ep_len(self) -> int:
        """Step limit of the next episode: ``max_steps`` +- ``episode_len_jitter`` (uniform), at least 1."""
        if self.len_jitter <= 0:
            return self.max_steps
        return max(1, int(round(self.max_steps * (1.0 + self.rng.uniform(-self.len_jitter, self.len_jitter)))))

    def _initial_offset(self) -> int:
        """Step offset of the very first episode, in [0, ep_len).  With ``num_envs`` the offsets are spread evenly:
        env i draws uniformly from the i-th of num_envs equal slots (so 24 envs cover all 24 phases of the cycle)."""
        if self.num_envs > 0:
            frac = ((self.index % self.num_envs) + float(self.rng.random())) / self.num_envs
        else:
            frac = float(self.rng.random())
        return min(self.ep_len - 1, int(frac * self.ep_len))

    def reset(self):
        """Start a new episode and return its first ``(screen, ram_vec)``."""
        data = self._load(self.selector.pick())
        for s in (self.visited, self.seen_tiles, self.seen_maps, self.seen_events, self.seen_waypoints):
            s.clear()
        self.tile_visit_counts.clear()
        self.caught_counts.clear()
        self.seen_interactions.clear()
        self.seen_warps.clear()
        if self._route_seed_stage:
            # Route checkpoint: the milestones its saver had passed are done (maps seen), so walking back over them and
            # forward again never pays their bonuses a second time; the observation's milestone bits stay consistent.
            self.seen_maps.update(stage_maps(self._route_seed_stage))
        if self.egg_route:
            # Egg errand: the event flags of the start state / checkpoint say which story maps are already behind it (the
            # egg in hand means Route 29 ... Mr. Pokemon's house), so their arrival bonuses are never paid again.
            self.seen_maps.update(implied_maps(self.reader.parse(self.emu.frame)))
        self.budget.clear()
        self.budget.update(self._budget_init)
        if self._intrinsic_scale != 1.0:
            self.budget["scale:explore"] = self._intrinsic_scale
        self._pending_stage, self._pending_events = None, set()
        self.blocked_count = self.consecutive_bumps = self.steps = 0
        self.last_dir = self.last_move = None
        self._battle_wait = None
        self.ep_len = self._draw_ep_len()
        self._partial = False
        if self._first_episode and self.stagger:
            # Only the step counter starts mid-episode: budgets, seen sets and the start state are the normal fresh
            # ones, so this first episode is simply cut short (truncated + bootstrapped like any time limit).
            self.steps = self._initial_offset()
            self._partial = self.steps > 0
        self._first_episode = False
        self._jitter(data)
        st = self.reader.parse(self.emu.frame)
        # Seed the caught species from the party so the starter does not pay a catch bonus later.
        for species_id in getattr(st, "party_species", ()):
            if species_id and species_id != 255:
                self.caught_counts[species_id] = self.caught_counts.get(species_id, 0) + 1
        self._last_stage = int(current_stage(st))
        self._ep_start_stage = self._ep_furthest = self._last_stage
        if self.dwell is not None:
            self.dwell.reset(st)
        if self.path_progress is not None:
            self.path_progress.reset(st)   # the start state's own progress (archive / jitter) is never paid
            self._ep_start_path = self.path_progress.best
            # The route checkpoint's own bucket is already in the archive: save again only for a higher one.
            self._route_saved_bucket = bucket_of(self.path_progress.best, self.route_every)
        if self.idle_press is not None:
            self.idle_press.reset()
        if self.dialogue_loop is not None:
            self.dialogue_loop.reset()
        if self.stagnation is not None:
            self.stagnation.reset()
        if self.heal_care is not None:
            self.heal_care.reset()
        # Flags already set in the start state are not progress: seed them so they pay nothing (was ~+300/episode).
        self.seen_events.update(initial_seen_events(st))
        self._event_snapshot = set(self.seen_events)
        screen, ram_vec, _ = self.get_obs()  # also records the start map as seen (it must not pay a milestone)
        self.start_phi = _story_phi(st, self.seen_maps)
        return screen, ram_vec

    def set_intrinsic_scale(self, scale: float) -> None:
        """Weight of the exploration payments (long_v4 decay): applied to the running and all later episodes."""
        self._intrinsic_scale = float(scale)
        if self._intrinsic_scale != 1.0:
            self.budget["scale:explore"] = self._intrinsic_scale
        else:
            self.budget.pop("scale:explore", None)

    def _maybe_archive(self, after) -> None:
        """Save the emulator state when this env is first to reach a new stage or key event (swarm sharing)."""
        """Snapshots are taken at stage ENTRY (a few rotating slots per stage) and at the first key-event flip, never
        mid-cutscene (battle or dialogue open): those are deferred until the screen is calm."""
        if self.archive is None:
            return
        new = self.seen_events - self._event_snapshot
        self._event_snapshot = set(self.seen_events)
        self._pending_events |= {fid for fid in new if fid in key_event_ids()}
        stage = int(current_stage(after))
        if stage > self._last_stage:
            self._pending_stage = stage if self._pending_stage is None else max(self._pending_stage, stage)
            self._last_stage = stage
        if (self._pending_stage is None and not self._pending_events) or after.battle_mode != 0 or dialogue_open(after):
            return
        data = self.emu.save_state()
        if self._pending_stage is not None:
            self.archive.save_rotating(f"stage{self._pending_stage}", data)
            self._pending_stage = None
        for fid in sorted(self._pending_events):
            self.archive.save(f"event{fid}", data)
        self._pending_events = set()

    def _maybe_archive_battle(self, before, after) -> None:
        """Save the emulator state shortly after a battle starts (battle flag 0 -> non-zero).

        Rate limited (one save per ``BATTLE_SAVE_GAP`` of this worker's steps), keyed by kind / map / enemy species
        so only new combinations are stored, bounded by the archive's ``max_files`` (oldest deleted).  The save waits
        up to ``BATTLE_SAVE_WAIT`` steps for the enemy species to be readable and is dropped if the battle ends."""
        if self.battle_archive is None:
            return
        if before.battle_mode == 0 and after.battle_mode != 0:
            self._battle_wait = BATTLE_SAVE_WAIT
        if self._battle_wait is None:
            return
        if after.battle_mode == 0:
            self._battle_wait = None
            return
        if int(getattr(after, "enemy_mon_species", 0)) == 0 and self._battle_wait > 0:
            self._battle_wait -= 1
            return
        self._battle_wait = None
        if self._total_steps - self._last_battle_save < BATTLE_SAVE_GAP:
            return
        lowhp = getattr(self, "battle_lowhp_weight", 0.0) > 0
        key = battle_archive_key(after, with_hp=lowhp, low_threshold=getattr(self, "battle_lowhp_threshold", 0.4))
        if self.battle_archive.has(key):
            return
        if self.battle_archive.save_fifo(key, self.emu.save_state()):
            self._last_battle_save = self._total_steps

    def _maybe_save_route(self, after) -> None:
        """Save a route checkpoint when the position reaches a path-progress bucket not yet saved in this episode.

        Calm states only (no battle, no open text, no white-out teleport); rate limited to one save per
        ``ROUTE_SAVE_GAP`` of this worker's steps (a blocked save is retried on the next steps).  The bucket is that of the
        CURRENT position, so a state saved after stepping back is filed under the bucket it really is in."""
        if self.route_archive is None or self.path_progress is None:
            return
        cur = self.path_progress.current
        if cur is None or getattr(after, "fainted", False) or after.battle_mode != 0 or dialogue_open(after):
            return
        bucket = bucket_of(cur, self.route_every)
        if bucket < 1 or bucket <= self._route_saved_bucket:
            return
        if self._total_steps - self._last_route_save < ROUTE_SAVE_GAP:
            return
        self._route_saved_bucket = bucket
        self._last_route_save = self._total_steps
        self.route_archive.save(bucket, milestone_stage(self.seen_maps), self.emu.save_state(),
                                phase=route_graph_phase(after) if self.egg_route else None)

    # -- stepping
    def step(self, act_idx: int):
        """Execute one policy action; returns the 7-tuple the trainer expects (see ``VectorEnv.step``)."""
        emu, stepper, macros, reader = self.emu, self.stepper, self.macros, self.reader
        before = reader.parse(emu.frame)
        exec_idx = pick_executed_action(int(act_idx), self.last_move, self.sticky_prob, self.rng,
                                        in_battle=before.battle_mode != 0)
        if exec_idx < 4:
            self.last_move = exec_idx
        action_name = ACTIONS[exec_idx]
        moved = False
        plain_press = False   # an A / B press in the plain overworld (no battle, no dialogue): the idle_press candidates

        if before.battle_mode != 0:
            # LEARNED BATTLE: the policy's button goes straight to the game (FIGHT / PKMN / PACK / RUN menus, the
            # move list, item and party screens, every text box). No tactical script, no text mashing in battle.
            emu.press(action_name, frames=8)
            emu.tick(12, render=False)
            self.blocked_count = 0
        elif dialogue_open(before):
            # Overworld dialogue ONLY is auto-advanced (episode feasibility: signs, NPC chatter, cutscenes, Nurse Joy).
            macros.mash_text(max_frames=80)
            self.blocked_count = 0
        elif action_name in BUTTON_ACTIONS:
            emu.press(action_name, frames=8)
            emu.tick(12, render=False)
            moved = False
            plain_press = True
        else:
            step_res = stepper.step(action_name)
            moved = bool(step_res.moved)
            if not moved:
                self.blocked_count += 1
                if self.blocked_count >= 4:
                    self._escape()
            else:
                self.blocked_count = 0

        busy = before.battle_mode != 0 or dialogue_open(before)
        # An A / B press (talking to an NPC, reading a sign) never moves; it is not a wall bump.
        if not moved and not busy and action_name not in BUTTON_ACTIONS:
            self.consecutive_bumps += 1
        else:
            self.consecutive_bumps = 0

        after = reader.parse(emu.frame)
        self._total_steps += 1
        self._maybe_archive_battle(before, after)
        unvisited_neighbor = False
        if 0 <= exec_idx < 4:
            dx, dy = _DELTA[action_name]
            neighbor = (before.map_group, before.map_number, before.x + dx, before.y + dy)
            unvisited_neighbor = neighbor not in self.seen_tiles

        comps: dict[str, float] = {}
        pr_kwargs = {"budget": self.budget} if self._budget_ok else {}
        if self.heal_reward:
            pr_kwargs["pay_heal"] = False   # src.heal_reward pays for HP actually restored instead of the flat terms
        if self.disable_bump and self.path_progress is not None:
            pr_kwargs["pay_bump"] = False   # long_v5: dwell + path progress cover wall hugging
        pr_info = StepInfo(consecutive_bumps=self.consecutive_bumps)
        species_before = len(self.caught_counts)
        step_rew, new_tile, new_map = progress_reward(
            before, after, moved, self.seen_tiles, self.seen_maps,
            unvisited_neighbor=unvisited_neighbor, caught_counts=self.caught_counts, seen_events=self.seen_events,
            seen_waypoints=self.seen_waypoints, info=pr_info,
            components=comps, goal_potential=self.goal_potential, **pr_kwargs,
        )
        step_rew += worker_interaction_reward(before, after, exec_idx, self.seen_interactions, self.seen_warps, comps,
                                              self.budget if self._budget_ok else None)
        if self.heal_reward:
            heal_pay = budget_pay(self.budget, "heal", heal_payment(before, after, self.heal_scale))
            if heal_pay:
                comps["heal"] = comps.get("heal", 0.0) + heal_pay
                step_rew += heal_pay
        # Count-based novelty (1/sqrt(visits)) and a small straight-line momentum bonus: generic exploration terms.
        v_count = self.tile_visit_counts.get((after.map_group, after.map_number, after.x, after.y), 0)
        curiosity = intrinsic_curiosity_reward(v_count, scale=0.08)
        curiosity = budget_pay(self.budget, "explore", curiosity) if self._budget_ok else curiosity
        in_play = not busy and after.battle_mode == 0 and not dialogue_open(after)  # no loop penalty in fights/text
        loop = -0.02 * min(4.0, (v_count - 10) / 5.0) if (v_count > 12 and in_play) else 0.0  # anti-looping pacing
        path_rew = 0.0
        if self.path_progress is not None:
            path_rew = self.path_progress.update(before, after)   # >= 0: only a new best progress tile pays
        retreat_pen = self.path_progress.retreat_step if self.path_progress is not None else 0.0   # <= 0 (long_v6)
        dwell_pen, level_exempt = 0.0, False
        if self.dwell is not None:
            # progress = story milestone / event, a species caught for the first time this episode, or a new best path tile
            progress = bool(pr_info.story_progress or len(self.caught_counts) > species_before or path_rew > 0.0)
            dwell_pen, level_exempt = self.dwell.update(before, after, new_tile=new_tile, new_map=new_map, progress=progress)
            if level_exempt and self.dwell_exempt_loop:
                loop = 0.0  # needed leveling is not "looping": the per-tile revisit cost is off while grinding below target
        if self.disable_loop and self.path_progress is not None:
            loop = 0.0      # long_v5: the per-tile revisit cost made standing still (pressing A) the cheapest policy
        stag_pen = 0.0
        if self.stagnation is not None and self.path_progress is not None:
            # long_v6: max(dwell, stagnation) per step, never the sum: only the excess over this step's dwell charge is paid here
            stag_pen = self.stagnation.update(before, after, new_best=path_rew > 0.0, level_exempt=level_exempt,
                                              dwell_charge=dwell_pen)
        idle_pen = 0.0
        if self.idle_press is not None and plain_press:
            idle_pen = self.idle_press.update(before, after)   # <= 0: an A / B press that changed nothing observable
        dlg_pen = 0.0
        if self.dialogue_loop is not None:
            dlg_pen = self.dialogue_loop.update(before, after, exec_idx == ACTIONS.index("a"))   # <= 0 (long_v6)
        self._maybe_save_route(after)
        if self.heal_trip_exempt and is_heal_trip(after, self.heal_trip_threshold):
            # long_v7: walking to heal is never a retreat / stall / dwell: those three costs are off while the lead is low or
            # the agent is in a Center / Mart (the retreat budget spent on this step is handed back).
            if retreat_pen and self.path_progress is not None:
                self.path_progress.retreat_spent = max(0.0, self.path_progress.retreat_spent + retreat_pen)
            retreat_pen = dwell_pen = stag_pen = 0.0
        heal_care_comps = self.heal_care.update(before, after) if self.heal_care is not None else None
        momentum = 0.02 if (action_name == self.last_dir and moved) else 0.0
        momentum = budget_pay(self.budget, "explore", momentum) if self._budget_ok else momentum
        self.last_dir = action_name if moved else None

        comps["curiosity"], comps["loop"], comps["momentum"] = curiosity, loop, momentum
        if self.dwell is not None:
            comps["dwell"] = dwell_pen
        if self.path_progress is not None:
            comps["path"] = path_rew
        if self.idle_press is not None:
            comps["idle_press"] = idle_pen
        if self.dialogue_loop is not None:
            comps["dialogue_loop"] = dlg_pen
        if self.path_progress is not None and self.path_progress.retreat_frac > 0:
            comps["retreat"] = retreat_pen
        if self.stagnation is not None:
            comps["stagnation"] = stag_pen
        step_rew += curiosity + loop + momentum + dwell_pen + path_rew + idle_pen + dlg_pen + retreat_pen + stag_pen
        if heal_care_comps:
            for k, v in heal_care_comps.items():   # long_v7: "lowhp" (<= 0), "survive" / "potion" (>= 0)
                comps[k] = comps.get(k, 0.0) + v
                step_rew += v

        # No termination on milestones or on the badge; the episode ends on the step limit (a time limit, not a goal) or,
        # with ``terminate_on_whiteout`` (long_v6), on a white-out: a TRUE termination (not truncated, so the trainer does
        # not bootstrap the value; the -15 white-out penalty of this step is the terminal reward).
        self.steps += 1
        whiteout = bool(getattr(after, "fainted", False) and not getattr(before, "fainted", False))
        terminated = self.terminate_on_whiteout and whiteout
        truncated = self.steps >= self.ep_len and not terminated  # time limit, not a real end
        done = truncated or terminated
        end_reason = "whiteout" if terminated else ("time_limit" if truncated else None)
        ended_maps = len(self.seen_maps)
        ended_state = self.state_name
        phi_end = _story_phi(after, self.seen_maps)
        phi_start = self.start_phi
        stage_end = int(current_stage(after))
        prev_furthest = self._ep_furthest
        self._ep_furthest = max(self._ep_furthest, stage_end)
        # Milestones first reached in this episode at this step (fastest-to-milestone records, src.milestone_stats):
        # [milestone index, steps since the episode start, from the base start state].  A stagger-offset first episode
        # has a wrong step count, so it reports nothing.
        ms_new = None if self._partial else [[s - 1, int(self.steps), is_base_start(ended_state)]
                                             for s in range(prev_furthest + 1, self._ep_furthest + 1)]
        ended_partial = self._partial and not terminated   # a white-out is a real episode end, not a stagger cut
        ended_ms = (self._ep_start_stage, self._ep_furthest,
                    None if self.path_progress is None else self.path_progress.best) if done else None
        ended_start_path = self._ep_start_path if done else None
        self._maybe_archive(after)
        final_obs = self._peek_obs(after) if truncated else None
        if done:
            self.reset()

        screen, ram_vec, st_now = self.get_obs()
        info_out = {
            "map": [st_now.map_group, st_now.map_number],
            "x": st_now.x,
            "y": st_now.y,
            "state": self.state_name,
            "battle": st_now.battle_mode != 0,
            "badges": st_now.badges,
            "ended_maps": ended_maps,
            "ended_state": ended_state,
            "reward_components": {k: float(v) for k, v in comps.items()},
            "route_phi": phi_end,  # -(story milestones left) after this step (feeds the entropy schedule, not the policy)
            "ep_start_phi": phi_start,  # the same quantity at the start of the episode this step belongs to
            "stage": stage_end,
            "truncated": bool(truncated),
            "terminated": bool(terminated),   # true termination (white-out with terminate_on_whiteout): no bootstrap
            "end_reason": end_reason,          # "whiteout" / "time_limit" / None while the episode runs on
            "partial_episode": bool(ended_partial and done),  # a staggered first episode that was cut short
            "ep_steps": int(self.steps),  # steps taken in the running episode (0 right after a reset)
            "ep_len": int(self.ep_len),
            "cur_furthest": int(self._ep_furthest),   # furthest story stage of the running episode (0 = none yet)
            "executed_action": exec_idx,
            # Dashboard fun-facts source: already-parsed RAM only (no extra reads); aggregated in LiveFeed.
            "fun": {
                "bm": int(st_now.battle_mode),
                "es": int(getattr(st_now, "enemy_mon_species", 0)),
                "ps": [int(v) for v in st_now.party_species],
                "pl": [int(v) for v in (getattr(st_now, "party_levels", ()) or ())],
                "ph": [[int(h[0]), int(h[1])] for h in (getattr(st_now, "party_hp", ()) or ())],
                "dc": bytes(getattr(st_now, "pokedex_caught", ()) or ()),
                "ds": bytes(getattr(st_now, "pokedex_seen", ()) or ()),
                "mo": int(getattr(st_now, "money", 0)),
            },
        }
        if self.action_mask:
            info_out["amask"] = pack_mask(self.last_mask)   # valid actions of the NEXT observation (the new episode's after a reset)
        if ended_ms is not None:   # the finished episode's milestone summary (src.milestone_stats)
            info_out["ended_start_stage"], info_out["ended_furthest"], info_out["ended_best_path"] = ended_ms
            info_out["ended_start_path"] = ended_start_path   # path progress the episode started at (source-split gain)
        if ms_new:
            info_out["ms_new"] = ms_new
        if final_obs is not None:
            info_out["final_obs"] = final_obs
        return screen, ram_vec, float(step_rew), bool(done), len(self.seen_maps), after.badges, info_out

    def _escape(self) -> None:
        """Reflexive escape after repeated blocked moves: step toward the least-visited open neighbor."""
        st_cur = self.reader.parse(self.emu.frame)
        tiles = getattr(st_cur, "tiles", {})
        candidates = []
        for d in ("left", "right", "up", "down"):
            if walkable(tiles.get(d, 0xFF), d):
                dx, dy = _DELTA[d]
                target = (st_cur.map_group, st_cur.map_number, st_cur.x + dx, st_cur.y + dy)
                candidates.append((self.tile_visit_counts.get(target, 0), d))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            escape_dir = candidates[0][1]
            ok = self.stepper.step(escape_dir)
            if not ok:
                dx, dy = _DELTA[escape_dir]
                target = (st_cur.map_group, st_cur.map_number, st_cur.x + dx, st_cur.y + dy)
                self.tile_visit_counts[target] = self.tile_visit_counts.get(target, 0) + 50
        else:
            self.emu.press("b", frames=8)
            self.emu.tick(12, render=False)
        self.blocked_count = 0

    def close(self) -> None:
        self.emu.close()


def _worker_fn(pipe, seed: int, start_states: list[str], max_steps: int, env_cfg: dict | None = None):
    """Child process running one EnvWorker behind a pipe (RESET / STEP / CLOSE).

    Any exception is sent back to the parent as a ``WorkerFailure`` (with the traceback) instead of dying silently.
    """
    worker = None
    try:
        worker = EnvWorker(seed, start_states, max_steps, env_cfg)
        while True:
            cmd, data_arg = pipe.recv()
            if cmd == "RESET":
                pipe.send(worker.reset())
            elif cmd == "STEP":
                pipe.send(worker.step(int(data_arg)))
            elif cmd == "MASK":   # valid-action mask of the current observation (after a RESET), packed bits
                pipe.send(pack_mask(worker.last_mask))
            elif cmd == "INTRINSIC":   # no reply: the parent does not wait for one
                worker.set_intrinsic_scale(float(data_arg))
            elif cmd == "CLOSE":
                break
    except (EOFError, KeyboardInterrupt):
        pass  # the parent went away or the user pressed Ctrl+C: just clean up
    except BaseException:
        try:
            pipe.send(WorkerFailure(traceback.format_exc()))
        except Exception:
            pass
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception:
                pass


class WorkerFailure:
    """Message a worker sends when it hits an exception (plain object so it pickles through the pipe)."""

    def __init__(self, text: str):
        self.text = text


class WorkerCrashed(RuntimeError):
    """A worker process died or raised; the message carries the env index, exit code and traceback if known."""


class VectorEnv:
    """Manages parallel PyBoy worker processes with multi-state curriculum distribution.

    ``env_cfg`` is forwarded to every ``EnvWorker`` (sticky/jitter/stage-balanced/swarm/seed).  After ``step`` the
    per-env telemetry is in ``last_infos``, the time-limit truncation flags in ``last_truncated`` and the final
    observations of truncated envs (before their reset) in ``last_final`` as ``{env: (screen, ram)}``.
    """

    def __init__(self, num_envs: int, start_states: list[str] | str, max_steps: int, env_cfg: dict | None = None,
                 worker_target=None):
        self.num_envs = num_envs
        worker_target = worker_target or _worker_fn  # tests inject crashing / stuck workers
        if isinstance(start_states, str):
            states_list = [s.strip() for s in start_states.split(",") if s.strip()]
        else:
            states_list = list(start_states)
        if not states_list:
            states_list = CURRICULUM_STATES

        self.pipes = []
        self.procs = []
        self.last_infos: list[dict | None] = [None] * num_envs  # per-env telemetry from the latest step()
        self.last_truncated = np.zeros(num_envs, dtype=bool)
        self.use_mask = bool((env_cfg or {}).get("action_mask", False))   # long_v8: per-env valid-action masks
        self.last_masks = np.ones((num_envs, MASK_N_ACTIONS), dtype=bool)   # (num_envs, 6), True = allowed
        self.last_final: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._closed = False
        try:
            for i in range(num_envs):
                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(
                    target=worker_target,
                    args=(child_conn, i, states_list, max_steps, env_cfg),
                    daemon=True,
                )
                p.start()
                # The parent must not keep the child's end open: otherwise a dead worker never gives EOFError.
                child_conn.close()
                self.pipes.append(parent_conn)
                self.procs.append(p)
        except BaseException:
            self.close()
            raise

    def _send(self, i: int, msg) -> None:
        try:
            self.pipes[i].send(msg)
        except (BrokenPipeError, EOFError, OSError, ConnectionError) as exc:
            raise WorkerCrashed(self._describe_death(i, f"could not send {msg[0]} ({exc!r})")) from exc

    def _describe_death(self, i: int, what: str) -> str:
        procs = getattr(self, "procs", None) or []
        code = "unknown"
        if i < len(procs):
            procs[i].join(timeout=1.0)
            code = procs[i].exitcode
        return f"env worker {i} died: {what} (exit code {code})"

    def _recv(self, i: int, poll_interval: float = 0.5):
        """Receive from worker ``i``; raises ``WorkerCrashed`` (never blocks forever) when it died or raised."""
        pipe = self.pipes[i]
        procs = getattr(self, "procs", None) or []
        poll = getattr(pipe, "poll", None)
        try:
            while poll is not None and i < len(procs) and not poll(poll_interval):
                if not procs[i].is_alive():
                    if poll(0):  # it answered and then exited
                        break
                    raise WorkerCrashed(self._describe_death(i, "process is no longer alive"))
            msg = pipe.recv()
        except (EOFError, OSError, ConnectionError) as exc:
            raise WorkerCrashed(self._describe_death(i, f"pipe closed ({exc!r})")) from exc
        if isinstance(msg, WorkerFailure):
            raise WorkerCrashed(f"env worker {i} raised an exception:\n{msg.text}")
        return msg

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            for i in range(len(self.pipes)):
                self._send(i, ("RESET", None))
            screens, rams = [], []
            for i in range(len(self.pipes)):
                s, r = self._recv(i)
                screens.append(s)
                rams.append(r)
            if getattr(self, "use_mask", False):   # valid-action masks of the first observations (workers answer MASK after a RESET)
                for i in range(len(self.pipes)):
                    self._send(i, ("MASK", None))
                for i in range(len(self.pipes)):
                    self.last_masks[i] = unpack_mask(self._recv(i))
        except BaseException:  # includes Ctrl+C: never leave workers (and their emulators) behind
            self.close()
            raise
        return np.array(screens, dtype=np.uint8), np.array(rams, dtype=np.float64)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        screens, rams, rews, dones, maps, badges = [], [], [], [], [], []
        self.last_final = {}
        self.last_truncated = np.zeros(self.num_envs, dtype=bool)
        replies = []
        try:
            for i, act in enumerate(actions):
                self._send(i, ("STEP", act))
            for i in range(len(self.pipes)):
                replies.append(self._recv(i))
        except BaseException:
            self.close()
            raise
        for i, reply in enumerate(replies):
            s, r, rew, d, m, b, info = reply
            final = info.pop("final_obs", None)
            if final is not None:
                self.last_final[i] = final
            self.last_truncated[i] = bool(info.get("truncated", False))
            self.last_infos[i] = info
            if getattr(self, "use_mask", False):
                self.last_masks[i] = unpack_mask(info.get("amask", 63))   # 63 = all six actions allowed
            screens.append(s)
            rams.append(r)
            rews.append(rew)
            dones.append(d)
            maps.append(m)
            badges.append(b)
        return (
            np.array(screens, dtype=np.uint8),
            np.array(rams, dtype=np.float64),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=bool),
            maps,
            badges,
        )

    def set_intrinsic_scale(self, scale: float) -> None:
        """Broadcast the exploration-payment weight to every worker (fire and forget; applies from their next step)."""
        for i in range(len(self.pipes)):
            self._send(i, ("INTRINSIC", float(scale)))

    def close(self):
        """Stop every worker; idempotent.  Workers that ignore CLOSE (stuck in a step) are terminated, then killed."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for pipe in getattr(self, "pipes", []):
            try:
                pipe.send(("CLOSE", None))
            except Exception:
                pass
        deadline = time.time() + 3.0
        for p in getattr(self, "procs", []):
            p.join(timeout=max(0.1, deadline - time.time()))
        for p in getattr(self, "procs", []):
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
                p.join(timeout=2.0)
        for pipe in getattr(self, "pipes", []):
            try:
                pipe.close()
            except Exception:
                pass


# --------------------------------------------------------------------------------------------- trainer helpers
def _finite_or_none(x) -> float | None:
    """JSON-safe number: NaN/inf become None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@torch.no_grad()
def policy_values(policy, screens_u8: np.ndarray, rams: np.ndarray, hx, device) -> torch.Tensor:
    """V(obs) for a batch of numpy observations ``(n, H, W, 3)`` uint8 / ``(n, D)``; one eval-mode forward pass.

    ``hx`` is the matching ``(1, n, H)`` hidden state (or None).  Used for the last observation of a rollout and for
    the final observation of time-limit-truncated episodes.  Returns a CPU tensor ``(n,)``.
    """
    s = torch.from_numpy(np.ascontiguousarray(screens_u8)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    r = torch.from_numpy(np.asarray(rams)).float().to(device)
    if policy.use_gru:
        _, val, _ = policy.forward_recurrent(s, r, hx)
    else:
        _, val = policy(s, r)
    return val.float().cpu()


@torch.no_grad()
def warmup_scaler(policy, vec_env, scaler, screens, rams, hx, rollouts: int, rollout_steps: int, device) -> dict:
    """Warm-up rollout(s) before update 0 (``long_v6``): step the envs with the current policy and feed ONLY the running
    reward scaler.  Nothing here is stored for training, no gradient step happens, nothing is logged as an update, the
    PPO update counter and the archives are untouched, and the envs just keep running their episodes.

    Why: every tile is novel at the start of a run, so the first rollout pays curiosity + new-tile bonuses on almost every
    step (mean reward 12.4 per env at update 0 in v5-1 against ~0 later, 254 of ~400 positive reward from those two terms).
    Without this the scaler starts from its (mean 0, var 1, count ~0) prior and its first batch statistics come from that
    burst alone, so the first real update would be scaled by a std that does not describe the run.  RND (Burda et al.
    2018) likewise initialises its observation / reward normalisation statistics with random-agent steps before training.

    Returns ``(screens, rams, hx, info)``; ``hx`` is carried on (zeroed where an episode ended).
    """
    std_before = scaler.std if scaler is not None else 1.0
    policy.eval()
    total, steps = 0.0, int(rollouts) * int(rollout_steps)
    for _ in range(steps):
        s = torch.from_numpy(screens).to(device).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        r = torch.from_numpy(rams).float().to(device)
        mkw = {"action_mask": torch.from_numpy(vec_env.last_masks).to(device)} if getattr(vec_env, "use_mask", False) else {}
        if policy.use_gru:
            logits, _val, next_hx = policy.forward_recurrent(s, r, hx, **mkw)
        else:
            logits, _val = policy(s, r, **mkw)
            next_hx = None
        actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        screens, rams, rews, dones, _maps, _badges = vec_env.step(actions)
        if scaler is not None:
            scaler.update(rews, dones)
        total += float(np.sum(rews))
        if next_hx is not None:
            if dones.any():
                next_hx[:, torch.from_numpy(np.asarray(dones)).to(device), :] = 0.0
            hx = next_hx
    n_env = max(1, len(screens))
    info = {"steps": steps, "mean_reward": total / n_env / max(1, int(rollouts)),
            "std_before": std_before, "std_after": scaler.std if scaler is not None else 1.0}
    return screens, rams, hx, info


EVAL_WORKER_INDEX = 9999   # RNG stream index of the evaluation env (training envs use 0..envs-1)


def eval_env_cfg(cfg: TrainConfig) -> dict:
    """Env config of the periodic-evaluation worker (long_v8): the training env config with every non-base start source
    switched off (no archives / checkpoints / swarm, no stagger offset, no length jitter, no start jitter) and its own seed."""
    env = worker_env_cfg(cfg, None, None, None)
    env.update({"swarm_dir": None, "swarm_prob": 0.0, "battle_dir": None, "battle_prob": 0.0, "route_dir": None,
                "route_prob": 0.0, "stagger_episodes": False, "episode_len_jitter": 0.0, "jitter_max": 0,
                "stage_balanced": False, "num_envs": 0, "seed": int(cfg.seed) + 104729})
    return env


class PeriodicEvaluator:
    """From-start evaluation of the CURRENT policy on one dedicated in-process env (``periodic_eval``; see ``src.periodic_eval``).

    Isolation (checked by tests/test_periodic_eval.py): its own emulator and ``EnvWorker`` (no archives, so nothing is
    written), its own ``torch.Generator`` for sampling (the global torch / numpy RNGs, which drive the PPO minibatch
    shuffles and the rollout sampling, are not consumed), its own GRU state, ``torch.no_grad()`` and eval mode (no BatchNorm
    statistic or parameter changes), and it never touches the reward scaler, the progress tracker or any counter.
    The early battle A-bias is switched off for the evaluation and restored afterwards.
    """

    def __init__(self, cfg: TrainConfig, device):
        self.cfg, self.device = cfg, device
        self.worker = None
        self.failures = 0

    def _worker(self):
        if self.worker is None:   # created lazily: no second emulator exists until the first evaluation
            self.worker = EnvWorker(EVAL_WORKER_INDEX, [EVAL_STATE], int(self.cfg.eval_max_steps), eval_env_cfg(self.cfg))
        return self.worker

    @torch.no_grad()
    def run(self, policy, update: int) -> dict:
        t0 = time.time()
        w = self._worker()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.cfg.seed) * 1000003 + int(update))
        saved_bias, was_training = float(policy.battle_a_bias), policy.training
        policy.set_battle_a_bias(0.0)
        policy.eval()
        try:
            episodes = []
            screen, ram = w.reset()   # the first episode always starts from the base start state
            for _ in range(int(self.cfg.eval_episodes)):
                hx = torch.zeros(1, 1, policy.gru_hidden_dim, device=self.device) if policy.use_gru else None
                ret, ms_steps, n_steps = 0.0, {}, 0
                while True:
                    s = torch.from_numpy(np.ascontiguousarray(screen)).to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
                    r = torch.from_numpy(np.asarray(ram)).float().to(self.device).unsqueeze(0)
                    mkw = {"action_mask": torch.from_numpy(w.last_mask.copy()).to(self.device).unsqueeze(0)} if w.action_mask else {}
                    if policy.use_gru:
                        logits, _v, hx = policy.forward_recurrent(s, r, hx, **mkw)
                    else:
                        logits, _v = policy(s, r, **mkw)
                    probs = torch.softmax(logits.float(), dim=-1).cpu()
                    act = int(torch.multinomial(probs, 1, generator=gen).item())
                    screen, ram, rew, done, _maps, _badges, info = w.step(act)
                    ret += float(rew)
                    n_steps += 1
                    for m_idx, m_steps, _base in (info.get("ms_new") or []):
                        ms_steps.setdefault(int(m_idx), int(m_steps))
                    if done:
                        # w.step() already started the next episode (a fresh base-start reset): its first obs is (screen, ram)
                        episodes.append(episode_record(ret, n_steps, info.get("ended_furthest"), info.get("ended_best_path"),
                                                       info.get("end_reason"), ms_steps))
                        break
        finally:
            policy.set_battle_a_bias(saved_bias)
            policy.train(was_training)
        return aggregate(episodes, _story.profile().count, update, time.time() - t0)

    def close(self) -> None:
        if self.worker is not None:
            try:
                self.worker.close()
            except Exception:
                pass
            self.worker = None


def fit_minibatches(envs: int, num_minibatches: int) -> int:
    """Minibatches are groups of whole environments; ``ppo_update`` splits them unevenly when the count does not
    divide ``envs`` (e.g. 7 envs, 4 minibatches -> groups of 2/2/2/1), so only the ``<= envs`` limit applies."""
    return max(1, min(int(num_minibatches), int(envs)))


MIB = 1024 * 1024
BYTES_PER_FRAME = 13 * MIB  # measured ~12.9 MiB of saved activations per frame (ResNet trunk, 144x160)
VRAM_BUDGET_FRACTION = 0.45  # share of VRAM for one minibatch's activations (gradients/optimizer/model need the rest)


def plan_minibatches(units: int, frames_per_unit: int, requested: int, vram_bytes: int | None, explicit: bool):
    """Pick the minibatch count so one minibatch's frames fit the VRAM budget.

    ``units`` are the things minibatches split (envs for a recurrent policy, samples otherwise) holding
    ``frames_per_unit`` frames each.  Returns ``(num_minibatches, message)``.  ``explicit`` (typed --minibatches) is
    never changed, only warned about.  ``vram_bytes`` None (CPU) means no cap.
    """
    requested = max(1, min(int(requested), units))
    if vram_bytes is None:
        return requested, f"minibatches: {requested} (no CUDA memory cap on CPU)"
    budget_frames = max(1, int(VRAM_BUDGET_FRACTION * vram_bytes / BYTES_PER_FRAME))
    units_per_mb = max(1, budget_frames // max(1, frames_per_unit))
    needed = min(units, math.ceil(units / units_per_mb))
    frames_now = math.ceil(units / requested) * frames_per_unit
    detail = (f"{frames_now} frames/minibatch at {requested} minibatches vs budget {budget_frames} frames "
              f"({VRAM_BUDGET_FRACTION:.0%} of {vram_bytes / 1024 ** 3:.1f} GiB / {BYTES_PER_FRAME // MIB} MiB per frame)")
    if requested >= needed:
        return requested, f"minibatches: {requested} ({detail}: fits)"
    if explicit:
        return requested, f"[warn] minibatches: keeping typed --minibatches {requested} ({detail}) - may run out of GPU memory; {needed} fit"
    return needed, f"minibatches: raised {requested} -> {needed} to fit GPU memory ({detail})"


def check_resume(save_path: str | Path) -> Path:
    """--resume needs an existing checkpoint (``SpatialVisionPolicy.load`` also looks in ``models/``); never train from scratch silently."""
    p = Path(save_path)
    for cand in (p, Path("models") / p.name):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"--resume given but no checkpoint at '{save_path}' (also looked in models/{p.name}). "
        "Pass --save <existing checkpoint> or drop --resume to start a fresh run."
    )


def resolve_start_states(cfg: TrainConfig, state: str | None, states: list[str] | None, curriculum: bool | None) -> list[str]:
    """--states wins; --no-curriculum means the single --state; else the config's start_states, else the curriculum."""
    if states:
        return list(states)
    if curriculum is False:
        return [state or "route29_entry"]
    if cfg.start_states:
        return list(cfg.start_states)
    return list(CURRICULUM_STATES)


def checkpoint_sidecar(save_path: str | Path) -> Path:
    """Optimizer state lives next to the checkpoint (the ``policy.save`` format is left alone)."""
    p = Path(save_path)
    return p.with_name(p.stem + ".optim.pt")


def best_path(save_path: str | Path) -> Path:
    """Best-smoothed-score weights live next to the latest checkpoint: ``logs/<run>.best.pt``."""
    p = Path(save_path)
    return p.with_name(p.stem + ".best.pt")


def _atomic(write, path: str | Path) -> None:
    """``write(tmp_path)`` then ``os.replace`` so a crash never leaves a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(str(tmp))
    os.replace(tmp, path)


def save_best(policy, save_path: str | Path) -> Path:
    out = best_path(save_path)
    _atomic(policy.save, out)
    return out


def save_checkpoint(policy, optimizer, save_path: str | Path, update: int, route_progress: float = 0.0,
                    extra: dict | None = None) -> None:
    """Atomic write of the latest weights and the sidecar (optimizer, update count, route progress, ``extra``)."""
    blob = {"optimizer": optimizer.state_dict(), "update": int(update), "route_progress": float(route_progress)}
    blob.update(extra or {})
    _atomic(policy.save, save_path)
    _atomic(lambda tmp: torch.save(blob, tmp), checkpoint_sidecar(save_path))


def load_sidecar(save_path: str | Path, device) -> dict | None:
    """The whole sidecar dict, or None when missing/unreadable (with a clear message)."""
    side = checkpoint_sidecar(save_path)
    if not side.exists():
        print(f"[resume] WARNING: optimizer sidecar {side} not found: weights loaded, Adam moments, update counter, "
              "reward scaler and route progress restart from zero.", flush=True)
        return None
    try:
        return torch.load(str(side), map_location=device)
    except Exception as e:
        print(f"[resume] WARNING: could not read sidecar {side} ({e}); Adam moments, update counter, reward scaler and "
              "route progress restart from zero.", flush=True)
        return None


def load_optimizer_state(optimizer, save_path: str | Path, device) -> bool:
    """Restore Adam state from the sidecar; returns False (with a clear message) when it cannot be restored."""
    side = checkpoint_sidecar(save_path)
    if not side.exists():
        print(f"[resume] WARNING: optimizer sidecar {side} not found: weights loaded, Adam moments restart from zero.", flush=True)
        return False
    try:
        blob = torch.load(str(side), map_location=device)
        optimizer.load_state_dict(blob["optimizer"])
        print(f"[resume] restored optimizer state from {side} (saved at update {blob.get('update', '?')}).", flush=True)
        return True
    except Exception as e:
        print(f"[resume] could not restore optimizer state from {side} ({e}); Adam moments restart from zero.", flush=True)
        return False


def worker_env_cfg(cfg: TrainConfig, swarm_dir: str | Path | None, battle_dir: str | Path | None = None,
                   route_dir: str | Path | None = None) -> dict:
    return {
        "route_dir": str(route_dir) if (cfg.route_ckpt and route_dir) else None,
        "route_prob": cfg.route_ckpt_prob if cfg.route_ckpt else 0.0,
        "route_every": cfg.route_ckpt_every,
        "route_per_bucket": cfg.route_ckpt_per_bucket,
        "route_max": cfg.route_ckpt_max,
        "route_floor": cfg.route_ckpt_weight_floor,
        "dialogue_loop": cfg.dialogue_loop_params(),
        "retreat": cfg.retreat_params(),
        "stagnation": cfg.stagnation_params(),
        "terminate_on_whiteout": cfg.terminate_on_whiteout,
        "stagger_episodes": cfg.stagger_episodes,
        "episode_len_jitter": cfg.episode_len_jitter,
        "num_envs": cfg.envs,
        "swarm_furthest_floor": cfg.swarm_weight_floor if cfg.swarm_furthest_weighting else None,
        "battle_dir": str(battle_dir) if (cfg.battle_archive and battle_dir) else None,
        "battle_prob": cfg.battle_archive_prob if cfg.battle_archive else 0.0,
        "battle_max": cfg.battle_archive_max,
        "sticky_prob": cfg.sticky_prob,
        "jitter_max": cfg.jitter_max,
        "stage_balanced": cfg.stage_balanced,
        "swarm_dir": str(swarm_dir) if (cfg.swarm and swarm_dir) else None,
        "swarm_prob": cfg.swarm_prob,
        "seed": cfg.seed,
        "goal_potential": cfg.goal_potential,
        "dwell": cfg.dwell_params(),
        "path_progress": cfg.path_progress_scale_or_none(),
        "dwell_exempt_loop": cfg.dwell_exempt_loop,
        "idle_press": cfg.idle_press_params(),
        "disable_bump_penalty": cfg.disable_bump_penalty,
        "disable_loop_penalty": cfg.disable_loop_penalty,
        "explore_cap": cfg.explore_cap_override(),
        "heal_reward": cfg.heal_reward,
        "heal_scale": cfg.heal_reward_scale,
        "heal_budget": cfg.heal_budget,
        "battle_lowhp_weight": cfg.battle_lowhp_weight if cfg.battle_archive else 0.0,
        "battle_lowhp_threshold": cfg.battle_lowhp_threshold,
        "heal_care": cfg.heal_care_params(),
        "heal_trip_exempt": cfg.heal_trip_exempt,
        "heal_trip_threshold": cfg.lowhp_exposure_threshold,
        "action_mask": cfg.battle_action_mask,
        "egg_errand_route": cfg.egg_errand_route,
    }


def train(
    cfg: TrainConfig,
    *,
    states: list[str],
    resume: bool = False,
    push_weights: bool = False,
    batch_size: int | None = None,
    live: bool = True,
    live_dir: str = "logs/live",
    runs_dir: str | None = "logs/runs",
    run_id: str | None = None,
    frame_every: int = 5,
    metrics_path: str = "logs/training_metrics.jsonl",
    minibatches_typed: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gru = cfg.recurrent
    _story.set_profile(cfg.egg_errand_route)   # milestone list, story block and observation width of this run
    obs_dim = deep_obs_dim()
    num_envs, rollout_steps, updates = cfg.envs, cfg.rollout_steps, cfg.updates
    save_path = cfg.save_path
    if resume:
        check_resume(save_path)  # raises before any worker is spawned
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Initializing SpatialVisionPolicy training on {device} (recurrent={use_gru}, config={cfg.name})...")

    use_aux = bool(cfg.aux_loss and use_gru)
    policy = SpatialVisionPolicy(obs_dim=obs_dim, num_actions=N_ACTIONS, use_gru=use_gru, gru_hidden_dim=512,
                                 aux_heads=use_aux)
    resumed = False
    sidecar = None
    if resume:
        try:
            policy = SpatialVisionPolicy.load(save_path, device=device, override_use_gru=use_gru,
                                              override_obs_dim=obs_dim, check_schema=True,
                                              override_aux_heads=use_aux)
        except CheckpointIncompatibleError as e:
            raise RuntimeError(f"--resume: {e}") from e
        except Exception as e:
            raise RuntimeError(f"--resume: could not load checkpoint '{save_path}': {e}") from e
        resumed = True
        print(f"Loaded existing checkpoint from {save_path} (use_gru={policy.use_gru}, obs_dim={policy.obs_dim})")

    policy.to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=1e-4)
    if resumed:
        sidecar = load_sidecar(save_path, device)
        if sidecar is not None:
            try:
                optimizer.load_state_dict(sidecar["optimizer"])
                print(f"[resume] restored optimizer state (saved at update {sidecar.get('update', '?')}).", flush=True)
            except Exception as e:
                print(f"[resume] WARNING: could not restore optimizer state ({e}); Adam moments restart from zero.", flush=True)
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr
    print(f"Policy ready with {policy.param_count:,} trainable parameters.")

    hp = PPOHParams(**dataclasses.asdict(cfg.ppo))
    hp.aux_coef = float(cfg.aux_coef) if (use_aux and policy.aux_enabled) else 0.0
    hp.aux_horizon = int(cfg.aux_horizon)
    if not policy.use_gru and batch_size:  # feed-forward minibatches are samples, sized by --batch-size
        hp.num_minibatches = max(1, math.ceil(num_envs * rollout_steps / int(batch_size)))
    explicit_mb = minibatches_typed or bool(batch_size and not policy.use_gru)
    vram = torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None
    units, per_unit = (num_envs, rollout_steps) if policy.use_gru else (num_envs * rollout_steps, 1)
    hp.num_minibatches, mb_msg = plan_minibatches(units, per_unit, hp.num_minibatches, vram, explicit_mb)
    print(f"[memory] {mb_msg}", flush=True)
    entropy_schedule = EntropySchedule(cfg.entropy_start, cfg.entropy_end)
    guard_params = cfg.entropy_guard_params()
    entropy_guard = EntropyGuard(**guard_params) if guard_params else None   # long_v5 safety net (coefficient only)
    last_entropy: float | None = None
    progress = RouteProgressTracker()
    scaler = RunningReturnScaler(num_envs, hp.gamma) if cfg.reward_scale == "running_std" else None
    start_update = 0
    best_reward = -float("inf")  # best smoothed (mean of the last BEST_WINDOW updates) raw mean episode reward
    if sidecar is not None:
        start_update = int(sidecar.get("update", -1)) + 1
        if "progress" in sidecar:
            progress.load_state_dict(sidecar["progress"])
        elif sidecar.get("route_progress"):
            progress.fracs.append(float(sidecar["route_progress"]))
        if scaler is not None and "reward_scaler" in sidecar:
            scaler.load_state_dict(sidecar["reward_scaler"])
        if sidecar.get("best_score") is not None:
            best_reward = float(sidecar["best_score"])
        print(f"[resume] continuing at update {start_update} (route_progress={progress.value:.3f}"
              f"{'' if scaler is None else f', reward std={scaler.std:.3f}'}).", flush=True)
    print(f"Reward scaling: {cfg.reward_scale}; BatchNorm in update: {hp.bn_mode}; value clip: {hp.vf_clip}", flush=True)

    run_name = run_id or cfg.run_id or time.strftime("%Y%m%d-%H%M%S")
    history_dir = Path(runs_dir) / run_name if runs_dir else None
    if history_dir is not None:
        cfg_path = dump_effective_config(cfg, history_dir)
        print(f"Effective config -> {cfg_path}", flush=True)
    archive_root = (Path(runs_dir).parent if runs_dir else Path("logs")) / "swarm" / run_name
    swarm_dir = archive_root if cfg.swarm else None
    battle_dir = archive_root / "battle" if cfg.battle_archive else None
    route_dir = archive_root / "route" if cfg.route_ckpt else None
    route_archive = (RouteCheckpointArchive(route_dir, cfg.route_ckpt_every, cfg.route_ckpt_per_bucket,
                                            cfg.route_ckpt_max, cfg.route_ckpt_weight_floor) if route_dir else None)

    print(
        f"Spawning {num_envs} vectorized PyBoy workers across states: {states} "
        f"(max_steps={cfg.max_steps}, sticky={cfg.sticky_prob}, jitter<={cfg.jitter_max}, "
        f"stage_balanced={cfg.stage_balanced}, swarm={'on -> ' + str(swarm_dir) if swarm_dir else 'off'})..."
    )
    print(
        f"[long_v3 features] stagger={cfg.stagger_episodes} (len jitter +-{cfg.episode_len_jitter:.0%}), "
        f"swarm_prob={cfg.swarm_prob} furthest_weighting={cfg.swarm_furthest_weighting}, "
        f"battle_archive={'on -> ' + str(battle_dir) + f' (prob {cfg.battle_archive_prob}, max {cfg.battle_archive_max})' if battle_dir else 'off'}, "
        f"battle_a_bias={cfg.battle_a_bias}->0 over {cfg.battle_a_bias_updates} updates",
        flush=True,
    )
    if cfg.dwell_penalty:
        print(
            f"[dwell penalty] window={cfg.dwell_window} steps, <= {cfg.dwell_max_tiles} tiles, "
            f"penalty {cfg.dwell_min_penalty}->{cfg.dwell_max_penalty}/step over {cfg.dwell_ramp_steps} steps, "
            f"budget {cfg.dwell_budget}/episode, level targets by stage {list(cfg.dwell_level_targets)}, "
            f"grind grace {cfg.dwell_grind_grace}, exempt loop={cfg.dwell_exempt_loop}",
            flush=True,
        )
    if cfg.name == "long_v4" or cfg.heal_reward or cfg.aux_loss or cfg.battle_lowhp_weight > 0 or cfg.intrinsic_decay_updates:
        print(
            f"[long_v4 features] gamma={cfg.ppo.gamma} lambda={cfg.ppo.gae_lambda} rollout={cfg.rollout_steps} "
            f"(whole rollout = one BPTT sequence, hidden state carried across rollouts) | entropy {cfg.entropy_start}->{cfg.entropy_end} | "
            f"explore budget {cfg.explore_budget} weight 1.0->{cfg.intrinsic_weight_end} over {cfg.intrinsic_decay_updates} updates | "
            f"heal reward={'x%g budget %g' % (cfg.heal_reward_scale, cfg.heal_budget) if cfg.heal_reward else 'off'} | "
            f"low-HP battle weight {cfg.battle_lowhp_weight} (<= {cfg.battle_lowhp_threshold:.0%} HP) | "
            f"aux heads={'coef %g horizon %d' % (cfg.aux_coef, cfg.aux_horizon) if hp.aux_coef > 0 else 'off'}",
            flush=True,
        )
    if cfg.idle_press_penalty or cfg.disable_bump_penalty or cfg.disable_loop_penalty or cfg.entropy_boost or cfg.entropy_anneal_updates:
        print(
            f"[long_v5 anti-collapse] bump penalty={'OFF' if cfg.disable_bump_penalty else 'on'} loop penalty={'OFF' if cfg.disable_loop_penalty else 'on'} | "
            f"idle-press penalty={'%g/press, budget %g' % (cfg.idle_press_cost, cfg.idle_press_budget) if cfg.idle_press_penalty else 'off'} | "
            f"entropy {cfg.entropy_start}->{cfg.entropy_end} (anneal >= {cfg.entropy_anneal_updates} updates) | "
            f"entropy boost={'coef %g below %g until update %d (release at +%g)' % (cfg.entropy_boost_coef, cfg.entropy_boost_threshold, cfg.entropy_boost_until, cfg.entropy_boost_margin) if cfg.entropy_boost else 'off'}",
            flush=True,
        )
    if (cfg.route_ckpt or cfg.dialogue_loop_penalty or cfg.warmup_rollout or cfg.retreat_penalty
            or cfg.stagnation_penalty or cfg.terminate_on_whiteout):
        print(
            f"[long_v6] route checkpoints={'every %d tiles, %d/bucket, <= %d files, %.0f%% of resets, weight floor %g -> %s' % (cfg.route_ckpt_every, cfg.route_ckpt_per_bucket, cfg.route_ckpt_max, 100 * cfg.route_ckpt_prob, cfg.route_ckpt_weight_floor, route_dir) if cfg.route_ckpt else 'off'} | "
            f"reset order: battle {cfg.battle_archive_prob if cfg.battle_archive else 0}, swarm {cfg.swarm_prob if cfg.swarm else 0}, route {cfg.route_ckpt_prob if cfg.route_ckpt else 0}, rest start state | "
            f"dialogue-loop penalty={'%g/step, > %d steps in %d or > %d idle talks, budget %g' % (cfg.dialogue_loop_cost, cfg.dialogue_loop_max_steps, cfg.dialogue_loop_window, cfg.dialogue_loop_max_talks, cfg.dialogue_loop_budget) if cfg.dialogue_loop_penalty else 'off'} | "
            f"warm-up rollouts={cfg.warmup_rollout} | "
            f"retreat cost={'%g x path scale per tile lost, budget %g' % (cfg.retreat_cost_frac, cfg.retreat_budget) if cfg.retreat_penalty else 'off'} | "
            f"stagnation penalty={'%g->%g/step after %d steps without a new best (ramp %d, budget %g)' % (cfg.stagnation_min, cfg.stagnation_max, cfg.stagnation_after, cfg.stagnation_ramp, cfg.stagnation_budget) if cfg.stagnation_penalty else 'off'} | "
            f"terminate on white-out={cfg.terminate_on_whiteout}",
            flush=True,
        )
    if cfg.source_split_metrics or cfg.heal_care_params() or cfg.heal_trip_exempt:
        print(
            f"[long_v7] source-split metrics={cfg.source_split_metrics} | "
            f"low-HP exposure={'%g->%g/step (ramp %d) below %.0f%% HP, budget %g' % (cfg.lowhp_exposure_min, cfg.lowhp_exposure_max, cfg.lowhp_exposure_ramp, 100 * cfg.lowhp_exposure_threshold, cfg.lowhp_exposure_budget) if cfg.lowhp_exposure_penalty else 'off'} | "
            f"battle-survive bonus={'+%g above %.0f%% HP, budget %g' % (cfg.battle_survive_amount, 100 * cfg.battle_survive_threshold, cfg.battle_survive_budget) if cfg.battle_survive_bonus else 'off'} | "
            f"potion-use reward={'+%g below %.0f%% HP, budget %g' % (cfg.potion_use_amount, 100 * cfg.potion_use_threshold, cfg.potion_use_budget) if cfg.potion_use_reward else 'off'} | "
            f"heal-trip exemption={cfg.heal_trip_exempt}",
            flush=True,
        )
    if cfg.path_progress:
        _steps = _story.profile().path_steps
        print(f"[path progress] +{cfg.path_progress_scale} per tile of new best progress along the story path "
              f"(best-ever per episode, never negative, start progress free; ~{cfg.path_progress_scale * _steps:.1f} for the whole {_steps}-step path)",
              flush=True)
    if cfg.egg_errand_route:
        _p = _story.profile()
        print(f"[egg errand] {_p.count} milestones {list(_p.names)} | story block {_p.feature_dim} floats, observation {obs_dim} | "
              f"path phases {list(_p.phase_steps)} = {_p.path_steps} steps (egg tile -> Elm's lab -> Falkner) | "
              f"story reward {_p.total_reward:g} | level targets {list(cfg.dwell_level_targets)}", flush=True)
    vec_env = VectorEnv(num_envs=num_envs, start_states=states, max_steps=cfg.max_steps,
                        env_cfg=worker_env_cfg(cfg, swarm_dir, battle_dir, route_dir))
    screens_cur, rams_cur = vec_env.reset()

    hidden = policy.gru_hidden_dim if use_gru else 512
    hx = torch.zeros(1, num_envs, hidden, device=device) if policy.use_gru else None
    if cfg.warmup_rollout > 0 and scaler is not None and not (sidecar is not None and "reward_scaler" in sidecar):
        try:
            screens_cur, rams_cur, hx, wu = warmup_scaler(policy, vec_env, scaler, screens_cur, rams_cur, hx,
                                                          cfg.warmup_rollout, rollout_steps, device)
        except BaseException:
            vec_env.close()
            raise
        print(f"[warmup] {cfg.warmup_rollout} rollout(s) x {rollout_steps} steps fed the reward scaler only "
              f"(mean reward {wu['mean_reward']:.2f}/env/rollout, scaler std {wu['std_before']:.3f} -> {wu['std_after']:.3f}); "
              f"no update, no gradient step, update counter untouched", flush=True)
    elif cfg.warmup_rollout > 0:
        print("[warmup] skipped (reward scaling off or the reward scaler was restored from the checkpoint)", flush=True)
    monitor = TerminalMonitor(total_updates=start_update + updates, jsonl_path=metrics_path)
    feed = None
    if live:
        feed = LiveFeed(
            live_dir,
            num_envs=num_envs,
            action_names=ACTIONS,
            history_dir=history_dir,
            frame_every=frame_every,
            run_info={
                "total_updates": start_update + updates,
                "rollout_steps": rollout_steps,
                "num_envs": num_envs,
                "lr": cfg.lr,
                "entropy_coef": [cfg.entropy_start, cfg.entropy_end],
                "recurrent": policy.use_gru,
                "resume": resume,
                "ms_records": True,   # this run logs the fastest-to-milestone records (dashboard shows the table)
                "egg_errand_route": bool(cfg.egg_errand_route),   # the dashboard draws the three route phases for such runs
                "milestone_labels": list(milestone_labels()),   # human names, in milestone order (dashboard chips / tables)
                "path_total": int(_story.profile().path_steps),
                "curriculum_states": sorted(set(states)),
                "device": str(device),
                "config_name": cfg.name,
                "config": cfg.to_dict(),
            },
        )
        print(f"Live dashboard feed -> {live_dir}  (view with: python -m tools.dashboard)", flush=True)
        if history_dir is not None:
            print(f"Run history (replay later) -> {history_dir}", flush=True)

    B, T = num_envs, rollout_steps
    # Rollout buffers live on the CPU; screens stay uint8 (4x smaller than float32) and ppo_update moves minibatches.
    buf_screens = torch.empty((B, T, *SCREEN_SHAPE), dtype=torch.uint8)
    buf_rams = torch.empty((B, T, obs_dim), dtype=torch.float32)
    buf_actions = torch.empty((B, T), dtype=torch.long)
    buf_logps = torch.empty((B, T), dtype=torch.float32)
    buf_values = torch.zeros((T, B), dtype=torch.float32)
    buf_rewards = torch.zeros((T, B), dtype=torch.float32)
    buf_dones = torch.zeros((T, B), dtype=torch.bool)
    buf_trunc = torch.zeros((T, B), dtype=torch.bool)
    buf_boot = torch.zeros((T, B), dtype=torch.float32)
    buf_raw = torch.zeros((T, B), dtype=torch.float32)  # unscaled rewards, for logging
    use_mask = bool(cfg.battle_action_mask)
    # long_v8: the valid-action mask each step's action was sampled under; stored so the PPO update re-applies the SAME mask.
    buf_masks = torch.ones((B, T, MASK_N_ACTIONS), dtype=torch.bool) if use_mask else None
    evaluator = PeriodicEvaluator(cfg, device) if cfg.periodic_eval else None   # long_v8: from-start evaluation, own env

    ms_stats = MilestoneStats()
    split_stats = SourceSplitStats() if cfg.source_split_metrics else None   # long_v7: learned vs head start, by reset source
    # Fastest-to-milestone records (steps from the base start state); rebuilt from this run's history.jsonl on resume.
    ms_records = (MilestoneRecords.from_history(history_dir / "history.jsonl")
                  if (resumed and history_dir is not None) else MilestoneRecords())
    last_intrinsic = 1.0
    recent_scores: deque[float] = deque(maxlen=BEST_WINDOW)
    reward_window_8: deque[float] = deque(maxlen=REWARD_CYCLE_WINDOW)  # mean over 8 updates: hides the lockstep cycle
    last_done: int | None = None  # last completed update index
    last_saved: int | None = None

    def save_latest(update_idx: int) -> None:
        nonlocal last_saved
        extra = {"progress": progress.state_dict(), "best_score": best_reward if math.isfinite(best_reward) else None,
                 "battle_a_bias": float(policy.battle_a_bias)}  # the schedule itself is in config.json
        if scaler is not None:
            extra["reward_scaler"] = scaler.state_dict()
        save_checkpoint(policy, optimizer, save_path, update_idx, progress.value, extra=extra)
        last_saved = update_idx

    try:
        for update in range(start_update, start_update + updates):
            t0 = time.time()
            buf_values.zero_(), buf_rewards.zero_(), buf_dones.zero_(), buf_trunc.zero_(), buf_boot.zero_()
            buf_raw.zero_()
            initial_hx = hx.detach().clone() if (policy.use_gru and hx is not None) else None
            entropy_coef = entropy_schedule.value(anneal_fraction(progress.value, update, cfg.entropy_anneal_updates))
            entropy_boosted = False
            if entropy_guard is not None:   # the coefficient only enters the PPO loss: rollout log-probs are unaffected
                entropy_coef, entropy_boosted = entropy_guard.coef(update, entropy_coef, last_entropy)
            # Early A-bias in battle menus: set before the rollout, constant until the PPO update below has used it, so
            # the rollout's log-probs (old_logps) and the update's evaluate_actions describe the same distribution.
            a_bias = battle_a_bias_value(update, cfg.battle_a_bias, cfg.battle_a_bias_updates)
            policy.set_battle_a_bias(a_bias)
            iw = intrinsic_weight_value(update, cfg.intrinsic_weight_end, cfg.intrinsic_decay_updates)
            if abs(iw - last_intrinsic) > 1e-9:   # workers only hear about a change (never for long_v3)
                vec_env.set_intrinsic_scale(iw)
                last_intrinsic = iw
            if cfg.lr_end_frac < 1.0:  # linear lr decay over this invocation's updates
                lr_now = cfg.lr * (1.0 - (1.0 - cfg.lr_end_frac) * min(1.0, (update - start_update) / max(1, updates)))
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

            # 1. Collect rollouts with batched inference
            policy.eval()
            phase = {"infer": 0.0, "env": 0.0, "feed": 0.0, "trunc": 0.0, "buf": 0.0}
            whiteouts = 0   # episodes ended by a white-out in this rollout (terminate_on_whiteout)
            mask_battle_steps = mask_active_steps = 0   # long_v8: env-steps in battle / of those with at least one masked action
            mask_valid_sum = 0.0                        # sum over those steps of the fraction of allowed actions
            resets_total = resets_route = 0   # resets in this rollout, and how many of them loaded a route checkpoint
            _pc = time.perf_counter
            for step in range(T):
                _t = _pc()
                screens_t = (
                    torch.from_numpy(screens_cur).to(device).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
                )
                rams_t = torch.from_numpy(rams_cur).float().to(device)

                # Valid-action mask of THIS observation (set by the previous step()/reset()); copied because step() overwrites it.
                step_mask_np = vec_env.last_masks.copy() if use_mask else None
                mkw = {"action_mask": torch.from_numpy(step_mask_np).to(device)} if use_mask else {}
                with torch.no_grad():
                    if policy.use_gru:
                        logits, val, next_hx = policy.forward_recurrent(screens_t, rams_t, hx, **mkw)
                    else:
                        logits, val = policy(screens_t, rams_t, **mkw)
                        next_hx = None

                    dist = torch.distributions.Categorical(logits=logits)
                    actions_t = dist.sample()
                    logp_t = dist.log_prob(actions_t)

                actions_np = actions_t.cpu().numpy()
                _t2 = _pc()
                phase["infer"] += _t2 - _t
                screens_next, rams_next, rews, dones, map_counts, badges = vec_env.step(actions_np)
                infos = vec_env.last_infos
                _t3 = _pc()
                phase["env"] += _t3 - _t2
                if feed is not None:
                    feed.observe(infos=infos, actions=actions_np, rewards=rews, dones=dones, screens=screens_next)
                _t4 = _pc()
                phase["feed"] += _t4 - _t3

                # Milestones first reached this step -> fastest-to-milestone records (live toast + events feed).
                for i, inf in enumerate(infos):
                    new = (inf or {}).get("ms_new")
                    if not new:
                        continue
                    for m_idx, m_steps, m_base in new:
                        for ev in ms_records.observe(i, m_idx, m_steps, update, bool(m_base)):
                            if feed is not None:
                                feed.record_milestone(ev, ms_records.snapshot())

                # Route progress per finished episode drives the entropy schedule.
                for i in np.nonzero(dones)[0]:
                    inf = infos[i] or {}
                    resets_total += 1
                    whiteouts += 1 if inf.get("end_reason") == "whiteout" else 0
                    resets_route += 1 if str(inf.get("state", "")).startswith(ROUTE_PREFIX) else 0   # the NEW episode's start
                    ms_stats.add_episode(inf.get("ended_start_stage"), inf.get("ended_furthest"),
                                         inf.get("ended_best_path"), partial=bool(inf.get("partial_episode")))
                    if split_stats is not None:
                        split_stats.add_episode(source_of(inf.get("ended_state")), inf.get("ended_start_stage"),
                                                inf.get("ended_furthest"), inf.get("ended_best_path"),
                                                inf.get("ended_start_path"), partial=bool(inf.get("partial_episode")))
                    if inf.get("partial_episode"):
                        continue  # a staggered first episode cut short: not a full-length progress sample
                    progress.add(inf.get("ep_start_phi"), inf.get("route_phi"))

                # Time-limit truncation: bootstrap with V(final observation), one batched pass for those envs.
                truncs = vec_env.last_truncated
                if truncs.any():
                    idx = np.nonzero(truncs)[0]
                    f_scr = np.stack([vec_env.last_final[i][0] for i in idx])
                    f_ram = np.stack([vec_env.last_final[i][1] for i in idx])
                    f_hx = next_hx[:, torch.from_numpy(idx).to(device), :] if next_hx is not None else None
                    buf_boot[step, torch.from_numpy(idx)] = policy_values(policy, f_scr, f_ram, f_hx, device)
                    buf_trunc[step] = torch.from_numpy(truncs)

                _t5 = _pc()
                phase["trunc"] += _t5 - _t4
                if policy.use_gru and next_hx is not None:
                    if dones.any():
                        done_mask = torch.from_numpy(dones).to(device)
                        next_hx[:, done_mask, :] = 0.0
                    hx = next_hx

                buf_screens[:, step] = torch.from_numpy(screens_cur).permute(0, 3, 1, 2)
                buf_rams[:, step] = rams_t.cpu()
                buf_actions[:, step] = actions_t.cpu()
                buf_logps[:, step] = logp_t.cpu()
                if use_mask:
                    buf_masks[:, step] = torch.from_numpy(step_mask_np)
                    in_battle = rams_cur[:, BATTLE_FLAG_INDEX] > 0.5   # observation of THIS step (rams_cur is replaced below)
                    nb = int(in_battle.sum())
                    mask_battle_steps += nb
                    mask_valid_sum += float(step_mask_np[in_battle].mean(axis=1).sum()) if nb else 0.0
                    mask_active_steps += int((~step_mask_np[in_battle].all(axis=1)).sum()) if nb else 0
                buf_raw[step] = torch.from_numpy(rews)
                buf_rewards[step] = torch.from_numpy(scaler.scale(rews, dones) if scaler is not None else rews)
                buf_dones[step] = torch.from_numpy(dones)
                buf_values[step] = val.cpu()

                screens_cur, rams_cur = screens_next, rams_next
                phase["buf"] += _pc() - _t5

            _t_roll = _pc()
            # 2. GAE with truncation bootstrapping and the value of the current observations
            last_value = policy_values(policy, screens_cur, rams_cur, hx, device)
            returns, advantages = compute_gae(
                buf_rewards, buf_values, buf_dones, last_value, hp.gamma, hp.gae_lambda,
                truncated=buf_trunc, bootstrap_values=buf_boot,
            )
            mean_rew = float(buf_raw.sum(dim=0).mean().item())  # raw (unscaled) reward per env per rollout
            reward_window_8.append(mean_rew)
            live_infos = [i for i in vec_env.last_infos if i]
            spread = phase_spread([i.get("ep_steps", 0) for i in live_infos], [i.get("ep_len", cfg.max_steps) for i in live_infos])
            total_samples = T * B

            # 3. PPO update (env-wise minibatches)
            batch = RolloutBatch(
                screens=buf_screens,
                rams=buf_rams,
                actions=buf_actions,
                old_logps=buf_logps,
                returns=returns.transpose(0, 1).contiguous(),
                advantages=advantages.transpose(0, 1).contiguous(),
                initial_hx=initial_hx,
                dones=buf_dones.transpose(0, 1).contiguous(),
                masks=buf_masks,
            )
            _t_upd0 = _pc()
            stats = ppo_update(policy, optimizer, batch, hp, entropy_coef=entropy_coef)
            if hx is not None:
                hx = hx.detach()
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t_upd1 = _pc()

            dt = time.time() - t0
            print(
                f"[time] update {update}: total {dt:.1f}s | rollout: policy {phase['infer']:.1f}s, envs {phase['env']:.1f}s, "
                f"feed {phase['feed']:.1f}s, truncation {phase['trunc']:.1f}s, buffers {phase['buf']:.1f}s | "
                f"gae+batch {_t_upd0 - _t_roll:.1f}s | ppo update {_t_upd1 - _t_upd0:.1f}s",
                flush=True,
            )
            eval_payload = None
            if evaluator is not None and is_eval_update(update, cfg.eval_every):   # long_v8: after dt, so dt excludes it
                try:
                    eval_payload = evaluator.run(policy, update)
                    print(f"[eval] update {update}: best progress mean {eval_payload['eval_best_progress_mean']} / max "
                          f"{eval_payload['eval_best_progress_max']}, white-outs {eval_payload['eval_whiteout_frac']:.0%}, "
                          f"return {eval_payload['eval_return_mean']} ({cfg.eval_episodes} x {cfg.eval_max_steps} steps from "
                          f"{EVAL_STATE}, {eval_payload['eval_seconds']}s)", flush=True)
                except Exception as e:   # an evaluation problem must never stop the training
                    evaluator.failures += 1
                    print(f"[eval] WARNING: evaluation at update {update} failed ({e!r}); training continues", flush=True)
            max_maps = max(map_counts)
            max_badges = max(badges)
            cur_lr = optimizer.param_groups[0]["lr"]
            mean_entropy = float(stats["entropy"])
            last_entropy = mean_entropy if math.isfinite(mean_entropy) else last_entropy
            mean_val_loss = float(stats["value_loss"])
            mean_pol_loss = float(stats["policy_loss"])
            mean_kl = float(stats["approx_kl"])
            mean_clip = float(stats["clip_frac"])
            extra = {
                "n_optimizer_steps": int(stats["n_optimizer_steps"]),
                "noise_scale": _finite_or_none(stats["noise_scale"]),
                "explained_variance": _finite_or_none(stats["explained_variance"]),
                "early_stopped": bool(stats["early_stopped"]),
                "approx_kl_first": _finite_or_none(stats["approx_kl_first"]),
                "grad_norm": _finite_or_none(stats["grad_norm"]),
                "entropy_coef": round(entropy_coef, 5),
                "route_progress": round(progress.value, 4),
                "reward_scale_std": round(scaler.std, 4) if scaler is not None else 1.0,
                "phase_spread": round(spread, 4),
                "reward_8": round(float(np.mean(reward_window_8)), 3),
                "battle_a_bias": round(a_bias, 4),
                "battle_archive_size": len(archive_files(battle_dir)) if battle_dir else 0,
                "swarm_archive_size": len(archive_files(swarm_dir)) if swarm_dir else 0,
                "intrinsic_weight": round(last_intrinsic, 4),
            }
            if entropy_guard is not None:
                extra["entropy_boost"] = 1.0 if entropy_boosted else 0.0
            if use_mask:
                extra["mask_battle_steps"] = int(mask_battle_steps)   # env-steps spent in battle this update
                extra["mask_valid_frac"] = round(mask_valid_sum / mask_battle_steps, 4) if mask_battle_steps else 1.0
                extra["mask_active_frac"] = round(mask_active_steps / mask_battle_steps, 4) if mask_battle_steps else 0.0
            if cfg.terminate_on_whiteout:
                extra["whiteout_episodes"] = int(whiteouts)   # episodes ended by a white-out (true terminations) this update
                extra["episodes_ended"] = int(resets_total)
            if route_archive is not None:
                counts = route_archive.counts()
                extra["route_ckpt_size"] = int(sum(counts.values()))
                extra["route_ckpt_max_bucket"] = int(max(counts)) if counts else 0
                extra["route_ckpt_buckets"] = {str(b): n for b, n in counts.items()}   # states per progress bucket
                extra["route_ckpt_resets"] = int(resets_route)
                extra["route_ckpt_reset_share"] = round(resets_route / resets_total, 4) if resets_total else 0.0
            ms_payload = ms_stats.end_update([i.get("cur_furthest") for i in live_infos])
            if split_stats is not None:
                ms_payload.update(split_stats.end_update())
            extra.update(ms_payload)
            if eval_payload:   # eval_* keys only on evaluation updates (metrics.jsonl, history.jsonl and state.json)
                extra.update(eval_payload)
            if hp.aux_coef > 0:
                extra["aux_map_loss"] = _finite_or_none(stats.get("aux_map_loss"))
                extra["aux_hp_loss"] = _finite_or_none(stats.get("aux_hp_loss"))
                extra["aux_battle_loss"] = _finite_or_none(stats.get("aux_battle_loss"))

            monitor.log_update(
                update=update,
                mean_reward=mean_rew,
                entropy=mean_entropy,
                value_loss=mean_val_loss,
                policy_loss=mean_pol_loss,
                approx_kl=mean_kl,
                clip_frac=mean_clip,
                lr=cur_lr,
                maps_discovered=max_maps,
                badges=max_badges,
                duration_sec=dt,
                samples=total_samples,
                extra=extra,
            )
            if feed is not None:
                feed.set_milestones(ms_payload)
                if eval_payload:
                    feed.set_eval(eval_payload, f"From-start evaluation at update {update}: best progress "
                                  f"{eval_payload['eval_best_progress_mean']} (max {eval_payload['eval_best_progress_max']})")
                feed.set_records(ms_records.snapshot(), ms_records.take_new())
                feed.on_update(
                    update,
                    total_samples,
                    metrics={
                        "update": update,
                        "mean_reward": round(mean_rew, 3),
                        "rolling_reward": round(float(monitor.last_rolling_avg or mean_rew), 3),
                        "entropy": round(mean_entropy, 4),
                        "value_loss": round(mean_val_loss, 4),
                        "policy_loss": round(mean_pol_loss, 4),
                        "approx_kl": round(mean_kl, 5),
                        "clip_frac": round(mean_clip, 4),
                        "lr": cur_lr,
                        "maps_discovered": max_maps,
                        "badges": max_badges,
                        "duration_sec": round(dt, 2),
                        "samples": total_samples,
                        "status": monitor.last_status,
                        **extra,
                    },
                )

            last_done = update
            recent_scores.append(mean_rew)
            smoothed = float(np.mean(recent_scores))
            if len(recent_scores) >= min(BEST_WINDOW, updates) and smoothed > best_reward:
                best_reward = smoothed
                save_best(policy, save_path)
            if (update + 1) % SAVE_EVERY == 0:
                save_latest(update)

        if last_done is not None and last_saved != last_done:
            save_latest(last_done)  # the final update is always saved
        print(f"\nTraining completed! Latest checkpoint: {save_path}; best (smoothed reward {best_reward:.2f}): {best_path(save_path)}", flush=True)
        if push_weights:
            try:
                from tools.sync_weights import sync_and_push
                sync_and_push(push=True, message=f"feat(weights): deep_policy.pt retrained (updates={updates}, rew={best_reward:.2f})")
            except Exception as e:
                print(f"[Warning] Could not auto-push weights: {e}", flush=True)
    finally:
        if last_done is not None and last_saved != last_done:  # the final update, Ctrl+C or an exception
            try:
                save_latest(last_done)
                print(f"[checkpoint] saved latest weights at update {last_done} -> {save_path}", flush=True)
            except Exception as e:  # never mask the original error
                print(f"[checkpoint] WARNING: final save failed: {e}", flush=True)
        if feed is not None:
            feed.finish()
        if evaluator is not None:
            evaluator.close()
        vec_env.close()


def build_parser() -> argparse.ArgumentParser:
    """CLI parser.  Every config-backed flag defaults to None: it overrides the config only when typed."""
    parser = argparse.ArgumentParser(description="Deep Visual PPO Training", allow_abbrev=False)
    parser.add_argument("--config", default="blueprint_v1", help="preset (long_v8 = long_v7 + battle-menu action mask, periodic from-start evaluation and the egg-errand route: Mr. Pokemon egg -> Elm -> Falkner, 11 milestones, 695-step phased path, 80 observation inputs; long_v7 = long_v6 + learned-vs-head-start metrics split by reset source and healing care: low-HP exposure cost, battle-survive and potion-use bonuses, heal-trip exemption from retreat/stagnation/dwell, stronger capped heal reward; long_v6 = long_v5 + route-checkpoint reverse-curriculum archive, dialogue-loop penalty, reward-scaler warm-up rollout; long_v5 = long_v4 + anti-collapse: bump/loop penalties off, no-effect A/B press penalty, path reward 0.3/tile, higher and slower-annealed entropy + entropy-boost safety net; long_v4 = long_v3 + dense best-ever path-progress reward, smaller decaying exploration budget, heal-for-HP reward, low-HP battle practice, aux prediction heads, gamma 0.999, lower entropy; long_v3 = long_v2 + dwell penalty with a level-target exemption + staggered episodes, battle practice, furthest-map resets; long_v2, blueprint_v1, long_v1, legacy are older) or a JSON config file")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--rollout", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="feed-forward policies only: samples per minibatch")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="PPO epochs per update")
    parser.add_argument("--minibatches", type=int, default=None, help="env-wise minibatches (<= --envs; uneven groups allowed). Typed values are never auto-reduced by the GPU-memory cap")
    parser.add_argument("--reward-scale", choices=("running_std", "none"), default=None,
                        help="divide rewards by the running std of the discounted return (blueprint_v1 default) or none (legacy)")
    parser.add_argument("--bn-mode", choices=("eval_refresh", "train"), default=None,
                        help="BatchNorm during the PPO update: eval mode + running-stat refresh (default) or train mode (legacy)")
    parser.add_argument("--vf-clip", action="store_const", const=True, default=None, dest="vf_clip", help="enable value-function clipping")
    parser.add_argument("--no-vf-clip", action="store_const", const=False, dest="vf_clip")
    parser.add_argument("--sticky-prob", type=float, default=None)
    parser.add_argument("--jitter-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None, help="periodic from-start evaluation interval in updates (only used when the config has periodic_eval)")
    parser.add_argument("--eval-episodes", type=int, default=None, help="episodes per from-start evaluation")
    parser.add_argument("--eval-max-steps", type=int, default=None, help="steps per from-start evaluation episode")
    parser.add_argument("--state", default=None, help="start state used with --no-curriculum (default route29_entry)")
    parser.add_argument("--states", default="", help="comma-separated list of start states")
    parser.add_argument("--curriculum", action="store_const", const=True, default=None, dest="curriculum",
                        help="use the config's start states / the built-in curriculum")
    parser.add_argument("--no-curriculum", action="store_const", const=False, dest="curriculum")
    parser.add_argument("--recurrent", action="store_const", const=True, default=None, dest="recurrent",
                        help="recurrent GRU temporal memory")
    parser.add_argument("--no-recurrent", action="store_const", const=False, dest="recurrent")
    parser.add_argument("--save", default=None, help="checkpoint path (default: config save_path)")
    parser.add_argument("--resume", action="store_true", default=False, help="warm-start from --save if it exists (and restore its optimizer sidecar)")
    parser.add_argument("--push-weights", action="store_true", default=False, help="commit and push updated weights to the current branch on completion (off by default)")
    parser.add_argument("--live", action="store_true", default=True, help="write live telemetry for tools.dashboard (default on)")
    parser.add_argument("--no-live", action="store_false", dest="live")
    parser.add_argument("--live-dir", default="logs/live")
    parser.add_argument("--runs-dir", default="logs/runs", help="where per-run replay history is written ('' to disable)")
    parser.add_argument("--run-id", default=None, help="name for this run's history folder (default: timestamp)")
    parser.add_argument("--frame-every", type=int, default=5, help="save a frame montage every N updates")
    parser.add_argument("--metrics-path", default="logs/training_metrics.jsonl", help="per-update metrics jsonl")
    return parser


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Config preset/file, then the CLI flags that were actually typed (None means not given)."""
    cfg = load_config(args.config)
    overrides = {
        "updates": args.updates,
        "rollout_steps": args.rollout,
        "max_steps": args.max_steps,
        "envs": args.envs,
        "lr": args.lr,
        "recurrent": args.recurrent,
        "save_path": args.save,
        "run_id": args.run_id,
        "sticky_prob": args.sticky_prob,
        "jitter_max": args.jitter_max,
        "seed": args.seed,
        "epochs": args.epochs,
        "num_minibatches": args.minibatches,
        "reward_scale": args.reward_scale,
        "bn_mode": args.bn_mode,
        "vf_clip": args.vf_clip,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "eval_max_steps": args.eval_max_steps,
    }
    typed ={k: v for k, v in overrides.items() if v is not None}
    if "save_path" not in typed and not getattr(args, "resume", False) and cfg.save_path == "logs/deep_policy.pt":
        # A fresh run must never reuse the shared default name: the cloud copy-back (or a second run) would overwrite
        # older weights. Give every run its own checkpoint, named after its run id.
        run = typed.get("run_id") or cfg.run_id or time.strftime("%Y%m%d-%H%M%S")
        typed["run_id"] = run
        typed["save_path"] = f"logs/{run}.pt"
        print(f"[config] checkpoint for this run: {typed['save_path']} (pass --save to choose, --resume to continue an old one)", flush=True)
    envs = typed.get("envs", cfg.envs)
    if "num_minibatches" not in typed:  # keep the preset's minibatch count valid for a different env count
        fit = fit_minibatches(envs, cfg.ppo.num_minibatches)
        if fit != cfg.ppo.num_minibatches:
            print(f"[config] {envs} envs: using {fit} minibatches instead of {cfg.ppo.num_minibatches} (cannot exceed envs)", flush=True)
            typed["num_minibatches"] = fit
    return apply_overrides(cfg, typed)


def main(argv=None):
    # The monitor prints π and status emoji; legacy Windows consoles (cp1252) would crash on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = build_config(args)
    except ConfigError as exc:
        parser.error(str(exc))

    states_list = [s.strip() for s in args.states.split(",") if s.strip()] if args.states else None
    states = resolve_start_states(cfg, args.state, states_list, args.curriculum)

    if args.resume:
        try:
            check_resume(cfg.save_path)
        except FileNotFoundError as exc:
            parser.error(str(exc))

    train(
        cfg,
        states=states,
        minibatches_typed=args.minibatches is not None,
        resume=args.resume,
        push_weights=args.push_weights,
        batch_size=args.batch_size,
        live=args.live,
        live_dir=args.live_dir,
        runs_dir=args.runs_dir or None,
        run_id=args.run_id,
        frame_every=args.frame_every,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
