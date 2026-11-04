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
  (applied inside the policy, so PPO log-probs stay valid).
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

from src.battle_obs import equip_supplies
from src.deep_policy import CheckpointIncompatibleError, SpatialVisionPolicy
from src.dwell import DwellTracker
from src.path_progress import PathProgress
from src.emulator import Emulator
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
    DEEP_OBS_DIM,
    N_ACTIONS,
    NEW_INTERACTION_REWARD,
    POKECENTER_MAPS,
    budget_pay,
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
from src.story_path import KEY_EVENT_REWARDS, MILESTONE_COUNT, current_stage, progress_potential
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
KEY_EVENT_IDS = frozenset(fid for fid, name in REVERSE_EVENT_FLAGS.items() if name in KEY_EVENT_REWARDS)
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
        return MILESTONE_COUNT
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


def battle_archive_key(state) -> str:
    """Diversity key of a battle: kind (1 wild / 2 trainer), map and enemy species."""
    return (f"m{int(getattr(state, 'battle_mode', 0))}_{int(state.map_group)}-{int(state.map_number)}"
            f"_s{int(getattr(state, 'enemy_mon_species', 0))}")


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

    def __init__(self, directory: str | Path, max_files: int = SWARM_MAX_FILES):
        self.dir = Path(directory)
        self.max_files = int(max_files)
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
        try:
            files = sorted(self.dir.glob("*.state"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for p in files[: max(0, len(files) - self.max_files)]:
            try:
                p.unlink()
            except OSError:
                pass

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
                 furthest_floor: float | None = None):
        self.states = list(start_states)
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
        if k > 0 and (swarm_on or self.battle_prob > 0):
            # One draw: [0, battle_prob) = battle archive, next swarm_prob = swarm archive, the rest = start states.
            u = float(self.rng.random())
            if u < self.battle_prob:
                key = self.battle_archive.sample(self.rng)
                if key is not None:
                    return BATTLE_PREFIX + key
            elif swarm_on and u < self.battle_prob + self.swarm_prob:
                key = self.archive.sample(self.rng, self.furthest_floor)
                if key is not None:
                    return SWARM_PREFIX + key
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
        self.battle_archive = (SwarmArchive(battle_dir, int(cfg.get("battle_max", 32))) if battle_dir else None)
        self.selector = StartStateSelector(
            names, self.index, stages, self.archive, float(cfg.get("swarm_prob", SWARM_SAMPLE_PROB)), self.rng,
            battle_archive=self.battle_archive, battle_prob=float(cfg.get("battle_prob", 0.0)),
            furthest_floor=cfg.get("swarm_furthest_floor"),
        )
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
        # Dwell penalty (long_v4, OFF when cfg has no "dwell" params): see src.dwell.
        self.dwell = DwellTracker.from_params(cfg.get("dwell"))
        self.dwell_exempt_loop = bool(cfg.get("dwell_exempt_loop", True))
        # Dense best-ever path-progress reward (long_v4, OFF when cfg has no "path_progress" scale): src.path_progress.
        self.path_progress = PathProgress.from_scale(cfg.get("path_progress"))
        self._last_stage = 0  # stage already archived / seen at the start of the episode
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
        if name.startswith(SWARM_PREFIX) and self.archive is not None:
            data = self.archive.load(name[len(SWARM_PREFIX):])
        elif name.startswith(BATTLE_PREFIX) and self.battle_archive is not None:
            data = self.battle_archive.load(name[len(BATTLE_PREFIX):])
        if data is None:
            name = name if not name.startswith((SWARM_PREFIX, BATTLE_PREFIX)) else self.selector.states[0]
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
        self.budget.clear()
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
        if self.dwell is not None:
            self.dwell.reset(st)
        if self.path_progress is not None:
            self.path_progress.reset(st)   # the start state's own progress (archive / jitter) is never paid
        # Flags already set in the start state are not progress: seed them so they pay nothing (was ~+300/episode).
        self.seen_events.update(initial_seen_events(st))
        self._event_snapshot = set(self.seen_events)
        screen, ram_vec, _ = self.get_obs()  # also records the start map as seen (it must not pay a milestone)
        self.start_phi = _story_phi(st, self.seen_maps)
        return screen, ram_vec

    def _maybe_archive(self, after) -> None:
        """Save the emulator state when this env is first to reach a new stage or key event (swarm sharing)."""
        """Snapshots are taken at stage ENTRY (a few rotating slots per stage) and at the first key-event flip, never
        mid-cutscene (battle or dialogue open): those are deferred until the screen is calm."""
        if self.archive is None:
            return
        new = self.seen_events - self._event_snapshot
        self._event_snapshot = set(self.seen_events)
        self._pending_events |= {fid for fid in new if fid in KEY_EVENT_IDS}
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
        key = battle_archive_key(after)
        if self.battle_archive.has(key):
            return
        if self.battle_archive.save_fifo(key, self.emu.save_state()):
            self._last_battle_save = self._total_steps

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
        # Count-based novelty (1/sqrt(visits)) and a small straight-line momentum bonus: generic exploration terms.
        v_count = self.tile_visit_counts.get((after.map_group, after.map_number, after.x, after.y), 0)
        curiosity = intrinsic_curiosity_reward(v_count, scale=0.08)
        curiosity = budget_pay(self.budget, "explore", curiosity) if self._budget_ok else curiosity
        in_play = not busy and after.battle_mode == 0 and not dialogue_open(after)  # no loop penalty in fights/text
        loop = -0.02 * min(4.0, (v_count - 10) / 5.0) if (v_count > 12 and in_play) else 0.0  # anti-looping pacing
        path_rew = 0.0
        if self.path_progress is not None:
            path_rew = self.path_progress.update(before, after)   # >= 0: only a new best progress tile pays
        dwell_pen = 0.0
        if self.dwell is not None:
            # progress = story milestone / event, a species caught for the first time this episode, or a new best path tile
            progress = bool(pr_info.story_progress or len(self.caught_counts) > species_before or path_rew > 0.0)
            dwell_pen, level_exempt = self.dwell.update(before, after, new_tile=new_tile, new_map=new_map, progress=progress)
            if level_exempt and self.dwell_exempt_loop:
                loop = 0.0  # needed leveling is not "looping": the per-tile revisit cost is off while grinding below target
        momentum = 0.02 if (action_name == self.last_dir and moved) else 0.0
        momentum = budget_pay(self.budget, "explore", momentum) if self._budget_ok else momentum
        self.last_dir = action_name if moved else None

        comps["curiosity"], comps["loop"], comps["momentum"] = curiosity, loop, momentum
        if self.dwell is not None:
            comps["dwell"] = dwell_pen
        if self.path_progress is not None:
            comps["path"] = path_rew
        step_rew += curiosity + loop + momentum + dwell_pen + path_rew

        # No termination on milestones or on the badge; the episode ends only on the step limit (a time limit, not a goal).
        self.steps += 1
        truncated = self.steps >= self.ep_len  # time limit, not a real end
        done = truncated
        ended_maps = len(self.seen_maps)
        ended_state = self.state_name
        phi_end = _story_phi(after, self.seen_maps)
        phi_start = self.start_phi
        stage_end = int(current_stage(after))
        ended_partial = self._partial
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
            "partial_episode": bool(ended_partial and done),  # a staggered first episode that was cut short
            "ep_steps": int(self.steps),  # steps taken in the running episode (0 right after a reset)
            "ep_len": int(self.ep_len),
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


def worker_env_cfg(cfg: TrainConfig, swarm_dir: str | Path | None, battle_dir: str | Path | None = None) -> dict:
    return {
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
    num_envs, rollout_steps, updates = cfg.envs, cfg.rollout_steps, cfg.updates
    save_path = cfg.save_path
    if resume:
        check_resume(save_path)  # raises before any worker is spawned
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Initializing SpatialVisionPolicy training on {device} (recurrent={use_gru}, config={cfg.name})...")

    policy = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=N_ACTIONS, use_gru=use_gru, gru_hidden_dim=512)
    resumed = False
    sidecar = None
    if resume:
        try:
            policy = SpatialVisionPolicy.load(save_path, device=device, override_use_gru=use_gru,
                                              override_obs_dim=DEEP_OBS_DIM, check_schema=True)
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
    if not policy.use_gru and batch_size:  # feed-forward minibatches are samples, sized by --batch-size
        hp.num_minibatches = max(1, math.ceil(num_envs * rollout_steps / int(batch_size)))
    explicit_mb = minibatches_typed or bool(batch_size and not policy.use_gru)
    vram = torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None
    units, per_unit = (num_envs, rollout_steps) if policy.use_gru else (num_envs * rollout_steps, 1)
    hp.num_minibatches, mb_msg = plan_minibatches(units, per_unit, hp.num_minibatches, vram, explicit_mb)
    print(f"[memory] {mb_msg}", flush=True)
    entropy_schedule = EntropySchedule(cfg.entropy_start, cfg.entropy_end)
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
    if cfg.path_progress:
        print(f"[path progress] +{cfg.path_progress_scale} per tile of new best progress along the story path "
              f"(best-ever per episode, never negative, start progress free; ~{cfg.path_progress_scale * 319:.1f} for the whole path)",
              flush=True)
    vec_env = VectorEnv(num_envs=num_envs, start_states=states, max_steps=cfg.max_steps,
                        env_cfg=worker_env_cfg(cfg, swarm_dir, battle_dir))
    screens_cur, rams_cur = vec_env.reset()

    hidden = policy.gru_hidden_dim if use_gru else 512
    hx = torch.zeros(1, num_envs, hidden, device=device) if policy.use_gru else None
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
    buf_rams = torch.empty((B, T, DEEP_OBS_DIM), dtype=torch.float32)
    buf_actions = torch.empty((B, T), dtype=torch.long)
    buf_logps = torch.empty((B, T), dtype=torch.float32)
    buf_values = torch.zeros((T, B), dtype=torch.float32)
    buf_rewards = torch.zeros((T, B), dtype=torch.float32)
    buf_dones = torch.zeros((T, B), dtype=torch.bool)
    buf_trunc = torch.zeros((T, B), dtype=torch.bool)
    buf_boot = torch.zeros((T, B), dtype=torch.float32)
    buf_raw = torch.zeros((T, B), dtype=torch.float32)  # unscaled rewards, for logging

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
            entropy_coef = entropy_schedule.value(progress.value)
            # Early A-bias in battle menus: set before the rollout, constant until the PPO update below has used it, so
            # the rollout's log-probs (old_logps) and the update's evaluate_actions describe the same distribution.
            a_bias = battle_a_bias_value(update, cfg.battle_a_bias, cfg.battle_a_bias_updates)
            policy.set_battle_a_bias(a_bias)
            if cfg.lr_end_frac < 1.0:  # linear lr decay over this invocation's updates
                lr_now = cfg.lr * (1.0 - (1.0 - cfg.lr_end_frac) * min(1.0, (update - start_update) / max(1, updates)))
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

            # 1. Collect rollouts with batched inference
            policy.eval()
            phase = {"infer": 0.0, "env": 0.0, "feed": 0.0, "trunc": 0.0, "buf": 0.0}
            _pc = time.perf_counter
            for step in range(T):
                _t = _pc()
                screens_t = (
                    torch.from_numpy(screens_cur).to(device).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
                )
                rams_t = torch.from_numpy(rams_cur).float().to(device)

                with torch.no_grad():
                    if policy.use_gru:
                        logits, val, next_hx = policy.forward_recurrent(screens_t, rams_t, hx)
                    else:
                        logits, val = policy(screens_t, rams_t)
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

                # Route progress per finished episode drives the entropy schedule.
                for i in np.nonzero(dones)[0]:
                    inf = infos[i] or {}
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
            max_maps = max(map_counts)
            max_badges = max(badges)
            cur_lr = optimizer.param_groups[0]["lr"]
            mean_entropy = float(stats["entropy"])
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
            }

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
        vec_env.close()


def build_parser() -> argparse.ArgumentParser:
    """CLI parser.  Every config-backed flag defaults to None: it overrides the config only when typed."""
    parser = argparse.ArgumentParser(description="Deep Visual PPO Training", allow_abbrev=False)
    parser.add_argument("--config", default="blueprint_v1", help="preset (long_v4 = long_v3 + dense best-ever path-progress reward; long_v3 = long_v2 + dwell penalty with a level-target exemption + staggered episodes, battle practice, furthest-map resets; long_v2, blueprint_v1, long_v1, legacy are older) or a JSON config file")
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
    }
    typed = {k: v for k, v in overrides.items() if v is not None}
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
