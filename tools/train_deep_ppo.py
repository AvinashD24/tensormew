"""High-throughput visual PPO training using SpatialVisionPolicy on vectorized PyBoy workers.

Parallel PyBoy processes step the game and stream raw screen buffers (144x160x3) plus the RAM/route feature vector to
one policy process that does batched inference and the PPO update (``src.ppo_update``: env-wise minibatches, GAE with
truncation bootstrapping, entropy schedule).

Every knob lives in ``src.train_config.TrainConfig``.  ``--config`` picks a preset (``blueprint_v1``, ``legacy``) or a
JSON file; every other CLI flag defaults to None and overrides the config only when it is typed.

Training-only environment features (all off in ``--config legacy``):
- sticky actions: with probability ``sticky_prob`` the executed action repeats the previous executed move; PPO stores
  the action the policy sampled;
- start jitter: 0..``jitter_max`` random safe moves after each reset (no reward, nothing added to the seen sets);
- stage-balanced start states: each story stage is sampled evenly;
- swarm archive (optional, file based): the first env to reach a new stage or key event saves its emulator state under
  ``<logs>/swarm/<run-id>/``; on reset an env loads from that archive with probability ``SWARM_SAMPLE_PROB``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim

from src import route_graph
from src.battle import tactical_battle_step
from src.deep_policy import SpatialVisionPolicy
from src.emulator import Emulator
from src.live_feed import LiveFeed
from src.macros import Macros
from src.mover import walkable
from src.ppo_update import EntropySchedule, PPOHParams, RolloutBatch, compute_gae, ppo_update  # noqa: F401
from src.ram_parser import RamParser, dialogue_open
from src.rl_env import (
    ACTIONS,
    DEEP_OBS_DIM,
    KEY_EVENT_REWARDS,
    StepInfo,
    initial_seen_events,
    intrinsic_curiosity_reward,
    observe,
    progress_reward,
)
from src.states import StateRegistry
from src.step import TileStepper
from src.story_flags import REVERSE_EVENT_FLAGS
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
]

_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

SWARM_SAMPLE_PROB = 0.3  # chance that a reset loads a state from the swarm archive instead of a start state
SWARM_MAX_FILES = 64  # the archive never holds more states than this (they are ~100 KB each)
SWARM_PREFIX = "swarm:"
KEY_EVENT_IDS = frozenset(fid for fid, name in REVERSE_EVENT_FLAGS.items() if name in KEY_EVENT_REWARDS)
ROUTE_PROGRESS_WINDOW = 64  # recent episodes averaged for the entropy schedule's progress signal
SCREEN_SHAPE = (3, 144, 160)


# --------------------------------------------------------------------------------------------- pure helpers
def pick_executed_action(act_idx: int, last_move: int | None, sticky_prob: float, rng) -> int:
    """Sticky actions: with probability ``sticky_prob`` repeat the previous executed movement (index < 4).

    ``act_idx`` is what the policy sampled (and what PPO stores); the return value is what the game executes.
    """
    if sticky_prob > 0.0 and last_move is not None and rng.random() < sticky_prob:
        return int(last_move)
    return int(act_idx)


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

    def sample(self, rng) -> str | None:
        keys = self.keys()
        return keys[int(rng.integers(0, len(keys)))] if keys else None


class StartStateSelector:
    """Chooses the start state of each episode of one env.

    Without ``stages`` the states cycle (offset by the env index, so envs start on different states).  With
    ``stages`` (state name -> story stage) the stages are cycled instead, so every stage is sampled evenly, and the
    states inside a stage cycle too.  Resets (not the very first pick) load from ``archive`` with probability
    ``swarm_prob``; archived picks are returned as ``"swarm:<key>"``.
    """

    def __init__(self, start_states, env_index: int = 0, stages: dict[str, int] | None = None,
                 archive: SwarmArchive | None = None, swarm_prob: float = SWARM_SAMPLE_PROB, rng=None):
        self.states = list(start_states)
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
        if k > 0 and self.archive is not None and self.swarm_prob > 0 and self.rng.random() < self.swarm_prob:
            key = self.archive.sample(self.rng)
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


def _route_covered(state) -> bool:
    try:
        return bool(route_graph.covers_map((state.map_group, state.map_number)))
    except Exception:
        return False


def _route_phi(state) -> float | None:
    """Route potential of ``state``, or None off the route graph."""
    if not _route_covered(state):
        return None
    try:
        return float(route_graph.route_potential(state))
    except Exception:
        return None


# --------------------------------------------------------------------------------------------- environment
class EnvWorker:
    """One headless PyBoy instance with the training reward, sticky actions, start jitter and swarm sharing.

    ``cfg`` keys (all optional): ``sticky_prob``, ``jitter_max``, ``stage_balanced``, ``swarm_dir``, ``swarm_prob``,
    ``seed``.  Call ``reset()`` before the first ``step()``.
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
        self.selector = StartStateSelector(
            names, self.index, stages, self.archive, float(cfg.get("swarm_prob", SWARM_SAMPLE_PROB)), self.rng
        )

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
        self.start_phi: float | None = None
        self._event_snapshot: set = set()

    # -- setup helpers
    def _state_bytes(self, name: str) -> bytes:
        if name not in self._bytes:
            self._bytes[name] = self.registry.load_bytes(name)
        return self._bytes[name]

    def _stage_of_states(self, names: list[str]) -> dict[str, int]:
        stages = {}
        for name in names:
            self.emu.load_state(self._state_bytes(name))
            stages[name] = int(route_graph.current_stage(self.reader.parse(self.emu.frame)))
        return stages

    def _equip_balls(self) -> None:
        try:
            b1, a1 = self.emu.symbol("wNumBalls")
            b2, a2 = self.emu.symbol("wBalls")
            if self.emu.read(b1, a1, 1)[0] == 0:
                self.emu.write(b1, a1, bytes([1]))
                self.emu.write(b2, a2, bytes([5, 10, 255]))
        except Exception:
            pass

    def _load(self, name: str) -> bytes:
        """Load a start state (or an archived ``swarm:<key>`` state) into the emulator; returns its bytes."""
        data = None
        if name.startswith(SWARM_PREFIX) and self.archive is not None:
            data = self.archive.load(name[len(SWARM_PREFIX):])
        if data is None:
            name = name if not name.startswith(SWARM_PREFIX) else self.selector.states[0]
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
        ram_vec = observe(st, self.visited, include_goals=True, include_route=True)
        return self._screen(), ram_vec, st

    def _peek_obs(self, st):
        """Observation of ``st`` without touching the visit bookkeeping (the final obs of a truncated episode)."""
        return self._screen(), observe(st, self.visited, include_goals=True, include_route=True)

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

    def reset(self):
        """Start a new episode and return its first ``(screen, ram_vec)``."""
        data = self._load(self.selector.pick())
        for s in (self.visited, self.seen_tiles, self.seen_maps, self.seen_events, self.seen_waypoints):
            s.clear()
        self.tile_visit_counts.clear()
        self.caught_counts.clear()
        self.blocked_count = self.consecutive_bumps = self.steps = 0
        self.last_dir = self.last_move = None
        self._jitter(data)
        st = self.reader.parse(self.emu.frame)
        # Flags already set in the start state are not progress: seed them so they pay nothing (was ~+300/episode).
        self.seen_events.update(initial_seen_events(st))
        self._event_snapshot = set(self.seen_events)
        self.start_phi = _route_phi(st)
        screen, ram_vec, _ = self.get_obs()
        return screen, ram_vec

    def _maybe_archive(self, after) -> None:
        """Save the emulator state when this env is first to reach a new stage or key event (swarm sharing)."""
        if self.archive is None or len(self.seen_events) == len(self._event_snapshot):
            return
        new = self.seen_events - self._event_snapshot
        self._event_snapshot = set(self.seen_events)
        keys = [f"event{fid}" for fid in sorted(new) if fid in KEY_EVENT_IDS]
        stage = int(route_graph.current_stage(after))
        if stage > 0:
            keys.append(f"stage{stage}")
        if keys:
            data = self.emu.save_state()
            for key in keys:
                self.archive.save(key, data)

    # -- stepping
    def step(self, act_idx: int):
        """Execute one policy action; returns the 7-tuple the trainer expects (see ``VectorEnv.step``)."""
        exec_idx = pick_executed_action(int(act_idx), self.last_move, self.sticky_prob, self.rng)
        if exec_idx < 4:
            self.last_move = exec_idx
        action_name = ACTIONS[exec_idx]
        emu, stepper, macros, reader = self.emu, self.stepper, self.macros, self.reader
        before = reader.parse(emu.frame)
        moved = False

        # Handle in-game events
        if before.battle_mode != 0:
            tactical_battle_step(before, macros)
            self.blocked_count = 0
        elif dialogue_open(before):
            macros.mash_text(max_frames=80)
            self.blocked_count = 0
        elif action_name == "a":
            emu.press("a", frames=8)
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

        if not moved and before.battle_mode == 0 and not dialogue_open(before):
            self.consecutive_bumps += 1
        else:
            self.consecutive_bumps = 0

        after = reader.parse(emu.frame)
        unvisited_neighbor = False
        if 0 <= exec_idx < 4:
            dx, dy = _DELTA[action_name]
            neighbor = (before.map_group, before.map_number, before.x + dx, before.y + dy)
            unvisited_neighbor = neighbor not in self.seen_tiles

        comps: dict[str, float] = {}
        step_rew, _, _ = progress_reward(
            before, after, moved, self.seen_tiles, self.seen_maps,
            unvisited_neighbor=unvisited_neighbor, caught_counts=self.caught_counts, seen_events=self.seen_events,
            seen_waypoints=self.seen_waypoints, info=StepInfo(consecutive_bumps=self.consecutive_bumps),
            components=comps,
        )
        # Curiosity and the momentum bonus are the fallback for areas the route graph does not cover.
        covered = _route_covered(after)
        v_count = self.tile_visit_counts.get((after.map_group, after.map_number, after.x, after.y), 0)
        curiosity = 0.0 if covered else intrinsic_curiosity_reward(v_count, scale=0.08)
        loop = -0.02 * min(4.0, (v_count - 10) / 5.0) if v_count > 12 else 0.0  # anti-looping pacing penalty
        momentum = 0.02 if (not covered and action_name == self.last_dir and moved) else 0.0
        self.last_dir = action_name if moved else None

        has_event = getattr(after, "has_event", None)
        gave_egg = callable(has_event) and has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
        quest_complete = bool(gave_egg and after.map_id == (24, 5))
        quest = 50.0 if quest_complete else 0.0

        comps["curiosity"], comps["loop"], comps["momentum"], comps["quest"] = curiosity, loop, momentum, quest
        step_rew += curiosity + loop + momentum + quest

        self.steps += 1
        truncated = self.steps >= self.max_steps and not quest_complete  # time limit, not a real end
        done = truncated or quest_complete
        ended_maps = len(self.seen_maps)
        ended_state = self.state_name
        phi_end = _route_phi(after)
        phi_start = self.start_phi
        stage_end = int(route_graph.current_stage(after))
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
            "route_phi": phi_end,  # route potential of the state this step's reward was computed on (None off-graph)
            "ep_start_phi": phi_start,  # route potential at the start of the episode this step belongs to
            "stage": stage_end,
            "truncated": bool(truncated),
            "executed_action": exec_idx,
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
    """Child process running one EnvWorker behind a pipe (RESET / STEP / CLOSE)."""
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
    finally:
        if worker is not None:
            worker.close()


class VectorEnv:
    """Manages parallel PyBoy worker processes with multi-state curriculum distribution.

    ``env_cfg`` is forwarded to every ``EnvWorker`` (sticky/jitter/stage-balanced/swarm/seed).  After ``step`` the
    per-env telemetry is in ``last_infos``, the time-limit truncation flags in ``last_truncated`` and the final
    observations of truncated envs (before their reset) in ``last_final`` as ``{env: (screen, ram)}``.
    """

    def __init__(self, num_envs: int, start_states: list[str] | str, max_steps: int, env_cfg: dict | None = None):
        self.num_envs = num_envs
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
        for i in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_worker_fn,
                args=(child_conn, i, states_list, max_steps, env_cfg),
                daemon=True,
            )
            p.start()
            self.pipes.append(parent_conn)
            self.procs.append(p)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        for pipe in self.pipes:
            pipe.send(("RESET", None))
        screens, rams = [], []
        for pipe in self.pipes:
            s, r = pipe.recv()
            screens.append(s)
            rams.append(r)
        return np.array(screens, dtype=np.uint8), np.array(rams, dtype=np.float64)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        for pipe, act in zip(self.pipes, actions):
            pipe.send(("STEP", act))
        screens, rams, rews, dones, maps, badges = [], [], [], [], [], []
        self.last_final = {}
        self.last_truncated = np.zeros(self.num_envs, dtype=bool)
        for i, pipe in enumerate(self.pipes):
            s, r, rew, d, m, b, info = pipe.recv()
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
        for pipe in self.pipes:
            try:
                pipe.send(("CLOSE", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2.0)


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
    """Largest divisor of ``envs`` that is <= num_minibatches (minibatches are whole environments)."""
    for k in range(min(num_minibatches, envs), 0, -1):
        if envs % k == 0:
            return k
    return 1


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


def save_checkpoint(policy, optimizer, save_path: str | Path, update: int, route_progress: float = 0.0) -> None:
    policy.save(save_path)
    torch.save(
        {"optimizer": optimizer.state_dict(), "update": int(update), "route_progress": float(route_progress)},
        str(checkpoint_sidecar(save_path)),
    )


def load_optimizer_state(optimizer, save_path: str | Path, device) -> bool:
    """Restore Adam state from the sidecar; returns False (with a clear message) when it cannot be restored."""
    side = checkpoint_sidecar(save_path)
    if not side.exists():
        print(f"[resume] optimizer sidecar {side} not found: weights loaded, Adam moments restart from zero.", flush=True)
        return False
    try:
        blob = torch.load(str(side), map_location=device)
        optimizer.load_state_dict(blob["optimizer"])
        print(f"[resume] restored optimizer state from {side} (saved at update {blob.get('update', '?')}).", flush=True)
        return True
    except Exception as e:
        print(f"[resume] could not restore optimizer state from {side} ({e}); Adam moments restart from zero.", flush=True)
        return False


def worker_env_cfg(cfg: TrainConfig, swarm_dir: str | Path | None) -> dict:
    return {
        "sticky_prob": cfg.sticky_prob,
        "jitter_max": cfg.jitter_max,
        "stage_balanced": cfg.stage_balanced,
        "swarm_dir": str(swarm_dir) if (cfg.swarm and swarm_dir) else None,
        "swarm_prob": SWARM_SAMPLE_PROB,
        "seed": cfg.seed,
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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gru = cfg.recurrent
    num_envs, rollout_steps, updates = cfg.envs, cfg.rollout_steps, cfg.updates
    save_path = cfg.save_path
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Initializing SpatialVisionPolicy training on {device} (recurrent={use_gru}, config={cfg.name})...")

    policy = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=5, use_gru=use_gru, gru_hidden_dim=512)
    resumed = False
    if resume and os.path.exists(save_path):
        try:
            policy = SpatialVisionPolicy.load(save_path, device=device, override_use_gru=use_gru, override_obs_dim=DEEP_OBS_DIM)
            resumed = True
            print(f"Loaded existing checkpoint from {save_path} (use_gru={policy.use_gru}, obs_dim={policy.obs_dim})")
        except Exception as e:
            print(f"Warning: could not load checkpoint ({e}), initializing fresh policy")

    policy.to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=1e-4)
    if resumed:
        load_optimizer_state(optimizer, save_path, device)
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr
    print(f"Policy ready with {policy.param_count:,} trainable parameters.")

    hp = PPOHParams(**dataclasses.asdict(cfg.ppo))
    if not policy.use_gru and batch_size:  # feed-forward minibatches are samples, sized by --batch-size
        hp.num_minibatches = max(1, math.ceil(num_envs * rollout_steps / int(batch_size)))
    entropy_schedule = EntropySchedule(cfg.entropy_start, cfg.entropy_end)
    progress = RouteProgressTracker()

    run_name = run_id or cfg.run_id or time.strftime("%Y%m%d-%H%M%S")
    history_dir = Path(runs_dir) / run_name if runs_dir else None
    if history_dir is not None:
        cfg_path = dump_effective_config(cfg, history_dir)
        print(f"Effective config -> {cfg_path}", flush=True)
    swarm_dir = (Path(runs_dir).parent if runs_dir else Path("logs")) / "swarm" / run_name if cfg.swarm else None

    print(
        f"Spawning {num_envs} vectorized PyBoy workers across states: {states} "
        f"(max_steps={cfg.max_steps}, sticky={cfg.sticky_prob}, jitter<={cfg.jitter_max}, "
        f"stage_balanced={cfg.stage_balanced}, swarm={'on -> ' + str(swarm_dir) if swarm_dir else 'off'})..."
    )
    vec_env = VectorEnv(num_envs=num_envs, start_states=states, max_steps=cfg.max_steps, env_cfg=worker_env_cfg(cfg, swarm_dir))
    screens_cur, rams_cur = vec_env.reset()

    best_reward = -float("inf")
    hidden = policy.gru_hidden_dim if use_gru else 512
    hx = torch.zeros(1, num_envs, hidden, device=device) if policy.use_gru else None
    monitor = TerminalMonitor(total_updates=updates, jsonl_path=metrics_path)
    feed = None
    if live:
        feed = LiveFeed(
            live_dir,
            num_envs=num_envs,
            action_names=ACTIONS,
            history_dir=history_dir,
            frame_every=frame_every,
            run_info={
                "total_updates": updates,
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

    try:
        for update in range(updates):
            t0 = time.time()
            buf_values.zero_(), buf_rewards.zero_(), buf_dones.zero_(), buf_trunc.zero_(), buf_boot.zero_()
            initial_hx = hx.detach().clone() if (policy.use_gru and hx is not None) else None
            entropy_coef = entropy_schedule.value(progress.value)

            # 1. Collect rollouts with batched inference
            policy.eval()
            for step in range(T):
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
                screens_next, rams_next, rews, dones, map_counts, badges = vec_env.step(actions_np)
                infos = vec_env.last_infos
                if feed is not None:
                    feed.observe(infos=infos, actions=actions_np, rewards=rews, dones=dones, screens=screens_next)

                # Route progress per finished episode drives the entropy schedule.
                for i in np.nonzero(dones)[0]:
                    inf = infos[i] or {}
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

                if policy.use_gru and next_hx is not None:
                    if dones.any():
                        done_mask = torch.from_numpy(dones).to(device)
                        next_hx[:, done_mask, :] = 0.0
                    hx = next_hx

                buf_screens[:, step] = torch.from_numpy(screens_cur).permute(0, 3, 1, 2)
                buf_rams[:, step] = rams_t.cpu()
                buf_actions[:, step] = actions_t.cpu()
                buf_logps[:, step] = logp_t.cpu()
                buf_rewards[step] = torch.from_numpy(rews)
                buf_dones[step] = torch.from_numpy(dones)
                buf_values[step] = val.cpu()

                screens_cur, rams_cur = screens_next, rams_next

            # 2. GAE with truncation bootstrapping and the value of the current observations
            last_value = policy_values(policy, screens_cur, rams_cur, hx, device)
            returns, advantages = compute_gae(
                buf_rewards, buf_values, buf_dones, last_value, hp.gamma, hp.gae_lambda,
                truncated=buf_trunc, bootstrap_values=buf_boot,
            )
            mean_rew = float(buf_rewards.sum(dim=0).mean().item())
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
            stats = ppo_update(policy, optimizer, batch, hp, entropy_coef=entropy_coef)
            if hx is not None:
                hx = hx.detach()

            dt = time.time() - t0
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
                "grad_norm": _finite_or_none(stats["grad_norm"]),
                "entropy_coef": round(entropy_coef, 5),
                "route_progress": round(progress.value, 4),
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

            if mean_rew > best_reward or (update + 1) % 10 == 0:
                best_reward = max(best_reward, mean_rew)
                save_checkpoint(policy, optimizer, save_path, update, progress.value)

        print(f"\nTraining completed! Saved best model to {save_path}", flush=True)
        if push_weights:
            try:
                from tools.sync_weights import sync_and_push
                sync_and_push(push=True, message=f"feat(weights): deep_policy.pt retrained (updates={updates}, rew={best_reward:.2f})")
            except Exception as e:
                print(f"[Warning] Could not auto-push weights: {e}", flush=True)
    finally:
        if feed is not None:
            feed.finish()
        vec_env.close()


def build_parser() -> argparse.ArgumentParser:
    """CLI parser.  Every config-backed flag defaults to None: it overrides the config only when typed."""
    parser = argparse.ArgumentParser(description="Deep Visual PPO Training")
    parser.add_argument("--config", default="blueprint_v1", help="preset (blueprint_v1, legacy) or a JSON config file")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--rollout", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="feed-forward policies only: samples per minibatch")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="PPO epochs per update")
    parser.add_argument("--minibatches", type=int, default=None, help="env-wise minibatches (must divide --envs)")
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
            print(f"[config] {envs} envs: using {fit} minibatches instead of {cfg.ppo.num_minibatches} (must divide envs)", flush=True)
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

    train(
        cfg,
        states=states,
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
