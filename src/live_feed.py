"""Live telemetry feed for the training dashboard.

The trainer calls :meth:`LiveFeed.observe` once per vectorized env step. The feed keeps
per-map visitation heatmaps, short position trails, action counts and an episode log in
memory, and periodically snapshots them to ``<out_dir>/state.json`` plus one PNG per
env (``env_<i>.png``). ``tools/dashboard.py`` serves those files to the browser.

All disk writes are throttled and atomic, so a slow or absent dashboard never blocks
training and a reader never sees a half-written file.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # Pillow is a project dependency; degrade to "no frames" if it is missing.
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    from src.game_data import get_map_name
except Exception:  # pragma: no cover

    def get_map_name(map_group: int, map_number: int) -> str:
        return f"Map ({map_group}, {map_number})"


TRAIL_LEN = 48
MAX_HEAT_MAPS = 12
MAX_CELLS_PER_MAP = 4000
MAX_EPISODES = 60
MAX_EVENTS = 80


def _map_key(map_id: Sequence[int]) -> str:
    return f"{int(map_id[0])},{int(map_id[1])}"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: reader momentarily holds the file
            time.sleep(0.01 * (attempt + 1))


class LiveFeed:
    """Aggregates rollout telemetry and snapshots it for the browser dashboard."""

    def __init__(
        self,
        out_dir: str | Path = "logs/live",
        *,
        num_envs: int,
        action_names: Sequence[str],
        frame_interval: float = 0.6,
        state_interval: float = 0.8,
        run_info: dict[str, Any] | None = None,
        clear: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = num_envs
        self.action_names = list(action_names)
        self.frame_interval = frame_interval
        self.state_interval = state_interval
        self.run_info = dict(run_info or {})

        if clear:
            for stale in self.out_dir.glob("env_*.png"):
                try:
                    stale.unlink()
                except OSError:
                    pass

        self.started = time.time()
        self.steps = 0
        self.total_samples = 0
        self.update = 0
        self.finished = False

        self._heat: dict[str, dict[tuple[int, int], int]] = {}
        self._map_visits: dict[str, int] = {}
        self._trails = [deque(maxlen=TRAIL_LEN) for _ in range(num_envs)]
        self._envs: list[dict[str, Any]] = [self._blank_env(i) for i in range(num_envs)]
        self._ep_return = np.zeros(num_envs, dtype=np.float64)
        self._ep_steps = np.zeros(num_envs, dtype=np.int64)
        self._action_total = np.zeros(len(self.action_names), dtype=np.int64)
        self._action_recent = np.zeros(len(self.action_names), dtype=np.int64)
        self._episodes: deque[dict[str, Any]] = deque(maxlen=MAX_EPISODES)
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self._best_maps = 0
        self._best_badges = 0
        self._best_ep_return = float("-inf")
        self._seen_map_keys: set[str] = set()

        self._last_frame_t = 0.0
        self._last_state_t = 0.0
        self._rate_t = self.started
        self._rate_steps = 0
        self._steps_per_sec = 0.0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _blank_env(i: int) -> dict[str, Any]:
        return {
            "id": i,
            "map": None,
            "map_name": "",
            "x": 0,
            "y": 0,
            "state": "",
            "battle": False,
            "badges": 0,
            "ep_return": 0.0,
            "ep_steps": 0,
            "last_action": "",
            "last_reward": 0.0,
        }

    def _event(self, kind: str, text: str, **extra: Any) -> None:
        self._events.append({"t": round(time.time(), 2), "kind": kind, "text": text, **extra})

    # --------------------------------------------------------------- ingestion
    def observe(
        self,
        *,
        infos: Sequence[dict[str, Any]] | None,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        screens: np.ndarray | None = None,
    ) -> None:
        """Record one vectorized env step. ``screens`` is only needed when a frame is due."""
        self.steps += 1
        now = time.time()

        for i in range(self.num_envs):
            act = int(actions[i])
            if 0 <= act < len(self.action_names):
                self._action_total[act] += 1
                self._action_recent[act] += 1
            rew = float(rewards[i])
            self._ep_return[i] += rew
            self._ep_steps[i] += 1

            env = self._envs[i]
            env["last_action"] = self.action_names[act] if 0 <= act < len(self.action_names) else ""
            env["last_reward"] = round(rew, 3)

            info = infos[i] if infos is not None and i < len(infos) else None
            if info:
                self._ingest_info(i, info)

            if bool(dones[i]):
                self._finish_episode(i, info or {})

            env["ep_return"] = round(float(self._ep_return[i]), 2)
            env["ep_steps"] = int(self._ep_steps[i])

        if now - self._last_frame_t >= self.frame_interval and screens is not None:
            self._write_frames(screens)
            self._last_frame_t = now

        if now - self._last_state_t >= self.state_interval:
            dt = now - self._rate_t
            if dt > 0:
                self._steps_per_sec = (self.steps - self._rate_steps) * self.num_envs / dt
            self._rate_t, self._rate_steps = now, self.steps
            self.flush()

    def _ingest_info(self, i: int, info: dict[str, Any]) -> None:
        env = self._envs[i]
        map_id = info.get("map")
        if map_id is not None:
            key = _map_key(map_id)
            x, y = int(info.get("x", 0)), int(info.get("y", 0))
            env.update(
                map=key,
                map_name=get_map_name(int(map_id[0]), int(map_id[1])),
                x=x,
                y=y,
                state=str(info.get("state", env["state"])),
                battle=bool(info.get("battle", False)),
                badges=int(info.get("badges", 0)),
            )
            heat = self._heat.setdefault(key, {})
            heat[(x, y)] = heat.get((x, y), 0) + 1
            self._map_visits[key] = self._map_visits.get(key, 0) + 1
            self._trails[i].append((key, x, y))
            if key not in self._seen_map_keys:
                self._seen_map_keys.add(key)
                self._event("map", f"env {i} entered {env['map_name']}", env=i, map=key)
        badges = int(info.get("badges", 0))
        if badges > self._best_badges:
            self._best_badges = badges
            self._event("badge", f"env {i} earned badge #{badges}", env=i)

    def _finish_episode(self, i: int, info: dict[str, Any]) -> None:
        ret = float(self._ep_return[i])
        maps = int(info.get("ended_maps", 0))
        episode = {
            "t": round(time.time(), 2),
            "env": i,
            "return": round(ret, 2),
            "steps": int(self._ep_steps[i]),
            "maps": maps,
            "state": str(info.get("ended_state", self._envs[i]["state"])),
        }
        self._episodes.append(episode)
        if maps > self._best_maps:
            self._best_maps = maps
            self._event("record", f"new best map coverage: {maps} maps (env {i})", env=i)
        if ret > self._best_ep_return:
            if self._best_ep_return != float("-inf"):
                self._event("record", f"new best episode return {ret:+.1f} (env {i})", env=i)
            self._best_ep_return = ret
        self._ep_return[i] = 0.0
        self._ep_steps[i] = 0
        self._trails[i].clear()

    def on_update(self, update: int, samples: int) -> None:
        """Called once per PPO update; resets the 'recent' action window."""
        self.update = update
        self.total_samples += samples
        self._action_recent[:] = 0
        self.flush()

    def finish(self) -> None:
        self.finished = True
        self.flush()

    # ---------------------------------------------------------------- snapshot
    def _write_frames(self, screens: np.ndarray) -> None:
        if Image is None:
            return
        for i in range(min(self.num_envs, len(screens))):
            try:
                path = self.out_dir / f"env_{i}.png"
                tmp = path.with_suffix(".tmp.png")
                Image.fromarray(np.ascontiguousarray(screens[i])).save(tmp, format="PNG", compress_level=1)
                for attempt in range(5):
                    try:
                        os.replace(tmp, path)
                        break
                    except PermissionError:
                        time.sleep(0.01 * (attempt + 1))
            except Exception:
                continue

    def _heat_payload(self) -> list[dict[str, Any]]:
        top = sorted(self._map_visits.items(), key=lambda kv: kv[1], reverse=True)[:MAX_HEAT_MAPS]
        out = []
        for key, visits in top:
            cells = self._heat.get(key, {})
            if len(cells) > MAX_CELLS_PER_MAP:
                cells = dict(sorted(cells.items(), key=lambda kv: kv[1], reverse=True)[:MAX_CELLS_PER_MAP])
            g, n = (int(p) for p in key.split(","))
            out.append(
                {
                    "map": key,
                    "name": get_map_name(g, n),
                    "visits": visits,
                    "cells": [[x, y, c] for (x, y), c in cells.items()],
                }
            )
        return out

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        trails = [[list(p) for p in tr] for tr in self._trails]
        return {
            "t": round(now, 2),
            "started": round(self.started, 2),
            "finished": self.finished,
            "update": self.update,
            "steps": self.steps,
            "samples": self.total_samples,
            "steps_per_sec": round(self._steps_per_sec, 1),
            "num_envs": self.num_envs,
            "run": self.run_info,
            "maps_seen": len(self._seen_map_keys),
            "best_maps": self._best_maps,
            "best_badges": self._best_badges,
            "best_ep_return": None if self._best_ep_return == float("-inf") else round(self._best_ep_return, 2),
            "actions": {
                "names": self.action_names,
                "total": self._action_total.tolist(),
                "recent": self._action_recent.tolist(),
            },
            "envs": self._envs,
            "trails": trails,
            "heat": self._heat_payload(),
            "episodes": list(self._episodes),
            "events": list(self._events),
        }

    def flush(self) -> None:
        try:
            _atomic_write(self.out_dir / "state.json", json.dumps(self.snapshot(), separators=(",", ":")))
            self._last_state_t = time.time()
        except Exception:
            pass
