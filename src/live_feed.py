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
    from src.game_data import get_map_name, get_species_name
except Exception:  # pragma: no cover

    def get_species_name(species_id: int) -> str:
        return f"Pokemon #{species_id}"

    def get_map_name(map_group: int, map_number: int) -> str:
        return f"Map ({map_group}, {map_number})"


TRAIL_LEN = 48
MAX_HEAT_MAPS = 12
MAX_CELLS_PER_MAP = 4000
MAX_EPISODES = 60
MAX_EVENTS = 80
# Per-step reward terms reported by ``rl_env.progress_reward(components=...)``; unknown keys are kept too.
COMPONENT_KEYS = (
    "step", "bump", "route", "tile", "neighbor", "map", "warp", "interaction", "event",
    "badge", "catch", "level", "battle", "damage", "heal", "whiteout", "behind_map", "other",
)


def dex_ids(bits: Any) -> set[int]:
    """Species numbers (1-based) set in a Pokedex bitmap (32 bytes, bit j of byte k = species k*8+j+1)."""
    out: set[int] = set()
    for k, b in enumerate(bits or ()):
        b = int(b)
        while b:
            low = b & -b
            out.add(k * 8 + low.bit_length())
            b ^= low
    return out


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
        history_dir: str | Path | None = None,
        frame_every: int = 5,
    ):
        # Per-env live screenshots are intentionally NOT written: 24 PNG encodes per call cost ~1.4 s on a network
        # volume and, run every rollout step, slowed training ~12x (238 s per update). Only the replay montage
        # (history_dir, every ``frame_every`` updates) is saved.
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Append-only per-run record (positions per update + frame montages) for replay after the run.
        self.history_dir = Path(history_dir) if history_dir else None
        self.frame_every = max(1, frame_every)
        if self.history_dir is not None:
            (self.history_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.run_id = self.history_dir.name if self.history_dir is not None else None
        self._upd_heat: dict[str, dict[tuple[int, int], int]] = {}
        self._upd_eps: list[dict[str, Any]] = []
        self._upd_events: list[dict[str, Any]] = []
        # Reward by component: {"pos": {name: sum of positive terms}, "neg": {name: sum of negative terms}}.
        self._upd_comp = self._blank_comp()
        self._run_comp = self._blank_comp()
        self._last_comp = self._blank_comp()
        self._milestones: dict[str, Any] | None = None   # latest milestone reach rates (src.milestone_stats), compact
        self._records: dict[str, Any] | None = None      # fastest-to-milestone records (MilestoneRecords.snapshot), compact
        self._eval: dict[str, Any] | None = None         # latest periodic from-start evaluation (long_v8), compact; None = none yet
        self._upd_records: list[dict[str, Any]] = []     # record events of the running update (go into history.jsonl)
        self._last_screens: np.ndarray | None = None
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
        # Cumulative "what the agents did" counters (run-wide, across all envs), fed by info["fun"].
        self._fun_on = False
        self._enc: dict[int, int] = {}      # species -> times battled (wild + trainer)
        self._got: dict[int, int] = {}      # species -> times newly caught/obtained
        self._dex_seen: set[int] = set()
        self._dex_own: set[int] = set()
        self._wild = self._trn = self._whiteouts = self._max_level = self._max_money = 0
        self._first_map: dict[str, tuple[int, str]] = {}   # map key -> (feed step first entered, name)
        self._fun_prev: list[dict[str, Any] | None] = [None] * num_envs
        self._env_party: dict[int, list[list[int]]] = {}

        self._last_frame_t = 0.0
        self._last_state_t = 0.0
        self._rate_t = self.started
        self._rate_steps = 0
        self._steps_per_sec = 0.0
        self._write_meta()

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

    @staticmethod
    def _blank_comp() -> dict[str, dict[str, float]]:
        return {"pos": {}, "neg": {}}

    def _add_components(self, comps: Any) -> None:
        if not isinstance(comps, dict):
            return
        for name, val in comps.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v == 0.0 or v != v or v in (float("inf"), float("-inf")):
                continue
            side = "pos" if v > 0 else "neg"
            key = str(name)
            for acc in (self._upd_comp, self._run_comp):
                acc[side][key] = acc[side].get(key, 0.0) + v

    @staticmethod
    def _comp_payload(comp: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        return {side: {k: round(v, 3) for k, v in comp[side].items()} for side in ("pos", "neg")}

    def _event(self, kind: str, text: str, **extra: Any) -> None:
        ev = {"t": round(time.time(), 2), "kind": kind, "text": text, **extra}
        self._events.append(ev)
        self._upd_events.append(ev)

    def _write_meta(self) -> None:
        if self.history_dir is None:
            return
        meta = {
            "run_id": self.run_id,
            "started": round(self.started, 2),
            "finished": self.finished,
            "updates_logged": self.update,
            "num_envs": self.num_envs,
            "action_names": self.action_names,
            "run": self.run_info,
            "species_names": self._species_names(),
            "map_names": {k: get_map_name(*(int(p) for p in k.split(","))) for k in self._seen_map_keys},
        }
        try:
            _atomic_write(self.history_dir / "meta.json", json.dumps(meta, indent=1))
        except Exception:
            pass

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
        if screens is not None:
            self._last_screens = screens

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
                self._add_components(info.get("reward_components"))

            if info:
                self._ingest_fun(i, info, bool(dones[i]))

            if bool(dones[i]):
                self._finish_episode(i, info or {})

            env["ep_return"] = round(float(self._ep_return[i]), 2)
            env["ep_steps"] = int(self._ep_steps[i])

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
            upd = self._upd_heat.setdefault(key, {})
            upd[(x, y)] = upd.get((x, y), 0) + 1
            self._map_visits[key] = self._map_visits.get(key, 0) + 1
            self._trails[i].append((key, x, y))
            if key not in self._seen_map_keys:
                self._seen_map_keys.add(key)
                self._first_map[key] = (self.steps, env["map_name"])
                self._event("map", f"env {i} entered {env['map_name']}", env=i, map=key)
        badges = int(info.get("badges", 0))
        if badges > self._best_badges:
            self._best_badges = badges
            self._event("badge", f"env {i} earned badge #{badges}", env=i)

    def _ingest_fun(self, i: int, info: dict[str, Any], done: bool) -> None:
        """Fold one env step's ``info["fun"]`` into the cumulative counters (cheap: no emulator reads)."""
        comps = info.get("reward_components")
        if isinstance(comps, dict) and comps.get("whiteout"):
            self._whiteouts += 1
        f = info.get("fun")
        if not isinstance(f, dict):
            return
        self._fun_on = True
        bm, es = int(f.get("bm", 0)), int(f.get("es", 0))
        own, seen = dex_ids(f.get("dc")), dex_ids(f.get("ds"))
        prev = self._fun_prev[i]
        if prev is not None:
            if bm and not prev["bm"]:
                prev["pending"], prev["kind"] = True, bm
            if prev.get("pending") and (es > 0 or not bm):
                if es > 0:
                    self._enc[es] = self._enc.get(es, 0) + 1
                if prev["kind"] == 2:
                    self._trn += 1
                else:
                    self._wild += 1
                prev["pending"] = False
            for sp in own - prev["own"]:
                self._got[sp] = self._got.get(sp, 0) + 1
        self._dex_own |= own
        self._dex_seen |= seen
        levels = f.get("pl") or ()
        hps = f.get("ph") or ()
        party = []
        for k, sp in enumerate(f.get("ps") or ()):
            if k < 6 and 1 <= int(sp) <= 251:
                row = [int(sp), int(levels[k]) if k < len(levels) else 0]
                if k < len(hps) and int(hps[k][1]) > 0:
                    row += [int(hps[k][0]), int(hps[k][1])]
                party.append(row)
        self._env_party[i] = party
        if levels:
            self._max_level = max(self._max_level, int(max(levels)))
        self._max_money = max(self._max_money, int(f.get("mo", 0)))
        # An episode reset reloads a start state: re-baseline instead of counting its Pokedex as new catches.
        self._fun_prev[i] = {"bm": bm, "own": own, "pending": False if done or prev is None else prev.get("pending", False),
                             "kind": 1 if prev is None else prev.get("kind", 1)}

    def _species_names(self) -> dict[str, str]:
        ids = set(self._enc) | set(self._got) | {r[0] for p in self._env_party.values() for r in p}
        return {str(sp): get_species_name(sp) for sp in ids}

    def fun_payload(self) -> dict[str, Any] | None:
        """Small cumulative record for history/live; None until any env has reported Pokemon facts."""
        if not self._fun_on:
            return None
        first = sorted(self._first_map.values())[-6:]
        # Showcase party = the env whose strongest Pokemon is highest (ties: higher level sum, then lower env id).
        best = max(self._env_party.items(), key=lambda kv: (max((r[1] for r in kv[1]), default=0),
                                                            sum(r[1] for r in kv[1]), -kv[0]), default=None)
        return {
            "party": best[1] if best and best[1] else [], "pe": best[0] if best else 0,
            # species battled but not (yet) in the Pokedex-owned set: drawn as silhouettes
            "unowned": sorted(sp for sp in self._enc if sp not in self._dex_own),
            "enc": {str(k): v for k, v in self._enc.items()},
            "got": {str(k): v for k, v in self._got.items()},
            "seen": len(self._dex_seen), "own": len(self._dex_own),
            "wild": self._wild, "trn": self._trn, "wo": self._whiteouts,
            "lvl": self._max_level, "money": self._max_money,
            "first": [[n, st] for st, n in first],
        }

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
        self._upd_eps.append(episode)
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

    def set_milestones(self, payload: dict[str, Any] | None) -> None:
        """Latest milestone success-rate payload (``MilestoneStats.end_update``); shown live in state.json."""
        self._milestones = dict(payload) if payload else None

    def record_milestone(self, ev: dict[str, Any], snapshot: dict[str, Any] | None = None) -> None:
        """A new fastest-to-milestone record (``MilestoneRecords`` event): live events-feed entry + state.json right away.

        Only the "from the base start state" records are announced; the "any start" ones live in the records table."""
        if snapshot is not None:
            self._records = snapshot
        if ev.get("scope") != "start":
            return
        name = str(ev.get("name", "milestone"))
        if ev.get("first"):
            text = f"First time reaching {name}: {ev['steps']} steps (env {ev.get('env')})"
        else:
            text = f"New record: {name} in {ev['steps']} steps (was {ev['prev']}, {ev['steps'] - ev['prev']:+d})"
        self._event("milestone", text, env=ev.get("env"), rec=ev)
        self.flush()

    def set_eval(self, payload: dict[str, Any], text: str | None = None) -> None:
        """Latest periodic from-start evaluation (``eval_*`` fields, long_v8): kept in state.json and announced as an event."""
        self._eval = dict(payload)
        if text:
            self._event("eval", text)

    def set_records(self, snapshot: dict[str, Any] | None, new_events: Sequence[dict[str, Any]] | None = None) -> None:
        """Records snapshot for state.json plus this update's new record events (written to the history record)."""
        self._records = snapshot
        if new_events:
            self._upd_records.extend(dict(e) for e in new_events)

    def on_update(self, update: int, samples: int, metrics: dict[str, Any] | None = None) -> None:
        """Called once per PPO update: persists the update's history and resets per-update windows."""
        self.update = update
        self.total_samples += samples
        self._write_history(update, metrics)
        self._last_comp = self._upd_comp
        self._upd_comp = self._blank_comp()
        self._action_recent[:] = 0
        self._upd_heat = {}
        self._upd_eps = []
        self._upd_events = []
        self._upd_records = []
        self.flush()

    def finish(self) -> None:
        self.finished = True
        if self.history_dir is not None and self._last_screens is not None:
            self._write_montage(self.update)  # always keep the final frame
        self._write_meta()
        self.flush()

    # ----------------------------------------------------------------- history
    def _write_history(self, update: int, metrics: dict[str, Any] | None) -> None:
        if self.history_dir is None:
            return
        record = {
            "update": update,
            "t": round(time.time(), 2),
            "steps": self.steps,
            "metrics": metrics or {},
            "envs": [
                [e["map"], e["x"], e["y"], e["ep_return"], e["ep_steps"], int(e["battle"])] for e in self._envs
            ],
            "heat": {k: [[x, y, c] for (x, y), c in cells.items()] for k, cells in self._upd_heat.items()},
            "actions": self._action_recent.tolist(),
            "components": self._comp_payload(self._upd_comp),
            "episodes": self._upd_eps,
            "events": self._upd_events,
        }
        fun = self.fun_payload()
        if fun is not None:
            record["fun"] = fun
        if self._upd_records:   # only on updates where a record happened, so replay can show it
            record["records"] = self._upd_records
        try:
            with open(self.history_dir / "history.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            if update == 0 or update % self.frame_every == 0:
                self._write_montage(update)
        except Exception:
            pass
        self._write_meta()

    def _write_montage(self, update: int) -> None:
        """One JPEG contact sheet of every env's current screen (small enough to keep many)."""
        if Image is None or self._last_screens is None or self.history_dir is None:
            return
        screens = self._last_screens
        n = min(self.num_envs, len(screens))
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        h, w = screens.shape[1], screens.shape[2]
        sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i in range(n):
            r, c = divmod(i, cols)
            sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = screens[i][:, :, :3]
        Image.fromarray(sheet).save(self.history_dir / "frames" / f"u{update:06d}.jpg", format="JPEG", quality=72)

    # ---------------------------------------------------------------- snapshot
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
            "run_id": self.run_id,
            "maps_seen": len(self._seen_map_keys),
            "best_maps": self._best_maps,
            "best_badges": self._best_badges,
            "best_ep_return": None if self._best_ep_return == float("-inf") else round(self._best_ep_return, 2),
            "actions": {
                "names": self.action_names,
                "total": self._action_total.tolist(),
                "recent": self._action_recent.tolist(),
            },
            "components": {
                "update": self._comp_payload(self._upd_comp),
                "last_update": self._comp_payload(self._last_comp),
                "run": self._comp_payload(self._run_comp),
            },
            "fun": self.fun_payload(),
            "milestones": self._milestones,
            "records": self._records,
            "eval": self._eval,
            "species_names": self._species_names(),
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
