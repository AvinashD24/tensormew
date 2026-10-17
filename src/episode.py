"""Play a snapshot forward and write a plain log. No model calls."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from src.config import ROM_SHA1, RUNS_DIR, STATES_DIR, sha1_bytes

_MAX_FRAMES = 600


class Script(Protocol):
    name: str

    def reset(self, state, seed: int) -> None: ...
    def act(self, state): ...


@dataclass(frozen=True)
class Action:
    kind: Literal["press", "wait", "done"]
    button: str | None = None
    frames: int = 8
    settle: int = 16

    def __str__(self) -> str:
        if self.kind == "press":
            return f"press:{self.button}"
        if self.kind == "wait":
            return f"wait:{self.frames}"
        return "done"


class ButtonScript:
    name = "buttons"

    def __init__(self, actions):
        self.actions = list(actions)
        self._i = 0

    def reset(self, state, seed: int) -> None:
        self._i = 0

    def act(self, state) -> Action:
        if self._i >= len(self.actions):
            return Action("done")
        action = self.actions[self._i]
        self._i += 1
        return action


class RandomScript:
    name = "random"

    def __init__(self, buttons=("up", "down", "left", "right", "a")):
        self.buttons = tuple(buttons)
        self._rng = random.Random()

    def reset(self, state, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, state) -> Action:
        return Action("press", button=self._rng.choice(self.buttons))


class FileStateSource:
    def __init__(self, root: Path = STATES_DIR):
        self.root = Path(root)

    def load_bytes(self, name: str) -> bytes:
        path = Path(name)
        if not path.exists():
            path = self.root / f"{name}.state"
        return path.read_bytes()


@dataclass
class EpisodeResult:
    episode_id: str
    outcome: Literal["step_budget", "timeout", "script_done", "error"]
    steps: int
    frames: int
    log_path: Path
    wall_s: float
    final_state: dict
    error: str | None = None


def _clamp(n: int) -> int:
    return max(1, min(_MAX_FRAMES, int(n)))


def _apply(action: Action, machine) -> None:
    if action.kind == "press":
        machine.press(action.button, frames=_clamp(action.frames))
        machine.tick(_clamp(action.settle), render=False)
    elif action.kind == "wait":
        machine.tick(_clamp(action.frames), render=False)


class EpisodeRunner:
    def __init__(self, machine, reader, states, *, runs_dir: Path = RUNS_DIR, observers=()):
        self.machine = machine
        self.reader = reader
        self.states = states
        self.runs_dir = Path(runs_dir)
        self.observers = list(observers)

    def run(self, script, state_name: str, *, max_steps: int = 200, timeout_s: float = 300.0, seed: int = 0, episode_id: str | None = None) -> EpisodeResult:
        import time

        started = time.perf_counter()
        if episode_id is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            episode_id = f"{state_name}-{script.name}-{seed}-{stamp}"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.runs_dir / f"{episode_id}.jsonl"
        outcome: Literal["step_budget", "timeout", "script_done", "error"] = "step_budget"
        error = None
        steps = 0
        final = {}
        data = b""
        try:
            data = self.states.load_bytes(state_name)
            self.machine.load_state(data)
            state = self.reader.parse(self.machine.frame)
            script.reset(state, seed)
            header = {
                "type": "start",
                "episode_id": episode_id,
                "script": script.name,
                "state": state_name,
                "state_sha1": sha1_bytes(data),
                "rom_sha1": ROM_SHA1,
                "seed": seed,
                "max_steps": max_steps,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._line(log_path, header, truncate=True)
            for obs in self.observers:
                try:
                    obs.on_start(header)
                except Exception as exc:
                    error = str(exc)
            while steps < max_steps:
                action = script.act(state)
                if action.kind == "done":
                    outcome = "script_done"
                    break
                _apply(action, self.machine)
                state = self.reader.parse(self.machine.frame)
                steps += 1
                rec = {
                    "type": "step",
                    "step": steps,
                    "frame": self.machine.frame,
                    "action": str(action),
                    "state": state.to_dict(),
                }
                self._line(log_path, rec)
                final = rec["state"]
                for obs in self.observers:
                    try:
                        obs.on_step(rec, state)
                    except Exception as exc:
                        error = str(exc)
                if time.perf_counter() - started > timeout_s:
                    outcome = "timeout"
                    break
            else:
                outcome = "step_budget"
        except Exception as exc:
            outcome = "error"
            error = str(exc)
        wall = time.perf_counter() - started
        result = EpisodeResult(episode_id, outcome, steps, self.machine.frame, log_path, wall, final, error)
        footer = {
            "type": "end",
            "outcome": outcome,
            "steps": steps,
            "frames": self.machine.frame,
            "wall_s": wall,
            "error": error,
        }
        self._line(log_path, footer)
        for obs in self.observers:
            try:
                obs.on_end(result)
            except Exception as exc:
                result.error = str(exc)
        return result

    def _line(self, path: Path, obj: dict, truncate: bool = False) -> None:
        mode = "w" if truncate else "a"
        with path.open(mode) as f:
            f.write(json.dumps(obj) + "\n")
            f.flush()
