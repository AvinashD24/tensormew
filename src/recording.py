"""Record button events and play them back from the same snapshot."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.config import RUNS_DIR, sha1_bytes
from src.emulator import InputEvent


class InputRecorder:
    def __init__(self, machine, runs_dir: Path = RUNS_DIR):
        self.machine = machine
        self.runs_dir = Path(runs_dir)
        self._events: list[InputEvent] = []
        self._unsub = None
        self._info: dict = {}

    def on_start(self, info: dict) -> None:
        self._info = dict(info)
        self._events = []
        self._unsub = self.machine.add_listener(self._events.append)

    def on_step(self, rec, state) -> None:
        return None

    def on_end(self, result) -> None:
        if self._unsub:
            self._unsub()
            self._unsub = None
        self.machine.tick(1, render=True)
        screen = sha1_bytes(self.machine.screen_image().tobytes())
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{result.episode_id}.inputs.jsonl"
        header = {
            "type": "header",
            "episode_id": result.episode_id,
            "state": self._info.get("state"),
            "state_sha1": self._info.get("state_sha1"),
            "rom_sha1": self._info.get("rom_sha1"),
            "rtc_locked": True,
            "total_frames": self.machine.frame,
            "final_state": result.final_state,
            "final_screen_sha1": screen,
        }
        with path.open("w") as f:
            f.write(json.dumps(header) + "\n")
            for event in self._events:
                kind = "p" if event.kind == "press" else "r"
                f.write(json.dumps({"f": event.frame, "b": event.button, "k": kind}) + "\n")
            f.flush()
        self.path = path


def load_inputs(path) -> tuple[dict, list[InputEvent]]:
    lines = Path(path).read_text().splitlines()
    header = json.loads(lines[0])
    events = []
    for line in lines[1:]:
        if not line.strip():
            continue
        row = json.loads(line)
        kind = "press" if row["k"] == "p" else "release"
        events.append(InputEvent(int(row["f"]), row["b"], kind))
    return header, events


@dataclass
class ReplayResult:
    final_state: dict
    screen_sha1: str
    matches: bool | None
    mismatches: list[str]


def replay(machine, reader, state_bytes, events, total_frames, *, expected: dict | None = None, on_frame: Callable | None = None, render_every: int = 0) -> ReplayResult:
    machine.load_state(state_bytes)
    by_frame: dict[int, list[InputEvent]] = {}
    for event in events:
        by_frame.setdefault(event.frame, []).append(event)
    last = max(total_frames, max((e.frame for e in events), default=0))
    for frame in range(last):
        for event in by_frame.get(frame, []):
            if event.kind == "press":
                machine.hold(event.button)
            else:
                machine.release(event.button)
        render = (render_every > 0 and frame % render_every == 0) or frame == last - 1
        machine.tick(1, render=render)
        if on_frame and render:
            on_frame(machine, frame)
    if machine.frame == 0 or last == 0:
        machine.tick(1, render=True)
    state = reader.parse(machine.frame)
    screen = sha1_bytes(machine.screen_image().tobytes())
    mismatches: list[str] = []
    matches = None
    if expected is not None:
        matches = True
        exp_state = expected.get("final_state") or {}
        got = state.to_dict()
        for key in ("map_group", "map_number", "x", "y", "party_count", "badges"):
            if exp_state.get(key) != got.get(key):
                mismatches.append(f"{key} {got.get(key)} != {exp_state.get(key)}")
                matches = False
        if expected.get("final_screen_sha1") and expected["final_screen_sha1"] != screen:
            mismatches.append("screen hash")
            matches = False
    return ReplayResult(state.to_dict(), screen, matches, mismatches)


class FrameDumper:
    def __init__(self, every: int = 2, scale: int = 2):
        self.every = every
        self.scale = scale
        self.frames = []

    def __call__(self, machine, frame: int) -> None:
        if frame % self.every != 0:
            return
        image = machine.screen_image().convert("RGB")
        if self.scale != 1:
            image = image.resize((image.width * self.scale, image.height * self.scale))
        self.frames.append(image)

    def save_gif(self, path, fps: int = 30) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            raise RuntimeError("no frames to write")
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=1000 / fps,
            loop=0,
        )

    def save_mp4(self, path, fps: int = 30) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found; brew install ffmpeg or use --gif")
        if not self.frames:
            raise RuntimeError("no frames to write")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = self.frames[0]
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{image.width}x{image.height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None
        for frame in self.frames:
            proc.stdin.write(frame.convert("RGB").tobytes())
        proc.stdin.close()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code}")
