"""Named save snapshots and the manifest that checks them."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import MANIFEST_PATH, ROM_SHA1, STATES_DIR, sha1_bytes

_NAME = re.compile(r"^[a-z0-9_]+$")


class StateNotFound(KeyError):
    pass


@dataclass
class StateEntry:
    name: str
    file: str
    description: str
    map: tuple[int, int]
    pos: tuple[int, int]
    badges: int
    party_count: int
    time_of_day: int
    rom_sha1: str
    state_sha1: str
    created_at: str
    tags: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["map"] = list(self.map)
        d["pos"] = list(self.pos)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "StateEntry":
        return cls(
            name=d["name"],
            file=d["file"],
            description=d.get("description", ""),
            map=tuple(d["map"]),
            pos=tuple(d["pos"]),
            badges=int(d["badges"]),
            party_count=int(d["party_count"]),
            time_of_day=int(d.get("time_of_day", 0)),
            rom_sha1=d["rom_sha1"],
            state_sha1=d["state_sha1"],
            created_at=d["created_at"],
            tags=list(d.get("tags") or []),
        )


@dataclass
class ValidationResult:
    name: str
    ok: bool
    problems: list[str]


class StateRegistry:
    def __init__(self, root: Path = STATES_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json" if self.root != STATES_DIR else MANIFEST_PATH
        if self.root != STATES_DIR:
            self.manifest_path = self.root / "manifest.json"

    def _load(self) -> dict:
        if not self.manifest_path.exists():
            return {"version": 1, "states": {}}
        return json.loads(self.manifest_path.read_text())

    def _save(self, doc: dict) -> None:
        doc["states"] = dict(sorted(doc["states"].items()))
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(tmp, self.manifest_path)

    def names(self) -> list[str]:
        return sorted(self._load()["states"])

    def get(self, name: str) -> StateEntry:
        doc = self._load()
        if name not in doc["states"]:
            raise StateNotFound(name)
        return StateEntry.from_json(doc["states"][name])

    def path(self, name: str) -> Path:
        entry = self.get(name)
        return self.root / entry.file

    def load_bytes(self, name: str) -> bytes:
        direct = self.root / (name if name.endswith(".state") else f"{name}.state")
        doc = self._load()
        if name not in doc.get("states", {}) and direct.exists():
            return direct.read_bytes()
        entry = self.get(name)
        path = self.root / entry.file
        if not path.exists():
            raise StateNotFound(f"missing file for {name}")
        data = path.read_bytes()
        got = sha1_bytes(data)
        if got != entry.state_sha1:
            raise ValueError(f"{name} hash {got} != manifest {entry.state_sha1}")
        return data

    def _check_name(self, name: str) -> None:
        if not _NAME.match(name):
            raise ValueError(f"illegal state name {name!r}")

    def _entry_from_state(self, name, description, tags, data, state) -> StateEntry:
        return StateEntry(
            name=name,
            file=f"{name}.state",
            description=description,
            map=(state.map_group, state.map_number),
            pos=(state.x, state.y),
            badges=state.badges,
            party_count=state.party_count,
            time_of_day=state.time_of_day,
            rom_sha1=ROM_SHA1,
            state_sha1=sha1_bytes(data),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tags=list(tags),
        )

    def capture(self, emu, parser, name, description="", tags=(), force=False, overwrite=False) -> StateEntry:
        self._check_name(name)
        state = parser.parse(emu.frame)
        reasons = state.not_ready_reasons()
        tag_list = list(tags)
        if reasons and not force:
            raise ValueError("not ready: " + ", ".join(reasons))
        if reasons and force and "forced" not in tag_list:
            tag_list.append("forced")
        doc = self._load()
        if name in doc["states"] and not overwrite:
            raise ValueError(f"{name} already registered")
        data = emu.save_state(self.root / f"{name}.state")
        entry = self._entry_from_state(name, description, tag_list, data, state)
        doc["states"][name] = entry.to_json()
        self._save(doc)
        return entry

    def register_file(self, path, emu, parser, name, description="", tags=(), force=False, overwrite=False) -> StateEntry:
        self._check_name(name)
        data = Path(path).read_bytes()
        emu.load_state(data)
        state = parser.parse(emu.frame)
        reasons = state.not_ready_reasons()
        tag_list = list(tags)
        if reasons and not force:
            raise ValueError("not ready: " + ", ".join(reasons))
        if reasons and force and "forced" not in tag_list:
            tag_list.append("forced")
        doc = self._load()
        if name in doc["states"] and not overwrite:
            raise ValueError(f"{name} already registered")
        dest = self.root / f"{name}.state"
        dest.write_bytes(data)
        entry = self._entry_from_state(name, description, tag_list, data, state)
        doc["states"][name] = entry.to_json()
        self._save(doc)
        return entry

    def validate(self, emu, parser, names=None) -> list[ValidationResult]:
        chosen = list(names) if names else self.names()
        results = []
        for name in chosen:
            problems = []
            try:
                entry = self.get(name)
            except StateNotFound:
                results.append(ValidationResult(name, False, ["missing manifest entry"]))
                continue
            path = self.root / entry.file
            if not path.exists():
                results.append(ValidationResult(name, False, ["missing file"]))
                continue
            data = path.read_bytes()
            if sha1_bytes(data) != entry.state_sha1:
                problems.append("state_sha1 mismatch")
            if entry.rom_sha1 != ROM_SHA1:
                problems.append("rom_sha1 mismatch")
            emu.load_state(data)
            state = parser.parse(emu.frame)
            if (state.map_group, state.map_number) != tuple(entry.map):
                problems.append(f"map {state.map_id} != {tuple(entry.map)}")
            if (state.x, state.y) != tuple(entry.pos):
                problems.append(f"pos {(state.x, state.y)} != {tuple(entry.pos)}")
            if state.badges != entry.badges:
                problems.append(f"badges {state.badges} != {entry.badges}")
            if state.party_count != entry.party_count:
                problems.append(f"party {state.party_count} != {entry.party_count}")
            results.append(ValidationResult(name, not problems, problems))
        return results
