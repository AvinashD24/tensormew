"""Route-checkpoint archive (``long_v6``): self-generated reverse curriculum along the story path.

Why: in run v5-1 the best agent got 97 of 99 steps of Route 29 once, but the swarm archive only stores a state on the
first arrival in a new story MAP, so no state near the END of Route 29 exists and the agents never practise the last
part of the route (typical per-episode best was ~30 tiles).  Here every worker saves an emulator state whenever its
position reaches a new multiple of ``every`` (12) tiles of path progress (``src.path_progress``) in the running
episode; ``route_ckpt_prob`` of the resets load one of these states, sampled toward the furthest buckets.

Literature: Go-Explore (Ecoffet et al., 2019 / Nature 2021) keeps an archive of reached states and restarts from them;
the reverse curriculum (Florensa et al., 2017) starts the agent near the goal and moves the starts backwards; the
backward algorithm of Salimans & Chen (2018) learns Montezuma's Revenge from restarts along ONE demonstration.  Here the
"demonstration" is the agents' own best progress, no human data.  Bucket size 12, 4 states per bucket, the weight
floor 0.2 and the 40 % reset share are guesses.

Files: ``rc_b<bucket>_<slot>_s<stage>.state`` in one directory shared by the worker processes of a run.  ``stage`` is
the number of story milestones the saving episode had already passed: a loader seeds its seen-map set with those
milestone maps so their bonuses are never paid again (the saver already got them).  Pure Python (no numpy / torch) so
the cloud launcher's config import chain stays light.

Reset order of ``StartStateSelector`` (one uniform draw u per reset, the very first episode is always a start state):
``[0, battle_prob)`` battle archive, next ``swarm_prob`` swarm archive, next ``route_prob`` route checkpoint, the rest a
start state (config validation keeps the three shares at or below 1).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROUTE_PREFIX = "route:"
ROUTE_CKPT_EVERY = 12       # tiles of path progress per bucket
ROUTE_PER_BUCKET = 4        # snapshots kept per bucket (the oldest is replaced)
ROUTE_MAX_FILES = 60        # hard bound over all buckets (about 100 KB each)
ROUTE_WEIGHT_FLOOR = 0.2    # sampling weight of the nearest bucket relative to the furthest (1.0)
ROUTE_SAVE_GAP = 64         # a worker saves at most one checkpoint per this many of its own steps

_NAME = re.compile(r"^rc_b(\d+)_(\d+)_s(\d+)(?:_p(\d+))?$")   # optional _p<phase>: the egg-errand phase (long_v8)


def validate_params(every, prob, per_bucket, max_files, floor) -> list[str]:
    errs: list[str] = []
    if every < 1:
        errs.append(f"route_ckpt_every must be >= 1 (got {every})")
    if not 0 <= prob <= 1:
        errs.append(f"route_ckpt_prob must be in [0, 1] (got {prob})")
    if per_bucket < 1:
        errs.append(f"route_ckpt_per_bucket must be >= 1 (got {per_bucket})")
    if max_files < 1:
        errs.append(f"route_ckpt_max must be >= 1 (got {max_files})")
    if not 0 < floor <= 1:
        errs.append(f"route_ckpt_weight_floor must be in (0, 1] (got {floor})")
    return errs


def bucket_of(progress, every: int = ROUTE_CKPT_EVERY) -> int:
    """Bucket index of a path progress value (0 for None / negative); bucket ``b`` covers ``[b * every, (b + 1) * every)``."""
    if progress is None or progress < 0:
        return 0
    return int(progress // every)


def bucket_weight(bucket: int, max_bucket: int, floor: float = ROUTE_WEIGHT_FLOOR) -> float:
    """``floor + (1 - floor) * bucket / max_bucket``: the furthest bucket gets 1.0, never below ``floor``."""
    if max_bucket <= 0:
        return 1.0
    return float(floor) + (1.0 - float(floor)) * max(0, bucket) / float(max_bucket)


def key_info(key: str) -> tuple[int, int, int] | None:
    """``(bucket, slot, stage)`` parsed from a checkpoint key (file stem), or None."""
    m = _NAME.match(key)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def key_phase(key: str) -> int | None:
    """The egg-errand phase (0 / 1 / 2) a checkpoint key was saved in (``..._p<phase>``), or None for a key without one."""
    m = _NAME.match(key)
    return int(m.group(4)) if m and m.group(4) is not None else None


def milestone_stage(seen_maps) -> int:
    """How many story milestones the episode has passed: 1 + the index of the furthest milestone map in ``seen_maps``."""
    from src.story_path import profile
    stage = 0
    for i, m in enumerate(profile().milestones):
        if m.map_id is not None and tuple(m.map_id) in seen_maps:
            stage = i + 1
    return stage


def stage_maps(stage: int) -> set:
    """The milestone maps ``(group, number)`` of the first ``stage`` milestones (what a stage-``stage`` saver had seen)."""
    from src.story_path import profile
    return {tuple(m.map_id) for m in profile().milestones[: max(0, int(stage))] if m.map_id is not None}


class RouteCheckpointArchive:
    """Bounded file archive of route checkpoints, shared by the worker processes of one run (all file based)."""

    def __init__(self, directory: str | Path, every: int = ROUTE_CKPT_EVERY, per_bucket: int = ROUTE_PER_BUCKET,
                 max_files: int = ROUTE_MAX_FILES, floor: float = ROUTE_WEIGHT_FLOOR):
        errs = validate_params(every, 0.0, per_bucket, max_files, floor)
        if errs:
            raise ValueError("; ".join(errs))
        self.dir = Path(directory)
        self.every, self.per_bucket, self.max_files, self.floor = int(every), int(per_bucket), int(max_files), float(floor)
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- listing
    def entries(self) -> list[tuple[int, int, int, float, Path]]:
        """``(bucket, slot, stage, mtime, path)`` for every checkpoint file."""
        out = []
        try:
            paths = list(self.dir.glob("rc_b*.state"))
        except OSError:
            return out
        for p in paths:
            info = key_info(p.stem)
            if info is None:
                continue
            try:
                out.append((info[0], info[1], info[2], p.stat().st_mtime, p))
            except OSError:
                pass
        return out

    def keys(self) -> list[str]:
        return sorted(e[4].stem for e in self.entries())

    def counts(self) -> dict[int, int]:
        """Number of stored states per bucket (only non-empty buckets)."""
        c: dict[int, int] = {}
        for b, *_ in self.entries():
            c[b] = c.get(b, 0) + 1
        return dict(sorted(c.items()))

    def max_bucket(self) -> int:
        c = self.counts()
        return max(c) if c else 0

    # -- writing
    def save(self, bucket: int, stage: int, data: bytes, phase: int | None = None) -> bool:
        """Store a snapshot for ``bucket``: fill a free slot (``per_bucket`` of them) else replace the OLDEST one in the
        bucket, then enforce the global bound.  Returns False only when the write failed.  ``phase`` (egg errand, long_v8)
        is recorded in the file name (``_p<phase>``) for diagnostics; the loader reads the real phase from the state's flags."""
        bucket = max(0, int(bucket))
        mine = {slot: (mtime, p) for b, slot, _s, mtime, p in self.entries() if b == bucket}
        free = [n for n in range(self.per_bucket) if n not in mine]
        old = None
        if free:
            slot = free[0]
        else:
            slot = min(mine, key=lambda n: (mine[n][0], n))
            old = mine[slot][1]
        suffix = "" if phase is None else f"_p{int(phase)}"
        target = self.dir / f"rc_b{bucket:03d}_{slot}_s{int(stage)}{suffix}.state"
        tmp = self.dir / f".rc_b{bucket:03d}_{slot}.{os.getpid()}.tmp"
        try:
            tmp.write_bytes(data)
            os.replace(tmp, target)
        except OSError:
            return False
        if old is not None and old != target:
            try:
                old.unlink()
            except OSError:
                pass
        self.prune()
        return True

    def prune(self) -> None:
        """Delete the oldest state of the fullest bucket (ties: the lowest bucket) until at most ``max_files`` remain, so
        every bucket keeps at least one state as long as possible and the furthest buckets are never starved."""
        entries = self.entries()
        while len(entries) > self.max_files:
            per: dict[int, list] = {}
            for e in entries:
                per.setdefault(e[0], []).append(e)
            fullest = max(per, key=lambda b: (len(per[b]), -b))
            victim = min(per[fullest], key=lambda e: (e[3], e[1]))
            try:
                victim[4].unlink()
            except OSError:
                pass
            entries = [e for e in entries if e[4] != victim[4]]

    # -- reading
    def sample(self, rng) -> str | None:
        """A checkpoint key: bucket drawn with ``bucket_weight`` (furthest bucket most likely, floor for the rest), then
        uniform inside the bucket.  None when the archive is empty.  Uses two ``rng.random()`` draws."""
        by_bucket: dict[int, list[str]] = {}
        for b, _slot, _stage, _mt, p in self.entries():
            by_bucket.setdefault(b, []).append(p.stem)
        if not by_bucket:
            return None
        buckets = sorted(by_bucket)
        top = buckets[-1]
        weights = [bucket_weight(b, top, self.floor) for b in buckets]
        x, acc, chosen = float(rng.random()) * sum(weights), 0.0, buckets[-1]
        for b, w in zip(buckets, weights):
            acc += w
            if x < acc:
                chosen = b
                break
        keys = sorted(by_bucket[chosen])
        return keys[min(len(keys) - 1, int(float(rng.random()) * len(keys)))]

    def load(self, key: str) -> tuple[bytes, int] | None:
        """``(state bytes, stage)`` of a checkpoint key, or None when it vanished (evicted by another worker)."""
        info = key_info(key)
        if info is None:
            return None
        try:
            return (self.dir / f"{key}.state").read_bytes(), info[2]
        except OSError:
            return None
