"""One tile of movement. Turn if needed, then walk. Wait out a warp."""

from __future__ import annotations

from dataclasses import dataclass

DIRS = ("up", "down", "left", "right")
WARP_SETTLE = 180


@dataclass(frozen=True)
class TileStep:
    direction: str
    turned: bool
    moved: bool
    map_before: tuple[int, int]
    map_after: tuple[int, int]
    warped: bool
    frames: int

    def __bool__(self) -> bool:
        return self.moved


class TileStepper:
    def __init__(self, machine, reader):
        self.machine = machine
        self.reader = reader

    def step(self, direction: str) -> TileStep:
        if direction not in DIRS:
            raise ValueError(f"unknown direction {direction!r}")
        before = self.reader.parse(self.machine.frame)
        frames = 0
        turned = before.facing != direction
        if turned:
            self.machine.press(direction, frames=8)
            frames += 8
        self.machine.press(direction, frames=8)
        self.machine.tick(12, render=False)
        frames += 20
        mid = self.reader.parse(self.machine.frame)
        warped = mid.map_id != before.map_id
        if warped:
            self.machine.tick(WARP_SETTLE, render=False)
            frames += WARP_SETTLE
        after = self.reader.parse(self.machine.frame)
        moved = after.pos != before.pos or after.map_id != before.map_id
        return TileStep(
            direction=direction,
            turned=turned,
            moved=moved,
            map_before=before.map_id,
            map_after=after.map_id,
            warped=warped,
            frames=frames,
        )
