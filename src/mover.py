"""Greedy overworld movement. No Start/Select. No LLM."""

from __future__ import annotations

from src.action_results import MoveResult, PathResult
from src.collision import (
    COLL_BOOKSHELF,
    COLL_BUOY,
    COLL_COUNTER,
    COLL_COUNTER_98,
    COLL_DOWN_WALL,
    COLL_FF,
    COLL_HEADBUTT_TREE,
    COLL_HEADBUTT_TREE_1D,
    COLL_INCENSE_BURNER,
    COLL_LEFT_WALL,
    COLL_MART_SHELF,
    COLL_PC,
    COLL_RADIO,
    COLL_RIGHT_WALL,
    COLL_TOWN_MAP,
    COLL_TV,
    COLL_UP_WALL,
    COLL_WALL,
    COLL_WATER,
    COLL_WATER_21,
    COLL_WATERFALL,
    COLL_WATERFALL_LEFT,
    COLL_WATERFALL_RIGHT,
    COLL_WATERFALL_UP,
    COLL_WHIRLPOOL,
    COLL_WHIRLPOOL_2C,
    COLL_WINDOW,
    allows_direction,
    is_cut_tree,
)
from src.step import DIRS, TileStepper

_BLOCKED = frozenset(
    {
        COLL_WALL,
        COLL_RIGHT_WALL,
        COLL_LEFT_WALL,
        COLL_UP_WALL,
        COLL_DOWN_WALL,
        COLL_WATER,
        COLL_WATER_21,
        COLL_WHIRLPOOL,
        COLL_WHIRLPOOL_2C,
        COLL_BUOY,
        COLL_WATERFALL,
        COLL_WATERFALL_LEFT,
        COLL_WATERFALL_RIGHT,
        COLL_WATERFALL_UP,
        COLL_COUNTER,
        COLL_COUNTER_98,
        COLL_HEADBUTT_TREE,
        COLL_HEADBUTT_TREE_1D,
        COLL_BOOKSHELF,
        COLL_PC,
        COLL_RADIO,
        COLL_TOWN_MAP,
        COLL_MART_SHELF,
        COLL_TV,
        COLL_WINDOW,
        COLL_INCENSE_BURNER,
        COLL_FF,
    }
)

_DELTA = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def walkable(tile: int, direction: str | None = None) -> bool:
    if is_cut_tree(tile):
        return False
    if tile in _BLOCKED:
        return False
    if direction is not None and not allows_direction(tile, direction):
        return False
    return True


def _manhattan(x0: int, y0: int, x1: int, y1: int) -> int:
    return abs(x0 - x1) + abs(y0 - y1)


def _reducing_dirs(x: int, y: int, tx: int, ty: int) -> list[str]:
    dx, dy = tx - x, ty - y
    order: list[str] = []
    if abs(dx) >= abs(dy):
        if dx > 0:
            order.append("right")
        elif dx < 0:
            order.append("left")
        if dy > 0:
            order.append("down")
        elif dy < 0:
            order.append("up")
    else:
        if dy > 0:
            order.append("down")
        elif dy < 0:
            order.append("up")
        if dx > 0:
            order.append("right")
        elif dx < 0:
            order.append("left")
    return order


class Mover:
    def __init__(self, machine, reader, *, stuck_limit: int = 8, max_steps: int = 80):
        self.machine = machine
        self.reader = reader
        self.stepper = TileStepper(machine, reader)
        self.stuck_limit = stuck_limit
        self.max_steps = max_steps
        self.stuck = 0

    def snapshot(self):
        return self.reader.parse(self.machine.frame)

    def _note_step(self, before, after, moved: bool) -> None:
        if after.map_id != before.map_id or moved:
            self.stuck = 0
            return
        self.stuck += 1

    def go_to(self, x: int, y: int) -> MoveResult:
        frames = 0
        steps = 0
        last = self.snapshot()
        if last.battle_mode != 0:
            return MoveResult(False, "go_to refused: in battle", pos=last.pos, stuck=self.stuck)
        for _ in range(self.max_steps):
            last = self.snapshot()
            if (last.x, last.y) == (x, y):
                return MoveResult(
                    True,
                    f"arrived ({x},{y})",
                    frames=frames,
                    pos=last.pos,
                    stuck=self.stuck,
                    steps=steps,
                )
            if last.battle_mode != 0:
                return MoveResult(
                    False,
                    "interrupted by battle",
                    frames=frames,
                    pos=last.pos,
                    stuck=self.stuck,
                    steps=steps,
                )
            candidates = [d for d in _reducing_dirs(last.x, last.y, x, y) if walkable(last.tiles.get(d, 0xFF), d)]
            if not candidates:
                return MoveResult(
                    False,
                    f"blocked going to ({x},{y}) from ({last.x},{last.y}) tiles={last.tiles}",
                    frames=frames,
                    pos=last.pos,
                    stuck=self.stuck,
                    steps=steps,
                )
            result = self.stepper.step(candidates[0])
            frames += result.frames
            steps += 1
            after = self.snapshot()
            self._note_step(last, after, result.moved)
            if self.stuck >= self.stuck_limit:
                return MoveResult(
                    False,
                    f"stuck at {after.pos} after {self.stuck} failed steps",
                    frames=frames,
                    pos=after.pos,
                    stuck=self.stuck,
                    steps=steps,
                )
        after = self.snapshot()
        return MoveResult(
            False,
            f"step budget going to ({x},{y}); now ({after.x},{after.y})",
            frames=frames,
            pos=after.pos,
            stuck=self.stuck,
            steps=steps,
        )

    def run_path(self, directions: list[str]) -> PathResult:
        frames = 0
        done = 0
        start = self.snapshot()
        if start.battle_mode != 0:
            return PathResult(False, "run_path refused: in battle", pos=start.pos)
        for direction in directions:
            if direction not in DIRS:
                return PathResult(
                    False,
                    f"bad direction {direction!r}",
                    frames=frames,
                    steps_done=done,
                    pos=self.snapshot().pos,
                )
            before = self.snapshot()
            result = self.stepper.step(direction)
            frames += result.frames
            done += 1
            after = self.snapshot()
            self._note_step(before, after, result.moved)
            if not result.moved:
                return PathResult(
                    False,
                    f"stopped after {done} steps at {after.pos} ({direction} blocked)",
                    frames=frames,
                    steps_done=done,
                    pos=after.pos,
                )
        end = self.snapshot()
        return PathResult(True, f"ran {done} steps", frames=frames, steps_done=done, pos=end.pos)
