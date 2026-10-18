"""Search the current map by walking tiles we have not stood on yet."""

from __future__ import annotations

from src.collision import is_ledge
from src.mover import walkable

_DELTA = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
_ORDER = ("up", "down", "left", "right")


def _toward(src: tuple[int, int], dst: tuple[int, int]) -> str | None:
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    if abs(dx) + abs(dy) != 1:
        return None
    for direction, (sx, sy) in _DELTA.items():
        if (sx, sy) == (dx, dy):
            return direction
    return None


class Explorer:
    """Pick a walkable step toward an unvisited tile. No map or route list."""

    def __init__(self):
        self.map_id: tuple[int, int] | None = None
        self.visited: set[tuple[int, int]] = set()
        self.tried: dict[tuple[int, int], set[str]] = {}
        self.trail: list[tuple[int, int]] = []

    def choose(self, state) -> str | None:
        self._ensure(tuple(state.map_id))
        pos = (state.x, state.y)
        self.visited.add(pos)
        self.tried.setdefault(pos, set())
        open_dirs = [
            direction
            for direction in _ORDER
            if walkable(int(state.tiles.get(direction, 0xFF)), direction)
        ]
        fresh = []
        for direction in open_dirs:
            if direction in self.tried[pos]:
                continue
            nx = pos[0] + _DELTA[direction][0]
            ny = pos[1] + _DELTA[direction][1]
            if (nx, ny) not in self.visited:
                fresh.append(direction)
        if fresh:
            regular_fresh = [d for d in fresh if not is_ledge(int(state.tiles.get(d, 0xFF)))]
            if regular_fresh:
                return regular_fresh[0]
            return fresh[0]
        untried = [direction for direction in open_dirs if direction not in self.tried[pos]]
        if untried:
            regular_untried = [d for d in untried if not is_ledge(int(state.tiles.get(d, 0xFF)))]
            if regular_untried:
                return regular_untried[0]
            return untried[0]
        if len(self.trail) >= 2 and self.trail[-1] == pos:
            return _toward(pos, self.trail[-2])
        return None

    def note(self, before, after, direction: str, moved: bool) -> None:
        origin = (before.x, before.y)
        self._ensure(tuple(before.map_id))
        self.tried.setdefault(origin, set()).add(direction)
        if tuple(after.map_id) != tuple(before.map_id):
            self._ensure(tuple(after.map_id))
            arrived = (after.x, after.y)
            self.visited.add(arrived)
            self.trail = [arrived]
            return
        if not moved:
            return
        arrived = (after.x, after.y)
        self.visited.add(arrived)
        if not self.trail or self.trail[-1] != origin:
            self.trail.append(origin)
        if self.trail[-1] != arrived:
            self.trail.append(arrived)

    def _ensure(self, map_id: tuple[int, int]) -> None:
        if self.map_id == map_id:
            return
        self.map_id = map_id
        self.visited = set()
        self.tried = {}
        self.trail = []
