"""Per-episode memory of the tiles the agent has seen, turned into policy features.

The policy used to see only the four adjacent tile ids, so it could not tell that an
opening one column over existed, and it forgot a bump on the next step. This keeps a
small map built from what the agent has actually stood next to (no stored routes):

* ``blocked`` flags per direction, so the policy can mask moves that cannot work
* the last action and whether it moved, so a bump is visible in the next observation
* the first step of the shortest known path to the nearest unvisited open tile
* a 5x5 egocentric grid of visited / open / blocked / unknown cells
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.mover import walkable

DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

GRID_RADIUS = 2
GRID_SIZE = (2 * GRID_RADIUS + 1) ** 2
# blocked(4) + last action one-hot(5) + last moved(1) + frontier dir(4) + frontier dist(1) + grid
NAV_DIM = 4 + 5 + 1 + 4 + 1 + GRID_SIZE
MAX_BFS_NODES = 4000
BUMP_MEMORY_STEPS = 12
FRONTIER_DIST_SCALE = 20.0


def _map(state) -> tuple[int, int]:
    return (int(state.map_group), int(state.map_number))


class LocalNav:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        # (map_id) -> {(x, y): {direction: bool open}}  edges known from visited cells
        self.edges: dict[tuple[int, int], dict[tuple[int, int], dict[str, bool]]] = {}
        self.bumps: dict[tuple, int] = {}
        self.tick = 0
        self.last_action = -1
        self.last_moved = True
        self._frontier_cache = None

    # --- recording ---------------------------------------------------------------
    def update(self, state) -> None:
        """Record what the four sides of the current tile allow."""
        known = self.edges.setdefault(_map(state), {})
        known[(state.x, state.y)] = {
            direction: walkable(int(state.tiles.get(direction, 0xFF)), direction) for direction in DIRS
        }

    def note_action(self, state_before, action: int, moved: bool) -> None:
        """Remember the last action. A failed step blocks that edge for a while (NPCs, ledges)."""
        self.tick += 1
        self.last_action = int(action)
        self.last_moved = bool(moved)
        if 0 <= action < 4 and not moved:
            key = (_map(state_before), state_before.x, state_before.y, DIRS[action])
            self.bumps[key] = self.tick

    # --- queries ------------------------------------------------------------------
    def _bumped(self, map_id, x: int, y: int, direction: str) -> bool:
        stamp = self.bumps.get((map_id, x, y, direction))
        return stamp is not None and self.tick - stamp <= BUMP_MEMORY_STEPS

    def blocked_flags(self, state) -> list[float]:
        """1.0 when a direction is known not to work from here."""
        map_id = _map(state)
        out = []
        for direction in DIRS:
            blocked = not walkable(int(state.tiles.get(direction, 0xFF)), direction)
            blocked = blocked or self._bumped(map_id, state.x, state.y, direction)
            out.append(1.0 if blocked else 0.0)
        return out

    def frontier(self, state) -> tuple[str | None, int]:
        """First step and length of the shortest known path to an unvisited open tile."""
        map_id = _map(state)
        known = self.edges.get(map_id, {})
        start = (state.x, state.y)
        key = (map_id, start, len(known), self.tick)
        if self._frontier_cache is not None and self._frontier_cache[0] == key:
            return self._frontier_cache[1]
        result = self._frontier(map_id, known, start)
        self._frontier_cache = (key, result)
        return result

    def _frontier(self, map_id, known, start) -> tuple[str | None, int]:
        if start not in known:
            return None, 0
        queue = deque([(start, None, 0)])
        seen = {start}
        nodes = 0
        while queue and nodes < MAX_BFS_NODES:
            (x, y), first, dist = queue.popleft()
            nodes += 1
            edges = known.get((x, y))
            if edges is None:
                continue
            for direction in DIRS:
                if not edges[direction] or self._bumped(map_id, x, y, direction):
                    continue
                dx, dy = _DELTA[direction]
                nxt = (x + dx, y + dy)
                if nxt in seen:
                    continue
                seen.add(nxt)
                step = first or direction
                if nxt not in known:
                    return step, dist + 1
                queue.append((nxt, step, dist + 1))
        return None, 0

    def grid(self, state) -> list[float]:
        """5x5 around the player: 1 visited, 0.5 open but unvisited, -1 blocked, 0 unknown."""
        map_id = _map(state)
        known = self.edges.get(map_id, {})
        out: list[float] = []
        for dy in range(-GRID_RADIUS, GRID_RADIUS + 1):
            for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):
                cell = (state.x + dx, state.y + dy)
                if cell in known:
                    out.append(1.0)
                    continue
                value = 0.0
                for direction in DIRS:
                    ddx, ddy = _DELTA[direction]
                    origin = (cell[0] - ddx, cell[1] - ddy)
                    edges = known.get(origin)
                    if edges is None:
                        continue
                    if edges[direction] and not self._bumped(map_id, origin[0], origin[1], direction):
                        value = 0.5
                        break
                    value = -1.0
                out.append(value)
        return out

    def potential(self, state) -> float:
        """Shaping potential in [-1, 0]: minus the known walking distance to the nearest
        unvisited open tile, over FRONTIER_DIST_SCALE. -1 when nothing known is left to try."""
        direction, dist = self.frontier(state)
        if direction is None:
            return -1.0
        return -min(dist, FRONTIER_DIST_SCALE) / FRONTIER_DIST_SCALE

    def features(self, state, *, in_menu: bool = False) -> np.ndarray:
        """NAV_DIM floats. ``in_menu`` (battle or dialogue) clears the movement mask."""
        blocked = [0.0] * 4 if in_menu else self.blocked_flags(state)
        last = [0.0] * 5
        if 0 <= self.last_action < 5:
            last[self.last_action] = 1.0
        direction, dist = self.frontier(state)
        front = [1.0 if direction == d else 0.0 for d in DIRS]
        dist_feature = min(dist, FRONTIER_DIST_SCALE) / FRONTIER_DIST_SCALE if direction else 0.0
        vec = blocked + last + [1.0 if self.last_moved else 0.0] + front + [dist_feature] + self.grid(state)
        return np.asarray(vec, dtype=np.float64)
