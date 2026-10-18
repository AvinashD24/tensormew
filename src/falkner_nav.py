"""Route from Elm's Lab all the way to Falkner's Gym in Violet City, read from ROM maps.

Active after delivering the Mystery Egg to Professor Elm, until Falkner is defeated (Zephyr Badge).
Provides deterministic shortest-path steps across:
1. Elm's Lab (24, 5) -> exit through south carpet after receiving aide's Poké Balls
2. New Bark Town (24, 4) -> west to Route 29 entrance at (0, 8..9)
3. Route 29 (24, 3) -> west to Cherrygrove City entrance at (0, 8..9)
4. Cherrygrove City (26, 3) -> north to Route 30 connection at (16..17, 0)
5. Route 30 (26, 1) -> north along the unblocked west corridor to Route 31 at (6, 0)
6. Route 31 (26, 2) -> west to Violet Gatehouse entrance at (5, 6..7)
7. Route 31 Violet Gate (26, 11) -> west to Violet City exit at (0, 4..5)
8. Violet City (10, 5) -> west and north to Violet Gym entrance at (18, 18)
9. Violet Gym (10, 7) -> S-curve bridge navigation to Leader Falkner at (4..5, 2..3) and challenge!
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

from src.mover import walkable
from src.return_nav import _DIRS, _Rom

# Map IDs
MAP_ELM_LAB = (24, 5)
MAP_NEW_BARK = (24, 4)
MAP_ROUTE_29 = (24, 3)
MAP_CHERRYGROVE = (26, 3)
MAP_ROUTE_30 = (26, 1)
MAP_ROUTE_31 = (26, 2)
MAP_VIOLET_GATE = (26, 11)
MAP_VIOLET_CITY = (10, 5)
MAP_VIOLET_GYM = (10, 7)


def _bfs_grid(cells: dict[tuple[int, int], int], w: int, h: int, targets: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Compute BFS distance field to any target in targets over walkable cells."""
    dist: dict[tuple[int, int], int] = {}
    queue = deque()
    for t in targets:
        dist[t] = 0
        queue.append(t)
    while queue:
        curr = queue.popleft()
        cx, cy = curr
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (cx + dx, cy + dy)
            if 0 <= nxt[0] < w and 0 <= nxt[1] < h and nxt not in dist:
                tile = cells.get(nxt, 0x07)
                if walkable(tile):
                    dist[nxt] = dist[curr] + 1
                    queue.append(nxt)
    return dist


class FalknerNav:
    def __init__(self):
        rom = _Rom()

        # 1. Route 29 westbound to (0, 8..9)
        w29, h29, cells29, _ = rom.grid("Route29")
        self.dist_r29 = _bfs_grid(cells29, w29, h29, [(0, 8), (0, 9)])

        # 2. Cherrygrove City to Route 30 north connection at (16, 0), (17, 0)
        wC, hC, cellsC, _ = rom.grid("CherrygroveCity")
        self.dist_cherry = _bfs_grid(cellsC, wC, hC, [(16, 0), (17, 0)])

        # 3. Route 30 northbound to Route 31 at (6, 0), (7, 0)
        w30, h30, cells30, _ = rom.grid("Route30")
        cells30_nav = dict(cells30)
        cells30_nav[(5, 23)] = 0x07  # Youngster Joey stands at (5, 23); detour west through column 4
        self.dist_r30 = _bfs_grid(cells30_nav, w30, h30, [(4, 0), (5, 0), (6, 0), (7, 0)])

        # 4. Route 31 westbound to Gatehouse door at (5, 6), (5, 7)
        w31, h31, cells31, _ = rom.grid("Route31")
        self.dist_r31 = _bfs_grid(cells31, w31, h31, [(5, 6), (5, 7)])

        # 5. Violet City to Violet Gym entrance at (18, 18)
        wV, hV, cellsV, _ = rom.grid("VioletCity")
        self.dist_violet = _bfs_grid(cellsV, wV, hV, [(18, 18)])

    def step_on_field(self, dist_field: dict[tuple[int, int], int], x: int, y: int) -> str | None:
        """Choose adjacent walkable tile with strictly lowest BFS distance."""
        best_d: str | None = None
        min_dist = dist_field.get((x, y), 999999)
        for direction, (dx, dy) in _DIRS.items():
            neighbor = (x + dx, y + dy)
            d = dist_field.get(neighbor)
            if d is not None and d < min_dist:
                min_dist = d
                best_d = direction
        return best_d


@lru_cache(maxsize=1)
def _get_falkner_nav() -> FalknerNav:
    return FalknerNav()


def violet_gym_step(x: int, y: int, facing: str) -> str:
    """Deterministic navigation along the elevated S-curve bridge in Violet Gym (10, 7)."""
    # Standing in front of Leader Falkner at (4..5, 2..3)
    if x in (4, 5) and y in (2, 3):
        if facing == "up":
            return "a"
        return "up"

    # Bottom straight section (entrance y=15 up to y=11)
    if y > 10:
        if x < 4:
            return "right"
        if x > 5:
            return "left"
        return "up"

    # East turn (y=9..10): move east onto the right-hand bridge
    if y in (9, 10):
        if x < 7:
            return "right"
        return "up"

    # West crossover (y=7..8): move west across to the left-hand bridge
    if y in (7, 8):
        if x > 2:
            return "left"
        return "up"

    # Return to center platform (y=4..6): move right back to center column 4..5
    if y in (4, 5, 6):
        if x < 4:
            return "right"
        if x > 5:
            return "left"
        return "up"

    return "up"


def falkner_nav_direction(state) -> str | None:
    """Returns the next deterministic move or button toward beating Falkner, or None if outside scope."""
    has_ev = getattr(state, "has_event", None)
    if callable(has_ev):
        if has_ev("EVENT_BEAT_FALKNER"):
            return None  # Gym badge already earned
        if not has_ev("EVENT_GAVE_MYSTERY_EGG_TO_ELM"):
            return None  # Still on egg delivery errand

    map_id = getattr(state, "map_id", None)
    x = getattr(state, "x", None)
    y = getattr(state, "y", None)
    facing = getattr(state, "facing", "down")

    if map_id is None or x is None or y is None:
        return None

    # 1. Inside Elm's Lab (24, 5) after delivering egg
    if map_id == MAP_ELM_LAB:
        if y == 11 and x in (4, 5):
            return "down"  # Step through exit warp mat
        if x < 5:
            return "right"
        if x > 5:
            return "left"
        return "down"

    # 2. In New Bark Town (24, 4) -> exit west into Route 29
    if map_id == MAP_NEW_BARK:
        if x == 0 and y in (8, 9):
            return "left"  # Step through western border into Route 29
        if y < 8:
            return "down"
        return "left"

    nav = _get_falkner_nav()

    # 3. On Route 29 (24, 3) -> head west to Cherrygrove City
    if map_id == MAP_ROUTE_29:
        if x == 0 and y in (8, 9):
            return "left"  # Step through western gate into Cherrygrove City
        step = nav.step_on_field(nav.dist_r29, x, y)
        if step is not None:
            return step
        # Fallback corridor
        return "left"

    # 4. In Cherrygrove City (26, 3) -> head north to Route 30
    if map_id == MAP_CHERRYGROVE:
        if y == 0 and x in (16, 17):
            return "up"  # Step through northern border into Route 30
        step = nav.step_on_field(nav.dist_cherry, x, y)
        if step is not None:
            return step
        if x < 16:
            return "right"
        if x > 17:
            return "left"
        return "up"

    # 5. On Route 30 (26, 1) -> head north along west path to Route 31
    if map_id == MAP_ROUTE_30:
        if y == 0 and x in (4, 5, 6, 7):
            return "up"  # Step through northern border into Route 31
        step = nav.step_on_field(nav.dist_r30, x, y)
        if step is not None:
            return step
        return "up"

    # 6. On Route 31 (26, 2) -> head west to Violet Gatehouse entrance
    if map_id == MAP_ROUTE_31:
        if x <= 5 and y in (6, 7):
            return "left"  # Step through doorway into Route 31 Violet Gate
        step = nav.step_on_field(nav.dist_r31, x, y)
        if step is not None:
            return step
        return "left"

    # 7. In Route 31 Violet Gate (26, 11) -> head west into Violet City
    if map_id == MAP_VIOLET_GATE:
        if x == 0 and y in (4, 5):
            return "left"  # Step through west doorway into Violet City
        return "left"

    # 8. In Violet City (10, 5) -> head to Violet Gym at (18, 18)
    if map_id == MAP_VIOLET_CITY:
        if x == 18 and y == 18:
            return "up"  # Enter Violet Gym doorway
        step = nav.step_on_field(nav.dist_violet, x, y)
        if step is not None:
            return step
        if x > 18:
            return "left"
        if x < 18:
            return "right"
        if y > 18:
            return "up"
        return "down"

    # 9. Inside Violet Gym (10, 7) -> navigate S-curve bridge and challenge Falkner
    if map_id == MAP_VIOLET_GYM:
        return violet_gym_step(x, y, facing)

    return None
