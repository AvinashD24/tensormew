"""Route from Mr. Pokémon's house back to Elm's Lab, read from the ROM's maps.

Only used while the Mystery Egg is in hand and not yet delivered. Distances are shortest-path
steps over the real collision grids of Route 30, Cherrygrove, Route 29 and New Bark Town,
joined by the ROM's map connections, so every step toward Elm's Lab raises the potential by the
same amount (no hand-placed coordinates, no dead-end maxima). Any shortest path crosses the
Cherrygrove rival trigger (x=33, y=6..7): the only way east is that two-row corridor.

``egg_return_potential`` returns None for maps it does not cover, so callers fall back to the
older per-map potentials.
"""

from __future__ import annotations

import re
from collections import deque
from functools import lru_cache
from pathlib import Path

from src.mover import walkable

ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = ROOT / "rom" / "pokemon_crystal.gbc"
SYM_PATH = ROOT / "rom" / "pokemon_crystal.sym"

# (group, number) -> symbol prefix. All four use the Johto outdoor tileset.
CHAIN = {
    (26, 1): "Route30",
    (26, 3): "CherrygroveCity",
    (24, 3): "Route29",
    (24, 4): "NewBarkTown",
}
GOAL_INTERIOR = (24, 5)  # Elm's Lab: walking through its door ends the stage
COLL_SYMBOL = "TilesetJohtoColl"

# Potential scale, kept consistent with the neighbouring stages so the flips are near-continuous:
# just outside Mr. Pokémon's house ~ the value of the get-egg stage, Elm's Lab door ~ stage 3 start.
PHI_LAB_DOOR = 135.0
PHI_HOUSE_EXIT = 96.0
HOUSE_EXIT_NODE = ((26, 1), 17, 6)

DOOR_DISCOUNT = 1.5        # interiors and doors sit just below the tile outside them
RIVAL_X = 33               # Cherrygrove: rival triggers on x=33, y=6..7
HEAL_BELOW = 0.5           # detour to the Pokécenter before the rival under this HP fraction
PC_MAP = (26, 5)
PC_DOOR_OUTSIDE = ((26, 3), 29, 4)

_DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _signed(b: int) -> int:
    return b - 256 if b > 127 else b


class _Rom:
    def __init__(self):
        self.rom = ROM_PATH.read_bytes()
        self.sym: dict[str, tuple[int, int]] = {}
        for line in SYM_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"([0-9a-f]+):([0-9a-f]+) (\S+)", line)
            if m:
                self.sym[m.group(3)] = (int(m.group(1), 16), int(m.group(2), 16))

    def off(self, bank: int, addr: int) -> int:
        return bank * 0x4000 + (addr - 0x4000) if bank else addr

    def grid(self, prefix: str):
        b, a = self.sym[prefix + "_MapAttributes"]
        o = self.off(b, a)
        h, w = self.rom[o + 1], self.rom[o + 2]
        bb, pp = self.rom[o + 3], self.rom[o + 4] | self.rom[o + 5] << 8
        blocks = self.rom[self.off(bb, pp):self.off(bb, pp) + h * w]
        cb, ca = self.sym[COLL_SYMBOL]
        co = self.off(cb, ca)
        cells = {}
        for by in range(h):
            for bx in range(w):
                blk = blocks[by * w + bx]
                for k in range(4):
                    cells[(bx * 2 + (k & 1), by * 2 + (k >> 1))] = self.rom[co + blk * 4 + k]
        conns = {}
        flags = self.rom[o + 11]
        p = o + 12
        for bit, name in ((8, "north"), (4, "south"), (2, "west"), (1, "east")):
            if flags & bit:
                c = self.rom[p:p + 12]
                p += 12
                conns[name] = (c[0], c[1], _signed(c[8]), _signed(c[9]))
        return w * 2, h * 2, cells, conns

    def warps(self, prefix: str):
        b, a = self.sym[prefix + "_MapEvents"]
        o = self.off(b, a)
        n = self.rom[o + 2]
        return [tuple(self.rom[o + 3 + 5 * k:o + 3 + 5 * k + 5]) for k in range(n)]  # y, x, id, grp, num


class ReturnNav:
    def __init__(self):
        rom = _Rom()
        self.size: dict[tuple[int, int], tuple[int, int]] = {}
        self.cells: dict[tuple[int, int], dict] = {}
        self.conns: dict[tuple[int, int], dict] = {}
        for mid, prefix in CHAIN.items():
            w, h, cells, conns = rom.grid(prefix)
            self.size[mid], self.cells[mid], self.conns[mid] = (w, h), cells, conns

        self.doors: set[tuple] = set()          # non-goal warp tiles: leave the chain, never routed through
        self.interior_outside: dict[tuple[int, int], tuple] = {}   # interior map -> node just outside its door
        self.goal = None
        for mid, prefix in CHAIN.items():
            for y, x, _wid, grp, num in rom.warps(prefix):
                node = (mid, x, y)
                if (grp, num) == GOAL_INTERIOR:
                    self.goal = node
                else:
                    self.doors.add(node)
                self.interior_outside[(grp, num)] = (mid, x, y + 1)
        if self.goal is None:
            raise RuntimeError("Elm's Lab door not found")

        self.rev: dict[tuple, list] = {}
        for mid in CHAIN:
            w, h = self.size[mid]
            for (x, y) in self.cells[mid]:
                src = (mid, x, y)
                if not self._standable(src):
                    continue
                for d, (dx, dy) in _DIRS.items():
                    dst = self._step(mid, x + dx, y + dy, d)
                    if dst is not None:
                        self.rev.setdefault(dst, []).append(src)

        self.dist = self._bfs(self.goal)
        self._pc_dist = None
        ref = self.dist.get(HOUSE_EXIT_NODE)
        if not ref:
            raise RuntimeError("no route from Mr. Pokémon's house to Elm's Lab")
        self.scale = (PHI_LAB_DOOR - PHI_HOUSE_EXIT) / ref
        self.max_dist = max(self.dist.values())

    # --- graph ---------------------------------------------------------------------------------
    def _standable(self, node) -> bool:
        mid, x, y = node
        if node in self.doors:
            return False
        return walkable(self.cells[mid][(x, y)])

    def _step(self, mid, nx, ny, direction):
        """Node reached by moving in `direction` onto (nx, ny) of map `mid`, or None if blocked."""
        w, h = self.size[mid]
        if 0 <= nx < w and 0 <= ny < h:
            target = (mid, nx, ny)
        else:
            conn = self.conns[mid].get({"up": "north", "down": "south", "left": "west", "right": "east"}[direction])
            if conn is None or (conn[0], conn[1]) not in CHAIN:
                return None
            omid = (conn[0], conn[1])
            ow, oh = self.size[omid]
            if direction == "up":
                tx, ty = nx + conn[3], oh - 1
            elif direction == "down":
                tx, ty = nx + conn[3], 0
            elif direction == "left":
                tx, ty = ow - 1, ny + conn[2]
            else:
                tx, ty = 0, ny + conn[2]
            if not (0 <= tx < ow and 0 <= ty < oh):
                return None
            target = (omid, tx, ty)
        tmid, tx, ty = target
        if target in self.doors:
            return None
        if not walkable(self.cells[tmid][(tx, ty)], direction):
            return None
        return target

    def _bfs(self, goal) -> dict:
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            node = queue.popleft()
            for prev in self.rev.get(node, ()):
                if prev not in dist:
                    dist[prev] = dist[node] + 1
                    queue.append(prev)
        return dist

    def _to_pc(self) -> dict:
        """Distances to Cherrygrove's Pokécenter door, with that door treated as the goal."""
        if self._pc_dist is None:
            door = ((26, 3), PC_DOOR_OUTSIDE[1], PC_DOOR_OUTSIDE[2] - 1)
            saved = self.doors
            self.doors = saved - {door}
            try:
                rev: dict[tuple, list] = {}
                for mid in CHAIN:
                    for (x, y) in self.cells[mid]:
                        src = (mid, x, y)
                        if not self._standable(src):
                            continue
                        for d, (dx, dy) in _DIRS.items():
                            dst = self._step(mid, x + dx, y + dy, d)
                            if dst is not None:
                                rev.setdefault(dst, []).append(src)
                dist = {door: 0}
                queue = deque([door])
                while queue:
                    node = queue.popleft()
                    for prev in rev.get(node, ()):
                        if prev not in dist:
                            dist[prev] = dist[node] + 1
                            queue.append(prev)
                self._pc_dist = dist
            finally:
                self.doors = saved
        return self._pc_dist

    # --- potential -----------------------------------------------------------------------------
    def phi_node(self, node) -> float:
        d = self.dist.get(node)
        if d is None:
            d = self.max_dist + 5
        return PHI_LAB_DOOR - self.scale * d

    def potential(self, mid, x, y, hp_fraction: float = 1.0) -> float | None:
        node = (mid, x, y)
        if mid in CHAIN:
            heal = hp_fraction < HEAL_BELOW and mid == (26, 3) and x < RIVAL_X
            if heal:
                pc = self._to_pc()
                pc_node = PC_DOOR_OUTSIDE
                base = self.phi_node(pc_node)
                if node == ((26, 3), PC_DOOR_OUTSIDE[1], PC_DOOR_OUTSIDE[2] - 1):
                    return base + 0.5
                d = pc.get(node)
                if d is not None:
                    return base - self.scale * (d - 1)
            if node in self.doors:
                outside = self.interior_outside_for_door(node)
                return self.phi_node(outside) - DOOR_DISCOUNT if outside else None
            if node == self.goal:
                return PHI_LAB_DOOR
            return self.phi_node(node)
        outside = self.interior_outside.get(mid)
        if outside is None:
            return None
        if mid == GOAL_INTERIOR:
            return PHI_LAB_DOOR + 3.0
        base = self.phi_node(outside)
        if mid == PC_MAP and hp_fraction < 0.999:
            return base + 0.5 - 0.1 * (abs(x - 3.0) * 1.5 + abs(y - 3.0))
        return base - DOOR_DISCOUNT + 0.9 * min(1.0, max(0.0, y / 7.0))

    def interior_outside_for_door(self, door_node):
        mid, x, y = door_node
        return (mid, x, y + 1) if (mid, x, y + 1) in self.dist or (mid, x, y + 1) in self.rev or True else None


@lru_cache(maxsize=1)
def _nav() -> ReturnNav | None:
    try:
        return ReturnNav()
    except (OSError, KeyError, RuntimeError):
        return None


def hp_fraction(state) -> float:
    hp = getattr(state, "party_hp", ()) or ()
    cur = sum(p[0] for p in hp)
    mx = sum(p[1] for p in hp)
    return cur / mx if mx > 0 else 1.0


def egg_return_potential(state) -> float | None:
    nav = _nav()
    if nav is None:
        return None
    mid = (int(getattr(state, "map_group", 0)), int(getattr(state, "map_number", 0)))
    return nav.potential(mid, int(getattr(state, "x", 0)), int(getattr(state, "y", 0)), hp_fraction(state))


def egg_return_active(state) -> bool:
    has_event = getattr(state, "has_event", None)
    return bool(
        callable(has_event)
        and has_event("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
        and not has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
    )


def egg_return_direction(state) -> str | None:
    """Next step ("up"/"down"/"left"/"right") along the shortest route back to Elm's Lab, or None.

    None when the route does not cover this map (e.g. inside a building) or the tile has no route.
    Every step toward the lab strictly lowers the BFS distance, so following it cannot dead-end on a
    wall or ledge the way a straight push east does.
    """
    nav = _nav()
    if nav is None:
        return None
    mid = (int(getattr(state, "map_group", 0)), int(getattr(state, "map_number", 0)))
    if mid not in CHAIN:
        return None
    x, y = int(getattr(state, "x", 0)), int(getattr(state, "y", 0))
    here = nav.dist.get((mid, x, y))
    if here is None or here == 0:
        return None
    best, best_d = None, here
    for d, (dx, dy) in _DIRS.items():
        dst = nav._step(mid, x + dx, y + dy, d)
        if dst is None:
            continue
        dd = nav.dist.get(dst)
        if dd is not None and dd < best_d:
            best, best_d = d, dd
    if best is None and nav.goal is not None:
        # The door tile is a warp, not a standable node: step onto it when it is the neighbour
        for d, (dx, dy) in _DIRS.items():
            if (mid, x + dx, y + dy) == nav.goal:
                return d
    return best
