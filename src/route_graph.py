"""OPTIONAL tile-level goal potential for the minimal story path (OFF by default; never part of the observation).

The graph holds every walkable tile of the route maps (New Bark -> Route 29 -> Cherrygrove -> Route 30 -> Route 31
-> Violet gatehouse -> Violet City -> Violet Gym), read from the ROM, with single-step edges (map connections and
warps, ledges and cut trees respected) and a BFS distance field to the single goal: the tile in front of Falkner.

``route_potential`` = -(steps to Falkner). The trainer only uses it when ``TrainConfig.goal_potential`` is True,
as a reward ``ROUTE_PROGRESS_WEIGHT * (Phi' - Phi)`` for ablations ("is the oracle shaping better?"). The default
runs do not use it, and the policy is never told the direction or distance (the old ``route_features`` and
``next_direction`` observation was removed: it was an oracle, not reinforcement learning).

``path_progress`` / ``path_total`` expose the same distance field as "tiles of the 319-step path already covered" for the
``long_v4`` best-ever reward (``src.path_progress``); that reward is also reward shaping only, never an observation.

Buildings next to the route (houses, marts, Pokecenters) are included as dead-end spurs so entering one lowers Phi
by the walk back out. The graph is built lazily, once per process.
"""

from __future__ import annotations

import threading
from collections import deque

from src.collision import (
    COLL_WARP_CARPET_DOWN,
    COLL_WARP_CARPET_LEFT,
    COLL_WARP_CARPET_RIGHT,
    COLL_WARP_CARPET_UP,
)
from src.mover import walkable
from src.return_nav import ROM_PATH, SYM_PATH, _DIRS, _Rom, _signed

STAGE_COUNT = 1
UNREACHABLE_EXTRA = 5   # tiles on the graph with no path to the goal sit this far past the farthest reachable one

# Maps the route walks through. Outdoor maps join through connections; houses and shops are loaded as spurs.
OUTDOOR_MAPS = ((24, 4), (24, 3), (26, 3), (26, 1), (26, 2), (10, 5))
ROUTE_MAPS = OUTDOOR_MAPS + ((26, 11), (10, 7))

# The one goal as (map, x, y): the tile in front of Falkner, who stands at (5, 1).
GOALS = (
    ((10, 7), 5, 2),
)

# Where the path starts as (map, x, y): states/new_bark_outside.state (outside New Bark Town).
PATH_START = ((24, 4), 3, 9)

DIR_NAMES = tuple(_DIRS)           # ("up", "down", "left", "right"), the feature order
_CARPET_DIR = {
    COLL_WARP_CARPET_DOWN: "down",
    COLL_WARP_CARPET_LEFT: "left",
    COLL_WARP_CARPET_UP: "up",
    COLL_WARP_CARPET_RIGHT: "right",
}
_CONN_SIDE = {"up": "north", "down": "south", "left": "west", "right": "east"}
_HEADER_SIZE = 9
_TILESET_SIZE = 15

Node = tuple  # ((group, number), x, y)


class _MapData:
    __slots__ = ("mid", "w", "h", "cells", "conns", "warps")

    def __init__(self, mid, w, h, cells, conns, warps):
        self.mid, self.w, self.h = mid, w, h
        self.cells, self.conns, self.warps = cells, conns, warps


class _Graph:
    def __init__(self):
        rom = _Rom()
        self.rom = rom
        attrs_name: dict[tuple[int, int], str] = {}
        for name, loc in rom.sym.items():
            if name.endswith("_MapAttributes"):
                attrs_name.setdefault(loc, name[: -len("_MapAttributes")])
        self._attrs_name = attrs_name

        self.maps: dict[tuple[int, int], _MapData] = {}
        for mid in ROUTE_MAPS:
            self._load(mid)
        spurs = []
        town_groups = {grp for grp, _num in OUTDOOR_MAPS}   # houses and shops share their town's map group;
        for mid in OUTDOOR_MAPS:                            # caves and towers (other groups) stay off the graph
            for _y, _x, _wid, grp, num in self.maps[mid].warps:
                dest = (grp, num)
                if grp in town_groups and dest not in self.maps and dest not in spurs:
                    spurs.append(dest)
        for mid in spurs:
            self._load(mid)

        self.warp_at: dict[Node, tuple] = {}
        for mid, m in self.maps.items():
            for y, x, wid, grp, num in m.warps:
                dest = (grp, num)
                dm = self.maps.get(dest)
                if dm is None or not 0 < wid <= len(dm.warps):
                    continue
                dy, dx = dm.warps[wid - 1][0], dm.warps[wid - 1][1]
                if (dx, dy) in dm.cells:
                    self.warp_at[(mid, x, y)] = (dest, dx, dy)

        # fwd[node] = landing node for (up, down, left, right), None where the step is blocked
        self.fwd: dict[Node, tuple] = {}
        rev: dict[Node, list] = {}
        for mid, m in self.maps.items():
            for (x, y), tile in m.cells.items():
                if not walkable(tile):
                    continue
                src = (mid, x, y)
                outs = tuple(self._step(src, d) for d in DIR_NAMES)
                self.fwd[src] = outs
                for dst in outs:
                    if dst is not None:
                        rev.setdefault(dst, []).append(src)

        self.goals: list[Node] = []
        for g in GOALS:
            if g not in self.fwd:
                raise RuntimeError(f"route goal {g} is not a walkable tile")
            self.goals.append(g)
        self.dist = [self._bfs(g, rev) for g in self.goals]
        self.max_dist = [max(d.values()) for d in self.dist]
        # tail[s] = steps from the goal of stage s to the end of the whole route
        self.tail = [0.0] * STAGE_COUNT
        for s in range(STAGE_COUNT - 2, -1, -1):
            leg = self.dist[s + 1].get(self.goals[s])
            if leg is None:
                raise RuntimeError(f"no route from goal {s} to goal {s + 1}")
            self.tail[s] = self.tail[s + 1] + leg

    # --- loading ------------------------------------------------------------------------------
    def _load(self, mid):
        rom = self.rom
        grp, num = mid
        pb, pa = rom.sym["MapGroupPointers"]
        po = rom.off(pb, pa) + 2 * (grp - 1)
        table = rom.rom[po] | rom.rom[po + 1] << 8
        h = rom.off(pb, table) + _HEADER_SIZE * (num - 1)
        bank, tileset, attr = rom.rom[h], rom.rom[h + 1], rom.rom[h + 3] | rom.rom[h + 4] << 8
        prefix = self._attrs_name[(bank, attr)]

        tb, ta = rom.sym["Tilesets"]
        te = rom.off(tb, ta) + _TILESET_SIZE * tileset   # entry 0 is the unused Tileset0
        co = rom.off(rom.rom[te + 6], rom.rom[te + 7] | rom.rom[te + 8] << 8)

        o = rom.off(bank, attr)
        mh, mw = rom.rom[o + 1], rom.rom[o + 2]
        bb, bp = rom.rom[o + 3], rom.rom[o + 4] | rom.rom[o + 5] << 8
        blocks = rom.rom[rom.off(bb, bp):rom.off(bb, bp) + mh * mw]
        cells = {}
        for by in range(mh):
            for bx in range(mw):
                blk = blocks[by * mw + bx]
                for k in range(4):
                    cells[(bx * 2 + (k & 1), by * 2 + (k >> 1))] = rom.rom[co + blk * 4 + k]
        conns = {}
        flags = rom.rom[o + 11]
        p = o + 12
        for bit, name in ((8, "north"), (4, "south"), (2, "west"), (1, "east")):
            if flags & bit:
                c = rom.rom[p:p + 12]
                p += 12
                conns[name] = (c[0], c[1], _signed(c[8]), _signed(c[9]))
        self.maps[mid] = _MapData(mid, mw * 2, mh * 2, cells, conns, rom.warps(prefix))

    # --- graph --------------------------------------------------------------------------------
    def _step(self, src: Node, direction: str):
        """Node reached by pressing `direction` on `src`, or None if blocked."""
        mid, x, y = src
        m = self.maps[mid]
        warp = self.warp_at.get(src)
        if warp is not None and _CARPET_DIR.get(m.cells[(x, y)]) == direction:
            return warp  # a carpet warp fires when you press its arrow while standing on it
        dx, dy = _DIRS[direction]
        nx, ny = x + dx, y + dy
        if 0 <= nx < m.w and 0 <= ny < m.h:
            target = (mid, nx, ny)
        else:
            conn = m.conns.get(_CONN_SIDE[direction]) if mid in OUTDOOR_MAPS else None
            if conn is None:
                return None
            omid = (conn[0], conn[1])
            om = self.maps.get(omid)
            if om is None or omid not in OUTDOOR_MAPS:
                return None
            if direction == "up":
                tx, ty = nx + conn[3], om.h - 1
            elif direction == "down":
                tx, ty = nx + conn[3], 0
            elif direction == "left":
                tx, ty = om.w - 1, ny + conn[2]
            else:
                tx, ty = 0, ny + conn[2]
            if not (0 <= tx < om.w and 0 <= ty < om.h):
                return None
            target = (omid, tx, ty)
        tm = self.maps[target[0]]
        tile = tm.cells.get((target[1], target[2]))
        if tile is None:
            return None
        if target in self.warp_at and tile not in _CARPET_DIR:
            return self.warp_at[target]  # stepping onto a door or stairs warps at once
        if not walkable(tile, direction):
            return None
        return target   # a carpet mat is walked onto first; pressing its arrow there is the warp (top of this method)

    @staticmethod
    def _bfs(goal: Node, rev: dict) -> dict:
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            node = queue.popleft()
            nd = dist[node] + 1
            for prev in rev.get(node, ()):
                if prev not in dist:
                    dist[prev] = nd
                    queue.append(prev)
        return dist

    # --- queries ------------------------------------------------------------------------------
    def goal_distance(self, stage: int, node: Node):
        """Steps from `node` to the stage's goal, or None when the tile is unknown or has no path."""
        return self.dist[stage].get(node)

    def _distance_or_nearby(self, stage: int, node: Node):
        """Distance for Phi: exact when known, else one more than the best neighbour, else None."""
        dist = self.dist[stage]
        d = dist.get(node)
        if d is not None:
            return d
        mid, x, y = node
        best = None
        for dx, dy in _DIRS.values():
            nd = dist.get((mid, x + dx, y + dy))
            if nd is not None and (best is None or nd < best):
                best = nd
        return None if best is None else best + 1

    def potential(self, stage: int, node: Node) -> float:
        d = self._distance_or_nearby(stage, node)
        if d is None:
            if node[0] not in self.maps:
                return 0.0
            d = self.max_dist[stage] + UNREACHABLE_EXTRA
        return -(d + self.tail[stage])


_graph_lock = threading.Lock()
_graph_cache: list = []  # [graph or None] once a load was attempted


def _graph() -> _Graph | None:
    if not _graph_cache:
        with _graph_lock:
            if not _graph_cache:
                try:
                    if not (ROM_PATH.exists() and SYM_PATH.exists()):
                        raise OSError("ROM or symbol file missing")
                    _graph_cache.append(_Graph())
                except (OSError, KeyError, IndexError, RuntimeError):
                    _graph_cache.append(None)
    return _graph_cache[0]


# --- public API ---------------------------------------------------------------------------------
def _map_id(state) -> tuple[int, int]:
    return int(getattr(state, "map_group", 0)), int(getattr(state, "map_number", 0))


def _node(state) -> Node:
    return (_map_id(state), int(getattr(state, "x", 0)), int(getattr(state, "y", 0)))


def covers_map(map_id: tuple[int, int]) -> bool:
    g = _graph()
    return g is not None and (int(map_id[0]), int(map_id[1])) in g.maps


def route_potential(state) -> float:
    """Phi = -(steps to Falkner); 0.0 off the graph. Used only by the optional ``goal_potential`` reward."""
    g = _graph()
    if g is None:
        return 0.0
    node = _node(state)
    if node[0] not in g.maps:
        return 0.0
    return float(g.potential(0, node))


def path_total() -> float | None:
    """Length of the minimal story path (BFS steps from the New Bark start tile to the tile in front of Falkner)."""
    g = _graph()
    return None if g is None else g.goal_distance(0, PATH_START)


def path_progress(state) -> float | None:
    """How far along the minimal story path the position is: ``path_total - steps_to_Falkner``, clamped to >= 0.

    Uses the EXACT BFS distance over the ROM graph (warps included), so every tile that is not on the shortest path
    counts as the progress of the path tile it branches from, minus the walk back: a dead-end corridor or a building
    spur never scores above its junction. None when the tile is off the graph or has no path to the goal. Used only
    by the ``path_progress`` reward (``src.path_progress``); never an observation.
    """
    g = _graph()
    if g is None:
        return None
    total = g.goal_distance(0, PATH_START)
    d = g.goal_distance(0, _node(state))
    if total is None or d is None:
        return None
    return float(max(0, total - d))


def goal_remaining(state) -> float | None:
    """Steps to Falkner, or None when the position is off the graph or has no path. Reporting / tests only."""
    g = _graph()
    if g is None:
        return None
    d = g.goal_distance(0, _node(state))
    return None if d is None else float(d)
