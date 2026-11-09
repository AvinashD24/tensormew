"""OPTIONAL tile-level goal potential for the minimal story path (OFF by default; never part of the observation).

The graph holds every walkable tile of the route maps (New Bark -> Route 29 -> Cherrygrove -> Route 30 -> Route 31
-> Violet gatehouse -> Violet City -> Violet Gym), read from the ROM, with single-step edges (map connections and
warps, ledge jumps and cut trees respected) and a BFS distance field to the single goal: the tile in front of Falkner.

``route_potential`` = -(steps to Falkner). The trainer only uses it when ``TrainConfig.goal_potential`` is True,
as a reward ``ROUTE_PROGRESS_WEIGHT * (Phi' - Phi)`` for ablations ("is the oracle shaping better?"). The default
runs do not use it, and the policy is never told the direction or distance (the old ``route_features`` and
``next_direction`` observation was removed: it was an oracle, not reinforcement learning).

``path_progress`` / ``path_total`` expose the same distance field as "steps of the 309-step path already covered" for the
``long_v4`` best-ever reward (``src.path_progress``); that reward is also reward shaping only, never an observation.
One step is one action: a ledge jump (two tiles) is one step, so the 309 steps cover 311 tiles (see ``_HOP_DIRS``).
Before the ledge fix the graph had no jump edges, no sideways walking along ledge rows and a 319-step route.

Buildings next to the route (houses, marts, Pokecenters) are included as dead-end spurs so entering one lowers Phi
by the walk back out. The graph is built lazily, once per process.

EGG ERRAND (``egg_errand_route``, ``long_v8``). The 309-step route above is the no-egg route and is UNREACHABLE at game
start: Route 30 is cut in half by the one-tile choke (5, 25), occupied by the Joey event (5, 26), two Rattata sprites
(5, 25) / (5, 24) and trainer Mikey (5, 23) until Elm has the Mystery Egg (hide flag 1812 is set together with
EVENT_GAVE_MYSTERY_EGG_TO_ELM by the script at ElmsLab.asm:345).  The real order is therefore three PHASES chosen from
event flags in RAM, each with its own BFS distance field and a goal tile:

  phase 0 (egg not yet received):   New Bark -> Cherrygrove -> Route 30 right-hand corridor -> Mr. Pokemon's house
                                    (the choke tiles are BLOCKED), goal = ``EGG_RECEIVE_GOAL``;
  phase 1 (egg received, not given): back through Route 30 / Cherrygrove / Route 29 to Elm's lab, goal =
                                    ``EGG_DELIVER_GOAL`` (choke still blocked, it is not on the way);
  phase 2 (egg given to Elm):       the opened graph, goal = the tile in front of Falkner (the 309-step route plus the walk
                                    from Elm's lab back to the New Bark start tile).

NPCs ON THE ROUTE (checked tile by tile in the emulator; only the sprites that cut Route 30 are modelled as blockers above).
Not modelled, all harmless for the distance except the first: the Guide Gent stands on (26,3) (32, 6) until the Cherrygrove tour
is taken (flag 1790), so with him there the real shortest way is the row-7 bypass, 2 steps longer (329 / 311 instead of 327 / 309;
the bypass costs at most a one-tile retreat in the BFS field); Mikey (5, 23) and Don (1, 7) on Route 30 stand on path tiles but
the equally short column-4 / column-2 detours cost nothing; trainers force a battle when the path crosses their sight line
(Mikey at (5, 24), Don at (1, 8..10), Abe in the gym at (4, 10) / (5, 10), Rod at (4, 6)); the Route 31 trainer at (21, 13) looks
left along row 13 and never sees the row-12 path.

``path_progress(state, egg=True)`` =``phase_offset + max(0, phase_len - BFS distance to the phase goal)``: continuous at the
phase borders (the goal tile of one phase is the start tile of the next), so the best-ever reward never jumps.
"""

from __future__ import annotations

import threading
from collections import deque

from src.collision import (
    COLL_HOP_DOWN,
    COLL_HOP_DOWN_LEFT,
    COLL_HOP_DOWN_RIGHT,
    COLL_HOP_LEFT,
    COLL_HOP_RIGHT,
    COLL_HOP_UP,
    COLL_HOP_UP_LEFT,
    COLL_HOP_UP_RIGHT,
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

# --- the egg errand (phased route) ---------------------------------------------------------------------------------
# Event flag ids (src.story_flags; 1812 = EVENT_ROUTE_30_BATTLE, the Route 30 blockers' hide flag, set together with 31).
FLAG_EGG_RECEIVED = 30      # EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON
FLAG_EGG_DELIVERED = 31     # EVENT_GAVE_MYSTERY_EGG_TO_ELM
FLAG_ROUTE30_OPEN = 1812    # EVENT_ROUTE_30_BATTLE (hide flag of Joey / the Rattatas on the choke)
# NPC sprites standing on the one-tile choke of Route 30 while flag 1812 is clear (Joey's event, two Rattatas, Mikey).
NPC_BLOCKERS = tuple(((26, 1), 5, y) for y in (23, 24, 25, 26))
# Mr. Pokemon's house (26, 10): the tile in front of Mr. Pokemon (3, 5) where the egg is handed over.
EGG_RECEIVE_GOAL = ((26, 10), 3, 6)
# Elm's lab (24, 5): the tile in front of Elm (5, 2) where the egg is delivered.
EGG_DELIVER_GOAL = ((24, 5), 5, 3)
EGG_PHASE_COUNT = 3

DIR_NAMES = tuple(_DIRS)           # ("up", "down", "left", "right"), the feature order
_CARPET_DIR = {
    COLL_WARP_CARPET_DOWN: "down",
    COLL_WARP_CARPET_LEFT: "left",
    COLL_WARP_CARPET_UP: "up",
    COLL_WARP_CARPET_RIGHT: "right",
}
_CONN_SIDE = {"up": "north", "down": "south", "left": "west", "right": "east"}

# Ledges (Pokemon Crystal, verified in the emulator on Route 29 and in the ROM collision grids of every route map):
# a hop tile is a normal walkable tile you may step onto from above or from the side and walk along sideways. While
# STANDING on it, pressing a hop direction of the tile jumps TWO tiles that way (over the cliff-face wall tile that the
# map always places on the far side) and lands there. Nothing can be walked up a ledge: the tile beyond a hop tile is a
# wall, so the way back is only through a gap in the ledge row, and entering a hop tile against its own hop
# direction is also refused here as a belt-and-braces rule. One jump is one action, so BFS distance drops by one.
_HOP_DIRS = {
    COLL_HOP_DOWN: ("down",),
    COLL_HOP_UP: ("up",),
    COLL_HOP_LEFT: ("left",),
    COLL_HOP_RIGHT: ("right",),
    COLL_HOP_DOWN_LEFT: ("down", "left"),
    COLL_HOP_DOWN_RIGHT: ("down", "right"),
    COLL_HOP_UP_LEFT: ("up", "left"),
    COLL_HOP_UP_RIGHT: ("up", "right"),
}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
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
        self._build_egg_phases()

    def _build_egg_phases(self) -> None:
        """BFS tables of the three egg-errand phases (see the module docstring); ``phase_dist`` stays None if a goal is missing."""
        self.phase_dist = self.phase_len = self.phase_off = None
        self.phase_total = None
        if any(n not in self.fwd for n in (EGG_RECEIVE_GOAL, EGG_DELIVER_GOAL)):
            return
        blocked = frozenset(NPC_BLOCKERS)
        rev: dict[Node, list] = {}
        for src, outs in self.fwd.items():
            if src in blocked:
                continue
            for dst in outs:
                if dst is not None and dst not in blocked:
                    rev.setdefault(dst, []).append(src)
        d0 = self._bfs(EGG_RECEIVE_GOAL, rev)
        d1 = self._bfs(EGG_DELIVER_GOAL, rev)
        d2 = self.dist[0]
        lens = (d0.get(PATH_START), d1.get(EGG_RECEIVE_GOAL), d2.get(EGG_DELIVER_GOAL))
        if any(n is None for n in lens):
            return
        self.phase_dist = (d0, d1, d2)
        self.phase_len = tuple(int(n) for n in lens)
        self.phase_off = (0, self.phase_len[0], self.phase_len[0] + self.phase_len[1])
        self.phase_total = sum(self.phase_len)

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
        if direction in _HOP_DIRS.get(m.cells[(x, y)], ()):
            land = m.cells.get((x + 2 * dx, y + 2 * dy))   # ledge jump: two tiles, same map, onto a walkable tile
            if land is not None and (mid, x + 2 * dx, y + 2 * dy) not in self.warp_at and walkable(land):
                return (mid, x + 2 * dx, y + 2 * dy)
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
        hops = _HOP_DIRS.get(tile)
        if hops is not None:
            # A ledge tile can be entered from above and from the side, never against its own hop direction.
            if _OPPOSITE[direction] in hops or not walkable(tile):
                return None
        elif not walkable(tile, direction):
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


def egg_phase(state) -> int:
    """Phase of the egg errand from the event flags of ``state``: 0 before the egg, 1 egg in hand, 2 egg given to Elm
    (hide flag 1812 or EVENT_GAVE_MYSTERY_EGG_TO_ELM set).  A state without ``has_event`` is phase 0."""
    fn = getattr(state, "has_event", None)
    if not callable(fn):
        return 0
    try:
        if fn(FLAG_ROUTE30_OPEN) or fn(FLAG_EGG_DELIVERED):
            return 2
        return 1 if fn(FLAG_EGG_RECEIVED) else 0
    except Exception:
        return 0


def egg_phase_info() -> dict | None:
    """``{"len": (l0, l1, l2), "off": (0, l0, l0 + l1), "total": l0 + l1 + l2}`` of the phased route, or None without the ROM."""
    g = _graph()
    if g is None or g.phase_dist is None:
        return None
    return {"len": g.phase_len, "off": g.phase_off, "total": g.phase_total}


def path_total(egg: bool = False) -> float | None:
    """Length of the minimal story path (BFS steps from the New Bark start tile to the tile in front of Falkner).

    With ``egg=True`` the phased egg-errand route: the sum of the three phase lengths (``egg_phase_info``)."""
    g = _graph()
    if g is None:
        return None
    if egg:
        return None if g.phase_total is None else float(g.phase_total)
    return g.goal_distance(0, PATH_START)


def _egg_progress(g, state) -> float | None:
    if g.phase_dist is None:
        return None
    phase = egg_phase(state)
    d = g.phase_dist[phase].get(_node(state))
    if d is None:
        return None
    return float(g.phase_off[phase] + max(0, g.phase_len[phase] - d))


def path_progress(state, egg: bool = False) -> float | None:
    """How far along the minimal story path the position is: ``path_total - steps_to_Falkner``, clamped to >= 0.

    Uses the EXACT BFS distance over the ROM graph (warps included), so every tile that is not on the shortest path
    counts as the progress of the path tile it branches from, minus the walk back: a dead-end corridor or a building
    spur never scores above its junction. None when the tile is off the graph or has no path to the goal. Used only
    by the ``path_progress`` reward (``src.path_progress``); never an observation.
    """
    g = _graph()
    if g is None:
        return None
    if egg:
        return _egg_progress(g, state)   # phase-aware: offset + (phase length - distance to the phase goal)
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
