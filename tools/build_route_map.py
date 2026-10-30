"""Build route_map.json: stitched Johto overworld, interiors, and the shortest story path.

Walks the ROM collision grids (same sources as src/return_nav.py) as one graph with map
connections and warps, then runs BFS between the story goals:

  Elm's Lab -> Mr. Pokemon -> Cherrygrove rival -> Elm's Lab (egg) -> Falkner

Usage: python tools/build_route_map.py out.json
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.collision import is_cut_tree  # noqa: E402
from src.falkner_nav import violet_gym_step  # noqa: E402
from src.mover import walkable  # noqa: E402
from src.return_nav import _Rom  # noqa: E402

JOHTO = "TilesetJohtoColl"
# id -> (symbol prefix, collision tileset, display name, outdoor?)
MAPS = {
    (24, 4): ("NewBarkTown", JOHTO, "New Bark Town", True),
    (24, 3): ("Route29", JOHTO, "Route 29", True),
    (26, 3): ("CherrygroveCity", JOHTO, "Cherrygrove City", True),
    (26, 1): ("Route30", JOHTO, "Route 30", True),
    (26, 2): ("Route31", JOHTO, "Route 31", True),
    (10, 5): ("VioletCity", JOHTO, "Violet City", True),
    (24, 5): ("ElmsLab", "TilesetLabColl", "Elm's Lab", False),
    (26, 10): ("MrPokemonsHouse", "TilesetTraditionalHouseColl", "Mr. Pokémon's House", False),
    (26, 11): ("Route31VioletGate", "TilesetGateColl", "Violet Gatehouse", False),
    (10, 7): ("VioletGym", None, "Violet Gym", False),
    (26, 5): ("CherrygrovePokecenter1F", "TilesetPokecenterColl", "Cherrygrove Pokécenter", False),
    (10, 10): ("VioletPokecenter1F", "TilesetPokecenterColl", "Violet Pokécenter", False),
}
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
GRASS = {0x10, 0x14, 0x18, 0x1C}
WATER = {0x21, 0x29}
LEDGE = {0x41, 0x42, 0x43, 0x44, 0x50, 0x51, 0x52, 0x53}
# Trainer and NPC tiles are not modelled: battles are sparse rewards, not walls. (falkner_nav
# walls off (5, 23) on Route 30, but that tile is the only way past the y=24 ledge row.)
NPC_BLOCKS: dict[tuple[int, int], set] = {}


def load_grid(rom: _Rom, prefix: str, coll: str):
    b, a = rom.sym[prefix + "_MapAttributes"]
    o = rom.off(b, a)
    h, w = rom.rom[o + 1], rom.rom[o + 2]
    bb, pp = rom.rom[o + 3], rom.rom[o + 4] | rom.rom[o + 5] << 8
    blocks = rom.rom[rom.off(bb, pp):rom.off(bb, pp) + h * w]
    cells = {}
    if coll:
        cb, ca = rom.sym[coll]
        co = rom.off(cb, ca)
        for by in range(h):
            for bx in range(w):
                blk = blocks[by * w + bx]
                for k in range(4):
                    cells[(bx * 2 + (k & 1), by * 2 + (k >> 1))] = rom.rom[co + blk * 4 + k]
    conns = {}
    flags = rom.rom[o + 11]
    p = o + 12
    for bit, name in ((8, "north"), (4, "south"), (2, "west"), (1, "east")):
        if flags & bit:
            c = rom.rom[p:p + 12]
            p += 12
            sg = lambda v: v - 256 if v > 127 else v
            conns[name] = (c[0], c[1], sg(c[8]), sg(c[9]))
    return w * 2, h * 2, cells, conns


class Graph:
    def __init__(self):
        rom = _Rom()
        self.size, self.cells, self.conns, self.warps = {}, {}, {}, {}
        self.warp_at = {}
        for mid, (prefix, coll, _n, _o) in MAPS.items():
            w, h, cells, conns = load_grid(rom, prefix, coll)
            self.size[mid], self.cells[mid], self.conns[mid] = (w, h), cells, conns
            self.warps[mid] = [(x, y, wid, (g, n)) for (y, x, wid, g, n) in rom.warps(prefix)]
            for (x, y, wid, dest) in self.warps[mid]:
                self.warp_at[(mid, x, y)] = (wid, dest)
        self.gym_cells = self._gym_bridge()

    def _gym_bridge(self):
        """Violet Gym floor from the verified S-curve route in falkner_nav.violet_gym_step."""
        x, y, facing = 4, 14, "up"
        tiles = [(4, 15), (4, 14)]
        for _ in range(80):
            a = violet_gym_step(x, y, facing)
            if a == "a":
                break
            dx, dy = DIRS[a]
            x, y, facing = x + dx, y + dy, a
            tiles.append((x, y))
        return tiles

    # --- movement graph ----------------------------------------------------------------------
    def walk_ok(self, mid, x, y, d):
        if mid == (10, 7):
            return (x, y) in set(self.gym_cells) or (x, y) == (5, 15)
        if (x, y) in NPC_BLOCKS.get(mid, ()):
            return False
        c = self.cells[mid].get((x, y))
        return c is not None and walkable(c, d)

    def step(self, node, d):
        mid, x, y = node
        w, h = self.size[mid]
        dx, dy = DIRS[d]
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            target = (mid, nx, ny)
        else:
            conn = self.conns[mid].get({"up": "north", "down": "south", "left": "west", "right": "east"}[d])
            if conn is None or (conn[0], conn[1]) not in MAPS:
                return None
            omid = (conn[0], conn[1])
            ow, oh = self.size[omid]
            if d == "up":
                tx, ty = nx + conn[3], oh - 1
            elif d == "down":
                tx, ty = nx + conn[3], 0
            elif d == "left":
                tx, ty = ow - 1, ny + conn[2]
            else:
                tx, ty = 0, ny + conn[2]
            if not (0 <= tx < ow and 0 <= ty < oh):
                return None
            target = (omid, tx, ty)
        tmid, tx, ty = target
        if target in self.warp_at:
            wid, dest = self.warp_at[target]
            if dest in MAPS and 0 < wid <= len(self.warps[dest]):
                wx, wy, _w, _d = self.warps[dest][wid - 1]
                return (dest, wx, wy)
            return None
        return target if self.walk_ok(tmid, tx, ty, d) else None

    def neighbors(self, node):
        for d in DIRS:
            nxt = self.step(node, d)
            if nxt is not None:
                yield nxt

    def path(self, a, b):
        prev = {a: None}
        q = deque([a])
        while q:
            n = q.popleft()
            if n == b:
                break
            for m in self.neighbors(n):
                if m not in prev:
                    prev[m] = n
                    q.append(m)
        if b not in prev:
            raise RuntimeError(f"no path {a} -> {b}")
        out, n = [], b
        while n is not None:
            out.append(n)
            n = prev[n]
        return out[::-1]


def tile_class(g: Graph, mid, x, y):
    if mid == (10, 7):
        return "f" if (x, y) in set(g.gym_cells) or (x, y) == (5, 15) else "p"  # p = pit
    c = g.cells[mid].get((x, y))
    if (mid, x, y) in g.warp_at:
        return "d"
    if c is None:
        return "w"
    if is_cut_tree(c):
        return "t"
    if c in WATER:
        return "a"
    if not walkable(c):
        return "w"
    if c in GRASS:
        return "g"
    if c in LEDGE:
        return "l"
    return "f"


def world_layout(g: Graph):
    """Tile origin of each outdoor map in one stitched coordinate system."""
    origin = {(24, 4): (0, 0)}
    todo = deque([(24, 4)])
    while todo:
        mid = todo.popleft()
        ox, oy = origin[mid]
        w, h = g.size[mid]
        for name, conn in g.conns[mid].items():
            om = (conn[0], conn[1])
            if om not in MAPS or om in origin:
                continue
            ow, oh = g.size[om]
            if name == "north":
                o = (ox - conn[3], oy - oh)
            elif name == "south":
                o = (ox - conn[3], oy + h)
            elif name == "west":
                o = (ox - ow, oy - conn[2])
            else:
                o = (ox + w, oy - conn[2])
            origin[om] = o
            todo.append(om)
    # Violet City joins Route 31 only through the gatehouse; put it west of the gate door.
    r31, vc = (26, 2), (10, 5)
    gx, gy = next((x, y) for (x, y, _w, d) in g.warps[r31] if d == (26, 11))
    vx, vy = next((x, y) for (x, y, _w, d) in g.warps[vc] if d == (26, 11))
    origin[vc] = (origin[r31][0] + gx - vx - 6, origin[r31][1] + gy - vy)
    return origin, ((gx, gy), (vx, vy))


def main(out: Path):
    g = Graph()
    elm_front = ((24, 5), 5, 3)             # tile below Elm (5, 2), facing up
    mrp_front = ((26, 10), 3, 6)            # tile below Mr. Pokémon (3, 5)
    rival = ((26, 3), 33, 7)                # Cherrygrove rival trigger (x=33, y=6..7)
    falkner = ((10, 7), 5, 2)               # approach tile below Falkner (5, 1)
    goals = [
        ("elm_start", "Elm's Lab", "Start: Professor Elm gives the starter and the egg errand", elm_front),
        ("mr_pokemon", "Mr. Pokémon", "Receive the Mystery Egg", mrp_front),
        ("rival", "Rival ambush", "Cherrygrove rival battle", rival),
        ("elm_egg", "Elm's Lab", "Deliver the Mystery Egg to Professor Elm", elm_front),
        ("falkner", "Falkner", "Violet Gym: defeat Falkner for the Zephyr Badge", falkner),
    ]
    legs, full = [], []
    for i in range(len(goals) - 1):
        p = g.path(goals[i][3], goals[i + 1][3])
        legs.append({"from": goals[i][0], "to": goals[i + 1][0], "steps": len(p) - 1, "nodes": p})
        full += p if not full else p[1:]

    origin, gate = world_layout(g)
    maps = {}
    for mid, (prefix, coll, name, outdoor) in MAPS.items():
        w, h = g.size[mid]
        if mid == (10, 7):
            w, h = 10, 16
        rows = ["".join(tile_class(g, mid, x, y) for x in range(w)) for y in range(h)]
        entry = {"id": f"{mid[0]}_{mid[1]}", "name": name, "w": w, "h": h, "rows": rows, "outdoor": outdoor}
        if outdoor:
            entry["ox"], entry["oy"] = origin[mid]
        maps[f"{mid[0]}_{mid[1]}"] = entry

    def seg_list(nodes):
        segs, cur, last = [], [], None
        for (mid, x, y) in nodes:
            if mid != last and cur:
                segs.append({"map": f"{last[0]}_{last[1]}", "pts": cur})
                cur = []
            cur.append([x, y])
            last = mid
        if cur:
            segs.append({"map": f"{last[0]}_{last[1]}", "pts": cur})
        return segs

    out_legs = []
    for i, leg in enumerate(legs):
        out_legs.append({
            "index": i, "from": leg["from"], "to": leg["to"], "steps": leg["steps"],
            "segments": seg_list(leg["nodes"]),
        })

    def outdoor_door(interior):
        """(outdoor map, x, y) of the door leading into `interior`."""
        for mid, (_p, _c, _n, outdoor) in MAPS.items():
            if not outdoor:
                continue
            for (x, y, _w, d) in g.warps[mid]:
                if d == interior:
                    return mid, x, y
        return None

    stars = []
    for i, (gid, name, desc, node) in enumerate(goals):
        mid = node[0]
        wm, wx, wy = (mid, node[1], node[2]) if MAPS[mid][3] else outdoor_door(mid)
        stars.append({"id": gid, "n": i + 1, "name": name, "desc": desc,
                      "map": f"{mid[0]}_{mid[1]}", "x": node[1], "y": node[2],
                      "wmap": f"{wm[0]}_{wm[1]}", "wx": wx, "wy": wy})

    def door(mid, dest_name):
        for (x, y, _w, d) in g.warps[mid]:
            if d == dest_name:
                return x, y
    heals = []
    for mid, dm, label in (((26, 3), (26, 5), "Cherrygrove Pokécenter"), ((10, 5), (10, 10), "Violet Pokécenter")):
        x, y = door(mid, dm)
        heals.append({"name": label, "map": f"{mid[0]}_{mid[1]}", "x": x, "y": y})

    data = {
        "maps": maps, "legs": out_legs, "stars": stars, "heals": heals,
        "gate": {"route31": list(gate[0]), "violet": list(gate[1])},
        "total_steps": sum(l["steps"] for l in legs),
    }
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print("legs:", [(l["from"], l["to"], l["steps"]) for l in legs], "total", data["total_steps"])
    print("origins:", {maps[k]["name"]: (maps[k]["ox"], maps[k]["oy"]) for k in maps if maps[k]["outdoor"]})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
