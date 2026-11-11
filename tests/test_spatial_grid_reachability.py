"""Ground-truth check of the semantic grid against the live emulator.

For each saved state: extract the grid, plan BFS paths over it, execute them in PyBoy and compare where the player really
ends up. Also: solid tiles / NPCs must bump, ledges must be one-way, and a trainer sight tile must start a battle.

Needs the ROM, ``pyboy`` and the saved states; skips otherwise. Moves use the env's press shape (8 frames held + 12 free,
``src.rl_env.press_button``). A press toward a new direction first TURNS the player without moving (DoPlayerMovement
.CheckTurning in pokecrystal), so the number of MOVES is asserted equal to the BFS length, not the number of presses.
"""

import unittest
from collections import deque
from pathlib import Path

import numpy as np

from src.spatial_grid import (
    CH_ENTITY, CH_ENTRY, CH_SIGHT, CH_TERRAIN, DIR_BITS, DIR_DELTA, GridExtractor, RADIUS, TERRAIN_FLOOR,
    TERRAIN_GRASS, TERRAIN_LEDGE,
)
from src.entity_classifier import ENT_NPC_MOVING

ROOT = Path(__file__).resolve().parent.parent
STATES = ("new_bark_outside", "route29_entry", "post_rival_cherrygrove")   # post_rival_cherrygrove is in Cherrygrove
MAX_DIST = 4
SOLID_TERRAIN = (1, 5, 6, 7, 8)   # wall, water, counter, field obstacle, furniture


def _open_machine():
    try:
        from src.emulator import Emulator
        return Emulator()
    except Exception as exc:  # no pyboy / wrong or missing ROM
        raise unittest.SkipTest(f"emulator unavailable: {exc}")


def _press(machine, direction):
    machine.press(direction, frames=8)
    machine.tick(12, render=False)


def _pos(machine):
    bx, ax = machine.symbol("wXCoord")
    by, ay = machine.symbol("wYCoord")
    return machine.read(bx, ax, 1)[0], machine.read(by, ay, 1)[0]


def _map(machine):
    return tuple(machine.read(*machine.symbol(s), 1)[0] for s in ("wMapGroup", "wMapNumber"))


def _walk(grid, allow_sight=False):
    """BFS over the grid from the player: {(gx, gy): [directions]} for floor / grass cells within MAX_DIST."""
    terrain, entity, entry, sight = grid[CH_TERRAIN], grid[CH_ENTITY], grid[CH_ENTRY], grid[CH_SIGHT]
    start = (RADIUS, RADIUS)
    paths = {start: []}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        if len(paths[(cx, cy)]) >= MAX_DIST:
            continue
        for name, (dx, dy) in DIR_DELTA.items():
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in paths or not (0 <= nx < grid.shape[2] and 0 <= ny < grid.shape[1]):
                continue
            if terrain[ny, nx] not in (TERRAIN_FLOOR, TERRAIN_GRASS) or entity[ny, nx] != 0:
                continue
            if entry[ny, nx] and not entry[ny, nx] & DIR_BITS[name]:
                continue
            if sight[ny, nx] and not allow_sight:
                continue
            paths[(nx, ny)] = paths[(cx, cy)] + [name]
            queue.append((nx, ny))
    return paths


def _run(machine, actions):
    """Execute BFS actions; a press that only turns is repeated once. Returns (final position, moves made)."""
    moves = 0
    for name in actions:
        before = _pos(machine)
        _press(machine, name)
        if _pos(machine) == before:
            _press(machine, name)
        if _pos(machine) != before:
            moves += 1
    return _pos(machine), moves


class GridReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.machine = _open_machine()

    @classmethod
    def tearDownClass(cls):
        cls.machine.close()

    def load(self, name):
        path = ROOT / "states" / f"{name}.state"
        if not path.exists():
            self.skipTest(f"missing state {name}")
        self.machine.load_state(path)
        self.machine.tick(4, render=False)
        return GridExtractor(self.machine)

    def test_live_collision_matches_rom_map_grids(self):
        """Every in-map square of the live window must equal the ROM-derived grid (src.return_nav._Rom.grid)."""
        from src.return_nav import _Rom
        rom = _Rom()
        prefixes = {(24, 4): "NewBarkTown", (24, 3): "Route29", (26, 3): "CherrygroveCity"}
        for name in STATES:
            with self.subTest(state=name):
                ex = self.load(name)
                prefix = prefixes.get(_map(self.machine))
                if prefix is None:
                    self.skipTest(f"no ROM grid prefix for map {_map(self.machine)}")
                _w, _h, cells, _conns = rom.grid(prefix)
                coll, px, py = ex.collision_window()
                compared = 0
                for gy in range(coll.shape[0]):
                    for gx in range(coll.shape[1]):
                        cell = (px + gx - RADIUS, py + gy - RADIUS)
                        if cell in cells:
                            self.assertEqual(int(coll[gy, gx]), cells[cell], f"{name}: square {cell}")
                            compared += 1
                self.assertGreater(compared, 50)

    def test_bfs_paths_match_emulator(self):
        for name in STATES:
            with self.subTest(state=name):
                ex = self.load(name)
                snapshot = self.machine.save_state()
                grid = ex.extract()
                px, py = _pos(self.machine)
                start_map = _map(self.machine)
                width, height = (self.machine.read(*self.machine.symbol(s), 1)[0] for s in ("wMapWidth", "wMapHeight"))
                moving = [(x, y) for y, x in zip(*np.nonzero(grid[CH_ENTITY] == ENT_NPC_MOVING))]
                checked = 0
                for (gx, gy), path in _walk(grid).items():
                    if not path or any(abs(gx - mx) + abs(gy - my) <= 2 for mx, my in moving):
                        continue   # a wandering NPC may legitimately step into the way
                    target = (px + gx - RADIUS, py + gy - RADIUS)
                    off_map = not (0 <= target[0] < 2 * width and 0 <= target[1] < 2 * height)
                    self.machine.load_state(snapshot)
                    got, moves = _run(self.machine, path)
                    if off_map:
                        # a connected-map border square: walking onto it must switch maps (coordinates then restart)
                        self.assertNotEqual(_map(self.machine), start_map, f"{name}: path {path} did not cross the map edge")
                    else:
                        self.assertEqual(got, target, f"{name}: path {path}")
                        self.assertEqual(moves, len(path), f"{name}: path {path}")
                    checked += 1
                self.assertGreater(checked, 0, f"{name}: no BFS target checked")

    def test_solid_neighbours_bump(self):
        for name in STATES:
            with self.subTest(state=name):
                ex = self.load(name)
                snapshot = self.machine.save_state()
                grid = ex.extract()
                start = _pos(self.machine)
                for direction, (dx, dy) in DIR_DELTA.items():
                    gx, gy = RADIUS + dx, RADIUS + dy
                    if grid[CH_TERRAIN, gy, gx] in SOLID_TERRAIN or grid[CH_ENTITY, gy, gx] != 0:
                        self.machine.load_state(snapshot)
                        _press(self.machine, direction)
                        _press(self.machine, direction)
                        self.assertEqual(_pos(self.machine), start, f"{name}: moved {direction} into a solid tile")

    def test_ledge_is_one_way(self):
        tested = False
        for name in STATES:
            ex = self.load(name)
            snapshot = self.machine.save_state()
            grid = ex.extract()
            px, py = _pos(self.machine)
            reachable = _walk(grid)
            for (gx, gy) in zip(*np.nonzero(grid[CH_TERRAIN] == TERRAIN_LEDGE)[::-1]):
                mask = int(grid[CH_ENTRY, gy, gx])
                names = [n for n, bit in DIR_BITS.items() if mask == bit]
                if not names:
                    continue
                d = names[0]
                dx, dy = DIR_DELTA[d]
                before, after = (gx - dx, gy - dy), (gx + dx, gy + dy)
                if before in reachable and after in reachable:
                    self.machine.load_state(snapshot)
                    _run(self.machine, reachable[before])
                    start = _pos(self.machine)
                    _press(self.machine, d)
                    self.machine.tick(40, render=False)
                    self.assertNotEqual(_pos(self.machine), start, f"{name}: could not hop the ledge {d}")
                    tested = True
        if not tested:
            self.skipTest("no ledge with both sides reachable within range in these states")

    def test_trainer_sight_tile_starts_a_battle(self):
        name = "agent_map_26_1"   # Route 30
        ex = self.load(name)
        snapshot = self.machine.save_state()
        grid = ex.extract()
        targets = [(c, p) for c, p in _walk(grid, allow_sight=True).items() if grid[CH_SIGHT, c[1], c[0]] and p]
        if not targets:
            self.skipTest("no unbeaten trainer sight tile within reach of this state")
        _cell, path = targets[0]
        self.machine.load_state(snapshot)
        _run(self.machine, path)
        bank, addr = self.machine.symbol("wBattleMode")
        for _ in range(30):   # the trainer exclaims, walks over and talks first
            self.machine.tick(10, render=False)
            if self.machine.read(bank, addr, 1)[0]:
                return
        self.fail("stood on a trainer sight tile but no battle started within 300 frames")


if __name__ == "__main__":
    unittest.main()
