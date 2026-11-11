"""Semantic spatial grid: a player-centred 15x15 view of the overworld read from WRAM / ROM, instead of screen pixels.

``GridExtractor(machine).extract()`` returns a ``uint8`` array ``(GRID_CHANNELS, 15, 15)``, player at (7, 7), +x east, +y
south. Planes (categorical planes are one-hot encoded inside the policy, see ``src.grid_policy``):

====  ===============  ======================================================================================
 0    CH_TERRAIN       TERRAIN_* class of the 16x16 square (floor, wall, ledge, grass, warp, water, counter,
                       field obstacle, interactive furniture)
 1    CH_ENTITY        ENT_* class of the object standing there (src.entity_classifier); the player is class 1
 2    CH_SIGHT         1 on every tile an UNBEATEN trainer would spot the player on (not blocked by walls: that is
                       how the game works, see entity_classifier)
 3    CH_ENTRY         4-bit mask of the move directions that may ENTER the tile (bit0 down, 1 left, 2 right, 3 up),
                       0 = unrestricted. Ledges, one-way walls and conveyor tiles use it.
 4    CH_COUNTER_FRONT 1 on a walkable tile that has a counter next to it and an NPC behind the counter (pressing A
                       there talks to the clerk / nurse)
 5    CH_FACING        on the player's tile only: 1 down, 2 up, 3 left, 4 right
 6    CH_VISITED       1 if the agent already stood on that tile this episode (from ``visited``)
====  ===============  ======================================================================================

Terrain comes from the game's real collision data: ``wOverworldMapBlocks`` (block ids, row stride ``wMapWidth + 6`` with a
3-block border that holds the connected maps) looked up in the tileset collision table (``wTilesetCollisionBank/Address``,
4 bytes per block = its 2x2 squares), then classified with the ROM's own ``CollisionPermissionTable`` (LAND / WATER / WALL
plus the TALK bit for cut / headbutt trees). Squares outside the buffered border are solid.

While a battle or a text box is open the grid is all zeros (``menu_open=True``): the policy reads that state from the RAM vector.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from src import collision as C
from src.entity_classifier import (
    ENT_NONE,
    ENT_NPC_MOVING,
    ENT_NPC_STILL,
    ENT_PLAYER,
    Entity,
    read_entities,
    rom_offset,
    sight_tiles,
)

GRID_SIZE = 15
RADIUS = GRID_SIZE // 2

CH_TERRAIN, CH_ENTITY, CH_SIGHT, CH_ENTRY, CH_COUNTER_FRONT, CH_FACING, CH_VISITED = range(7)
GRID_CHANNELS = 7

TERRAIN_FLOOR, TERRAIN_WALL, TERRAIN_LEDGE, TERRAIN_GRASS, TERRAIN_WARP, TERRAIN_WATER, TERRAIN_COUNTER = range(7)
TERRAIN_FIELD_OBSTACLE = 7      # cut / headbutt tree: a wall until the field move is available
TERRAIN_FURNITURE = 8           # bookshelf, PC, TV, shelf...: blocks movement, A interacts
TERRAIN_CLASSES = 9

# PlayerFacing plane / entry mask directions
DIR_BITS = {"down": 1, "left": 2, "right": 4, "up": 8}
DIR_DELTA = {"down": (0, 1), "left": (-1, 0), "right": (1, 0), "up": (0, -1)}
_FACING_VALUE = {0: 1, 4: 2, 8: 3, 12: 4}   # wPlayerDirection -> CH_FACING value

# constants/collision_constants.asm: LAND_TILE 0, WATER_TILE 1, WALL_TILE 0x0f, TALK 0x10
_PERM_BASE_MASK = 0x0F
_PERM_WATER = 0x01
_PERM_WALL = 0x0F
_PERM_TALK = 0x10

_HOP_TILES = frozenset({C.COLL_HOP_RIGHT, C.COLL_HOP_LEFT, C.COLL_HOP_UP, C.COLL_HOP_DOWN, C.COLL_HOP_DOWN_RIGHT,
                        C.COLL_HOP_DOWN_LEFT, C.COLL_HOP_UP_RIGHT, C.COLL_HOP_UP_LEFT})
_COUNTER_TILES = frozenset({C.COLL_COUNTER, C.COLL_COUNTER_98})
_FURNITURE_TILES = frozenset({C.COLL_BOOKSHELF, C.COLL_PC, C.COLL_RADIO, C.COLL_TOWN_MAP, C.COLL_MART_SHELF, C.COLL_TV,
                              C.COLL_WINDOW, C.COLL_INCENSE_BURNER})
_GRASS_TILES = frozenset({C.COLL_TALL_GRASS_10, C.COLL_LONG_GRASS, C.COLL_TALL_GRASS, C.COLL_LONG_GRASS_1C})

_BORDER_BLOCKS = 3          # MAP_CONNECTION_PADDING_WIDTH
_BLOCKS_BUFFER_LEN = 1300   # wOverworldMapBlocks size


@lru_cache(maxsize=1)
def _default_rom() -> bytes:
    from src.config import ROM_PATH

    return Path(ROM_PATH).read_bytes()


def classify_terrain(coll: int, perm: int) -> tuple[int, int]:
    """(terrain class, entry mask) for one collision byte given its CollisionPermissionTable entry."""
    base = perm & _PERM_BASE_MASK
    if coll in C.WARP_TILES:
        terrain = TERRAIN_WARP
    elif coll in _HOP_TILES:
        terrain = TERRAIN_LEDGE
    elif coll in _COUNTER_TILES:
        terrain = TERRAIN_COUNTER
    elif base == _PERM_WALL:
        if perm & _PERM_TALK:
            terrain = TERRAIN_FIELD_OBSTACLE
        elif coll in _FURNITURE_TILES:
            terrain = TERRAIN_FURNITURE
        else:
            terrain = TERRAIN_WALL
    elif base == _PERM_WATER:
        terrain = TERRAIN_WATER
    elif coll in _GRASS_TILES:
        terrain = TERRAIN_GRASS
    else:
        terrain = TERRAIN_FLOOR
    mask = 0
    if terrain in (TERRAIN_FLOOR, TERRAIN_LEDGE, TERRAIN_GRASS, TERRAIN_WARP):
        for name, bit in DIR_BITS.items():
            if C.allows_direction(coll, name):
                mask |= bit
        if mask == 0b1111:
            mask = 0   # unrestricted
    return terrain, mask


def _shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """out[y, x] = a[y + dy, x + dx], zero outside."""
    out = np.zeros_like(a)
    h, w = a.shape
    ys, ye = max(0, -dy), min(h, h - dy)
    xs, xe = max(0, -dx), min(w, w - dx)
    if ys < ye and xs < xe:
        out[ys:ye, xs:xe] = a[ys + dy:ye + dy, xs + dx:xe + dx]
    return out


class GridExtractor:
    """Reads the grid from a ``Machine`` (src.emulator). ``rom`` / ``perm_table`` can be injected for tests."""

    def __init__(self, machine, rom: bytes | None = None, perm_table: bytes | None = None, symbols=None):
        self.machine = machine
        self.rom = rom if rom is not None else _default_rom()
        self._symbol = symbols or machine.symbol
        self._sym_cache: dict[str, tuple[int, int]] = {}
        if perm_table is None:
            bank, addr = self.symbol("CollisionPermissionTable")
            off = rom_offset(bank, addr)
            perm_table = self.rom[off:off + 256]
        if len(perm_table) != 256:
            raise ValueError("CollisionPermissionTable must be 256 bytes")
        self._terrain_lut = np.zeros(256, dtype=np.uint8)
        self._entry_lut = np.zeros(256, dtype=np.uint8)
        for coll in range(256):
            terrain, mask = classify_terrain(coll, perm_table[coll])
            self._terrain_lut[coll] = terrain
            self._entry_lut[coll] = mask
        self._coll_tables: dict[tuple[int, int], np.ndarray] = {}

    # -- memory helpers -------------------------------------------------------------------------------------
    def symbol(self, name: str) -> tuple[int, int]:
        if name not in self._sym_cache:
            self._sym_cache[name] = self._symbol(name)
        return self._sym_cache[name]

    def _u8(self, name: str) -> int:
        bank, addr = self.symbol(name)
        return self.machine.read(bank, addr, 1)[0]

    def _collision_table(self) -> np.ndarray:
        bank = self._u8("wTilesetCollisionBank")
        b, a = self.symbol("wTilesetCollisionAddress")
        lo, hi = self.machine.read(b, a, 2)
        addr = lo | (hi << 8)
        key = (bank, addr)
        if key not in self._coll_tables:
            off = rom_offset(bank, addr)
            raw = self.rom[off:off + 256 * 4]
            table = np.full(256 * 4, C.COLL_WALL, dtype=np.uint8)   # blocks past the tileset's table read as solid
            table[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
            self._coll_tables[key] = table
        return self._coll_tables[key]

    # -- extraction -----------------------------------------------------------------------------------------
    def collision_window(self) -> tuple[np.ndarray, int, int]:
        """Collision bytes of the 15x15 squares around the player (solid outside the buffered map), plus the player x, y."""
        px, py = self._u8("wXCoord"), self._u8("wYCoord")
        width, height = self._u8("wMapWidth"), self._u8("wMapHeight")
        table = self._collision_table()
        base_bank, base_addr = self.symbol("wOverworldMapBlocks")
        stride = width + 2 * _BORDER_BLOCKS

        cx = px - RADIUS + np.arange(GRID_SIZE)
        cy = py - RADIUS + np.arange(GRID_SIZE)
        bx, by = cx >> 1, cy >> 1                      # floor division, also for negative squares
        in_x = (bx >= -_BORDER_BLOCKS) & (bx < width + _BORDER_BLOCKS)
        in_y = (by >= -_BORDER_BLOCKS) & (by < height + _BORDER_BLOCKS)

        blocks = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
        coll = np.full((GRID_SIZE, GRID_SIZE), C.COLL_WALL, dtype=np.uint8)
        if in_x.any() and in_y.any():
            bx0, bx1 = int(bx[in_x].min()), int(bx[in_x].max())
            rows: dict[int, np.ndarray] = {}
            for row in sorted({int(v) for v in by[in_y]}):
                addr = base_addr + (row + _BORDER_BLOCKS) * stride + (bx0 + _BORDER_BLOCKS)
                if addr + (bx1 - bx0 + 1) > base_addr + _BLOCKS_BUFFER_LEN:
                    continue   # would run past the buffer: treat as solid
                rows[row] = np.frombuffer(self.machine.read(base_bank, addr, bx1 - bx0 + 1), dtype=np.uint8)
            for gy in range(GRID_SIZE):
                row = rows.get(int(by[gy])) if in_y[gy] else None
                if row is None:
                    continue
                for gx in range(GRID_SIZE):
                    if in_x[gx]:
                        blocks[gy, gx] = row[int(bx[gx]) - bx0]
                        sub = (int(cx[gx]) & 1) + 2 * (int(cy[gy]) & 1)
                        coll[gy, gx] = table[blocks[gy, gx] * 4 + sub]
        return coll, px, py

    def extract(self, visited=None, menu_open: bool = False, entities: list[Entity] | None = None) -> np.ndarray:
        """The (GRID_CHANNELS, 15, 15) uint8 grid. ``visited``: set of (map_group, map_number, x, y) stood on this episode."""
        grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        if menu_open:
            return grid
        coll, px, py = self.collision_window()
        grid[CH_TERRAIN] = self._terrain_lut[coll]
        grid[CH_ENTRY] = self._entry_lut[coll]

        if entities is None:
            entities = read_entities(self.machine.read, self.symbol, self.rom)
        for ent in entities:
            gx, gy = RADIUS + ent.dx, RADIUS + ent.dy
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                grid[CH_ENTITY, gy, gx] = ent.cls
            for sx, sy in sight_tiles(ent):
                tx, ty = RADIUS + sx, RADIUS + sy
                if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
                    grid[CH_SIGHT, ty, tx] = 1
        grid[CH_ENTITY, RADIUS, RADIUS] = ENT_PLAYER

        # counter front: walkable tile, a counter next to it, an NPC behind the counter
        npc = np.isin(grid[CH_ENTITY], (ENT_NPC_STILL, ENT_NPC_MOVING))
        counter = grid[CH_TERRAIN] == TERRAIN_COUNTER
        walkable = np.isin(grid[CH_TERRAIN], (TERRAIN_FLOOR, TERRAIN_GRASS)) & (grid[CH_ENTITY] == ENT_NONE)
        front = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for dx, dy in DIR_DELTA.values():
            front |= walkable & _shift(counter, dx, dy) & _shift(npc, 2 * dx, 2 * dy)
        grid[CH_COUNTER_FRONT] = front

        grid[CH_FACING, RADIUS, RADIUS] = _FACING_VALUE.get(self._u8("wPlayerDirection"), 0)

        if visited:
            group, number = self._u8("wMapGroup"), self._u8("wMapNumber")
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    if (group, number, px + gx - RADIUS, py + gy - RADIUS) in visited:
                        grid[CH_VISITED, gy, gx] = 1
        return grid
