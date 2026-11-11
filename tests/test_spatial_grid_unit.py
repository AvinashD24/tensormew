"""Unit tests for the semantic grid: a fake machine with hand-built WRAM and a synthetic ROM, no emulator or real ROM."""

import re
import unittest
from pathlib import Path

import numpy as np

from src import collision as C
from src.entity_classifier import (
    ENT_ITEM, ENT_NPC_MOVING, ENT_NPC_STILL, ENT_PLAYER, ENT_TRAINER, ENT_TRAINER_BEATEN, Entity, classify_object,
    sight_tiles,
)
from src.spatial_grid import (
    CH_COUNTER_FRONT, CH_ENTITY, CH_ENTRY, CH_FACING, CH_SIGHT, CH_TERRAIN, CH_VISITED, GRID_CHANNELS, GRID_SIZE,
    GridExtractor, TERRAIN_COUNTER, TERRAIN_FIELD_OBSTACLE, TERRAIN_FLOOR, TERRAIN_GRASS, TERRAIN_LEDGE,
    TERRAIN_WALL, TERRAIN_WARP, TERRAIN_WATER, classify_terrain,
)

SYM_PATH = Path(__file__).resolve().parent.parent / "rom" / "pokemon_crystal.sym"
_SYMS = {}
for _line in SYM_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
    _m = re.match(r"([0-9a-f]+):([0-9a-f]+) (\S+)", _line)
    if _m:
        _SYMS[_m.group(3)] = (int(_m.group(1), 16), int(_m.group(2), 16))

LAND, WATER, WALL, TALK = 0x00, 0x01, 0x0F, 0x10
PERM = bytearray([LAND] * 256)
for _c in (C.COLL_WALL, C.COLL_COUNTER, C.COLL_BOOKSHELF):
    PERM[_c] = WALL
PERM[C.COLL_CUT_TREE] = WALL | TALK

PERM[C.COLL_WATER] = WATER

COLL_BANK, COLL_ADDR = 0x10, 0x4000
# block ids of the synthetic tileset (4 collision bytes each)
BLOCKS = {0: C.COLL_FLOOR, 1: C.COLL_WALL, 2: C.COLL_TALL_GRASS, 3: C.COLL_HOP_DOWN, 4: C.COLL_COUNTER,
          5: C.COLL_CUT_TREE, 6: C.COLL_WATER, 7: C.COLL_DOOR, 8: C.COLL_UP_WALL,
          # one counter row (the bottom two squares) under a floor row, like a Pokecenter counter
          9: (C.COLL_FLOOR, C.COLL_FLOOR, C.COLL_COUNTER, C.COLL_COUNTER)}


def make_rom() -> bytearray:
    rom = bytearray(0x16 * 0x4000)
    base = COLL_BANK * 0x4000
    for blk, coll in BLOCKS.items():
        rom[base + blk * 4:base + blk * 4 + 4] = bytes(coll) if isinstance(coll, tuple) else bytes([coll] * 4)
    return rom


class FakeMachine:
    """WRAM as a dict; ``symbol`` resolves the real addresses from rom/pokemon_crystal.sym."""

    def __init__(self):
        self.mem = {}

    def symbol(self, name):
        return _SYMS[name]

    def put(self, name, data, offset=0):
        _bank, addr = _SYMS[name]
        for i, b in enumerate(bytes(data)):
            self.mem[addr + offset + i] = b

    def read(self, bank, addr, n=1):
        return bytes(self.mem.get(addr + i, 0) for i in range(n))


def build_machine(width=10, height=9, fill=0, player=(8, 8), direction=0):
    """A width x height block map, all ``fill`` blocks, with a 3-block border of floor blocks."""
    m = FakeMachine()
    stride = width + 6
    buf = bytearray(1300)
    for by in range(height + 6):
        for bx in range(stride):
            inside = 3 <= bx < width + 3 and 3 <= by < height + 3
            buf[by * stride + bx] = fill if inside else 0
    m.put("wOverworldMapBlocks", buf)
    m.put("wMapWidth", [width])
    m.put("wMapHeight", [height])
    m.put("wXCoord", [player[0]])
    m.put("wYCoord", [player[1]])
    m.put("wMapGroup", [24])
    m.put("wMapNumber", [4])
    m.put("wTilesetCollisionBank", [COLL_BANK])
    m.put("wTilesetCollisionAddress", [COLL_ADDR & 0xFF, COLL_ADDR >> 8])
    # player struct (slot 0) at map (px + 4, py + 4)
    m.put("wObjectStructs", [0] * 0x28)
    m.put("wObjectStructs", [player[0] + 4, player[1] + 4], offset=0x10)
    m.put("wPlayerDirection", [direction])   # lives inside the player struct (+8), so set it after the struct
    return m


def set_block(m, width, bx, by, block):
    _b, addr = _SYMS["wOverworldMapBlocks"]
    m.mem[addr + (by + 3) * (width + 6) + (bx + 3)] = block


def add_object(m, index, *, sx, sy, kind, sprite=7, movement=1, direction=0, sight=0, script=0, slot=None):
    """Map object ``index`` (1..15) backed by live struct ``slot``; (sx, sy) are map-square coordinates."""
    slot = index if slot is None else slot
    mo = bytearray(16)
    mo[0], mo[1], mo[8], mo[9] = slot, sprite, kind, sight
    mo[10], mo[11] = script & 0xFF, script >> 8
    m.put("wMapObjects", mo, offset=index * 16)
    st = bytearray(0x28)
    st[0x03], st[0x08], st[0x10], st[0x11] = movement, direction, sx + 4, sy + 4
    m.put("wObjectStructs", st, offset=slot * 0x28)


def empty_objects(m):
    for i in range(1, 16):
        m.put("wMapObjects", [0xFF] + [0] * 15, offset=i * 16)


def extractor(m, rom=None):
    return GridExtractor(m, rom=bytes(rom or make_rom()), perm_table=bytes(PERM))


class TerrainClassificationTests(unittest.TestCase):
    def test_classes(self):
        cls = lambda coll: classify_terrain(coll, PERM[coll])[0]
        self.assertEqual(cls(C.COLL_FLOOR), TERRAIN_FLOOR)
        self.assertEqual(cls(C.COLL_WALL), TERRAIN_WALL)
        self.assertEqual(cls(C.COLL_TALL_GRASS), TERRAIN_GRASS)
        self.assertEqual(cls(C.COLL_HOP_DOWN), TERRAIN_LEDGE)
        self.assertEqual(cls(C.COLL_DOOR), TERRAIN_WARP)
        self.assertEqual(cls(C.COLL_WATER), TERRAIN_WATER)
        self.assertEqual(cls(C.COLL_COUNTER), TERRAIN_COUNTER)
        self.assertEqual(cls(C.COLL_CUT_TREE), TERRAIN_FIELD_OBSTACLE)

    def test_entry_masks(self):
        self.assertEqual(classify_terrain(C.COLL_HOP_DOWN, PERM[C.COLL_HOP_DOWN])[1], 0b0001)   # only moving down enters
        self.assertEqual(classify_terrain(C.COLL_FLOOR, 0)[1], 0)
        self.assertEqual(classify_terrain(C.COLL_UP_WALL, 0)[1], 0b1110)   # cannot be entered moving down


class GridShapeAndTerrainTests(unittest.TestCase):
    def test_shape_dtype_and_bounds(self):
        m = build_machine(fill=0)
        empty_objects(m)
        grid = extractor(m).extract()
        self.assertEqual(grid.shape, (GRID_CHANNELS, GRID_SIZE, GRID_SIZE))
        self.assertEqual(grid.dtype, np.uint8)
        self.assertEqual(grid[CH_ENTITY, 7, 7], ENT_PLAYER)
        self.assertTrue((grid[CH_TERRAIN] <= TERRAIN_FIELD_OBSTACLE).all())

    def test_block_lookup_and_subsquares(self):
        """Player at square (8, 8) = block (4, 4). Put a wall block at (5, 4): squares x 10..11, y 8..9 turn solid."""
        m = build_machine(fill=0)
        empty_objects(m)
        set_block(m, 10, 5, 4, 1)
        grid = extractor(m).extract()
        # grid cell (gx, gy) = square (8 - 7 + gx, 8 - 7 + gy); squares x 10, 11 -> gx 9, 10; y 8, 9 -> gy 7, 8
        self.assertEqual(grid[CH_TERRAIN, 7, 9], TERRAIN_WALL)
        self.assertEqual(grid[CH_TERRAIN, 8, 10], TERRAIN_WALL)
        self.assertEqual(grid[CH_TERRAIN, 7, 8], TERRAIN_FLOOR)
        self.assertEqual(grid[CH_TERRAIN, 9, 9], TERRAIN_FLOOR)

    def test_border_blocks_are_read_and_beyond_is_solid(self):
        m = build_machine(width=10, height=9, fill=0, player=(0, 0))
        empty_objects(m)
        grid = extractor(m).extract()
        # square (-6, y) is the outermost buffered column (block -3): floor; (-7, y) is outside the buffer: solid
        self.assertEqual(grid[CH_TERRAIN, 7, 1], TERRAIN_FLOOR)   # x = -6
        self.assertEqual(grid[CH_TERRAIN, 7, 0], TERRAIN_WALL)    # x = -7
        self.assertEqual(grid[CH_TERRAIN, 0, 7], TERRAIN_WALL)    # y = -7

    def test_special_tiles_and_entry_plane(self):
        m = build_machine(fill=0)
        empty_objects(m)
        set_block(m, 10, 5, 4, 3)    # hop-down block at squares x 10..11, y 8..9
        set_block(m, 10, 4, 5, 2)    # grass at squares x 8..9, y 10..11
        grid = extractor(m).extract()
        self.assertEqual(grid[CH_TERRAIN, 7, 9], TERRAIN_LEDGE)
        self.assertEqual(grid[CH_ENTRY, 7, 9], 0b0001)
        self.assertEqual(grid[CH_TERRAIN, 9, 7], TERRAIN_GRASS)

    def test_menu_open_returns_zeros(self):
        m = build_machine()
        empty_objects(m)
        self.assertFalse(extractor(m).extract(menu_open=True).any())

    def test_facing_and_visited(self):
        m = build_machine(direction=8)   # left
        empty_objects(m)
        grid = extractor(m).extract(visited={(24, 4, 7, 8), (24, 4, 99, 99), (1, 1, 8, 8)})
        self.assertEqual(grid[CH_FACING, 7, 7], 3)
        self.assertEqual(grid[CH_VISITED, 7, 6], 1)          # square (7, 8) is one tile west
        self.assertEqual(int(grid[CH_VISITED].sum()), 1)     # other map / far tile do not count


class EntityTests(unittest.TestCase):
    def test_classify_object_kinds(self):
        player = bytearray(0x28)
        player[0x10], player[0x11] = 20, 20
        def struct(x, y, movement=1, direction=0, flags=0):
            s = bytearray(0x28)
            s[0x03], s[0x04], s[0x08], s[0x10], s[0x11] = movement, flags, direction, x, y
            return bytes(s)
        mo = lambda kind, sight=0: bytes([1, 7, 0, 0, 0, 0, 0, 0, kind, sight, 0, 0, 0, 0, 0, 0])
        self.assertEqual(classify_object(mo(1), struct(22, 20), player, False).cls, ENT_ITEM)
        npc = classify_object(mo(0), struct(21, 19), player, False)
        self.assertEqual((npc.dx, npc.dy, npc.cls), (1, -1, ENT_NPC_STILL))
        self.assertEqual(classify_object(mo(0), struct(21, 19, movement=2), player, False).cls, ENT_NPC_MOVING)
        self.assertEqual(classify_object(mo(2, 3), struct(20, 24, direction=4), player, False).cls, ENT_TRAINER)
        self.assertEqual(classify_object(mo(2, 3), struct(20, 24), player, True).cls, ENT_TRAINER_BEATEN)
        self.assertIsNone(classify_object(mo(0), struct(21, 19, flags=1), player, False))   # invisible

    def test_sight_tiles_follow_facing_through_walls(self):
        t = Entity(dx=3, dy=0, cls=ENT_TRAINER, facing=8, sight=3)   # faces left
        self.assertEqual(sight_tiles(t), [(2, 0), (1, 0), (0, 0)])
        self.assertEqual(sight_tiles(Entity(0, 0, ENT_TRAINER_BEATEN, 8, 3)), [])

    def test_trainer_defeat_flag_read_from_header(self):
        """Trainer header word -> event flag id -> bit in wEventFlags decides beaten vs hostile."""
        rom = make_rom()
        bank, ptr = 0x14, 0x5000
        off = bank * 0x4000 + (ptr - 0x4000)
        rom[off], rom[off + 1] = 0x0B, 0x01          # flag id 0x010B -> byte 33, bit 3
        for beaten in (False, True):
            m = build_machine()
            empty_objects(m)
            m.put("wMapScriptsBank", [bank])
            add_object(m, 1, sx=11, sy=8, kind=2, movement=8, direction=8, sight=4, script=ptr)
            m.put("wEventFlags", [0x08 if beaten else 0], offset=33)
            grid = extractor(m, rom).extract()
            self.assertEqual(grid[CH_ENTITY, 7, 10], ENT_TRAINER_BEATEN if beaten else ENT_TRAINER)
            self.assertEqual(int(grid[CH_SIGHT].sum()), 0 if beaten else 4)
            if not beaten:
                self.assertTrue(grid[CH_SIGHT, 7, 6:10].all())   # four tiles west of the trainer, walls or not

    def test_counter_front_marks_tile_before_the_counter(self):
        """Nurse (y=8) | counter (y=9) | floor (y=10): the floor tile below the counter is the interaction front."""
        m = build_machine(fill=0)
        empty_objects(m)
        set_block(m, 10, 5, 4, 9)                           # squares x 10..11: y=8 floor, y=9 counter
        add_object(m, 1, sx=10, sy=8, kind=0, movement=1)   # nurse behind the counter
        grid = extractor(m).extract()
        self.assertEqual(grid[CH_TERRAIN, 8, 9], TERRAIN_COUNTER)   # square (10, 9)
        self.assertEqual(grid[CH_ENTITY, 7, 9], ENT_NPC_STILL)      # square (10, 8)
        self.assertEqual(grid[CH_COUNTER_FRONT, 9, 9], 1)           # square (10, 10)
        self.assertEqual(int(grid[CH_COUNTER_FRONT].sum()), 1)


try:
    import torch  # noqa: F401
    TORCH = True
except ImportError:
    TORCH = False


@unittest.skipUnless(TORCH, "needs torch")
class GridPolicyTests(unittest.TestCase):
    def setUp(self):
        from src.grid_policy import SpatialGridPolicy
        self.cls = SpatialGridPolicy
        self.policy = SpatialGridPolicy(obs_dim=80, use_gru=True)

    def test_param_count_is_small(self):
        self.assertLess(self.policy.param_count, 1_000_000)
        self.assertGreater(self.policy.param_count, 100_000)

    def test_act_and_batched_forward(self):
        import torch
        grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        action, logp, value = self.policy.act(grid, np.zeros(80, dtype=np.float32))
        self.assertTrue(0 <= action < 6)
        logits, values = self.policy.forward(torch.zeros(5, GRID_CHANNELS, GRID_SIZE, GRID_SIZE, dtype=torch.uint8),
                                             torch.zeros(5, 80))
        self.assertEqual(tuple(logits.shape), (5, 6))
        self.assertEqual(tuple(values.shape), (5,))

    def test_sequence_evaluation_shapes(self):
        import torch
        B, T = 2, 4
        grids = torch.randint(0, 3, (B, T, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=torch.uint8)
        logp, ent, values = self.policy.evaluate_actions_sequence(grids, torch.zeros(B, T, 80),
                                                                  torch.zeros(B, T, dtype=torch.long))
        self.assertEqual(tuple(logp.shape), (B, T))
        self.assertEqual(tuple(values.shape), (B, T))

    def test_gradients_flow_to_conv_trunk(self):
        import torch
        grid = torch.randint(0, 3, (3, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=torch.uint8)
        logits, values = self.policy.forward(grid, torch.zeros(3, 80))
        (logits.sum() + values.sum()).backward()
        self.assertGreater(float(self.policy.vision_trunk.conv1.weight.grad.abs().sum()), 0.0)

    def test_checkpoint_roundtrip_and_schema_guard(self):
        import tempfile
        from src.deep_policy import CheckpointIncompatibleError, SpatialVisionPolicy
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "grid.pt"
            self.policy.save(path)
            loaded = self.cls.load(path, check_schema=True)
            self.assertIsInstance(loaded, self.cls)
            for (k, a), (_k, b) in zip(self.policy.state_dict().items(), loaded.state_dict().items()):
                self.assertTrue(torch_equal(a, b), k)
            # a pixel checkpoint must not load as a grid policy
            pix = Path(d) / "pix.pt"
            import torch
            torch.save({"state_dict": {}, "obs_dim": 80, "num_actions": 6}, str(pix))
            with self.assertRaises(CheckpointIncompatibleError):
                self.cls.load(pix)


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


if __name__ == "__main__":
    unittest.main()
