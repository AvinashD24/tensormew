"""Overworld entities (NPCs, trainers, item balls) read from WRAM, for the semantic grid (src/spatial_grid.py).

Everything here comes from the game's own structures (pret/pokecrystal ``constants/map_object_constants.asm`` and
``home/trainers.asm``), not from hand-written story knowledge: the policy is told WHAT is on a tile, never WHERE to go.

Facts this module relies on (all checked against pokecrystal):

* ``wObjectStructs`` holds 13 live objects of 0x28 bytes (slot 0 is the player). ``OBJECT_MAP_X/Y`` (+0x10/+0x11) carry a
  +4 offset, so positions are used RELATIVE to the player's own struct and the offset cancels.
* ``wMapObjects`` holds 16 map objects of 16 bytes (index 0 is the player). Byte 0 is the live struct slot (0xFF = not
  loaded), byte 1 the sprite, byte 8 the type (low nibble: 0 script, 1 item ball, 2 trainer), byte 9 the trainer sight
  range, bytes 0xA-0xB the script pointer (for a trainer: the trainer header, whose first word is its event flag).
* A trainer notices the player when the player is in the same row / column, the trainer FACES the player (live
  ``OBJECT_DIRECTION`` +8: 0 down, 4 up, 8 left, 12 right) and the distance is <= the sight range. The game does NOT check for
  walls or other objects in between (``FacingPlayerDistance``), so the sight ray is drawn through them too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# object_struct (wObjectStructs)
OBJECT_STRUCT_LEN = 0x28
NUM_OBJECT_STRUCTS = 13
OBJ_MOVEMENT_TYPE = 0x03
OBJ_FLAGS1 = 0x04
OBJ_DIRECTION = 0x08
OBJ_MAP_X = 0x10
OBJ_MAP_Y = 0x11
OBJ_FLAG_INVISIBLE = 0x01  # OBJECT_FLAGS1 bit 0

# map_object (wMapObjects)
MAPOBJECT_LEN = 16
NUM_OBJECTS = 16
MO_STRUCT_ID = 0
MO_SPRITE = 1
MO_TYPE = 8
MO_SIGHT_RANGE = 9
MO_SCRIPT_PTR = 0x0A

OBJECTTYPE_SCRIPT = 0
OBJECTTYPE_ITEMBALL = 1
OBJECTTYPE_TRAINER = 2

# SpriteMovementData indexes that never move by themselves (still / standing in a direction).
_STATIONARY_MOVEMENT = frozenset({0, 1, 6, 7, 8, 9})

# OBJECT_DIRECTION value -> (dx, dy) of the tile the object faces
FACING_DELTA = {0: (0, 1), 4: (0, -1), 8: (-1, 0), 12: (1, 0)}

# Entity classes (plane CH_ENTITY of the grid)
ENT_NONE = 0
ENT_PLAYER = 1
ENT_TRAINER = 2          # unbeaten trainer: starts a battle when the player enters its sight line
ENT_TRAINER_BEATEN = 3
ENT_ITEM = 4             # item ball on the ground
ENT_NPC_STILL = 5        # script NPC / prop that stays put (clerk, nurse, sign-like object)
ENT_NPC_MOVING = 6       # script NPC that walks around (can block a corridor, then moves on)
ENT_OTHER = 7
ENTITY_CLASSES = 8


@dataclass(frozen=True)
class Entity:
    dx: int                  # tile offset from the player (+x east)
    dy: int                  # tile offset from the player (+y south)
    cls: int                 # ENT_*
    facing: int = 0          # OBJECT_DIRECTION (0 down, 4 up, 8 left, 12 right)
    sight: int = 0           # trainer sight range in tiles (0 for non-trainers)
    sprite: int = 0


def sight_tiles(entity: Entity) -> list[tuple[int, int]]:
    """Offsets (relative to the player) a hostile trainer would spot the player on: 1..sight tiles along its facing."""
    if entity.cls != ENT_TRAINER or entity.sight <= 0:
        return []
    fx, fy = FACING_DELTA.get(entity.facing, (0, 0))
    return [(entity.dx + fx * d, entity.dy + fy * d) for d in range(1, entity.sight + 1)]


def rom_offset(bank: int, addr: int) -> int:
    """File offset of a ROM (bank, address) pair; bank 0 / addresses below 0x4000 are not banked."""
    return bank * 0x4000 + (addr - 0x4000) if (bank and addr >= 0x4000) else addr


def _flag_set(flags: bytes, flag_id: int) -> bool:
    byte_index, bit = divmod(flag_id, 8)
    return byte_index < len(flags) and bool(flags[byte_index] & (1 << bit))


def classify_object(map_object: bytes, struct: bytes, player_struct: bytes, beaten: bool) -> Entity | None:
    """One map object + its live struct -> Entity (None if invisible). Pure: no emulator access."""
    if struct[OBJ_FLAGS1] & OBJ_FLAG_INVISIBLE:
        return None
    dx = struct[OBJ_MAP_X] - player_struct[OBJ_MAP_X]
    dy = struct[OBJ_MAP_Y] - player_struct[OBJ_MAP_Y]
    kind = map_object[MO_TYPE] & 0x0F
    facing = struct[OBJ_DIRECTION]
    sprite = map_object[MO_SPRITE]
    if kind == OBJECTTYPE_TRAINER:
        if beaten:
            return Entity(dx, dy, ENT_TRAINER_BEATEN, facing, 0, sprite)
        return Entity(dx, dy, ENT_TRAINER, facing, map_object[MO_SIGHT_RANGE], sprite)
    if kind == OBJECTTYPE_ITEMBALL:
        return Entity(dx, dy, ENT_ITEM, facing, 0, sprite)
    if kind == OBJECTTYPE_SCRIPT:
        moving = struct[OBJ_MOVEMENT_TYPE] not in _STATIONARY_MOVEMENT
        return Entity(dx, dy, ENT_NPC_MOVING if moving else ENT_NPC_STILL, facing, 0, sprite)
    return Entity(dx, dy, ENT_OTHER, facing, 0, sprite)


def read_entities(
    read: Callable[[int, int, int], bytes],
    symbol: Callable[[str], tuple[int, int]],
    rom: bytes | None,
) -> list[Entity]:
    """All visible non-player objects on the current map, positions relative to the player.

    ``read(bank, addr, n)`` / ``symbol(name)`` are the Machine methods; ``rom`` (file bytes) is only needed to look up
    whether a trainer was already beaten (without it every trainer counts as unbeaten).
    """
    sbank, saddr = symbol("wObjectStructs")
    player = read(sbank, saddr, OBJECT_STRUCT_LEN)
    mbank, maddr = symbol("wMapObjects")
    map_objects = read(mbank, maddr, MAPOBJECT_LEN * NUM_OBJECTS)
    flags = b""
    scripts_bank = None
    out: list[Entity] = []
    for index in range(1, NUM_OBJECTS):  # 0 is the player
        mo = map_objects[index * MAPOBJECT_LEN:(index + 1) * MAPOBJECT_LEN]
        slot = mo[MO_STRUCT_ID]
        if slot >= NUM_OBJECT_STRUCTS or slot == 0:  # 0xFF: not loaded
            continue
        struct = read(sbank, saddr + slot * OBJECT_STRUCT_LEN, OBJECT_STRUCT_LEN)
        beaten = False
        if (mo[MO_TYPE] & 0x0F) == OBJECTTYPE_TRAINER and rom is not None:
            if not flags:
                fb, fa = symbol("wEventFlags")
                flags = read(fb, fa, 256)
                bb, ba = symbol("wMapScriptsBank")
                scripts_bank = read(bb, ba, 1)[0]
            ptr = mo[MO_SCRIPT_PTR] | (mo[MO_SCRIPT_PTR + 1] << 8)
            off = rom_offset(scripts_bank, ptr)
            if 0 <= off < len(rom) - 1:
                beaten = _flag_set(flags, rom[off] | (rom[off + 1] << 8))
        entity = classify_object(mo, struct, player, beaten)
        if entity is not None:
            out.append(entity)
    return out
