# Semantic spatial grid (implemented spec)

Replaces the ResNet-18 screen trunk with a 15x15 player-centred grid read from WRAM/ROM. Code: `src/spatial_grid.py`,
`src/entity_classifier.py`, `src/grid_policy.py`. Tests: `tests/test_spatial_grid_unit.py` (synthetic WRAM, no emulator),
`tests/test_spatial_grid_reachability.py` (live PyBoy; needs `pyboy==2.8.1`, the states' PyBoy version).

## What changed from the first draft (verified against `rom/pokemon_crystal.sym` and pret/pokecrystal)

| Draft | Reality |
| --- | --- |
| `wPlayerDirection` at `01:dcb9` | `01:d4de` (inside the player object struct; `dcb9` is `wScreenSave`) |
| `wObjectStructs` `01:d4d7`, `wEventFlags` `01:d9a0` | `01:d4d6`, `01:da72`. Always resolve by symbol. |
| object fields `+0x07/+0x0e/+0x0f` | direction `+8`, map X/Y `+0x10/+0x11` (carry a +4 offset; positions are taken relative to the player struct) |
| `wOverworldMapBlocks` is collision | block ids; row stride `wMapWidth + 6`, 3-block border holding connected maps; 4 collision bytes per block come from the tileset collision table |
| trainer ray stops at walls | the game checks only same row/column + facing + distance <= sight range (`FacingPlayerDistance`); no obstruction test |
| trainer defeated flag "from wEventFlags" | first word of the trainer header the map object's script pointer points to is the flag id |
| counter = special tile class | WALL in `CollisionPermissionTable`; the interaction front is derived from an NPC behind it |
| "story target" entity class | removed: it is a goal hint (the repo removed route oracles) |
| steps == BFS length | a press toward a new direction first turns without moving; tests count MOVES |

## Observation: `(7, 15, 15)` uint8

0 terrain (9 classes, from the ROM's own LAND/WATER/WALL + TALK table), 1 entity (8 classes), 2 trainer sight, 3 entry-direction
mask (ledges, one-way walls), 4 counter front, 5 player facing, 6 visited. Zeros while a battle / text box is open.
The policy one-hot encodes the categorical planes (29 conv input channels), conv 3x3 -> conv 3x3/2 -> 4x4 pool -> 1024.

## Policy

`SpatialGridPolicy(SpatialVisionPolicy)`: same RAM trunk, fusion, GRU, heads, masking, aux head, PPO paths; ~0.7M parameters
with the defaults (ram 128, fusion 256, GRU 256). Checkpoints carry `grid_schema`; a pixel checkpoint cannot load as a grid policy.

## Not done yet

* Trainer / env integration: `rl_env`/`tools/train_deep_ppo.py` still produce screens. The policy accepts grids in the `screen`
  argument (`(B, 7, 15, 15)` / `(B, T, 7, 15, 15)`), so the buffer change is the only plumbing needed.
* Live ledge and trainer-sight tests skip: none of the committed states has one in reach.
* No A/B against the ResNet policy and no throughput measurement yet.
