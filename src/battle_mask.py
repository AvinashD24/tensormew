"""Valid-action masking for the learned battle (flag ``battle_action_mask``, OFF everywhere except ``long_v8``).

Idea (Huang & Ontanon 2020, "A closer look at invalid action masking in policy gradient algorithms"): when some
actions provably do nothing in the current state, set their logits to a large negative number BEFORE the softmax so
they are never sampled (and carry no gradient); the policy-gradient estimator stays valid because the SAME masked
distribution is used for sampling, for the stored log-probs and for the PPO re-evaluation. Here the masked
actions are the buttons that provably leave a battle menu / text box unchanged, so exploration is not spent on them.

The context is read from the BACKGROUND TILEMAP (``wTilemap``, 20 x 18 tile ids: what the player sees) plus the moves
byte string; the cursor RAM (``wMenuCursorY/X``, ``wBattleMenuCursorPosition``) is stale outside the menu that owns it,
so it is NOT used to decide the context. Every rule below was checked against the real emulator (press the button in
that state with the worker's own ``press(8 frames) + tick(12)`` and compare the screen, the tilemap and the menu / battle
WRAM; see ``tests/test_battle_mask.py`` and the README design note). Unknown screens return an all-True mask.

Verified rules (wild battle, Crystal ROM)::

    main FIGHT/PKMN/PACK/RUN 2 x 2 menu  edges do not wrap: FIGHT up,left | PKMN up,right | PACK down,left | RUN down,right
                                        are no-ops; B is a no-op everywhere in this menu.
    move list (cursor on a move)        left, right are no-ops; with ONE move up and down are no-ops too (with 2-4 moves
                                        the list WRAPS, so up/down are real moves and are NOT masked).
    party list (solid arrow in col 0)   left, right are no-ops (up/down wrap, A selects, B leaves: not masked).
    battle text box (no cursor drawn)   up, down, left, right are no-ops (A and B advance / dismiss: never masked).

NOT masked (no rule, or not verified): the bag / item list, yes/no prompts, the party sub-menu, summary pages, every
animation / transition frame, anything outside a battle (``battle_mode == 0`` -> all True).
"""

from __future__ import annotations

import numpy as np

UP, DOWN, LEFT, RIGHT, BTN_A, BTN_B = range(6)
N_ACTIONS = 6
ACTION_NAMES = ("up", "down", "left", "right", "a", "b")

TILEMAP_ADDR = 0xC4A0   # wTilemap (WRAM bank 0)
TILEMAP_W, TILEMAP_H = 20, 18
TILEMAP_SIZE = TILEMAP_W * TILEMAP_H

ARROW, HOLLOW_ARROW = 0xED, 0xEC
BOX_TL, BOX_H, BOX_TR, BOX_V, BOX_BL, BOX_BR = 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E
FIGHT_TILES = (0x85, 0x88, 0x86, 0x87, 0x93)   # "FIGHT" in the game's charmap
_MAIN_CELLS = {(14, 9): (0, 0), (14, 15): (0, 1), (16, 9): (1, 0), (16, 15): (1, 1)}   # arrow (row, col) -> (grid row, col)

CTX_OTHER, CTX_MAIN, CTX_MOVES, CTX_PARTY, CTX_TEXT = "other", "main", "moves", "party", "text"


def all_true() -> np.ndarray:
    return np.ones(N_ACTIONS, dtype=bool)


def _tile(tm, row: int, col: int) -> int:
    return int(tm[row * TILEMAP_W + col])


def _arrows(tm, code: int) -> list[tuple[int, int]]:
    return [(i // TILEMAP_W, i % TILEMAP_W) for i in range(TILEMAP_SIZE) if tm[i] == code]


def _text_box(tm) -> bool:
    """The full-width 6-row battle text box (rows 12-17) is on screen."""
    if _tile(tm, 12, 0) != BOX_TL or _tile(tm, 12, 19) != BOX_TR:
        return False
    if _tile(tm, 17, 0) != BOX_BL or _tile(tm, 17, 19) != BOX_BR:
        return False
    if not all(_tile(tm, r, 0) == BOX_V and _tile(tm, r, 19) == BOX_V for r in range(13, 17)):
        return False
    # A menu box being drawn INSIDE the text area (the frames before its cursor appears) is not a text box: the next
    # frame's input would be read by that menu.
    if any(_tile(tm, 12, c) != BOX_H or _tile(tm, 17, c) != BOX_H for c in range(1, 19)):
        return False
    if any(_tile(tm, r, c) == BOX_V for r in range(13, 17) for c in range(1, 19)):
        return False
    # Text must already be printed (an empty box is a transition frame).
    return any(_tile(tm, r, c) not in (0x7F, 0x00) for r in (14, 16) for c in range(1, 19))


def classify(tm, n_moves: int = 4) -> tuple[str, tuple]:
    """Context of the battle screen described by ``tm`` (``TILEMAP_SIZE`` tile ids): ``(kind, detail)``.

    ``main`` -> ``(grid_row, grid_col)``; ``moves`` -> ``(index,)``; ``party`` -> ``(party_row,)``; ``text`` / ``other`` -> ``()``.
    """
    if tm is None or len(tm) < TILEMAP_SIZE:
        return CTX_OTHER, ()
    arrows = _arrows(tm, ARROW)
    hollow = _arrows(tm, HOLLOW_ARROW)
    if not arrows and not hollow:
        return (CTX_TEXT, ()) if _text_box(tm) else (CTX_OTHER, ())
    if len(arrows) != 1 or hollow:
        return CTX_OTHER, ()
    r, c = arrows[0]
    if (r, c) in _MAIN_CELLS and tuple(_tile(tm, 14, 10 + k) for k in range(5)) == FIGHT_TILES \
            and _tile(tm, 12, 8) == BOX_TL:
        return CTX_MAIN, _MAIN_CELLS[(r, c)]
    if c == 5 and 13 <= r <= 16 and _tile(tm, r, 4) == BOX_V and _tile(tm, 12, 0) == BOX_BL \
            and _tile(tm, 17, 4) == BOX_BL and _tile(tm, 8, 0) == BOX_TL and r - 13 < max(1, int(n_moves)):
        return CTX_MOVES, (r - 13,)
    if c == 0 and 1 <= r <= 11 and r % 2 == 1 and _tile(tm, 14, 0) == BOX_TL and _tile(tm, 17, 0) == BOX_BL:
        return CTX_PARTY, (r,)
    return CTX_OTHER, ()


def mask_for(kind: str, detail: tuple = (), n_moves: int = 4) -> np.ndarray:
    """Boolean mask over (up, down, left, right, a, b) for a classified context; True = allowed."""
    m = all_true()
    if kind == CTX_MAIN:
        row, col = detail
        if row == 0:
            m[UP] = False
        if row == 1:
            m[DOWN] = False
        if col == 0:
            m[LEFT] = False
        if col == 1:
            m[RIGHT] = False
        m[BTN_B] = False
    elif kind == CTX_MOVES:
        m[LEFT] = m[RIGHT] = False
        if int(n_moves) <= 1:
            m[UP] = m[DOWN] = False
    elif kind == CTX_PARTY:
        m[LEFT] = m[RIGHT] = False
    elif kind == CTX_TEXT:
        m[UP] = m[DOWN] = m[LEFT] = m[RIGHT] = False
    return sanitize(m)


def sanitize(m: np.ndarray) -> np.ndarray:
    """Safety net: A and B are never both masked and the mask is never all False."""
    if not m.any() or (not m[BTN_A] and not m[BTN_B]):
        return all_true()
    return m


def moves_count(moves) -> int:
    return sum(1 for v in (moves or ()) if int(v) != 0)


def battle_mask_from_tilemap(tm, n_moves: int = 4) -> np.ndarray:
    kind, detail = classify(tm, n_moves)
    return mask_for(kind, detail, n_moves)


def read_tilemap(emu):
    return bytes(emu.read(0, TILEMAP_ADDR, TILEMAP_SIZE))


def battle_action_mask(emu, state) -> np.ndarray:
    """Mask for the current emulator state: all True outside a battle, otherwise by the visible battle screen."""
    if int(getattr(state, "battle_mode", 0)) == 0:
        return all_true()
    try:
        return battle_mask_from_tilemap(read_tilemap(emu), moves_count(getattr(state, "battle_mon_moves", ())) or 4)
    except Exception:   # a failed read must never break training: no mask is always safe
        return all_true()


def pack_mask(m: np.ndarray) -> int:
    return int(sum(1 << i for i in range(N_ACTIONS) if bool(m[i])))


def unpack_mask(bits: int) -> np.ndarray:
    return np.array([bool((int(bits) >> i) & 1) for i in range(N_ACTIONS)], dtype=bool)
