"""Battle-relevant observation features, read straight from RAM, plus the starting supplies.

Battles are LEARNED, not scripted: the policy presses the buttons itself (see ``tools/train_deep_ppo.EnvWorker``).
To be able to learn fighting, switching, healing and item use, it gets these non-directional facts:

    in-battle flag, wild / trainer, enemy species + level + HP fraction + status,
    active mon species + level + HP fraction + status + party slot,
    party HP fractions and levels (6 slots), the active mon's 4 move ids and PP, which moves are usable,
    the battle-menu cursor (FIGHT / PKMN / PACK / RUN), the move-list cursor and the raw menu cursor x / y,
    number of healing items and of Poke Balls in the bag.

Nothing here says what to do. Outside a battle the battle-only fields are zero (the battle WRAM holds junk
there), while party and bag fields stay valid so the policy can learn to heal before it fights.

``equip_supplies`` gives the agent Poke Balls and Potions at every reset so catching and healing items can be
learned from the first episode (the start state outside New Bark only has the starter and one Potion).
"""

from __future__ import annotations

import numpy as np

MAX_SPECIES = 251.0
MAX_PARTY = 6
MAX_PP = 40.0
STATUS_BITS = ((0x07, "sleep"), (0x08, "poison"), (0x10, "burn"), (0x20, "freeze"), (0x40, "paralysis"))
N_STATUS = len(STATUS_BITS)

# Item ids (pokecrystal constants/item_constants.asm)
POKE_BALL = 0x05
POTION = 0x12
HEAL_ITEMS = frozenset({0x0E, 0x0F, 0x10, 0x11, 0x12})  # Full Restore, Max / Hyper / Super Potion, Potion
POTION_SUPPLY = 5
BALL_SUPPLY = 10

BATTLE_FEATURE_NAMES = (
    ("in_battle", "is_wild", "is_trainer")
    + ("enemy_species", "enemy_level", "enemy_hp")
    + tuple(f"enemy_{n}" for _, n in STATUS_BITS)
    + ("active_species", "active_level", "active_hp", "active_slot")
    + tuple(f"active_{n}" for _, n in STATUS_BITS)
    + tuple(f"party{i}_hp" for i in range(MAX_PARTY))
    + tuple(f"party{i}_level" for i in range(MAX_PARTY))
    + tuple(f"move{i}_id" for i in range(4))
    + tuple(f"move{i}_pp" for i in range(4))
    + tuple(f"move{i}_usable" for i in range(4))
    + ("menu_cursor", "move_cursor", "cursor_y", "cursor_x")
    + ("heal_items", "poke_balls")
)
BATTLE_FEATURE_DIM = len(BATTLE_FEATURE_NAMES)


def _frac(cur: int, mx: int) -> float:
    return float(cur) / float(mx) if mx > 0 else 0.0


def _status(byte: int) -> list[float]:
    return [1.0 if (byte & mask) else 0.0 for mask, _ in STATUS_BITS]


def heal_item_count(state) -> int:
    items = getattr(state, "items", ()) or ()
    counts = getattr(state, "item_counts", ()) or ()
    return int(sum(c for i, c in zip(items, counts) if i in HEAL_ITEMS))


def ball_count(state) -> int:
    balls = tuple(getattr(state, "balls", ()) or ())
    total = 0
    for i in range(0, len(balls) - 1, 2):
        if balls[i] == 255 or balls[i] == 0:
            break
        total += int(balls[i + 1])
    return total


def battle_features(state) -> np.ndarray:
    """BATTLE_FEATURE_DIM floats in [0, 1] (see the module docstring); never raises on missing fields."""
    battle = int(getattr(state, "battle_mode", 0)) != 0
    f: list[float] = [1.0 if battle else 0.0, 1.0 if getattr(state, "battle_mode", 0) == 1 else 0.0,
                      1.0 if getattr(state, "battle_mode", 0) == 2 else 0.0]
    if battle:
        f += [getattr(state, "enemy_mon_species", 0) / MAX_SPECIES, getattr(state, "enemy_level", 0) / 100.0,
              _frac(getattr(state, "enemy_hp", 0), getattr(state, "enemy_max_hp", 0))]
        f += _status(getattr(state, "enemy_status", 0))
        f += [getattr(state, "battle_mon_species", 0) / MAX_SPECIES, getattr(state, "battle_mon_level", 0) / 100.0,
              _frac(getattr(state, "battle_mon_hp", 0), getattr(state, "battle_mon_max_hp", 0)),
              getattr(state, "cur_battle_mon", 0) / float(MAX_PARTY)]
        f += _status(getattr(state, "battle_mon_status", 0))
    else:
        f += [0.0] * (3 + N_STATUS + 4 + N_STATUS)

    hp = list(getattr(state, "party_hp", ()) or ())[:MAX_PARTY]
    lv = list(getattr(state, "party_levels", ()) or ())[:MAX_PARTY]
    f += [_frac(c, m) for c, m in hp] + [0.0] * (MAX_PARTY - len(hp))
    f += [v / 100.0 for v in lv] + [0.0] * (MAX_PARTY - len(lv))

    if battle:
        moves = tuple(getattr(state, "battle_mon_moves", ()) or ())[:4]
        pps = [int(p) & 0x3F for p in (getattr(state, "battle_mon_pp", ()) or ())[:4]]  # top 2 bits are PP Ups
    else:  # lead mon's moves are known from the party, PP only in battle
        pm = getattr(state, "party_moves", ()) or ()
        moves = tuple(pm[0])[:4] if pm else ()
        pps = []
    moves = list(moves) + [0] * (4 - len(moves))
    pps = list(pps) + [0] * (4 - len(pps))
    f += [m / MAX_SPECIES for m in moves]
    f += [p / MAX_PP for p in pps]
    f += [1.0 if (battle and m > 0 and p > 0) else 0.0 for m, p in zip(moves, pps)]

    if battle:
        f += [getattr(state, "battle_menu_cursor", 0) / 4.0, getattr(state, "cur_move_num", 0) / 4.0,
              min(getattr(state, "menu_cursor_y", 0), 15) / 8.0, min(getattr(state, "menu_cursor_x", 0), 15) / 8.0]
    else:
        f += [0.0] * 4
    f += [min(heal_item_count(state), 20) / 10.0, min(ball_count(state), 40) / 20.0]
    out = np.asarray(f, dtype=np.float64)
    assert out.shape == (BATTLE_FEATURE_DIM,), out.shape
    return np.clip(out, 0.0, 1.5)


def equip_supplies(emu, balls: int = BALL_SUPPLY, potions: int = POTION_SUPPLY) -> None:
    """Make sure the bag holds at least ``balls`` Poke Balls and ``potions`` Potions (never removes anything).

    Hard-coded on purpose and documented: it lets healing and catching be learned from episode one. Any failure
    (missing symbol, odd RAM) is ignored so a start state is never broken by it.
    """
    try:
        nb, ab = emu.symbol("wNumBalls")
        pb, pa = emu.symbol("wBalls")
        raw = emu.read(pb, pa, 25)
        pairs = []
        for i in range(0, 24, 2):
            if raw[i] in (0, 255):
                break
            pairs.append([raw[i], raw[i + 1]])
        for p in pairs:
            if p[0] == POKE_BALL:
                p[1] = max(p[1], balls)
                break
        else:
            pairs.append([POKE_BALL, balls])
        emu.write(nb, ab, bytes([len(pairs)]))
        emu.write(pb, pa, bytes([v for p in pairs for v in p] + [255]))
    except Exception:
        pass
    try:
        ni, ai = emu.symbol("wNumItems")
        ib, ia = emu.symbol("wItems")
        n = emu.read(ni, ai, 1)[0]
        raw = emu.read(ib, ia, 41)
        pairs = [[raw[2 * i], raw[2 * i + 1]] for i in range(min(n, 20)) if raw[2 * i] not in (0, 255)]
        for p in pairs:
            if p[0] == POTION:
                p[1] = max(p[1], potions)
                break
        else:
            if len(pairs) < 20:
                pairs.append([POTION, potions])
        emu.write(ni, ai, bytes([len(pairs)]))
        emu.write(ib, ia, bytes([v for p in pairs for v in p] + [255]))
    except Exception:
        pass
