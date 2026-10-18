"""Combat logic and intelligent move selection for Generation 2 battles."""

from __future__ import annotations

# Gen 2 Type Chart multiplier lookup: (attacking_type, defending_type) -> multiplier
# Types: 0=normal, 1=fighting, 2=flying, 3=poison, 4=ground, 5=rock, 6=bug, 7=ghost, 8=steel,
# 9=fire, 10=water, 11=grass, 12=electric, 13=psychic, 14=ice, 15=dragon, 16=dark
TYPE_CHART: dict[tuple[int, int], float] = {
    # Ground vs Flying is immune (0.0x) - Mud-slap against Falkner does 0!
    (4, 2): 0.0,
    # Electric vs Flying (2.0x)
    (12, 2): 2.0,
    # Rock vs Flying (2.0x)
    (5, 2): 2.0,
    # Ice vs Flying (2.0x)
    (14, 2): 2.0,
    # Grass vs Flying (0.5x)
    (11, 2): 0.5,
    # Bug vs Flying (0.5x)
    (6, 2): 0.5,
    # Fighting vs Flying (0.5x)
    (1, 2): 0.5,
    # Ghost vs Normal (0.0x)
    (7, 0): 0.0,
    # Normal vs Ghost (0.0x)
    (0, 7): 0.0,
    # Fire vs Grass/Bug/Steel/Ice (2.0x)
    (9, 11): 2.0,
    (9, 6): 2.0,
    (9, 8): 2.0,
    (9, 14): 2.0,
    # Water vs Fire/Ground/Rock (2.0x)
    (10, 9): 2.0,
    (10, 4): 2.0,
    (10, 5): 2.0,
}

# Common Gen 2 Early-Game Moves: id -> (name, type, power, accuracy)
MOVE_DATA: dict[int, tuple[str, int, int, int]] = {
    1: ("Pound", 0, 40, 100),
    2: ("Karate Chop", 1, 50, 100),
    10: ("Scratch", 0, 40, 100),
    33: ("Tackle", 0, 35, 95),
    43: ("Leer", 0, 0, 100),
    45: ("Growl", 0, 0, 100),
    52: ("Ember", 9, 40, 100),
    55: ("Water Gun", 10, 40, 100),
    71: ("Absorb", 11, 20, 100),
    72: ("Mega Drain", 11, 40, 100),
    84: ("Thunder Shock", 12, 40, 100),
    98: ("Quick Attack", 0, 40, 100),
    106: ("Harden", 0, 0, 100),
    108: ("Smokescreen", 0, 0, 100),
    111: ("Defense Curl", 0, 0, 100),
    189: ("Mud-Slap", 4, 20, 100),
}


def move_power(move_id: int) -> int:
    return MOVE_DATA.get(move_id, ("Unknown", 0, 40, 100))[2]


def move_type(move_id: int) -> int:
    return MOVE_DATA.get(move_id, ("Unknown", 0, 40, 100))[1]


def type_multiplier(atk_type: int, def_type: int) -> float:
    return TYPE_CHART.get((atk_type, def_type), 1.0)


def score_move(move_id: int, enemy_types: tuple[int, ...] = (2, 0)) -> float:
    """Score a move against enemy defending types (default Flying/Normal for Falkner)."""
    if move_id == 0:
        return -1.0
    name, m_type, power, acc = MOVE_DATA.get(move_id, ("Unknown", 0, 35, 95))
    if power == 0:
        # Non-damaging status move has lower priority during rush to KO
        return 5.0
    mult = 1.0
    for dt in enemy_types:
        mult *= type_multiplier(m_type, dt)
    return float(power) * mult * (acc / 100.0)


def pick_best_move(state, enemy_types: tuple[int, ...] = (2, 0)) -> int:
    """Pick 1-indexed move slot (1..4) with highest combat effectiveness."""
    if not state.party_moves:
        return 1
    moves = state.party_moves[0]  # Active Pokémon moves
    best_slot = 1
    best_score = -1.0
    for slot, move_id in enumerate(moves, start=1):
        if move_id == 0:
            continue
        score = score_move(move_id, enemy_types)
        if score > best_score:
            best_score = score
            best_slot = slot
    return best_slot


def tactical_battle_step(state, macros, enemy_types: tuple[int, ...] = (2, 0)):
    """Execute optimal combat/capture action:
    - In wild battles (battle_mode == 1) with available balls and open party:
      - If species is not already caught in party:
        - Weaken first if enemy HP > 40%.
        - Catch with Pokéball when HP <= 40%.
    - In trainer battles, or when already caught/no balls:
      - Attack using the best move to KO and gain EXP.
    """
    if getattr(state, "battle_mode", 0) == 1 and getattr(state, "num_balls", 0) > 0 and getattr(state, "party_count", 0) < 6:
        enemy_spec = getattr(state, "enemy_species", 0)
        party_spec = getattr(state, "party_species", ())
        already_caught = enemy_spec > 0 and enemy_spec in party_spec
        if not already_caught:
            cur_hp = getattr(state, "enemy_hp", 0)
            max_hp = getattr(state, "enemy_max_hp", 0)
            if max_hp > 0 and (cur_hp / max_hp) > 0.40:
                slot = pick_best_move(state, enemy_types)
                return macros.choose_move(slot)
            return macros.use_pokeball()
    slot = pick_best_move(state, enemy_types)
    return macros.choose_move(slot)

