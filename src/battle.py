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


class _ItemGuard:
    """Stops the bot re-opening PACK every turn when an item is not actually healing.

    In trainer battles the game rejects an unusable item and drops back to the menu, so
    "use item" repeated forever (hundreds of turns in live runs). Two tries without an HP gain
    block items until HP rises, the battle ends, or enough frames pass to be a new battle.
    """

    MAX_FAILS = 2
    NEW_BATTLE_FRAMES = 3000

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_hp = None
        self.fails = 0
        self.last_frame = None
        self.fainted: set[int] = set()   # party slots known to be unusable this battle
        self.last_switch: int | None = None  # slot we tried to bring in last time
        self.attempts: dict[int, int] = {}   # failed switch tries per party slot

    def observe_frame(self, frame) -> None:
        """A long gap since the last decision means a new battle: forget per-battle memory."""
        if isinstance(frame, int):
            if self.last_frame is not None and frame - self.last_frame > self.NEW_BATTLE_FRAMES:
                self.reset()
            self.last_frame = frame

    def allows(self, cur_hp: int, frame) -> bool:
        self.observe_frame(frame)
        if self.last_hp is not None:
            if cur_hp > self.last_hp:
                self.fails = 0
            else:
                self.fails += 1
        return self.fails < self.MAX_FAILS

    def note_use(self, cur_hp: int):
        self.last_hp = cur_hp


def pick_replacement_slot(state, guard: "_ItemGuard") -> int | None:
    """Hardcoded choice of which party slot (1-indexed) to send in next, or None if nothing is alive.

    Only real party members count (padding slots past party_count are ignored), the active Pokémon
    is never picked, and slots already written off this battle are skipped. Fewest failed tries wins,
    then most HP, then highest level. If the book-keeping has ruled out every Pokémon that still shows
    HP, the memory is wiped and they are tried again rather than looping on the whiteout dialog.
    """
    party_hp = tuple(getattr(state, "party_hp", ()) or ())
    levels = tuple(getattr(state, "party_levels", ()) or ())
    count = getattr(state, "party_count", None)
    n = len(party_hp) if not isinstance(count, int) or count <= 0 else min(count, len(party_hp))
    active_slot = min(getattr(state, "cur_battle_mon", 0), max(n - 1, 0)) + 1

    def candidates(skip: set[int]) -> list[int]:
        return [i + 1 for i in range(n) if party_hp[i][0] > 0 and i + 1 != active_slot and (i + 1) not in skip]

    cands = candidates(guard.fainted)
    if not cands:
        cands = candidates(set())
        if not cands:
            return None
        guard.fainted.clear()
        guard.attempts.clear()

    def key(slot: int):
        lvl = levels[slot - 1] if slot - 1 < len(levels) else 0
        return (-guard.attempts.get(slot, 0), party_hp[slot - 1][0], lvl)

    return max(cands, key=key)


_GUARDS: dict[int, _ItemGuard] = {}


def _item_guard(macros) -> _ItemGuard:
    return _GUARDS.setdefault(id(macros), _ItemGuard())


def tactical_battle_step(state, macros, enemy_types: tuple[int, ...] = (2, 0)):
    """Execute optimal combat/capture/item/switch action:
    - 0. Faint Recovery & Switch:
         - If all party Pokémon are fainted, clear whiteout dialogue to respawn.
         - If active Pokémon is fainted (cur_hp == 0), find first conscious party
           member and switch to it immediately!
    - 1. In-Battle Item Usage:
         - In trainer battles (where fleeing is impossible), if active mon is low HP
           (<= 30%) and pack has items, use item (Potion) to heal!
    - 2. Self-Preservation: In wild battles, if active Pokémon HP is low (<= 30%),
         flee immediately to prevent getting knocked out / white-out!
    - 3. Goal-Oriented Progress (Anti-Grinding): In wild battles, if species is already
         caught and party has sufficient level (>= 7), flee immediately to stop blind
         grinding and resume advancing toward the Gym!
    - 4. Strategic Capture: In wild battles with open party and balls:
         - Weaken first if enemy HP > 40%.
         - Catch with Pokéball when HP <= 40%.
    - 5. Tactical Switching (Healthy Reserve): In trainer battles, if active mon is
         at critical HP, no healing items remain, and a conscious backup mon exists
         with healthy HP (> 50%), switch to it!
    - 6. Decisive Combat: Attack using the highest-scoring move against defending types.
    """
    mode = getattr(state, "battle_mode", 0)

    # 1. Health check: active Pokémon HP
    party_hp = getattr(state, "party_hp", ())
    party_levels = getattr(state, "party_levels", ())
    active_idx = getattr(state, "cur_battle_mon", 0)
    active_idx = min(active_idx, len(party_hp) - 1) if party_hp else 0

    b_cur = getattr(state, "battle_mon_hp", 0)
    b_max = getattr(state, "battle_mon_max_hp", 0)
    if b_max > 0:
        cur_hp, max_hp = b_cur, b_max
    else:
        cur_hp = party_hp[active_idx][0] if party_hp else 1
        max_hp = party_hp[active_idx][1] if party_hp else 1

    guard = _item_guard(macros)
    guard.observe_frame(getattr(getattr(macros, "machine", None), "frame", None))

    # 0. Faint Check: active Pokémon fainted
    if cur_hp == 0:
        guard.fainted.add(active_idx + 1)
        if guard.last_switch is not None:
            # The previous pick did not bring anyone in; two misses and the slot is written off
            n = guard.attempts.get(guard.last_switch, 0) + 1
            guard.attempts[guard.last_switch] = n
            if n >= 2:
                guard.fainted.add(guard.last_switch)
        slot = pick_replacement_slot(state, guard)
        if slot is None:
            guard.last_switch = None
            return macros.handle_whiteout()
        guard.last_switch = slot
        return macros.switch_fainted_pokemon(slot)

    guard.last_switch = None  # active Pokémon is up: the last switch (if any) worked

    is_low_hp = max_hp > 0 and (cur_hp / max_hp) <= 0.30
    num_items = getattr(state, "num_items", 0)

    # 1. TRAINER BATTLES (mode == 2): Running away is physically impossible in Gen 2.
    # Under no circumstances should the policy ever attempt to flee a trainer battle.
    if mode == 2:
        if not is_low_hp:
            guard.last_hp, guard.fails = None, 0
        if is_low_hp:
            # First priority: heal active Pokémon if Pack has items (e.g. Potions), unless the
            # last tries did not raise HP (item rejected): then switch or fight instead.
            frame = getattr(getattr(macros, "machine", None), "frame", None)
            if num_items > 0 and guard.allows(cur_hp, frame):
                guard.note_use(cur_hp)
                return macros.use_battle_item(item_slot=1, target_slot=active_idx + 1)
            # Second priority: switch to a conscious backup Pokémon if low HP and out of items
            backup = pick_replacement_slot(state, guard)
            if backup is not None:
                return macros.switch_pokemon(backup)
        # Decisive combat: Fight to the end using the best offensive move!
        slot = pick_best_move(state, enemy_types)
        return macros.choose_move(slot)

    # 2. WILD BATTLES (mode == 1): Strategic capture, self-preservation, and anti-grinding
    has_event = getattr(state, "has_event", None)
    has_egg = callable(has_event) and has_event("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
    gave_egg = callable(has_event) and has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM")

    if mode == 1:
        # Flee if active Pokémon is at risk of fainting
        if is_low_hp:
            return macros.run_away()

        # Urgent Quest Delivery: When returning Mystery Egg to Elm, avoid wild battle distractions
        if has_egg and not gave_egg:
            return macros.run_away()

        enemy_spec = getattr(state, "enemy_species", 0)
        party_spec = getattr(state, "party_species", ())
        already_caught = enemy_spec > 0 and enemy_spec in party_spec

        # Anti-grinding: if already caught and level >= 7, flee to progress towards gym
        lead_level = party_levels[0] if party_levels else 5
        if already_caught and lead_level >= 7:
            return macros.run_away()

        # Capture path for uncaptured wild Pokémon
        if not already_caught and getattr(state, "num_balls", 0) > 0 and getattr(state, "party_count", 0) < 6:
            e_cur = getattr(state, "enemy_hp", 0)
            e_max = getattr(state, "enemy_max_hp", 0)
            if e_max > 0 and (e_cur / e_max) > 0.40:
                slot = pick_best_move(state, enemy_types)
                return macros.choose_move(slot)
            return macros.use_pokeball()

    # Fallback combat
    slot = pick_best_move(state, enemy_types)
    return macros.choose_move(slot)



