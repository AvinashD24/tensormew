"""Overworld environment. Rewards come from new tiles, maps, story milestones, gyms, and caught Pokémon."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ram_parser import dialogue_open
from src.local_nav import NAV_DIM, LocalNav
from src.step import DIRS
from src.story_flags import EVENT_FLAGS, REVERSE_EVENT_FLAGS

ACTIONS = DIRS + ("a",)
N_ACTIONS = len(ACTIONS)
# 4 neighbor tiles, xy, badges, battle, dialogue, 4 visited-neighbor flags,
# party_count, max_level, caught_count, story_stage
BASE_OBS_DIM = 17
# Then the LocalNav block: blocked mask (indices MASK_START..+4), last action, frontier, 5x5 grid.
MASK_START = BASE_OBS_DIM
# Which story milestones are done, as a multi-hot (the "events vector" in the pokemonred_puffer
# write-up). Order is fixed; the checkpoint width depends on it.
OBS_EVENT_FLAGS = (
    "EVENT_GOT_A_POKEMON_FROM_ELM",
    "EVENT_LEARNED_TO_CATCH_POKEMON",
    "EVENT_RIVAL_CHERRYGROVE_CITY",
    "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON",
    "EVENT_GAVE_MYSTERY_EGG_TO_ELM",
    "EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST",
    "EVENT_GOT_TM31_MUD_SLAP",
    "EVENT_BEAT_FALKNER",
    "EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE",
    "EVENT_GOT_HM01_CUT",
)
OBS_DIM = BASE_OBS_DIM + NAV_DIM + len(OBS_EVENT_FLAGS)
_NEIGHBOR_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
UNVISITED_NEIGHBOR_BONUS = 0.2
# Frontier-distance shaping (potential-based: gamma * phi(s') - phi(s)). Scale is per tile of
# progress toward the nearest unvisited open tile when it is 20 tiles away.
FRONTIER_SHAPING_WEIGHT = 4.0
SHAPING_GAMMA = 0.99
# Puffer-style one-time exploration payments for things that are not just new coordinates.
NEW_INTERACTION_REWARD = 0.5  # first A press that opens a script from a given tile and facing
NEW_WARP_REWARD = 1.0         # first time a map-to-map transition is taken
# Stepping back into a map the run has already finished ends the episode with this penalty.
BEHIND_MAP_PENALTY = 2.0

# Key Transit Maps along the path to the First Gym (Falkner in Violet City)
GYM_PATH_WAYPOINTS: dict[tuple[int, int], float] = {
    (24, 3): 3.0,   # Route 29
    (26, 3): 5.0,   # Cherrygrove City
    (26, 5): 3.0,   # Cherrygrove Poké Center
    (26, 1): 5.0,   # Route 30
    (26, 10): 8.0,  # Mr. Pokémon's House (Egg quest)
    (26, 2): 8.0,   # Route 31
    (26, 11): 5.0,  # Route 31 Violet Gate
    (10, 5): 12.0,  # Violet City
    (10, 10): 3.0,  # Violet Poké Center
    (10, 7): 20.0,  # Violet Gym
}

# Major Story Milestone Rewards
KEY_EVENT_REWARDS: dict[str, float] = {
    "EVENT_GOT_A_POKEMON_FROM_ELM": 15.0,
    "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON": 20.0,
    "EVENT_GAVE_MYSTERY_EGG_TO_ELM": 20.0,
    "EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST": 10.0,
    "EVENT_RIVAL_CHERRYGROVE_CITY": 15.0,
    "EVENT_LEARNED_TO_CATCH_POKEMON": 5.0,
    "EVENT_BEAT_FALKNER": 50.0,
    "EVENT_GOT_TM31_MUD_SLAP": 10.0,
    "EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE": 10.0,
    "EVENT_GOT_HM01_CUT": 20.0,
}


OBJECTIVE_TILE_BOOST = 2.0

# Level reward shaped like the pokemonred write-up: full credit for early levels, then a quarter.
# Continuous at LEVEL_CAP_SUM (the write-up's version jumps by 15 there, which would dwarf a gym badge).
LEVEL_REWARD_SCALE = 5.0
LEVEL_CAP_SUM = 15


def level_reward_value(level_sum: float) -> float:
    if level_sum < LEVEL_CAP_SUM:
        return float(level_sum)
    return LEVEL_CAP_SUM + (level_sum - LEVEL_CAP_SUM) / 4.0

_STAGE_OBJECTIVE_MAPS: dict[str, frozenset] = {
    # Fetch the Mystery Egg: Route 29 -> Cherrygrove -> Route 30 -> Mr. Pokemon's house.
    "get_egg": frozenset({(24, 3), (26, 3), (26, 1), (26, 10)}),
    # Bring it back to Elm.
    "return_egg": frozenset({(26, 10), (26, 1), (26, 3), (24, 3), (24, 4), (24, 5)}),
    # Walk to Violet City and the gym.
    "to_gym": frozenset({(24, 3), (26, 3), (26, 1), (26, 2), (26, 11), (10, 5), (10, 7)}),
    # After Falkner: the Togepi egg in Violet's Pokemon Center.
    "post_gym": frozenset({(10, 5), (10, 7), (10, 10)}),
}


def story_objective_maps(state) -> frozenset:
    """Maps where the current story objective lives. New tiles there pay OBJECTIVE_TILE_BOOST x.

    This is the "map boost until the objective is done" idea from pokemonred_puffer: the agent
    cannot read the dialogue that says where to go, so exploring the right maps pays more.
    """
    has_event = getattr(state, "has_event", None)
    if not callable(has_event):
        return frozenset()
    if not has_event("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"):
        return _STAGE_OBJECTIVE_MAPS["get_egg"]
    if not has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM"):
        return _STAGE_OBJECTIVE_MAPS["return_egg"]
    if has_event("EVENT_BEAT_FALKNER") and not has_event("EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE"):
        return _STAGE_OBJECTIVE_MAPS["post_gym"]
    if has_event("EVENT_BEAT_FALKNER"):
        return frozenset()
    return _STAGE_OBJECTIVE_MAPS["to_gym"]


def catch_reward_for_species(count: int, base: float = 10.0) -> float:
    """Calculate reward for catching a Pokémon species with diminishing returns for duplicates.

    1st catch of species: 10.0 (high novelty bonus)
    2nd catch of species: 5.0 (duplicate, 50%)
    3rd catch of species: 2.5 (duplicate, 25%)
    4th+ catch: max(0.5, 10.0 * 0.5**(count - 1))
    """
    if count <= 1:
        return base
    return max(0.5, base * (0.5 ** (count - 1)))


def story_stage(state) -> float:
    """Return normalized story progress from 0.0 (start) to 1.0 (badge earned)."""
    if getattr(state, "badges", 0) > 0:
        return 1.0
    has_event = getattr(state, "has_event", None)
    if callable(has_event):
        if has_event("EVENT_BEAT_FALKNER"):
            return 1.0
        if has_event("EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST"):
            return 0.65
        if has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM"):
            return 0.55
        if has_event("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"):
            return 0.45
        if has_event("EVENT_RIVAL_CHERRYGROVE_CITY"):
            return 0.35
        if has_event("EVENT_GOT_A_POKEMON_FROM_ELM"):
            return 0.20
    map_id = getattr(state, "map_id", (getattr(state, "map_group", 0), getattr(state, "map_number", 0)))
    if map_id == (10, 7):
        return 0.85
    if map_id == (10, 5):
        return 0.75
    return 0.0


def _route29_potential(x: float, y: float, base: float = 0.0) -> float:
    """Accurate topological potential for Route 29.

    Topological structure:
    1. Southern trap (x < 31, y >= 11): Enclosed on North by an impassable hop-down ledge
       and on West by walls at x=14. The ONLY escape is East back towards x=31.
    2. 1-space-wide choke point at x=31 (y in 10..16): The critical gap cutting vertically
       through the ledge barrier into the northern route.
    3. Transition corridor (31 <= x <= 37, 8 <= y <= 10): Moves East-North around trees.
    4. Northern Highway (x <= 37, y <= 7): Clear horizontal route running West directly to
       Cherrygrove City at (0, 6).
    """
    if x < 31 and y >= 10:
        # Southern trap and Cut-tree dead-end: heavily penalize moving west; reward moving east
        return base + 5.0 + (x - 14) * 0.3
    elif 31 <= x <= 32 and 10 <= y <= 16:
        # Choke point gap: strongly reward moving UP through the 1-tile gap
        return base + 16.0 + (16 - y) * 1.5
    elif x <= 37 and y <= 7:
        # Northern highway heading west to Cherrygrove City (x -> 0)
        return base + 31.0 + (37 - x) * 0.35
    elif 31 <= x <= 37 and 8 <= y <= 10:
        # Connector into northern highway: moves East-North around trees
        return base + 25.0 + (x - 31) * 0.4 + (10 - y) * 1.0
    else:
        # Eastern approach from New Bark Town (x in 33..59)
        return base + 10.0 + max(0.0, (59 - x) * 0.2)


def is_party_low_hp(state) -> bool:
    """Return True if any party member is not at full health."""
    hp_list = getattr(state, "party_hp", ()) or ()
    if not hp_list:
        return False
    for cur, max_hp in hp_list:
        if max_hp > 0 and cur < max_hp:
            return True
    return False


def _pokecenter_potential(x: float, y: float, base: float, is_low_hp: bool) -> float:
    """Potential gradient inside a Pokémon Center.

    If low HP: pulls player towards Nurse Joy at counter (3, 3) facing UP.
    If fully healed: pulls player towards exit door at (3.5, 7.0) to return to overworld.
    """
    if is_low_hp:
        # Distance to counter at (3, 3) facing UP
        dist_x = abs(x - 3.0)
        dist_y = abs(y - 3.0)
        dist = dist_x * 1.5 + dist_y
        return base + max(0.0, 8.0 - 0.7 * dist)
    else:
        # Distance to exit doorway warp at (3.5, 8.0) - stepping down at y=7 is rewarded
        dist = abs(x - 3.5) + abs(y - 8.0)
        return base + max(0.0, 7.0 - 0.7 * dist)


def _building_exit_potential(x: float, y: float, base: float) -> float:
    """Guides agent towards the exit door warp (y=8.0) of incidental indoor buildings."""
    dist = abs(x - 2.5) + abs(y - 8.0)
    return base + max(0.0, 6.0 - 0.7 * dist)


def navigation_potential(state) -> float:
    """Potential-based reward shaping Phi(s) guiding the agent through the complete story quest.

    Creates an invariant directional gradient (Ng, Harada & Russell, 1999) that rewards
    moving forward along the transit corridor and penalizes retreating backwards.
    """
    map_id = getattr(state, "map_id", (getattr(state, "map_group", 0), getattr(state, "map_number", 0)))
    x = float(getattr(state, "x", 0))
    y = float(getattr(state, "y", 0))

    has_event = getattr(state, "has_event", None)
    has_egg = callable(has_event) and has_event("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
    gave_egg = callable(has_event) and has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
    beat_falkner = callable(has_event) and has_event("EVENT_BEAT_FALKNER")
    has_aide_egg = callable(has_event) and has_event("EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE")
    low_hp = is_party_low_hp(state)

    # Pokémon Center handling across all stages
    if map_id in ((26, 5), (10, 10), (10, 13)):
        # In Stage 4 (post-Falkner), Violet PC (10, 10) holds Elm's Aide with the Togepi Egg
        if beat_falkner and not has_aide_egg and map_id == (10, 10):
            # Elm's Aide is at (4, 3); target is standing in front at (4, 4)
            dist_aide = abs(x - 4.0) + abs(y - 4.0)
            return 260.0 + max(0.0, 10.0 - 1.0 * dist_aide)
        base_pc = 245.0 if map_id == (10, 10) else (170.0 if gave_egg else 45.0)
        return _pokecenter_potential(x, y, base_pc, low_hp)

    # Incidental indoor buildings: guide agent straight back out to overworld
    if map_id in ((20, 1), (20, 2)):  # Pokecenter 2F / Cable Club -> exit stairs at (0, 7)
        dist = abs(x - 0.5) + abs(y - 7.5)
        return 40.0 + max(0.0, 6.0 - 0.7 * dist)
    if map_id in ((26, 4), (26, 6), (26, 7), (26, 8), (26, 9)):  # Cherrygrove Mart & houses
        return _building_exit_potential(x, y, base=40.0)
    if map_id in ((24, 6), (24, 7), (24, 8), (24, 9)):  # New Bark houses
        return _building_exit_potential(x, y, base=5.0)
    if map_id in ((10, 6), (10, 8), (10, 9), (10, 11)):  # Violet Mart & houses
        return _building_exit_potential(x, y, base=220.0)

    # Stage 1: Heading to Mr. Pokémon / Cherrygrove (prior to getting egg)
    if not has_egg:
        if map_id == (24, 4):  # New Bark Town -> exit west into Route 29 (x -> 0)
            return max(0.0, 10.0 - x * 0.5)
        elif map_id == (24, 3):  # Route 29 -> exit west towards Cherrygrove
            return _route29_potential(x, y, base=0.0)
        elif map_id in ((26, 3), (24, 7)):  # Cherrygrove City -> exit north towards Route 30 (y -> 0)
            if low_hp:
                # When low HP, guide toward Pokecenter at (29, 3)
                dist_pc = abs(x - 29.0) + abs(y - 3.0)
                return 45.0 + max(0.0, 6.0 - 0.3 * dist_pc)
            return 45.0 + max(0.0, (20.0 - y) * 0.5)
        elif map_id == (26, 1):  # Route 30 -> head to Mr. Pokémon's House at (17, 5)
            # Route 30 branch geometry: left side (x < 12, y < 40) is blocked by battling trainers!
            if y > 40.0:
                # South entrance: move north and east towards fork at (15, 40)
                dist_fork = abs(x - 15.0) + abs(y - 40.0)
                return 55.0 + max(0.0, 15.0 - 0.5 * dist_fork)
            else:
                # North of fork: stay on east grass path (x >= 12) leading to (17, 5)
                dist_house = abs(x - 17.0) + abs(y - 5.0)
                pen = max(0.0, 12.0 - x) * 2.0  # Heavy penalty for entering blocked left path
                return 70.0 + max(0.0, 20.0 - 0.5 * dist_house) - pen
        elif map_id in ((26, 10), (24, 10)):  # Inside Mr. Pokémon's House -> approach Mr. Pokémon at (3, 5)
            dist_mr = abs(x - 3.0) + abs(y - 5.0)
            return 90.0 + max(0.0, 5.0 - 0.5 * dist_mr)

    # Stage 2: Return Egg to Elm in New Bark Town
    elif not gave_egg:
        if map_id in ((26, 10), (24, 10)):  # Inside Mr. Pokémon's House -> exit doorway at (2.5, 8.0)
            dist_exit = abs(x - 2.5) + abs(y - 8.0)
            return 90.0 + max(0.0, 5.0 - 0.7 * dist_exit)
        elif map_id == (26, 1):  # Route 30 -> head south through east grass path to Cherrygrove at (10, 53)
            if y <= 40.0:
                dist_fork = abs(x - 15.0) + abs(y - 40.0)
                pen = max(0.0, 12.0 - x) * 2.0
                return 75.0 + max(0.0, 15.0 - 0.5 * dist_fork) - pen
            else:
                dist_south = abs(x - 10.0) + abs(y - 53.0)
                return 75.0 + max(0.0, 15.0 - 0.3 * dist_south)
        elif map_id in ((26, 3), (24, 7)):  # Cherrygrove -> head east to Route 29 gate at (39, 7)
            dist_gate = abs(x - 39.0) + abs(y - 7.0)
            pen_south = max(0.0, y - 8.0) * 0.8  # Penalty for wandering into south cul-de-sac/trees
            return 90.0 + max(0.0, 15.0 - 0.4 * dist_gate) - pen_south
        elif map_id == (24, 3):  # Route 29 -> head east to New Bark Town gate at (59, 9)
            dist_new_bark = abs(x - 59.0) + abs(y - 9.0)
            return 100.0 + max(0.0, 25.0 - 0.35 * dist_new_bark)
        elif map_id == (24, 4):  # New Bark Town -> Elm Lab entrance at (6, 4)
            dist_lab = abs(x - 6.0) + abs(y - 4.0)
            return 125.0 + max(0.0, 10.0 - 0.6 * dist_lab)
        elif map_id == (24, 5):  # Inside Elm's Lab -> approach Elm at (4, 2)
            dist_elm = abs(x - 4.0) + abs(y - 2.0)
            return 135.0 + max(0.0, 5.0 - 0.7 * dist_elm)

    # Stage 4: Post-Falkner -> Collect Togepi Pokémon Egg from Elm's Aide in Violet Poké Center
    elif beat_falkner and not has_aide_egg:
        if map_id == (10, 7):  # Inside Violet Gym -> exit doorway at (4.5, 16.0)
            dist_exit = abs(x - 4.5) + abs(y - 16.0)
            return 255.0 + max(0.0, 5.0 - 0.5 * dist_exit)
        elif map_id == (10, 5):  # Violet City -> head to Violet Poké Center at (31, 25)
            dist_pc = abs(x - 31.0) + abs(y - 25.0)
            return 255.0 + max(0.0, 15.0 - 0.4 * dist_pc)

    # Stage 3: Head to Violet City Gym & Falkner
    else:
        if map_id == (24, 5):  # Inside Elm's Lab -> exit doorway at (4.5, 12.0)
            dist_exit = abs(x - 4.5) + abs(y - 12.0)
            return 135.0 + max(0.0, 5.0 - 0.5 * dist_exit)
        elif map_id == (24, 4):  # In New Bark Town -> head west into Route 29
            return 135.0 + max(0.0, 10.0 - x * 0.5)
        elif map_id == (24, 3):  # Route 29 -> head west to Cherrygrove
            return _route29_potential(x, y, base=135.0)
        elif map_id in ((26, 3), (24, 7)):  # Cherrygrove -> head north to Route 30
            if low_hp:
                dist_pc = abs(x - 29.0) + abs(y - 3.0)
                return 175.0 + max(0.0, 6.0 - 0.3 * dist_pc)
            return 175.0 + max(0.0, (20.0 - y) * 0.5)
        elif map_id == (26, 1):  # Route 30 -> head north along unblocked western path into Route 31
            dist_x = abs(x - 8.5)
            pen_x = max(0.0, dist_x - 2.5) * 0.4
            return 190.0 + max(0.0, (54.0 - y) * 0.45) - pen_x
        elif map_id == (26, 2):  # Route 31 -> head west to Violet Gate at (4, 6..7)
            dist_gate = abs(x - 4.0) + abs(y - 6.5) * 0.4
            return 215.0 + max(0.0, 20.0 - 0.45 * dist_gate)
        elif map_id == (26, 11):  # Route 31 Violet Gate -> head west into Violet City (x -> 0)
            return 235.0 + max(0.0, (9.0 - x) * 0.8)
        elif map_id == (10, 5):  # Violet City -> Violet Gym (10, 7) at (18, 17)
            if low_hp:
                dist_pc = abs(x - 31.0) + abs(y - 25.0)
                return 245.0 + max(0.0, 8.0 - 0.4 * dist_pc)
            dist_gym = abs(x - 18.0) + abs(y - 17.0)
            return 245.0 + max(0.0, 22.0 - 0.5 * dist_gym)
        elif map_id == (10, 7):  # Violet Gym -> Falkner (x in 4..5, y -> 2)
            dist_falkner = abs(x - 4.5) + abs(y - 2.5)
            return 270.0 + max(0.0, 20.0 - 1.2 * dist_falkner)

    return 0.0


def progress_reward(
    before,
    after,
    moved: bool,
    seen_tiles: set,
    seen_maps: set,
    unvisited_neighbor: bool = False,
    caught_counts: dict[int, int] | None = None,
    seen_events: set[int] | None = None,
    seen_waypoints: set[tuple[int, int]] | None = None,
    info: StepInfo | None = None,
    shaping: float | None = None,
    legacy_potential: bool | None = None,
) -> tuple[float, bool, bool]:
    """Return reward and whether this step discovered a tile or a map."""
    reward = -0.01
    new_tile = False
    new_map = False

    # Potential-based shaping (Ng et al., 1999). ``shaping`` is the env's frontier-distance delta.
    # The hand-coded per-map potential runs when no shaping is given, or when asked for (hybrid).
    use_legacy = (shaping is None) if legacy_potential is None else legacy_potential
    if shaping is not None:
        reward += shaping
    if use_legacy and moved:
        phi_before = navigation_potential(before)
        phi_after = navigation_potential(after)
        delta_phi = phi_after - phi_before
        reward += delta_phi

    # Non-movement check: do not penalize wall-bump if in battle or dialogue
    if not moved:
        is_busy = getattr(before, "battle_mode", 0) != 0 or dialogue_open(before)
        if not is_busy:
            bump_count = getattr(info, "consecutive_bumps", 1) if info else 1
            reward -= 0.05 * min(4, max(1, bump_count))
    else:
        tile = (after.map_group, after.map_number, after.x, after.y)
        if tile not in seen_tiles:
            seen_tiles.add(tile)
            on_objective = (after.map_group, after.map_number) in story_objective_maps(after)
            reward += OBJECTIVE_TILE_BOOST if on_objective else 1.0
            new_tile = True
        # Chosen neighbor was unseen at the start of the step, including the discovery itself.
        if unvisited_neighbor:
            reward += UNVISITED_NEIGHBOR_BONUS

    map_id = (after.map_group, after.map_number)
    if map_id not in seen_maps:
        seen_maps.add(map_id)
        reward += 10.0
        new_map = True

    # Waypoints towards Gym & Quest Delivery
    if seen_waypoints is not None:
        if map_id in GYM_PATH_WAYPOINTS and map_id not in seen_waypoints:
            seen_waypoints.add(map_id)
            reward += GYM_PATH_WAYPOINTS[map_id]

        # Mystery Egg return delivery waypoint: after getting egg from Mr. Pokémon, reward returning to New Bark / Elm
        has_event = getattr(after, "has_event", None)
        if callable(has_event) and has_event("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON") and not has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM"):
            egg_key = (map_id[0], map_id[1] + 1000)
            if map_id in ((24, 4), (24, 5)) and egg_key not in seen_waypoints:
                seen_waypoints.add(egg_key)
                reward += 15.0

    # Badges
    if getattr(after, "badges", 0) > getattr(before, "badges", 0):
        reward += 30.0

    # Catching Pokémon (with diminishing returns for duplicate species)
    caught_any = False
    before_count = getattr(before, "party_count", 0)
    after_count = getattr(after, "party_count", 0)
    if after_count > before_count:
        caught_any = True
        if info is not None:
            info.caught_pokemon = True
        has_ev = getattr(after, "has_event", None)
        has_egg_now = callable(has_ev) and has_ev("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
        gave_egg_now = callable(has_ev) and has_ev("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
        if has_egg_now and not gave_egg_now:
            # After grabbing the egg, wild catching is an errand delay and yields no bonus
            reward += 0.0
        else:
            species_list = getattr(after, "party_species", ())
            for idx in range(before_count, min(after_count, len(species_list))):
                species_id = species_list[idx]
                if caught_counts is not None and species_id not in (0, 255):
                    caught_counts[species_id] = caught_counts.get(species_id, 0) + 1
                    reward += catch_reward_for_species(caught_counts[species_id])
                else:
                    reward += 10.0

    # Also check if party was full and caught straight to PC box
    before_dex = getattr(before, "pokedex_caught_count", 0)
    after_dex = getattr(after, "pokedex_caught_count", 0)
    if not caught_any and after_dex > before_dex:
        if info is not None:
            info.caught_pokemon = True
        has_ev = getattr(after, "has_event", None)
        has_egg_now = callable(has_ev) and has_ev("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
        gave_egg_now = callable(has_ev) and has_ev("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
        if has_egg_now and not gave_egg_now:
            reward += 0.0
        elif caught_counts is not None:
            caught_counts[0] = caught_counts.get(0, 0) + 1
            reward += catch_reward_for_species(caught_counts[0])
        else:
            reward += 10.0

    # Story Milestones & Event Flags
    if seen_events is not None and getattr(after, "event_flags", None):
        for i, b in enumerate(after.event_flags):
            if b == 0:
                continue
            for bit in range(8):
                if b & (1 << bit):
                    flag_id = i * 8 + bit
                    if flag_id not in seen_events:
                        seen_events.add(flag_id)
                        flag_name = REVERSE_EVENT_FLAGS.get(flag_id)
                        if flag_name:
                            reward += KEY_EVENT_REWARDS.get(flag_name, 10.0)
                            if info is not None:
                                info.story_progress = True
                        else:
                            reward += 2.0

    # Combat & Growth: Pokémon Level-Ups
    before_levels = sum(getattr(before, "party_levels", ()) or [0])
    after_levels = sum(getattr(after, "party_levels", ()) or [0])
    if after_levels > before_levels:
        reward += LEVEL_REWARD_SCALE * (level_reward_value(after_levels) - level_reward_value(before_levels))
        if info is not None:
            info.level_up = True

    # Combat: Battle Won (with diminishing returns on early wild battles)
    if getattr(before, "battle_mode", 0) != 0 and getattr(after, "battle_mode", 0) == 0:
        if not getattr(after, "fainted", False):
            lead_lvl = max(getattr(before, "party_levels", (0,)) or [0])
            has_ev = getattr(after, "has_event", None)
            has_egg_now = callable(has_ev) and has_ev("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
            gave_egg_now = callable(has_ev) and has_ev("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
            if getattr(before, "battle_mode", 0) == 2:
                # Trainer battles are key milestones towards the gym (Rival Cherrygrove = 15.0)
                reward += 5.0
            elif has_egg_now and not gave_egg_now:
                # Urgent egg delivery errand: wild fights are time sinks, not rewarded
                reward += 0.0
            elif lead_lvl < 7:
                # Early wild battle victory provides initial EXP
                reward += 3.0
            else:
                # Diminishing returns: blind wild grinding yields no incentive over progressing
                reward += 0.2

    # HP Dynamics: Damage Taken, Healing, and Pokémon Center Interaction
    is_pc = map_id in ((26, 5), (10, 10), (10, 13))
    hp_b = getattr(before, "party_hp", ())
    hp_a = getattr(after, "party_hp", ())
    if hp_b and hp_a:
        cur_b = sum(p[0] for p in hp_b)
        max_b = sum(p[1] for p in hp_b)
        cur_a = sum(p[0] for p in hp_a)
        max_a = sum(p[1] for p in hp_a)
        if max_b > 0 and cur_a < cur_b:
            # Damage penalty proportional to HP fraction lost
            dmg_frac = (cur_b - cur_a) / max_b
            reward -= dmg_frac * 2.0
        elif max_a > 0 and cur_a > cur_b:
            heal_frac = (cur_a - cur_b) / max_a
            if is_pc:
                # Paid once per Pokémon Center per episode, so taking damage to heal again earns nothing.
                heal_key = (map_id[0], map_id[1] + 2000)
                if seen_waypoints is None or heal_key not in seen_waypoints:
                    if seen_waypoints is not None:
                        seen_waypoints.add(heal_key)
                    reward += 15.0 + heal_frac * 10.0
                    if info is not None:
                        info.healed_party = True
            else:
                reward += heal_frac * 3.0
        elif is_pc and cur_b < max_b:
            # Low HP in Pokémon Center: reward choosing to interact with Nurse Joy
            if dialogue_open(after) and not dialogue_open(before):
                reward += 2.0

        # Critical danger penalty when active Pokémon is below 25% HP
        lead_cur, lead_max = hp_a[0]
        if lead_max > 0 and (lead_cur / lead_max) <= 0.25:
            reward -= 0.05

    # Combat: Fainted (White-out penalty: catastrophic reset of progress)
    if getattr(after, "fainted", False) and not getattr(before, "fainted", False):
        reward -= 25.0

    return reward, new_tile, new_map


def observe(state, visited: set[tuple[int, int, int, int]], nav: LocalNav | None = None) -> np.ndarray:
    """17 base features, plus the LocalNav block when ``nav`` is given."""
    tiles = [float(state.tiles.get(direction, 0)) / 255.0 for direction in DIRS]
    pos = [state.x / 64.0, state.y / 64.0]
    flags = [
        state.badges / 8.0,
        1.0 if state.battle_mode else 0.0,
        1.0 if dialogue_open(state) else 0.0,
    ]
    neighbors = []
    for direction, (dx, dy) in zip(DIRS, _NEIGHBOR_DELTAS):
        key = (state.map_group, state.map_number, state.x + dx, state.y + dy)
        neighbors.append(1.0 if key in visited else 0.0)
    extra = [
        float(getattr(state, "party_count", 0)) / 6.0,
        float(max(getattr(state, "party_levels", (0,)) or [0])) / 20.0,
        float(getattr(state, "pokedex_caught_count", 0)) / 10.0,
        story_stage(state),
    ]
    base = np.asarray(tiles + pos + flags + neighbors + extra, dtype=np.float64)
    if nav is None:
        return base
    nav.update(state)
    in_menu = bool(state.battle_mode) or dialogue_open(state)
    has_event = getattr(state, "has_event", None)
    events = [1.0 if callable(has_event) and has_event(name) else 0.0 for name in OBS_EVENT_FLAGS]
    return np.concatenate([base, nav.features(state, in_menu=in_menu), np.asarray(events)])


@dataclass
class StepInfo:
    discovered_tile: bool = False
    discovered_map: bool = False
    frontier: bool = False
    caught_pokemon: bool = False
    story_progress: bool = False
    level_up: bool = False
    healed_party: bool = False
    consecutive_bumps: int = 0


@dataclass
class CrystalEnv:
    machine: object
    reader: object
    stepper: object
    macros: object
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    frontier_prob: float = 0.3
    max_steps: int = 64
    ignore_maps: frozenset = frozenset()  # (group, number) maps already completed; entering one ends the episode
    # "hybrid": hand-coded stage direction plus learned-map frontier distance (default);
    # "frontier": frontier distance only; "legacy": hand-coded per-map potential only.
    shaping: str = "hybrid"

    def __post_init__(self):
        self.seen_tiles: set = set()
        self.seen_maps: set = set()
        self.visited: set[tuple[int, int, int, int]] = set()
        self.frontier: list[bytes] = []
        self.frontier_maps: set[tuple[int, int]] = set()
        self.caught_counts: dict[int, int] = {}
        self.seen_events: set[int] = set()
        self.seen_waypoints: set[tuple[int, int]] = set()
        self.start: bytes | None = None
        self.steps = 0
        self.consecutive_bumps = 0
        self.nav = LocalNav()
        self.seen_interactions: set[tuple] = set()
        self.seen_warps: set[tuple] = set()
        self._phi = 0.0

    def reset(self, start: bytes | None = None) -> np.ndarray:
        if start is not None:
            self.start = start
        blob = self.start
        if self.frontier and self.rng.random() < self.frontier_prob:
            blob = self.frontier[int(self.rng.integers(0, len(self.frontier)))]
        if blob is not None:
            self.machine.load_state(blob)
        self.steps = 0
        self.consecutive_bumps = 0
        self.nav.reset()
        self.seen_interactions = set()
        self.seen_warps = set()
        # Discovery bonuses start over each episode. The frontier list does not.
        self.seen_tiles = set()
        self.seen_maps = set()
        self.visited = set()
        self.caught_counts = {}
        self.seen_events = set()
        self.seen_waypoints = set()
        state = self.reader.parse(self.machine.frame)
        self.seen_maps.add((state.map_group, state.map_number))
        tile = (state.map_group, state.map_number, state.x, state.y)
        self.seen_tiles.add(tile)
        self.visited.add(tile)
        map_id = (state.map_group, state.map_number)
        if map_id in GYM_PATH_WAYPOINTS:
            self.seen_waypoints.add(map_id)
        # Seed initial caught counts from party
        for species_id in getattr(state, "party_species", ()):
            if species_id and species_id != 255:
                self.caught_counts[species_id] = self.caught_counts.get(species_id, 0) + 1
        # Seed initial event flags
        if getattr(state, "event_flags", None):
            for i, b in enumerate(state.event_flags):
                for bit in range(8):
                    if b & (1 << bit):
                        self.seen_events.add(i * 8 + bit)
        self.nav.update(state)
        self._phi = self.nav.potential(state)
        return observe(state, self.visited, self.nav)

    def _interaction_reward(self, before, after, action: int, busy: bool) -> float:
        """One-time payments, as in pokemonred_puffer's NPC/sign/warp terms."""
        bonus = 0.0
        if action == 4 and not busy and dialogue_open(after):
            # In a Pokémon Center, do not award interaction bonus if party is already at full health
            if (before.map_group, before.map_number) in ((26, 5), (10, 10), (10, 13)):
                hp_b = getattr(before, "party_hp", ()) or ()
                cur_b = sum(p[0] for p in hp_b) if hp_b else 0
                max_b = sum(p[1] for p in hp_b) if hp_b else 0
                if cur_b >= max_b:
                    return bonus
            key = (before.map_group, before.map_number, before.x, before.y, before.facing)
            if key not in self.seen_interactions:
                self.seen_interactions.add(key)
                bonus += NEW_INTERACTION_REWARD
        before_map = (before.map_group, before.map_number)
        after_map = (after.map_group, after.map_number)
        if before_map != after_map:
            edge = (before_map, after_map)
            if edge not in self.seen_warps:
                self.seen_warps.add(edge)
                bonus += NEW_WARP_REWARD
        return bonus

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        before = self.reader.parse(self.machine.frame)
        info = StepInfo()
        if before.battle_mode:
            from src.battle import tactical_battle_step
            tactical_battle_step(before, self.macros)
            moved = False
        elif dialogue_open(before):
            self.macros.mash_text(max_frames=80)
            moved = False
        elif action == 4:
            self.machine.press("a", frames=8)
            self.machine.tick(16, render=False)
            moved = False
        else:
            moved = self.stepper.step(ACTIONS[action]).moved
        if not moved and not before.battle_mode and not dialogue_open(before):
            self.consecutive_bumps += 1
        else:
            self.consecutive_bumps = 0
        info.consecutive_bumps = self.consecutive_bumps
        after = self.reader.parse(self.machine.frame)
        if not before.battle_mode and not dialogue_open(before):
            self.nav.note_action(before, action, moved)
        # Read the neighbor before progress_reward records the tile this step discovers.
        unvisited_neighbor = False
        if 0 <= action < 4:
            dx, dy = _NEIGHBOR_DELTAS[action]
            neighbor = (before.map_group, before.map_number, before.x + dx, before.y + dy)
            unvisited_neighbor = neighbor not in self.seen_tiles
        busy = bool(before.battle_mode) or dialogue_open(before)
        shaping = None
        if self.shaping in ("frontier", "hybrid"):
            self.nav.update(after)
            phi_after = self.nav.potential(after)
            same_map = (before.map_group, before.map_number) == (after.map_group, after.map_number)
            # No shaping across a warp (the map bonus covers it) or inside battle and dialogue.
            shaping = 0.0
            if same_map and not busy:
                shaping = FRONTIER_SHAPING_WEIGHT * (SHAPING_GAMMA * phi_after - self._phi)
            self._phi = phi_after
        reward, info.discovered_tile, info.discovered_map = progress_reward(
            before,
            after,
            moved,
            self.seen_tiles,
            self.seen_maps,
            unvisited_neighbor=unvisited_neighbor,
            caught_counts=self.caught_counts,
            seen_events=self.seen_events,
            seen_waypoints=self.seen_waypoints,
            info=info,
            shaping=shaping,
            legacy_potential=self.shaping in ("legacy", "hybrid"),
        )
        reward += self._interaction_reward(before, after, action, busy)
        tile = (after.map_group, after.map_number, after.x, after.y)
        self.visited.add(tile)
        map_id = (after.map_group, after.map_number)
        if (info.discovered_map or after.badges > before.badges) and map_id not in self.frontier_maps:
            self.frontier.append(self.machine.save_state())
            self.frontier_maps.add(map_id)
            info.frontier = True
        self.steps += 1
        done = self.steps >= self.max_steps
        if map_id in self.ignore_maps and (before.map_group, before.map_number) not in self.ignore_maps:
            reward, done = -BEHIND_MAP_PENALTY, True
        if not getattr(self.machine, "headless", True):
            self.machine.tick(1, render=True)
        return observe(after, self.visited, self.nav), reward, done, info
