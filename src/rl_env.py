"""Overworld environment. Rewards come from new tiles, maps, story milestones, gyms, and caught Pokémon."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ram_parser import dialogue_open
from src.step import DIRS
from src.story_flags import EVENT_FLAGS, REVERSE_EVENT_FLAGS

ACTIONS = DIRS + ("a",)
N_ACTIONS = len(ACTIONS)
# 4 neighbor tiles, xy, badges, battle, dialogue, 4 visited-neighbor flags,
# party_count, max_level, caught_count, story_stage
OBS_DIM = 17
_NEIGHBOR_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
UNVISITED_NEIGHBOR_BONUS = 0.2

# Key Transit Maps along the path to the First Gym (Falkner in Violet City)
GYM_PATH_WAYPOINTS: dict[tuple[int, int], float] = {
    (24, 3): 3.0,   # Route 29
    (24, 7): 4.0,   # Route 29 Gate / West
    (24, 6): 4.0,   # Route 29 Gate / Route 46
    (26, 3): 5.0,   # Cherrygrove City
    (26, 1): 5.0,   # Route 30
    (26, 10): 8.0,  # Mr. Pokémon's House (Egg quest)
    (26, 2): 8.0,   # Route 31
    (26, 11): 5.0,  # Route 31 Violet Gate
    (10, 5): 12.0,  # Violet City
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
def navigation_potential(state) -> float:
    """Potential-based reward shaping Phi(s) guiding the agent towards Falkner's Gym.

    Creates an invariant directional gradient (Ng, Harada & Russell, 1999) that rewards
    moving forward along the transit corridor and penalizes retreating backwards.
    """
    map_id = getattr(state, "map_id", (getattr(state, "map_group", 0), getattr(state, "map_number", 0)))
    x = float(getattr(state, "x", 0))
    y = float(getattr(state, "y", 0))

    has_event = getattr(state, "has_event", None)
    has_egg = callable(has_event) and has_event("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
    gave_egg = callable(has_event) and has_event("EVENT_GAVE_MYSTERY_EGG_TO_ELM")

    # Stage 1: Heading to Mr. Pokémon / Cherrygrove (prior to getting egg)
    if not has_egg:
        if map_id == (24, 4):  # New Bark Town -> exit west into Route 29 (x -> 0)
            return max(0.0, 10.0 - x * 0.5)
        elif map_id == (24, 3):  # Route 29 -> exit west towards Cherrygrove (x -> 0)
            # Route 29 has an enclosed dead-end cul-de-sac at x <= 20, y >= 13.
            # The true highway branches North at x=24 up to y=7, then runs west to x=8, then drops to (0, 11).
            if x <= 20 and y >= 13:
                return 10.0 + (x * 0.5)  # Penalize deeper intrusion; reward moving east to escape
            elif x <= 28 and y > 7:
                return 15.0 + (60.0 - x) * 0.3 + (15.0 - y) * 0.8  # Pull north towards y=7 highway
            else:
                return 20.0 + max(0.0, (60.0 - x) * 0.5)
        elif map_id in ((26, 3), (24, 7)):  # Cherrygrove City -> exit north towards Route 30 (y -> 0)
            return 45.0 + max(0.0, (20.0 - y) * 0.5)
        elif map_id in ((26, 2), (24, 6)):  # Route 30 -> head north to Mr. Pokémon (y -> 0)
            return 60.0 + max(0.0, (40.0 - y) * 0.5)
        elif map_id in ((26, 10), (24, 10)):  # Mr. Pokémon's House
            return 85.0

    # Stage 2: Return Egg to Elm in New Bark Town
    elif not gave_egg:
        if map_id in ((26, 2), (24, 6)):  # Route 30 -> head south to Cherrygrove (y -> 40)
            return 70.0 + y * 0.3
        elif map_id in ((26, 3), (24, 7)):  # Cherrygrove -> head east to Route 29 (x -> 20)
            return 85.0 + x * 0.3
        elif map_id == (24, 3):  # Route 29 -> head east to New Bark (x -> 60)
            return 95.0 + x * 0.3
        elif map_id == (24, 4):  # New Bark Town -> Elm Lab
            return 120.0
        elif map_id == (24, 5):  # Inside Elm's Lab
            return 130.0

    # Stage 3: Head to Violet City Gym & Falkner
    else:
        if map_id in ((24, 4), (24, 5)):  # In New Bark -> head west
            return 130.0 + max(0.0, 10.0 - x * 0.5)
        elif map_id == (24, 3):  # Route 29 -> head west to Cherrygrove
            if x <= 20 and y >= 13:
                return 135.0 + (x * 0.5)
            elif x <= 28 and y > 7:
                return 140.0 + (60.0 - x) * 0.3 + (15.0 - y) * 0.8
            else:
                return 145.0 + max(0.0, (60.0 - x) * 0.5)
        elif map_id in ((26, 3), (24, 7)):  # Cherrygrove -> head north to Route 30
            return 175.0 + max(0.0, (20.0 - y) * 0.5)
        elif map_id in ((26, 2), (24, 6)):  # Route 30 -> head north-west to Route 31
            return 190.0 + max(0.0, (40.0 - y) * 0.5)
        elif map_id in ((26, 1), (24, 2)):  # Route 31 -> head west to Violet City (x -> 0)
            return 215.0 + max(0.0, (40.0 - x) * 0.5)
        elif map_id == (10, 5):  # Violet City -> Violet Gym (10, 7)
            return 240.0
        elif map_id == (10, 7):  # Violet Gym -> Falkner
            return 260.0

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
) -> tuple[float, bool, bool]:
    """Return reward and whether this step discovered a tile or a map."""
    reward = -0.01
    new_tile = False
    new_map = False

    # Goal-Directed Navigation Potential Shaping (Ng et al., 1999)
    if moved:
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
            reward += 1.0
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
        if caught_counts is not None:
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
        level_diff = after_levels - before_levels
        reward += 5.0 * level_diff
        if info is not None:
            info.level_up = True

    # Combat: Battle Won (with diminishing returns on early wild battles)
    if getattr(before, "battle_mode", 0) != 0 and getattr(after, "battle_mode", 0) == 0:
        if not getattr(after, "fainted", False):
            lead_lvl = max(getattr(before, "party_levels", (0,)) or [0])
            if getattr(before, "battle_mode", 0) == 2:
                # Trainer battles are key milestones towards the gym
                reward += 5.0
            elif lead_lvl < 7:
                # Early wild battle victory provides initial EXP
                reward += 3.0
            else:
                # Diminishing returns: blind wild grinding yields no incentive over progressing
                reward += 0.2

    # HP Dynamics: Damage Taken, Healing, and Critical Low-HP Danger
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
            # Healing reward (Pokémon Center, Mom, potions)
            heal_frac = (cur_a - cur_b) / max_a
            reward += heal_frac * 3.0

        # Critical danger penalty when active Pokémon is below 25% HP
        lead_cur, lead_max = hp_a[0]
        if lead_max > 0 and (lead_cur / lead_max) <= 0.25:
            reward -= 0.05

    # Combat: Fainted (White-out penalty: catastrophic reset of progress)
    if getattr(after, "fainted", False) and not getattr(before, "fainted", False):
        reward -= 25.0

    return reward, new_tile, new_map


def observe(state, visited: set[tuple[int, int, int, int]]) -> np.ndarray:
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
    return np.asarray(tiles + pos + flags + neighbors + extra, dtype=np.float64)


@dataclass
class StepInfo:
    discovered_tile: bool = False
    discovered_map: bool = False
    frontier: bool = False
    caught_pokemon: bool = False
    story_progress: bool = False
    level_up: bool = False
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
        return observe(state, self.visited)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        before = self.reader.parse(self.machine.frame)
        info = StepInfo()
        if before.battle_mode:
            self.macros.choose_move()
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
        # Read the neighbor before progress_reward records the tile this step discovers.
        unvisited_neighbor = False
        if 0 <= action < 4:
            dx, dy = _NEIGHBOR_DELTAS[action]
            neighbor = (before.map_group, before.map_number, before.x + dx, before.y + dy)
            unvisited_neighbor = neighbor not in self.seen_tiles
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
        )
        tile = (after.map_group, after.map_number, after.x, after.y)
        self.visited.add(tile)
        map_id = (after.map_group, after.map_number)
        if (info.discovered_map or after.badges > before.badges) and map_id not in self.frontier_maps:
            self.frontier.append(self.machine.save_state())
            self.frontier_maps.add(map_id)
            info.frontier = True
        self.steps += 1
        done = self.steps >= self.max_steps
        if not self.machine.headless:
            self.machine.tick(1, render=True)
        return observe(after, self.visited), reward, done, info
