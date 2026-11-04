"""Environment helpers: observation, reward and the legacy single-process ``CrystalEnv``.

Design (see research/analysis/design_hardcoded_vs_learned.md):
- The policy is NOT told where to go. The observation holds state (position, map id, party, battle, bag, event flags,
  which story milestones are done) but no direction to a goal and no distance to it.
- Rewards are generic: first-arrival milestones on the minimal story path (``src.story_path``), the badge, small
  novelty bonuses (new tiles / maps / warps / interactions), survival (HP, white-out) and battle terms (damage
  dealt, KO, win, level-ups, catches, Pokecenter heals).
- Battles are played by the policy with raw button presses; only overworld dialogue is auto-advanced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.battle_obs import BATTLE_FEATURE_DIM, battle_features
from src.local_nav import NAV_DIM, LocalNav
from src.ram_parser import dialogue_open
from src.step import DIRS
from src.story_flags import REVERSE_EVENT_FLAGS
from src import story_path as _story
from src.story_path import (
    BADGE_REWARD,
    KEY_EVENT_REWARDS,
    MAP_MILESTONES,
    STORY_FEATURE_DIM,
    story_features,
    story_fraction,
)

try:  # optional goal potential (OFF by default); without the ROM the term is simply 0
    from src import route_graph
except ImportError:  # pragma: no cover - exercised through monkeypatching in tests
    route_graph = None

# Six buttons: the four directions, A and B. B is needed to back out of battle menus and the move list.
# START and SELECT are deliberately absent: the START menu (items outside battle, saving) is never needed to
# reach the badge, and the Pokecenter / in-battle item menus are reached with A and the directions.
ACTIONS = DIRS + ("a", "b")
N_ACTIONS = len(ACTIONS)
BUTTON_ACTIONS = ("a", "b")
ACTION_A = ACTIONS.index("a")
ACTION_B = ACTIONS.index("b")
# 4 neighbor tiles, xy, badges, battle, dialogue, 4 visited-neighbor flags,
# party_count, max_level, caught_count, story fraction
BASE_OBS_DIM = 17
# Then the LocalNav block: blocked mask (indices MASK_START..+4), last action, frontier, 5x5 grid.
MASK_START = BASE_OBS_DIM
# Plain state flags (not goals): fixed order, the checkpoint width depends on it.
OBS_EVENT_FLAGS = (
    "EVENT_GOT_A_POKEMON_FROM_ELM",
    "EVENT_LEARNED_TO_CATCH_POKEMON",
    "EVENT_BEAT_FALKNER",
    "EVENT_GOT_TM31_MUD_SLAP",
)
OBS_DIM = BASE_OBS_DIM + NAV_DIM + len(OBS_EVENT_FLAGS) + STORY_FEATURE_DIM + BATTLE_FEATURE_DIM
# Observation of the deep (screen + RAM vector) trainer: 17 base + story milestones + battle features.
# ``DEEP_OBS_DIM`` is the base story profile (77); ``deep_obs_dim()`` follows the active profile (80 with ``egg_errand_route``).
DEEP_OBS_DIM = BASE_OBS_DIM + STORY_FEATURE_DIM + BATTLE_FEATURE_DIM


def deep_obs_dim(egg: bool | None = None) -> int:
    """Width of the deep observation for the story profile of this process (or of the egg / base profile when given)."""
    p = _story.profile() if egg is None else _story.profile_for(egg)
    return BASE_OBS_DIM + p.feature_dim + BATTLE_FEATURE_DIM

# Bumped whenever the observation or action layout changes; stored in checkpoints (see SpatialVisionPolicy.load).
OBS_SCHEMA = "v2-no-oracle-learned-battles"
_NEIGHBOR_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
UNVISITED_NEIGHBOR_BONUS = 0.02
# Frontier-distance shaping (potential-based: gamma * phi(s') - phi(s)) over tiles the agent has itself seen. It
# knows nothing about the story; it only pays for reaching the edge of what the agent has explored.
FRONTIER_SHAPING_WEIGHT = 4.0
SHAPING_GAMMA = 0.99
# Puffer-style one-time exploration payments for things that are not just new coordinates.
NEW_INTERACTION_REWARD = 0.5  # first A press that opens a script from a given tile and facing
NEW_WARP_REWARD = 1.0         # first time a map-to-map transition is taken
NEW_TILE_REWARD = 0.1         # first visit of a tile in an episode
NEW_MAP_REWARD = 0.5          # first visit of a map that is not a story milestone (houses, shops, ...)
# Stepping back into a map the run has already finished ends the episode with this penalty (legacy CrystalEnv only).
BEHIND_MAP_PENALTY = 2.0
# OPTIONAL tile-level goal potential (TrainConfig.goal_potential, default OFF): weight * (Phi(s') - Phi(s)) with
# Phi = -(BFS steps to Falkner). Telescopes to weight * total path progress. Off in every normal run: it is an oracle.
ROUTE_PROGRESS_WEIGHT = 0.08
# Per-step cost while the lead Pokemon is at <= 25% HP (survival pressure; there is no heal before Cherrygrove).
LOW_HP_STEP_PENALTY = 0.01

# Per-episode caps on the shaping sources that are not the story milestones. The story path pays ~130 in total
# (src.story_path.TOTAL_STORY_REWARD); each source below is held to a fraction of that so none can outbid it.
# They apply only when ``progress_reward`` gets a ``budget`` dict (fresh per episode); the first payments are
# full-size, later ones zero.
BATTLE_BUDGET = 20.0    # damage dealt, KOs and battle wins
LEVEL_BUDGET = 15.0     # level-ups
CATCH_BUDGET = 6.0      # new Pokemon
HEAL_BUDGET = 15.0      # Pokecenter heals, Nurse Joy, healing items
EXPLORE_BUDGET = 30.0   # new tiles / maps / warps / interactions / curiosity-like terms
BUDGET_CAPS: dict[str, float] = {
    "battle": BATTLE_BUDGET,
    "level": LEVEL_BUDGET,
    "catch": CATCH_BUDGET,
    "heal": HEAL_BUDGET,
    "explore": EXPLORE_BUDGET,
}
DAMAGE_DEALT_WEIGHT = 1.5  # per fraction of the enemy's max HP removed (pays while the same enemy stays out)
KO_REWARD = 1.0            # the enemy's HP reaching 0
WILD_WIN_FULL = 3.0        # reward for each of the first WILD_WIN_FULL_COUNT genuine wild wins in an episode
WILD_WIN_FULL_COUNT = 3
WILD_WIN_LATE = 0.2
TRAINER_WIN = 5.0
PC_HEAL_BASE = 15.0        # Pokecenter heal pays (PC_HEAL_BASE + PC_HEAL_FRAC) x the fraction of max HP healed
PC_HEAL_FRAC = 10.0
ALLY_FAINT_PENALTY = 1.0   # one of the player's Pokemon fainted but the party is not wiped
WHITEOUT_PENALTY = 15.0    # the whole party fainted
KEY_EVENT_DEFAULT_NAMED = 0.0  # named flags that are not story milestones pay nothing
UNNAMED_EVENT_REWARD = 0.05    # any other flag that flips: tiny, explore-budgeted, so incidental flags cannot swamp
POKECENTER_MAPS = ((26, 5), (10, 10), (10, 13))


def budget_pay(budget: dict | None, key: str, value: float) -> float:
    """Amount of ``value`` that fits under BUDGET_CAPS[key] given what ``budget`` already paid; records it.

    No budget (or a non-positive value, or an unknown key) returns ``value`` unchanged. Callers that add their
    own exploration terms (curiosity, momentum) should route them through ``budget_pay(budget, "explore", v)``.
    """
    if budget is None or value <= 0:
        return value
    # Optional per-episode overrides (long_v4): "cap:<key>" replaces the cap, "scale:<key>" multiplies the payment
    # before the cap is applied (a decaying intrinsic weight).  Absent keys leave the behaviour unchanged.
    cap = budget.get("cap:" + key, BUDGET_CAPS.get(key))
    if cap is None:
        return value
    value = value * budget.get("scale:" + key, 1.0)
    paid = budget.get(key, 0.0)
    pay = max(0.0, min(value, cap - paid))
    budget[key] = paid + pay
    return pay

# Level reward shaped like the pokemonred write-up: full credit for early levels, then a quarter.
# Continuous at LEVEL_CAP_SUM (the write-up's version jumps by 15 there, which would dwarf a gym badge).
LEVEL_REWARD_SCALE = 5.0
LEVEL_CAP_SUM = 15


def level_reward_value(level_sum: float) -> float:
    if level_sum < LEVEL_CAP_SUM:
        return float(level_sum)
    return LEVEL_CAP_SUM + (level_sum - LEVEL_CAP_SUM) / 4.0

def catch_reward_for_species(count: int, base: float = 3.0) -> float:
    """Calculate reward for catching a Pokémon species with diminishing returns for duplicates.

    1st catch of species: 3.0 (small novelty bonus; catching is not on the story path)
    2nd catch of species: 1.5 (duplicate, 50%)
    4th+ catch: max(0.5, 3.0 * 0.5**(count - 1))
    """
    if count <= 1:
        return base
    return max(0.5, base * (0.5 ** (count - 1)))


def story_stage(state, seen_maps=None) -> float:
    """Fraction of the story milestones done, 0.0 (start) to 1.0 (badge earned). Progress state, not a direction."""
    return story_fraction(state, seen_maps)


def is_party_low_hp(state) -> bool:
    """Return True if any party member is not at full health."""
    hp_list = getattr(state, "party_hp", ()) or ()
    for cur, max_hp in hp_list:
        if max_hp > 0 and cur < max_hp:
            return True
    return False


def route_potential(state) -> float:
    """Optional goal potential Phi = -(BFS steps to Falkner); 0 without the route graph. Used only with goal_potential."""
    if route_graph is None:
        return 0.0
    try:
        return float(route_graph.route_potential(state))
    except Exception:
        return 0.0


def _route_covers(*states) -> bool:
    if route_graph is None:
        return False
    try:
        return all(route_graph.covers_map((s.map_group, s.map_number)) for s in states)
    except Exception:
        return False


def initial_seen_events(state) -> set[int]:
    """Flag ids already set in ``state.event_flags``.

    The worker must seed ``seen_events`` with this after EVERY reset (``seen_events.clear()`` then
    ``seen_events.update(initial_seen_events(state))``). Otherwise every flag that is already set
    pays its milestone again on the first step, about +300 per episode.
    """
    seen: set[int] = set()
    for i, b in enumerate(getattr(state, "event_flags", None) or ()):
        if not b:
            continue
        for bit in range(8):
            if b & (1 << bit):
                seen.add(i * 8 + bit)
    return seen


def behind_map_penalty_applies(before, after, seen_maps_done, state=None) -> bool:
    """Whether stepping from ``before`` into ``after`` is a forbidden step back into a finished map.

    DO NOT wire this into training (episode end plus a -2 penalty): ending an episode at -2 beats about -10
    of step costs, so it is an escape hatch. Only the legacy ``CrystalEnv`` (with ``ignore_maps`` set) uses it.

    ``seen_maps_done`` is the set of finished (group, number) maps (``CrystalEnv.ignore_maps``).
    ``before``/``after`` are parsed states or (group, number) tuples.
    """
    def mid(x):
        return x if isinstance(x, tuple) else (x.map_group, x.map_number)

    b, a = mid(before), mid(after)
    return a in seen_maps_done and b not in seen_maps_done


def _party_level_gain(before, after) -> int:
    """Level-ups only: positive change per party slot present in both states (a catch adds a slot, not levels)."""
    lb = tuple(getattr(before, "party_levels", ()) or ())
    la = tuple(getattr(after, "party_levels", ()) or ())
    if len(lb) == len(la):
        return max(0, sum(la) - sum(lb))
    n = min(len(lb), len(la))
    return sum(max(0, la[i] - lb[i]) for i in range(n))


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
    components: dict[str, float] | None = None,
    budget: dict | None = None,
    goal_potential: bool = False,
    pay_heal: bool = True,
    pay_bump: bool = True,
) -> tuple[float, bool, bool]:
    """Return reward and whether this step discovered a tile or a map.

    ``components``, when given, receives each term under a stable key (step, bump, route, tile, neighbor, map,
    milestone, warp, interaction, event, badge, catch, level, battle, damage, heal, whiteout, behind_map, other);
    the values sum exactly to the returned reward.

    Generic terms only (no scripts, no direction hints):
    - ``milestone``: first arrival (per episode, tracked by ``seen_maps``) on a map of the minimal story path
      (``src.story_path.MILESTONES``), plus ``badge`` for the Zephyr Badge and ``event`` for EVENT_BEAT_FALKNER;
    - ``tile`` / ``neighbor`` / ``map``: small novelty bonuses, ``event``: unnamed flags pay 0.05 (explore budget);
    - battle: ``battle`` = damage dealt, KO and wins; ``damage`` = HP lost, ally faint, low-HP pressure;
      ``whiteout``; ``level``; ``catch`` (small); ``heal`` at a Pokecenter / with items;
    - ``route``: OPTIONAL tile potential toward Falkner, only when ``goal_potential`` is True (default OFF).

    ``shaping`` is the env's frontier-distance delta (potential over the agent's own explored tiles), logged as
    ``other``. ``budget`` is a per-episode counter dict (fresh at every reset): with it, battle, level-up, catch,
    heal and exploration payments stop at the BUDGET_CAPS totals; without it behavior is uncapped. A non-moving
    step whose ``info.consecutive_bumps`` is 0 is not a bump (an A / B press), so it pays no bump cost.
    ``pay_heal=False`` skips every ``heal`` payment here (the caller pays ``src.heal_reward`` instead, long_v4).
    ``pay_bump=False`` drops the wall-bump cost (long_v5, with the dense path-progress reward on).
    """
    new_tile = False
    new_map = False
    terms: dict[str, float] = {}

    def add(key: str, value: float, budget_key: str | None = None) -> None:
        if budget_key is not None:
            value = budget_pay(budget, budget_key, value)
        terms[key] = terms.get(key, 0.0) + value

    add("step", -0.01)
    faint_ctx = bool(getattr(after, "fainted", False) or getattr(before, "fainted", False))
    in_battle_b = getattr(before, "battle_mode", 0) != 0
    in_battle_a = getattr(after, "battle_mode", 0) != 0

    if shaping is not None:
        add("other", shaping)
    if goal_potential and _route_covers(before, after):
        route_term = ROUTE_PROGRESS_WEIGHT * (route_potential(after) - route_potential(before))
        if faint_ctx:
            route_term = min(0.0, route_term)  # the white-out teleport must not pay for "progress"
        add("route", route_term)

    # Non-movement check: do not penalize wall-bump if in battle or dialogue
    if not moved:
        is_busy = in_battle_b or dialogue_open(before)
        bump_count = getattr(info, "consecutive_bumps", 1) if info else 1
        if pay_bump and not is_busy and not (info is not None and bump_count == 0):
            add("bump", -0.05 * min(4, max(1, bump_count)))
    else:
        tile = (after.map_group, after.map_number, after.x, after.y)
        if tile not in seen_tiles:
            seen_tiles.add(tile)
            add("tile", NEW_TILE_REWARD, "explore")
            new_tile = True
        # Chosen neighbor was unseen at the start of the step, including the discovery itself.
        if unvisited_neighbor:
            add("neighbor", UNVISITED_NEIGHBOR_BONUS, "explore")

    map_id = (after.map_group, after.map_number)
    if map_id not in seen_maps:
        seen_maps.add(map_id)
        milestone = _story.profile().map_milestones.get(map_id)
        if milestone is not None:
            add("milestone", milestone.reward)  # on the story path: not part of any shaping budget
            if info is not None:
                info.story_progress = True
        else:
            add("map", NEW_MAP_REWARD, "explore")
        new_map = True

    # Badges
    if getattr(after, "badges", 0) > getattr(before, "badges", 0):
        add("badge", BADGE_REWARD)

    # Catching Pokémon (small, with diminishing returns for duplicate species)
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
                add("catch", catch_reward_for_species(caught_counts[species_id]), "catch")
            else:
                add("catch", catch_reward_for_species(1), "catch")

    # Also check if party was full and caught straight to PC box
    before_dex = getattr(before, "pokedex_caught_count", 0)
    after_dex = getattr(after, "pokedex_caught_count", 0)
    if not caught_any and after_dex > before_dex:
        caught_any = True
        if info is not None:
            info.caught_pokemon = True
        if caught_counts is not None:
            caught_counts[0] = caught_counts.get(0, 0) + 1
            add("catch", catch_reward_for_species(caught_counts[0]), "catch")
        else:
            add("catch", catch_reward_for_species(1), "catch")

    # Event flags: only the profile's key events pay (EVENT_BEAT_FALKNER; with the egg errand also the egg received / given
    # to Elm); any other flip pays a token amount or nothing.
    key_events = _story.profile().key_event_rewards
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
                        if flag_name in key_events:
                            add("event", key_events[flag_name])
                            if info is not None:
                                info.story_progress = True
                        elif flag_name:
                            add("event", KEY_EVENT_DEFAULT_NAMED)
                        else:
                            add("event", UNNAMED_EVENT_REWARD, "explore")

    # Growth: Pokémon level-ups (a caught Pokémon's own level is not a level-up)
    level_gain = _party_level_gain(before, after)
    if level_gain > 0:
        before_levels = sum(getattr(before, "party_levels", ()) or [0])
        add(
            "level",
            LEVEL_REWARD_SCALE * (level_reward_value(before_levels + level_gain) - level_reward_value(before_levels)),
            "level",
        )
        if info is not None:
            info.level_up = True

    # Battle, generic: damage dealt to the SAME enemy while it stays out, its KO, and the win when the fight ends.
    # (Whatever buttons the policy pressed to get there; nothing is scripted.)
    if in_battle_b and in_battle_a:
        e_max = getattr(after, "enemy_max_hp", 0)
        same_enemy = (
            e_max > 0
            and e_max == getattr(before, "enemy_max_hp", 0)
            and getattr(after, "enemy_mon_species", 0) == getattr(before, "enemy_mon_species", 0)
        )
        e_before, e_after = getattr(before, "enemy_hp", 0), getattr(after, "enemy_hp", 0)
        if same_enemy and e_after < e_before:
            add("battle", DAMAGE_DEALT_WEIGHT * (e_before - e_after) / e_max, "battle")
            if e_after == 0:
                add("battle", KO_REWARD, "battle")
    if in_battle_b and not in_battle_a:
        if not getattr(after, "fainted", False) and not caught_any:
            if getattr(before, "battle_mode", 0) == 2:
                add("battle", TRAINER_WIN, "battle")
            else:
                # A genuine win leaves the wild Pokémon fainted (its HP stays in RAM after the battle);
                # a flee leaves it alive. Unknown enemy HP (no max) counts as a win.
                e_max = max(getattr(after, "enemy_max_hp", 0), getattr(before, "enemy_max_hp", 0))
                won = e_max <= 0 or getattr(after, "enemy_hp", 0) == 0
                if not won:
                    add("battle", 0.0)
                else:
                    if budget is not None:
                        # The early-wild rule keys off the episode's own win count (the party level stays put).
                        wins = budget.get("wild_wins", 0)
                        budget["wild_wins"] = wins + 1
                        early = wins < WILD_WIN_FULL_COUNT
                    else:
                        early = max(getattr(before, "party_levels", (0,)) or [0]) < 7
                    add("battle", WILD_WIN_FULL if early else WILD_WIN_LATE, "battle")

    # HP dynamics: damage taken, healing (Pokémon Center or items), ally faint
    is_pc = map_id in POKECENTER_MAPS
    same_map = (getattr(before, "map_group", None), getattr(before, "map_number", None)) == map_id
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
            add("damage", -dmg_frac * 2.0)
        elif max_a > 0 and cur_a > cur_b:
            heal_frac = (cur_a - cur_b) / max_a
            leveled = level_gain > 0 or sum(p[1] for p in hp_a) != sum(p[1] for p in hp_b)
            if faint_ctx or cur_b == 0 or not same_map or leveled:
                pass  # white-out recovery (the teleport to a Pokécenter) or a level-up is not a heal to pay for
            elif not pay_heal:
                pass  # long_v4: src.heal_reward pays for HP actually restored instead
            elif is_pc:
                # Paid once per Pokémon Center per episode, so taking damage to heal again earns nothing.
                # Proportional to the fraction healed: one missing HP pays almost nothing.
                heal_key = (map_id[0], map_id[1] + 2000)
                if seen_waypoints is None or heal_key not in seen_waypoints:
                    if seen_waypoints is not None and heal_frac >= 0.25:
                        seen_waypoints.add(heal_key)
                    add("heal", heal_frac * (PC_HEAL_BASE + PC_HEAL_FRAC), "heal")
                    if info is not None:
                        info.healed_party = True
            else:
                add("heal", heal_frac * 3.0, "heal")  # a Potion or other item
                if info is not None:
                    info.healed_party = True
        elif is_pc and cur_b < max_b and same_map and not faint_ctx:
            # Low HP in Pokémon Center: reward choosing to interact with Nurse Joy
            if pay_heal and dialogue_open(after) and not dialogue_open(before):
                add("heal", 2.0, "heal")

        # One of the player's Pokémon fainted (not the white-out, which has its own penalty)
        fainted_b = sum(1 for p in hp_b if p[0] == 0)
        fainted_a = sum(1 for p in hp_a if p[0] == 0)
        if fainted_a > fainted_b and not getattr(after, "fainted", False):
            add("damage", -ALLY_FAINT_PENALTY * (fainted_a - fainted_b))

        # Danger pressure while the lead Pokémon is below 25% HP
        lead_cur, lead_max = hp_a[0]
        if lead_max > 0 and (lead_cur / lead_max) <= 0.25:
            add("damage", -LOW_HP_STEP_PENALTY)

    # White-out: the whole party fainted
    if getattr(after, "fainted", False) and not getattr(before, "fainted", False):
        add("whiteout", -WHITEOUT_PENALTY)

    # The reward is the sum of the terms, so the components add up to it exactly.
    reward = sum(terms.values())
    if components is not None:
        for key, value in terms.items():
            components[key] = components.get(key, 0.0) + value
    return reward, new_tile, new_map


def intrinsic_curiosity_reward(visit_count: int, scale: float = 0.08) -> float:
    """Count-based novelty bonus 1/sqrt(N) to propel exploration and break loops."""
    return float(scale / np.sqrt(max(1, visit_count)))


def observe(
    state,
    visited: set[tuple[int, int, int, int]],
    nav: LocalNav | None = None,
    deep: bool = False,
    seen_maps=None,
) -> np.ndarray:
    """Observation vector.

    - ``nav`` given: 17 base + LocalNav block + event flags + story block + battle block = ``OBS_DIM``.
    - ``deep=True`` (trainer / evaluator / play_live): 17 base + story block + battle block = ``DEEP_OBS_DIM``.
    - neither: just the 17 base features.

    The story block says which milestones are done and the current map id; the battle block (``src.battle_obs``)
    holds the battle, party and bag facts. There is NO next-step direction and NO distance to a goal anywhere.
    ``seen_maps`` (maps visited this episode) lets the milestone bits survive leaving a map; without it they are
    derived from the maps present in ``visited``.
    """
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
    if seen_maps is None and (deep or nav is not None):
        seen_maps = {k[:2] for k in visited}
    extra = [
        float(getattr(state, "party_count", 0)) / 6.0,
        float(max(getattr(state, "party_levels", (0,)) or [0])) / 20.0,
        float(getattr(state, "pokedex_caught_count", 0)) / 10.0,
        story_stage(state, seen_maps),
    ]
    base = np.asarray(tiles + pos + flags + neighbors + extra, dtype=np.float64)
    if deep:
        return np.concatenate([base, story_features(state, seen_maps), battle_features(state)])
    if nav is None:
        return base
    nav.update(state)
    in_menu = bool(state.battle_mode) or dialogue_open(state)
    has_event = getattr(state, "has_event", None)
    events = [1.0 if callable(has_event) and has_event(name) else 0.0 for name in OBS_EVENT_FLAGS]
    return np.concatenate([base, nav.features(state, in_menu=in_menu), np.asarray(events),
                           story_features(state, seen_maps), battle_features(state)])


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
    reward_components: dict = field(default_factory=dict)


def press_button(machine, name: str) -> None:
    """One raw A / B press (or a direction press inside a battle menu), as the policy asked for it."""
    machine.press(name, frames=8)
    machine.tick(12, render=False)


@dataclass
class CrystalEnv:
    """Legacy single-process env (tabular / MLP policies). The deep trainer uses ``tools.train_deep_ppo.EnvWorker``."""

    machine: object
    reader: object
    stepper: object
    macros: object
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    frontier_prob: float = 0.3
    max_steps: int = 64
    ignore_maps: frozenset = frozenset()  # (group, number) maps already completed; entering one ends the episode
    # "hybrid" / "frontier": learned-map frontier distance (potential over the agent's own explored tiles);
    # "none": no frontier shaping. (The hand-coded per-map potentials were removed with the oracle.)
    shaping: str = "hybrid"
    goal_potential: bool = False  # optional BFS potential toward Falkner (OFF by default; an oracle, for ablations)

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
        self.budget: dict = {}

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
        self.budget = {}
        state = self.reader.parse(self.machine.frame)
        self.seen_maps.add((state.map_group, state.map_number))
        tile = (state.map_group, state.map_number, state.x, state.y)
        self.seen_tiles.add(tile)
        self.visited.add(tile)
        # Seed initial caught counts from party
        for species_id in getattr(state, "party_species", ()):
            if species_id and species_id != 255:
                self.caught_counts[species_id] = self.caught_counts.get(species_id, 0) + 1
        # Seed initial event flags so flags already set at the start never pay
        self.seen_events.update(initial_seen_events(state))
        self.nav.update(state)
        self._phi = self.nav.potential(state)
        return observe(state, self.visited, self.nav, seen_maps=self.seen_maps)

    def _interaction_reward(self, before, after, action: int, busy: bool, components: dict | None = None) -> float:
        """One-time payments, as in pokemonred_puffer's NPC/sign/warp terms (explore-budgeted)."""
        bonus = 0.0
        if action == ACTION_A and not busy and dialogue_open(after):
            # In a Pokémon Center, do not award interaction bonus if party is already at full health
            if (before.map_group, before.map_number) in POKECENTER_MAPS:
                hp_b = getattr(before, "party_hp", ()) or ()
                cur_b = sum(p[0] for p in hp_b) if hp_b else 0
                max_b = sum(p[1] for p in hp_b) if hp_b else 0
                if cur_b >= max_b:
                    return bonus
            key = (before.map_group, before.map_number, before.x, before.y, before.facing)
            if key not in self.seen_interactions:
                self.seen_interactions.add(key)
                pay = budget_pay(self.budget, "explore", NEW_INTERACTION_REWARD)
                bonus += pay
                if components is not None:
                    components["interaction"] = components.get("interaction", 0.0) + pay
        before_map = (before.map_group, before.map_number)
        after_map = (after.map_group, after.map_number)
        if before_map != after_map:
            edge = (before_map, after_map)
            if edge not in self.seen_warps:
                self.seen_warps.add(edge)
                warp = budget_pay(self.budget, "explore", NEW_WARP_REWARD)
                bonus += warp
                if components is not None:
                    components["warp"] = components.get("warp", 0.0) + warp
        return bonus

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        before = self.reader.parse(self.machine.frame)
        info = StepInfo()
        name = ACTIONS[action]
        moved = False
        if before.battle_mode:
            # Battles are learned: the policy's button goes straight to the game (no tactical script).
            press_button(self.machine, name)
        elif dialogue_open(before):
            # Overworld dialogue only is auto-advanced (episode feasibility); battle text is never mashed.
            self.macros.mash_text(max_frames=80)
        elif name in BUTTON_ACTIONS:
            press_button(self.machine, name)
            self.machine.tick(4, render=False)
        else:
            moved = self.stepper.step(name).moved
        if not moved and name not in BUTTON_ACTIONS and not before.battle_mode and not dialogue_open(before):
            self.consecutive_bumps += 1  # an A / B press is not a bump
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
        components: dict[str, float] = {}
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
            components=components,
            budget=self.budget,
            goal_potential=self.goal_potential,
        )
        reward += self._interaction_reward(before, after, action, busy, components)
        tile = (after.map_group, after.map_number, after.x, after.y)
        self.visited.add(tile)
        map_id = (after.map_group, after.map_number)
        if (info.discovered_map or after.badges > before.badges) and map_id not in self.frontier_maps:
            self.frontier.append(self.machine.save_state())
            self.frontier_maps.add(map_id)
            info.frontier = True
        self.steps += 1
        done = self.steps >= self.max_steps
        if behind_map_penalty_applies(before, after, self.ignore_maps, after):
            reward, done = -BEHIND_MAP_PENALTY, True
            components = {"behind_map": reward}
        info.reward_components = components
        if not getattr(self.machine, "headless", True):
            self.machine.tick(1, render=True)
        return observe(after, self.visited, self.nav, seen_maps=self.seen_maps), reward, done, info
