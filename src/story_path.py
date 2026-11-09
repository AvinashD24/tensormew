"""The minimal story path from the New Bark start state to the Zephyr Badge, and nothing else.

Two PROFILES (``set_profile``; the default is ``base`` so every preset up to ``long_v7`` is unchanged):

``base`` (the original list, 8 milestones)::

    leave New Bark -> Route 29 -> Cherrygrove City -> Route 30 -> Route 31 -> Violet gatehouse
    -> Violet City -> Violet Gym -> beat Falkner (Zephyr Badge).

  The base route is the "no egg" shortcut and is NOT reachable from game start: Route 30 is cut in half by NPC sprites
  (Joey's event, two Rattatas, trainer Mikey) that only disappear once Elm has the Mystery Egg (hide flag 1812, set together
  with EVENT_GAVE_MYSTERY_EGG_TO_ELM).  Runs that use it (long_v3..long_v7) can therefore never pass Route 30 from the
  start state; the milestones after it are only reachable from archive / checkpoint states taken after the egg errand.

``egg`` (``TrainConfig.egg_errand_route``, 11 milestones; ``long_v8``) puts the REQUIRED errand back in::

    Route 29 -> Cherrygrove -> Route 30 -> Mr. Pokemon's house (get the Mystery Egg: EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON)
    -> rival fight at Cherrygrove (scripted, on the way back) -> Elm's lab in New Bark (give the egg:
    EVENT_GAVE_MYSTERY_EGG_TO_ELM + Route 30 opens) -> Cherrygrove -> Route 30 -> Route 31 -> Violet gatehouse -> Violet City
    -> Violet Gym -> beat Falkner (Zephyr Badge).

  Only taking a starter from Elm is a hard gate before that, and the start state already holds Cyndaquil.  The Togepi egg
  and Sprout Tower stay optional.  ``src.route_graph`` measures the distance along these three phases
  (``path_progress(state, egg=True)``) from the same event flags.

EXTENDING: append a ``Milestone`` (a map the agent must reach, an event flag, or the badge) to a profile list and give it a
reward; the milestone count, the observation block and the reward all follow.  The next hard gate after the badge is
Route 32 -> Azalea Town (needs nothing but walking), but the run stops at the badge for now.

None of this tells the policy WHERE to go. Milestones are paid once per episode on first arrival and exposed to the
policy only as "which milestones are done" bits.

The module-level names (``MILESTONES``, ``MILESTONE_COUNT``, ``STORY_FEATURE_DIM``, ...) are the BASE profile and never
change.  Code that must follow the run's profile calls ``profile()`` (or passes ``profile=``) at use time; the trainer's
workers, the trainer and the evaluator call ``set_profile`` once at startup.  No numpy at import time (the cloud launcher
imports this through ``src.train_config`` under a bare python).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class Milestone:
    name: str
    map_id: tuple[int, int] | None  # arrival at this (group, number) map completes it; None = event / badge based
    reward: float
    event: str | None = None        # an event-flag milestone (EVENT_...): done while the flag is set, paid on its first flip


BADGE_REWARD = 60.0
FALKNER_EVENT = "EVENT_BEAT_FALKNER"
EGG_RECEIVED_EVENT = "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"
EGG_DELIVERED_EVENT = "EVENT_GAVE_MYSTERY_EGG_TO_ELM"
FALKNER_EVENT_REWARD = 10.0

MILESTONES: tuple[Milestone, ...] = (
    Milestone("route29", (24, 3), 5.0),
    Milestone("cherrygrove", (26, 3), 10.0),
    Milestone("route30", (26, 1), 10.0),
    Milestone("route31", (26, 2), 10.0),
    Milestone("violet_gate", (26, 11), 5.0),   # the gatehouse between Route 31 and Violet City
    Milestone("violet_city", (10, 5), 15.0),
    Milestone("violet_gym", (10, 7), 15.0),
    Milestone("beat_falkner", None, 0.0),      # paid as the badge (BADGE_REWARD) + EVENT_BEAT_FALKNER
)

# The egg-errand list: the same common milestones with the same rewards, plus Mr. Pokemon's house (+10), the egg in hand
# (+15) and the egg given to Elm (+30; also what opens Route 30).  Total 195 (base 140), see ``StoryProfile.total_reward``.
EGG_MILESTONES: tuple[Milestone, ...] = (
    Milestone("route29", (24, 3), 5.0),
    Milestone("cherrygrove", (26, 3), 10.0),
    Milestone("route30", (26, 1), 10.0),
    Milestone("mr_pokemon_house", (26, 10), 10.0),
    Milestone("egg_received", None, 15.0, EGG_RECEIVED_EVENT),
    Milestone("egg_delivered", None, 30.0, EGG_DELIVERED_EVENT),
    Milestone("route31", (26, 2), 10.0),
    Milestone("violet_gate", (26, 11), 5.0),
    Milestone("violet_city", (10, 5), 15.0),
    Milestone("violet_gym", (10, 7), 15.0),
    Milestone("beat_falkner", None, 0.0),
)

# Length of the minimal story path in ACTIONS (BFS over the ROM graph, ledge jumps = one action).  Hard-coded so the cloud
# launcher and the dashboard need no ROM; tests/test_route_graph.py asserts them against ``route_graph``.  The base number
# is New Bark start tile -> the tile in front of Falkner (319 before the ledge fix).  The egg route is three phases:
# start -> Mr. Pokemon's egg tile, -> Elm's lab delivery tile, -> the tile in front of Falkner (the last one is the 309-step
# route plus the walk from Elm's lab back to the New Bark start tile).
STORY_PATH_STEPS = 309
EGG_PHASE_STEPS = (185, 183, 327)
EGG_PATH_STEPS = sum(EGG_PHASE_STEPS)   # 695


class StoryProfile:
    """One milestone list and everything derived from it (counts, dims, reward tables)."""

    def __init__(self, name: str, milestones: tuple[Milestone, ...], path_steps: int, phase_steps: tuple[int, ...] | None):
        self.name = name
        self.milestones = tuple(milestones)
        self.count = len(self.milestones)
        self.map_milestones: dict[tuple[int, int], Milestone] = {m.map_id: m for m in self.milestones if m.map_id is not None}
        self.event_milestones: tuple[tuple[int, Milestone], ...] = tuple(
            (i, m) for i, m in enumerate(self.milestones) if m.event is not None)
        # Named event flags that pay (everything else pays UNNAMED_EVENT_REWARD or nothing, see rl_env).
        self.key_event_rewards: dict[str, float] = {FALKNER_EVENT: FALKNER_EVENT_REWARD}
        self.key_event_rewards.update({m.event: m.reward for _i, m in self.event_milestones})
        self.total_reward = sum(m.reward for m in self.milestones) + BADGE_REWARD + FALKNER_EVENT_REWARD
        self.feature_dim = self.count + 2    # milestone done bits + map group / number (scaled)
        self.path_steps = int(path_steps)
        self.phase_steps = tuple(phase_steps) if phase_steps else None
        self.egg = bool(self.event_milestones)
        self.names = tuple(m.name for m in self.milestones)

    def __repr__(self) -> str:
        return f"StoryProfile({self.name!r}, {self.count} milestones, feature_dim={self.feature_dim})"


BASE_PROFILE = StoryProfile("base", MILESTONES, STORY_PATH_STEPS, None)
EGG_PROFILE = StoryProfile("egg", EGG_MILESTONES, EGG_PATH_STEPS, EGG_PHASE_STEPS)
_active: StoryProfile = BASE_PROFILE


def profile_for(egg: bool) -> StoryProfile:
    return EGG_PROFILE if egg else BASE_PROFILE


def profile() -> StoryProfile:
    """The profile of this process (``base`` unless ``set_profile`` switched it)."""
    return _active


def set_profile(egg) -> StoryProfile:
    """Switch this process to the egg (True / ``"egg"``) or base (False / ``"base"``) profile; returns it."""
    global _active
    if isinstance(egg, StoryProfile):
        _active = egg
    else:
        _active = profile_for(egg is True or egg == "egg")
    return _active


@contextmanager
def use_profile(egg):
    """Temporarily switch the profile (tests, tools); restores the previous one on exit."""
    prev = _active
    try:
        yield set_profile(egg)
    finally:
        set_profile(prev)


# --- base-profile module constants (unchanged; import-time users such as the legacy observation keep working) ---------
MILESTONE_COUNT = BASE_PROFILE.count
MAP_MILESTONES: dict[tuple[int, int], Milestone] = BASE_PROFILE.map_milestones
KEY_EVENT_REWARDS: dict[str, float] = BASE_PROFILE.key_event_rewards
TOTAL_STORY_REWARD = BASE_PROFILE.total_reward
STORY_FEATURE_DIM = BASE_PROFILE.feature_dim   # milestone done bits + map group / number (scaled)
START_MAP = (24, 4)                             # New Bark Town, where states/new_bark_outside.state starts


def _map_of(state) -> tuple[int, int]:
    return int(getattr(state, "map_group", 0)), int(getattr(state, "map_number", 0))


def beat_falkner(state) -> bool:
    if getattr(state, "badges", 0) > 0:
        return True
    return _has_event(state, FALKNER_EVENT)


def _has_event(state, name: str) -> bool:
    fn = getattr(state, "has_event", None)
    try:
        return bool(callable(fn) and fn(name))
    except Exception:
        return False


def milestones_done(state, seen_maps=None, profile: StoryProfile | None = None) -> list[bool]:
    """Done flags in milestone order. Map milestones count once the map was seen this episode (or is current); event
    milestones while their flag is set.  A beaten gym implies the whole path behind it (even from a mid-path start state),
    and a done event milestone implies every milestone before it in the list (the egg in hand means Mr. Pokemon's house,
    Route 30, Cherrygrove and Route 29 are behind you)."""
    p = profile or _active
    maps = set(seen_maps) if seen_maps is not None else set()
    maps.add(_map_of(state))
    done = []
    for m in p.milestones:
        if m.map_id is not None:
            done.append(m.map_id in maps)
        elif m.event is not None:
            done.append(_has_event(state, m.event))
        else:
            done.append(beat_falkner(state))
    if done[-1]:
        return [True] * p.count
    last_event = max((i for i, m in p.event_milestones if done[i]), default=-1)
    for i in range(last_event):
        done[i] = True
    return done


def implied_maps(state, profile: StoryProfile | None = None) -> set[tuple[int, int]]:
    """Milestone maps already passed according to the EVENT flags of ``state`` (empty in the base profile): a state with the egg
    in hand has been through Route 29, Cherrygrove, Route 30 and Mr. Pokemon's house.  Workers add them to ``seen_maps`` at
    reset so a loaded checkpoint never pays those arrival bonuses again."""
    p = profile or _active
    last = max((i for i, m in p.event_milestones if _has_event(state, m.event)), default=-1)
    return {tuple(m.map_id) for m in p.milestones[:max(last, 0)] if m.map_id is not None}


def story_fraction(state, seen_maps=None) -> float:
    p = _active
    return sum(milestones_done(state, seen_maps, p)) / float(p.count)


def story_features(state, seen_maps=None) -> "np.ndarray":
    """``profile().feature_dim`` floats: which milestones are done, plus the current map id. No direction, no distance."""
    import numpy as np   # lazy: tools.cloud_train_deep imports src.train_config -> dwell -> story_path under a bare WSL python
    done = milestones_done(state, seen_maps, _active)
    g, n = _map_of(state)
    return np.asarray([1.0 if d else 0.0 for d in done] + [g / 32.0, n / 32.0], dtype=np.float64)


def current_stage(state, profile: StoryProfile | None = None) -> int:
    """Coarse position along the path for start-state balancing and the swarm archive (not an observation).

    0 = New Bark / unknown or interior map, k = milestone k - 1 (a map milestone while standing in its map, an event milestone
    while its flag is set; in the base profile Route 29 = 1 ... Violet Gym = 7), ``count`` = Falkner beaten.
    """
    p = profile or _active
    if beat_falkner(state):
        return p.count
    m = p.map_milestones.get(_map_of(state))
    stage = 0 if m is None else p.milestones.index(m) + 1
    for i, em in p.event_milestones:
        if i + 1 > stage and _has_event(state, em.event):
            stage = i + 1
    return stage


def progress_potential(state, seen_maps) -> float:
    """-(milestones left): 0 when the whole path is done. Only for the entropy schedule / reporting."""
    p = _active
    return -float(p.count - sum(milestones_done(state, seen_maps, p)))
