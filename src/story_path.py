"""The minimal story path from the New Bark start state to the Zephyr Badge, and nothing else.

What is REQUIRED (checked against the ROM-derived notes: ``src/story_flags.py``, ``tools/build_route_map.py``,
``src/route_graph.py`` and the audit in ``research/analysis``):

    leave New Bark -> Route 29 -> Cherrygrove City -> Route 30 -> Route 31 -> Violet gatehouse
    -> Violet City -> Violet Gym -> beat Falkner (Zephyr Badge).

Only taking a starter from Elm is a hard gate, and the start state already holds Cyndaquil. The Mystery Egg
errand (Mr. Pokemon, the Cherrygrove rival, returning to Elm) and the Togepi egg are NOT required, so they are not
goals, not rewarded and not in the observation. Sprout Tower is optional too.

EXTENDING: append a ``Milestone`` to ``MILESTONES`` (a map the agent must reach, or an event/badge) and give it a
reward; ``MILESTONE_COUNT``, the observation block and the reward all follow. The next hard gate after the badge is
Route 32 -> Azalea Town (needs nothing but walking), but the run stops at the badge for now.

None of this tells the policy WHERE to go. Milestones are paid once per episode on first arrival and exposed to the
policy only as "which milestones are done" bits.
"""

from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class Milestone:
    name: str
    map_id: tuple[int, int] | None  # arrival at this (group, number) map completes it; None = event / badge based
    reward: float


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
MILESTONE_COUNT = len(MILESTONES)
MAP_MILESTONES: dict[tuple[int, int], Milestone] = {m.map_id: m for m in MILESTONES if m.map_id is not None}
BADGE_REWARD = 60.0
FALKNER_EVENT = "EVENT_BEAT_FALKNER"
# Named event flags that pay (everything else pays UNNAMED_EVENT_REWARD or nothing, see rl_env).
KEY_EVENT_REWARDS: dict[str, float] = {FALKNER_EVENT: 10.0}
TOTAL_STORY_REWARD = sum(m.reward for m in MILESTONES) + BADGE_REWARD + sum(KEY_EVENT_REWARDS.values())

STORY_FEATURE_DIM = MILESTONE_COUNT + 2   # milestone done bits + map group / number (scaled)
START_MAP = (24, 4)                        # New Bark Town, where states/new_bark_outside.state starts
# Length of the minimal story path in ACTIONS (BFS over the ROM graph, ledge jumps = one action), New Bark start tile ->
# the tile in front of Falkner. Hard-coded here so the cloud launcher and the dashboard need no ROM; tests/test_route_graph.py
# asserts it equals ``route_graph.path_total()``. It was 319 before the ledge fix (no jump edges in the graph).
STORY_PATH_STEPS = 309


def _map_of(state) -> tuple[int, int]:
    return int(getattr(state, "map_group", 0)), int(getattr(state, "map_number", 0))


def beat_falkner(state) -> bool:
    if getattr(state, "badges", 0) > 0:
        return True
    fn = getattr(state, "has_event", None)
    try:
        return bool(callable(fn) and fn(FALKNER_EVENT))
    except Exception:
        return False


def milestones_done(state, seen_maps=None) -> list[bool]:
    """Done flags in ``MILESTONES`` order. Map milestones count once the map was seen this episode (or is current)."""
    maps = set(seen_maps) if seen_maps is not None else set()
    maps.add(_map_of(state))
    done = [m.map_id in maps if m.map_id is not None else beat_falkner(state) for m in MILESTONES]
    if done[-1]:  # a beaten gym implies the whole path behind it, even from a mid-path start state
        done = [True] * MILESTONE_COUNT
    return done


def story_fraction(state, seen_maps=None) -> float:
    return sum(milestones_done(state, seen_maps)) / float(MILESTONE_COUNT)


def story_features(state, seen_maps=None) -> "np.ndarray":
    """STORY_FEATURE_DIM floats: which milestones are done, plus the current map id. No direction, no distance."""
    import numpy as np   # lazy: tools.cloud_train_deep imports src.train_config -> dwell -> story_path under a bare WSL python
    done = milestones_done(state, seen_maps)
    g, n = _map_of(state)
    return np.asarray([1.0 if d else 0.0 for d in done] + [g / 32.0, n / 32.0], dtype=np.float64)


def current_stage(state) -> int:
    """Coarse position along the path for start-state balancing and the swarm archive (not an observation).

    0 = New Bark / unknown or interior map, k = the map of milestone k - 1 (Route 29 = 1 ... Violet Gym = 7),
    MILESTONE_COUNT = Falkner beaten.
    """
    if beat_falkner(state):
        return MILESTONE_COUNT
    m = MAP_MILESTONES.get(_map_of(state))
    return 0 if m is None else MILESTONES.index(m) + 1


def progress_potential(state, seen_maps) -> float:
    """-(milestones left): 0 when the whole path is done. Only for the entropy schedule / reporting."""
    return -float(MILESTONE_COUNT - sum(milestones_done(state, seen_maps)))
