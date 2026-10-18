import unittest
from types import SimpleNamespace

from src import gym_prep
from src.gym_prep import (
    CHERRYGROVE,
    PC_DOOR_OUTSIDE,
    ROUTE_30,
    TARGET_LEVEL,
    _DIRS,
    _nav,
    needs_training,
    prep_direction,
)


def make_state(mid=ROUTE_30, x=10, y=45, level=8, hp=(25, 25), gave_egg=True, beat_falkner=False, battle_mode=0):
    events = set()
    if gave_egg:
        events.add("EVENT_GAVE_MYSTERY_EGG_TO_ELM")
    if beat_falkner:
        events.add("EVENT_BEAT_FALKNER")
    return SimpleNamespace(
        map_group=mid[0], map_number=mid[1], x=x, y=y, battle_mode=battle_mode,
        party_levels=(level, 3, 3), party_hp=(hp, (15, 15), (15, 15)),
        has_event=lambda name: name in events,
    )


def walk(state, limit=400):
    """Follow prep_direction on the real ROM map until it stops; returns the last state."""
    nav = _nav()
    for _ in range(limit):
        d = prep_direction(state)
        if d is None:
            return state
        mid = (state.map_group, state.map_number)
        dx, dy = _DIRS[d]
        nxt = nav._step(mid, state.x + dx, state.y + dy, d)
        if nxt is None:
            return state
        state.map_group, state.map_number = nxt[0]
        state.x, state.y = nxt[1], nxt[2]
        if d == "up" and (nxt[0], nxt[1], nxt[2]) == (CHERRYGROVE, PC_DOOR_OUTSIDE[0], PC_DOOR_OUTSIDE[1] - 1):
            return state
    return state


class GymPrepTests(unittest.TestCase):
    def setUp(self):
        gym_prep._last_pace = None

    def test_training_only_between_egg_delivery_and_falkner_while_under_level(self):
        self.assertTrue(needs_training(make_state(level=8)))
        self.assertFalse(needs_training(make_state(level=TARGET_LEVEL)))
        self.assertFalse(needs_training(make_state(gave_egg=False)))
        self.assertFalse(needs_training(make_state(beat_falkner=True)))

    def test_nothing_in_battle_or_on_other_maps(self):
        self.assertIsNone(prep_direction(make_state(battle_mode=1)))
        self.assertIsNone(prep_direction(make_state(mid=(24, 3))))
        self.assertIsNone(prep_direction(make_state(level=TARGET_LEVEL)))

    def test_healthy_in_cherrygrove_leaves_the_route_to_walk_north(self):
        self.assertIsNone(prep_direction(make_state(mid=CHERRYGROVE, x=20, y=8)))

    def test_walks_to_tall_grass_then_paces_between_grass_tiles(self):
        nav = _nav()
        state = make_state(x=10, y=48)
        for _ in range(300):
            if nav.is_grass(ROUTE_30, state.x, state.y):
                break
            state = walk_one(state)
        self.assertTrue(nav.is_grass(ROUTE_30, state.x, state.y), "never reached tall grass")
        seen = []
        for _ in range(6):
            state = walk_one(state)
            seen.append((state.x, state.y))
            self.assertTrue(nav.is_grass(ROUTE_30, state.x, state.y), "left the grass while pacing")
        self.assertGreater(len(set(seen)), 1)

    def test_low_hp_walks_back_to_the_pokemon_center_door(self):
        state = walk(make_state(x=10, y=40, hp=(5, 25)))
        self.assertEqual((state.map_group, state.map_number), CHERRYGROVE)
        self.assertEqual((state.x, state.y), (PC_DOOR_OUTSIDE[0], PC_DOOR_OUTSIDE[1] - 1))


def walk_one(state):
    nav = _nav()
    d = prep_direction(state)
    dx, dy = _DIRS[d]
    nxt = nav._step((state.map_group, state.map_number), state.x + dx, state.y + dy, d)
    state.map_group, state.map_number, state.x, state.y = nxt[0][0], nxt[0][1], nxt[1], nxt[2]
    return state


if __name__ == "__main__":
    unittest.main()
