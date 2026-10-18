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


class VioletHealTests(unittest.TestCase):
    def test_gym_exit_retraces_the_forward_route_in_reverse(self):
        from src.falkner_nav import violet_gym_step

        # Walk the forward route (entrance -> Falkner), recording every tile, ignoring walls.
        delta = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        x, y, facing, forward = 4, 14, "up", [(4, 14)]
        for _ in range(200):
            step = violet_gym_step(x, y, facing)
            if step == "a":
                break
            facing = step
            x, y = x + delta[step][0], y + delta[step][1]
            forward.append((x, y))
        self.assertEqual(violet_gym_step(x, y, facing), "a", "forward route should end facing Falkner")

        # From any tile on it, the exit steps must stay on the route and reach the bottom of the straight.
        # the bottom straight is two tiles wide (x 4..5, y 11..15), including the two exit tiles at y=15
        on_route = set(forward) | {(x, y) for x in (4, 5) for y in range(11, 16)}
        for sx, sy in forward:
            x, y = sx, sy
            for _ in range(200):
                if y >= 15:
                    break
                d = gym_prep.gym_exit_step(x, y)
                x, y = x + delta[d][0], y + delta[d][1]
                self.assertIn((x, y), on_route, f"left the route at {(x, y)} starting from {(sx, sy)}")
            self.assertGreaterEqual(y, 15, f"never reached the exit from {(sx, sy)}")

    def test_hurt_in_the_gym_heads_for_the_exit_and_a_healthy_lead_is_left_alone(self):
        hurt = make_state(mid=gym_prep.VIOLET_GYM, x=4, y=5, level=15, hp=(10, 35))
        self.assertEqual(prep_direction(hurt), gym_prep.gym_exit_step(4, 5))
        self.assertIsNone(prep_direction(make_state(mid=gym_prep.VIOLET_GYM, x=4, y=5, level=15, hp=(30, 35))))

    def test_walks_across_violet_city_from_the_gym_door_to_the_pokemon_center(self):
        state = walk(make_state(mid=gym_prep.VIOLET_CITY, x=18, y=17, level=15, hp=(10, 35)), limit=600)
        self.assertEqual((state.x, state.y), gym_prep.VIOLET_PC_DOOR)

    def test_arriving_in_violet_city_hurt_heals_before_the_gym(self):
        d = prep_direction(make_state(mid=gym_prep.VIOLET_CITY, x=18, y=17, level=15, hp=(34, 35)))
        self.assertIn(d, _DIRS)
        self.assertIsNone(prep_direction(make_state(mid=gym_prep.VIOLET_CITY, x=18, y=17, level=15, hp=(35, 35))))

    def test_no_prep_once_falkner_is_beaten(self):
        self.assertIsNone(prep_direction(make_state(mid=gym_prep.VIOLET_CITY, x=18, y=17, hp=(5, 35), beat_falkner=True)))


def walk_one(state):
    nav = _nav()
    d = prep_direction(state)
    dx, dy = _DIRS[d]
    nxt = nav._step((state.map_group, state.map_number), state.x + dx, state.y + dy, d)
    state.map_group, state.map_number, state.x, state.y = nxt[0][0], nxt[0][1], nxt[1], nxt[2]
    return state


if __name__ == "__main__":
    unittest.main()
