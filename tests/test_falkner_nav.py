"""Tests for src/falkner_nav.py: deterministic navigation from Elm's Lab through Falkner defeat."""

import unittest
from types import SimpleNamespace

from src.falkner_nav import (
    MAP_CHERRYGROVE,
    MAP_ELM_LAB,
    MAP_NEW_BARK,
    MAP_ROUTE_29,
    MAP_ROUTE_30,
    MAP_ROUTE_31,
    MAP_VIOLET_CITY,
    MAP_VIOLET_GATE,
    MAP_VIOLET_GYM,
    falkner_nav_direction,
    violet_gym_step,
)


def _make_state(map_id, x, y, facing="down", events=()):
    ev_set = set(events)
    return SimpleNamespace(
        map_id=map_id,
        x=x,
        y=y,
        facing=facing,
        has_event=lambda ev: ev in ev_set,
        is_battle=False,
    )


class TestFalknerNav(unittest.TestCase):
    def test_inactive_before_egg_delivered(self):
        st = _make_state(MAP_ELM_LAB, 5, 3, events=())
        self.assertIsNone(falkner_nav_direction(st))

    def test_inactive_after_falkner_beaten(self):
        st = _make_state(MAP_VIOLET_GYM, 5, 3, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM", "EVENT_BEAT_FALKNER"))
        self.assertIsNone(falkner_nav_direction(st))

    def test_elm_lab_exit_navigation(self):
        # Facing Elm at (5, 3) post-egg
        st = _make_state(MAP_ELM_LAB, 5, 3, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st), "down")

        # Walking down carpet at (5, 11) -> step out
        st_exit = _make_state(MAP_ELM_LAB, 5, 11, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_exit), "down")

    def test_new_bark_town_exit_to_route29(self):
        # Outside lab at (6, 4)
        st = _make_state(MAP_NEW_BARK, 6, 4, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st), "down")

        # On path at (6, 8) -> head west
        st_path = _make_state(MAP_NEW_BARK, 6, 8, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_path), "left")

        # At border (0, 8) -> step into Route 29
        st_border = _make_state(MAP_NEW_BARK, 0, 8, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_border), "left")

    def test_route29_westbound(self):
        # East entrance from New Bark at (59, 8)
        st = _make_state(MAP_ROUTE_29, 59, 8, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        step = falkner_nav_direction(st)
        self.assertIn(step, ("left", "up", "down"))

        # At west gate (0, 8) -> step into Cherrygrove
        st_gate = _make_state(MAP_ROUTE_29, 0, 8, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_gate), "left")

    def test_cherrygrove_northbound(self):
        # East entrance from Route 29 at (39, 6)
        st = _make_state(MAP_CHERRYGROVE, 39, 6, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        step = falkner_nav_direction(st)
        self.assertIn(step, ("left", "up"))

        # At northern connection (16, 0) -> step into Route 30
        st_conn = _make_state(MAP_CHERRYGROVE, 16, 0, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_conn), "up")

    def test_route30_northbound_along_west_path(self):
        # South entrance at (7, 53)
        st = _make_state(MAP_ROUTE_30, 7, 53, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        step = falkner_nav_direction(st)
        self.assertIn(step, ("up", "left", "right"))

        # At north connection (6, 0) -> step into Route 31
        st_exit = _make_state(MAP_ROUTE_30, 6, 0, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_exit), "up")

    def test_route31_westbound_to_gatehouse(self):
        # South entrance at (25, 17)
        st = _make_state(MAP_ROUTE_31, 25, 17, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        step = falkner_nav_direction(st)
        self.assertIn(step, ("up", "left"))

        # At gatehouse doorway (5, 7) -> step left into gatehouse
        st_door = _make_state(MAP_ROUTE_31, 5, 7, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_door), "left")

    def test_violet_gatehouse(self):
        # Inside gatehouse at (9, 4)
        st = _make_state(MAP_VIOLET_GATE, 9, 4, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st), "left")

        # At west door (0, 4) -> step into Violet City
        st_exit = _make_state(MAP_VIOLET_GATE, 0, 4, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_exit), "left")

    def test_violet_city_to_gym(self):
        # East entrance from gatehouse at (39, 25) with healthy party -> heads to Gym
        st = _make_state(MAP_VIOLET_CITY, 39, 25, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        step = falkner_nav_direction(st)
        self.assertIn(step, ("left", "up"))

        # Outside Violet Gym at (18, 18) -> step into Gym
        st_gym = _make_state(MAP_VIOLET_CITY, 18, 18, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        self.assertEqual(falkner_nav_direction(st_gym), "up")

    def test_violet_city_routes_to_pokecenter_when_damaged(self):
        from src.falkner_nav import MAP_VIOLET_POKECENTER
        # Entering Violet City at (39, 25) with fainted/damaged Pokémon
        st = _make_state(MAP_VIOLET_CITY, 39, 25, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st.party_hp = ((0, 20), (5, 15))
        step = falkner_nav_direction(st)
        self.assertEqual(step, "left")

        # Outside Violet PC doorway at (31, 26) when damaged -> step up to enter
        st_door = _make_state(MAP_VIOLET_CITY, 31, 26, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st_door.party_hp = ((0, 20), (5, 15))
        self.assertEqual(falkner_nav_direction(st_door), "up")

        # Inside Violet PC at (3, 3) facing up with damaged party -> press A to talk to Nurse Joy
        st_nurse = _make_state(MAP_VIOLET_POKECENTER, 3, 3, facing="up", events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st_nurse.party_hp = ((0, 20), (5, 15))
        self.assertEqual(falkner_nav_direction(st_nurse), "a")

        # Inside Violet PC fully healed on exit mat at (3, 7) -> step down to exit
        st_healed = _make_state(MAP_VIOLET_POKECENTER, 3, 7, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st_healed.party_hp = ((20, 20), (15, 15))
        self.assertEqual(falkner_nav_direction(st_healed), "down")

    def test_route31_tall_grass_training_when_underleveled(self):
        # On Route 31 at (20, 10) with Cyndaquil Lv 8 (no Ember) -> routes towards grass patch
        st = _make_state(MAP_ROUTE_31, 20, 10, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st.party_levels = (8,)
        st.party_moves = ((33, 43, 108, 0),)
        st.party_hp = ((25, 25),)
        step = falkner_nav_direction(st)
        self.assertIn(step, ("left", "down", "up", "right"))

        # Inside grass at (9, 12) -> paces right to encounter wild Pokémon
        st_grass = _make_state(MAP_ROUTE_31, 9, 12, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st_grass.party_levels = (8,)
        st_grass.party_moves = ((33, 43, 108, 0),)
        st_grass.party_hp = ((25, 25),)
        self.assertEqual(falkner_nav_direction(st_grass), "right")

    def test_violet_city_routes_to_kyle_house_with_bellsprout(self):
        from src.falkner_nav import MAP_VIOLET_KYLES_HOUSE
        # In Violet City at (21, 30) with Bellsprout (69) and healthy party -> step up to enter Kyle's house
        st = _make_state(MAP_VIOLET_CITY, 21, 30, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st.party_species = (155, 69)
        st.party_hp = ((25, 25), (15, 15))
        self.assertEqual(falkner_nav_direction(st), "up")

        # Inside Kyle's house at (2, 3) facing right -> talk to Kyle
        st_kyle = _make_state(MAP_VIOLET_KYLES_HOUSE, 2, 3, facing="right", events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st_kyle.party_species = (155, 69)
        self.assertEqual(falkner_nav_direction(st_kyle), "a")

        # Inside Kyle's house after trade (Onix 95 received) at exit mat (3, 7) -> step down to exit
        st_traded = _make_state(MAP_VIOLET_KYLES_HOUSE, 3, 7, events=("EVENT_GAVE_MYSTERY_EGG_TO_ELM",))
        st_traded.party_species = (155, 95)
        self.assertEqual(falkner_nav_direction(st_traded), "down")

    def test_violet_gym_s_curve_and_falkner_challenge(self):
        # Entrance at (4, 15) -> walk up
        self.assertEqual(violet_gym_step(4, 15, "up"), "up")

        # At (4, 10) -> turn right
        self.assertEqual(violet_gym_step(4, 10, "up"), "right")

        # Facing Falkner at (5, 3) facing up -> press A to challenge!
        self.assertEqual(violet_gym_step(5, 3, "up"), "a")

        # In front of Falkner facing right -> turn up
        self.assertEqual(violet_gym_step(5, 3, "right"), "up")


if __name__ == "__main__":
    unittest.main()
