import unittest
from dataclasses import dataclass

from src.rl_env import navigation_potential, story_stage, progress_reward


@dataclass
class MockState:
    map_group: int
    map_number: int
    x: float
    y: float
    events: set[str]
    cherrygrove_scene: int = 0
    elms_lab_scene: int = 0
    badges: int = 0
    battle_mode: int = 0
    script_running: int = 0
    script_mode: int = 0
    party_hp: tuple = ((20, 20),)

    @property
    def map_id(self):
        return (self.map_group, self.map_number)

    @property
    def beat_rival(self) -> bool:
        if "EVENT_GAVE_MYSTERY_EGG_TO_ELM" in self.events:
            return True
        if "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON" not in self.events:
            return False
        return self.cherrygrove_scene == 0

    def has_event(self, flag: str) -> bool:
        if flag == "EVENT_RIVAL_CHERRYGROVE_CITY":
            return self.beat_rival
        return flag in self.events


class EggRivalElmPolicyTests(unittest.TestCase):
    def test_rival_flag_behavior(self):
        # Initial state: no egg, rival not beaten
        st0 = MockState(24, 4, 0, 9, {"EVENT_GOT_A_POKEMON_FROM_ELM"}, cherrygrove_scene=0)
        self.assertFalse(st0.has_event("EVENT_RIVAL_CHERRYGROVE_CITY"))
        self.assertFalse(st0.beat_rival)

        # After getting egg: cherrygrove scene is 1 (pending rival battle)
        st_egg = MockState(26, 10, 2, 6, {"EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}, cherrygrove_scene=1)
        self.assertFalse(st_egg.has_event("EVENT_RIVAL_CHERRYGROVE_CITY"))
        self.assertFalse(st_egg.beat_rival)

        # After beating rival: cherrygrove scene resets to 0
        st_rival_beaten = MockState(26, 3, 33, 6, {"EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}, cherrygrove_scene=0)
        self.assertTrue(st_rival_beaten.has_event("EVENT_RIVAL_CHERRYGROVE_CITY"))
        self.assertTrue(st_rival_beaten.beat_rival)

        # After giving egg to Elm
        st_egg_given = MockState(24, 5, 4, 3, {"EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON", "EVENT_GAVE_MYSTERY_EGG_TO_ELM"}, cherrygrove_scene=0)
        self.assertTrue(st_egg_given.has_event("EVENT_RIVAL_CHERRYGROVE_CITY"))
        self.assertTrue(st_egg_given.beat_rival)

    def test_story_stage_monotonicity(self):
        st_start = MockState(24, 4, 0, 9, set())
        st_starter = MockState(24, 4, 0, 9, {"EVENT_GOT_A_POKEMON_FROM_ELM"})
        st_egg = MockState(26, 10, 2, 6, {"EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}, cherrygrove_scene=1)
        st_rival = MockState(26, 3, 33, 6, {"EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}, cherrygrove_scene=0)
        st_elm = MockState(24, 5, 4, 3, {"EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON", "EVENT_GAVE_MYSTERY_EGG_TO_ELM"})

        stages = [story_stage(s) for s in (st_start, st_starter, st_egg, st_rival, st_elm)]
        for i in range(len(stages) - 1):
            self.assertGreater(stages[i + 1], stages[i], f"Stage {i+1} ({stages[i+1]}) should be > Stage {i} ({stages[i]})")

    def test_stage_2a_route30_to_rival_gradient(self):
        # Route 30 doorstep at (17, 5) stepping out of Mr. Pokemon house
        egg_events = {"EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}
        st_house_door = MockState(26, 10, 2.5, 7.0, egg_events, cherrygrove_scene=1)
        st_r30_doorstep = MockState(26, 1, 17.0, 5.0, egg_events, cherrygrove_scene=1)
        self.assertGreaterEqual(navigation_potential(st_r30_doorstep), navigation_potential(st_house_door), "No cliff exiting Mr. Pokemon house")

        # Moving south down Route 30 east grass path
        st_r30_mid = MockState(26, 1, 15.0, 25.0, egg_events, cherrygrove_scene=1)
        st_r30_fork = MockState(26, 1, 15.0, 40.0, egg_events, cherrygrove_scene=1)
        st_r30_south = MockState(26, 1, 10.0, 53.0, egg_events, cherrygrove_scene=1)
        self.assertGreater(navigation_potential(st_r30_mid), navigation_potential(st_r30_doorstep))
        self.assertGreater(navigation_potential(st_r30_fork), navigation_potential(st_r30_mid))
        self.assertGreater(navigation_potential(st_r30_south), navigation_potential(st_r30_fork))

        # Reaching Rival encounter at (33, 6..7) in Cherrygrove
        st_cg_north = MockState(26, 3, 29.0, 0.0, egg_events, cherrygrove_scene=1)
        st_cg_rival = MockState(26, 3, 33.0, 6.5, egg_events, cherrygrove_scene=1)
        self.assertGreater(navigation_potential(st_cg_north), navigation_potential(st_r30_south))
        self.assertGreater(navigation_potential(st_cg_rival), navigation_potential(st_cg_north))

    def test_stage_2b_rival_to_elm_gradient(self):
        egg_events = {"EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}
        # Defeated rival in Cherrygrove, moving east towards Route 29
        st_cg_post_rival = MockState(26, 3, 34.0, 6.0, egg_events, cherrygrove_scene=0)
        st_cg_gate = MockState(26, 3, 39.0, 7.0, egg_events, cherrygrove_scene=0)
        self.assertGreater(navigation_potential(st_cg_gate), navigation_potential(st_cg_post_rival))

        # Route 29 Eastbound
        st_r29_west = MockState(24, 3, 0.0, 6.0, egg_events, cherrygrove_scene=0)
        st_r29_mid = MockState(24, 3, 30.0, 6.0, egg_events, cherrygrove_scene=0)
        st_r29_east = MockState(24, 3, 59.0, 9.0, egg_events, cherrygrove_scene=0)
        self.assertGreaterEqual(navigation_potential(st_r29_west), navigation_potential(st_cg_gate))
        self.assertGreater(navigation_potential(st_r29_mid), navigation_potential(st_r29_west))
        self.assertGreater(navigation_potential(st_r29_east), navigation_potential(st_r29_mid))

        # Entering New Bark Town and heading into Elm's Lab
        st_new_bark = MockState(24, 4, 0.0, 9.0, egg_events, cherrygrove_scene=0)
        st_lab_entrance = MockState(24, 4, 6.0, 4.0, egg_events, cherrygrove_scene=0)
        st_lab_inside_door = MockState(24, 5, 4.5, 11.0, egg_events, cherrygrove_scene=0)
        st_lab_elm = MockState(24, 5, 4.5, 2.5, egg_events, cherrygrove_scene=0)

        self.assertGreaterEqual(navigation_potential(st_new_bark), navigation_potential(st_r29_east))
        self.assertGreater(navigation_potential(st_lab_entrance), navigation_potential(st_new_bark))
        self.assertGreater(navigation_potential(st_lab_inside_door), navigation_potential(st_lab_entrance))
        self.assertGreater(navigation_potential(st_lab_elm), navigation_potential(st_lab_inside_door))


if __name__ == "__main__":
    unittest.main()
