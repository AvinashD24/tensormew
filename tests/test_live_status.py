import unittest
from types import SimpleNamespace

from src.live_status import checklist_line, story_snapshot
from src.story_flags import EVENT_FLAGS


def _state(*names, species=(155,)):
    flags = bytearray(256)
    for name in names:
        i = EVENT_FLAGS[name]
        flags[i // 8] |= 1 << (i % 8)
    st = SimpleNamespace(event_flags=tuple(flags), party_species=species)
    st.has_event = lambda n: bool(flags[EVENT_FLAGS[n] // 8] & (1 << (EVENT_FLAGS[n] % 8)))
    return st


class LiveStatusTests(unittest.TestCase):
    def test_next_objective_follows_flags(self):
        snap = story_snapshot(_state("EVENT_GOT_A_POKEMON_FROM_ELM"))
        self.assertTrue(snap["done"]["starter"])
        self.assertFalse(snap["done"]["egg from Mr. Pokémon"])
        self.assertIn("Mystery Egg", snap["next"])

    def test_egg_in_party_detected(self):
        snap = story_snapshot(_state("EVENT_GOT_A_POKEMON_FROM_ELM", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON", species=(155, 253)))
        self.assertTrue(snap["egg_in_party"])
        self.assertIn("[x] egg from Mr. Pokémon", checklist_line(snap))
        self.assertIn("Deliver", snap["next"])

    def test_unreadable_flags(self):
        st = SimpleNamespace(event_flags=(), party_species=(), has_event=lambda n: False)
        self.assertEqual(story_snapshot(st)["next"], "event flags unreadable")


class MilestoneTrackerTests(unittest.TestCase):
    def _st(self, group, number, mode=0, hp=((5, 20),), events=()):
        from types import SimpleNamespace

        return SimpleNamespace(
            map_group=group, map_number=number, battle_mode=mode, party_hp=hp,
            event_flags=b"\x01", has_event=lambda name: name in events,
            fainted=bool(hp) and all(h == 0 for h, _ in hp),
        )

    def test_rival_heal_and_gym_sequence(self):
        from src.live_status import MilestoneTracker

        t = MilestoneTracker()
        t.update(self._st(26, 3), 1)
        t.update(self._st(26, 3, mode=2), 2)
        t.update(self._st(26, 3, mode=0), 3)
        t.update(self._st(26, 5, hp=((5, 20),)), 4)
        t.update(self._st(26, 5, hp=((20, 20),)), 5)
        t.update(self._st(10, 7, mode=2), 6)
        t.update(self._st(10, 7, events=("EVENT_BEAT_FALKNER",)), 7)
        for key in ("cherrygrove", "rival_fight", "rival_beaten", "pokecenter", "healed",
                    "gym_entered", "falkner_fight", "falkner_beaten"):
            self.assertIn(key, t.reached)
        self.assertEqual(t.reached["rival_beaten"], 3)
        self.assertEqual(t.heals, 1)
