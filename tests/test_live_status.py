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
