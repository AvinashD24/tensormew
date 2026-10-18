import unittest

from src.story_flags import (
    EVENT_FLAGS,
    is_flag_set,
    get_story_summary,
    next_story_milestone,
)


class StoryFlagsTests(unittest.TestCase):
    def test_flag_setting_and_getting(self):
        buf = bytearray(256)
        flag_id = EVENT_FLAGS["EVENT_GOT_A_POKEMON_FROM_ELM"]  # 26
        byte_idx = flag_id // 8
        bit_idx = flag_id % 8
        buf[byte_idx] |= (1 << bit_idx)

        self.assertTrue(is_flag_set(buf, flag_id))
        self.assertFalse(is_flag_set(buf, EVENT_FLAGS["EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"]))

    def test_milestone_progression(self):
        buf = bytearray(256)
        self.assertIn("starter Pokémon", next_story_milestone(buf))

        # Give starter
        f_starter = EVENT_FLAGS["EVENT_GOT_A_POKEMON_FROM_ELM"]
        buf[f_starter // 8] |= (1 << (f_starter % 8))
        self.assertIn("Mr. Pokémon", next_story_milestone(buf))

        # Got mystery egg
        f_egg = EVENT_FLAGS["EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"]
        buf[f_egg // 8] |= (1 << (f_egg % 8))
        self.assertIn("Deliver the Mystery Egg", next_story_milestone(buf))

        # Delivered egg
        f_delivered = EVENT_FLAGS["EVENT_GAVE_MYSTERY_EGG_TO_ELM"]
        buf[f_delivered // 8] |= (1 << (f_delivered % 8))
        self.assertIn("Talk to Mom", next_story_milestone(buf))

        # Talked to mom
        f_mom = EVENT_FLAGS["EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST"]
        buf[f_mom // 8] |= (1 << (f_mom % 8))
        self.assertIn("Falkner", next_story_milestone(buf))

        # Beat Falkner
        f_falkner = EVENT_FLAGS["EVENT_BEAT_FALKNER"]
        buf[f_falkner // 8] |= (1 << (f_falkner % 8))
        self.assertIn("Zephyr Badge earned", next_story_milestone(buf))

    def test_summary_dictionary(self):
        buf = bytearray(256)
        f_starter = EVENT_FLAGS["EVENT_GOT_A_POKEMON_FROM_ELM"]
        buf[f_starter // 8] |= (1 << (f_starter % 8))
        summary = get_story_summary(buf)
        self.assertTrue(summary["EVENT_GOT_A_POKEMON_FROM_ELM"])
        self.assertFalse(summary["EVENT_BEAT_FALKNER"])
