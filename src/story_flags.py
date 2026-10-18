"""Event flags and map scene scripts for Pokémon Crystal story progression."""

from __future__ import annotations

# Bank and address of event flags in Pokémon Crystal (US 1.1)
EVENT_FLAGS_BANK = 1
EVENT_FLAGS_ADDR = 0xDA72
EVENT_FLAGS_BYTES = 256  # 2048 flags

# Key Johto event flags from pret/pokecrystal constants/event_flags.asm
EVENT_FLAGS = {
    # Starters & Elm
    "EVENT_GOT_A_POKEMON_FROM_ELM": 26,
    "EVENT_GOT_CYNDAQUIL_FROM_ELM": 27,
    "EVENT_GOT_TOTODILE_FROM_ELM": 28,
    "EVENT_GOT_CHIKORITA_FROM_ELM": 29,
    "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON": 30,
    "EVENT_GAVE_MYSTERY_EGG_TO_ELM": 31,
    "EVENT_ELM_CALLED_ABOUT_STOLEN_POKEMON": 67,
    "EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST": 64,
    "EVENT_REFUSED_TO_TAKE_EGG_FROM_ELMS_AIDE": 44,
    "EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE": 45,
    "EVENT_SHOWED_TOGEPI_TO_ELM": 85,
    "EVENT_GOT_EVERSTONE_FROM_ELM": 86,
    "EVENT_GOT_SS_TICKET_FROM_ELM": 36,
    "EVENT_GOT_MASTER_BALL_FROM_ELM": 125,

    # Route 29 / Cherrygrove
    "EVENT_DUDE_TALKED_TO_YOU": 65,
    "EVENT_LEARNED_TO_CATCH_POKEMON": 66,
    "EVENT_RIVAL_CHERRYGROVE_CITY": 1726,
    "EVENT_GOT_MYSTIC_WATER_IN_CHERRYGROVE": 77,

    # Route 30 / Mr. Pokémon
    "EVENT_GOT_BERRY_FROM_ROUTE_30_HOUSE": 39,

    # HMs & TMs
    "EVENT_GOT_HM01_CUT": 14,
    "EVENT_GOT_HM02_FLY": 15,
    "EVENT_GOT_HM03_SURF": 16,
    "EVENT_GOT_HM04_STRENGTH": 17,
    "EVENT_GOT_HM05_FLASH": 18,
    "EVENT_GOT_HM06_WHIRLPOOL": 19,
    "EVENT_GOT_TM31_MUD_SLAP": 6,

    # Gym Badges & Bosses
    "EVENT_BEAT_FALKNER": 1213,
    "EVENT_BEAT_BUGSY": 1214,
    "EVENT_BEAT_WHITNEY": 1215,
    "EVENT_BEAT_MORTY": 1216,
    "EVENT_BEAT_CHUCK": 1217,
    "EVENT_BEAT_JASMINE": 1218,
    "EVENT_BEAT_PRYCE": 1219,
    "EVENT_BEAT_CLAIR": 1220,
}

REVERSE_EVENT_FLAGS = {v: k for k, v in EVENT_FLAGS.items()}


def is_flag_set(flags_bytes: bytes | bytearray | list[int], flag_id: int) -> bool:
    """Check if a specific event flag is set in the 256-byte wEventFlags buffer."""
    if not (0 <= flag_id < EVENT_FLAGS_BYTES * 8):
        raise ValueError(f"flag_id {flag_id} out of range [0, 2047]")
    byte_index = flag_id // 8
    if byte_index >= len(flags_bytes):
        return False
    bit_index = flag_id % 8
    return bool(flags_bytes[byte_index] & (1 << bit_index))


def get_story_summary(flags_bytes: bytes | bytearray | list[int]) -> dict[str, bool]:
    """Return dictionary of recognized story flag statuses."""
    return {
        name: is_flag_set(flags_bytes, flag_id)
        for name, flag_id in EVENT_FLAGS.items()
    }


def next_story_milestone(flags_bytes: bytes | bytearray | list[int]) -> str:
    """Return a human-readable string describing the current story objective."""
    if not is_flag_set(flags_bytes, EVENT_FLAGS["EVENT_GOT_A_POKEMON_FROM_ELM"]):
        return "Get starter Pokémon from Professor Elm in New Bark Town lab."
    if not is_flag_set(flags_bytes, EVENT_FLAGS["EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"]):
        return "Travel through Route 29 & 30 to Mr. Pokémon's house to get Mystery Egg."
    if not is_flag_set(flags_bytes, EVENT_FLAGS["EVENT_GAVE_MYSTERY_EGG_TO_ELM"]):
        return "Deliver the Mystery Egg back to Professor Elm in New Bark Town."
    if not is_flag_set(flags_bytes, EVENT_FLAGS["EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST"]):
        return "Talk to Mom in New Bark Town before departing."
    if not is_flag_set(flags_bytes, EVENT_FLAGS["EVENT_BEAT_FALKNER"]):
        return "Head west past Cherrygrove through Route 30/31 to Violet City Gym and defeat Falkner."
    return "Zephyr Badge earned. Continue toward Azalea Town."
