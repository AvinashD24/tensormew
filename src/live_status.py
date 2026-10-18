"""Story progress snapshot for live runs: which milestones are done and what comes next."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.game_data import get_map_name, get_species_name
from src.story_flags import EVENT_FLAGS, next_story_milestone

EGG_SPECIES = 253  # party slot species id of an unhatched egg
STATUS_DIR = Path("logs") / "live_status"

# Checklist order is the quest order up to the first badge.
CHECKLIST = (
    ("starter", "EVENT_GOT_A_POKEMON_FROM_ELM"),
    ("egg from Mr. Pokémon", "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"),
    ("egg given to Elm", "EVENT_GAVE_MYSTERY_EGG_TO_ELM"),
    ("talked to Mom", "EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST"),
    ("Falkner beaten", "EVENT_BEAT_FALKNER"),
)


RIVAL_MAP = (26, 3)  # Cherrygrove: the forced rival trainer fight


class RivalTracker:
    """Cherrygrove rival fight: `fighting` while the trainer battle is on, `beaten` once it is won.

    EVENT_RIVAL_CHERRYGROVE_CITY is set in every post-starter save, so it cannot tell these apart.
    """

    def __init__(self):
        self.fighting = False
        self.beaten = False

    def update(self, state) -> None:
        """Feed every parsed state in order; a won fight is seen as the battle ending where it began."""
        on_map = (getattr(state, "map_group", 0), getattr(state, "map_number", 0)) == RIVAL_MAP
        mode = getattr(state, "battle_mode", 0)
        was_fighting = self.fighting
        self.fighting = on_map and mode == 2
        if was_fighting and mode == 0 and not getattr(state, "fainted", False):
            self.beaten = True

    @property
    def label(self) -> str:
        return "beaten" if self.beaten else ("FIGHT" if self.fighting else "-")


def seed_seen_events(state, seen_events: set) -> None:
    """Mark flags already set at start as seen so they are not logged as fresh milestone rewards."""
    for i, b in enumerate(getattr(state, "event_flags", None) or ()):
        for bit in range(8):
            if b & (1 << bit):
                seen_events.add(i * 8 + bit)


def story_snapshot(state, rival: str = "-") -> dict:
    flags = getattr(state, "event_flags", ()) or ()
    done = {label: bool(flags) and state.has_event(name) for label, name in CHECKLIST}
    species = tuple(getattr(state, "party_species", ()) or ())
    return {
        "done": done,
        "rival": rival,
        "egg_in_party": EGG_SPECIES in species,
        "next": next_story_milestone(flags) if flags else "event flags unreadable",
        "known_flags": sorted(n for n, i in EVENT_FLAGS.items() if flags and state.has_event(n)),
    }


def checklist_line(snap: dict) -> str:
    marks = " ".join(f"[{'x' if ok else ' '}] {label}" for label, ok in snap["done"].items())
    egg = " (egg in party)" if snap["egg_in_party"] else ""
    return f"{marks} [rival: {snap.get('rival', '-')}]{egg} | NEXT: {snap['next']}"


def write_status(tag: str, state, snap: dict, step: int, cumulative_reward: float) -> None:
    """One small JSON per run so tools.live_status can show every window at once."""
    lead = ""
    species = getattr(state, "party_species", ()) or ()
    if species:
        lead = get_species_name(species[0])
    payload = {
        "tag": tag,
        "pid": os.getpid(),
        "updated": time.time(),
        "step": step,
        "map": get_map_name(state.map_group, state.map_number),
        "pos": [state.x, state.y],
        "return": round(cumulative_reward, 2),
        "party": getattr(state, "party_count", 0),
        "lead": lead,
        "badges": getattr(state, "badges", 0),
        **snap,
    }
    try:
        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_DIR / f".{tag}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(STATUS_DIR / f"{tag}.json")
    except OSError:
        pass
