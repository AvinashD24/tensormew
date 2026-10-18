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
POKECENTER_MAPS = {(26, 5): "Cherrygrove", (10, 10): "Violet"}

# Quest order from the first step to the Zephyr Badge: (key, label)
MILESTONES = (
    ("starter", "got starter from Elm"),
    ("cherrygrove", "reached Cherrygrove City"),
    ("pokecenter", "entered a Poké Center"),
    ("healed", "healed at a Poké Center"),
    ("rival_fight", "rival fight started"),
    ("rival_beaten", "rival beaten"),
    ("mr_pokemon", "entered Mr. Pokémon's house"),
    ("got_egg", "got Mystery Egg"),
    ("gave_egg", "gave egg to Elm"),
    ("talked_mom", "talked to Mom"),
    ("violet", "reached Violet City"),
    ("gym_entered", "entered Violet Gym"),
    ("falkner_fight", "Falkner battle started"),
    ("falkner_beaten", "Falkner beaten (Zephyr Badge)"),
)
MILESTONE_LABELS = dict(MILESTONES)


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


class MilestoneTracker:
    """Records the step each quest milestone was first reached, plus heal and blackout counts.

    Feed it every parsed state in order. `update` returns the milestones newly reached by that state.
    """

    def __init__(self):
        self.reached: dict[str, int] = {}
        self.rival = RivalTracker()
        self.heals = 0
        self.blackouts = 0
        self._prev = None

    def _mark(self, key: str, step: int, new: list) -> None:
        if key not in self.reached:
            self.reached[key] = step
            new.append(key)

    def update(self, state, step: int) -> list[str]:
        new: list[str] = []
        self.rival.update(state)
        map_id = (getattr(state, "map_group", 0), getattr(state, "map_number", 0))
        mode = getattr(state, "battle_mode", 0)
        has = getattr(state, "has_event", None)

        def ev(name: str) -> bool:
            return bool(getattr(state, "event_flags", None)) and callable(has) and has(name)

        if ev("EVENT_GOT_A_POKEMON_FROM_ELM"):
            self._mark("starter", step, new)
        if map_id == (26, 3):
            self._mark("cherrygrove", step, new)
        if map_id in POKECENTER_MAPS:
            self._mark("pokecenter", step, new)
        if self.rival.fighting:
            self._mark("rival_fight", step, new)
        if self.rival.beaten:
            self._mark("rival_beaten", step, new)
        if map_id == (26, 10):
            self._mark("mr_pokemon", step, new)
        if ev("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"):
            self._mark("got_egg", step, new)
        if ev("EVENT_GAVE_MYSTERY_EGG_TO_ELM"):
            self._mark("gave_egg", step, new)
        if ev("EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST"):
            self._mark("talked_mom", step, new)
        if map_id[0] == 10 and map_id[1] in (5, 7, 10):
            self._mark("violet", step, new)
        if map_id == (10, 7):
            self._mark("gym_entered", step, new)
            if mode == 2:
                self._mark("falkner_fight", step, new)
        if ev("EVENT_BEAT_FALKNER"):
            self._mark("falkner_beaten", step, new)

        prev = self._prev
        if prev is not None:
            party = getattr(state, "party_hp", ()) or ()
            hp_now = sum(h for h, _ in party)
            hp_max = sum(m for _, m in party)
            hp_prev = sum(h for h, _ in getattr(prev, "party_hp", ()) or ())
            if map_id in POKECENTER_MAPS and mode == 0 and hp_max > 0 and hp_now == hp_max and hp_prev < hp_now:
                self.heals += 1
                self._mark("healed", step, new)
            if getattr(state, "fainted", False) and not getattr(prev, "fainted", False) and mode == 0:
                self.blackouts += 1
        self._prev = state
        return new

    def summary(self) -> dict:
        return {
            "rival": self.rival.label,
            "milestones": [[k, st] for k, st in self.reached.items()],
            "heals": self.heals,
            "blackouts": self.blackouts,
        }


def seed_seen_events(state, seen_events: set) -> None:
    """Mark flags already set at start as seen so they are not logged as fresh milestone rewards."""
    for i, b in enumerate(getattr(state, "event_flags", None) or ()):
        for bit in range(8):
            if b & (1 << bit):
                seen_events.add(i * 8 + bit)


def milestone_line(key: str, step: int, tracker: MilestoneTracker) -> str:
    return f"[Milestone {len(tracker.reached)}/{len(MILESTONES)}] step {step}: {MILESTONE_LABELS[key]}"


def story_snapshot(state, tracker: "MilestoneTracker | None" = None) -> dict:
    flags = getattr(state, "event_flags", ()) or ()
    done = {label: bool(flags) and state.has_event(name) for label, name in CHECKLIST}
    species = tuple(getattr(state, "party_species", ()) or ())
    return {
        "done": done,
        **(tracker.summary() if tracker else {"rival": "-", "milestones": [], "heals": 0, "blackouts": 0}),
        "egg_in_party": EGG_SPECIES in species,
        "next": next_story_milestone(flags) if flags else "event flags unreadable",
        "known_flags": sorted(n for n, i in EVENT_FLAGS.items() if flags and state.has_event(n)),
    }


def checklist_line(snap: dict) -> str:
    marks = " ".join(f"[{'x' if ok else ' '}] {label}" for label, ok in snap["done"].items())
    egg = " (egg in party)" if snap["egg_in_party"] else ""
    extra = (
        f" [rival: {snap.get('rival', '-')} | heals: {snap.get('heals', 0)}"
        f" | blackouts: {snap.get('blackouts', 0)} | milestones: {len(snap.get('milestones', []))}/{len(MILESTONES)}]"
    )
    return f"{marks}{extra}{egg} | NEXT: {snap['next']}"


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
