"""When to ask the LLM. Pure GameState logic — no button presses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.collision import coll_name, is_cut_tree
from src.ram_parser import dialogue_open, GameState

STUCK_THRESHOLD = 8


class TriggerKind(str, Enum):
    CUT_BLOCKED = "CUT_BLOCKED"
    STUCK = "STUCK"
    BATTLE = "BATTLE"
    DIALOGUE = "DIALOGUE"


@dataclass(frozen=True)
class TriggerEvent:
    kind: TriggerKind
    detail: str

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.detail}"


def _cut_sides(state: GameState) -> list[str]:
    return [side for side, tile in state.tiles.items() if is_cut_tree(tile)]


def evaluate(
    state: GameState,
    *,
    stuck_count: int = 0,
    stuck_threshold: int = STUCK_THRESHOLD,
    mover_idle: bool = True,
) -> TriggerEvent | None:
    """Return the highest-priority trigger, or None if the mover can keep going."""
    if state.battle_mode != 0:
        return TriggerEvent(
            TriggerKind.BATTLE,
            f"battle_mode={state.battle_mode} party_hp={list(state.party_hp)}",
        )

    in_dialogue = dialogue_open(state)
    if in_dialogue and mover_idle:
        return TriggerEvent(
            TriggerKind.DIALOGUE,
            f"script={state.script_running} script_mode={state.script_mode}",
        )

    cuts = _cut_sides(state)
    if cuts:
        names = " ".join(f"{side}={coll_name(state.tiles[side])}" for side in cuts)
        return TriggerEvent(
            TriggerKind.CUT_BLOCKED,
            f"cut tree {names} facing={state.facing} pos=({state.x},{state.y})",
        )

    if in_dialogue:
        return None
    if stuck_count >= stuck_threshold:
        return TriggerEvent(
            TriggerKind.STUCK,
            f"same tile {state.pos} for {stuck_count} steps",
        )
    return None
