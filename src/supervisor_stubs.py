"""Stand-in tool handlers until Dev A wires mover and macros."""

from __future__ import annotations

from src.action_results import ActionResult


def stub_go_to(x: int, y: int) -> ActionResult:
    return ActionResult(ok=True, message=f"stub go_to {x},{y}")


def stub_run_path(directions: list[str]) -> ActionResult:
    return ActionResult(ok=True, message=f"stub run_path {' '.join(directions)}")


def stub_use_cut() -> ActionResult:
    return ActionResult(ok=True, message="stub use_cut")


def stub_mash_text(max_frames: int = 600) -> ActionResult:
    return ActionResult(ok=True, message=f"stub mash_text {max_frames}", frames=max_frames)


def stub_choose_move(slot: int) -> ActionResult:
    return ActionResult(ok=True, message=f"stub choose_move {slot}")


def stub_switch_pokemon(slot: int) -> ActionResult:
    return ActionResult(ok=True, message=f"stub switch_pokemon {slot}")


def stub_use_pokeball() -> ActionResult:
    return ActionResult(ok=True, message="stub use_pokeball")


STUB_REGISTRY: dict = {
    "go_to": stub_go_to,
    "run_path": stub_run_path,
    "use_cut": stub_use_cut,
    "mash_text": stub_mash_text,
    "choose_move": stub_choose_move,
    "switch_pokemon": stub_switch_pokemon,
    "use_pokeball": stub_use_pokeball,
}

