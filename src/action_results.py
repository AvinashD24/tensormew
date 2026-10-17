"""Shared result objects for Dev A execution and Dev B supervisor dispatch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionResult:
    ok: bool
    message: str
    frames: int = 0


@dataclass
class MoveResult(ActionResult):
    pos: tuple[int, int, int, int] | None = None
    stuck: int = 0
    steps: int = 0


@dataclass
class PathResult(ActionResult):
    steps_done: int = 0
    pos: tuple[int, int, int, int] | None = None


@dataclass
class MacroResult(ActionResult):
    pass
