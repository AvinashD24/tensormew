"""Shared tool result. Dev A movers/macros should return this shape."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    message: str
    frames: int = 0
