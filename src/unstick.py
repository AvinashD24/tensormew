"""Hardcoded escalation for being stuck on one tile, e.g. an NPC standing in a corridor.

The policy only ever presses a direction into the blocker. After a few turns on the same tile this
talks to whatever is in the way (A), then waits for it to move, then walks around it, and finally backs
off a few tiles and tries again.
"""

from __future__ import annotations

from dataclasses import dataclass

_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

TALK_AT = 6        # turns on the same tile before pressing A
WAIT_AT = 10       # then wait for the blocker to walk off
DETOUR_EVERY = 6   # then try a detour every this many turns (alternating sides)
DETOUR_FROM = 14
RETREAT_AT = 38    # then back off and come again
CYCLE = 44


@dataclass
class UnstickResult:
    kind: str
    detail: str = ""
    moved: bool = False


class Unsticker:
    def __init__(self):
        self.pos = None
        self.count = 0
        self.detours = 0

    def note(self, state) -> int:
        """Call once per overworld turn. Returns how many turns in a row were spent on this tile."""
        pos = (state.map_group, state.map_number, state.x, state.y)
        if pos == self.pos:
            self.count += 1
        else:
            self.pos, self.count, self.detours = pos, 0, 0
        return self.count

    @staticmethod
    def _walkable(state, direction: str) -> bool:
        from src.mover import walkable

        return walkable(getattr(state, "tiles", {}).get(direction, 0xFF), direction)

    def act(self, state, heading: str | None, emu, macros, stepper, visit_counts: dict | None = None):
        """Run the escalation step due at the current count, or return None when nothing is due."""
        n = self.count % CYCLE if self.count else 0
        if not heading or n < TALK_AT:
            return None
        visit_counts = visit_counts or {}

        if n == TALK_AT:
            emu.press(heading, frames=6)   # turn to face the blocker without moving
            emu.tick(10, render=False)
            emu.press("a", frames=8)
            emu.tick(30, render=False)
            macros.mash_text(max_frames=200)
            return UnstickResult("talk", f"pressed A facing {heading}")

        if n == WAIT_AT:
            emu.tick(90, render=False)
            return UnstickResult("wait", "waiting for the blocker to move")

        if n >= RETREAT_AT and n == RETREAT_AT:
            back = _OPPOSITE[heading]
            steps = 0
            for _ in range(3):
                if stepper.step(back).moved:
                    steps += 1
                else:
                    break
            emu.tick(60, render=False)
            return UnstickResult("retreat", f"backed off {steps} tile(s)", moved=steps > 0)

        if n >= DETOUR_FROM and (n - DETOUR_FROM) % DETOUR_EVERY == 0:
            perps = ("left", "right") if heading in ("up", "down") else ("up", "down")

            def visits(d):
                dx, dy = _DELTA[d]
                return visit_counts.get((state.map_group, state.map_number, state.x + dx, state.y + dy), 0)

            order = sorted((p for p in perps if self._walkable(state, p)), key=visits)
            if self.detours % 2 and len(order) == 2:
                order.reverse()
            self.detours += 1
            for side in order:
                sidestepped = 0
                for _ in range(2):
                    if stepper.step(side).moved:
                        sidestepped += 1
                    else:
                        break
                if not sidestepped:
                    continue
                ahead = 0
                for _ in range(4):
                    if stepper.step(heading).moved:
                        ahead += 1
                    else:
                        break
                if ahead:
                    return UnstickResult("detour", f"went {side} x{sidestepped}, then {heading} x{ahead}", moved=True)
            return UnstickResult("detour", "no way around", moved=False)
        return None
