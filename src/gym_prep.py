"""Get the lead Pokémon strong enough for Falkner before walking into his gym.

Falkner (Crystal) leads a Pidgey Lv 7 and a Pidgeotto Lv 9-10, and both use Mud-Slap, which is super
effective on Cyndaquil and lowers accuracy. Guides agree the starter should be about Lv 12+ (Cyndaquil
learns Ember at Lv 12). Without this the scripted route in `falkner_nav` walks in with a Lv 8 lead.

While the lead is under TARGET_LEVEL, after the egg is delivered and before Falkner is beaten, this
module takes over movement on Route 30 and Cherrygrove City:
  * lead HP under HEAL_BELOW -> walk to Cherrygrove's Pokémon Center door (the existing Pokémon Center
    logic in play_live heals inside);
  * otherwise on Route 30 -> walk to the nearest tall grass and pace back and forth to farm wild battles.
Anywhere else it returns None and the normal route continues.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

from src.mover import walkable
from src.return_nav import _DIRS, _Rom

TARGET_LEVEL = 13
HEAL_BELOW = 0.5

ROUTE_30 = (26, 1)
CHERRYGROVE = (26, 3)
PC_DOOR_OUTSIDE = (29, 4)  # standing here, "up" walks onto the Pokémon Center door at (29, 3)
TALL_GRASS = 0x18
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

_last_pace: str | None = None


def lead_level(state) -> int:
    levels = tuple(getattr(state, "party_levels", ()) or ())
    return levels[0] if levels else 0


def lead_hp_fraction(state) -> float:
    hp = tuple(getattr(state, "party_hp", ()) or ())
    if not hp or hp[0][1] <= 0:
        return 1.0
    return hp[0][0] / hp[0][1]


def needs_training(state) -> bool:
    has = getattr(state, "has_event", None)
    if not callable(has):
        return False
    return (
        bool(has("EVENT_GAVE_MYSTERY_EGG_TO_ELM"))
        and not has("EVENT_BEAT_FALKNER")
        and 0 < lead_level(state) < TARGET_LEVEL
    )


class _PrepNav:
    """Walking graph over Route 30 and Cherrygrove City, read from the ROM's maps."""

    def __init__(self):
        rom = _Rom()
        self.cells: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        self.size: dict[tuple[int, int], tuple[int, int]] = {}
        for mid, prefix in ((ROUTE_30, "Route30"), (CHERRYGROVE, "CherrygroveCity")):
            w, h, cells, conns = rom.grid(prefix)
            self.cells[mid], self.size[mid] = cells, (w, h)
            if mid == ROUTE_30:
                self.south_dx = conns["south"][3]   # Route 30 (x, h) -> Cherrygrove (x + dx, 0)
            else:
                self.north_dx = conns["north"][3]   # Cherrygrove (x, -1) -> Route 30 (x + dx, h - 1)
        self.rev: dict[tuple, list] = {}
        for mid, cells in self.cells.items():
            for (x, y), tile in cells.items():
                if not walkable(tile):
                    continue
                for d, (dx, dy) in _DIRS.items():
                    dst = self._step(mid, x + dx, y + dy, d)
                    if dst is not None:
                        self.rev.setdefault(dst, []).append((mid, x, y))
        self.grass = [
            (ROUTE_30, x, y) for (x, y), t in self.cells[ROUTE_30].items() if t == TALL_GRASS
        ]
        self._fields: dict[str, dict] = {}

    def _step(self, mid, nx, ny, direction):
        w, h = self.size[mid]
        if 0 <= nx < w and 0 <= ny < h:
            tmid, tx, ty = mid, nx, ny
        elif mid == ROUTE_30 and direction == "down" and ny >= h:
            tmid, tx, ty = CHERRYGROVE, nx + self.south_dx, 0
        elif mid == CHERRYGROVE and direction == "up" and ny < 0:
            tmid, tx, ty = ROUTE_30, nx + self.north_dx, self.size[ROUTE_30][1] - 1
        else:
            return None
        tw, th = self.size[tmid]
        if not (0 <= tx < tw and 0 <= ty < th):
            return None
        if not walkable(self.cells[tmid][(tx, ty)], direction):
            return None
        return (tmid, tx, ty)

    def _field(self, name: str, goals: list) -> dict:
        if name not in self._fields:
            dist = {g: 0 for g in goals}
            queue = deque(goals)
            while queue:
                node = queue.popleft()
                for prev in self.rev.get(node, ()):
                    if prev not in dist:
                        dist[prev] = dist[node] + 1
                        queue.append(prev)
            self._fields[name] = dist
        return self._fields[name]

    def toward(self, name: str, goals: list, mid, x, y) -> str | None:
        dist = self._field(name, goals)
        here = dist.get((mid, x, y))
        if here is None:
            return None
        best = None
        for d, (dx, dy) in _DIRS.items():
            nxt = self._step(mid, x + dx, y + dy, d)
            if nxt is not None and dist.get(nxt, 1 << 30) < here:
                if best is None or dist[nxt] < best[0]:
                    best = (dist[nxt], d)
        return best[1] if best else None

    def is_grass(self, mid, x, y) -> bool:
        return self.cells.get(mid, {}).get((x, y)) == TALL_GRASS


@lru_cache(maxsize=1)
def _nav() -> _PrepNav:
    return _PrepNav()


def prep_direction(state) -> str | None:
    """Direction to step while training or healing, or None to leave movement to the normal route."""
    global _last_pace
    if not needs_training(state) or getattr(state, "battle_mode", 0):
        return None
    mid = (state.map_group, state.map_number)
    if mid not in (ROUTE_30, CHERRYGROVE):
        return None
    x, y = state.x, state.y
    nav = _nav()

    if lead_hp_fraction(state) < HEAL_BELOW:
        if mid == CHERRYGROVE and (x, y) == PC_DOOR_OUTSIDE:
            return "up"
        return nav.toward("heal", [(CHERRYGROVE,) + PC_DOOR_OUTSIDE], mid, x, y)

    if mid != ROUTE_30:
        return None  # healthy in Cherrygrove: the normal route walks north to Route 30

    if nav.is_grass(mid, x, y):
        # Pace between neighbouring grass tiles: every step in tall grass can start a wild battle.
        options = [
            d for d, (dx, dy) in _DIRS.items()
            if nav.is_grass(mid, x + dx, y + dy) and nav._step(mid, x + dx, y + dy, d) is not None
        ]
        if options:
            pick = _OPPOSITE.get(_last_pace) if _OPPOSITE.get(_last_pace) in options else options[0]
            _last_pace = pick
            return pick
    return nav.toward("grass", nav.grass, mid, x, y)
