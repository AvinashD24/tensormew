"""Read Crystal RAM into a frozen snapshot. Names only; no addresses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from src.collision import coll_name
from src.options import options_text

# wTextboxFlags bit 0 stays set for fast text. An open box is the script engine.
def dialogue_open(state: "GameState") -> bool:
    return state.script_mode != 0 or state.script_running != 0


TEXT_GATE: Callable[["GameState"], bool] = dialogue_open

_FACING = {8: "down", 4: "up", 2: "left", 1: "right"}
_TIME = {0: "morning", 1: "day", 2: "night"}


class MemoryBus(Protocol):
    def symbol(self, name: str) -> tuple[int, int]: ...
    def read(self, bank: int, addr: int, n: int = 1) -> bytes: ...


class SymbolError(KeyError):
    pass


@dataclass(frozen=True)
class Sym:
    name: str
    size: int = 1
    required: bool = True


def _party_hp() -> tuple[Sym, ...]:
    names = []
    for i in range(1, 7):
        names.append(Sym(f"wPartyMon{i}HP", size=2))
        names.append(Sym(f"wPartyMon{i}MaxHP", size=2))
        names.append(Sym(f"wPartyMon{i}Moves", size=4))
        names.append(Sym(f"wPartyMon{i}Level"))
    return tuple(names)


SYMBOLS: tuple[Sym, ...] = (
    Sym("wMapGroup"),
    Sym("wMapNumber"),
    Sym("wXCoord"),
    Sym("wYCoord"),
    Sym("wFacingDirection"),
    Sym("wPlayerDirection"),
    Sym("wTileUp"),
    Sym("wTileDown"),
    Sym("wTileLeft"),
    Sym("wTileRight"),
    Sym("wPlayerTileCollision"),
    Sym("wBattleMode"),
    Sym("wScriptRunning"),
    Sym("wScriptMode"),
    Sym("wTextboxFlags"),
    Sym("wMapStatus"),
    Sym("wPartyCount"),
    Sym("wPartySpecies", size=6),
    *_party_hp(),
    Sym("wJohtoBadges"),
    Sym("wKantoBadges"),
    Sym("wNumItems"),
    Sym("wNumKeyItems"),
    Sym("wNumBalls"),
    Sym("wTimeOfDay"),
    Sym("wOptions"),
    Sym("wMoney", size=3),
    Sym("wTMsHMs", size=57),
)


@dataclass(frozen=True)
class GameState:
    map_group: int
    map_number: int
    x: int
    y: int
    facing: str
    facing_raw: int
    player_direction: int
    tiles: dict[str, int]
    player_tile: int
    battle_mode: int
    script_running: int
    textbox_flags: int
    map_status: int
    party_count: int
    party_species: tuple[int, ...]
    party_hp: tuple[tuple[int, int], ...]
    badge_bits: int
    kanto_badge_bits: int
    num_items: int
    num_key_items: int
    num_balls: int
    time_of_day: int
    frame: int = 0
    party_levels: tuple[int, ...] = ()
    party_moves: tuple[tuple[int, ...], ...] = ()
    options: int = 0
    money: int = 0
    hm_counts: tuple[int, ...] = ()
    script_mode: int = 0

    @property
    def map_id(self) -> tuple[int, int]:
        return (self.map_group, self.map_number)

    @property
    def pos(self) -> tuple[int, int, int, int]:
        return (self.map_group, self.map_number, self.x, self.y)

    @property
    def badges(self) -> int:
        return bin(self.badge_bits).count("1")

    @property
    def in_battle(self) -> bool:
        return self.battle_mode != 0

    @property
    def in_script(self) -> bool:
        return self.script_running != 0

    @property
    def in_dialogue(self) -> bool:
        return TEXT_GATE(self)

    @property
    def ready(self) -> bool:
        return not self.not_ready_reasons()

    @property
    def fainted(self) -> bool:
        return self.party_count > 0 and all(hp == 0 for hp, _ in self.party_hp)

    def not_ready_reasons(self) -> list[str]:
        reasons = []
        if (self.map_group, self.map_number) == (0, 0):
            reasons.append("map=(0,0)")
        if self.in_battle:
            reasons.append("battle")
        if self.in_script:
            reasons.append("script")
        elif self.in_dialogue:
            reasons.append("dialogue")
        return reasons

    def to_dict(self) -> dict:
        return {
            "map_group": self.map_group,
            "map_number": self.map_number,
            "x": self.x,
            "y": self.y,
            "facing": self.facing,
            "facing_raw": self.facing_raw,
            "player_direction": self.player_direction,
            "tiles": dict(self.tiles),
            "player_tile": self.player_tile,
            "battle_mode": self.battle_mode,
            "script_running": self.script_running,
            "textbox_flags": self.textbox_flags,
            "map_status": self.map_status,
            "party_count": self.party_count,
            "party_species": list(self.party_species),
            "party_hp": [list(pair) for pair in self.party_hp],
            "badge_bits": self.badge_bits,
            "kanto_badge_bits": self.kanto_badge_bits,
            "badges": self.badges,
            "num_items": self.num_items,
            "num_key_items": self.num_key_items,
            "num_balls": self.num_balls,
            "time_of_day": self.time_of_day,
            "ready": self.ready,
            "frame": self.frame,
            "party_levels": list(self.party_levels),
            "party_moves": [list(m) for m in self.party_moves],
            "options": self.options,
            "money": self.money,
            "hm_counts": list(self.hm_counts),
            "script_mode": self.script_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        tiles = d.get("tiles") or {"up": 0, "down": 0, "left": 0, "right": 0}
        hp = tuple(tuple(pair) for pair in d.get("party_hp") or [])
        species = tuple(d.get("party_species") or [])
        return cls(
            map_group=int(d.get("map_group", 0)),
            map_number=int(d.get("map_number", 0)),
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            facing=str(d.get("facing", "?")),
            facing_raw=int(d.get("facing_raw", 0)),
            player_direction=int(d.get("player_direction", 0)),
            tiles={k: int(tiles[k]) for k in ("up", "down", "left", "right")},
            player_tile=int(d.get("player_tile", 0)),
            battle_mode=int(d.get("battle_mode", 0)),
            script_running=int(d.get("script_running", 0)),
            textbox_flags=int(d.get("textbox_flags", 0)),
            map_status=int(d.get("map_status", 0)),
            party_count=int(d.get("party_count", 0)),
            party_species=species,
            party_hp=hp,
            badge_bits=int(d.get("badge_bits", 0)),
            kanto_badge_bits=int(d.get("kanto_badge_bits", 0)),
            num_items=int(d.get("num_items", 0)),
            num_key_items=int(d.get("num_key_items", 0)),
            num_balls=int(d.get("num_balls", 0)),
            time_of_day=int(d.get("time_of_day", 0)),
            frame=int(d.get("frame", 0)),
            party_levels=tuple(d.get("party_levels") or []),
            party_moves=tuple(tuple(m) for m in (d.get("party_moves") or [])),
            options=int(d.get("options", 0)),
            money=int(d.get("money", 0)),
            hm_counts=tuple(d.get("hm_counts") or []),
            script_mode=int(d.get("script_mode", 0)),
        )

    def to_text(self) -> str:
        tiles = " ".join(f"{side}={coll_name(self.tiles[side])}" for side in ("up", "down", "left", "right"))
        mode = "overworld" if self.battle_mode == 0 else f"battle({self.battle_mode})"
        hp = ", ".join(f"{cur}/{mx}" for cur, mx in self.party_hp)
        when = _TIME.get(self.time_of_day, str(self.time_of_day))
        return "\n".join(
            [
                f"map=({self.map_group},{self.map_number}) pos=({self.x},{self.y}) facing={self.facing}",
                f"adjacent: {tiles} under={coll_name(self.player_tile)}",
                f"mode={mode} script={self.script_running} script_mode={self.script_mode} text={self.textbox_flags}",
                f"party={self.party_count} hp=[{hp}]",
                f"badges={self.badges} bag: items={self.num_items} key={self.num_key_items} balls={self.num_balls}",
                f"time={when} {options_text(self.options)}",
            ]
        )


class RamParser:
    def __init__(self, bus: MemoryBus, symbols: tuple[Sym, ...] = SYMBOLS):
        missing = []
        resolved: dict[str, tuple[int, int]] = {}
        for sym in symbols:
            try:
                resolved[sym.name] = bus.symbol(sym.name)
            except Exception:
                if sym.required:
                    missing.append(sym.name)
        if missing:
            raise SymbolError("missing symbols: " + ", ".join(missing))
        self.bus = bus
        self.symbols = symbols
        self.resolved = resolved

    def read_raw(self) -> dict[str, int | tuple[int, ...]]:
        out: dict[str, int | tuple[int, ...]] = {}
        for sym in self.symbols:
            if sym.name not in self.resolved:
                continue
            bank, addr = self.resolved[sym.name]
            raw = self.bus.read(bank, addr, sym.size)
            if sym.size == 1:
                out[sym.name] = raw[0]
            elif sym.size == 2:
                out[sym.name] = int.from_bytes(raw, "big")
            else:
                out[sym.name] = tuple(raw)
        return out

    def parse(self, frame: int = 0) -> GameState:
        raw = self.read_raw()
        count = max(0, min(6, int(raw["wPartyCount"])))
        hp = []
        for i in range(1, count + 1):
            hp.append((int(raw[f"wPartyMon{i}HP"]), int(raw[f"wPartyMon{i}MaxHP"])))
        species_all = raw["wPartySpecies"]
        if not isinstance(species_all, tuple):
            species_all = (int(species_all),)
        levels = tuple(int(raw[f"wPartyMon{i}Level"]) for i in range(1, count + 1))
        moves = tuple(tuple(int(b) for b in raw[f"wPartyMon{i}Moves"]) for i in range(1, count + 1))
        money_bytes = raw["wMoney"]
        if not isinstance(money_bytes, tuple):
            money_bytes = (int(money_bytes),)
        money_text = "".join(f"{b:02x}" for b in money_bytes)
        money = int(money_text) if money_text.isdigit() else 0
        tmhm = raw["wTMsHMs"]
        if not isinstance(tmhm, tuple):
            tmhm = (int(tmhm),)
        hm_counts = tuple(int(b) for b in tmhm[50:57])
        facing_raw = int(raw["wFacingDirection"])
        return GameState(
            map_group=int(raw["wMapGroup"]),
            map_number=int(raw["wMapNumber"]),
            x=int(raw["wXCoord"]),
            y=int(raw["wYCoord"]),
            facing=_FACING.get(facing_raw, f"?{facing_raw:#04x}"),
            facing_raw=facing_raw,
            player_direction=int(raw["wPlayerDirection"]),
            tiles={
                "up": int(raw["wTileUp"]),
                "down": int(raw["wTileDown"]),
                "left": int(raw["wTileLeft"]),
                "right": int(raw["wTileRight"]),
            },
            player_tile=int(raw["wPlayerTileCollision"]),
            battle_mode=int(raw["wBattleMode"]),
            script_running=int(raw["wScriptRunning"]),
            textbox_flags=int(raw["wTextboxFlags"]),
            script_mode=int(raw["wScriptMode"]),
            map_status=int(raw["wMapStatus"]),
            party_count=count,
            party_species=tuple(species_all[:count]),
            party_hp=tuple(hp),
            badge_bits=int(raw["wJohtoBadges"]),
            kanto_badge_bits=int(raw["wKantoBadges"]),
            num_items=int(raw["wNumItems"]),
            num_key_items=int(raw["wNumKeyItems"]),
            num_balls=int(raw["wNumBalls"]),
            time_of_day=int(raw["wTimeOfDay"]),
            frame=frame,
            party_levels=levels,
            party_moves=moves,
            options=int(raw["wOptions"]),
            money=money,
            hm_counts=hm_counts,
        )
