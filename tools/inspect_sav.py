"""Load a cartridge save or a snapshot and print one line. Does not register it."""

import argparse
import sys
from pathlib import Path

from src.config import RAM_PATH
from src.emulator import Emulator
from src.options import options_text
from src.ram_parser import RamParser
from src.states import StateNotFound, StateRegistry

SRAM = 32768


def report(state) -> str:
    cut = any(0x0F in moves for moves in state.party_moves)
    return (
        f"badges={state.badges} map={state.map_id} "
        f"party={list(zip(state.party_species, state.party_moves))} cut={'yes' if cut else 'NO'}\n"
        f"wOptions {state.options:#04x} {options_text(state.options)}"
    )


def play_until_overworld(emu, reader, max_seconds: float, window: bool):
    emu.tick(1200, render=window)
    emu.press("start", frames=8)
    emu.tick(80, render=window)
    emu.press("a", frames=8)
    budget = int(max_seconds * 60)
    used = 0
    stable = 0
    while used < budget and stable < 30:
        state = reader.parse(emu.frame)
        if state.map_id == (0, 0):
            stable = 0
            if used % 20 == 0:
                emu.press("a", frames=4)
                used += 4
        else:
            stable += 1
        emu.tick(1, render=window)
        used += 1
    emu.tick(30, render=False)
    return reader.parse(emu.frame)


def _with_ram(raw: bytes):
    backup = RAM_PATH.read_bytes() if RAM_PATH.exists() else None
    RAM_PATH.write_bytes(raw[:SRAM])
    return backup


def _restore_ram(backup) -> None:
    if backup is None:
        if RAM_PATH.exists():
            RAM_PATH.unlink()
    else:
        RAM_PATH.write_bytes(backup)


def inspect_cartridge(path: Path, max_seconds: float, window: bool) -> int:
    raw = path.read_bytes()
    if len(raw) < SRAM:
        print(f"save is {len(raw)} bytes; Crystal SRAM is {SRAM}", file=sys.stderr)
        return 1
    if len(raw) > SRAM:
        print(f"warning: truncating {len(raw)} bytes to {SRAM}", file=sys.stderr)
    backup = _with_ram(raw)
    emu = None
    try:
        emu = Emulator(headless=not window, rtc_lock=False)
        state = play_until_overworld(emu, RamParser(emu), max_seconds, window)
        print(report(state))
        if state.map_id == (0, 0):
            print("still on the title screen. rerun with --window", file=sys.stderr)
            return 1
        return 0
    finally:
        if emu is not None:
            emu.close()
        _restore_ram(backup)


def inspect_snapshot(data: bytes) -> int:
    emu = Emulator(headless=True, rtc_lock=True)
    try:
        emu.load_state(data, apply_options=False)
        emu.tick(30, render=False)
        print(report(RamParser(emu).parse(emu.frame)))
        return 0
    finally:
        emu.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Load a .sav, .srm, or .state and print badges, party, map, options")
    parser.add_argument("target", help="file path or registered snapshot name")
    parser.add_argument("--max-seconds", type=float, default=90)
    parser.add_argument("--window", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.target)
    if path.exists():
        if path.suffix.lower() in {".state"}:
            return inspect_snapshot(path.read_bytes())
        return inspect_cartridge(path, args.max_seconds, args.window)
    try:
        data = StateRegistry().load_bytes(args.target)
    except StateNotFound:
        print(f"no file or snapshot named {args.target}", file=sys.stderr)
        return 1
    return inspect_snapshot(data)


if __name__ == "__main__":
    sys.exit(main())
