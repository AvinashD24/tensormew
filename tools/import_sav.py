"""Turn a cartridge .sav into a named snapshot."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import RAM_PATH
from src.emulator import Emulator
from src.ram_parser import RamParser
from src.states import StateRegistry
from tools.inspect_sav import SRAM, play_until_overworld, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Import a Crystal .sav as a named snapshot")
    parser.add_argument("path", nargs="?")
    parser.add_argument("--name", required=False)
    parser.add_argument("--desc", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--keep-ram", action="store_true")
    args = parser.parse_args(argv)
    if not args.path or not args.name:
        parser.error("PATH.sav and --name are required")

    raw = Path(args.path).read_bytes()
    if len(raw) < SRAM:
        print(f"save is {len(raw)} bytes; Crystal SRAM is {SRAM}", file=sys.stderr)
        return 1
    if len(raw) > SRAM:
        print(f"warning: truncating {len(raw)} bytes to {SRAM}", file=sys.stderr)
        raw = raw[:SRAM]

    backup = None
    existed = RAM_PATH.exists()
    if existed:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = Path(str(RAM_PATH) + f".bak-{stamp}")
        backup.write_bytes(RAM_PATH.read_bytes())
    RAM_PATH.write_bytes(raw)

    emu = None
    try:
        emu = Emulator(headless=not args.window, rtc_lock=False)
        state = play_until_overworld(emu, RamParser(emu), args.max_seconds, bool(args.window))
        print(report(state))
        if not state.ready:
            print("budget exhausted. rerun with --window and finish in tools.make_state", file=sys.stderr)
            print(state.to_text())
            return 1
        before = state.options
        after = emu.apply_training_options()
        print(f"wOptions {before:#04x} -> {after:#04x} text=FAST style=SET scene=OFF")
        tags = [t for t in args.tags.split(",") if t]
        entry = StateRegistry().capture(emu, RamParser(emu), args.name, description=args.desc, tags=tags)
        print(f"registered {entry.name}")
        return 0
    finally:
        if emu is not None:
            emu.close()
        if not args.keep_ram:
            if backup is not None:
                RAM_PATH.write_bytes(backup.read_bytes())
            elif RAM_PATH.exists():
                RAM_PATH.unlink()


if __name__ == "__main__":
    sys.exit(main())
