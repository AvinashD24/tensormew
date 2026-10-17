"""List, show, validate, and register snapshots."""

import argparse
import json
import sys

from src.emulator import Emulator
from src.ram_parser import RamParser
from src.states import StateRegistry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Named Crystal snapshots")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("name")

    val = sub.add_parser("validate")
    val.add_argument("names", nargs="*")

    cap = sub.add_parser("capture-from")
    cap.add_argument("path")
    cap.add_argument("name")
    cap.add_argument("--desc", default="")
    cap.add_argument("--tags", default="")
    cap.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    registry = StateRegistry()

    if args.cmd == "list":
        names = registry.names()
        if not names:
            print("no states")
        for name in names:
            entry = registry.get(name)
            g, n = entry.map
            x, y = entry.pos
            print(f"{name}  map=({g},{n}) pos=({x},{y}) badges={entry.badges}  {entry.description}")
        print("state files are not in git; share them by hand. manifest.json is committed.")
        return 0

    if args.cmd == "show":
        entry = registry.get(args.name)
        print(json.dumps(entry.to_json(), indent=2))
        emu = Emulator(headless=True, rtc_lock=True)
        try:
            data = registry.load_bytes(args.name)
            emu.load_state(data)
            print(RamParser(emu).parse(emu.frame).to_text())
        finally:
            emu.close()
        return 0

    if args.cmd == "validate":
        if not registry.names() and not args.names:
            print("FAIL empty manifest")
            return 1
        emu = Emulator(headless=True, rtc_lock=True)
        try:
            results = registry.validate(emu, RamParser(emu), args.names or None)
        finally:
            emu.close()
        failed = False
        for result in results:
            status = "PASS" if result.ok else "FAIL"
            extra = "" if result.ok else " " + "; ".join(result.problems)
            print(f"{status} {result.name}{extra}")
            failed = failed or not result.ok
        return 1 if failed else 0

    tags = [t for t in args.tags.split(",") if t]
    emu = Emulator(headless=True, rtc_lock=True)
    try:
        entry = registry.register_file(
            args.path, emu, RamParser(emu), args.name, description=args.desc, tags=tags, force=args.force
        )
    finally:
        emu.close()
    print(json.dumps(entry.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
