"""Play in a window and save several named snapshots."""

import argparse
import queue
import sys
import threading
import time

from src.emulator import Emulator
from src.ram_parser import RamParser
from src.states import StateRegistry

HELP = """commands: save NAME [description] | force NAME [description] | pos | list | speed N | quit | help"""


def _stdin(commands: queue.Queue) -> None:
    for line in sys.stdin:
        commands.put(line.rstrip("\n"))
    commands.put("quit")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play Crystal and save named snapshots")
    parser.add_argument("--from", dest="from_name", default=None)
    parser.add_argument("--from-path", default=None)
    parser.add_argument("--autosave-every", type=float, default=60)
    args = parser.parse_args(argv)

    registry = StateRegistry()
    emu = Emulator(headless=False, rtc_lock=False, speed=1)
    reader = RamParser(emu)
    if args.from_name:
        emu.load_state(registry.load_bytes(args.from_name))
    elif args.from_path:
        emu.load_state(args.from_path)

    commands: queue.Queue = queue.Queue()
    threading.Thread(target=_stdin, args=(commands,), daemon=True).start()
    print(HELP)
    saved = []
    last_auto = time.monotonic()
    auto_i = 0
    try:
        while True:
            if not emu.tick(1, render=True):
                break
            try:
                line = commands.get_nowait()
            except queue.Empty:
                line = None
            if line is not None:
                if not _command(line, emu, reader, registry, saved):
                    break
            if emu.frame % 30 == 0:
                _status(reader.parse(emu.frame))
            if args.autosave_every and time.monotonic() - last_auto >= args.autosave_every:
                last_auto = time.monotonic()
                state = reader.parse(emu.frame)
                if state.ready:
                    dest = registry.root / "_autosave" / f"auto_{auto_i % 5}.state"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    emu.save_state(dest)
                    auto_i += 1
    finally:
        emu.close()
    print("saved this session:", ", ".join(saved) or "(none)")
    return 0


def _status(state) -> None:
    reasons = state.not_ready_reasons()
    ready = "READY" if not reasons else "NOT READY: " + ", ".join(reasons)
    g, n, x, y = state.pos
    sys.stdout.write(
        f"\rmap=({g},{n}) pos=({x},{y}) facing={state.facing} badges={state.badges}  {ready}   "
    )
    sys.stdout.flush()


def _command(line, emu, reader, registry, saved) -> bool:
    parts = line.split()
    if not parts:
        return True
    cmd = parts[0]
    if cmd == "quit":
        return False
    if cmd == "help":
        print("\n" + HELP)
        return True
    if cmd == "pos":
        print("\n" + reader.parse(emu.frame).to_text())
        return True
    if cmd == "list":
        for name in registry.names():
            print(name)
        return True
    if cmd == "speed" and len(parts) == 2:
        emu.set_speed(int(parts[1]))
        return True
    if cmd in ("save", "force") and len(parts) >= 2:
        name = parts[1]
        description = " ".join(parts[2:])
        try:
            entry = registry.capture(
                emu, reader, name, description=description, force=(cmd == "force")
            )
        except ValueError as exc:
            print(f"\n{exc}. Wait until READY, or use force.")
            return True
        saved.append(name)
        print(f"\nsaved {entry.name} {entry.map} {entry.pos}")
        return True
    print("\n" + HELP)
    return True


if __name__ == "__main__":
    sys.exit(main())
