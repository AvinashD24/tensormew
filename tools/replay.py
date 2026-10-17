"""Replay a recorded button log from its snapshot."""

import argparse
import sys
from pathlib import Path

from src.config import sha1_bytes
from src.emulator import Emulator
from src.episode import FileStateSource
from src.ram_parser import RamParser
from src.recording import FrameDumper, load_inputs, replay
from src.states import StateNotFound, StateRegistry


def _source(name: str):
    registry = StateRegistry()
    try:
        registry.get(name)
        return registry
    except StateNotFound:
        return FileStateSource()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay recorded Crystal inputs")
    parser.add_argument("path")
    parser.add_argument("--state", default=None)
    parser.add_argument("--gif", default=None)
    parser.add_argument("--mp4", default=None)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    header, events = load_inputs(args.path)
    name = args.state or header.get("state")
    if not name:
        print("no state name in the header; pass --state", file=sys.stderr)
        return 2
    data = _source(str(name)).load_bytes(str(name))
    if header.get("state_sha1") and sha1_bytes(data) != header["state_sha1"]:
        print("state file does not match the recording", file=sys.stderr)
        return 2

    dumper = FrameDumper(every=args.every) if (args.gif or args.mp4) else None
    emu = Emulator(headless=not args.window, rtc_lock=True, speed=1 if args.window else 0)
    try:
        result = replay(
            emu,
            RamParser(emu),
            data,
            events,
            int(header.get("total_frames") or 0),
            expected=header if args.verify else None,
            on_frame=dumper,
            render_every=args.every if dumper else 0,
        )
    finally:
        emu.close()

    if dumper is not None:
        if args.gif:
            dumper.save_gif(args.gif, fps=args.fps)
            print(f"gif={args.gif}")
        if args.mp4:
            try:
                dumper.save_mp4(args.mp4, fps=args.fps)
                print(f"mp4={args.mp4}")
            except RuntimeError as exc:
                gif = str(Path(args.mp4).with_suffix(".gif"))
                dumper.save_gif(gif, fps=args.fps)
                print(f"warning: {exc}; wrote {gif}")
    if args.verify:
        print("MATCH" if result.matches else "MISMATCH " + "; ".join(result.mismatches))
        return 0 if result.matches else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
