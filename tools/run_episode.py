"""Play a named snapshot forward with a button script."""

import argparse
import sys

from src.emulator import Emulator
from src.episode import Action, ButtonScript, EpisodeRunner, FileStateSource, RandomScript
from src.ram_parser import RamParser
from src.states import StateNotFound, StateRegistry


def _buttons(text: str) -> ButtonScript:
    actions = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("wait:"):
            actions.append(Action("wait", frames=int(part.split(":")[1])))
        else:
            actions.append(Action("press", button=part))
    return ButtonScript(actions)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a snapshot forward")
    parser.add_argument("--state", required=True)
    parser.add_argument("--script", choices=("random", "buttons"), default="random")
    parser.add_argument("--buttons", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--gif", default=None)
    args = parser.parse_args(argv)

    registry = StateRegistry()
    try:
        registry.get(args.state)
        source = registry
    except StateNotFound:
        source = FileStateSource()

    script = RandomScript() if args.script == "random" else _buttons(args.buttons)
    emu = Emulator(headless=not args.window, rtc_lock=True)
    observers = []
    recorder = None
    if args.record or args.gif:
        from src.recording import InputRecorder

        recorder = InputRecorder(emu)
        observers.append(recorder)
    try:
        result = EpisodeRunner(emu, RamParser(emu), source, observers=observers).run(
            script, args.state, max_steps=args.steps, timeout_s=args.timeout, seed=args.seed
        )
    finally:
        emu.close()
    print(f"{result.outcome} steps={result.steps} log={result.log_path}")
    if result.error:
        print(result.error)
    if args.gif and recorder is not None:
        from src.recording import FrameDumper, load_inputs, replay

        header, events = load_inputs(recorder.path)
        dumper = FrameDumper(every=2)
        replay_emu = Emulator(headless=True, rtc_lock=True)
        try:
            data = source.load_bytes(args.state)
            replay(
                replay_emu,
                RamParser(replay_emu),
                data,
                events,
                header.get("total_frames") or 0,
                on_frame=dumper,
                render_every=2,
            )
            dumper.save_gif(args.gif)
        finally:
            replay_emu.close()
        print(f"gif={args.gif}")
    return 0 if result.outcome != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
