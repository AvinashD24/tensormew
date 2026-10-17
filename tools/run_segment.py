"""Run one scripted segment from a named snapshot."""

import argparse
import sys

from src.emulator import Emulator
from src.llm_supervisor import Supervisor
from src.ram_parser import RamParser
from src.segment import run_segment
from src.states import StateRegistry
from src.tool_dispatch import execution_registry

_EXIT = {"advanced": 0, "stuck": 2, "budget": 3, "error": 1}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a checkpoint segment")
    parser.add_argument("--state", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--log", default="logs/decisions.jsonl")
    args = parser.parse_args(argv)

    emu = None
    try:
        data = StateRegistry().load_bytes(args.state)
        emu = Emulator(headless=not args.window, rtc_lock=True)
        emu.load_state(data)
        reader = RamParser(emu)
        supervisor = Supervisor(
            client=None,
            registry=execution_registry(emu, reader),
            log_path=None,
        )
        result = run_segment(
            emu,
            reader,
            supervisor,
            max_steps=args.steps,
            decision_log=args.log,
        )
        print(
            f"outcome={result.outcome} steps={result.steps} "
            f"start_map={result.start_map} end_map={result.end_map} "
            f"start_badges={result.start_badges} end_badges={result.end_badges} "
            f"cut_cleared={result.cut_cleared} log_path={result.log_path}"
        )
        return _EXIT.get(result.outcome, 1)
    except Exception as exc:
        print(f"error={exc}")
        return 1
    finally:
        if emu is not None:
            emu.close()


if __name__ == "__main__":
    sys.exit(main())
