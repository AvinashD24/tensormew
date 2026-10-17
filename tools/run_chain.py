"""Explore from a checkpoint. Save a checkpoint each time the map changes."""

import argparse
import sys

from src.chain import run_chain
from src.emulator import Emulator
from src.llm_supervisor import Supervisor
from src.ram_parser import RamParser
from src.run_log import DEFAULT_DB, RunLog
from src.states import StateRegistry
from src.tool_dispatch import execution_registry

_EXIT = {"advanced": 0, "no_exit": 0, "stuck": 2, "budget": 3, "error": 1}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Explore until the map changes, then continue")
    parser.add_argument("--state", required=True)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--log", default="logs/decisions.jsonl")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args(argv)

    emu = None
    run_log = RunLog(args.db)
    registry = StateRegistry()
    try:
        emu = Emulator(headless=not args.window, rtc_lock=True)
        emu.load_state(registry.load_bytes(args.state))
        reader = RamParser(emu)
        supervisor = Supervisor(
            client=None,
            registry=execution_registry(emu, reader),
            log_path=None,
        )
        chain = run_chain(
            emu,
            reader,
            supervisor,
            state_name=args.state,
            hops=args.hops,
            max_steps=args.steps,
            decision_log=args.log,
            run_log=run_log,
            registry=registry,
        )
        print(
            f"stopped={chain.stopped} hops={len(chain.results)} "
            f"checkpoints={','.join(chain.checkpoints) or '-'}"
        )
        for result in chain.results:
            print(
                f"outcome={result.outcome} steps={result.steps} "
                f"map={result.start_map}->{result.end_map}"
            )
        return _EXIT.get(chain.stopped, 1)
    except Exception as exc:
        print(f"error={exc}")
        return 1
    finally:
        run_log.close()
        if emu is not None:
            emu.close()


if __name__ == "__main__":
    sys.exit(main())
