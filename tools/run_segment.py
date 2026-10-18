"""Run one scripted segment from a named snapshot."""

import argparse
import sys

from src.emulator import Emulator
from src.llm_supervisor import Supervisor
from src.ram_parser import RamParser
from src.run_log import DEFAULT_DB, RunLog
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
    parser.add_argument("--target-map", default="", help="group,number to reach, e.g. 24,3")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--policy", default="", help="path to ppo_policy.npz; walks with PPO instead of search")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="render live visual reward & entropy graph alongside PyBoy (defaults to True if --window and --policy)",
    )
    args = parser.parse_args(argv)
    target_map = None
    if args.target_map:
        group, number = args.target_map.split(",")
        target_map = (int(group), int(number))

    emu = None
    run_log = RunLog(args.db)
    should_plot = bool(args.window) if args.plot is None else bool(args.plot)
    plotter = None
    try:
        data = StateRegistry().load_bytes(args.state)
        emu = Emulator(headless=not args.window, rtc_lock=True)
        emu.load_state(data)
        reader = RamParser(emu)
        policy = None
        if args.policy:
            from src.ppo import load_policy

            policy = load_policy(args.policy)
        if should_plot:
            from src.visual_metrics import LiveMetricsPlotter

            plotter = LiveMetricsPlotter(
                title=f"TensorMew: Segment Run ({args.state})",
                enabled=True,
                headless=not args.window and not args.plot,
                save_path="logs/segment_reward_entropy.png",
            )
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
            target_map=target_map,
            run_log=run_log,
            state_name=args.state,
            policy=policy,
            plotter=plotter,
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
        if plotter is not None:
            saved_plot = plotter.save()
            if saved_plot:
                print(f"saved_plot={saved_plot}")
            plotter.close()
        run_log.close()
        if emu is not None:
            emu.close()


if __name__ == "__main__":
    sys.exit(main())
