"""Count how often a saved policy leaves the map it starts on."""

import argparse
import sys

import numpy as np

from src.emulator import Emulator
from src.macros import Macros
from src.ppo import Policy
from src.ram_parser import RamParser
from src.rl_env import CrystalEnv
from src.states import StateRegistry
from src.step import TileStepper


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Count map exits under a saved policy")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", action="store_true", help="render PyBoy visual window")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="render live visual reward & entropy graph alongside PyBoy (defaults to True if --window)",
    )
    args = parser.parse_args(argv)

    start = StateRegistry().load_bytes(args.state)
    from src.ppo import load_policy

    policy = load_policy(args.policy, rng=np.random.default_rng(args.seed))
    emu = Emulator(headless=not args.window, rtc_lock=True, speed=0 if not args.window else args.speed)
    should_plot = args.window if args.plot is None else bool(args.plot)
    plotter = None
    if should_plot:
        from src.visual_metrics import LiveMetricsPlotter

        plotter = LiveMetricsPlotter(
            title="TensorMew: Policy Evaluation (Exits, Returns & Entropy)",
            enabled=True,
            headless=not args.window and not args.plot,
            save_path="logs/eval_exits_reward_entropy.png",
        )
    try:
        reader = RamParser(emu)
        env = CrystalEnv(
            emu,
            reader,
            TileStepper(emu, reader),
            Macros(emu, reader),
            rng=np.random.default_rng(args.seed),
            frontier_prob=0.0,
            max_steps=args.steps,
        )
        exits = 0
        for _episode in range(args.episodes):
            obs = env.reset(start)
            origin = reader.parse(emu.frame).map_id
            left = False
            for _step in range(args.steps):
                probs = policy.probs(obs)
                entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
                val = policy.value(obs)
                action, _logp = policy.act(obs)
                obs, reward, done, info = env.step(action)
                if plotter is not None:
                    plotter.log_step(
                        reward=reward,
                        entropy=entropy,
                        done=done,
                        action=action,
                        value=val,
                        info=info,
                    )
                if reader.parse(emu.frame).map_id != origin:
                    left = True
                    break
                if done:
                    break
            if left:
                exits += 1
        print(f"episodes={args.episodes} exits={exits} rate={exits}/{args.episodes}")
        return 0
    finally:
        if plotter is not None:
            saved_plot = plotter.save()
            if saved_plot:
                print(f"saved_plot={saved_plot}")
            plotter.close()
        emu.close()


if __name__ == "__main__":
    sys.exit(main())
