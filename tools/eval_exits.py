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
    args = parser.parse_args(argv)

    start = StateRegistry().load_bytes(args.state)
    policy = Policy(rng=np.random.default_rng(args.seed))
    policy.load(args.policy)
    emu = Emulator(headless=True, rtc_lock=True, speed=0)
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
                action, _logp = policy.act(obs)
                obs, _reward, done, _info = env.step(action)
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
        emu.close()


if __name__ == "__main__":
    sys.exit(main())
