"""Train the navigation policy from scratch on live Crystal rewards."""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.emulator import Emulator
from src.macros import Macros
from src.ppo import Policy, collect_rollout, ppo_update
from src.ram_parser import RamParser
from src.rl_env import CrystalEnv
from src.states import StateRegistry
from src.step import TileStepper


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO by playing")
    parser.add_argument("--state", default="new_bark_outside")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="logs/ppo_policy.npz")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    emu = Emulator(headless=True, rtc_lock=True)
    try:
        reader = RamParser(emu)
        start = StateRegistry().load_bytes(args.state)
        env = CrystalEnv(
            emu,
            reader,
            TileStepper(emu, reader),
            Macros(emu, reader),
            rng=np.random.default_rng(args.seed),
            max_steps=args.rollout,
        )
        env.start = start
        policy = Policy(rng=np.random.default_rng(args.seed))
        if args.resume and Path(args.save).exists():
            policy.load(args.save)
            print(f"loaded={args.save}")
        for update in range(args.updates):
            batch = collect_rollout(env, policy, args.rollout)
            delta = ppo_update(policy, batch)
            mean_reward = float(np.mean(batch["rew"]))
            print(
                f"update={update} mean_reward={mean_reward:.3f} "
                f"weight_change={delta:.5f} frontier={len(env.frontier)}"
            )
        policy.save(args.save)
        print(f"saved={args.save}")
        return 0
    finally:
        emu.close()


if __name__ == "__main__":
    sys.exit(main())
