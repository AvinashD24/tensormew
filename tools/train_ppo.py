"""Train the navigation policy from scratch on live Crystal rewards."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.emulator import Emulator
from src.macros import Macros
from src.ppo import Policy, collect_rollout, ppo_update
from src.ram_parser import RamParser
from src.rl_env import CrystalEnv
from src.states import StateRegistry
from src.step import TileStepper


def _one_rollout(payload):
    """Play one batch in its own emulator. Returns lists plus how many new maps appeared."""
    state_bytes, steps, seed, weights = payload
    emu = Emulator(headless=True, rtc_lock=True, speed=0)
    try:
        reader = RamParser(emu)
        env = CrystalEnv(
            emu,
            reader,
            TileStepper(emu, reader),
            Macros(emu, reader),
            rng=np.random.default_rng(seed),
            max_steps=steps,
        )
        env.start = state_bytes
        policy = Policy(rng=np.random.default_rng(seed))
        policy.w = np.array(weights["w"], copy=True)
        policy.b = np.array(weights["b"], copy=True)
        policy.vw = np.array(weights["vw"], copy=True)
        policy.vb = float(weights["vb"])
        batch = collect_rollout(env, policy, steps)
        return batch, len(env.frontier)
    finally:
        emu.close()


def _merge(parts):
    merged = {"obs": [], "actions": [], "logp": [], "rew": [], "ret": [], "adv": []}
    for batch, _maps in parts:
        for key in merged:
            merged[key].extend(batch[key])
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO by playing")
    parser.add_argument("--state", default="new_bark_outside")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--envs", type=int, default=1, help="parallel emulators per update")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="logs/ppo_policy.npz")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--speed", type=int, default=10)
    args = parser.parse_args(argv)

    emu = Emulator(headless=not args.window, rtc_lock=True, speed=0 if not args.window else args.speed)
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
        weights = lambda: {
            "w": policy.w,
            "b": policy.b,
            "vw": policy.vw,
            "vb": policy.vb,
        }
        pool = ProcessPoolExecutor(max_workers=args.envs) if args.envs > 1 else None
        try:
            for update in range(args.updates):
                payloads = [
                    (start, args.rollout, args.seed + update * args.envs + i, weights())
                    for i in range(args.envs)
                ]
                if args.envs == 1 and not args.window:
                    batch, frontier = _one_rollout(payloads[0])
                    parts = [(batch, frontier)]
                elif args.envs == 1:
                    batch = collect_rollout(env, policy, args.rollout)
                    parts = [(batch, len(env.frontier))]
                else:
                    parts = list(pool.map(_one_rollout, payloads))
                    batch = _merge(parts)
                delta = ppo_update(policy, batch, batch_size=args.batch_size)
                mean_reward = float(np.mean(batch["rew"]))
                frontier = sum(item[1] for item in parts)
                print(
                    f"update={update} samples={len(batch['rew'])} mean_reward={mean_reward:.3f} "
                    f"weight_change={delta:.5f} new_maps={frontier}",
                    flush=True,
                )
        finally:
            if pool is not None:
                pool.shutdown()
        policy.save(args.save)
        print(f"saved={args.save}")
        return 0
    finally:
        emu.close()


if __name__ == "__main__":
    sys.exit(main())
