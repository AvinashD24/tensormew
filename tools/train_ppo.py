"""Train the navigation policy from scratch on live Crystal rewards."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.emulator import Emulator
from src.macros import Macros
from src.ppo import MlpPolicy, Policy, collect_rollout, ppo_update
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
        if weights.get("model_type") == "mlp":
            policy = MlpPolicy(rng=np.random.default_rng(seed), hidden_dim=weights.get("hidden_dim", 64))
        else:
            policy = Policy(rng=np.random.default_rng(seed))
        policy.set_weights(weights)
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
    parser.add_argument("--mlp", action="store_true", help="train 2-layer MLP policy instead of linear")
    parser.add_argument("--hidden-dim", type=int, default=64, help="hidden dimension for MLP policy")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="render live visual reward & entropy graph alongside PyBoy (defaults to True if --window)",
    )
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

        use_mlp = args.mlp
        if args.resume and Path(args.save).exists():
            data = np.load(args.save)
            if "w1" in data or str(data.get("model_type", "")) == "mlp":
                use_mlp = True

        if use_mlp:
            policy = MlpPolicy(rng=np.random.default_rng(args.seed), hidden_dim=args.hidden_dim)
        else:
            policy = Policy(rng=np.random.default_rng(args.seed))

        if args.resume and Path(args.save).exists():
            policy.load(args.save)
            print(f"loaded={args.save} (type={policy.__class__.__name__})")

        weights = lambda: policy.get_weights()
        should_plot = args.window if args.plot is None else bool(args.plot)
        plotter = None
        if should_plot:
            from src.visual_metrics import LiveMetricsPlotter

            plotter = LiveMetricsPlotter(
                title=f"TensorMew: PPO Training ({policy.__class__.__name__})",
                enabled=True,
                headless=not args.window and not args.plot,
                save_path="logs/training_reward_entropy.png",
            )

        pool = ProcessPoolExecutor(max_workers=args.envs) if args.envs > 1 else None
        try:
            for update in range(args.updates):
                payloads = [
                    (start, args.rollout, args.seed + update * args.envs + i, weights())
                    for i in range(args.envs)
                ]
                if args.envs == 1 and not args.window and not args.plot:
                    batch, frontier = _one_rollout(payloads[0])
                    parts = [(batch, frontier)]
                    if plotter is not None:
                        for s_idx in range(len(batch["rew"])):
                            obs = batch["obs"][s_idx]
                            probs = policy.probs(obs)
                            ent = -float(np.sum(probs * np.log(probs + 1e-8)))
                            plotter.log_step(
                                reward=batch["rew"][s_idx],
                                entropy=ent,
                                action=batch["actions"][s_idx],
                            )
                elif args.envs == 1:
                    def _on_step(data: dict) -> None:
                        if plotter is not None:
                            plotter.log_step(
                                reward=data["reward"],
                                entropy=data["entropy"],
                                done=data["done"],
                                action=data["action"],
                                value=data["value"],
                                info=data["info"],
                            )

                    batch = collect_rollout(env, policy, args.rollout, on_step=_on_step)
                    parts = [(batch, len(env.frontier))]
                else:
                    parts = list(pool.map(_one_rollout, payloads))
                    batch = _merge(parts)
                    if plotter is not None:
                        for s_idx in range(len(batch["rew"])):
                            obs = batch["obs"][s_idx]
                            probs = policy.probs(obs)
                            ent = -float(np.sum(probs * np.log(probs + 1e-8)))
                            plotter.log_step(
                                reward=batch["rew"][s_idx],
                                entropy=ent,
                                action=batch["actions"][s_idx],
                            )
                delta = ppo_update(policy, batch, batch_size=args.batch_size)
                mean_reward = float(np.mean(batch["rew"]))
                frontier = sum(item[1] for item in parts)
                if plotter is not None:
                    recent_entropy = float(
                        np.mean(
                            [
                                -np.sum(policy.probs(o) * np.log(policy.probs(o) + 1e-8))
                                for o in batch["obs"][-min(len(batch["obs"]), args.batch_size) :]
                            ]
                        )
                    )
                    plotter.log_update(
                        update=update,
                        mean_reward=mean_reward,
                        mean_entropy=recent_entropy,
                        weight_change=delta,
                        new_maps=frontier,
                    )
                print(
                    f"update={update} samples={len(batch['rew'])} mean_reward={mean_reward:.3f} "
                    f"weight_change={delta:.5f} new_maps={frontier}",
                    flush=True,
                )
        finally:
            if pool is not None:
                pool.shutdown()
            if plotter is not None:
                saved_plot = plotter.save()
                if saved_plot:
                    print(f"saved_plot={saved_plot}")
                plotter.close()
        policy.save(args.save)
        print(f"saved={args.save} (type={policy.__class__.__name__})")
        return 0
    finally:
        emu.close()


if __name__ == "__main__":
    sys.exit(main())
