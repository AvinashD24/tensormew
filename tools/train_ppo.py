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
    state_bytes, steps, seed, weights, *rest = payload
    shaping = rest[0] if rest else "frontier"
    ignore_maps = rest[1] if len(rest) > 1 else frozenset()
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
            shaping=shaping,
            ignore_maps=ignore_maps,
        )
        env.start = state_bytes
        if weights.get("model_type") == "mlp":
            policy = MlpPolicy(rng=np.random.default_rng(seed), hidden_dim=weights.get("hidden_dim", 64))
        else:
            policy = Policy(rng=np.random.default_rng(seed))
        policy.set_weights(weights)
        meta = []
        batch = collect_rollout(
            env,
            policy,
            steps,
            on_step=lambda d: meta.append((d["map_id"], d["badges"], d["info"])),
        )
        batch["meta"] = meta  # lets the parent feed map/badge/discovery data to the live plot
        return batch, len(env.frontier)
    finally:
        emu.close()


def _demo(start, steps, seed, policy, speed, shaping, ignore_maps):
    """Play one episode of the current policy in a visible PyBoy window, then close it."""
    emu = Emulator(headless=False, rtc_lock=True, speed=speed)
    try:
        reader = RamParser(emu)
        env = CrystalEnv(
            emu,
            reader,
            TileStepper(emu, reader),
            Macros(emu, reader),
            rng=np.random.default_rng(seed),
            max_steps=steps,
            shaping=shaping,
            ignore_maps=ignore_maps,
        )
        env.start = start
        collect_rollout(env, policy, steps)
    finally:
        emu.close()


def _meta_kwargs(batch, idx):
    """Map, badge and discovery info for step ``idx`` of a worker batch, if the worker sent it."""
    meta = batch.get("meta")
    if not meta or idx >= len(meta):
        return {}
    map_id, badges, info = meta[idx]
    return {"map_id": map_id, "badges": badges, "info": info}


def _normalized_entropy(policy, batch):
    """Mean entropy as a fraction of the maximum for the moves each state allows.

    The action mask removes blocked moves, so raw entropy falls in corridors even when the
    policy is still fully random over what it can do. This corrects for that.
    """
    from src.rl_env import MASK_START

    values = []
    for obs in batch["obs"]:
        probs = policy.probs(obs)
        allowed = 5 - int(np.sum(obs[MASK_START : MASK_START + 4] > 0.5)) if len(obs) >= MASK_START + 4 else 5
        allowed = 5 if allowed <= 1 else allowed
        values.append(-float(np.sum(probs * np.log(probs + 1e-8))) / np.log(allowed))
    return float(np.mean(values))


def _merge(parts):
    merged = {"obs": [], "actions": [], "logp": [], "rew": [], "ret": [], "adv": [], "meta": []}
    for batch, _maps in parts:
        for key in merged:
            merged[key].extend(batch.get(key, []))
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO by playing")
    parser.add_argument("--state", default="new_bark_outside")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--envs", type=int, default=1, help="parallel emulators per update")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.03, help="entropy bonus weight")
    parser.add_argument("--target-kl", type=float, default=0.02, help="stop a PPO update early past 1.5x this KL")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="logs/ppo_policy.npz")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mlp", action="store_true", help="train 2-layer MLP policy instead of linear")
    parser.add_argument("--hidden-dim", type=int, default=64, help="hidden dimension for MLP policy")
    parser.add_argument("--shaping", choices=("frontier", "legacy"), default="frontier")
    parser.add_argument(
        "--behind-maps",
        default="",
        help="maps already completed, e.g. '24,4;24,5'. Entering one ends the episode with a penalty.",
    )
    parser.add_argument(
        "--show-every",
        type=int,
        default=0,
        help="train windowless at full speed, and open PyBoy for one episode every N updates",
    )
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="render live visual reward & entropy graph alongside PyBoy (defaults to True if --window)",
    )
    args = parser.parse_args(argv)
    ignore_maps = frozenset(
        tuple(int(n) for n in item.split(",")) for item in args.behind_maps.split(";") if item.strip()
    )

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
            shaping=args.shaping,
            ignore_maps=ignore_maps,
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
                refresh_fps=4.0 if args.window else 8.0,
                save_path="logs/training_reward_entropy.png",
                jsonl_path="logs/training_metrics.jsonl",
            )

        pool = ProcessPoolExecutor(max_workers=args.envs) if args.envs > 1 else None
        try:
            for update in range(args.updates):
                payloads = [
                    (start, args.rollout, args.seed + update * args.envs + i, weights(), args.shaping, ignore_maps)
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
                                done=s_idx == len(batch["rew"]) - 1,
                                action=batch["actions"][s_idx],
                                probabilities=probs,
                                **_meta_kwargs(batch, s_idx),
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
                                badges=data.get("badges", 0),
                                map_id=data.get("map_id"),
                                probabilities=data.get("probabilities"),
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
                                done=(s_idx + 1) % args.rollout == 0,
                                action=batch["actions"][s_idx],
                                probabilities=probs,
                                **_meta_kwargs(batch, s_idx),
                            )
                stats = ppo_update(
                    policy,
                    batch,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    entropy_coef=args.entropy_coef,
                    target_kl=args.target_kl,
                )
                norm_entropy = _normalized_entropy(policy, batch)
                mean_reward = float(np.mean(batch["rew"]))
                frontier = sum(item[1] for item in parts)
                if plotter is not None:
                    plotter.log_update(
                        update=update,
                        mean_reward=mean_reward,
                        mean_entropy=stats.mean_entropy,
                        weight_change=stats.weight_change,
                        new_maps=frontier,
                        approx_kl=stats.approx_kl,
                        clip_frac=stats.clip_frac,
                        value_mse=stats.value_mse,
                    )
                if args.show_every and (update + 1) % args.show_every == 0:
                    print(f"showing episode after update {update}", flush=True)
                    _demo(start, args.rollout, args.seed + update, policy, args.speed, args.shaping, ignore_maps)
                print(
                    f"update={update} samples={len(batch['rew'])} mean_reward={mean_reward:.3f} "
                    f"weight_change={stats.weight_change:.5f} kl={stats.approx_kl:.4f} "
                    f"clip={stats.clip_frac:.2f} H={stats.mean_entropy:.3f} Hnorm={norm_entropy:.2f} new_maps={frontier}",
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
