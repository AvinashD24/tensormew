"""Wall-clock PPO training (headless, max speed) with periodic visual policy runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.ppo import MlpPolicy, Policy, ppo_update
from src.states import StateRegistry
from tools.train_ppo import _merge, _one_rollout


def _visual_run(
    *,
    state: str,
    policy_path: str,
    steps: int,
    speed: int,
    plot: bool,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "tools.run_segment",
        "--state",
        state,
        "--policy",
        policy_path,
        "--steps",
        str(steps),
        "--window",
        "--speed",
        str(speed),
    ]
    if plot:
        cmd.append("--plot")
    else:
        cmd.append("--no-plot")
    print(f"watch_start cmd={' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    print(f"watch_done exit={result.returncode}", flush=True)
    return int(result.returncode)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train PPO headless for a duration; open PyBoy with latest weights on an interval",
    )
    parser.add_argument("--state", default="new_bark_outside")
    parser.add_argument("--duration-minutes", type=float, default=60.0)
    parser.add_argument("--watch-every-minutes", type=float, default=10.0)
    parser.add_argument("--rollout", type=int, default=64)
    default_envs = 1 if sys.platform == "win32" else 4
    parser.add_argument(
        "--envs",
        type=int,
        default=default_envs,
        help="parallel emulators (use 1 on Windows; >1 needs PyBoy importable in worker processes)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="logs/ppo_policy.npz")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--watch-steps", type=int, default=120, help="steps per visual run_segment")
    parser.add_argument("--watch-speed", type=int, default=4, help="PyBoy speed during watch runs")
    parser.add_argument(
        "--watch-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="live metrics plot during watch runs",
    )
    parser.add_argument(
        "--train-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="live matplotlib dashboard while headless training runs (PyBoy stays off-screen)",
    )
    args = parser.parse_args(argv)

    if args.train_plot or args.watch_plot:
        os.environ.setdefault("MPLBACKEND", "TkAgg")

    duration_s = args.duration_minutes * 60.0
    watch_s = args.watch_every_minutes * 60.0
    deadline = time.monotonic() + duration_s
    next_watch = time.monotonic() + watch_s
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start = StateRegistry().load_bytes(args.state)
    use_mlp = args.mlp
    if args.resume and save_path.exists():
        data = np.load(save_path)
        if "w1" in data or str(data.get("model_type", "")) == "mlp":
            use_mlp = True

    if use_mlp:
        policy = MlpPolicy(rng=np.random.default_rng(args.seed), hidden_dim=args.hidden_dim)
    else:
        policy = Policy(rng=np.random.default_rng(args.seed))

    if args.resume and save_path.exists():
        policy.load(str(save_path))
        print(f"loaded={save_path} (type={policy.__class__.__name__})", flush=True)

    weights = lambda: policy.get_weights()
    update = 0
    pool = ProcessPoolExecutor(max_workers=args.envs) if args.envs > 1 else None
    plotter = None
    if args.train_plot:
        from src.visual_metrics import LiveMetricsPlotter

        plotter = LiveMetricsPlotter(
            title=f"TensorMew: Marathon PPO ({policy.__class__.__name__})",
            enabled=True,
            headless=False,
            refresh_fps=8.0,
            save_path="logs/marathon_reward_entropy.png",
            jsonl_path="logs/marathon_metrics.jsonl",
        )
        if not plotter._initialized_gui:
            print(
                "train_plot: no live window (see stderr above); open logs/marathon_reward_entropy.png after updates",
                flush=True,
            )

    print(
        f"marathon_start duration_min={args.duration_minutes} watch_every_min={args.watch_every_minutes} "
        f"rollout={args.rollout} envs={args.envs} train_plot={args.train_plot} save={save_path}",
        flush=True,
    )

    try:
        while time.monotonic() < deadline:
            payloads = [
                (start, args.rollout, args.seed + update * args.envs + i, weights())
                for i in range(args.envs)
            ]
            if args.envs == 1:
                batch, frontier_maps = _one_rollout(payloads[0])
                parts = [(batch, frontier_maps)]
            else:
                parts = list(pool.map(_one_rollout, payloads))
                batch = _merge(parts)
            frontier = sum(item[1] for item in parts)

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
                    )

            stats = ppo_update(policy, batch, batch_size=args.batch_size)
            mean_reward = float(np.mean(batch["rew"]))
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
            policy.save(str(save_path))
            elapsed_min = (time.monotonic() - (deadline - duration_s)) / 60.0
            print(
                f"update={update} elapsed_min={elapsed_min:.1f} samples={len(batch['rew'])} "
                f"mean_reward={mean_reward:.3f} weight_change={stats.weight_change:.5f} "
                f"kl={stats.approx_kl:.4f} clip={stats.clip_frac:.2f} new_maps={frontier} saved={save_path}",
                flush=True,
            )
            update += 1

            now = time.monotonic()
            while now >= next_watch and next_watch <= deadline:
                _visual_run(
                    state=args.state,
                    policy_path=str(save_path),
                    steps=args.watch_steps,
                    speed=args.watch_speed,
                    plot=args.watch_plot,
                )
                next_watch += watch_s
                now = time.monotonic()
    finally:
        if pool is not None:
            pool.shutdown()
        if plotter is not None:
            saved_plot = plotter.save()
            if saved_plot:
                print(f"saved_plot={saved_plot}", flush=True)
            plotter.close()
        policy.save(str(save_path))
        print(f"marathon_done updates={update} saved={save_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
