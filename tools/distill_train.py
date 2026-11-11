"""Train the grid policy to imitate a pixel teacher from shards written by ``tools.distill_collect``.

    python -m tools.distill_train --data data/distill/v8 --config long_v8 --out logs/grid_from_v8.pt --steps 3000 --eval

The result is an ordinary ``SpatialGridPolicy`` checkpoint (recurrent, same observation layout as ``--config``), ready for PPO:

    python -m tools.train_deep_ppo --config long_v8 --visual grid --resume --save logs/grid_from_v8.pt

Only the actor is distilled. The critic and the optimizer start fresh (the teacher's values are in a different reward scale), so
the first PPO updates should use a small learning rate (``--lr 5e-5``) until the new critic has caught up.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src import story_path as _story
from src.distill import ChunkSampler, distill_loss, load_shard, uniform_baseline
from src.grid_policy import SpatialGridPolicy
from src.rl_env import N_ACTIONS, deep_obs_dim
from src.train_config import apply_overrides, load_config


def to_tensors(batch: dict, device) -> dict:
    return {
        "grids": torch.from_numpy(batch["grids"]).to(device).float(),
        "rams": torch.from_numpy(batch["rams"]).to(device),
        "probs": torch.from_numpy(batch["probs"]).to(device),
        "masks": torch.from_numpy(batch["masks"]).to(device),
        "dones": torch.from_numpy(batch["dones"]).to(device),
    }


@torch.no_grad()
def validate(student, sampler: ChunkSampler, device, batches: int, batch_size: int, burn_in: int) -> tuple[float, float]:
    student.eval()
    sampler.rng = np.random.default_rng(12345)   # the same chunks every time
    kls, agrees = [], []
    for _ in range(batches):
        b = to_tensors(sampler.sample(batch_size), device)
        logits, _ = student.sequence_outputs(b["grids"], b["rams"], None, b["dones"], b["masks"])
        kl, agree = distill_loss(logits, b["probs"], burn_in)
        kls.append(float(kl))
        agrees.append(float(agree))
    student.train()
    return float(np.mean(kls)), float(np.mean(agrees))


def split_shards(shards: list[dict], holdout_frac: float) -> tuple[list[dict], list[dict]]:
    """Whole shards stay together; with a single shard its last ``holdout_frac`` becomes the held-out part."""
    if len(shards) > 1:
        n = max(1, int(round(len(shards) * holdout_frac)))
        return shards[:-n], shards[-n:]
    s = shards[0]
    cut = int(len(s["grids"]) * (1.0 - holdout_frac))
    return [{k: v[:cut] for k, v in s.items()}], [{k: v[cut:] for k, v in s.items()}]


def evaluate_policy(label: str, policy, cfg, device) -> None:
    from tools.train_deep_ppo import PeriodicEvaluator

    ev = PeriodicEvaluator(cfg, device)
    try:
        res = ev.run(policy, 0)
    finally:
        ev.close()
    print(f"[eval] {label}: {res}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory with shard_*.npz")
    p.add_argument("--config", required=True, help="preset / config of the teacher (and of the PPO fine-tune afterwards)")
    p.add_argument("--out", required=True, help="where to write the student checkpoint")
    p.add_argument("--steps", type=int, default=3000, help="optimizer steps")
    p.add_argument("--batch", type=int, default=32, help="chunks per step")
    p.add_argument("--chunk", type=int, default=80, help="steps per chunk")
    p.add_argument("--burn-in", type=int, default=16, help="leading steps per chunk that only warm up the GRU")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--holdout", type=float, default=0.15, help="fraction of shards kept for validation")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--eval", action="store_true", help="from-start evaluation of the finished student (one emulator)")
    p.add_argument("--eval-teacher", default=None, help="also evaluate this pixel checkpoint the same way (for comparison)")
    p.add_argument("--eval-episodes", type=int, default=4)
    p.add_argument("--eval-max-steps", type=int, default=2048)
    args = p.parse_args(argv)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    cfg = apply_overrides(load_config(args.config), {"visual": "grid", "eval_episodes": args.eval_episodes,
                                                     "eval_max_steps": args.eval_max_steps})
    _story.set_profile(cfg.egg_errand_route)
    obs_dim = deep_obs_dim(time_aware=cfg.time_feature)
    torch.manual_seed(args.seed)

    paths = sorted(Path(args.data).glob("shard_*.npz"))
    if not paths:
        print(f"no shard_*.npz in {args.data}", file=sys.stderr)
        return 1
    shards = [load_shard(x) for x in paths]
    if shards[0]["rams"].shape[1] != obs_dim:
        print(f"shards have {shards[0]['rams'].shape[1]}-dim RAM vectors but --config {args.config} uses {obs_dim}", file=sys.stderr)
        return 1
    train_shards, val_shards = split_shards(shards, args.holdout)
    train = ChunkSampler(train_shards, args.chunk, args.burn_in, args.seed)
    val = ChunkSampler(val_shards, args.chunk, args.burn_in, args.seed + 1)
    base = uniform_baseline(val_shards)
    print(f"[data] {len(shards)} shards, {train.steps} train / {val.steps} held-out steps | uniform-policy baseline on held-out: "
          f"KL {base['kl']:.3f}, top-1 agreement {base['agree']:.3f}", flush=True)

    student = SpatialGridPolicy(obs_dim=obs_dim, num_actions=N_ACTIONS, use_gru=cfg.recurrent).to(device)
    student.train()
    print(f"[student] {student.param_count:,} parameters on {device}", flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / max(1, args.steps)))))

    best_kl, best_state, t0 = float("inf"), None, time.time()
    for step in range(1, args.steps + 1):
        b = to_tensors(train.sample(args.batch), device)
        logits, _ = student.sequence_outputs(b["grids"], b["rams"], None, b["dones"], b["masks"])
        kl, agree = distill_loss(logits, b["probs"], args.burn_in)
        opt.zero_grad(set_to_none=True)
        kl.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"step {step:5d} | train KL {float(kl):.4f} top-1 {float(agree):.3f} | lr {sched.get_last_lr()[0]:.2e} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            vkl, vag = validate(student, val, device, batches=8, batch_size=args.batch, burn_in=args.burn_in)
            print(f"  [val] step {step}: held-out KL {vkl:.4f}, top-1 agreement {vag:.3f}", flush=True)
            if vkl < best_kl:
                best_kl = vkl
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    if best_state is not None:
        student.load_state_dict(best_state)
    student.to("cpu").save(args.out)
    print(f"[done] best held-out KL {best_kl:.4f} -> {args.out}", flush=True)

    if args.eval or args.eval_teacher:
        student.to(device).eval()
        if args.eval:
            evaluate_policy("grid student (distilled)", student, cfg, device)
        if args.eval_teacher:
            from src.deep_policy import SpatialVisionPolicy

            pix_cfg = apply_overrides(cfg, {"visual": "pixels"})
            teacher = SpatialVisionPolicy.load(args.eval_teacher, device=device, override_use_gru=cfg.recurrent,
                                               override_obs_dim=obs_dim, check_schema=True)
            evaluate_policy(f"pixel teacher ({args.eval_teacher})", teacher, pix_cfg, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
