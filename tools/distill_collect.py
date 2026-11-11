"""Record a trained pixel policy playing, as training data for the grid student (see ``src.distill``).

    python -m tools.distill_collect --teacher models/v8-1.pt --config long_v8 --procs 4 --steps 30000 --out data/distill/v8

Each process owns one emulator and one copy of the teacher; it samples actions from the teacher's own distribution (so the
states visited are the ones the teacher actually reaches, with its sticky actions / start jitter / archives from ``--config``) and
writes one shard ``<out>/shard_<i>.npz``. The teacher is evaluated exactly as during training (eval mode, no early battle A-bias,
the same valid-action mask).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from src.distill import save_shard


def _dual_worker_class():
    """EnvWorker that also exposes the semantic grid of the observation it returns (``obs_grid``) next to the pixels."""
    from src.ram_parser import dialogue_open
    from src.spatial_grid import GridExtractor
    from tools.train_deep_ppo import EnvWorker

    class DualWorker(EnvWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._extractor = GridExtractor(self.emu)
            self._last_grid = None
            self.obs_grid = None

        def _screen(self, st=None):
            st = st if st is not None else self.reader.parse(self.emu.frame)
            self._last_grid = self._extractor.extract(visited=self.visited,
                                                      menu_open=bool(st.battle_mode) or dialogue_open(st))
            return self.emu._pb.screen.ndarray[:, :, :3].copy()

        def get_obs(self):
            out = super().get_obs()
            self.obs_grid = self._last_grid   # the grid of THIS observation (a later _peek_obs must not overwrite it)
            return out

    return DualWorker


def collect(idx: int, args) -> None:
    import torch

    from src import story_path as _story
    from src.deep_policy import SpatialVisionPolicy
    from src.rl_env import deep_obs_dim
    from src.train_config import load_config
    from tools.train_deep_ppo import resolve_start_states, worker_env_cfg

    cfg = load_config(args.config)
    _story.set_profile(cfg.egg_errand_route)
    obs_dim = deep_obs_dim(time_aware=cfg.time_feature)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    torch.manual_seed(args.seed * 1000 + idx)
    teacher = SpatialVisionPolicy.load(args.teacher, device=device, override_use_gru=cfg.recurrent,
                                       override_obs_dim=obs_dim, check_schema=True)
    teacher.eval()
    teacher.set_battle_a_bias(0.0)

    root = Path(args.archive_root) / f"collect_{idx}"
    env_cfg = worker_env_cfg(cfg, root if cfg.swarm else None, root / "battle" if cfg.battle_archive else None,
                             root / "route" if cfg.route_ckpt else None)
    env_cfg.update({"seed": args.seed * 1000 + idx, "num_envs": args.procs})
    states = resolve_start_states(cfg, None, None, None)
    w = _dual_worker_class()(idx, states, cfg.max_steps, env_cfg)

    grids, rams, probs, masks, dones = [], [], [], [], []
    screen, ram = w.reset()
    grid = w.obs_grid
    hx = torch.zeros(1, 1, teacher.gru_hidden_dim, device=device) if teacher.use_gru else None
    t0, episodes = time.time(), 0
    try:
        with torch.no_grad():
            for step in range(args.steps):
                mask = np.asarray(w.last_mask, dtype=bool).copy()
                s = teacher.visual_batch(np.asarray(screen)[None], device)
                r = torch.from_numpy(np.asarray(ram)).float().to(device).unsqueeze(0)
                mkw = {"action_mask": torch.from_numpy(mask).to(device).unsqueeze(0)} if w.action_mask else {}
                if teacher.use_gru:
                    logits, _v, hx = teacher.forward_recurrent(s, r, hx, **mkw)
                else:
                    logits, _v = teacher(s, r, **mkw)
                p = torch.softmax(logits.float(), dim=-1)[0]
                action = int(torch.multinomial(p, 1).item())
                grids.append(grid)
                rams.append(np.asarray(ram, dtype=np.float32))
                probs.append(p.cpu().numpy())
                masks.append(mask)
                screen, ram, _rew, done, _maps, _badges, _info = w.step(action)
                grid = w.obs_grid
                dones.append(bool(done))
                if done:
                    episodes += 1
                    if hx is not None:
                        hx.zero_()
                if (step + 1) % 2000 == 0:
                    print(f"[collect {idx}] {step + 1}/{args.steps} steps, {episodes} episodes, "
                          f"{(step + 1) / (time.time() - t0):.0f} steps/s", flush=True)
    finally:
        w.close()
        if grids:
            out = Path(args.out) / f"shard_{idx}.npz"
            save_shard(out, grids, rams, probs, masks, dones)
            print(f"[collect {idx}] wrote {len(grids)} steps -> {out}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True, help="pixel checkpoint to imitate (read only)")
    p.add_argument("--config", required=True, help="preset / config the teacher was trained with (long_v8 for v8-1)")
    p.add_argument("--out", required=True, help="directory for the shards")
    p.add_argument("--procs", type=int, default=4, help="parallel emulators")
    p.add_argument("--steps", type=int, default=30000, help="env steps recorded per process")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive-root", default="logs/swarm/distill", help="start-state archives the collectors build (own dirs)")
    p.add_argument("--cpu", action="store_true", help="run the teacher on the CPU even if CUDA exists")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=collect, args=(i, args)) for i in range(args.procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if bad:
        print(f"collector processes {bad} failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
