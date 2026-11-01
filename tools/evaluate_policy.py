"""Greedy evaluation of a trained checkpoint: argmax actions, no sticky actions, no noise.

Usage:
    python -m tools.evaluate_policy --checkpoint models/deep_policy.pt --episodes 8
    python -m tools.evaluate_policy --checkpoint models/deep_policy.pt --start-state new_bark_outside \
        --episodes 16 --max-steps 1500 --jitter 6 --name run42

Each episode starts from the saved state (plus 0..``--jitter`` random safe moves so the episodes differ) and the
policy plays until it clears the last stage goal or runs out of steps. Reported per episode and as mean with a
bootstrap 95% interval: steps to reach each stage goal (leg times), events reached, maps visited, final stage,
total reward and its components, and success (reached stage 3, which means egg fetched, egg delivered, Falkner beaten).

The game-step logic (``execute_action``) mirrors the training worker (``tools.train_deep_ppo._worker_fn``) so it can
be shared later; this module deliberately does not import the trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

STAGE_NAMES = ("GET_EGG", "RETURN_EGG", "TO_GYM", "POST_GYM")
SUCCESS_STAGE = 3
# Event-flag fallback for the story stage when src.route_graph is not importable.
_STAGE_FLAGS = (
    "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON",  # stage 0 -> 1
    "EVENT_GAVE_MYSTERY_EGG_TO_ELM",  # stage 1 -> 2
    "EVENT_BEAT_FALKNER",  # stage 2 -> 3
)
_REPORTED_EVENTS = (
    "EVENT_GOT_A_POKEMON_FROM_ELM",
    "EVENT_LEARNED_TO_CATCH_POKEMON",
    "EVENT_RIVAL_CHERRYGROVE_CITY",
    "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON",
    "EVENT_GAVE_MYSTERY_EGG_TO_ELM",
    "EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST",
    "EVENT_GOT_TM31_MUD_SLAP",
    "EVENT_BEAT_FALKNER",
    "EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE",
    "EVENT_GOT_HM01_CUT",
)
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


# ----------------------------------------------------------------------------- state helpers
def _has_event(state, name: str) -> bool:
    fn = getattr(state, "has_event", None)
    try:
        return bool(callable(fn) and fn(name))
    except Exception:
        return False


def stage_of(state) -> int:
    """Story stage 0..3: ``route_graph.current_stage`` when available, else event flags."""
    try:
        from src import route_graph

        return int(route_graph.current_stage(state))
    except Exception:
        pass
    stage = 0
    for name in _STAGE_FLAGS:
        if not _has_event(state, name):
            break
        stage += 1
    return stage


def _map_of(state) -> tuple[int, int]:
    return (int(state.map_group), int(state.map_number))


# ----------------------------------------------------------------------------- game stepping
@dataclass
class StepContext:
    """Everything ``execute_action`` needs; per-episode counters live here."""

    emu: Any
    reader: Any
    stepper: Any
    macros: Any
    tile_visit_counts: dict = field(default_factory=dict)
    blocked_count: int = 0
    consecutive_bumps: int = 0


def execute_action(ctx: StepContext, act_idx: int, actions: tuple[str, ...]) -> tuple[Any, Any, bool]:
    """Turn an action index into one game step, as the training worker does.

    Battle is handled by the tactical controller, open dialogue by text mashing, ``a`` is a button press and
    directions go through ``TileStepper`` (with the worker's 4-bump escape). Returns ``(before, after, moved)``.
    """
    from src.battle import tactical_battle_step
    from src.mover import walkable
    from src.ram_parser import dialogue_open

    emu, reader, stepper, macros = ctx.emu, ctx.reader, ctx.stepper, ctx.macros
    before = reader.parse(emu.frame)
    name = actions[act_idx]
    moved = False
    if before.battle_mode != 0:
        tactical_battle_step(before, macros)
        ctx.blocked_count = 0
    elif dialogue_open(before):
        macros.mash_text(max_frames=80)
        ctx.blocked_count = 0
    elif name == "a":
        emu.press("a", frames=8)
        emu.tick(12, render=False)
    else:
        res = stepper.step(name)
        moved = bool(res.moved)
        if not moved:
            ctx.blocked_count += 1
            if ctx.blocked_count >= 4:
                cur = reader.parse(emu.frame)
                tiles = getattr(cur, "tiles", {})
                cands = []
                for d in ("left", "right", "up", "down"):
                    if walkable(tiles.get(d, 0xFF), d):
                        dx, dy = _DELTA[d]
                        cands.append((ctx.tile_visit_counts.get((cur.map_group, cur.map_number, cur.x + dx, cur.y + dy), 0), d))
                if cands:
                    cands.sort(key=lambda c: c[0])
                    stepper.step(cands[0][1])
                else:
                    emu.press("b", frames=8)
                    emu.tick(12, render=False)
                ctx.blocked_count = 0
        else:
            ctx.blocked_count = 0
    if not moved and before.battle_mode == 0 and not dialogue_open(before):
        ctx.consecutive_bumps += 1
    else:
        ctx.consecutive_bumps = 0
    return before, reader.parse(emu.frame), moved


def obs_vector(state, visited: set, obs_dim: int) -> np.ndarray:
    """Observation in the layout the checkpoint was trained with (17 base, +4 goals, +route features)."""
    from src.rl_env import DEEP_OBS_DIM, observe

    if obs_dim == DEEP_OBS_DIM:
        return observe(state, visited, include_goals=True, include_route=True)
    if obs_dim == 21:
        return observe(state, visited, include_goals=True)
    if obs_dim == 17:
        return observe(state, visited)
    raise ValueError(f"unsupported checkpoint obs_dim {obs_dim} (expected 17, 21 or {DEEP_OBS_DIM})")


class GameEnv:
    """Headless PyBoy episode wrapper: ``reset(seed)`` and ``step(action) -> (obs..., reward, components, done)``."""

    def __init__(self, start_state: str, obs_dim: int, max_steps: int, jitter: int = 6):
        from src.emulator import Emulator
        from src.macros import Macros
        from src.ram_parser import RamParser
        from src.rl_env import ACTIONS
        from src.states import StateRegistry
        from src.step import TileStepper

        self.actions = tuple(ACTIONS)
        self.start_state, self.obs_dim, self.max_steps, self.jitter = start_state, obs_dim, max_steps, jitter
        self._data = StateRegistry().load_bytes(start_state)
        self.emu = Emulator(headless=True, speed=0, rtc_lock=True)
        self.reader = RamParser(self.emu)
        self.ctx = StepContext(self.emu, self.reader, TileStepper(self.emu, self.reader), Macros(self.emu, self.reader))

    def _equip_balls(self) -> None:
        try:
            b1, a1 = self.emu.symbol("wNumBalls")
            b2, a2 = self.emu.symbol("wBalls")
            if self.emu.read(b1, a1, 1)[0] == 0:
                self.emu.write(b1, a1, bytes([1]))
                self.emu.write(b2, a2, bytes([5, 10, 255]))
        except Exception:
            pass

    def _observe(self):
        st = self.reader.parse(self.emu.frame)
        tile = (st.map_group, st.map_number, st.x, st.y)
        self.seen_tiles.add(tile)
        self.seen_maps.add((st.map_group, st.map_number))
        self.ctx.tile_visit_counts[tile] = self.ctx.tile_visit_counts.get(tile, 0) + 1
        screen = self.emu.pyboy.screen.ndarray[:, :, :3].copy()
        return screen, obs_vector(st, self.seen_tiles, self.obs_dim), st

    def reset(self, seed: int):
        from src import rl_env
        from src.ram_parser import dialogue_open

        self.emu.load_state(self._data)
        self._equip_balls()
        self.seen_tiles: set = set()
        self.seen_maps: set = set()
        self.caught_counts: dict = {}
        self.seen_waypoints: set = set()
        self.ctx.tile_visit_counts.clear()
        self.ctx.blocked_count = self.ctx.consecutive_bumps = 0
        self.steps = 0
        st = self.reader.parse(self.emu.frame)
        init = getattr(rl_env, "initial_seen_events", None)
        self.seen_events: set = set(init(st)) if callable(init) else set()
        rng = np.random.default_rng(seed)
        for _ in range(int(rng.integers(0, self.jitter + 1)) if self.jitter > 0 else 0):  # start jitter
            cur = self.reader.parse(self.emu.frame)
            if cur.battle_mode != 0 or dialogue_open(cur):
                break
            self.ctx.stepper.step(self.actions[int(rng.integers(0, 4))])
        return self._observe()

    def step(self, act_idx: int):
        from src.rl_env import StepInfo, progress_reward

        before, after, moved = execute_action(self.ctx, act_idx, self.actions)
        unvisited = False
        if 0 <= act_idx < 4:
            dx, dy = _DELTA[self.actions[act_idx]]
            unvisited = (before.map_group, before.map_number, before.x + dx, before.y + dy) not in self.seen_tiles
        comps: dict[str, float] = {}
        reward, _, _ = progress_reward(
            before, after, moved, self.seen_tiles, self.seen_maps,
            unvisited_neighbor=unvisited, caught_counts=self.caught_counts, seen_events=self.seen_events,
            seen_waypoints=self.seen_waypoints, info=StepInfo(consecutive_bumps=self.ctx.consecutive_bumps),
            components=comps,
        )
        self.steps += 1
        screen, obs, st = self._observe()
        return screen, obs, st, float(reward), comps, self.steps >= self.max_steps

    def close(self) -> None:
        self.emu.close()


# ----------------------------------------------------------------------------- policy wrapper
class GreedyPolicy:
    """Argmax over a ``SpatialVisionPolicy`` (carries the GRU state when the checkpoint is recurrent)."""

    def __init__(self, policy, device: str = "cpu"):
        self.policy, self.device, self.hx = policy, device, None
        self.obs_dim = int(policy.obs_dim)

    def reset(self) -> None:
        self.hx = None

    def act(self, screen: np.ndarray, obs: np.ndarray) -> int:
        ram = np.asarray(obs, dtype=np.float32)
        if getattr(self.policy, "use_gru", False):
            probs, self.hx = self.policy.probs_recurrent(screen, ram, self.hx, device=self.device)
        else:
            probs = self.policy.probs(screen, ram, device=self.device)
        return int(np.argmax(probs))


# ----------------------------------------------------------------------------- episodes and stats
def run_episode(env, policy, seed: int, max_steps: int) -> dict:
    """Play one greedy episode and return its metrics. ``env`` and ``policy`` are duck-typed (see GameEnv, GreedyPolicy)."""
    policy.reset()
    screen, obs, st = env.reset(seed)
    stage = stage_of(st)
    leg_steps: dict[int, int] = {}  # leg k is complete the first step the stage is >= k + 1
    events_at: dict[str, int] = {name: 0 for name in _REPORTED_EVENTS if _has_event(st, name)}
    maps = {_map_of(st)}
    total = 0.0
    comps: dict[str, float] = {}
    steps = 0
    t0 = time.time()
    while steps < max_steps:
        act = policy.act(screen, obs)
        screen, obs, st, reward, c, done = env.step(act)
        steps += 1
        total += reward
        for k, v in (c or {}).items():
            comps[k] = comps.get(k, 0.0) + float(v)
        maps.add(_map_of(st))
        stage = max(stage, stage_of(st))
        for k in range(stage):
            leg_steps.setdefault(k, steps)
        for name in _REPORTED_EVENTS:
            if name not in events_at and _has_event(st, name):
                events_at[name] = steps
        if stage >= SUCCESS_STAGE or done:
            break
    return {
        "seed": seed,
        "steps": steps,
        "final_stage": stage,
        "final_stage_name": STAGE_NAMES[min(stage, len(STAGE_NAMES) - 1)],
        "success": stage >= SUCCESS_STAGE,
        "leg_steps": {str(k): v for k, v in sorted(leg_steps.items())},
        "events": events_at,
        "maps_visited": len(maps),
        "maps": sorted(f"{g},{n}" for g, n in maps),
        "total_reward": round(total, 3),
        "components": {k: round(v, 3) for k, v in sorted(comps.items())},
        "seconds": round(time.time() - t0, 1),
    }


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high) with a percentile bootstrap interval; NaNs when there are no values."""
    if not values:
        return (float("nan"),) * 3
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return (float(arr[0]),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _stat(values: list[float]) -> dict:
    mean, lo, hi = bootstrap_ci(values)
    return {"n": len(values), "mean": mean, "ci_low": lo, "ci_high": hi}


def aggregate(episodes: list[dict]) -> dict:
    """Mean and bootstrap interval across episodes. Leg times only average the episodes that reached the leg."""
    legs = {}
    for k in range(SUCCESS_STAGE):
        reached = [e["leg_steps"][str(k)] for e in episodes if str(k) in e["leg_steps"]]
        legs[str(k)] = {"name": STAGE_NAMES[k], "reached": len(reached), "of": len(episodes), **_stat(reached)}
    comp_keys = sorted({k for e in episodes for k in e["components"]})
    event_names = sorted({n for e in episodes for n in e["events"]})
    return {
        "episodes": len(episodes),
        "success_rate": _stat([1.0 if e["success"] else 0.0 for e in episodes]),
        "final_stage": _stat([float(e["final_stage"]) for e in episodes]),
        "total_reward": _stat([e["total_reward"] for e in episodes]),
        "steps": _stat([float(e["steps"]) for e in episodes]),
        "maps_visited": _stat([float(e["maps_visited"]) for e in episodes]),
        "legs": legs,
        "components": {k: _stat([e["components"].get(k, 0.0) for e in episodes]) for k in comp_keys},
        "events_reached": {n: sum(1 for e in episodes if n in e["events"]) for n in event_names},
    }


def format_report(result: dict) -> str:
    agg = result["aggregate"]
    f = lambda s, d=1: f"{s['mean']:.{d}f} [{s['ci_low']:.{d}f}, {s['ci_high']:.{d}f}]"
    lines = [
        f"Greedy evaluation: {result['checkpoint']}  start={result['start_state']}  episodes={agg['episodes']}  "
        f"max_steps={result['max_steps']}",
        "",
        f"{'ep':>3} {'seed':>5} {'steps':>6} {'stage':<11} {'ok':>3} {'maps':>4} {'reward':>9}  legs (steps to finish each stage goal)",
    ]
    for i, e in enumerate(result["episodes"]):
        legs = " ".join(f"{STAGE_NAMES[int(k)][:6]}={v}" for k, v in e["leg_steps"].items()) or "-"
        lines.append(
            f"{i:>3} {e['seed']:>5} {e['steps']:>6} {e['final_stage_name']:<11} {'yes' if e['success'] else 'no':>3} "
            f"{e['maps_visited']:>4} {e['total_reward']:>9.1f}  {legs}"
        )
    lines += ["", "Mean [95% bootstrap CI]",
              f"  success rate   {f(agg['success_rate'], 2)}",
              f"  final stage    {f(agg['final_stage'], 2)}",
              f"  total reward   {f(agg['total_reward'])}",
              f"  steps          {f(agg['steps'], 0)}",
              f"  maps visited   {f(agg['maps_visited'])}"]
    for k, leg in agg["legs"].items():
        t = f(leg, 0) if leg["reached"] else "never reached"
        lines.append(f"  leg {k} {leg['name']:<10} reached {leg['reached']}/{leg['of']}  steps {t}")
    if agg["components"]:
        lines.append("  reward by component (mean per episode)")
        for k, s in sorted(agg["components"].items(), key=lambda kv: -abs(kv[1]["mean"])):
            lines.append(f"    {k:<12} {f(s)}")
    if agg["events_reached"]:
        lines.append("  events reached: " + ", ".join(f"{n.removeprefix('EVENT_').lower()} {c}/{agg['episodes']}" for n, c in agg["events_reached"].items()))
    return "\n".join(lines)


def evaluate(
    env,
    policy,
    episodes: int,
    max_steps: int,
    seed: int = 0,
    meta: dict | None = None,
    on_episode: Callable[[int, dict], None] | None = None,
) -> dict:
    results = []
    for i in range(episodes):
        ep = run_episode(env, policy, seed + i, max_steps)
        results.append(ep)
        if on_episode:
            on_episode(i, ep)
    return {
        **(meta or {}),
        "max_steps": max_steps,
        "success_definition": f"reached stage {SUCCESS_STAGE} (egg fetched, egg delivered, Falkner beaten)",
        "episodes": results,
        "aggregate": aggregate(results),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Greedy evaluation of a SpatialVisionPolicy checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--start-state", default="new_bark_outside")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=int, default=6, help="max random safe moves after reset (0 disables)")
    ap.add_argument("--name", default=None, help="output name: logs/eval/<name>.json (default: checkpoint stem)")
    ap.add_argument("--out-dir", default="logs/eval")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from src.deep_policy import SpatialVisionPolicy

    net = SpatialVisionPolicy.load(args.checkpoint, device=args.device)
    net.to(args.device)
    policy = GreedyPolicy(net, args.device)
    env = GameEnv(args.start_state, policy.obs_dim, args.max_steps, args.jitter)
    try:
        result = evaluate(
            env, policy, args.episodes, args.max_steps, args.seed,
            meta={"checkpoint": str(args.checkpoint), "start_state": args.start_state, "jitter": args.jitter, "seed": args.seed},
            on_episode=lambda i, e: print(f"episode {i}: stage {e['final_stage_name']} in {e['steps']} steps, reward {e['total_reward']:.1f}", flush=True),
        )
    finally:
        env.close()
    print()
    print(format_report(result))
    name = args.name or Path(args.checkpoint).stem
    out = Path(args.out_dir) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
