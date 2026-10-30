"""Live training dashboard: game screens, charts, coverage heatmaps and run health.

Usage:
    python -m tools.dashboard                 # watch logs/ from a running (or finished) training run
    python -m tools.dashboard --demo          # preview the UI with a synthetic run, no training needed
    python -m tools.dashboard --port 9000 --live-dir logs/live --metrics logs/training_metrics.jsonl

Serves a single page on http://127.0.0.1:8765 using only the standard library. Training
writes telemetry through ``src.live_feed.LiveFeed``; this server only reads those files.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC_DIR = Path(__file__).resolve().parent / "dashboard_static"
METRICS_TAIL_BYTES = 8 * 1024 * 1024


def read_metrics(path: Path, since: int = -1) -> list[dict]:
    """Per-update records from the metrics JSONL (skips per-step records, bounded tail read)."""
    if not path.is_file():
        return []
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if size > METRICS_TAIL_BYTES:
            fh.seek(size - METRICS_TAIL_BYTES)
            fh.readline()  # drop the partial first line
        raw = fh.read().decode("utf-8", errors="replace")
    out: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "update" in rec and "type" not in rec and rec["update"] > since:
            out.append(rec)
    return out


def make_handler(live_dir: Path, metrics_path: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TensorMewDashboard/1.0"

        def log_message(self, *args):  # keep the training terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            route = url.path
            try:
                if route in ("/", "/index.html"):
                    page = (STATIC_DIR / "index.html").read_bytes()
                    return self._send(200, page, "text/html; charset=utf-8")
                if route == "/api/metrics":
                    since = int(parse_qs(url.query).get("since", ["-1"])[0])
                    body = json.dumps(read_metrics(metrics_path, since)).encode()
                    return self._send(200, body, "application/json")
                if route == "/api/live":
                    state = live_dir / "state.json"
                    if not state.is_file():
                        return self._send(200, b"null", "application/json")
                    return self._send(200, state.read_bytes(), "application/json")
                if route.startswith("/frames/env_") and route.endswith(".png"):
                    name = route.rsplit("/", 1)[-1]
                    frame = live_dir / name
                    if not frame.is_file():
                        return self._send(404, b"", "image/png")
                    return self._send(200, frame.read_bytes(), "image/png")
            except (ConnectionError, OSError):
                return
            except Exception as exc:  # pragma: no cover
                return self._send(500, str(exc).encode(), "text/plain")
            self._send(404, b"not found", "text/plain")

    return Handler


# --------------------------------------------------------------------------- demo
def _demo_frame(env: int, map_idx: int, x: int, y: int, battle: bool, t: float):
    """Procedural 160x144 'screen' so the demo grid looks alive."""
    import numpy as np

    palettes = [(120, 180, 90), (90, 140, 200), (200, 170, 100), (150, 110, 170), (110, 170, 150)]
    base = np.array(palettes[map_idx % len(palettes)], dtype=np.uint8)
    img = np.empty((144, 160, 3), dtype=np.uint8)
    img[:] = base
    gy, gx = np.mgrid[0:144, 0:160]
    checker = ((gx // 16 + gy // 16) % 2).astype(np.uint8) * 14
    img = np.clip(img.astype(np.int16) + checker[..., None] - 7, 0, 255).astype(np.uint8)
    if battle:
        img[:] = (img * 0.55 + np.array([120, 30, 30]) * 0.45).astype(np.uint8)
        img[100:144, :, :] = 245
    px = int((x * 7 + 80 + 12 * math.sin(t + env)) % 144) + 8
    py = int((y * 5 + 72) % 112) + 8
    img[py : py + 14, px : px + 10] = (230, 60, 60)
    img[py : py + 5, px : px + 10] = (250, 250, 250)
    return img


def run_demo(live_dir: Path, metrics_path: Path, stop: threading.Event) -> None:
    """Drive the real LiveFeed + TerminalMonitor with a synthetic training run."""
    import numpy as np

    from src.live_feed import LiveFeed
    from src.terminal_monitor import TerminalMonitor

    rng = random.Random(7)
    n_envs = 12
    actions = ("up", "down", "left", "right", "a")
    feed = LiveFeed(
        live_dir,
        num_envs=n_envs,
        action_names=actions,
        run_info={
            "total_updates": 400,
            "rollout_steps": 128,
            "num_envs": n_envs,
            "lr": 3e-4,
            "entropy_coef": 0.02,
            "recurrent": True,
            "resume": False,
            "curriculum_states": ["route29_entry", "new_bark_outside", "post_rival_cherrygrove"],
            "device": "demo",
        },
    )
    monitor = TerminalMonitor(total_updates=400, jsonl_path=metrics_path)
    maps = [(24, 4), (24, 3), (26, 1), (26, 3), (24, 5)]
    pos = [[rng.randint(3, 20), rng.randint(3, 16)] for _ in range(n_envs)]
    midx = [rng.randint(0, 2) for _ in range(n_envs)]
    ep_steps = [0] * n_envs
    update = 0
    steps_in_update = 0
    t0 = time.time()
    seen = 1
    while not stop.is_set():
        acts = np.array([rng.choices(range(5), weights=[3, 3, 3, 3, 1])[0] for _ in range(n_envs)])
        rews = np.array([rng.gauss(0.08, 0.5) for _ in range(n_envs)], dtype=np.float32)
        dones = np.zeros(n_envs, dtype=bool)
        infos = []
        screens = []
        for i in range(n_envs):
            dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))[int(acts[i])]
            pos[i][0] = max(0, min(30, pos[i][0] + dx))
            pos[i][1] = max(0, min(24, pos[i][1] + dy))
            ep_steps[i] += 1
            if rng.random() < 0.004 and midx[i] < len(maps) - 1:
                midx[i] += 1
                pos[i] = [rng.randint(3, 20), rng.randint(3, 16)]
                rews[i] += 6.0
            if ep_steps[i] >= 300:
                dones[i] = True
            battle = rng.random() < 0.02
            infos.append(
                {
                    "map": list(maps[midx[i]]),
                    "x": pos[i][0],
                    "y": pos[i][1],
                    "state": "route29_entry",
                    "battle": battle,
                    "badges": 0,
                    "ended_maps": midx[i] + 1,
                    "ended_state": "route29_entry",
                }
            )
            screens.append(_demo_frame(i, midx[i], pos[i][0], pos[i][1], battle, time.time() - t0))
            if dones[i]:
                ep_steps[i] = 0
                midx[i] = rng.randint(0, 1)
        feed.observe(infos=infos, actions=acts, rewards=rews, dones=dones, screens=np.stack(screens))
        steps_in_update += 1
        if steps_in_update >= 40:
            steps_in_update = 0
            x = update
            progress = 1 - math.exp(-x / 90)
            reward = -8 + 70 * progress + rng.gauss(0, 6) + (35 if x in (120, 121, 122) else 0)
            collapse = 170 <= x <= 185
            entropy = max(0.18, 1.55 * math.exp(-x / 260) + rng.gauss(0, 0.04) - (0.9 if collapse else 0))
            kl = abs(rng.gauss(0.012, 0.008)) + (0.09 if x in (60, 61) else 0)
            seen = max(seen, 1 + int(progress * 4.2))
            with contextlib.redirect_stdout(io.StringIO()):
                monitor.log_update(
                    update=update,
                    mean_reward=reward,
                    entropy=entropy,
                    value_loss=max(0.2, 40 * math.exp(-x / 70) + rng.gauss(0, 2)),
                    policy_loss=rng.gauss(0, 0.02),
                    approx_kl=kl,
                    clip_frac=min(0.5, kl * 4 + rng.random() * 0.03),
                    lr=3e-4,
                    maps_discovered=seen,
                    badges=0,
                    duration_sec=2.0 + rng.random(),
                    samples=40 * n_envs,
                )
            feed.on_update(update, 40 * n_envs)
            update += 1
        time.sleep(0.04)
    feed.finish()


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live-dir", default="logs/live")
    parser.add_argument("--metrics", default="logs/training_metrics.jsonl")
    parser.add_argument("--demo", action="store_true", help="drive the UI with a synthetic training run")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    stop = threading.Event()
    live_dir, metrics_path = Path(args.live_dir), Path(args.metrics)
    if args.demo:
        tmp = Path(tempfile.mkdtemp(prefix="tensormew_demo_"))
        live_dir, metrics_path = tmp / "live", tmp / "metrics.jsonl"
        threading.Thread(target=run_demo, args=(live_dir, metrics_path, stop), daemon=True).start()
        print(f"Demo mode: synthetic run in {tmp}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(live_dir, metrics_path))
    url = f"http://{args.host}:{args.port}/"
    print(f"Dashboard: {url}   (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
