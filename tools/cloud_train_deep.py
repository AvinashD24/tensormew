"""Run high-throughput 11.6M Spatial Vision PPO training on the remote RunPod RTX 4090.

Syncs codebase, runs the vectorized GPU training engine across 12 parallel emulators,
and syncs the trained 11.6M model checkpoint back to logs/deep_policy.pt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

DEFAULT_POD_HOST = os.environ.get("RUNPOD_SSH_HOST", "")
DEFAULT_POD_PORT = os.environ.get("RUNPOD_SSH_PORT", "")
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_runpod")
REMOTE_ROOT = "/workspace/tensormew"

# (remote path relative to the repo, local path): live feed for the dashboard, replayable run
# histories, and the per-update metrics log. Directories end in "/" so rsync copies their contents.
LOG_PULLS = [
    ("logs/live/", "logs/live/"),
    ("logs/runs/", "logs/runs/"),
    ("logs/training_metrics.jsonl", "logs/training_metrics.jsonl"),
]


def run_cmd(cmd: list[str]) -> int:
    print(f"--> {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd)
    return res.returncode


def build_pull_cmd(ssh_cmd: str, host: str, remote_rel: str, local_rel: str) -> list[str]:
    """rsync command that copies one telemetry path from the pod into the local logs/."""
    return ["rsync", "-az", "-e", ssh_cmd, f"root@{host}:{REMOTE_ROOT}/{remote_rel}", f"./{local_rel}"]


def pull_logs(ssh_cmd: str, host: str, quiet: bool = True) -> None:
    """Best-effort copy of live telemetry and run histories (paths may not exist yet early in a run)."""
    for remote_rel, local_rel in LOG_PULLS:
        target = Path(local_rel)
        (target if local_rel.endswith("/") else target.parent).mkdir(parents=True, exist_ok=True)
        cmd = build_pull_cmd(ssh_cmd, host, remote_rel, local_rel)
        if quiet:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            run_cmd(cmd)


def watch_loop(ssh_cmd: str, host: str, every: float, stop: threading.Event) -> None:
    while not stop.wait(every):
        pull_logs(ssh_cmd, host, quiet=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cloud training of 11.6M Vision Policy on RTX 4090")
    parser.add_argument("--host", default=DEFAULT_POD_HOST, help="pod SSH host (or set RUNPOD_SSH_HOST)")
    parser.add_argument("--port", default=DEFAULT_POD_PORT, help="pod SSH port (or set RUNPOD_SSH_PORT)")
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--envs", type=int, default=16, help="parallel emulators on pod (matched to dedicated cores)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--state", default="route29_entry")
    parser.add_argument("--states", default="", help="comma-separated list of curriculum states")
    parser.add_argument("--curriculum", action="store_true", default=True, help="use multi-state curriculum resets")
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum")
    parser.add_argument("--recurrent", action="store_true", default=True, help="use recurrent GRU temporal memory (13.2M model)")
    parser.add_argument("--no-recurrent", action="store_false", dest="recurrent")
    parser.add_argument("--save", default="logs/deep_policy.pt")
    parser.add_argument("--resume", action="store_true", default=False, help="warm-start from the remote checkpoint (off by default)")
    parser.add_argument("--run-id", default="", help="name for this run's saved history (default: timestamp)")
    parser.add_argument("--watch", action="store_true", help="sync pod telemetry down while training and open the live dashboard locally")
    parser.add_argument("--sync-every", type=float, default=5.0, help="seconds between telemetry syncs with --watch")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true", help="with --watch: sync telemetry only; run tools.dashboard yourself")
    args = parser.parse_args(argv)
    if not args.host or not args.port:
        parser.error("pod address missing: pass --host/--port or set RUNPOD_SSH_HOST / RUNPOD_SSH_PORT (pods get a new address each start)")

    ssh_cmd = f"ssh -i {args.key} -p {args.port} -o StrictHostKeyChecking=no"

    print("Step 1: Syncing codebase and Spatial Vision architecture to RunPod...", flush=True)
    sync_up = [
        "rsync",
        "-avz",
        "-e",
        ssh_cmd,
        "--exclude",
        ".venv",
        "--exclude",
        ".git",
        "--exclude",
        "__pycache__",
        "--exclude",
        ".pytest_cache",
        "--exclude",
        ".DS_Store",
        "--exclude",
        "/logs/",  # old checkpoints/telemetry stay local; the pod trains from scratch unless --resume
        "--exclude",
        "/recordings/",
        "./",
        f"root@{args.host}:/workspace/tensormew",
    ]
    if run_cmd(sync_up) != 0:
        print("Failed to sync codebase to pod.", file=sys.stderr)
        return 1

    extra_flags = []
    if not args.curriculum:
        extra_flags.append("--no-curriculum")
    else:
        extra_flags.append("--curriculum")
    if not args.recurrent:
        extra_flags.append("--no-recurrent")
    else:
        extra_flags.append("--recurrent")
    if args.states:
        extra_flags.append(f"--states {args.states}")

    flags_str = " ".join(extra_flags)
    model_desc = "13.2M Recurrent Spatial Vision Policy" if args.recurrent else "11.6M Spatial Vision Policy"

    print(
        f"Step 2: Launching {args.updates} updates of {model_desc} on RTX 4090 GPU (curriculum={args.curriculum})...",
        flush=True,
    )
    train_remote = [
        "ssh",
        "-i",
        args.key,
        "-p",
        args.port,
        "-o",
        "StrictHostKeyChecking=no",
        f"root@{args.host}",
        f"cd /workspace/tensormew && PYTHONUNBUFFERED=1 python3 -m tools.train_deep_ppo --updates {args.updates} "
        f"--rollout {args.rollout} --max-steps {args.max_steps} --envs {args.envs} --batch-size {args.batch_size} --lr {args.lr} "
        f"--state {args.state} --save {args.save} {'--resume ' if args.resume else ''}"
        f"{f'--run-id {args.run_id} ' if args.run_id else ''}{flags_str}",
    ]

    stop = threading.Event()
    dashboard = None
    if args.watch:
        print(f"Watching: syncing pod telemetry every {args.sync_every:g}s and serving the dashboard locally...", flush=True)
        pull_logs(ssh_cmd, args.host, quiet=True)
        threading.Thread(target=watch_loop, args=(ssh_cmd, args.host, args.sync_every, stop), daemon=True).start()
        if not args.no_dashboard:
            dashboard = subprocess.Popen([sys.executable, "-m", "tools.dashboard", "--port", str(args.dashboard_port)])

    def train_and_sync() -> int:
        try:
            train_rc = run_cmd(train_remote)
        finally:
            stop.set()
            # Always bring the telemetry home, even if training failed or the run was interrupted.
            print("Syncing run history and live telemetry back to logs/ ...", flush=True)
            pull_logs(ssh_cmd, args.host, quiet=False)
        if train_rc != 0:
            print("Remote GPU training encountered an error.", file=sys.stderr)
            return 1

        print("Step 3: Syncing trained checkpoint back to local machine...", flush=True)
        sync_down = [
            "rsync",
            "-avz",
            "-e",
            ssh_cmd,
            f"root@{args.host}:/workspace/tensormew/{args.save}",
            f"./{args.save}",
        ]
        if run_cmd(sync_down) != 0:
            print("Failed to sync checkpoint back to local machine.", file=sys.stderr)
            return 1

        print(f"\n11.6M Parameter GPU Training complete! Checkpoint saved to {args.save}.", flush=True)
        return 0

    interrupted = False
    try:
        rc = train_and_sync()
    except KeyboardInterrupt:
        interrupted, rc = True, 130
    if dashboard is not None:
        # Keep the dashboard up after everything has synced so the finished run can be replayed.
        if interrupted:
            dashboard.terminate()
        else:
            print(f"Dashboard still running at http://127.0.0.1:{args.dashboard_port}/ (pick the run under Source to replay). Ctrl+C to stop.", flush=True)
            try:
                dashboard.wait()
            except KeyboardInterrupt:
                dashboard.terminate()
    return rc


if __name__ == "__main__":
    sys.exit(main())
