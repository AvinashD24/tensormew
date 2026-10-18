"""Run high-throughput 11.6M Spatial Vision PPO training on the remote RunPod RTX 4090.

Syncs codebase, runs the vectorized GPU training engine across 12 parallel emulators,
and syncs the trained 11.6M model checkpoint back to logs/deep_policy.pt.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_POD_HOST = "69.145.85.83"
DEFAULT_POD_PORT = "18602"
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_runpod")


def run_cmd(cmd: list[str]) -> int:
    print(f"--> {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd)
    return res.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cloud training of 11.6M Vision Policy on RTX 4090")
    parser.add_argument("--host", default=DEFAULT_POD_HOST)
    parser.add_argument("--port", default=DEFAULT_POD_PORT)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--envs", type=int, default=8, help="parallel emulators on pod (matched to dedicated cores)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--state", default="route29_entry")
    parser.add_argument("--save", default="logs/deep_policy.pt")
    args = parser.parse_args(argv)

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
        "./",
        f"root@{args.host}:/workspace/tensormew",
    ]
    if run_cmd(sync_up) != 0:
        print("Failed to sync codebase to pod.", file=sys.stderr)
        return 1

    print(
        f"Step 2: Launching {args.updates} updates of 11.6M Spatial Vision Policy on RTX 4090 GPU...",
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
        f"--rollout {args.rollout} --envs {args.envs} --batch-size {args.batch_size} --lr {args.lr} "
        f"--state {args.state} --save {args.save} --resume",
    ]
    if run_cmd(train_remote) != 0:
        print("Remote GPU training encountered an error.", file=sys.stderr)
        return 1

    print("Step 3: Syncing trained 11.6M checkpoint back to local machine...", flush=True)
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


if __name__ == "__main__":
    sys.exit(main())
