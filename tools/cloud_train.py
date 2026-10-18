"""Run high-throughput multi-worker PPO training on the remote RunPod instance.

Syncs code to the pod, executes training across parallel emulators, and syncs
the trained checkpoint back to the local logs/ directory.
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
    parser = argparse.ArgumentParser(description="High-throughput cloud training on RunPod")
    parser.add_argument("--host", default=DEFAULT_POD_HOST)
    parser.add_argument("--port", default=DEFAULT_POD_PORT)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--envs", type=int, default=12, help="parallel emulators on pod")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--state", default="new_bark_outside")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--save", default="logs/ppo_policy.npz")
    args = parser.parse_args(argv)

    ssh_cmd = f"ssh -i {args.key} -p {args.port} -o StrictHostKeyChecking=no"

    print("Step 1: Syncing codebase to RunPod instance...", flush=True)
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

    print(f"Step 2: Launching {args.updates} updates across {args.envs} parallel emulators...", flush=True)
    train_remote = [
        "ssh",
        "-i",
        args.key,
        "-p",
        args.port,
        "-o",
        "StrictHostKeyChecking=no",
        f"root@{args.host}",
        f"cd /workspace/tensormew && python3 -m tools.train_ppo --mlp --hidden-dim {args.hidden_dim} "
        f"--updates {args.updates} --rollout {args.rollout} --envs {args.envs} --batch-size {args.batch_size} "
        f"--state {args.state} --save {args.save} --resume",
    ]
    if run_cmd(train_remote) != 0:
        print("Remote training encountered an error.", file=sys.stderr)
        return 1

    print("Step 3: Syncing updated checkpoint back to local machine...", flush=True)
    sync_down = [
        "rsync",
        "-avz",
        "-e",
        ssh_cmd,
        f"root@{args.host}:/workspace/tensormew/{args.save}",
        f"./{args.save}",
    ]
    if run_cmd(sync_down) != 0:
        print("Failed to sync weights back to local machine.", file=sys.stderr)
        return 1

    print(f"Training complete! Checkpoint saved locally to {args.save}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
