"""Run Spatial Vision PPO training on a remote RunPod GPU pod.

Syncs the codebase, checks the pod (GPU, cores, disk, python deps; installs pyboy if missing), runs the
vectorized training engine and syncs telemetry and the checkpoint back. With --detach the job runs under
setsid/tmux on the pod so a dropped SSH session does not kill it; --attach reconnects and resumes syncing.

Exit codes: 0 ok, 1 sync/general failure, 2 preflight or config failure, 3 remote training failed,
4 lost the remote job (it died without finishing), 130 interrupted.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
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

JOB_DIR = "/workspace/tensormew_job"  # outside the repo so code syncs never touch it
REMOTE_CONFIG = f"{REMOTE_ROOT}/.cloud_config.json"
PYBOY_SPEC = "pyboy==2.8.1"
MIN_DISK_GB = 5
REQUIRED_MODULES = ("torch", "numpy", "pyboy", "PIL")

EXIT_OK, EXIT_FAIL, EXIT_PREFLIGHT, EXIT_TRAIN, EXIT_LOST = 0, 1, 2, 3, 4

PREFLIGHT_SCRIPT = (
    'echo "GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"; '
    'echo "CORES=$(nproc 2>/dev/null)"; '
    'echo "DISK_GB=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc 0-9)"; '
    + " ".join(
        f'python3 -c "import {m}" >/dev/null 2>&1 && echo "DEP_{m}=ok" || echo "DEP_{m}=missing";'
        for m in REQUIRED_MODULES
    )
)


def run_cmd(cmd: list[str]) -> int:
    print(f"--> {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd)
    return res.returncode


def ssh_opts(key: str, port: str) -> list[str]:
    return ["-i", key, "-p", str(port), "-o", "StrictHostKeyChecking=no"]


def build_ssh_cmd(key: str, port: str, host: str, remote: str) -> list[str]:
    """Run one remote shell command line on the pod."""
    return ["ssh", *ssh_opts(key, port), "-o", "ServerAliveInterval=30", f"root@{host}", remote]


def build_pull_cmd(ssh_cmd: str, host: str, remote_rel: str, local_rel: str) -> list[str]:
    """rsync command that copies one telemetry path from the pod into the local logs/."""
    return ["rsync", "-az", "-e", ssh_cmd, f"root@{host}:{REMOTE_ROOT}/{remote_rel}", f"./{local_rel}"]


def remote_capture(key: str, port: str, host: str, remote: str, timeout: float = 60, stdin: str | None = None):
    """Run a remote command and return (returncode, stdout); (255, '') if ssh could not run or timed out."""
    try:
        res = subprocess.run(
            build_ssh_cmd(key, port, host, remote), capture_output=True, text=True, timeout=timeout, input=stdin
        )
    except (subprocess.TimeoutExpired, OSError):
        return 255, ""
    return res.returncode, res.stdout


def parse_preflight(output: str) -> dict:
    """Parse the KEY=VALUE lines printed by PREFLIGHT_SCRIPT."""
    kv = {}
    for line in output.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    info = {
        "gpu": kv.get("GPU") or None,
        "cores": int(kv["CORES"]) if kv.get("CORES", "").isdigit() else None,
        "disk_gb": int(kv["DISK_GB"]) if kv.get("DISK_GB", "").isdigit() else None,
        "deps": {m: kv.get(f"DEP_{m}") == "ok" for m in REQUIRED_MODULES},
    }
    info["missing"] = [m for m, ok in info["deps"].items() if not ok]
    return info


def build_install_cmd(spec: str = PYBOY_SPEC) -> str:
    # Ubuntu 24.04 images mark the system python externally managed, so plain pip refuses.
    return f"pip install --break-system-packages {shlex.quote(spec)}"


def cap_envs(envs: int, cores: int | None) -> tuple[int, str | None]:
    """Cap parallel emulators at the pod's core count. Returns (envs, warning or None)."""
    if cores and envs > cores:
        return cores, f"--envs {envs} exceeds the pod's {cores} cores; capping to {cores} (emulators are CPU-bound)"
    return envs, None


def preflight_problems(info: dict) -> list[str]:
    """Fatal problems only (pyboy is auto-installed before this runs)."""
    errs = []
    if not info["gpu"]:
        errs.append("no GPU visible (nvidia-smi failed); is this a GPU pod?")
    if info["disk_gb"] is not None and info["disk_gb"] < MIN_DISK_GB:
        errs.append(f"only {info['disk_gb']} GB free on /workspace (need >= {MIN_DISK_GB})")
    still = [m for m in info["missing"] if m != "pyboy"]
    if still:
        errs.append(f"python modules missing on the pod: {', '.join(still)} (use a PyTorch image)")
    return errs


def run_preflight(args) -> tuple[bool, int]:
    """Inspect the pod, install pyboy if needed, and return (ok, effective_envs)."""
    print("Preflight: checking GPU, cores, disk and python deps on the pod...", flush=True)
    rc, out = remote_capture(args.key, args.port, args.host, PREFLIGHT_SCRIPT)
    if not out.strip():
        print("Preflight failed: could not run commands over SSH (wrong address? pods get a new one each start).", file=sys.stderr)
        return False, args.envs
    info = parse_preflight(out)
    print(f"  GPU: {info['gpu']}  cores: {info['cores']}  free disk: {info['disk_gb']} GB", flush=True)
    if "pyboy" in info["missing"]:
        print(f"  pyboy missing; installing {PYBOY_SPEC} ...", flush=True)
        remote_capture(args.key, args.port, args.host, build_install_cmd(), timeout=600)
        _, out2 = remote_capture(args.key, args.port, args.host, 'python3 -c "import pyboy" && echo ok')
        if "ok" not in out2:
            print(f"Preflight failed: could not install {PYBOY_SPEC} on the pod.", file=sys.stderr)
            return False, args.envs
        info["missing"].remove("pyboy")
    problems = preflight_problems(info)
    for p in problems:
        print(f"Preflight failed: {p}", file=sys.stderr)
    envs, warn = cap_envs(args.envs, info["cores"])
    if warn:
        print(f"  WARNING: {warn}", flush=True)
    return not problems, envs


def build_remote_train_cmd(args, envs: int, config_path: str | None = None) -> str:
    """The one place that builds the remote trainer command line. Adjust flags here."""
    parts = [
        "PYTHONUNBUFFERED=1", "python3", "-m", "tools.train_deep_ppo",
        "--updates", str(args.updates), "--rollout", str(args.rollout), "--max-steps", str(args.max_steps),
        "--envs", str(envs), "--batch-size", str(args.batch_size), "--lr", str(args.lr),
        "--state", args.state, "--save", args.save,
        "--curriculum" if args.curriculum else "--no-curriculum",
        "--recurrent" if args.recurrent else "--no-recurrent",
    ]
    if args.states:
        parts += ["--states", args.states]
    if args.resume:
        parts.append("--resume")
    if args.run_id:
        parts += ["--run-id", args.run_id]
    if config_path:
        parts += ["--config", config_path]
    return f"cd {REMOTE_ROOT} && " + " ".join(shlex.quote(p) if not p.startswith("PYTHONUNBUFFERED=") else p for p in parts)


def build_runner_script(train_cmd: str, job_dir: str = JOB_DIR) -> str:
    """Bash script for the pod: logs everything, records its pid, and writes the exit code to a done marker."""
    return (
        "#!/bin/bash\n"
        f"exec >> {job_dir}/train.log 2>&1\n"
        f"echo $$ > {job_dir}/pid\n"
        'echo "=== job start $(date -u +%FT%TZ) ==="\n'
        f"bash -c {shlex.quote(train_cmd)}\n"
        "rc=$?\n"
        'echo "=== job end rc=$rc $(date -u +%FT%TZ) ==="\n'
        f"echo $rc > {job_dir}/done\n"
        "exit $rc\n"
    )


def build_install_runner_cmd(job_dir: str = JOB_DIR) -> str:
    """Remote command that stores the runner script (read from stdin) and clears the previous job's markers."""
    return f"mkdir -p {job_dir} && rm -f {job_dir}/done {job_dir}/pid && cat > {job_dir}/run.sh && chmod +x {job_dir}/run.sh"


def build_launch_cmd(job_dir: str = JOB_DIR) -> str:
    """Start run.sh detached: tmux if present, otherwise setsid nohup."""
    return (
        "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t train 2>/dev/null; "
        f"tmux new-session -d -s train 'bash {job_dir}/run.sh'; "
        f"else setsid nohup bash {job_dir}/run.sh >/dev/null 2>&1 </dev/null & fi; sleep 1; echo launched"
    )


def build_status_cmd(job_dir: str = JOB_DIR) -> str:
    return (
        f"if [ -f {job_dir}/done ]; then echo done:$(cat {job_dir}/done); "
        f"elif [ -f {job_dir}/pid ] && kill -0 $(cat {job_dir}/pid) 2>/dev/null; then echo running; "
        f"elif [ -f {job_dir}/run.sh ]; then echo dead; else echo none; fi"
    )


def parse_status(out: str) -> tuple[str, int | None]:
    """Return (state, rc): state in running/done/dead/none/unknown."""
    out = out.strip().splitlines()[-1].strip() if out.strip() else ""
    if out.startswith("done:"):
        tail = out[5:].strip()
        return "done", int(tail) if tail.lstrip("-").isdigit() else 1
    if out in ("running", "dead", "none"):
        return out, None
    return "unknown", None


def build_tail_cmd(lines: int = 5, job_dir: str = JOB_DIR) -> str:
    return f"tail -n {lines} {job_dir}/train.log 2>/dev/null"


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


def poll_detached(args, ssh_cmd: str, max_failures: int = 120) -> int:
    """Poll the detached job until it finishes; sync telemetry as we go. Returns an exit code."""
    failures = 0
    last_tail = ""
    while True:
        rc, out = remote_capture(args.key, args.port, args.host, build_status_cmd(), timeout=30)
        state, job_rc = parse_status(out) if rc == 0 else ("unknown", None)
        if state == "unknown":
            failures += 1
            if failures == 1 or failures % 10 == 0:
                print(f"  (cannot reach pod, retrying; {failures} failed polls. The job keeps running on the pod; use --attach later)", flush=True)
            if failures >= max_failures:
                print("Giving up: pod unreachable. Re-run with --attach once it is back (new address if restarted).", file=sys.stderr)
                return EXIT_LOST
        else:
            failures = 0
            if args.watch:
                pull_logs(ssh_cmd, args.host, quiet=True)
            _, tail = remote_capture(args.key, args.port, args.host, build_tail_cmd(1), timeout=30)
            tail = tail.strip()
            if tail and tail != last_tail:
                print(f"  [{state}] {tail[-200:]}", flush=True)
                last_tail = tail
            if state == "done":
                return EXIT_OK if job_rc == 0 else EXIT_TRAIN
            if state in ("dead", "none"):
                print("Remote job is not running and left no done marker (killed or pod restarted).", file=sys.stderr)
                return EXIT_LOST
        time.sleep(max(args.sync_every, 1.0))


def pull_checkpoint(args, ssh_cmd: str) -> int:
    print("Syncing trained checkpoint back to local machine...", flush=True)
    sync_down = ["rsync", "-avz", "-e", ssh_cmd, f"root@{args.host}:{REMOTE_ROOT}/{args.save}", f"./{args.save}"]
    if run_cmd(sync_down) != 0:
        print("Failed to sync checkpoint back to local machine.", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


def resolve_config(args) -> tuple[str | None, bool]:
    """Validate --config locally. Returns (config arg for the remote trainer or None, needs_upload)."""
    if not args.config:
        return None, False
    from src.train_config import PRESETS, ConfigError, load_config

    try:
        load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_PREFLIGHT)
    if args.config in PRESETS:
        return args.config, False
    return REMOTE_CONFIG, True


def build_sync_up_cmd(ssh_cmd: str, host: str) -> list[str]:
    return [
        "rsync", "-avz", "-e", ssh_cmd,
        "--exclude", ".venv", "--exclude", ".git", "--exclude", "__pycache__",
        "--exclude", ".pytest_cache", "--exclude", ".DS_Store",
        "--exclude", "/logs/",  # old checkpoints/telemetry stay local; the pod trains from scratch unless --resume
        "--exclude", "/recordings/",
        "./", f"root@{host}:{REMOTE_ROOT}",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cloud training of the Spatial Vision Policy on a RunPod GPU")
    parser.add_argument("--host", default=DEFAULT_POD_HOST, help="pod SSH host (or set RUNPOD_SSH_HOST)")
    parser.add_argument("--port", default=DEFAULT_POD_PORT, help="pod SSH port (or set RUNPOD_SSH_PORT)")
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--envs", type=int, default=16, help="parallel emulators on pod (capped to the pod's core count)")
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
    parser.add_argument("--config", default="", help="training config: preset name (blueprint_v1, legacy) or a JSON file; passed to the remote trainer")
    parser.add_argument("--detach", action="store_true", help="run the job under tmux/setsid on the pod so it survives a dropped SSH session, then poll")
    parser.add_argument("--attach", action="store_true", help="reattach to a running detached job: resume telemetry sync, then pull results")
    parser.add_argument("--skip-preflight", action="store_true", help="skip the pod checks and dependency install")
    parser.add_argument("--watch", action="store_true", help="sync pod telemetry down while training and open the live dashboard locally")
    parser.add_argument("--sync-every", type=float, default=5.0, help="seconds between telemetry syncs with --watch")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true", help="with --watch: sync telemetry only; run tools.dashboard yourself")
    args = parser.parse_args(argv)
    if not args.host or not args.port:
        parser.error("pod address missing: pass --host/--port or set RUNPOD_SSH_HOST / RUNPOD_SSH_PORT (pods get a new address each start)")
    if args.attach and args.detach:
        parser.error("--attach and --detach are mutually exclusive")

    config_arg, upload_config = resolve_config(args)
    ssh_cmd = "ssh " + " ".join(ssh_opts(args.key, args.port))
    envs = args.envs

    if not args.attach:
        print("Step 1: Syncing codebase and Spatial Vision architecture to RunPod...", flush=True)
        if run_cmd(build_sync_up_cmd(ssh_cmd, args.host)) != 0:
            print("Failed to sync codebase to pod.", file=sys.stderr)
            return EXIT_FAIL
        if upload_config:
            if run_cmd(["rsync", "-az", "-e", ssh_cmd, args.config, f"root@{args.host}:{REMOTE_CONFIG}"]) != 0:
                print("Failed to upload the config file.", file=sys.stderr)
                return EXIT_FAIL
        if not args.skip_preflight:
            ok, envs = run_preflight(args)
            if not ok:
                return EXIT_PREFLIGHT

    train_cmd = build_remote_train_cmd(args, envs, config_arg)
    model_desc = "13.2M Recurrent Spatial Vision Policy" if args.recurrent else "11.6M Spatial Vision Policy"

    stop = threading.Event()
    dashboard = None
    if args.watch:
        print(f"Watching: syncing pod telemetry every {args.sync_every:g}s and serving the dashboard locally...", flush=True)
        pull_logs(ssh_cmd, args.host, quiet=True)
        if not args.detach and not args.attach:  # detached modes sync inside the poll loop
            threading.Thread(target=watch_loop, args=(ssh_cmd, args.host, args.sync_every, stop), daemon=True).start()
        if not args.no_dashboard:
            dashboard = subprocess.Popen([sys.executable, "-m", "tools.dashboard", "--port", str(args.dashboard_port)])

    def finish(train_rc: int) -> int:
        stop.set()
        # Always bring the telemetry home, even if training failed or the run was interrupted.
        print("Syncing run history and live telemetry back to logs/ ...", flush=True)
        pull_logs(ssh_cmd, args.host, quiet=False)
        if train_rc == EXIT_LOST:
            return EXIT_LOST
        if train_rc != 0:
            print("Remote GPU training encountered an error.", file=sys.stderr)
            return EXIT_TRAIN
        rc = pull_checkpoint(args, ssh_cmd)
        if rc == 0:
            print(f"\nGPU training complete! Checkpoint saved to {args.save}.", flush=True)
        return rc

    def start_detached() -> int:
        _, out = remote_capture(args.key, args.port, args.host, build_status_cmd(), timeout=30)
        if parse_status(out)[0] == "running":
            print("A detached job is already running on this pod. Use --attach to follow it.", file=sys.stderr)
            return EXIT_FAIL
        rc, _ = remote_capture(args.key, args.port, args.host, build_install_runner_cmd(), stdin=build_runner_script(train_cmd))
        if rc != 0:
            print("Failed to install the runner script on the pod.", file=sys.stderr)
            return EXIT_FAIL
        rc, out = remote_capture(args.key, args.port, args.host, build_launch_cmd(), timeout=60)
        if rc != 0 or "launched" not in out:
            print("Failed to launch the detached job.", file=sys.stderr)
            return EXIT_FAIL
        print(f"Detached job started (log: {JOB_DIR}/train.log on the pod). Safe to close this terminal; "
              "reattach with --attach.", flush=True)
        return EXIT_OK

    interrupted = False
    try:
        if args.attach:
            _, out = remote_capture(args.key, args.port, args.host, build_status_cmd(), timeout=30)
            state, _rc = parse_status(out)
            if state in ("none", "unknown"):
                print("No detached job found on the pod (or pod unreachable).", file=sys.stderr)
                return EXIT_FAIL
            print(f"Attached to remote job (state: {state}).", flush=True)
            rc = finish(poll_detached(args, ssh_cmd))
        elif args.detach:
            print(f"Step 2: Launching {args.updates} updates of {model_desc} detached on the pod (envs={envs})...", flush=True)
            rc = start_detached()
            if rc == 0:
                rc = finish(poll_detached(args, ssh_cmd))
        else:
            print(f"Step 2: Launching {args.updates} updates of {model_desc} on the pod (envs={envs}, curriculum={args.curriculum})...", flush=True)
            try:
                train_rc = run_cmd(build_ssh_cmd(args.key, args.port, args.host, train_cmd))
            finally:
                stop.set()
            rc = finish(train_rc)
    except KeyboardInterrupt:
        interrupted, rc = True, 130
        if args.detach or args.attach:
            print("\nInterrupted. The remote job keeps running; reattach with --attach.", flush=True)
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
