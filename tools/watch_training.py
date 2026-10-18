"""Real-time terminal monitor for deep RL training.

Usage:
    python -m tools.watch_training
    python -m tools.watch_training --tail 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

CANDIDATE_LOGS = [
    Path("logs/training.log"),
    Path("/Users/shrey/.gemini/antigravity-ide/brain/1369e291-1f53-4e41-b038-cda0fcd72bcd/.system_generated/tasks/task-6304.log"),
]


def find_log_path() -> Path | None:
    for p in CANDIDATE_LOGS:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def is_table_line(line: str) -> bool:
    line_s = line.strip()
    if not line_s:
        return False
    # Show table borders, headers, and update rows
    if line_s.startswith("=") or line_s.startswith("-") or "Upd |" in line_s:
        return True
    # Rows start with digits (the update index) followed by '|'
    parts = line_s.split("|")
    if len(parts) >= 6 and parts[0].strip().isdigit():
        return True
    if "Training completed!" in line_s or "Initializing" in line_s or "Spawning" in line_s:
        return True
    return False



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Live monitor for RL training metrics")
    parser.add_argument("--tail", type=int, default=25, help="number of recent table lines to show initially")
    parser.add_argument("--no-follow", action="store_true", help="exit after printing current history")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    log_path = find_log_path()
    if log_path is None:
        print("Waiting for training log to be created...", flush=True)
        while log_path is None:
            time.sleep(1.0)
            log_path = find_log_path()

    print(f"Tracking RL training from {log_path} (Pod: tensormew-cpu-train / f4m58oyp37mg7m)...", flush=True)

    # Initial history display
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    table_lines = [line for line in lines if is_table_line(line)]
    for line in table_lines[-args.tail:]:
        sys.stdout.write(line)
    sys.stdout.flush()

    if args.no_follow:
        return 0

    # Live follow loop
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)  # Seek to end
        while True:
            line = f.readline()
            if line:
                if is_table_line(line):
                    sys.stdout.write(line)
                    sys.stdout.flush()
            else:
                time.sleep(0.5)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
