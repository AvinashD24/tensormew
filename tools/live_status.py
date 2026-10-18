"""Show where every running play_live window is in the story. Usage: python -m tools.live_status [--watch 5]."""

from __future__ import annotations

import argparse
import json
import sys
import time

from src.live_status import CHECKLIST, STATUS_DIR

SHORT = {
    "starter": "start",
    "egg from Mr. Pokémon": "egg",
    "egg given to Elm": "gave",
    "talked to Mom": "mom",
    "Falkner beaten": "falk",
}


def render() -> str:
    rows = []
    for path in sorted(STATUS_DIR.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if not rows:
        return "no live runs found (start tools.play_live first)"
    header = f"{'run':8s} {'step':>5s} {'map':24s} {'pos':>8s} {'ret':>8s} {'pty':>3s} " + " ".join(
        f"{SHORT[l]:>5s}" for l, _ in CHECKLIST
    ) + f" {'rival':>6s}  next"
    lines = [header, "-" * len(header)]
    now = time.time()
    for r in rows:
        stale = "" if now - r["updated"] < 120 else "  (stale)"
        marks = " ".join(f"{('yes' if r['done'].get(l) else '-'):>5s}" for l, _ in CHECKLIST) + f" {r.get('rival', '-'):>6s}"
        egg = "*egg*" if r.get("egg_in_party") else ""
        lines.append(
            f"{r['tag']:8s} {r['step']:5d} {r['map']:24s} {str(tuple(r['pos'])):>8s} {r['return']:8.1f} "
            f"{r['party']:3d} {marks}  {r['next']} {egg}{stale}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize live play_live runs")
    parser.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    while True:
        print(render(), flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
