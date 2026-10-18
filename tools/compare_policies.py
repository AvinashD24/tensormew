"""Compare policies from finished live runs. Usage: python -m tools.compare_policies [--prefix w] [--kinds new,og].

Reads logs/live_status/<prefix><wave><kind><n>.json (tags like w1new3) and prints, per policy kind, how many runs
reached each milestone and the median step they reached it at.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

from src.live_status import MILESTONES, STATUS_DIR


def load(prefix: str, kinds: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {k: [] for k in kinds}
    for path in sorted(STATUS_DIR.glob(f"{prefix}*.json")):
        m = re.fullmatch(rf"{re.escape(prefix)}\d+({'|'.join(kinds)})\d+", path.stem)
        if not m:
            continue
        try:
            out[m.group(1)].append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare policies across finished live runs")
    parser.add_argument("--prefix", default="w")
    parser.add_argument("--kinds", default="new,og")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    runs = load(args.prefix, kinds)

    print(f"{'':32s}" + "".join(f"{k + ' reached':>14s}{k + ' med step':>14s}" for k in kinds))
    for key, label in MILESTONES:
        row = f"{label:32s}"
        for k in kinds:
            steps = [dict(r.get("milestones", [])).get(key) for r in runs[k]]
            hit = [s for s in steps if s is not None]
            med = f"{statistics.median(hit):.0f}" if hit else "-"
            row += f"{f'{len(hit)}/{len(runs[k])}':>14s}{med:>14s}"
        print(row)
    print()
    for k in kinds:
        rs = runs[k]
        if not rs:
            continue
        steps = [r["step"] for r in rs]
        print(
            f"{k}: runs={len(rs)} median final step={statistics.median(steps):.0f} "
            f"median return={statistics.median(r['return'] for r in rs):.0f} "
            f"heals={sum(r.get('heals', 0) for r in rs)} blackouts={sum(r.get('blackouts', 0) for r in rs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
