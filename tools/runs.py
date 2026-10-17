"""Inspect the local run database."""

import argparse
import sys

from src.run_log import DEFAULT_DB, RunLog


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Query logs/runs.db")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("progress")
    recent = sub.add_parser("recent")
    recent.add_argument("--limit", type=int, default=15)
    fails = sub.add_parser("failures")
    fails.add_argument("--limit", type=int, default=30)
    show = sub.add_parser("show")
    show.add_argument("run_id", type=int)
    args = parser.parse_args(argv)

    log = RunLog(args.db)
    try:
        if args.cmd == "progress":
            stats = log.progress()
            print(
                f"runs={stats['runs']} advanced={stats['advanced']} "
                f"stuck={stats['stuck']} errors={stats['errors']} "
                f"max_badges={stats['max_badges']}"
            )
            return 0
        if args.cmd == "recent":
            for row in log.recent_runs(args.limit):
                print(
                    f"#{row['id']} {row['outcome']} state={row['state_name']} "
                    f"badges={row['start_badges']}->{row['end_badges']} "
                    f"map=({row['start_map_group']},{row['start_map_number']})->"
                    f"({row['end_map_group']},{row['end_map_number']}) "
                    f"steps={row['steps']} {row['error'] or ''}"
                )
            return 0
        if args.cmd == "failures":
            rows = log.failed_steps(args.limit)
            if not rows:
                print("no failed steps")
                return 0
            for row in rows:
                print(
                    f"run#{row['run_id']} {row['state_name']} {row['actor']} "
                    f"tool={row['tool']} trigger={row['trigger']} "
                    f"reason={row['reason']}"
                )
            return 0
        for row in log.run_steps(args.run_id):
            print(
                f"{row['actor']} tool={row['tool']} ok={row['ok']} "
                f"map=({row['map_group']},{row['map_number']}) "
                f"pos=({row['x']},{row['y']}) trigger={row['trigger']} "
                f"reason={row['reason']}"
            )
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
