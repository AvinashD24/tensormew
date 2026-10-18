"""List the checkpoints that exist. Does not invent a 25-town script."""

import argparse
import json
import sys
from pathlib import Path


def campaign_names(manifest: dict) -> list[str]:
    return sorted(manifest.get("states", {}))


def summarize(rows: list[dict]) -> dict:
    left = sum(1 for row in rows if row.get("end_map") != row.get("start_map"))
    advanced = sum(1 for row in rows if row.get("outcome") == "advanced")
    return {"ran": len(rows), "advanced": advanced, "left_map": left}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Index saved checkpoints")
    parser.add_argument("--manifest", default="states/manifest.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    names = campaign_names(manifest)
    for name in names:
        print(name)
    print(f"checkpoints={len(names)} ran=0")
    if args.execute:
        print("execute requires a policy run; not started")
    return 0


if __name__ == "__main__":
    sys.exit(main())
