"""Sync trained model weights from logs/ to models/ and push to GitHub.

Enforces that any retraining or updated weights are publicly committed and tracked
in the repository so all team members have access to the latest checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

WEIGHT_PAIRS = [
    ("logs/deep_policy.pt", "models/deep_policy.pt"),
    ("logs/ppo_policy.npz", "models/ppo_policy.npz"),
]


def file_md5(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_and_push(
    push: bool = True,
    message: str | None = None,
    custom_pairs: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Sync weights from logs/ to models/ and optionally git push them to GitHub.

    Returns the list of updated model names.
    """
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    pairs = custom_pairs or WEIGHT_PAIRS
    updated: list[str] = []

    for src_rel, dst_rel in pairs:
        src = repo_root / src_rel
        dst = repo_root / dst_rel

        if not src.is_file():
            continue

        src_hash = file_md5(src)
        dst_hash = file_md5(dst)

        if src_hash != dst_hash:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            updated.append(dst_rel)
            print(f"[sync_weights] Updated {dst_rel} from {src_rel} (MD5: {src_hash[:8]}...)", flush=True)

    if not updated:
        print("[sync_weights] All model weights in models/ are already up to date with logs/.", flush=True)
        return []

    if not push:
        print(f"[sync_weights] Copied {len(updated)} weight file(s) locally. Push skipped (--no-push).", flush=True)
        return updated

    # Git pull, stage, commit, and push
    commit_msg = message or f"feat(weights): update {', '.join(Path(u).name for u in updated)} from retraining"
    try:
        # Pull latest first
        subprocess.run(["git", "fetch", "origin"], check=True, capture_output=True)
        subprocess.run(["git", "pull", "origin", "main"], check=True, capture_output=True)

        # Stage updated models
        for u in updated:
            subprocess.run(["git", "add", u], check=True)

        # Commit
        res = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
        if res.returncode != 0 and "nothing to commit" in res.stdout:
            print("[sync_weights] No changes to commit.", flush=True)
            return updated

        # Pull again in case remote advanced
        subprocess.run(["git", "pull", "origin", "main"], check=True, capture_output=True)

        # Push to origin
        print(f"[sync_weights] Pushing {', '.join(updated)} to GitHub origin/main...", flush=True)
        push_res = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True, check=True)
        print("[sync_weights] Successfully pushed updated weights to GitHub!", flush=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        print(f"[sync_weights] Warning: Git operation failed: {stderr}", file=sys.stderr, flush=True)

    return updated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sync weights to models/ and push to GitHub")
    parser.add_argument("--no-push", action="store_true", help="copy weights to models/ without pushing")
    parser.add_argument("-m", "--message", help="custom git commit message")
    args = parser.parse_args(argv)

    sync_and_push(push=not args.no_push, message=args.message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
