"""Boss lookup from data/knowledge/bosses.json. No boss stats live in Python."""

import json
from pathlib import Path

_DEFAULT_BOSSES = Path(__file__).resolve().parent.parent / "data" / "knowledge" / "bosses.json"


def lookup_boss(name: str, path=None) -> dict | None:
    target = _DEFAULT_BOSSES if path is None else Path(path)
    if not target.is_file():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(raw, dict):
        return None
    boss = raw.get(name)
    if not isinstance(boss, dict):
        return None
    return boss
