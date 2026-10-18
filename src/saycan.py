"""Party affordance check. Not a damage calculator."""

import importlib.util


def metamon_available() -> bool:
    return importlib.util.find_spec("metamon") is not None


def evaluate(party: dict, gym: str) -> dict:
    count = int(party.get("party_count") or 0)
    levels = party.get("levels") or []
    highest = max(levels or [0])
    if count < 1 or highest < 10:
        return {
            "verdict": "VETO",
            "reason": f"party cannot attempt {gym}: count {count}, highest level {highest}",
        }
    return {
        "verdict": "APPROVE",
        "reason": f"party of {count} at level {highest} can attempt {gym}",
    }
