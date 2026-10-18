"""Best snapshot per map, capped so the buffer stays small."""

from __future__ import annotations

from collections.abc import Hashable


class FrontierBuffer:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._by_map: dict[Hashable, dict] = {}

    def add(self, state_bytes: bytes, map_id, badges, score) -> None:
        """Keep the higher score for this map. Evict the lowest score when full."""
        incoming = float(score)
        current = self._by_map.get(map_id)
        if current is not None:
            if incoming > current["meta"]["score"]:
                current["state"] = bytes(state_bytes)
                current["meta"]["badges"] = badges
                current["meta"]["score"] = incoming
            return
        if len(self._by_map) >= self.capacity:
            worst_id = min(self._by_map, key=lambda key: self._by_map[key]["meta"]["score"])
            if incoming <= self._by_map[worst_id]["meta"]["score"]:
                return
            del self._by_map[worst_id]
        self._by_map[map_id] = {
            "state": bytes(state_bytes),
            "meta": {"map": map_id, "badges": badges, "score": incoming},
        }

    def sample(self, rng) -> bytes | None:
        if not self._by_map:
            return None
        keys = list(self._by_map)
        return self._by_map[keys[_pick(rng, len(keys))]]["state"]

    def __len__(self) -> int:
        return len(self._by_map)

    @property
    def entries(self) -> list[dict]:
        return [
            {"state": item["state"], "meta": dict(item["meta"])}
            for item in self._by_map.values()
        ]


def _pick(rng, count: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(0, count))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(count))
    return int(rng.randint(0, count - 1))
