"""Keep exploring. Each time the map changes, save that arrival and continue."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.segment import SegmentResult, run_segment


def discovered_name(map_id: tuple[int, int]) -> str:
    return f"agent_map_{map_id[0]}_{map_id[1]}"


@dataclass
class ChainResult:
    results: list[SegmentResult] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    stopped: str = "budget"


def run_chain(
    machine,
    reader,
    supervisor,
    *,
    state_name: str,
    hops: int = 1,
    max_steps: int = 30,
    decision_log: str = "logs/decisions.jsonl",
    run_log=None,
    registry=None,
) -> ChainResult:
    """Explore until a new map, badge, or cleared Cut tree, then repeat."""
    chain = ChainResult()
    for _ in range(hops):
        result = run_segment(
            machine,
            reader,
            supervisor,
            max_steps=max_steps,
            decision_log=decision_log,
            run_log=run_log,
            state_name=state_name,
        )
        chain.results.append(result)
        if result.outcome != "advanced":
            chain.stopped = result.outcome
            break
        name = discovered_name(tuple(result.end_map))
        description = f"Arrived on map {tuple(result.end_map)} by walking."
        if registry is not None:
            registry.capture(
                machine,
                reader,
                name,
                description=description,
                tags=("route_test", "agent"),
                overwrite=True,
            )
        if run_log is not None and result.run_id is not None:
            run_log.set_checkpoint(result.run_id, name)
        chain.checkpoints.append(name)
        state_name = name
        chain.stopped = "advanced"
    return chain
