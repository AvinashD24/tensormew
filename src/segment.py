"""Play one checkpoint forward until the map, badges, or a Cut tree changes."""

from __future__ import annotations

from dataclasses import dataclass

from src.action_results import ActionResult
from src.agent_loop import act, append_decision
from src.collision import is_cut_tree
from src.llm_supervisor import Decision
from src.mover import Mover, walkable
from src.supervisor_stubs import STUB_REGISTRY
from src.tool_dispatch import execution_registry
from src.triggers import evaluate

_DIRS = ("up", "down", "left", "right")


@dataclass
class SegmentResult:
    outcome: str  # "advanced" | "budget" | "stuck" | "error"
    steps: int
    start_map: tuple
    end_map: tuple
    start_badges: int
    end_badges: int
    cut_cleared: bool
    log_path: str


def run_segment(
    machine,
    reader,
    supervisor,
    *,
    max_steps: int = 30,
    decision_log: str = "logs/decisions.jsonl",
) -> SegmentResult:
    """Step until the map or badges change, a starting Cut tree is gone, or we stop."""
    log_path = str(decision_log)
    mover = _prepare_mover(machine, reader, supervisor)
    start_map: tuple = ()
    end_map: tuple = ()
    start_badges = 0
    end_badges = 0
    steps = 0
    try:
        state = reader.parse(getattr(machine, "frame", 0))
        start_map = tuple(state.map_id)
        end_map = start_map
        start_badges = int(state.badges)
        end_badges = start_badges
        started_with_cut = _adjacent_cut(state)
        streak_kind: str | None = None
        streak = 0
        while steps < max_steps:
            end_map = tuple(state.map_id)
            end_badges = int(state.badges)
            cut_cleared = _cut_cleared(started_with_cut, state)
            if _advanced(state, start_map, start_badges, cut_cleared):
                return _finish(
                    "advanced",
                    steps,
                    start_map,
                    end_map,
                    start_badges,
                    end_badges,
                    cut_cleared,
                    log_path,
                )
            if mover.stuck >= mover.stuck_limit:
                return _finish(
                    "stuck", steps, start_map, end_map, start_badges, end_badges, False, log_path
                )

            trigger = evaluate(state, stuck_count=mover.stuck)
            if trigger is None and not getattr(supervisor, "always_on", False):
                streak_kind = None
                streak = 0
                direction = _walkable_direction(state)
                if direction is None:
                    mover.stuck += 1
                else:
                    path_result = mover.run_path([direction])
                    _log_script(decision_log, machine, state, direction, path_result)
                steps += 1
            else:
                kind = _trigger_kind(trigger) if trigger is not None else ""
                if kind and kind == streak_kind:
                    streak += 1
                elif kind:
                    streak_kind = kind
                    streak = 1
                else:
                    streak_kind = None
                    streak = 0
                act(
                    supervisor,
                    machine,
                    reader,
                    actor="llm",
                    stuck_count=mover.stuck,
                    decision_log=decision_log,
                )
                steps += 1

            state = reader.parse(getattr(machine, "frame", 0))
            end_map = tuple(state.map_id)
            end_badges = int(state.badges)
            cut_cleared = _cut_cleared(started_with_cut, state)
            if _advanced(state, start_map, start_badges, cut_cleared):
                return _finish(
                    "advanced",
                    steps,
                    start_map,
                    end_map,
                    start_badges,
                    end_badges,
                    cut_cleared,
                    log_path,
                )
            if streak >= 3 or mover.stuck >= mover.stuck_limit:
                return _finish(
                    "stuck", steps, start_map, end_map, start_badges, end_badges, False, log_path
                )
        return _finish(
            "budget", steps, start_map, end_map, start_badges, end_badges, False, log_path
        )
    except Exception:
        return _finish(
            "error", steps, start_map, end_map, start_badges, end_badges, False, log_path
        )


def _prepare_mover(machine, reader, supervisor) -> Mover:
    registry = getattr(supervisor, "registry", None)
    bound = _mover_from_registry(registry)
    if bound is not None:
        return bound
    mover = Mover(machine, reader)
    if _registry_unset(registry):
        supervisor.registry = execution_registry(machine, reader, mover=mover)
    return mover


def _mover_from_registry(registry) -> Mover | None:
    if not registry:
        return None
    owner = getattr(registry.get("run_path"), "__self__", None)
    if isinstance(owner, Mover):
        return owner
    return None


def _registry_unset(registry) -> bool:
    if not registry:
        return True
    return all(registry.get(name) is fn for name, fn in STUB_REGISTRY.items())


def _adjacent_cut(state) -> bool:
    tiles = getattr(state, "tiles", None) or {}
    return any(is_cut_tree(int(tile)) for tile in tiles.values())


def _cut_cleared(started_with_cut: bool, state) -> bool:
    return started_with_cut and not _adjacent_cut(state)


def _advanced(state, start_map: tuple, start_badges: int, cut_cleared: bool) -> bool:
    return tuple(state.map_id) != tuple(start_map) or int(state.badges) != start_badges or cut_cleared


def _walkable_direction(state) -> str | None:
    facing = getattr(state, "facing", None)
    order: list[str] = []
    if facing in _DIRS:
        order.append(facing)
    order.extend(direction for direction in _DIRS if direction not in order)
    tiles = getattr(state, "tiles", None) or {}
    for direction in order:
        if walkable(int(tiles.get(direction, 0xFF))):
            return direction
    return None


def _trigger_kind(trigger) -> str:
    kind = getattr(trigger, "kind", trigger)
    return str(getattr(kind, "value", kind))


def _log_script(decision_log, machine, state, direction: str, result: ActionResult) -> None:
    if decision_log is None:
        return
    decision = Decision(
        name="run_path",
        arguments={"directions": [direction]},
        reason="segment script",
        model="script",
        input_tokens=0,
        output_tokens=0,
        est_usd=0.0,
    )
    append_decision(
        decision_log,
        state=state,
        frame=getattr(machine, "frame", 0),
        actor="script",
        trigger=None,
        decision=decision,
        result=result,
    )


def _finish(
    outcome: str,
    steps: int,
    start_map: tuple,
    end_map: tuple,
    start_badges: int,
    end_badges: int,
    cut_cleared: bool,
    log_path: str,
) -> SegmentResult:
    return SegmentResult(
        outcome=outcome,
        steps=steps,
        start_map=tuple(start_map),
        end_map=tuple(end_map),
        start_badges=start_badges,
        end_badges=end_badges,
        cut_cleared=cut_cleared,
        log_path=log_path,
    )
