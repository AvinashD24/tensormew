"""Play one checkpoint forward until the map, badges, or a Cut tree changes."""

from __future__ import annotations

from dataclasses import dataclass

from src.action_results import ActionResult
from src.agent_loop import act, append_decision
from src.collision import is_cut_tree
from src.llm_supervisor import Decision
from src.explore import Explorer
from src.mover import Mover, _reducing_dirs, walkable
from src.rl_env import ACTIONS, observe
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
    run_id: int | None = None


def run_segment(
    machine,
    reader,
    supervisor,
    *,
    max_steps: int = 30,
    decision_log: str = "logs/decisions.jsonl",
    target_map: tuple[int, int] | None = None,
    target_xy: tuple[int, int] | None = None,
    run_log=None,
    state_name: str = "",
    policy=None,
) -> SegmentResult:
    """Step until the target map or tile, a badge, or a starting Cut tree changes."""
    log_path = str(decision_log)
    mover = _prepare_mover(machine, reader, supervisor)
    explorer = Explorer()
    visited: set[tuple[int, int, int, int]] = set()
    previous = None
    last_direction = None
    last_moved = False
    start_map: tuple = ()
    end_map: tuple = ()
    start_badges = 0
    end_badges = 0
    steps = 0
    run_id = None
    try:
        state = reader.parse(getattr(machine, "frame", 0))
        start_map = tuple(state.map_id)
        end_map = start_map
        start_badges = int(state.badges)
        end_badges = start_badges
        if run_log is not None:
            run_id = run_log.start_run(
                state_name=state_name,
                start_map=start_map,
                start_badges=start_badges,
            )
        started_with_cut = _adjacent_cut(state)
        streak_kind: str | None = None
        streak = 0
        while steps < max_steps:
            if previous is not None and last_direction is not None:
                explorer.note(previous, state, last_direction, last_moved)
            end_map = tuple(state.map_id)
            end_badges = int(state.badges)
            cut_cleared = _cut_cleared(started_with_cut, state)
            if _advanced(state, start_map, start_badges, cut_cleared, target_map, target_xy, steps):
                return _close(
                    run_log, run_id, "advanced", steps, start_map, end_map,
                    start_badges, end_badges, cut_cleared, log_path,
                )
            if mover.stuck >= mover.stuck_limit:
                return _close(
                    run_log, run_id, "stuck", steps, start_map, end_map,
                    start_badges, end_badges, False, log_path,
                )

            trigger = evaluate(state, stuck_count=mover.stuck)
            if trigger is None and not getattr(supervisor, "always_on", False):
                streak_kind = None
                streak = 0
                if policy is None:
                    direction = _step_direction(state, target_xy, explorer)
                    actor = "script"
                    reason = None
                else:
                    visited.add((state.map_group, state.map_number, state.x, state.y))
                    action, _logp = policy.act(observe(state, visited))
                    direction = ACTIONS[int(action)]
                    actor = "rl"
                    reason = f"policy chose {direction}"
                if direction is None:
                    mover.stuck += 1
                    last_direction = None
                else:
                    if direction == "a":
                        machine.press("a", frames=8)
                        machine.tick(12, render=False)
                        path_result = ActionResult(True, "policy pressed a", frames=20)
                        last_moved = False
                    else:
                        path_result = mover.run_path([direction])
                        last_moved = bool(path_result.ok)
                    previous = state
                    last_direction = direction
                    _log_script(
                        decision_log,
                        machine,
                        state,
                        direction,
                        path_result,
                        target_map=target_map,
                        run_log=run_log,
                        run_id=run_id,
                        actor=actor,
                        reason=reason,
                    )
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
                    run_log=run_log,
                    run_id=run_id,
                )
                steps += 1

            state = reader.parse(getattr(machine, "frame", 0))
            end_map = tuple(state.map_id)
            end_badges = int(state.badges)
            cut_cleared = _cut_cleared(started_with_cut, state)
            if _advanced(state, start_map, start_badges, cut_cleared, target_map, target_xy, steps):
                return _close(
                    run_log, run_id, "advanced", steps, start_map, end_map,
                    start_badges, end_badges, cut_cleared, log_path,
                )
            if streak >= 3 or mover.stuck >= mover.stuck_limit:
                return _close(
                    run_log, run_id, "stuck", steps, start_map, end_map,
                    start_badges, end_badges, False, log_path,
                )
        return _close(
            run_log, run_id, "budget", steps, start_map, end_map,
            start_badges, end_badges, False, log_path,
        )
    except Exception as exc:
        return _close(
            run_log, run_id, "error", steps, start_map, end_map,
            start_badges, end_badges, False, log_path,
            error=f"{type(exc).__name__}: {exc}",
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


def _advanced(
    state,
    start_map: tuple,
    start_badges: int,
    cut_cleared: bool,
    target_map: tuple[int, int] | None,
    target_xy: tuple[int, int] | None,
    steps: int,
) -> bool:
    if steps <= 0 and target_map is None and target_xy is None:
        return False
    on_target_map = target_map is not None and tuple(state.map_id) == tuple(target_map)
    on_target_tile = (
        target_xy is not None
        and (state.x, state.y) == tuple(target_xy)
        and (target_map is None or tuple(state.map_id) == tuple(target_map))
    )
    if on_target_map or on_target_tile:
        return True
    if target_map is not None or target_xy is not None:
        return False
    return tuple(state.map_id) != tuple(start_map) or int(state.badges) != start_badges or cut_cleared


def _step_direction(state, target_xy: tuple[int, int] | None, explorer: Explorer) -> str | None:
    if target_xy is not None:
        tiles = getattr(state, "tiles", None) or {}
        for direction in _reducing_dirs(state.x, state.y, target_xy[0], target_xy[1]):
            if walkable(int(tiles.get(direction, 0xFF))):
                return direction
        return None
    return explorer.choose(state)


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


def _log_script(
    decision_log,
    machine,
    state,
    direction: str,
    result: ActionResult,
    *,
    target_map: tuple[int, int] | None = None,
    run_log=None,
    run_id: int | None = None,
    actor: str = "script",
    reason: str | None = None,
) -> None:
    if decision_log is None:
        return
    if reason is None:
        if target_map is None:
            reason = "exploring unvisited tiles"
        else:
            reason = f"segment script toward map {tuple(target_map)}"
    tool = "a" if direction == "a" else "run_path"
    arguments = {} if direction == "a" else {"directions": [direction]}
    decision = Decision(
        name=tool,
        arguments=arguments,
        reason=reason,
        model="ppo" if actor == "rl" else "script",
        input_tokens=0,
        output_tokens=0,
        est_usd=0.0,
    )
    append_decision(
        decision_log,
        state=state,
        frame=getattr(machine, "frame", 0),
        actor=actor,
        trigger=None,
        decision=decision,
        result=result,
        run_log=run_log,
        run_id=run_id,
    )


def _close(
    run_log,
    run_id,
    outcome,
    steps,
    start_map,
    end_map,
    start_badges,
    end_badges,
    cut_cleared,
    log_path,
    error: str = "",
) -> SegmentResult:
    if run_log is not None and run_id is not None:
        run_log.finish_run(
            run_id,
            outcome=outcome,
            end_map=end_map or (0, 0),
            end_badges=end_badges,
            steps=steps,
            cut_cleared=cut_cleared,
            error=error,
        )
    return _finish(
        outcome, steps, start_map, end_map, start_badges, end_badges, cut_cleared, log_path, run_id
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
    run_id: int | None = None,
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
        run_id=run_id,
    )
