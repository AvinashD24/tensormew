"""One overworld step: read RAM, ask on a trigger, run the real tool, log why."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.action_results import ActionResult
from src.llm_supervisor import Decision, Supervisor, TriggerContext
from src.triggers import TriggerEvent, evaluate

DEFAULT_DECISION_LOG = Path("logs/decisions.jsonl")


def step(
    supervisor: Supervisor,
    state_text: str,
    trigger: TriggerContext | TriggerEvent | str | None = None,
    *,
    hard: bool = False,
) -> tuple[Decision, ActionResult] | None:
    decision = supervisor.decide(state_text, trigger, hard=hard)
    if decision is None:
        return None
    result = supervisor.run_tool(decision.name, {**decision.arguments, "reason": decision.reason})
    return decision, result


def act(
    supervisor: Supervisor,
    machine,
    reader,
    *,
    actor: str = "llm",
    hard: bool = False,
    stuck_count: int = 0,
    decision_log: str | Path | None = DEFAULT_DECISION_LOG,
    run_log=None,
    run_id: int | None = None,
) -> tuple[Decision, ActionResult] | None:
    state = reader.parse(machine.frame)
    trigger = evaluate(state, stuck_count=stuck_count)
    decision = supervisor.decide(state.to_text(), trigger, hard=hard)
    if decision is None:
        return None
    result = supervisor.run_tool(decision.name, {**decision.arguments, "reason": decision.reason})
    if decision_log is not None:
        append_decision(
            decision_log,
            state=state,
            frame=getattr(machine, "frame", 0),
            actor=actor,
            trigger=trigger,
            decision=decision,
            result=result,
            run_log=run_log,
            run_id=run_id,
        )
    return decision, result


def append_decision(
    path: str | Path,
    *,
    state,
    frame: int,
    actor: str,
    trigger,
    decision: Decision,
    result: ActionResult,
    run_log=None,
    run_id: int | None = None,
) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if trigger is None:
        trigger_text = ""
    else:
        kind = getattr(getattr(trigger, "kind", trigger), "value", getattr(trigger, "kind", trigger))
        detail = getattr(trigger, "detail", "")
        trigger_text = f"{kind}: {detail}".strip(": ") if detail else str(kind)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame": frame,
        "map": [state.map_group, state.map_number],
        "pos": [state.x, state.y],
        "badges": state.badges,
        "actor": actor,
        "trigger": trigger_text,
        "tool": decision.name,
        "arguments": decision.arguments,
        "reason": decision.reason,
        "ok": result.ok,
        "est_usd": decision.est_usd,
        "input_tokens": decision.input_tokens,
        "output_tokens": decision.output_tokens,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    if run_log is not None and run_id is not None:
        run_log.add_step(run_id, row)
