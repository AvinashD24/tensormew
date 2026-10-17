"""One decision step. Triggers come from Dev A; handlers stay in the registry."""

from __future__ import annotations

from src.action_results import ActionResult
from src.llm_supervisor import Decision, Supervisor, TriggerContext


def step(
    supervisor: Supervisor,
    state_text: str,
    trigger: TriggerContext | str | None = None,
    *,
    hard: bool = False,
) -> tuple[Decision, ActionResult] | None:
    decision = supervisor.decide(state_text, trigger, hard=hard)
    if decision is None:
        return None
    result = supervisor.run_tool(decision.name, decision.arguments)
    return decision, result
