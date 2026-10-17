"""LLM supervisor: tool calls only, with token cost logging."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.action_results import ActionResult
from src.supervisor_stubs import STUB_REGISTRY
from src.tools_schema import anthropic_tools, parse_tool_call, split_reason

DEFAULT_MODEL = os.environ.get("POKEMIND_MODEL", "claude-haiku-4-5")
HARD_MODEL = os.environ.get("POKEMIND_HARD_MODEL", DEFAULT_MODEL)
# Rough Haiku-class rates (USD / 1e6 tokens). Override if pricing changes.
INPUT_USD_PER_MTOK = float(os.environ.get("POKEMIND_INPUT_USD_PER_MTOK", "1.0"))
OUTPUT_USD_PER_MTOK = float(os.environ.get("POKEMIND_OUTPUT_USD_PER_MTOK", "5.0"))

SYSTEM_PROMPT = """You are playing Pokémon Crystal (GBC). The state text is authoritative.
Reply only by calling one tool. Never name or suggest raw buttons (Up, Down, A, B, Start, Select).
On trigger CUT_BLOCKED, prefer use_cut when the party can Cut.
On trigger BATTLE, prefer choose_move unless HP is critical, then switch_pokemon.
On trigger STUCK or DIALOGUE, prefer mash_text or a short run_path.
Every tool call must include reason: one sentence explaining why this action, given the state and trigger.
"""


class SupervisorError(RuntimeError):
    pass


@dataclass(frozen=True)
class TriggerContext:
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class Decision:
    name: str
    arguments: dict
    reason: str
    model: str
    input_tokens: int
    output_tokens: int
    est_usd: float


def format_user_message(state_text: str, trigger: TriggerContext | str | None) -> str:
    if trigger is None:
        return state_text
    if isinstance(trigger, str):
        line = trigger.strip()
        return f"{state_text}\n\ntrigger={line}" if line else state_text
    detail = trigger.detail.strip()
    kind = getattr(trigger.kind, "value", trigger.kind)
    body = f"{kind}: {detail}" if detail else str(kind)
    return f"{state_text}\n\ntrigger={body}"


def estimate_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * INPUT_USD_PER_MTOK + output_tokens * OUTPUT_USD_PER_MTOK) / 1_000_000


class Supervisor:
    def __init__(
        self,
        *,
        client: Any = None,
        model: str | None = None,
        hard_model: str | None = None,
        always_on: bool = False,
        registry: dict[str, Callable[..., ActionResult]] | None = None,
        log_path: str | Path | None = "logs/supervisor.jsonl",
    ):
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.hard_model = hard_model or HARD_MODEL
        self.always_on = always_on
        self.registry = registry if registry is not None else dict(STUB_REGISTRY)
        self.log_path = Path(log_path) if log_path else None

    def decide(
        self,
        state_text: str,
        trigger: TriggerContext | str | None = None,
        *,
        hard: bool = False,
    ) -> Decision | None:
        if not self.always_on and trigger is None:
            return None
        model = self.hard_model if hard else self.model
        user = format_user_message(state_text, trigger)
        response = self._client().messages.create(
            model=model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            tools=anthropic_tools(),
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user}],
        )
        name, arguments = _first_tool_use(response)
        call = parse_tool_call(name, arguments)
        reason, handler_args = split_reason(call.arguments)
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        decision = Decision(
            name=call.name,
            arguments=handler_args,
            reason=reason,
            model=getattr(response, "model", None) or model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            est_usd=estimate_usd(in_tok, out_tok),
        )
        self._log(decision)
        return decision

    def run_tool(self, name: str, arguments: dict | None) -> ActionResult:
        call = parse_tool_call(name, arguments)
        _reason, handler_args = split_reason(call.arguments)
        fn = self.registry.get(call.name)
        if fn is None:
            return ActionResult(ok=False, message=f"no handler for {call.name}")
        return fn(**handler_args)

    def _client(self):
        if self.client is not None:
            return self.client
        import anthropic

        self.client = anthropic.Anthropic()
        return self.client

    def _log(self, decision: Decision) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": decision.model,
            "input_tokens": decision.input_tokens,
            "output_tokens": decision.output_tokens,
            "tool_name": decision.name,
            "est_usd": decision.est_usd,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def _first_tool_use(response) -> tuple[str, dict]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            raw = getattr(block, "input", None) or {}
            if not isinstance(raw, dict):
                raise SupervisorError("tool input was not an object")
            return block.name, raw
    raise SupervisorError("model returned no tool call")
