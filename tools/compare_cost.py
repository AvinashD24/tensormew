"""Compare LLM-on-triggers vs LLM-every-step call counts on one trace."""

from __future__ import annotations

import os
from types import SimpleNamespace

from src.llm_supervisor import Supervisor

SAMPLE_TRACE: list[tuple[str, str | None]] = [
    ("map=(1,1) pos=(2,3) step=1", None),
    ("map=(1,1) pos=(2,3) step=2", "STUCK: same tile"),
    ("map=(1,1) pos=(3,3) step=3", None),
    ("map=(1,1) pos=(4,3) step=4", None),
    ("map=(1,1) pos=(4,3) step=5", "DIALOGUE: text box"),
    ("map=(1,1) pos=(4,4) step=6", None),
]


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs.get("model", "fake"),
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="mash_text",
                    input={"reason": "test", "max_frames": 30},
                    id="toolu_fake",
                )
            ],
        )


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def count_calls(steps: list[tuple[str, str | None]], *, always_on: bool, client) -> int:
    """steps is (state_text, trigger or None). Return how many times client.messages.create is called."""
    messages = client.messages
    original = messages.create
    calls = 0

    def create(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    messages.create = create
    try:
        supervisor = Supervisor(client=client, always_on=always_on, registry={}, log_path=None)
        for state_text, trigger in steps:
            supervisor.decide(state_text, trigger)
    finally:
        messages.create = original
    return calls


def make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        return anthropic.Anthropic()
    return FakeClient()


def comparison_line(steps: list[tuple[str, str | None]], client) -> str:
    triggers_only = count_calls(steps, always_on=False, client=client)
    always_on = count_calls(steps, always_on=True, client=client)
    return f"triggers_only={triggers_only} always_on={always_on}"


def main(argv: list[str] | None = None) -> int:
    del argv
    print(comparison_line(SAMPLE_TRACE, make_client()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
