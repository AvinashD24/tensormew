import json
import os
import unittest
from types import SimpleNamespace

from src.llm_supervisor import SYSTEM_PROMPT, Supervisor, format_user_message
from src.supervisor_stubs import stub_use_cut


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["messages"][0]["content"]
        if "CUT_BLOCKED" in user:
            name, payload = "use_cut", {"reason": "tree blocks the path"}
        elif "BATTLE" in user:
            name, payload = "choose_move", {"slot": 1, "reason": "wild fight"}
        else:
            name, payload = "mash_text", {"max_frames": 120, "reason": "dialogue is open"}
        return SimpleNamespace(
            model=kwargs["model"],
            usage=SimpleNamespace(input_tokens=20, output_tokens=8),
            content=[SimpleNamespace(type="tool_use", name=name, input=payload, id="toolu_test")],
        )


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


class SupervisorTests(unittest.TestCase):
    def test_cut_blocked_returns_use_cut_and_logs(self):
        client = FakeClient()
        log_path = self._log_path("cut.jsonl")
        supervisor = Supervisor(client=client, always_on=False, log_path=log_path)
        state = "map=(21,14) pos=(3,7) facing=up"
        decision = supervisor.decide(state, "CUT_BLOCKED: tree to the north")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.name, "use_cut")
        self.assertEqual(decision.arguments, {})
        self.assertEqual(decision.reason, "tree blocks the path")
        sent = client.messages.calls[0]
        self.assertIn("CUT_BLOCKED: tree to the north", sent["messages"][0]["content"])
        self.assertIn(state, sent["messages"][0]["content"])
        self.assertEqual(sent["system"], SYSTEM_PROMPT)
        self.assertGreaterEqual(len(sent["tools"]), 6)
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual(rows[-1]["tool_name"], "use_cut")
        self.assertEqual(rows[-1]["input_tokens"], 20)
        self.assertGreater(rows[-1]["est_usd"], 0)
        result = supervisor.run_tool("use_cut", {"reason": "tree blocks the path"})
        self.assertTrue(result.ok)
        self.assertEqual(result.message, stub_use_cut().message)

    def test_skips_llm_without_trigger_unless_always_on(self):
        client = FakeClient()
        quiet = Supervisor(client=client, always_on=False, log_path=None)
        self.assertIsNone(quiet.decide("map=(1,1) pos=(0,0)"))
        self.assertEqual(client.messages.calls, [])
        baseline = Supervisor(client=client, always_on=True, log_path=None)
        decision = baseline.decide("map=(1,1) pos=(0,0)")
        self.assertEqual(decision.name, "mash_text")
        self.assertEqual(len(client.messages.calls), 1)

    def test_run_tool_parses_args_into_stub(self):
        supervisor = Supervisor(client=FakeClient(), log_path=None)
        result = supervisor.run_tool("go_to", {"x": 8, "y": 12, "reason": "closer to the door"})
        self.assertEqual(result.message, "stub go_to 8,12")

    def test_format_trigger_object(self):
        from src.llm_supervisor import TriggerContext

        text = format_user_message("state", TriggerContext("BATTLE", "wild"))
        self.assertIn("trigger=BATTLE: wild", text)

    def _log_path(self, name):
        root = self.id().replace(".", "_")
        path = __import__("pathlib").Path("logs") / "test" / root / name
        if path.exists():
            path.unlink()
        return path


@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("POKEMIND_LIVE") == "1", "live API off")
class LiveSupervisorTests(unittest.TestCase):
    def test_one_live_tool_call(self):
        supervisor = Supervisor(always_on=True, log_path=None)
        decision = supervisor.decide(
            "map=(21,14) pos=(3,7) facing=up\nadjacent: up=cut_tree",
            "CUT_BLOCKED: tree to the north",
        )
        self.assertEqual(decision.name, "use_cut")


if __name__ == "__main__":
    unittest.main()
