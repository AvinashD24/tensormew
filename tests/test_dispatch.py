import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.action_results import MacroResult
from src.agent_loop import act
from src.collision import COLL_CUT_TREE, COLL_FLOOR
from src.config import ROM_PATH, STATES_DIR, SYM_PATH
from src.llm_supervisor import Supervisor
from src.macros import Macros
from src.ram_parser import GameState
from src.tool_dispatch import execution_registry
from src.triggers import TriggerKind, evaluate


def gs(**kw):
    base = {
        "map_group": 21,
        "map_number": 14,
        "x": 3,
        "y": 7,
        "facing": "up",
        "tiles": {"up": COLL_CUT_TREE, "down": COLL_FLOOR, "left": COLL_FLOOR, "right": COLL_FLOOR},
        "battle_mode": 0,
        "script_running": 0,
        "textbox_flags": 0,
        "badge_bits": 0b00001111,
    }
    base.update(kw)
    return GameState.from_dict(base)


class FakeMachine:
    def __init__(self):
        self.presses = []
        self.frame = 0

    def press(self, button, frames=8):
        self.presses.append((button, frames))
        self.frame += frames

    def tick(self, n=1, render=True):
        self.frame += n
        return True


class ScriptedReader:
    def __init__(self, states):
        self.states = list(states)
        self.i = 0

    def parse(self, frame=0):
        state = self.states[min(self.i, len(self.states) - 1)]
        self.i += 1
        return state


class FakeMessages:
    def create(self, **kwargs):
        return SimpleNamespace(
            model=kwargs["model"],
            usage=SimpleNamespace(input_tokens=12, output_tokens=6),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="use_cut",
                    input={"reason": "a Cut tree is blocking the path"},
                    id="toolu_cut",
                )
            ],
        )


class DispatchTests(unittest.TestCase):
    def test_cut_trigger_runs_real_macro_and_logs_reason(self):
        machine = FakeMachine()
        clear = gs(tiles={"up": COLL_FLOOR, "down": COLL_FLOOR, "left": COLL_FLOOR, "right": COLL_FLOOR})
        reader = ScriptedReader([gs(), gs(), clear])
        registry = execution_registry(machine, reader)
        self.assertIs(registry["use_cut"].__func__, Macros.use_cut)
        log_path = Path("logs/test/dispatch/decisions.jsonl")
        if log_path.exists():
            log_path.unlink()
        supervisor = Supervisor(client=SimpleNamespace(messages=FakeMessages()), registry=registry, log_path=None)
        outcome = act(supervisor, machine, reader, decision_log=log_path)
        self.assertIsNotNone(outcome)
        decision, result = outcome
        self.assertEqual(decision.name, "use_cut")
        self.assertIsInstance(result, MacroResult)
        self.assertNotIn("stub", result.message)
        self.assertTrue(machine.presses)
        row = json.loads(log_path.read_text().strip().splitlines()[-1])
        self.assertEqual(row["tool"], "use_cut")
        self.assertEqual(row["reason"], "a Cut tree is blocking the path")
        self.assertEqual(row["actor"], "llm")
        self.assertIn("CUT_BLOCKED", row["trigger"])
        self.assertEqual(row["map"], [21, 14])
        self.assertEqual(row["badges"], 4)
        self.assertNotIn("reason", row["arguments"])
        event = evaluate(gs())
        self.assertEqual(event.kind, TriggerKind.CUT_BLOCKED)


ILEX = STATES_DIR / "ilex_src.state"
HAS_LIVE = ROM_PATH.exists() and SYM_PATH.exists() and ILEX.exists()


@unittest.skipUnless(HAS_LIVE, "ilex_src state or ROM missing")
class LiveDispatchTests(unittest.TestCase):
    def test_ilex_checkpoint_dispatches_real_use_cut(self):
        from src.emulator import Emulator
        from src.ram_parser import RamParser
        from src.states import StateRegistry

        emu = Emulator(headless=True, rtc_lock=True)
        try:
            parser = RamParser(emu)
            emu.load_state(StateRegistry().load_bytes("ilex_src"))
            emu.tick(30, render=False)
            registry = execution_registry(emu, parser)
            supervisor = Supervisor(
                client=SimpleNamespace(messages=FakeMessages()),
                registry=registry,
                log_path=None,
                always_on=True,
            )
            log_path = Path("logs/test/dispatch/ilex.jsonl")
            outcome = act(supervisor, emu, parser, decision_log=log_path)
            self.assertIsNotNone(outcome)
            decision, result = outcome
            self.assertEqual(decision.name, "use_cut")
            self.assertIsInstance(result, MacroResult)
            self.assertFalse(result.message.startswith("stub"))
            row = json.loads(log_path.read_text().strip().splitlines()[-1])
            self.assertEqual(row["reason"], "a Cut tree is blocking the path")
        finally:
            emu.close()


if __name__ == "__main__":
    unittest.main()
