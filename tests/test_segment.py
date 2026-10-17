import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.collision import COLL_CUT_TREE, COLL_FLOOR
from src.config import ROM_PATH, STATES_DIR, SYM_PATH
from src.llm_supervisor import Supervisor
from src.macros import Macros
from src.ram_parser import GameState
from src.segment import SegmentResult, run_segment
from src.tool_dispatch import execution_registry


def gs(**kw):
    base = {
        "map_group": 21,
        "map_number": 14,
        "x": 3,
        "y": 7,
        "facing": "up",
        "tiles": {"up": COLL_FLOOR, "down": COLL_FLOOR, "left": COLL_FLOOR, "right": COLL_FLOOR},
        "battle_mode": 0,
        "script_running": 0,
        "textbox_flags": 0,
        "badge_bits": 0b00001111,
    }
    base.update(kw)
    return GameState.from_dict(base)


def cut_tiles():
    return {"up": COLL_CUT_TREE, "down": COLL_FLOOR, "left": COLL_FLOOR, "right": COLL_FLOOR}


def floor_tiles():
    return {"up": COLL_FLOOR, "down": COLL_FLOOR, "left": COLL_FLOOR, "right": COLL_FLOOR}


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


class BoomMessages:
    def create(self, **kwargs):
        raise AssertionError("LLM should not be called without a trigger")


class FakeMessages:
    calls = 0

    def create(self, **kwargs):
        FakeMessages.calls += 1
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


class SegmentTests(unittest.TestCase):
    def test_no_trigger_takes_scripted_walkable_step(self):
        machine = FakeMachine()
        reader = ScriptedReader([gs()])
        sentinel = {"run_path": lambda directions: None, "use_cut": lambda: None}
        supervisor = Supervisor(
            client=SimpleNamespace(messages=BoomMessages()),
            registry=sentinel,
            log_path=None,
        )
        log_path = Path("logs/test/segment/script.jsonl")
        if log_path.exists():
            log_path.unlink()
        result = run_segment(machine, reader, supervisor, max_steps=1, decision_log=log_path)
        self.assertIsInstance(result, SegmentResult)
        self.assertIn(result.outcome, {"budget", "advanced"})
        self.assertGreaterEqual(result.steps, 1)
        self.assertIs(supervisor.registry, sentinel)
        self.assertTrue(machine.presses)
        row = json.loads(log_path.read_text().strip().splitlines()[-1])
        self.assertEqual(row["actor"], "script")
        self.assertEqual(row["tool"], "run_path")
        self.assertEqual(row["reason"], "segment script")
        self.assertEqual(row["arguments"]["directions"], ["up"])

    def test_cut_tree_clears_after_act(self):
        FakeMessages.calls = 0
        machine = FakeMachine()
        # parses: loop, act, use_cut start stay on the tree; use_cut after is floor
        reader = ScriptedReader([gs(tiles=cut_tiles()), gs(tiles=cut_tiles()), gs(tiles=cut_tiles()), gs(tiles=floor_tiles())])
        supervisor = Supervisor(
            client=SimpleNamespace(messages=FakeMessages()),
            log_path=None,
        )
        log_path = Path("logs/test/segment/cut.jsonl")
        if log_path.exists():
            log_path.unlink()
        result = run_segment(machine, reader, supervisor, max_steps=5, decision_log=log_path)
        self.assertEqual(result.outcome, "advanced")
        self.assertTrue(result.cut_cleared)
        self.assertGreaterEqual(result.steps, 1)
        self.assertIs(supervisor.registry["use_cut"].__func__, Macros.use_cut)
        self.assertTrue(machine.presses)
        row = json.loads(log_path.read_text().strip().splitlines()[-1])
        self.assertEqual(row["tool"], "use_cut")
        self.assertEqual(row["actor"], "llm")

    def test_same_trigger_three_times_is_stuck(self):
        FakeMessages.calls = 0
        machine = FakeMachine()
        reader = ScriptedReader([gs(tiles=cut_tiles())])
        supervisor = Supervisor(
            client=SimpleNamespace(messages=FakeMessages()),
            log_path=None,
        )
        log_path = Path("logs/test/segment/stuck.jsonl")
        if log_path.exists():
            log_path.unlink()
        result = run_segment(machine, reader, supervisor, max_steps=10, decision_log=log_path)
        self.assertEqual(result.outcome, "stuck")
        self.assertFalse(result.cut_cleared)
        self.assertEqual(result.steps, 3)
        self.assertEqual(FakeMessages.calls, 3)


ILEX = STATES_DIR / "ilex_src.state"
HAS_LIVE = ROM_PATH.exists() and SYM_PATH.exists() and ILEX.exists()


@unittest.skipUnless(HAS_LIVE, "ilex_src state or ROM missing")
class LiveSegmentTests(unittest.TestCase):
    def test_ilex_segment_returns_without_api_spend(self):
        from src.emulator import Emulator
        from src.ram_parser import RamParser
        from src.states import StateRegistry

        emu = Emulator(headless=True, rtc_lock=True)
        try:
            parser = RamParser(emu)
            emu.load_state(StateRegistry().load_bytes("ilex_src"))
            emu.tick(30, render=False)
            log_path = Path("logs/test/segment/ilex.jsonl")
            if log_path.exists():
                log_path.unlink()
            supervisor = Supervisor(
                client=SimpleNamespace(messages=FakeMessages()),
                registry=execution_registry(emu, parser),
                log_path=None,
            )
            result = run_segment(emu, parser, supervisor, max_steps=5, decision_log=log_path)
            self.assertIsInstance(result, SegmentResult)
            self.assertGreaterEqual(result.steps, 1)
            self.assertNotEqual(result.outcome, "error")
            lines = [line for line in log_path.read_text().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 1)
        finally:
            emu.close()


if __name__ == "__main__":
    unittest.main()
