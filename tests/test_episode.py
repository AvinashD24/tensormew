import json
import tempfile
import unittest
from pathlib import Path

from src.config import ROM_PATH, SYM_PATH
from src.episode import Action, ButtonScript, EpisodeRunner, FileStateSource, RandomScript
from src.ram_parser import GameState


def gs(x=0):
    return GameState.from_dict(
        {"map_group": 1, "map_number": 1, "x": x, "y": 1, "facing": "down", "tiles": {"up": 0, "down": 0, "left": 0, "right": 0}}
    )


class FakeMachine:
    def __init__(self):
        self.frame = 7
        self.presses = []

    def load_state(self, src):
        self.frame = 0

    def press(self, button, frames=8):
        self.presses.append(button)
        self.frame += frames

    def tick(self, n=1, render=True):
        self.frame += n
        return True


class FakeReader:
    def __init__(self):
        self.n = 0

    def parse(self, frame=0):
        state = gs(self.n)
        self.n += 1
        return state


class FakeSource:
    def load_bytes(self, name):
        return b"STATE"


class Boom:
    name = "boom"

    def reset(self, state, seed):
        return None

    def act(self, state):
        raise RuntimeError("nope")


class EpisodeTests(unittest.TestCase):
    def test_script_done(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = EpisodeRunner(FakeMachine(), FakeReader(), FakeSource(), runs_dir=Path(folder))
            actions = [Action("press", "up"), Action("press", "up"), Action("press", "a")]
            result = runner.run(ButtonScript(actions), "lab", max_steps=10, timeout_s=30)
            self.assertEqual(result.outcome, "script_done")
            self.assertEqual(result.steps, 3)
            lines = [json.loads(line) for line in result.log_path.read_text().splitlines()]
            self.assertEqual(lines[0]["type"], "start")
            self.assertEqual(sum(1 for line in lines if line["type"] == "step"), 3)
            self.assertEqual(lines[-1]["type"], "end")

    def test_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = EpisodeRunner(FakeMachine(), FakeReader(), FakeSource(), runs_dir=Path(folder))
            result = runner.run(RandomScript(), "lab", max_steps=5, timeout_s=30, seed=1)
            self.assertEqual(result.outcome, "step_budget")
            self.assertEqual(result.steps, 5)
            result = runner.run(RandomScript(), "lab", max_steps=5, timeout_s=0, seed=1)
            self.assertEqual(result.outcome, "timeout")
            self.assertEqual(result.steps, 1)

    def test_error_still_closes_log(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = EpisodeRunner(FakeMachine(), FakeReader(), FakeSource(), runs_dir=Path(folder))
            result = runner.run(Boom(), "lab", max_steps=5, timeout_s=30)
            self.assertEqual(result.outcome, "error")
            self.assertIn("nope", result.error)
            self.assertEqual(json.loads(result.log_path.read_text().splitlines()[-1])["type"], "end")

    def test_same_seed_same_actions(self):
        a = RandomScript()
        b = RandomScript()
        state = gs()
        a.reset(state, 7)
        b.reset(state, 7)
        self.assertEqual([str(a.act(state)) for _ in range(20)], [str(b.act(state)) for _ in range(20)])


@unittest.skipUnless(ROM_PATH.exists() and SYM_PATH.exists(), "ROM not present")
class DeterminismTests(unittest.TestCase):
    def test_two_runs_match(self):
        from src.emulator import Emulator
        from src.ram_parser import RamParser

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            emu = Emulator(headless=True, rtc_lock=True)
            try:
                emu.tick(400, render=False)
                emu.save_state(folder / "boot.state")
            finally:
                emu.close()
            positions = []
            for _ in range(2):
                emu = Emulator(headless=True, rtc_lock=True)
                try:
                    result = EpisodeRunner(emu, RamParser(emu), FileStateSource(folder), runs_dir=folder).run(
                        RandomScript(), "boot", max_steps=40, timeout_s=60, seed=1
                    )
                finally:
                    emu.close()
                lines = [json.loads(line) for line in result.log_path.read_text().splitlines() if '"type": "step"' in line]
                positions.append([(line["state"]["x"], line["state"]["y"], line["frame"]) for line in lines])
            self.assertEqual(positions[0], positions[1])
