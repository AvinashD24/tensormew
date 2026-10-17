import unittest

from src.ram_parser import GameState
from src.step import TileStepper


def gs(**kw):
    base = {
        "map_group": 1,
        "map_number": 1,
        "x": 5,
        "y": 5,
        "facing": "down",
        "tiles": {"up": 0, "down": 0, "left": 0, "right": 0},
    }
    base.update(kw)
    return GameState.from_dict(base)


class FakeMachine:
    def __init__(self):
        self.presses = []
        self.ticks = []
        self.frame = 0

    def press(self, button, frames=8):
        self.presses.append((button, frames))
        self.frame += frames

    def tick(self, n=1, render=True):
        self.ticks.append((n, render))
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


class StepTests(unittest.TestCase):
    def test_already_facing_presses_once(self):
        machine = FakeMachine()
        reader = ScriptedReader([gs(facing="up"), gs(facing="up", y=4), gs(facing="up", y=4)])
        step = TileStepper(machine, reader).step("up")
        self.assertEqual(machine.presses, [("up", 8)])
        self.assertFalse(step.turned)
        self.assertTrue(step.moved)

    def test_turn_then_walk(self):
        machine = FakeMachine()
        reader = ScriptedReader([gs(facing="down"), gs(facing="up"), gs(facing="up", y=4)])
        step = TileStepper(machine, reader).step("up")
        self.assertEqual(machine.presses, [("up", 8), ("up", 8)])
        self.assertTrue(step.turned)

    def test_warp_waits_180(self):
        machine = FakeMachine()
        reader = ScriptedReader(
            [
                gs(facing="up"),
                gs(facing="up", map_group=2),
                gs(facing="up", map_group=2),
            ]
        )
        step = TileStepper(machine, reader).step("up")
        self.assertTrue(step.warped)
        self.assertIn((180, False), machine.ticks)

    def test_blocked(self):
        machine = FakeMachine()
        same = gs(facing="up")
        step = TileStepper(machine, ScriptedReader([same, same, same])).step("up")
        self.assertFalse(step.moved)

    def test_bad_direction(self):
        with self.assertRaises(ValueError):
            TileStepper(FakeMachine(), ScriptedReader([gs()])).step("start")
