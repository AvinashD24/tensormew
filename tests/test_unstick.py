import unittest
from types import SimpleNamespace

from src.unstick import Unsticker


class FakeEmu:
    def __init__(self):
        self.log = []

    def press(self, b, frames=8):
        self.log.append(("press", b))

    def tick(self, n=1, render=True):
        self.log.append(("tick", n))


class FakeStepper:
    def __init__(self, free):
        self.free = set(free)
        self.calls = []

    def step(self, d):
        self.calls.append(d)
        return SimpleNamespace(moved=d in self.free)


class FakeMacros:
    def __init__(self):
        self.text = 0

    def mash_text(self, max_frames=0):
        self.text += 1


def _state(x=5, y=5):
    return SimpleNamespace(map_group=26, map_number=3, x=x, y=y, tiles={"left": 0x01, "right": 0x01, "up": 0x01, "down": 0x01})


class UnstickTests(unittest.TestCase):
    def test_counter_resets_on_movement(self):
        u = Unsticker()
        for _ in range(5):
            u.note(_state())
        self.assertEqual(u.count, 4)
        u.note(_state(6, 5))
        self.assertEqual(u.count, 0)

    def test_nothing_before_the_talk_threshold(self):
        u = Unsticker()
        u.count = 3
        self.assertIsNone(u.act(_state(), "right", FakeEmu(), FakeMacros(), FakeStepper([])))

    def test_talks_then_waits_then_detours(self):
        u, emu, macros = Unsticker(), FakeEmu(), FakeMacros()
        u.count = 6
        self.assertEqual(u.act(_state(), "right", emu, macros, FakeStepper([])).kind, "talk")
        self.assertIn(("press", "a"), emu.log)
        self.assertEqual(macros.text, 1)
        u.count = 10
        self.assertEqual(u.act(_state(), "right", emu, macros, FakeStepper([])).kind, "wait")
        u.count = 14
        res = u.act(_state(), "right", emu, macros, FakeStepper(["up", "right"]))
        self.assertEqual(res.kind, "detour")
        self.assertTrue(res.moved)

    def test_detour_in_a_one_tile_corridor_reports_no_way(self):
        u = Unsticker()
        u.count = 14
        res = u.act(_state(), "right", FakeEmu(), FakeMacros(), FakeStepper([]))
        self.assertFalse(res.moved)

    def test_detour_tries_a_side_the_tile_table_calls_blocked(self):
        # Route 30 beside Mikey: the left tile is a ledge row (0xA3) the table rejects for "left",
        # but the player can walk along it, so the sidestep must still be attempted.
        state = SimpleNamespace(map_group=26, map_number=1, x=5, y=24,
                                tiles={"left": 0xA3, "right": 0x07, "up": 0x00, "down": 0x00})
        u = Unsticker()
        u.count = 14
        stepper = FakeStepper(["left", "up"])
        res = u.act(state, "up", FakeEmu(), FakeMacros(), stepper)
        self.assertEqual(res.kind, "detour")
        self.assertTrue(res.moved)
        self.assertIn("left", stepper.calls)

    def test_retreat_backs_off(self):
        u = Unsticker()
        u.count = 38
        st = FakeStepper(["left"])
        res = u.act(_state(), "right", FakeEmu(), FakeMacros(), st)
        self.assertEqual(res.kind, "retreat")
        self.assertEqual(st.calls[0], "left")
