import tempfile
import unittest
from pathlib import Path

from src.ram_parser import GameState
from src.states import StateRegistry


def ready(**kw):
    base = {
        "map_group": 1,
        "map_number": 2,
        "x": 3,
        "y": 4,
        "facing": "down",
        "party_count": 1,
        "badge_bits": 1,
        "tiles": {"up": 0, "down": 0, "left": 0, "right": 0},
    }
    base.update(kw)
    return GameState.from_dict(base)


class FakeMachine:
    def __init__(self, blob=b"S1"):
        self.blob = blob
        self.loaded = None
        self.frame = 0

    def save_state(self, path=None):
        if path is not None:
            Path(path).write_bytes(self.blob)
        return self.blob

    def load_state(self, src):
        self.loaded = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
        self.frame = 0


class FakeParser:
    def __init__(self, state):
        self.state = state

    def parse(self, frame=0):
        return self.state


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.reg = StateRegistry(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_capture_and_load(self):
        entry = self.reg.capture(FakeMachine(), FakeParser(ready()), "a", description="start")
        self.assertTrue((self.root / "a.state").exists())
        self.assertEqual(entry.map, (1, 2))
        self.assertEqual(entry.pos, (3, 4))
        self.assertEqual(entry.badges, 1)
        self.assertTrue(entry.state_sha1)
        self.assertTrue(entry.rom_sha1)
        self.assertTrue(entry.created_at)
        self.assertEqual(self.reg.load_bytes("a"), b"S1")
        (self.root / "a.state").write_bytes(b"NO")
        with self.assertRaises(ValueError):
            self.reg.load_bytes("a")

    def test_not_ready_and_overwrite(self):
        machine = FakeMachine()
        parser = FakeParser(ready(map_group=0, map_number=0))
        with self.assertRaises(ValueError) as caught:
            self.reg.capture(machine, parser, "a")
        self.assertIn("map=(0,0)", str(caught.exception))
        entry = self.reg.capture(machine, parser, "a", force=True)
        self.assertIn("forced", entry.tags)
        with self.assertRaises(ValueError):
            self.reg.capture(machine, FakeParser(ready()), "a")
        self.reg.capture(machine, FakeParser(ready()), "a", overwrite=True)

    def test_validate_and_names(self):
        self.reg.capture(FakeMachine(), FakeParser(ready()), "a")
        results = self.reg.validate(FakeMachine(), FakeParser(ready()), ["a"])
        self.assertTrue(results[0].ok)
        bad = self.reg.validate(FakeMachine(), FakeParser(ready(x=9, y=9)), ["a"])
        self.assertFalse(bad[0].ok)
        self.assertTrue(any("pos" in p for p in bad[0].problems))
        missing = self.reg.validate(FakeMachine(), FakeParser(ready()), ["gone"])
        self.assertIn("missing", missing[0].problems[0])
        with self.assertRaises(ValueError):
            self.reg.capture(FakeMachine(), FakeParser(ready()), "Bad/Name")
