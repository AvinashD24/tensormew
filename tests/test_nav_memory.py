import unittest
from pathlib import Path

from src.nav_memory import NavMemory


class NavMemoryTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("logs/test/nav.db")
        if self.path.exists():
            self.path.unlink()
        self.mem = NavMemory(self.path)

    def tearDown(self):
        self.mem.close()

    def test_same_map_step_is_ignored(self):
        self.mem.record((1, 1), (2, 3), "left", (1, 1), (1, 3))
        self.assertEqual(self.mem.exits_from((1, 1)), [])

    def test_map_change_is_stored(self):
        self.mem.record((1, 1), (0, 4), "left", (1, 2), (9, 4))
        exits = self.mem.exits_from((1, 1))
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["direction"], "left")
        self.assertEqual(exits[0]["dst_map"], (1, 2))
        self.assertEqual(exits[0]["src_xy"], (0, 4))

    def test_repeat_increments_seen_count(self):
        self.mem.record((3, 3), (1, 1), "up", (4, 4), (1, 8))
        self.mem.record((3, 3), (1, 1), "up", (4, 4), (1, 9))
        exits = self.mem.exits_from((3, 3))
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["seen_count"], 2)
        self.assertEqual(exits[0]["dst_xy"], (1, 9))


if __name__ == "__main__":
    unittest.main()
