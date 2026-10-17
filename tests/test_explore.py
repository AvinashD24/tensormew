import unittest

from src.collision import COLL_FLOOR, COLL_WALL
from src.explore import Explorer
from src.ram_parser import GameState


def gs(**kw):
    base = {
        "map_group": 1,
        "map_number": 1,
        "x": 0,
        "y": 0,
        "facing": "up",
        "tiles": {"up": COLL_FLOOR, "down": COLL_WALL, "left": COLL_WALL, "right": COLL_WALL},
        "battle_mode": 0,
        "script_running": 0,
        "textbox_flags": 1,
    }
    base.update(kw)
    return GameState.from_dict(base)


class ExplorerTests(unittest.TestCase):
    def test_walks_the_only_open_unvisited_tile(self):
        explorer = Explorer()
        first = explorer.choose(gs())
        self.assertEqual(first, "up")
        explorer.note(gs(), gs(y=-1), "up", moved=True)
        second = explorer.choose(gs(y=-1, tiles={"up": COLL_FLOOR, "down": COLL_FLOOR, "left": COLL_WALL, "right": COLL_WALL}))
        self.assertEqual(second, "up")

    def test_does_not_repeat_a_blocked_direction(self):
        explorer = Explorer()
        state = gs(tiles={"up": COLL_FLOOR, "down": COLL_FLOOR, "left": COLL_WALL, "right": COLL_WALL})
        first = explorer.choose(state)
        explorer.note(state, state, first, moved=False)
        second = explorer.choose(state)
        self.assertNotEqual(second, first)
        self.assertIn(second, {"up", "down"})

    def test_new_map_forgets_the_old_trail(self):
        explorer = Explorer()
        explorer.choose(gs())
        explorer.note(gs(), gs(map_number=2, x=5, y=5), "up", moved=True)
        self.assertEqual(explorer.map_id, (1, 2))
        self.assertEqual(explorer.visited, {(5, 5)})


if __name__ == "__main__":
    unittest.main()
