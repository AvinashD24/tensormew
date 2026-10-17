import unittest

from src.collision import COLL_CUT_TREE, COLL_FLOOR, COLL_WALL, COLL_WATER
from src.mover import Mover, walkable
from src.ram_parser import GameState
from src.step import DIRS

_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class Grid:
    """Machine + reader: first press in a new facing turns; second press walks."""

    def __init__(self, x=5, y=5, facing="down", tiles=None, battle_mode=0, blocked=None):
        self.x = x
        self.y = y
        self.facing = facing
        self.tiles = tiles or {d: COLL_FLOOR for d in DIRS}
        self.battle_mode = battle_mode
        self.blocked = set(blocked or ())
        self.frame = 0
        self.presses = []
        self._mid_step = False

    def press(self, button, frames=8):
        self.presses.append((button, frames))
        self.frame += frames
        if button not in DIRS:
            return
        if self.facing != button:
            self.facing = button
            self._mid_step = False
            return
        nx, ny = self.x + _DELTA[button][0], self.y + _DELTA[button][1]
        if (nx, ny) in self.blocked:
            return
        if not walkable(self.tiles.get(button, COLL_FLOOR)):
            return
        self.x, self.y = nx, ny

    def tick(self, n=1, render=True):
        self.frame += n
        return True

    def parse(self, frame=0):
        return GameState.from_dict(
            {
                "map_group": 24,
                "map_number": 3,
                "x": self.x,
                "y": self.y,
                "facing": self.facing,
                "tiles": dict(self.tiles),
                "battle_mode": self.battle_mode,
            }
        )


class WalkableTests(unittest.TestCase):
    def test_floor_and_wall(self):
        self.assertTrue(walkable(COLL_FLOOR))
        self.assertFalse(walkable(COLL_WALL))
        self.assertFalse(walkable(COLL_WATER))
        self.assertFalse(walkable(COLL_CUT_TREE))


class MoverTests(unittest.TestCase):
    def test_already_there(self):
        world = Grid()
        result = Mover(world, world).go_to(5, 5)
        self.assertTrue(result.ok)
        self.assertEqual(result.steps, 0)

    def test_greedy_right(self):
        world = Grid(facing="right")
        result = Mover(world, world).go_to(7, 5)
        self.assertTrue(result.ok)
        self.assertEqual((world.x, world.y), (7, 5))
        self.assertGreaterEqual(result.steps, 2)

    def test_blocked_by_wall(self):
        world = Grid(tiles={"up": 0, "down": 0, "left": 0, "right": COLL_WALL})
        result = Mover(world, world).go_to(8, 5)
        self.assertFalse(result.ok)
        self.assertIn("blocked", result.message)

    def test_run_path(self):
        world = Grid(facing="up")
        result = Mover(world, world).run_path(["up", "right"])
        self.assertTrue(result.ok)
        self.assertEqual(result.steps_done, 2)
        self.assertEqual((world.x, world.y), (6, 4))

    def test_run_path_bad_dir(self):
        world = Grid()
        result = Mover(world, world).run_path(["start"])
        self.assertFalse(result.ok)

    def test_go_to_refuses_battle(self):
        world = Grid(battle_mode=1)
        self.assertFalse(Mover(world, world).go_to(6, 5).ok)

    def test_stuck_when_tile_never_changes(self):
        world = Grid(facing="right", blocked={(6, 5), (7, 5), (8, 5), (9, 5)})
        result = Mover(world, world, stuck_limit=2, max_steps=10).go_to(9, 5)
        self.assertFalse(result.ok)
        self.assertGreaterEqual(result.stuck, 2)
        self.assertIn("stuck", result.message)
