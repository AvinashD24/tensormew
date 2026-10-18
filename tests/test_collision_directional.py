import unittest

from src.collision import (
    COLL_HOP_DOWN,
    COLL_HOP_LEFT,
    COLL_HOP_RIGHT,
    COLL_HOP_UP,
    COLL_WALK_DOWN,
    COLL_FLOOR,
    COLL_WALL,
    allows_direction,
    is_ledge,
    is_warp,
    COLL_DOOR,
    COLL_WARP_CARPET_DOWN,
)
from src.mover import walkable


class DirectionalCollisionTests(unittest.TestCase):
    def test_hop_down_directional_allowances(self):
        # Can only hop down
        self.assertTrue(allows_direction(COLL_HOP_DOWN, "down"))
        self.assertFalse(allows_direction(COLL_HOP_DOWN, "up"))
        self.assertFalse(allows_direction(COLL_HOP_DOWN, "left"))
        self.assertFalse(allows_direction(COLL_HOP_DOWN, "right"))

    def test_hop_left_directional_allowances(self):
        self.assertTrue(allows_direction(COLL_HOP_LEFT, "left"))
        self.assertFalse(allows_direction(COLL_HOP_LEFT, "right"))
        self.assertFalse(allows_direction(COLL_HOP_LEFT, "up"))
        self.assertFalse(allows_direction(COLL_HOP_LEFT, "down"))

    def test_hop_right_directional_allowances(self):
        self.assertTrue(allows_direction(COLL_HOP_RIGHT, "right"))
        self.assertFalse(allows_direction(COLL_HOP_RIGHT, "left"))

    def test_walkable_with_direction(self):
        # Normal floor is walkable in any direction
        self.assertTrue(walkable(COLL_FLOOR, "up"))
        self.assertTrue(walkable(COLL_FLOOR, "down"))

        # Wall is blocked in any direction
        self.assertFalse(walkable(COLL_WALL, "down"))

        # Hop down ledge: walkable DOWN, blocked UP
        self.assertTrue(walkable(COLL_HOP_DOWN, "down"))
        self.assertFalse(walkable(COLL_HOP_DOWN, "up"))
        self.assertFalse(walkable(COLL_HOP_DOWN, "left"))

    def test_is_ledge_and_is_warp(self):
        self.assertTrue(is_ledge(COLL_HOP_DOWN))
        self.assertTrue(is_ledge(COLL_WALK_DOWN))
        self.assertFalse(is_ledge(COLL_FLOOR))

        self.assertTrue(is_warp(COLL_DOOR))
        self.assertTrue(is_warp(COLL_WARP_CARPET_DOWN))
        self.assertFalse(is_warp(COLL_FLOOR))
