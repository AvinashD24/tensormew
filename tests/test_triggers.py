import unittest

from src.collision import COLL_CUT_TREE, COLL_FLOOR, COLL_WALL
from src.ram_parser import GameState
from src.triggers import TriggerKind, evaluate


def gs(**kw):
    base = {
        "map_group": 7,
        "map_number": 1,
        "x": 10,
        "y": 12,
        "facing": "up",
        "tiles": {"up": COLL_FLOOR, "down": COLL_FLOOR, "left": COLL_FLOOR, "right": COLL_FLOOR},
    }
    base.update(kw)
    return GameState.from_dict(base)


class TriggerTests(unittest.TestCase):
    def test_battle_wins(self):
        event = evaluate(gs(battle_mode=1, script_running=1, tiles={"up": COLL_CUT_TREE, "down": 0, "left": 0, "right": 0}))
        self.assertEqual(event.kind, TriggerKind.BATTLE)

    def test_dialogue_when_idle(self):
        event = evaluate(gs(script_running=1), mover_idle=True)
        self.assertEqual(event.kind, TriggerKind.DIALOGUE)

    def test_no_stuck_during_script(self):
        event = evaluate(gs(script_running=1), stuck_count=20, mover_idle=False)
        self.assertIsNone(event)

    def test_cut_blocked(self):
        event = evaluate(gs(tiles={"up": COLL_CUT_TREE, "down": COLL_WALL, "left": 0, "right": 0}))
        self.assertEqual(event.kind, TriggerKind.CUT_BLOCKED)
        self.assertIn("up", event.detail)

    def test_stuck(self):
        event = evaluate(gs(), stuck_count=8)
        self.assertEqual(event.kind, TriggerKind.STUCK)

    def test_not_stuck_yet(self):
        self.assertIsNone(evaluate(gs(), stuck_count=3))
