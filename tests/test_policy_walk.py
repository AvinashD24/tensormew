import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.collision import COLL_FLOOR, COLL_WALL
from src.llm_supervisor import Supervisor
from src.ram_parser import GameState
from src.segment import run_segment


def gs(**kw):
    base = {
        "map_group": 1,
        "map_number": 1,
        "x": 0,
        "y": 0,
        "facing": "up",
        "tiles": {"up": COLL_WALL, "down": COLL_WALL, "left": COLL_FLOOR, "right": COLL_WALL},
        "battle_mode": 0,
        "script_running": 0,
        "script_mode": 0,
        "textbox_flags": 1,
    }
    base.update(kw)
    return GameState.from_dict(base)


class FakeMachine:
    def __init__(self):
        self.presses = []
        self.frame = 0

    def press(self, button, frames=8):
        self.presses.append(button)
        self.frame += frames

    def tick(self, n=1, render=True):
        self.frame += n
        return True


class WarpReader:
    def __init__(self, machine):
        self.machine = machine

    def parse(self, frame=0):
        if "left" in self.machine.presses:
            return gs(map_group=2, map_number=2, x=3, y=3)
        return gs()


class LeftPolicy:
    def act(self, obs):
        return 2, -0.1  # left


class Boom:
    def create(self, **kwargs):
        raise AssertionError("llm should stay out of ordinary walking")


class PolicyWalkTests(unittest.TestCase):
    def test_policy_walks_and_llm_is_not_called(self):
        machine = FakeMachine()
        log_path = Path("logs/test/policy_walk.jsonl")
        if log_path.exists():
            log_path.unlink()
        supervisor = Supervisor(client=SimpleNamespace(messages=Boom()), registry={}, log_path=None)
        result = run_segment(
            machine,
            WarpReader(machine),
            supervisor,
            max_steps=5,
            decision_log=log_path,
            policy=LeftPolicy(),
        )
        self.assertEqual(result.outcome, "advanced")
        self.assertEqual(result.end_map, (2, 2))
        row = json.loads(log_path.read_text().strip().splitlines()[-1])
        self.assertEqual(row["actor"], "rl")
        self.assertEqual(row["reason"], "policy chose left")


if __name__ == "__main__":
    unittest.main()
