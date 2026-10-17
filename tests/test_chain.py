import unittest
from pathlib import Path
from types import SimpleNamespace

from src.chain import discovered_name, run_chain
from src.collision import COLL_FLOOR, COLL_WALL
from src.llm_supervisor import Supervisor
from src.ram_parser import GameState
from src.run_log import RunLog


def gs(**kw):
    tiles = {"up": COLL_WALL, "down": COLL_WALL, "left": COLL_FLOOR, "right": COLL_WALL}
    base = {
        "map_group": 1,
        "map_number": 1,
        "x": 0,
        "y": 0,
        "facing": "up",
        "tiles": tiles,
        "battle_mode": 0,
        "script_running": 0,
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
    """One open direction. After a step, the map id changes. No stored route."""

    def __init__(self, machine):
        self.machine = machine

    def parse(self, frame=0):
        if self.machine.presses:
            return gs(map_group=9, map_number=9, x=4, y=4)
        return gs()


class FakeRegistry:
    def __init__(self):
        self.captured = []

    def capture(self, emu, parser, name, description="", tags=(), force=False, overwrite=False):
        self.captured.append({"name": name, "description": description, "tags": tuple(tags)})
        return name


class Boom:
    def create(self, **kwargs):
        raise AssertionError("llm should not be required to find a warp")


class ChainTests(unittest.TestCase):
    def test_discovers_a_new_map_without_a_direction_table(self):
        machine = FakeMachine()
        registry = FakeRegistry()
        db = Path("logs/test/chain.db")
        if db.exists():
            db.unlink()
        run_log = RunLog(db)
        log_path = Path("logs/test/chain.jsonl")
        if log_path.exists():
            log_path.unlink()
        try:
            supervisor = Supervisor(client=SimpleNamespace(messages=Boom()), registry={}, log_path=None)
            chain = run_chain(
                machine,
                WarpReader(machine),
                supervisor,
                state_name="start",
                hops=2,
                max_steps=5,
                decision_log=log_path,
                run_log=run_log,
                registry=registry,
            )
            self.assertEqual(chain.results[0].outcome, "advanced")
            self.assertEqual(chain.results[0].end_map, (9, 9))
            self.assertEqual(chain.checkpoints[0], discovered_name((9, 9)))
            self.assertEqual(registry.captured[0]["tags"], ("route_test", "agent"))
            self.assertNotIn("left", registry.captured[0]["description"].lower())
            named = [row["checkpoint_name"] for row in run_log.recent_runs(5)]
            self.assertIn("agent_map_9_9", named)
        finally:
            run_log.close()


if __name__ == "__main__":
    unittest.main()
