"""Live PyBoy checks. Skip when Yash's .state files are not on disk."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.config import ROM_PATH, STATES_DIR, SYM_PATH
from src.mover import Mover
from src.ram_parser import RamParser
from src.triggers import TriggerKind, evaluate

NEW_BARK = STATES_DIR / "new_bark_outside.state"
ROUTE29 = STATES_DIR / "route29_entry.state"
SMOKE_WALL = STATES_DIR / "smoke_wall.state"
HAS_ROM = ROM_PATH.exists() and SYM_PATH.exists()


@unittest.skipUnless(HAS_ROM, "ROM not present")
class LiveBootTests(unittest.TestCase):
    def test_stack_constructs_on_real_pyboy(self):
        from src.emulator import Emulator
        from src.macros import Macros

        emu = Emulator(headless=True, rtc_lock=True)
        try:
            parser = RamParser(emu)
            emu.tick(200, render=False)
            state = parser.parse(emu.frame)
            self.assertTrue(parser.resolved)
            mover = Mover(emu, parser, max_steps=2)
            macros = Macros(emu, parser)
            event = evaluate(state)
            path = mover.run_path(["right"])
            self.assertIsInstance(path.ok, bool)
            self.assertIsInstance(macros.mash_text(max_frames=32).ok, bool)
            self.assertTrue(state.to_text())
            if event is not None:
                self.assertIn(event.kind, list(TriggerKind))
        finally:
            emu.close()


@unittest.skipUnless(HAS_ROM and NEW_BARK.exists(), "new_bark_outside.state missing")
class LiveNewBarkTests(unittest.TestCase):
    def test_go_to_moves_one_tile(self):
        from src.emulator import Emulator

        emu = Emulator(headless=True, rtc_lock=True)
        try:
            parser = RamParser(emu)
            emu.load_state(NEW_BARK)
            emu.tick(30, render=False)
            before = parser.parse(emu.frame)
            self.assertNotEqual(before.map_id, (0, 0), msg=before.to_text())
            target_x = before.x + 1
            result = Mover(emu, parser, max_steps=8).go_to(target_x, before.y)
            after = parser.parse(emu.frame)
            self.assertTrue(result.ok, result.message)
            self.assertEqual((after.x, after.y), (target_x, before.y), after.to_text())
        finally:
            emu.close()


@unittest.skipUnless(HAS_ROM and SMOKE_WALL.exists(), "smoke_wall.state missing")
class LiveWallTests(unittest.TestCase):
    def test_wall_does_not_claim_success(self):
        from src.emulator import Emulator

        emu = Emulator(headless=True, rtc_lock=True)
        try:
            parser = RamParser(emu)
            emu.load_state(SMOKE_WALL)
            emu.tick(30, render=False)
            before = parser.parse(emu.frame)
            mover = Mover(emu, parser, max_steps=4)
            result = mover.go_to(before.x, before.y + 20)
            self.assertFalse(result.ok)
        finally:
            emu.close()


@unittest.skipUnless(HAS_ROM and ROUTE29.exists(), "route29_entry.state missing")
class LiveRoute29Tests(unittest.TestCase):
    def test_run_path_changes_x_or_y(self):
        from src.emulator import Emulator

        emu = Emulator(headless=True, rtc_lock=True)
        try:
            parser = RamParser(emu)
            emu.load_state(ROUTE29)
            emu.tick(30, render=False)
            before = parser.parse(emu.frame)
            result = Mover(emu, parser).run_path(["left", "left"])
            after = parser.parse(emu.frame)
            self.assertNotEqual(before.pos, after.pos, f"{result.message}\n{after.to_text()}")
        finally:
            emu.close()
