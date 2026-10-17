import tempfile
import unittest
from pathlib import Path

from src.config import ROM_PATH, SYM_PATH
from src.emulator import BUTTONS, Emulator, InputEvent, RomMismatchError

ROM_OK = ROM_PATH.exists() and SYM_PATH.exists()


class RomCheckTests(unittest.TestCase):
    def test_wrong_bytes_rejected_before_boot(self):
        with tempfile.NamedTemporaryFile(suffix=".gbc") as handle:
            handle.write(b"not a rom")
            handle.flush()
            with self.assertRaises(RomMismatchError):
                Emulator(rom=handle.name, verify_rom=True)


@unittest.skipUnless(ROM_OK, "ROM not present")
class EmulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emu = Emulator(headless=True, rtc_lock=True)
        cls.emu.tick(600, render=False)

    @classmethod
    def tearDownClass(cls):
        cls.emu.close()

    def test_tick_advances_frame(self):
        emu = Emulator(headless=True, rtc_lock=True)
        try:
            self.assertTrue(emu.tick(300, render=False))
            self.assertEqual(emu.frame, 300)
        finally:
            emu.close()

    def test_save_load_restores_wram_and_resets_frame(self):
        snap = self.emu.save_state()
        ram = self.emu.read(1, 0xD000, 256)
        self.assertTrue(snap)
        self.emu.tick(200, render=False)
        self.emu.load_state(snap)
        self.assertEqual(self.emu.frame, 0)
        self.assertEqual(self.emu.read(1, 0xD000, 256), ram)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.state"
            data = self.emu.save_state(path)
            self.assertEqual(path.read_bytes(), data)
            self.emu.tick(10, render=False)
            self.emu.load_state(path)
            self.assertEqual(self.emu.frame, 0)

    def test_press_emits_events_and_unsubscribe(self):
        snap = self.emu.save_state()
        self.emu.load_state(snap)
        seen = []
        unsub = self.emu.add_listener(seen.append)
        self.emu.press("a", frames=5)
        self.assertEqual(seen, [InputEvent(0, "a", "press"), InputEvent(5, "a", "release")])
        self.assertEqual(self.emu.frame, 5)
        unsub()
        self.emu.press("a", frames=1)
        self.assertEqual(len(seen), 2)

    def test_unknown_button(self):
        with self.assertRaises(ValueError):
            self.emu.press("x")

    def test_screen_and_read(self):
        image = self.emu.screen_image()
        self.assertEqual(image.size, (160, 144))
        self.assertEqual(len(self.emu.read(0, 0xC2FA, 4)), 4)
        self.assertEqual(len(self.emu.read(1, 0xD000, 1)), 1)

    def test_time_of_day_uses_symbol(self):
        self.emu.set_time_of_day("DAY")
        bank, addr = self.emu.symbol("wTimeOfDay")
        self.assertEqual(self.emu.read(bank, addr, 1), bytes([1]))
        source = Path(__file__).resolve().parents[1] / "src" / "emulator.py"
        text = source.read_text()
        body = text.split("def set_time_of_day")[1].split("def seed_rng")[0]
        self.assertNotIn("0x", body)

    def test_two_runs_match(self):
        dumps = []
        for _ in range(2):
            emu = Emulator(headless=True, rtc_lock=True)
            try:
                emu.tick(30, render=False)
                emu.seed_rng(1, 2)
                for button in ("a", "start", "down", "up", "left"):
                    emu.press(button, frames=4)
                dumps.append(emu.read(1, 0xD000, 256))
            finally:
                emu.close()
        self.assertEqual(dumps[0], dumps[1])
        self.assertTrue(set(BUTTONS))
