import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.config import ROM_PATH, SYM_PATH
from src.emulator import InputEvent
from src.recording import FrameDumper, InputRecorder, load_inputs, replay


class FakeMachine:
    def __init__(self):
        self.frame = 0
        self.listeners = []
        self.log = []
        self.loaded = None

    def add_listener(self, fn):
        self.listeners.append(fn)
        return lambda: self.listeners.remove(fn)

    def hold(self, button):
        self.log.append((self.frame, button, "press"))
        for fn in list(self.listeners):
            fn(InputEvent(self.frame, button, "press"))

    def release(self, button):
        self.log.append((self.frame, button, "release"))
        for fn in list(self.listeners):
            fn(InputEvent(self.frame, button, "release"))

    def tick(self, n=1, render=True):
        self.frame += n
        return True

    def load_state(self, src):
        self.loaded = src
        self.frame = 0

    def screen_image(self):
        return Image.new("RGBA", (160, 144), (self.frame % 255, 0, 0, 255))


class FakeResult:
    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.final_state = {"map_group": 1, "map_number": 1, "x": 2, "y": 3, "party_count": 1, "badges": 0}


class FakeReader:
    def parse(self, frame=0):
        from src.ram_parser import GameState

        return GameState.from_dict(
            {
                "map_group": 1,
                "map_number": 1,
                "x": 2,
                "y": 3,
                "facing": "down",
                "party_count": 1,
                "tiles": {"up": 0, "down": 0, "left": 0, "right": 0},
            }
        )


class RecordingTests(unittest.TestCase):
    def test_roundtrip_and_replay(self):
        machine = FakeMachine()
        recorder = InputRecorder(machine, runs_dir=Path(tempfile.mkdtemp()))
        recorder.on_start({"state": "lab", "state_sha1": "abc", "rom_sha1": "def"})
        machine.hold("up")
        machine.tick(4)
        machine.release("up")
        recorder.on_end(FakeResult("ep1"))
        header, events = load_inputs(recorder.path)
        self.assertEqual(header["episode_id"], "ep1")
        self.assertEqual([(e.frame, e.button, e.kind) for e in events], [(0, "up", "press"), (4, "up", "release")])

        other = FakeMachine()
        replay(other, FakeReader(), b"STATE", events, header["total_frames"])
        self.assertEqual(other.log, [(0, "up", "press"), (4, "up", "release")])

    def test_gif(self):
        dumper = FrameDumper(every=4, scale=1)
        machine = FakeMachine()
        for frame in range(40):
            machine.frame = frame
            dumper(machine, frame)
        self.assertEqual(len(dumper.frames), 10)
        path = Path(tempfile.mkdtemp()) / "a.gif"
        dumper.save_gif(path)
        self.assertGreater(path.stat().st_size, 0)


@unittest.skipUnless(ROM_PATH.exists() and SYM_PATH.exists(), "ROM not present")
class RomReplayTests(unittest.TestCase):
    def test_replay_matches(self):
        from src.emulator import Emulator
        from src.episode import EpisodeRunner, RandomScript
        from src.ram_parser import RamParser

        folder = Path(tempfile.mkdtemp())
        boot = Emulator(headless=True, rtc_lock=True)
        try:
            boot.tick(300, render=False)
            data = boot.save_state(folder / "boot.state")
        finally:
            boot.close()

        emu = Emulator(headless=True, rtc_lock=True)
        recorder = InputRecorder(emu, runs_dir=folder)
        try:
            EpisodeRunner(emu, RamParser(emu), _Bytes(data), runs_dir=folder, observers=[recorder]).run(
                RandomScript(("start", "a", "b", "up", "down", "left", "right")),
                "boot",
                max_steps=30,
                timeout_s=60,
                seed=1,
                episode_id="rec1",
            )
        finally:
            emu.close()

        header, events = load_inputs(folder / "rec1.inputs.jsonl")
        fresh = Emulator(headless=True, rtc_lock=True)
        try:
            result = replay(fresh, RamParser(fresh), data, events, header["total_frames"], expected=header, render_every=0)
        finally:
            fresh.close()
        self.assertTrue(result.matches, result.mismatches)


class _Bytes:
    def __init__(self, data):
        self.data = data

    def load_bytes(self, name):
        return self.data
