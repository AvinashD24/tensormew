"""PyBoy wrapper. Holds the emulator; does not subclass it."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

from src.config import ROM_PATH, ROM_SHA1, SYM_PATH, sha1_file

BUTTONS: tuple[str, ...] = ("up", "down", "left", "right", "a", "b", "start", "select")
InputListener = Callable[["InputEvent"], None]

_TIME = {"MORN": (0, 8), "DAY": (1, 12), "NITE": (2, 20)}


class RomMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputEvent:
    frame: int
    button: str
    kind: Literal["press", "release"]


class Machine(Protocol):
    @property
    def frame(self) -> int: ...

    def tick(self, n: int = 1, render: bool = True) -> bool: ...
    def press(self, button: str, frames: int = 8) -> None: ...
    def hold(self, button: str) -> None: ...
    def release(self, button: str) -> None: ...
    def save_state(self, path=None) -> bytes: ...
    def load_state(self, src: bytes | Path, *, apply_options: bool = True) -> None: ...
    def read(self, bank: int, addr: int, n: int = 1) -> bytes: ...
    def symbol(self, name: str) -> tuple[int, int]: ...
    def screen_image(self): ...
    def add_listener(self, fn: InputListener) -> Callable[[], None]: ...


class Emulator:
    def __init__(
        self,
        rom=ROM_PATH,
        sym=SYM_PATH,
        *,
        headless: bool = True,
        speed: int | None = None,
        verify_rom: bool = True,
        rtc_lock: bool = True,
    ):
        rom = Path(rom)
        sym = Path(sym)
        if verify_rom:
            got = sha1_file(rom)
            if got != ROM_SHA1:
                raise RomMismatchError(f"rom sha1 {got} != {ROM_SHA1}")
        from pyboy import PyBoy

        if os.environ.get("PYBOY_VISUAL") == "1":
            headless = False
        sound = os.environ.get("PYBOY_SOUND", "0") == "1"
        self._pb = PyBoy(str(rom), window="null" if headless else "SDL2", symbols=str(sym), sound=sound)
        if speed is None:
            # 0 is uncapped. A window defaults to 10x so the walk is visible but not realtime.
            speed = 0 if headless else int(os.environ.get("PYBOY_SPEED", "10"))
        self._pb.set_emulation_speed(speed)
        self._frame = 0
        self._listeners: list[InputListener] = []
        self._rtc_lock = rtc_lock
        self._closed = False
        if rtc_lock:
            self._lock_rtc()

    def _lock_rtc(self) -> None:
        try:
            self._pb.rtc_lock_experimental(True)
        except Exception:
            pass

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def pyboy(self):
        return self._pb

    def tick(self, n: int = 1, render: bool = True) -> bool:
        if not self._headless:
            ok = True
            for _ in range(max(1, n)):
                ok = self._pb.tick(1, True)
                self._frame += 1
            return bool(ok)
        ok = self._pb.tick(n, render)
        self._frame += n
        return bool(ok)

    def _check(self, button: str) -> None:
        if button not in BUTTONS:
            raise ValueError(f"unknown button {button!r}")

    def _emit(self, button: str, kind: Literal["press", "release"]) -> None:
        event = InputEvent(self._frame, button, kind)
        for fn in list(self._listeners):
            fn(event)

    def hold(self, button: str) -> None:
        self._check(button)
        self._pb.button_press(button)
        self._emit(button, "press")

    def release(self, button: str) -> None:
        self._check(button)
        self._pb.button_release(button)
        self._emit(button, "release")

    def press(self, button: str, frames: int = 8) -> None:
        self.hold(button)
        self.tick(frames, render=False)
        self.release(button)

    def save_state(self, path=None) -> bytes:
        buf = io.BytesIO()
        self._pb.save_state(buf)
        data = buf.getvalue()
        if path is not None:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return data

    def load_state(self, src: bytes | Path, *, apply_options: bool = True) -> None:
        if isinstance(src, (bytes, bytearray)):
            handle = io.BytesIO(src)
            own = False
        else:
            handle = open(src, "rb")
            own = True
        try:
            self._pb.load_state(handle)
        finally:
            if own:
                handle.close()
        self._frame = 0
        if self._rtc_lock:
            self._lock_rtc()
        if apply_options:
            self.apply_training_options()

    def read(self, bank: int, addr: int, n: int = 1) -> bytes:
        out = bytearray()
        for i in range(n):
            a = addr + i
            if 0xD000 <= a < 0xE000:
                out.append(int(self._pb.memory[bank, a]) & 0xFF)
            else:
                out.append(int(self._pb.memory[a]) & 0xFF)
        return bytes(out)

    def write(self, bank: int, addr: int, data: bytes) -> None:
        for i, value in enumerate(data):
            a = addr + i
            if 0xD000 <= a < 0xE000:
                self._pb.memory[bank, a] = value
            else:
                self._pb.memory[a] = value

    def symbol(self, name: str) -> tuple[int, int]:
        bank, addr = self._pb.symbol_lookup(name)
        return int(bank), int(addr)

    def set_time_of_day(self, name: str) -> None:
        if name not in _TIME:
            raise ValueError(f"unknown time of day {name!r}")
        tod, hour = _TIME[name]
        bank, addr = self.symbol("wTimeOfDay")
        self.write(bank, addr, bytes([tod]))
        bank, addr = self.symbol("hHours")
        self.write(bank, addr, bytes([hour]))

    def apply_training_options(self) -> int:
        from src.options import training_options

        bank, addr = self.symbol("wOptions")
        current = self.read(bank, addr, 1)[0]
        updated = training_options(current)
        self.write(bank, addr, bytes([updated]))
        return updated

    def seed_rng(self, add: int, sub: int) -> None:
        bank, addr = self.symbol("hRandomAdd")
        self.write(bank, addr, bytes([add & 0xFF]))
        bank, addr = self.symbol("hRandomSub")
        self.write(bank, addr, bytes([sub & 0xFF]))

    def screen_image(self):
        return self._pb.screen.image.copy()

    def set_speed(self, speed: int) -> None:
        self._pb.set_emulation_speed(speed)

    def add_listener(self, fn: InputListener) -> Callable[[], None]:
        self._listeners.append(fn)

        def unsubscribe() -> None:
            if fn in self._listeners:
                self._listeners.remove(fn)

        return unsubscribe

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pb.stop(save=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
