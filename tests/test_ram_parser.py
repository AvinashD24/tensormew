import json
import unittest
from pathlib import Path

from src.config import ROM_PATH, SYM_PATH
from src.ram_parser import SYMBOLS, GameState, RamParser, SymbolError


class FakeBus:
    def __init__(self, symbols, mem):
        self.symbols = symbols
        self.mem = mem

    def symbol(self, name):
        if name not in self.symbols:
            raise KeyError(name)
        return self.symbols[name]

    def read(self, bank, addr, n=1):
        return bytes(self.mem.get((bank, addr + i), 0) for i in range(n))


def _layout(values):
    symbols = {}
    mem = {}
    addr = 0xD000
    for sym in SYMBOLS:
        symbols[sym.name] = (1, addr)
        raw = values.get(sym.name, 0)
        if sym.size == 1:
            mem[(1, addr)] = raw
        elif sym.size == 2:
            mem[(1, addr)] = (raw >> 8) & 0xFF
            mem[(1, addr + 1)] = raw & 0xFF
        else:
            for i, byte in enumerate(raw if isinstance(raw, tuple) else (raw,) * sym.size):
                mem[(1, addr + i)] = byte
        addr += sym.size
    return symbols, mem


class RamParserTests(unittest.TestCase):
    def test_missing_symbols_named_together(self):
        symbols, mem = _layout({})
        del symbols["wMapGroup"]
        del symbols["wXCoord"]
        with self.assertRaises(SymbolError) as caught:
            RamParser(FakeBus(symbols, mem))
        text = str(caught.exception)
        self.assertIn("wMapGroup", text)
        self.assertIn("wXCoord", text)

    def test_facing_tiles_and_hp(self):
        symbols, mem = _layout(
            {
                "wMapGroup": 24,
                "wMapNumber": 3,
                "wXCoord": 10,
                "wYCoord": 12,
                "wFacingDirection": 4,
                "wTileUp": 0x12,
                "wPartyCount": 2,
                "wPartyMon1HP": 300,
                "wPartyMon1MaxHP": 320,
                "wJohtoBadges": 0b00000011,
            }
        )
        state = RamParser(FakeBus(symbols, mem)).parse()
        self.assertEqual(state.facing, "up")
        self.assertIn("up=CUT_TREE", state.to_text())
        self.assertEqual(state.party_hp[0], (300, 320))
        self.assertEqual(len(state.party_hp), 2)
        self.assertEqual(state.badges, 2)
        self.assertLessEqual(len(state.to_text().splitlines()), 6)
        for word in ("map=", "pos=", "facing=", "adjacent:", "party=", "badges=", "time="):
            self.assertIn(word, state.to_text())

    def test_not_ready_order(self):
        symbols, mem = _layout({})
        state = RamParser(FakeBus(symbols, mem)).parse()
        self.assertIn("map=(0,0)", state.not_ready_reasons())
        symbols, mem = _layout({"wMapGroup": 1, "wMapNumber": 1, "wBattleMode": 1})
        self.assertIn("battle", RamParser(FakeBus(symbols, mem)).parse().not_ready_reasons())
        symbols, mem = _layout({"wMapGroup": 1, "wMapNumber": 1, "wScriptRunning": 1})
        self.assertIn("script", RamParser(FakeBus(symbols, mem)).parse().not_ready_reasons())

    def test_json_roundtrip(self):
        symbols, mem = _layout({"wMapGroup": 1, "wMapNumber": 2, "wXCoord": 3, "wYCoord": 4})
        state = RamParser(FakeBus(symbols, mem)).parse()
        again = GameState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(again.pos, state.pos)
        self.assertEqual(again.facing, state.facing)

    def test_no_emulator_import_and_no_hex_addresses(self):
        text = Path("src/ram_parser.py").read_text()
        self.assertNotIn("src.emulator", text)
        self.assertNotIn("0x", text)


@unittest.skipUnless(ROM_PATH.exists() and SYM_PATH.exists(), "ROM not present")
class RomResolveTests(unittest.TestCase):
    def test_all_symbols_resolve(self):
        try:
            from src.emulator import Emulator
        except ImportError:
            self.skipTest("emulator missing")
        emu = Emulator(headless=True, rtc_lock=True)
        try:
            parser = RamParser(emu)
            self.assertEqual(len(parser.resolved), len(SYMBOLS))
        finally:
            emu.close()
