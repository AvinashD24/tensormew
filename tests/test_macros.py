import unittest

from src.collision import COLL_CUT_TREE, COLL_FLOOR
from src.macros import Macros
from src.ram_parser import GameState


def gs(**kw):
    base = {
        "map_group": 7,
        "map_number": 1,
        "x": 5,
        "y": 5,
        "facing": "up",
        "tiles": {"up": COLL_FLOOR, "down": COLL_FLOOR, "left": COLL_FLOOR, "right": COLL_FLOOR},
        "battle_mode": 0,
        "script_running": 0,
        "textbox_flags": 0,
    }
    base.update(kw)
    return GameState.from_dict(base)


class FakeMachine:
    def __init__(self):
        self.presses = []
        self.frame = 0

    def press(self, button, frames=8):
        self.presses.append((button, frames))
        self.frame += frames

    def tick(self, n=1, render=True):
        self.frame += n
        return True


class ReadableMachine(FakeMachine):
    """FakeMachine with RAM reads (all zeros) so macros take their real-emulator code path."""

    def read(self, bank, addr, n=1):
        return bytes([0] * n)


class ScriptedReader:
    def __init__(self, states):
        self.states = list(states)
        self.i = 0

    def parse(self, frame=0):
        state = self.states[min(self.i, len(self.states) - 1)]
        self.i += 1
        return state


class MacroTests(unittest.TestCase):
    def test_mash_text_ignores_text_speed_bit(self):
        machine = FakeMachine()
        macros = Macros(machine, ScriptedReader([gs(textbox_flags=1, script_running=0, script_mode=0)]))
        result = macros.mash_text(max_frames=600)
        self.assertTrue(result.ok)
        self.assertEqual(result.message, "no dialogue open")
        self.assertEqual(machine.presses, [])

    def test_mash_text_stops_when_clear(self):
        talking = gs(script_running=1)
        clear = gs(script_running=0)
        macros = Macros(FakeMachine(), ScriptedReader([talking, talking, clear]))
        result = macros.mash_text(max_frames=600)
        self.assertTrue(result.ok)

    def test_use_cut_faces_tree_and_presses_a(self):
        before = gs(facing="down", tiles={"up": COLL_CUT_TREE, "down": 0, "left": 0, "right": 0})
        after = gs(facing="up", tiles={"up": COLL_FLOOR, "down": 0, "left": 0, "right": 0})
        machine = FakeMachine()
        macros = Macros(machine, ScriptedReader([before, after]))
        result = macros.use_cut()
        self.assertTrue(result.ok)
        buttons = [b for b, _ in machine.presses]
        self.assertEqual(buttons[0], "up")
        self.assertGreaterEqual(buttons.count("a"), 2)

    def test_use_cut_no_tree(self):
        macros = Macros(FakeMachine(), ScriptedReader([gs()]))
        result = macros.use_cut()
        self.assertFalse(result.ok)

    def test_use_cut_in_battle(self):
        macros = Macros(FakeMachine(), ScriptedReader([gs(battle_mode=1)]))
        self.assertFalse(macros.use_cut().ok)

    def test_choose_move_slot_2(self):
        machine = FakeMachine()
        macros = Macros(machine, ScriptedReader([gs(battle_mode=2)]))
        result = macros.choose_move(2)
        self.assertTrue(result.ok)
        self.assertIn(("down", 8), machine.presses)

    def test_choose_move_overworld(self):
        macros = Macros(FakeMachine(), ScriptedReader([gs()]))
        self.assertFalse(macros.choose_move(1).ok)

    def test_switch_slot_bounds(self):
        macros = Macros(FakeMachine(), ScriptedReader([gs(battle_mode=1)]))
        self.assertFalse(macros.switch_pokemon(0).ok)
        self.assertFalse(macros.switch_pokemon(7).ok)

    def test_switch_fainted_pokemon_in_battle(self):
        machine = FakeMachine()
        macros = Macros(machine, ScriptedReader([gs(battle_mode=2)]))
        result = macros.switch_fainted_pokemon(2)
        self.assertTrue(result.ok)
        buttons = [b for b, _ in machine.presses]
        self.assertIn("down", buttons)
        self.assertIn("a", buttons)

    def test_switch_fainted_pokemon_overworld(self):
        macros = Macros(FakeMachine(), ScriptedReader([gs(battle_mode=0)]))
        self.assertFalse(macros.switch_fainted_pokemon(2).ok)

    def test_handle_whiteout(self):
        machine = FakeMachine()
        macros = Macros(machine, ScriptedReader([gs(battle_mode=1)]))
        result = macros.handle_whiteout()
        self.assertTrue(result.ok)

    def test_use_battle_item_in_battle(self):
        machine = FakeMachine()
        macros = Macros(machine, ScriptedReader([gs(battle_mode=2, num_items=1)]))
        result = macros.use_battle_item(1, 1)
        self.assertTrue(result.ok)
        buttons = [b for b, _ in machine.presses]
        self.assertIn("down", buttons)
        self.assertIn("a", buttons)

    def test_use_battle_item_no_items(self):
        macros = Macros(FakeMachine(), ScriptedReader([gs(battle_mode=2, num_items=0)]))
        self.assertFalse(macros.use_battle_item(1, 1).ok)

    def test_run_away_refused_in_trainer_battle(self):
        macros = Macros(FakeMachine(), ScriptedReader([gs(battle_mode=2)]))
        res = macros.run_away()
        self.assertFalse(res.ok)
        self.assertIn("cannot run from a trainer battle", res.message)

    def test_run_away_allowed_in_wild_battle(self):
        machine = FakeMachine()
        macros = Macros(machine, ScriptedReader([gs(battle_mode=1)]))
        res = macros.run_away()
        self.assertTrue(res.ok)
        buttons = [b for b, _ in machine.presses]
        self.assertIn("right", buttons)
        self.assertIn("a", buttons)

    def test_switch_fainted_pokemon_closed_loop_memory(self):
        # The cursor is never read: the macro steps DOWN and presses A, and the first pick works here.
        machine = ReadableMachine()
        macros = Macros(machine, ScriptedReader([gs(battle_mode=2), gs(battle_mode=2, battle_mon_hp=15)]))
        res = macros.switch_fainted_pokemon(4)
        self.assertTrue(res.ok)
        buttons = [b for b, _ in machine.presses]
        self.assertEqual(buttons.count("down"), 1)
        self.assertNotIn("up", buttons)
        self.assertIn("a", buttons)


    def test_switch_fainted_pokemon_tries_next_slot_when_one_is_rejected(self):
        machine = ReadableMachine()
        # the first row is rejected by the game ("no will to fight"), the next DOWN press lands on one that goes in
        states = [gs(battle_mode=2), gs(battle_mode=2, battle_mon_hp=0), gs(battle_mode=2, battle_mon_hp=11)]
        macros = Macros(machine, ScriptedReader(states))
        res = macros.switch_fainted_pokemon(3)
        self.assertTrue(res.ok)
        self.assertIn("2 down-press", res.message)
        buttons = [b for b, _ in machine.presses]
        self.assertEqual(buttons.count("down"), 2)
        self.assertNotIn("up", buttons)

    def test_switch_fainted_pokemon_cycles_every_row_then_gives_up(self):
        machine = ReadableMachine()
        macros = Macros(machine, ScriptedReader([gs(battle_mode=2, party_count=3)]))
        res = macros.switch_fainted_pokemon(2)
        self.assertFalse(res.ok)
        # 3 party rows + CANCEL, one DOWN press each, never the same row twice in a row without moving
        self.assertEqual([b for b, _ in machine.presses].count("down"), 4)
