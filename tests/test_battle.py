import unittest
from types import SimpleNamespace

from src.battle import pick_best_move, score_move, type_multiplier


class BattleTests(unittest.TestCase):
    def test_type_multipliers(self):
        # Ground (4) vs Flying (2) is immune (0.0x)
        self.assertEqual(type_multiplier(4, 2), 0.0)
        # Electric (12) vs Flying (2) is super effective (2.0x)
        self.assertEqual(type_multiplier(12, 2), 2.0)
        # Rock (5) vs Flying (2) is super effective (2.0x)
        self.assertEqual(type_multiplier(5, 2), 2.0)
        # Fire (9) vs Grass (11) is 2.0x
        self.assertEqual(type_multiplier(9, 11), 2.0)

    def test_score_move(self):
        # Tackle (33): normal damaging move vs flying/normal
        score_tackle = score_move(33, (2, 0))
        self.assertGreater(score_tackle, 0)

        # Mud-Slap (189): ground move vs flying/normal -> 0 damage due to immunity
        score_mudslap = score_move(189, (2, 0))
        self.assertEqual(score_mudslap, 0.0)

        # Leer (43): status move
        score_leer = score_move(43, (2, 0))
        self.assertEqual(score_leer, 5.0)

        # Damaging move should score higher than status move
        self.assertGreater(score_tackle, score_leer)

    def test_pick_best_move_slot(self):
        # Slot 1: Tackle (33), Slot 2: Leer (43), Slot 3: Empty (0), Slot 4: Empty (0)
        state = SimpleNamespace(party_moves=((33, 43, 0, 0),))
        # Tackle should be chosen (slot 1) over status move Leer
        self.assertEqual(pick_best_move(state, enemy_types=(2, 0)), 1)

        # Slot 1: Mud-Slap (189), Slot 2: Ember (52)
        state_ember = SimpleNamespace(party_moves=((189, 52, 0, 0),))
        # Against Falkner (Flying), Mud-slap does 0, Ember does full damage -> choose slot 2!
        self.assertEqual(pick_best_move(state_ember, enemy_types=(2, 0)), 2)
