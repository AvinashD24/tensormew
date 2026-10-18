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

    def test_tactical_battle_weaken_first(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Wild battle, uncaptured species, HP is 100% (20/20) -> should choose move to weaken
        state = SimpleNamespace(
            battle_mode=1,
            num_balls=5,
            party_count=1,
            party_species=(155,),  # Cyndaquil
            party_moves=((33, 0, 0, 0),),
            enemy_species=16,  # Pidgey
            enemy_hp=20,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.choose_move.assert_called_once_with(1)
        macros.use_pokeball.assert_not_called()

    def test_tactical_battle_throws_ball_when_weakened(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Wild battle, uncaptured species, HP is 30% (6/20) -> should throw ball
        state = SimpleNamespace(
            battle_mode=1,
            num_balls=5,
            party_count=1,
            party_species=(155,),
            party_moves=((33, 0, 0, 0),),
            enemy_species=16,
            enemy_hp=6,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.use_pokeball.assert_called_once()
        macros.choose_move.assert_not_called()

    def test_tactical_battle_attacks_when_species_already_caught(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Already have Pidgey in party -> should attack to KO and gain exp
        state = SimpleNamespace(
            battle_mode=1,
            num_balls=5,
            party_count=2,
            party_species=(155, 16),
            party_moves=((33, 0, 0, 0),),
            enemy_species=16,
            enemy_hp=5,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.choose_move.assert_called_once_with(1)
        macros.use_pokeball.assert_not_called()

    def test_tactical_battle_attacks_in_trainer_battle(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Trainer battle (battle_mode=2) -> balls cannot be used
        state = SimpleNamespace(
            battle_mode=2,
            num_balls=5,
            party_count=1,
            party_species=(155,),
            party_moves=((33, 0, 0, 0),),
            enemy_species=16,
            enemy_hp=5,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.choose_move.assert_called_once_with(1)
        macros.use_pokeball.assert_not_called()

    def test_tactical_battle_flees_when_player_low_hp(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Active Pokémon HP is 5/20 (25% <= 30%) in wild battle -> flee to prevent wipeout!
        state = SimpleNamespace(
            battle_mode=1,
            num_balls=5,
            party_count=1,
            party_species=(155,),
            party_hp=((5, 20),),
            party_levels=(5,),
            party_moves=((33, 0, 0, 0),),
            enemy_species=16,
            enemy_hp=5,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.run_away.assert_called_once()
        macros.choose_move.assert_not_called()
        macros.use_pokeball.assert_not_called()

    def test_tactical_battle_flees_when_already_caught_and_level_sufficient(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Active Pokémon is level 8 (>= 7) and Pidgey is already caught -> flee to stop grinding!
        state = SimpleNamespace(
            battle_mode=1,
            num_balls=5,
            party_count=2,
            party_species=(155, 16),
            party_hp=((20, 20), (18, 18)),
            party_levels=(8, 4),
            party_moves=((33, 0, 0, 0),),
            enemy_species=16,
            enemy_hp=10,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.run_away.assert_called_once()
        macros.choose_move.assert_not_called()
        macros.use_pokeball.assert_not_called()

    def test_tactical_battle_switches_fainted_pokemon_when_backup_alive(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Active Pokémon fainted (0/20 HP), but Pidgey (slot 2) is alive (15/15 HP)
        state = SimpleNamespace(
            battle_mode=2,
            party_count=2,
            party_species=(155, 16),
            party_hp=((0, 20), (15, 15)),
            party_levels=(5, 4),
            party_moves=((33, 0, 0, 0), (33, 0, 0, 0)),
            enemy_species=152,
            enemy_hp=12,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.switch_fainted_pokemon.assert_called_once_with(2)
        macros.choose_move.assert_not_called()
        macros.handle_whiteout.assert_not_called()

    def test_tactical_battle_switches_to_highest_hp_backup_when_fainted(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Team with fainted mons in slots 1, 3, 5, and alive mons in slot 2 (13 HP) and slot 4 (16 HP)
        state = SimpleNamespace(
            battle_mode=2,
            party_count=5,
            party_species=(155, 19, 161, 10, 165),
            party_hp=((0, 21), (13, 13), (0, 16), (16, 16), (0, 15)),
            party_levels=(6, 2, 3, 3, 3),
            party_moves=((33, 0, 0, 0),) * 5,
            enemy_species=152,
            enemy_hp=2,
            enemy_max_hp=21,
        )
        tactical_battle_step(state, macros)
        # Should choose Caterpie (slot 4, 16 HP) over Rattata (slot 2, 13 HP)
        macros.switch_fainted_pokemon.assert_called_once_with(4)
        macros.choose_move.assert_not_called()

    def test_tactical_battle_handles_whiteout_when_all_fainted(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # All party Pokémon fainted
        state = SimpleNamespace(
            battle_mode=2,
            party_count=2,
            party_species=(155, 16),
            party_hp=((0, 20), (0, 15)),
            party_levels=(5, 4),
            party_moves=((33, 0, 0, 0), (33, 0, 0, 0)),
            enemy_species=152,
            enemy_hp=12,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.handle_whiteout.assert_called_once()
        macros.switch_fainted_pokemon.assert_not_called()
        macros.choose_move.assert_not_called()

    def test_tactical_battle_uses_item_in_trainer_battle_when_low_hp(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Trainer battle: active mon at 5/20 HP (25% <= 30%), pack has 1 Potion
        state = SimpleNamespace(
            battle_mode=2,
            num_items=1,
            party_count=1,
            party_species=(155,),
            party_hp=((5, 20),),
            party_levels=(5,),
            party_moves=((33, 0, 0, 0),),
            enemy_species=152,
            enemy_hp=10,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.use_battle_item.assert_called_once_with(item_slot=1, target_slot=1)
        macros.choose_move.assert_not_called()

    def test_tactical_battle_switches_to_healthy_backup_in_trainer_battle_when_no_items(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Trainer battle: active mon low HP (5/20), no items, but healthy backup (slot 2: 18/18)
        state = SimpleNamespace(
            battle_mode=2,
            num_items=0,
            party_count=2,
            party_species=(155, 16),
            party_hp=((5, 20), (18, 18)),
            party_levels=(5, 5),
            party_moves=((33, 0, 0, 0), (33, 0, 0, 0)),
            enemy_species=152,
            enemy_hp=10,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        macros.switch_pokemon.assert_called_once_with(2)
        macros.choose_move.assert_not_called()

    def test_tactical_battle_never_runs_away_in_trainer_battle_even_if_low_and_no_backups(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Trainer battle: active mon at critical 2/20 HP, 0 items, and no backups
        state = SimpleNamespace(
            battle_mode=2,
            num_items=0,
            party_count=1,
            party_species=(155,),
            party_hp=((2, 20),),
            party_levels=(5,),
            party_moves=((33, 0, 0, 0),),
            enemy_species=152,
            enemy_hp=10,
            enemy_max_hp=20,
        )
        tactical_battle_step(state, macros)
        # MUST attack with choose_move, and NEVER call run_away!
        macros.choose_move.assert_called_once_with(1)
        macros.run_away.assert_not_called()
        macros.switch_pokemon.assert_not_called()

    def test_tactical_battle_never_runs_away_in_trainer_battle_with_egg(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Rival battle while carrying mystery egg (has_egg=True, gave_egg=False)
        state = SimpleNamespace(
            battle_mode=2,
            is_trainer_battle=True,
            other_trainer_class=1,  # Rival
            num_items=0,
            party_count=1,
            party_species=(155,),
            party_hp=((3, 20),),
            party_levels=(5,),
            party_moves=((33, 0, 0, 0),),
            enemy_species=152,
            enemy_hp=10,
            enemy_max_hp=20,
            has_event=lambda f: f == "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON",
        )
        tactical_battle_step(state, macros)
        # Even with Mystery Egg and low HP, trainer battles MUST NEVER flee!
        macros.run_away.assert_not_called()
        macros.choose_move.assert_called_once_with(1)





class TrainerItemLoopTests(unittest.TestCase):
    def _state(self, hp):
        return SimpleNamespace(
            battle_mode=2, battle_mon_hp=hp, battle_mon_max_hp=20, party_hp=((hp, 20),), party_levels=(6,),
            cur_battle_mon=0, num_items=3, party_moves=((33, 0, 0, 0),),
        )

    def test_rejected_item_stops_after_two_tries(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        macros.machine.frame = 100
        for _ in range(6):  # HP never rises: the game keeps rejecting the item
            tactical_battle_step(self._state(3), macros)
        self.assertEqual(macros.use_battle_item.call_count, 2)
        self.assertGreaterEqual(macros.choose_move.call_count, 4)

    def test_working_item_keeps_being_used(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        macros.machine.frame = 100
        for hp in (3, 5, 6):  # HP rises each time and stays at or below 30%
            tactical_battle_step(self._state(hp), macros)
        self.assertEqual(macros.use_battle_item.call_count, 3)


class FaintedSwitchLoopTests(unittest.TestCase):
    def test_does_not_repeat_slots_the_game_rejected(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        macros.machine.frame = 500
        # Stale party HP: every slot looks alive, 1/3/5 rank highest. Slots 1, 3, 5 are really fainted,
        # so the active Pokémon stays at 0 HP after each attempted switch.
        hp = ((30, 30), (10, 30), (28, 30), (9, 30), (26, 30), (8, 30))
        state = SimpleNamespace(
            battle_mode=2, battle_mon_hp=0, battle_mon_max_hp=30, party_hp=hp,
            party_levels=(10, 5, 9, 5, 8, 4), cur_battle_mon=0, num_items=0, party_moves=((33, 0, 0, 0),) * 6,
        )
        tried = []
        for _ in range(5):
            tactical_battle_step(state, macros)
            tried.append(macros.switch_fainted_pokemon.call_args.args[0])
        self.assertEqual(tried[:3], [3, 5, 2])
        self.assertNotIn(1, tried)
        self.assertEqual(len(tried), len(set(tried)))

    def test_all_fainted_goes_to_whiteout(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        macros.machine.frame = 900
        state = SimpleNamespace(
            battle_mode=2, battle_mon_hp=0, battle_mon_max_hp=30, party_hp=((0, 30), (0, 30)),
            party_levels=(5, 5), cur_battle_mon=0, num_items=0, party_moves=((33, 0, 0, 0),) * 2,
        )
        tactical_battle_step(state, macros)
        macros.handle_whiteout.assert_called_once()


class ReplacementPickerTests(unittest.TestCase):
    def _state(self, hp, active=0, count=None):
        return SimpleNamespace(
            battle_mode=2, battle_mon_hp=0, battle_mon_max_hp=30, party_hp=hp,
            party_levels=(5,) * len(hp), cur_battle_mon=active, num_items=0,
            party_moves=((33, 0, 0, 0),) * len(hp), party_count=count if count is not None else len(hp),
        )

    def test_ignores_padding_slots_past_party_count(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        macros.machine.frame = 10
        # Slots 4-6 are leftover bytes, not real Pokémon
        state = self._state(((0, 20), (12, 20), (0, 20), (99, 99), (99, 99), (99, 99)), count=3)
        tactical_battle_step(state, macros)
        macros.switch_fainted_pokemon.assert_called_once_with(2)

    def test_never_whiteout_loops_while_someone_still_shows_hp(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        macros.machine.frame = 20
        state = self._state(((0, 20), (12, 20)))
        for _ in range(6):  # the game keeps rejecting slot 2
            tactical_battle_step(state, macros)
        macros.handle_whiteout.assert_not_called()
        self.assertEqual(macros.switch_fainted_pokemon.call_count, 6)


class RivalTrackerTests(unittest.TestCase):
    def test_fight_then_win_latches(self):
        from src.live_status import RIVAL_MAP, RivalTracker

        def st(mode, fainted=False):
            return SimpleNamespace(map_group=RIVAL_MAP[0], map_number=RIVAL_MAP[1], battle_mode=mode, fainted=fainted)

        t = RivalTracker()
        t.update(st(0))
        self.assertEqual(t.label, "-")
        t.update(st(2))
        self.assertEqual(t.label, "FIGHT")
        t.update(st(0))
        self.assertEqual(t.label, "beaten")

    def test_loss_does_not_count(self):
        from src.live_status import RIVAL_MAP, RivalTracker

        t = RivalTracker()
        t.update(SimpleNamespace(map_group=RIVAL_MAP[0], map_number=RIVAL_MAP[1], battle_mode=2, fainted=False))
        t.update(SimpleNamespace(map_group=RIVAL_MAP[0], map_number=RIVAL_MAP[1], battle_mode=0, fainted=True))
        self.assertFalse(t.beaten)


class FalknerPrepBattleTests(unittest.TestCase):
    def test_ignores_mystery_egg_as_potion(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Active mon at 3/20 HP (low), pack only has Mystery Egg (ID 18)
        state = SimpleNamespace(
            battle_mode=2,
            num_items=1,
            items=(18,),
            party_count=1,
            party_species=(155,),
            party_hp=((3, 20),),
            party_levels=(8,),
            party_moves=((33, 0, 0, 0),),
            enemy_species=21,  # Spearow
            enemy_hp=15,
            enemy_max_hp=27,
        )
        tactical_battle_step(state, macros)
        # Must NOT call use_battle_item because 18 is not a healing item!
        macros.use_battle_item.assert_not_called()
        macros.choose_move.assert_called_once()

    def test_uses_verified_potion(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Pack has Mystery Egg (18) at slot 1, Potion (20) at slot 2
        state = SimpleNamespace(
            battle_mode=2,
            num_items=2,
            items=(18, 20),
            party_count=1,
            party_species=(155,),
            party_hp=((3, 20),),
            party_levels=(8,),
            party_moves=((33, 0, 0, 0),),
            enemy_species=21,
            enemy_hp=15,
            enemy_max_hp=27,
        )
        tactical_battle_step(state, macros)
        # Should target slot 2 where the Potion is!
        macros.use_battle_item.assert_called_once_with(item_slot=2, target_slot=1)

    def test_catches_bellsprout_en_route_to_falkner(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        state = SimpleNamespace(
            battle_mode=1,
            num_balls=5,
            party_count=5,
            party_species=(155, 161, 187, 16, 165),
            party_hp=((25, 25), (13, 13), (15, 15), (15, 15), (15, 15)),
            party_levels=(8, 2, 3, 3, 3),
            party_moves=((33, 43, 108, 0),),
            enemy_species=69,  # Bellsprout!
            enemy_hp=4,
            enemy_max_hp=15,
            has_event=lambda ev: ev == "EVENT_GAVE_MYSTERY_EGG_TO_ELM",
        )
        tactical_battle_step(state, macros)
        macros.use_pokeball.assert_called_once()

    def test_trains_cyndaquil_before_ember(self):
        from unittest.mock import MagicMock
        from src.battle import tactical_battle_step

        macros = MagicMock()
        # Wild battle with Pidgey; Cyndaquil is Lv 8 and doesn't know Ember yet
        state = SimpleNamespace(
            battle_mode=1,
            num_balls=0,
            party_count=5,
            party_species=(155, 161, 187, 16, 165),
            party_hp=((25, 25), (13, 13), (15, 15), (15, 15), (15, 15)),
            party_levels=(8, 2, 3, 3, 3),
            party_moves=((33, 43, 108, 0),),
            enemy_species=16,  # Pidgey
            enemy_hp=12,
            enemy_max_hp=15,
            has_event=lambda ev: ev == "EVENT_GAVE_MYSTERY_EGG_TO_ELM",
        )
        tactical_battle_step(state, macros)
        # Should attack to gain EXP, NOT run away!
        macros.run_away.assert_not_called()
        macros.choose_move.assert_called_once()
