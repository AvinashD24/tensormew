"""Button sequences. No LLM. No Start/Select except party switch in battle."""

from __future__ import annotations

from src.action_results import MacroResult
from src.collision import is_cut_tree
from src.ram_parser import dialogue_open

DIRS = ("up", "down", "left", "right")


class Macros:
    def __init__(self, machine, reader):
        self.machine = machine
        self.reader = reader

    def _state(self):
        return self.reader.parse(self.machine.frame)

    def mash_text(self, max_frames: int = 1800) -> MacroResult:
        """Advance an open script until wScriptMode == 0."""
        used = 0
        presses = 0
        if not dialogue_open(self._state()):
            return MacroResult(True, "no dialogue open", frames=0)
        while used < max_frames:
            if not dialogue_open(self._state()):
                self.machine.tick(12, render=False)
                used += 12
                if not dialogue_open(self._state()):
                    return MacroResult(True, f"text clear after {presses} presses", frames=used)
            # Gen 2 Naming Screen (e.g. Rival naming in Elm's Lab): pressing A just enters
            # letters into the name. START jumps cursor to 'END', and A confirms the default name.
            has_mem = hasattr(self.machine, "symbol") and hasattr(self.machine, "read")
            if has_mem:
                try:
                    b_name, a_name = self.machine.symbol("wNamingScreenType")
                    if self.machine.read(b_name, a_name, 1)[0] != 0:
                        self.machine.press("start", frames=6)
                        self.machine.tick(16, render=False)
                        self.machine.press("a", frames=6)
                        self.machine.tick(24, render=False)
                        used += 52
                        presses += 1
                        continue
                except Exception:
                    pass

            self.machine.press("a", frames=4)
            self.machine.tick(24, render=False)
            used += 28
            presses += 1
        if not dialogue_open(self._state()):
            return MacroResult(True, f"text clear after {presses} presses", frames=used)
        return MacroResult(False, f"text still open after {max_frames} frames", frames=used)

    def use_cut(self) -> MacroResult:
        state = self._state()
        if state.battle_mode != 0:
            return MacroResult(False, "use_cut refused: in battle")
        facing_tree = [
            side for side in DIRS if is_cut_tree(state.tiles.get(side, 0xFF))
        ]
        if not facing_tree:
            return MacroResult(False, "no adjacent Cut tree")
        direction = state.facing if state.facing in facing_tree else facing_tree[0]
        frames = 0
        if state.facing != direction:
            self.machine.press(direction, frames=8)
            self.machine.tick(8, render=False)
            frames += 16
        # Gen 2: A facing a Cut tree → "use CUT?" → A confirms.
        self.machine.press("a", frames=8)
        self.machine.tick(40, render=False)
        self.machine.press("a", frames=8)
        self.machine.tick(90, render=False)
        frames += 146
        after = self._state()
        still = [side for side in DIRS if is_cut_tree(after.tiles.get(side, 0xFF))]
        if still and after.facing in still and after.tiles.get(after.facing) == state.tiles.get(direction):
            return MacroResult(False, "Cut tree still present", frames=frames)
        return MacroResult(True, f"used Cut facing {direction}", frames=frames)

    def _reset_to_battle_fight(self) -> int:
        """Back out of open battle submenus and reset cursor to FIGHT (top-left)."""
        frames = 0
        # Press B twice to cancel any open submenus (PKMN, PACK, STATS, etc.)
        for _ in range(2):
            self.machine.press("b", frames=6)
            self.machine.tick(8, render=False)
            frames += 14
        # Press UP then LEFT to ensure top-left FIGHT is selected
        self.machine.press("up", frames=6)
        self.machine.tick(8, render=False)
        self.machine.press("left", frames=6)
        self.machine.tick(8, render=False)
        frames += 28
        return frames

    def choose_move(self, slot: int | None = None) -> MacroResult:
        state = self._state()
        if state.battle_mode == 0:
            return MacroResult(False, "choose_move refused: not in battle")
        if slot is None:
            from src.battle import pick_best_move
            slot = pick_best_move(state)
        if slot < 1 or slot > 4:
            return MacroResult(False, f"move slot {slot} not in 1–4")
        frames = self._reset_to_battle_fight()
        # At FIGHT: press A to open move list
        self.machine.press("a", frames=8)
        self.machine.tick(16, render=False)
        frames += 24
        for _ in range(slot - 1):
            self.machine.press("down", frames=8)
            self.machine.tick(8, render=False)
            frames += 16
        self.machine.press("a", frames=8)
        self.machine.tick(32, render=False)
        frames += 40
        return MacroResult(True, f"chose move slot {slot}", frames=frames)

    def switch_pokemon(self, slot: int) -> MacroResult:
        if slot < 1 or slot > 6:
            return MacroResult(False, f"party slot {slot} not in 1–6")
        state = self._state()
        if state.battle_mode == 0:
            return MacroResult(False, "switch_pokemon refused: not in battle")
        frames = self._reset_to_battle_fight()
        # From FIGHT (top-left), press RIGHT to move to PKMN (top-right)
        self.machine.press("right", frames=8)
        self.machine.tick(16, render=False)
        frames += 24
        # Open PKMN menu
        self.machine.press("a", frames=8)
        self.machine.tick(36, render=False)
        frames += 44

        has_mem = hasattr(self.machine, "read")
        if has_mem:
            cur_y = self.machine.read(0, 0xcfa9, 1)[0]
            for _ in range(6):
                if cur_y <= slot:
                    break
                self.machine.press("up", frames=8)
                self.machine.tick(22, render=False)
                frames += 30
                cur_y = self.machine.read(0, 0xcfa9, 1)[0]
            for _ in range(6):
                if cur_y >= slot:
                    break
                self.machine.press("down", frames=8)
                self.machine.tick(22, render=False)
                frames += 30
                cur_y = self.machine.read(0, 0xcfa9, 1)[0]
        else:
            for _ in range(5):
                self.machine.press("up", frames=8)
                self.machine.tick(16, render=False)
                frames += 24
            for _ in range(slot - 1):
                self.machine.press("down", frames=8)
                self.machine.tick(16, render=False)
                frames += 24

        # Select Pokémon and confirm switch
        self.machine.press("a", frames=8)
        self.machine.tick(32, render=False)
        frames += 40
        self.machine.press("a", frames=8)
        self.machine.tick(56, render=False)
        frames += 64
        # Clear entrance dialogue
        for _ in range(3):
            self.machine.press("b", frames=6)
            self.machine.tick(20, render=False)
            frames += 26
        return MacroResult(True, f"switched to party slot {slot}", frames=frames)

    def _party_cursor(self) -> int:
        try:
            val = self.machine.read(1, 0xD0D8, 1)[0]
            if val != 0:
                return val
        except Exception:
            pass
        try:
            return self.machine.read(0, 0xCFA9, 1)[0]
        except Exception:
            return 0

    def _move_party_cursor(self, slot: int) -> int:
        """Walk the party-menu cursor to `slot` (1-indexed) using up/down, checking the cursor each press."""
        frames = 0
        cur_y = self._party_cursor()
        for _ in range(8):
            if cur_y == slot or cur_y == 0:
                break
            self.machine.press("up" if cur_y > slot else "down", frames=8)
            self.machine.tick(22, render=False)
            frames += 30
            new_y = self._party_cursor()
            if new_y == cur_y:  # cursor did not move: menu is not where we think it is
                break
            cur_y = new_y
        return frames

    def switch_fainted_pokemon(self, slot: int = 2) -> MacroResult:
        """Bring in a Pokémon that can fight after the active one fainted.

        Tries `slot` first, then the other party slots that show HP, then every remaining slot, one
        after another with the arrow keys and A. After each try the battle state is read back: the
        switch only counts once the battle Pokémon has HP. A rejected pick ("no will to fight") is
        dismissed and the next slot is tried, so the party menu can no longer be left stuck on one
        Pokémon.
        """
        state = self._state()
        if state.battle_mode == 0:
            return MacroResult(False, "switch_fainted_pokemon refused: not in battle")
        if slot < 1 or slot > 6:
            return MacroResult(False, f"party slot {slot} not in 1–6")
        frames = 0

        if not hasattr(self.machine, "read"):
            # Fallback for mock environments
            self.machine.press("b", frames=6)
            self.machine.tick(16, render=False)
            frames += 22
            for _ in range(5):
                self.machine.press("up", frames=8)
                self.machine.tick(16, render=False)
                frames += 24
            for _ in range(slot - 1):
                self.machine.press("down", frames=8)
                self.machine.tick(16, render=False)
                frames += 24
            self.machine.press("a", frames=8)
            self.machine.tick(32, render=False)
            self.machine.press("a", frames=8)
            self.machine.tick(56, render=False)
            frames += 96
            return MacroResult(True, f"switched fainted Pokémon to party slot {slot}", frames=frames)

        # Dismiss any open text modal (e.g. "There's no will to fight!")
        self.machine.press("b", frames=6)
        self.machine.tick(16, render=False)
        frames += 22
        # If the party menu is not open yet (faint message / 'Use next POKéMON?'), advance to it
        if self._party_cursor() == 0:
            for _ in range(3):
                self.machine.press("a", frames=6)
                self.machine.tick(16, render=False)
                frames += 22
            for _ in range(6):
                self.machine.tick(16, render=False)
                frames += 16
                if self._party_cursor() > 0:
                    break

        count = getattr(state, "party_count", 0) or len(getattr(state, "party_hp", ()) or ()) or 6
        count = max(1, min(6, count))
        party_hp = tuple(getattr(state, "party_hp", ()) or ())
        active = getattr(state, "cur_battle_mon", 0) + 1
        with_hp = [
            s for s in range(1, count + 1)
            if s != active and s - 1 < len(party_hp) and party_hp[s - 1][0] > 0
        ]
        rest = [s for s in range(1, count + 1) if s != active and s not in with_hp]
        order = [slot] + [s for s in with_hp if s != slot] + [s for s in rest if s != slot]

        for attempt, cand in enumerate(order):
            cur_c = self._party_cursor()
            if cur_c > 0:
                frames += self._move_party_cursor(cand)
            else:
                # Blind fallback: reset to top with UP x5, then step DOWN (cand - 1) times
                for _ in range(5):
                    self.machine.press("up", frames=6)
                    self.machine.tick(16, render=False)
                    frames += 22
                for _ in range(cand - 1):
                    self.machine.press("down", frames=6)
                    self.machine.tick(16, render=False)
                    frames += 22

            self.machine.press("a", frames=8)   # select the Pokémon
            self.machine.tick(32, render=False)
            self.machine.press("a", frames=8)   # SWITCH (top option); harmless if no submenu opened
            self.machine.tick(56, render=False)
            frames += 96
            after = self._state()
            if after.battle_mode == 0 or getattr(after, "battle_mon_hp", 0) > 0:
                # Clear the '<PLAYER> sent out <MON>!' text
                for _ in range(4):
                    self.machine.press("b", frames=6)
                    self.machine.tick(20, render=False)
                    frames += 26
                return MacroResult(True, f"switched fainted Pokémon to party slot {cand}", frames=frames)
            # Rejected: close the message / submenu and move on to the next slot
            for _ in range(2):
                self.machine.press("b", frames=6)
                self.machine.tick(16, render=False)
                frames += 22
        return MacroResult(False, f"no party Pokémon could be sent out (tried {order})", frames=frames)

    def handle_whiteout(self) -> MacroResult:
        """Dismiss blackout / whiteout text when entire party has fainted."""
        frames = 0
        for _ in range(12):
            self.machine.press("a", frames=6)
            self.machine.tick(16, render=False)
            frames += 22
            self.machine.press("b", frames=6)
            self.machine.tick(16, render=False)
            frames += 22
        return MacroResult(True, "cleared whiteout dialog", frames=frames)

    def use_battle_item(self, item_slot: int = 1, target_slot: int = 1) -> MacroResult:
        """Navigate PACK during battle to use an item (e.g. Potion) on a party Pokémon."""
        state = self._state()
        if state.battle_mode == 0:
            return MacroResult(False, "use_battle_item refused: not in battle")
        if state.num_items <= 0:
            return MacroResult(False, "use_battle_item refused: no items in pack")
        frames = self._reset_to_battle_fight()
        # From FIGHT (top-left), press DOWN to move to PACK (bottom-left)
        self.machine.press("down", frames=8)
        self.machine.tick(16, render=False)
        frames += 24
        # Open PACK
        self.machine.press("a", frames=8)
        self.machine.tick(32, render=False)
        frames += 40
        # Ensure we are in ITEM POCKET: press left twice
        for _ in range(2):
            self.machine.press("left", frames=6)
            self.machine.tick(12, render=False)
            frames += 18
        # Reset item cursor to top
        for _ in range(5):
            self.machine.press("up", frames=6)
            self.machine.tick(8, render=False)
            frames += 14
        # Navigate to item_slot
        for _ in range(item_slot - 1):
            self.machine.press("down", frames=6)
            self.machine.tick(8, render=False)
            frames += 14
        # Select item
        self.machine.press("a", frames=8)
        self.machine.tick(20, render=False)
        frames += 28
        # Confirm "USE"
        self.machine.press("a", frames=8)
        self.machine.tick(28, render=False)
        frames += 36
        # Select target Pokémon: reset to slot 1
        for _ in range(5):
            self.machine.press("up", frames=6)
            self.machine.tick(8, render=False)
            frames += 14
        for _ in range(target_slot - 1):
            self.machine.press("down", frames=6)
            self.machine.tick(8, render=False)
            frames += 14
        # Apply item to target Pokémon
        self.machine.press("a", frames=8)
        self.machine.tick(80, render=False)
        frames += 88
        # Dismiss text ("<MON> recovered HP!")
        for _ in range(3):
            self.machine.press("b", frames=6)
            self.machine.tick(12, render=False)
            frames += 18
        return MacroResult(True, f"used item slot {item_slot} on party slot {target_slot}", frames=frames)

    def use_pokeball(self) -> MacroResult:
        state = self._state()
        if state.battle_mode == 0:
            return MacroResult(False, "use_pokeball refused: not in battle")
        if state.num_balls <= 0:
            return MacroResult(False, "use_pokeball refused: no Pokéballs in pack")
        frames = self._reset_to_battle_fight()
        # From FIGHT (top-left), press DOWN to move to PACK (bottom-left)
        self.machine.press("down", frames=8)
        self.machine.tick(16, render=False)
        frames += 24
        # Open PACK
        self.machine.press("a", frames=8)
        self.machine.tick(32, render=False)
        frames += 40
        # Switch from ITEM POCKET right to BALL POCKET
        self.machine.press("right", frames=8)
        self.machine.tick(20, render=False)
        frames += 28
        # Select first Ball
        self.machine.press("a", frames=8)
        self.machine.tick(20, render=False)
        frames += 28
        # Confirm "USE"
        self.machine.press("a", frames=8)
        self.machine.tick(160, render=False)
        frames += 168
        return MacroResult(True, "used Pokéball", frames=frames)

    def run_away(self) -> MacroResult:
        state = self._state()
        if state.battle_mode == 0:
            return MacroResult(False, "run_away refused: not in battle")
        if state.battle_mode == 2 or getattr(state, "is_trainer_battle", False) or (getattr(state, "other_trainer_class", 0) > 0):
            return MacroResult(False, "run_away refused: cannot run from a trainer battle")
        frames = self._reset_to_battle_fight()
        # From FIGHT (top-left), move down to PACK, then right to RUN (bottom-right)
        self.machine.press("down", frames=8)
        self.machine.tick(16, render=False)
        frames += 24
        self.machine.press("right", frames=8)
        self.machine.tick(16, render=False)
        frames += 24
        # Select RUN
        self.machine.press("a", frames=8)
        self.machine.tick(48, render=False)
        frames += 56
        return MacroResult(True, "ran away from battle", frames=frames)




