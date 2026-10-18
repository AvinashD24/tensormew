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
                continue
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
        self.machine.tick(12, render=False)
        frames += 20
        # Open PKMN menu
        self.machine.press("a", frames=8)
        self.machine.tick(24, render=False)
        frames += 32
        # Reset party cursor to slot 1 (top)
        for _ in range(5):
            self.machine.press("up", frames=6)
            self.machine.tick(6, render=False)
            frames += 12
        for _ in range(slot - 1):
            self.machine.press("down", frames=8)
            self.machine.tick(8, render=False)
            frames += 16
        # Select Pokémon and confirm switch
        self.machine.press("a", frames=8)
        self.machine.tick(16, render=False)
        self.machine.press("a", frames=8)
        self.machine.tick(32, render=False)
        frames += 48
        return MacroResult(True, f"switched to party slot {slot}", frames=frames)

    def switch_fainted_pokemon(self, slot: int = 2) -> MacroResult:
        """Switch to an alive party Pokémon after active Pokémon has fainted."""
        state = self._state()
        if state.battle_mode == 0:
            return MacroResult(False, "switch_fainted_pokemon refused: not in battle")
        if slot < 1 or slot > 6:
            return MacroResult(False, f"party slot {slot} not in 1–6")
        frames = 0
        # 1. Advance through faint dialog and confirm YES on 'Use next POKéMON?'
        for _ in range(2):
            self.machine.press("a", frames=6)
            self.machine.tick(16, render=False)
            frames += 22
        # Allow party menu to fully load/fade in
        self.machine.tick(32, render=False)
        frames += 32
        # 2. Dismiss any text popup (e.g. "There's no will to fight!" if A was pressed on fainted mon)
        self.machine.press("b", frames=6)
        self.machine.tick(16, render=False)
        frames += 22
        # 3. In party menu, reset cursor to top (slot 1)
        for _ in range(5):
            self.machine.press("up", frames=6)
            self.machine.tick(10, render=False)
            frames += 16
        # 4. Navigate down to the target alive Pokémon (1-indexed slot)
        for _ in range(slot - 1):
            self.machine.press("down", frames=8)
            self.machine.tick(14, render=False)
            frames += 22
        # 5. Press A to select the Pokémon
        self.machine.press("a", frames=8)
        self.machine.tick(24, render=False)
        frames += 32
        # 6. Press A to confirm SWITCH (top option in popup)
        self.machine.press("a", frames=8)
        self.machine.tick(48, render=False)
        frames += 56
        # 7. Clear entrance dialog text ('<PLAYER> sent out <MON>!')
        for _ in range(4):
            self.machine.press("b", frames=6)
            self.machine.tick(16, render=False)
            frames += 22
        return MacroResult(True, f"switched fainted Pokémon to party slot {slot}", frames=frames)

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




