"""Map supervisor tool names onto the live mover and macros."""

from __future__ import annotations

from src.macros import Macros
from src.mover import Mover


def execution_registry(machine, reader, *, mover: Mover | None = None, macros: Macros | None = None) -> dict:
    mover = mover or Mover(machine, reader)
    macros = macros or Macros(machine, reader)
    return {
        "go_to": mover.go_to,
        "run_path": mover.run_path,
        "use_cut": macros.use_cut,
        "mash_text": macros.mash_text,
        "choose_move": macros.choose_move,
        "switch_pokemon": macros.switch_pokemon,
        "use_pokeball": macros.use_pokeball,
    }

