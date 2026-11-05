# Coding Conventions

**Analysis Date:** 2026-11-05

Scope: the Python package (`src/`, `tools/`, `app.py`, `crystal_smoke.py`) and the React frontend in `tensormewweb/`. Python is the bulk of the repo; the frontend follows its own, simpler conventions (see "Frontend (tensormewweb)" below).

## Naming Patterns

**Files:**
- Python modules are lowercase `snake_case.py`, named after the concept they hold: `src/battle.py`, `src/run_log.py`, `src/tool_dispatch.py`, `src/return_nav.py`.
- CLI entry points live in `tools/` and are also `snake_case.py`: `tools/run_episode.py`, `tools/train_ppo.py`, `tools/pod_checkpoint.py`.
- Tests are `tests/test_<module>.py`, mirroring the module they cover (`src/ppo.py` -> `tests/test_ppo.py`, `src/nav_memory.py` -> `tests/test_nav_memory.py`).
- React components are `PascalCase.jsx` (`tensormewweb/src/components/PokedexTab.jsx`, `tensormewweb/src/components/ui/XpBar.jsx`). Data modules are `camelCase.js` (`tensormewweb/src/data/pokedexData.js`).

**Functions:**
- `snake_case`, verbs for actions: `pick_best_move` (`src/battle.py`), `format_user_message` (`src/llm_supervisor.py`), `load_decisions` and `summarize` (`app.py`), `retrieve_checkpoint` (`tools/pod_checkpoint.py`).
- Module-private helpers get a single leading underscore and are not re-exported: `_state`, `_now` (`src/run_log.py`), `_party_hp()` (`src/ram_parser.py`), `_batch_policy_stats` (`src/ppo.py`). About 38 `def _...` helpers exist in `src/`.
- Predicates read as questions: `dialogue_open`, `is_cut_tree`, `needs_training`, `prep_direction` (returns a direction or `None`).
- Converters use `to_` / `from_`: `StateEntry.to_json` / `StateEntry.from_json` (`src/states.py`), `GameState.from_dict` (`src/ram_parser.py`).

**Variables:**
- `snake_case`. Short local names are normal for numeric and array code (`w`, `b`, `vw`, `obs`, `logp`, `kls`) in `src/ppo.py`.
- Constants are `UPPER_SNAKE_CASE` at module top: `TYPE_CHART`, `MOVE_DATA`, `HEALING_ITEMS` (`src/battle.py`), `ROM_SHA1`, `STATES_DIR` (`src/config.py`), `DEFAULT_MODEL` (`src/llm_supervisor.py`), `TARGET_LEVEL` (`src/gym_prep.py`).
- Module-private constants get a leading underscore: `_MAX_FRAMES = 600` (`src/episode.py`), `_NAME = re.compile(...)` (`src/states.py`), `_DIRS` (`src/macros.py`).
- Environment-driven settings are read once at import time with `os.environ.get` and a project prefix: `POKEMIND_MODEL`, `POKEMIND_INPUT_USD_PER_MTOK` (`src/llm_supervisor.py`), `PYBOY_VISUAL`, `PYBOY_SPEED` (`src/emulator.py`).

**Types / Classes:**
- `PascalCase`: `Policy`, `MlpPolicy`, `PpoUpdateStats` (`src/ppo.py`), `GameState` (`src/ram_parser.py`), `EpisodeRunner` (`src/episode.py`), `Supervisor` (`src/llm_supervisor.py`).
- Structural contracts use `typing.Protocol`, not ABCs: `MemoryBus` (`src/ram_parser.py`), `Script` (`src/episode.py`).
- Private helper classes use a leading underscore: `_ItemGuard` (`src/battle.py`), `_Story` in tests.

## Code Style

**Formatting:**
- No formatter is configured. No `pyproject.toml`, `setup.cfg`, `.flake8`, `ruff.toml`, `.editorconfig`, or pre-commit config exists at the repo root.
- Written style, which new code should match: 4-space indent, double quotes for strings, trailing commas in multi-line literals, blank line between top-level definitions.
- Line length is soft. Most lines stay under 100 characters (144 lines in `src/` exceed 100; 19 exceed 120). Longest lines are in `src/ram_parser.py`. Wrap new code at about 100.

**Linting:**
- No linter is configured (no ruff, flake8, pylint, or mypy config). Do not assume a lint step will catch problems; match the surrounding file by hand.
- Frontend has no ESLint or Prettier config in `tensormewweb/` (`tensormewweb/package.json` lists only Vite and React tooling).

**Python version and syntax:**
- Every module under `src/` and most under `tools/` begin with `from __future__ import annotations` (the first code line after the module docstring).
- Use PEP 604 unions (`TriggerContext | str | None`, `int | None`) and builtin generics (`dict[tuple[int, int], float]`, `list[str]`). Do not import `typing.List` / `typing.Dict` / `Optional`.
- Use `Callable` from `typing` for injected callables (`src/ppo.py`, `src/ram_parser.py`) and `collections.abc.Callable` in newer tools (`tools/pod_checkpoint.py`).

## Import Organization

**Order:**
1. Module docstring.
2. `from __future__ import annotations`.
3. Standard library (`json`, `dataclasses`, `pathlib`, `typing`, `datetime`).
4. Third-party (`numpy as np` in `src/ppo.py`). The LLM client is injected into `Supervisor` rather than imported in `src/llm_supervisor.py`.
5. Project imports, always absolute from the `src` package, with one blank line between groups.

Example from `src/ppo.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os

import numpy as np

from src.rl_env import MASK_START, N_ACTIONS, OBS_DIM
```

**Rules observed:**
- Relative imports are never used (zero `from .` imports in `src/`, `tools/`, `tests/`, `app.py`). Write `from src.battle import pick_best_move`, not `from .battle import ...`.
- Imports of heavy optional dependencies are deferred into the function that needs them, so modules import cleanly without a ROM or GPU: `import torch` inside `load_checkpoint` in `tools/pod_checkpoint.py`; `from src.emulator import Emulator` inside a test method in `tests/test_live_execution.py`; `from src.recording import InputRecorder` inside `main` in `tools/run_episode.py`.
- Nothing in the repo uses `import *`.

**Path Aliases:**
- None. Everything resolves from the repo root, which is on `sys.path` when running `python -m tools.<name>` or `pytest` from the root.

## Module Layout

- Module docstring first, one or two sentences, stating what the module does and what it does not do. Examples: `"""Button sequences. No LLM. No Start/Select except party switch in battle."""` (`src/macros.py`); `"""Named save snapshots and the manifest that checks them."""` (`src/states.py`). 40 of 42 `src/` modules have one.
- Constants and regexes next, then exception classes, then dataclasses, then functions and classes.
- Large tables (type charts, move data, prompts) are plain module-level literals with an inline comment on each group, e.g. `TYPE_CHART` in `src/battle.py` and `SYSTEM_PROMPT` in `src/llm_supervisor.py`.

## Data Modeling

**Dataclasses are the default record type:**
- Immutable value objects use `@dataclass(frozen=True)`: `TriggerContext`, `Decision` (`src/llm_supervisor.py`), `Action` (`src/episode.py`), `Sym` (`src/ram_parser.py`).
- Mutable results and accumulators use plain `@dataclass`: `ActionResult`, `MoveResult`, `PathResult`, `MacroResult` (`src/action_results.py`), `PpoUpdateStats` (`src/ppo.py`), `StateEntry` (`src/states.py`).
- Defaults that are mutable use `field(default_factory=...)`: `tags: list[str] = field(default_factory=list)` (`src/states.py`), `rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))` (`src/rl_env.py`).
- Serialization is hand-written per class with `to_json()` / `from_json()` and `dataclasses.asdict`, not a serialization library. Tuples are converted to lists on the way out and back to tuples on the way in (`src/states.py`).

**Dict-shaped data is used at boundaries:**
- JSONL rows, decision logs, and tool arguments stay as `dict` (`load_decisions` returns `list[dict]` in `app.py`). Access fields with `row.get("est_usd")` and coerce with helpers such as `_as_float` / `_as_int` (`app.py`) rather than assuming keys exist.

## Error Handling

The codebase uses four distinct strategies. Pick the one that matches the layer you are editing.

**1. Custom exceptions subclass a built-in base, and carry domain meaning:**
```python
class RomMismatchError(RuntimeError): ...   # src/emulator.py
class SupervisorError(RuntimeError): ...    # src/llm_supervisor.py
class SymbolError(KeyError): ...            # src/ram_parser.py
class StateNotFound(KeyError): ...          # src/states.py
```
Use `KeyError` subclasses for lookups by name and `RuntimeError` subclasses for runtime refusals. Do not add a new base exception hierarchy.

**2. Game actions return result objects instead of raising:**
- Macros, movement, and tool dispatch return `ActionResult` or a subclass with `ok: bool`, `message: str`, and `frames: int` (`src/action_results.py`). Callers branch on `result.ok`.
```python
if state.battle_mode != 0:
    return MacroResult(False, "use_cut refused: in battle")
```
(`src/macros.py`). Prefer this for anything the LLM or a script can trigger, because the message is fed back as the tool result.

**3. Best-effort blocks swallow exceptions on purpose:**
- Reading optional RAM symbols and optional UI state is wrapped in `try: ... except Exception: pass` (`src/macros.py` lines ~46, ~69, ~168, ~217; `src/emulator.py` ~89; `src/terminal_monitor.py` ~234). Use this only when a failure is expected and the fallback is a safe default. Add a comment saying why.
- Per-item loops that should skip bad inputs use `continue` (`src/visual_metrics.py`).

**4. Run-level errors are captured as data, not propagated:**
- Long-running episodes and segments catch `Exception as exc`, store `str(exc)` in a result or log row, and mark `outcome="error"` (`src/episode.py` ~L177, `src/segment.py` ~L246). The run log (`src/run_log.py`) records `error TEXT` on the `runs` table.

**Narrow excepts for parsing and I/O:**
- Parse errors and I/O errors are caught by name: `except json.JSONDecodeError: continue` (`app.py`), `except (OSError, json.JSONDecodeError, UnicodeError):` (`src/knowledge.py`), `except (OSError, KeyError, RuntimeError):` (`src/return_nav.py`). Prefer this over bare `except Exception` whenever the failure modes are known.
- Bare `except:` is not used anywhere in the repo. Do not introduce it.

**Validation:**
- Invalid input raises `ValueError` with the offending value in the message: `raise ValueError(f"checkpoint {path!r} has no state_dict")` (`tools/pod_checkpoint.py`). `RuntimeError` is used for environment or invariant failures.
- Guard clauses at the top of a function are the norm; deep nesting is avoided.

## Logging

**Framework:** None. The `logging` module is not used. Output goes to `print` or to a structured store.

**Patterns:**
- Human-facing live output uses `print(..., flush=True)` so terminal monitors update immediately (`src/terminal_monitor.py`, `src/visual_metrics.py`).
- CLI error messages go to `sys.stderr`: `print("Failed to sync codebase to pod.", file=sys.stderr)` (`tools/cloud_train.py`).
- Structured run history goes to SQLite (`logs/runs.db`, `src/run_log.py`) and JSONL (`logs/decisions.jsonl`, `src/agent_loop.py`, `app.py`). Prefer writing a row over printing when the data is meant to be analyzed later.
- Relative output paths are used (`Path("logs/runs.db")`, `Path("logs") / "live_status"`). All of `logs/`, `runs/`, and `*.state` outputs are gitignored by `.gitignore`.

## Comments

**When to comment:**
- Explain *why* a non-obvious constant or rule exists, especially when it encodes game knowledge or a past failure. Examples: `# Ground vs Flying is immune (0.0x) - Mud-slap against Falkner does 0!` (`src/battle.py`); `# wTextboxFlags bit 0 stays set for fast text. An open box is the script engine.` (`src/ram_parser.py`).
- Map game RAM and ROM facts to their source name (`wPartyMenuActionText`, `wNamingScreenType`) so the reader can find them in `rom/pokemon_crystal.sym`.
- Use section comments inside long literal tables to group entries.

**Avoid:**
- Restating what the code does. `# Slot 1: Tackle` inside a test is fine because it documents fixture data; `# increment counter` above `x += 1` is not.

**JSDoc/TSDoc:** Not used. Python uses docstrings instead.

## Docstrings

- Public functions and classes have a one-line docstring in imperative or descriptive form: `"""Score a move against enemy defending types (default Flying/Normal for Falkner)."""`, `"""Read a decisions JSONL file. Missing files yield []. Bad lines are skipped."""` (`app.py`).
- Multi-line docstrings are used when behavior has edge cases, with a blank line after the summary. Example: `_ItemGuard` in `src/battle.py` explains the "two tries then block" rule.
- Dataclasses and Protocols often carry a one-line docstring describing their role: `"""Diagnostics from one ``ppo_update`` call."""` (`src/ppo.py`). Double backticks are used for code references in docstrings.
- Private helpers frequently have no docstring; the name carries the meaning.

## Function Design

**Size:** Most functions are short (under about 40 lines). Long procedural code lives in the `tools/` CLI scripts and in `src/rl_env.py`; those are the exception, not the template.

**Parameters:**
- Keyword-only arguments for flags and config in public constructors and helpers: `Supervisor(*, client, always_on, log_path, ...)` (`src/llm_supervisor.py`), `retrieve_checkpoint(host=..., port=..., key=..., run=..., loader=..., clock=...)` (`tools/pod_checkpoint.py`), `clip` in `_batch_policy_stats(..., *, clip)` (`src/ppo.py`).
- Defaults are literal values and `None` for optional collaborators. Mutable defaults are not used.
- Inject side-effecting collaborators as parameters so tests can replace them: a `run` callable for subprocesses, a `loader` for file I/O, a `clock` for time, an `rng` for randomness (`src/rl_env.py` takes `rng: np.random.Generator`). See "Dependency injection" in `TESTING.md`.
- Hardware and emulator objects are duck-typed. `Macros(machine, reader)` (`src/macros.py`) accepts anything with `press`, `tick`, `frame`, and `read`; do not add `isinstance` checks against the emulator class.

**Return Values:**
- Return `bool` for simple predicates, `int` for exit codes from `main`, `None` for "not found" in lookups (`prep_direction`, `exits_from` returns `[]` rather than `None`).
- Prefer returning empty collections over `None` for list-producing functions (`load_decisions` returns `[]` when the file is missing).
- Tuples are used for small fixed records such as coordinates and `(map_group, map_number)` pairs. Use `tuple(...)` for hashable map keys (`seen_maps` in `src/rl_env.py`).

**CLI entry points:**
Every `tools/*.py` script that is run directly follows this shape:
```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a snapshot forward")
    parser.add_argument("--state", required=True)
    ...
    args = parser.parse_args(argv)
    ...
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
Taking `argv=None` lets tests call `main([...])` directly. Return an integer exit code; do not call `sys.exit` inside `main`.

## Module Design

**Exports:**
- No `__all__` lists are defined. Names are public by default unless prefixed with `_`.
- `src/__init__.py`, `tests/__init__.py`, and `tools/__init__.py` exist but are empty. Import from the defining module (`from src.ram_parser import GameState`), not from a package.

**Barrel Files:** Not used. Do not add re-export `__init__.py` files.

**Cross-module dependencies:**
- `src/` modules depend downward on `src/config.py`, `src/collision.py`, `src/ram_parser.py`, and `src/action_results.py`. `src/ppo.py` imports constants from `src/rl_env.py`; keep constants in the lower-level module to avoid cycles.
- Tests import the module under test plus shared constants (`from src.config import ROM_PATH, SYM_PATH`).

## Frontend (tensormewweb)

Stack: React 19 with Vite 6 (`tensormewweb/package.json`, `tensormewweb/vite.config.js`). No TypeScript, no ESLint, no Prettier, no tests.

**Components:**
- Function components only, each in its own file, with a default export named the same as the file: `export default function PokedexTab() {...}` (`tensormewweb/src/components/PokedexTab.jsx`). Named exports are used only for data and singletons (`export const soundEffects = new RetroAudio();`, `tensormewweb/src/utils/soundEffects.js`).
- Props are destructured in the signature: `export default function Navbar({ activeTab, setActiveTab, isMuted, setIsMuted })` (`tensormewweb/src/components/Navbar.jsx`).
- Local UI state with `useState`, named as the `[value, setValue]` pair. Event handlers are named `handleX` (`handleMonClick`, `handleCloseModal` in `PokedexTab.jsx`).
- Reusable pieces that are not tabs live in `tensormewweb/src/components/ui/` (`XpBar.jsx`).

**Data:**
- Static content lives in `tensormewweb/src/data/*.js` as `export const UPPER_SNAKE` arrays (`POKEDEX_DATA`, `BLOG_POSTS`, `SCREENSHOTS_DATA`). Components import and filter them; they do not fetch.

**Styling:**
- Single global stylesheet `tensormewweb/src/index.css`. Class names are kebab-case (`app-container`, `main-content`, `site-footer`, `footer-inner`). Use `className=`, not inline style, for anything reusable; inline `style={{...}}` appears only for one-off values.

**Syntax:**
- Double quotes, semicolons at the end of statements, JSX comments in `{/* ... */}` form (`tensormewweb/src/App.jsx`).
- Icons come from `lucide-react` (`import { Search, CheckCircle2, X } from "lucide-react";`).
- Imports are relative within the frontend (`./components/Navbar`, `../data/pokedexData`) with the file extension omitted.

---

*Convention analysis: 2026-11-05*
