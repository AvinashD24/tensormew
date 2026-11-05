# Testing Patterns

**Analysis Date:** 2026-11-05

## Test Framework

**Runner:**
- Python `unittest` (standard library). Every test file is written as `unittest.TestCase` subclasses, so the suite runs under both `unittest` and `pytest`.
- `pytest` is the documented runner (`README.md` line 18: `pytest tests/ -q`; `SPEC.md` line 112), but it is not declared in `requirements.txt` and is not installed in the local interpreter checked for this analysis. Install it separately (`pip install pytest`) before following the documented command.
- Config: none. No `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`, or `conftest.py` exists. Test discovery uses defaults, which pick up `tests/test_*.py`.
- Python used locally: 3.14.3.

**Assertion Library:**
- `unittest` built-ins only: `assertEqual`, `assertTrue`, `assertFalse`, `assertIn`, `assertNotIn`, `assertGreater`, `assertIsInstance`, `assertRaises`. No `pytest` assertion rewriting or third-party matchers.
- `numpy.testing` is not used. Compare float arrays by value with `assertEqual` on scalars, or by converting to lists.

**Run Commands:**
```bash
pytest tests/ -q                                  # Run all tests (documented; needs pytest installed)
python -m unittest discover -t . -s tests -q      # Run all tests with the stdlib runner
python -m unittest tests.test_macros tests.test_battle -q   # Run specific modules (verified: 49 tests OK)
pytest tests/test_ppo.py -q                       # Run one module (as used in .planning/ROADMAP.md)
python crystal_smoke.py --headless                # Live smoke check, needs ROM (see SPEC.md line 266)
```
- Run from the repository root. Tests import `src.*`, `tools.*`, and `app` as top-level packages.
- Watch mode and coverage commands are not configured.

## Test File Organization

**Location:**
- All tests live in a single top-level directory, `tests/`, separate from source. Tests are not co-located with modules.
- `tests/__init__.py` exists (empty), so `tests` is a package and `python -m unittest discover -t .` resolves imports from the root.

**Naming:**
- File: `tests/test_<module>.py`, where `<module>` matches the `src/` or `tools/` module under test. Examples: `tests/test_battle.py` -> `src/battle.py`; `tests/test_pod_checkpoint.py` -> `tools/pod_checkpoint.py`; `tests/test_falkner_nav.py` -> `src/falkner_nav.py`.
- Some files are cross-cutting and named by behavior, not module: `tests/test_curriculum_and_recurrent_training.py`, `tests/test_collision_directional.py`, `tests/test_egg_rival_elm_policy.py`, `tests/test_live_execution.py`, `tests/test_policy_walk.py`.
- Test class: `<Subject>Tests(unittest.TestCase)`, e.g. `BattleTests`, `RewardTests`, `NavMemoryTests`, `LoadDecisionsTests`, `PodCheckpointTests`. About 68 `TestCase` classes exist across 42 files.
- Test method: `test_<behavior in snake_case>`, phrased as the outcome: `test_malformed_line_is_skipped`, `test_missing_file_returns_empty`, `test_repeat_increments_seen_count`, `test_tactical_battle_throws_ball_when_weakened`.

**Structure:**
```
tests/
├── __init__.py
├── test_<module>.py          # one file per src/ or tools/ module (most)
├── test_<behavior>.py        # cross-cutting behavior
└── __pycache__/              # generated, ignored by .gitignore
```
- Test output directories are written under `logs/test/...` relative to the repo root (`tests/test_nav_memory.py`, `tests/test_chain.py`, `tests/test_dispatch.py`). The `logs/` directory is gitignored.

## Test Structure

**Suite Organization:**
```python
import unittest
from types import SimpleNamespace

from src.battle import pick_best_move, score_move, type_multiplier


class BattleTests(unittest.TestCase):
    def test_type_multipliers(self):
        # Ground (4) vs Flying (2) is immune (0.0x)
        self.assertEqual(type_multiplier(4, 2), 0.0)
        # Electric (12) vs Flying (2) is super effective (2.0x)
        self.assertEqual(type_multiplier(12, 2), 2.0)

    def test_pick_best_move_slot(self):
        state = SimpleNamespace(party_moves=((33, 43, 0, 0),))
        self.assertEqual(pick_best_move(state, enemy_types=(2, 0)), 1)


if __name__ == "__main__":
    unittest.main()
```
(Pattern from `tests/test_battle.py`.)

**Patterns:**
- One behavior per test method, with a comment above each assertion that explains the game-level reason for the expected value (`# Mud-Slap (189): ground move vs flying/normal -> 0 damage due to immunity`). Keep this habit: the expected values come from Gen 2 rules, not from the implementation.
- Each file ends with `if __name__ == "__main__": unittest.main()` in about half the files (21 of 42). Include it in new files so they can run directly.
- Test methods are grouped by the function under test, in the order they appear in the source module.

**Setup and teardown:**
- Per-test setup with `setUp` / `tearDown` when each test needs a fresh resource. Example: `tests/test_nav_memory.py` deletes any previous `logs/test/nav.db`, creates a `NavMemory`, and closes it in `tearDown`. Use this shape for anything that writes to disk or a database.
- Class-level setup with `@classmethod setUpClass` / `tearDownClass` only for expensive resources. It appears in 4 files. `tests/test_emulator.py` is the reference: it boots PyBoy once, ticks 600 frames, and closes in `tearDownClass`. Do not use it for cheap fixtures.
- Temporary files use `tempfile.TemporaryDirectory()` as a context manager or `tempfile.NamedTemporaryFile(suffix=...)`, in 22 test files. Example from `tests/test_app.py`:
```python
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "decisions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = load_decisions(path)
```

**Assertion patterns:**
- Equality on structured results: `self.assertEqual(loaded[1]["pos"], [4, 8])`.
- Boolean outcomes that are part of a result object: `self.assertTrue(tile)`, `self.assertTrue(result.ok)`.
- Exceptions: `with self.assertRaises(RomMismatchError): Emulator(rom=handle.name, verify_rom=True)` (`tests/test_emulator.py`).
- Relational checks for reward and ranking logic: `self.assertGreater(walk, bump)` (`tests/test_ppo.py`, `tests/test_battle.py`). Prefer relational assertions when the exact number is a tuning parameter.
- Module-level constants are referenced in assertions instead of magic numbers where possible (`TARGET_LEVEL`, `CHERRYGROVE` in `tests/test_gym_prep.py`).
- `subTest` is not used anywhere. If you need to loop over cases, either write separate methods or use `subTest` for the first time deliberately.

## Test Data and Builders

There are no pytest fixtures and no shared fixture modules. Each test file builds its own inputs with small helper functions at module top. Follow this pattern.

**Builder with overrides (preferred):**
```python
def gs(**kw):
    base = {
        "map_group": 21,
        "map_number": 14,
        "x": 3,
        "y": 7,
        "facing": "up",
        "tiles": {"up": COLL_FLOOR, "down": COLL_FLOOR, "left": COLL_FLOOR, "right": COLL_FLOOR},
        "battle_mode": 0,
        "script_running": 0,
        "textbox_flags": 0,
        "badge_bits": 0b00001111,
    }
    base.update(kw)
    return GameState.from_dict(base)
```
(`tests/test_segment.py`, same shape in `tests/test_macros.py` as `gs(**kw)` returning `GameState.from_dict(base)`.)

Other builders in the same style:
- `_row(**overrides)` in `tests/test_app.py` builds a decision-log dict with a full default set of keys, then `base.update(overrides)`.
- `make_state(mid=ROUTE_30, x=10, y=45, ...)` in `tests/test_gym_prep.py` returns a `SimpleNamespace` with a `has_event` lambda for story flags.

**Lightweight state objects:**
- `types.SimpleNamespace` is the usual stand-in for a game-state or party object when only a few attributes matter (12 test files). Use it when the code under test reads attributes but never calls methods.
- Use `GameState.from_dict(...)` when the code under test calls real `GameState` methods (`to_text`, `has_event`).

**Location:**
- Builders and fake classes are defined at the top of the test module that uses them. There is no shared `tests/helpers.py` or `tests/factories/`. If two files need the same builder, copy it, or move it into the test module that owns the behavior. Do not create a shared helpers module without updating this document.

**Real game fixtures:**
- Saved emulator snapshots live in `states/` (for example `states/route29_entry.state`, `states/new_bark_outside.state`, `states/smoke_wall.state`) and `recordings/`. They are loaded by name through `src/states.py` (`StateRegistry`) or by path (`STATES_DIR / "new_bark_outside.state"`).
- The real ROM is `rom/pokemon_crystal.gbc` with symbols in `rom/pokemon_crystal.sym`. Paths come from `src/config.py` (`ROM_PATH`, `SYM_PATH`, `STATES_DIR`).

## Mocking

**Framework:** `unittest.mock` where needed, but the repo prefers hand-written fakes.

**Hand-written fakes (dominant pattern):**
- Fakes are small classes whose methods record calls and return canned values. They are named `Fake<Thing>` or `_<Thing>`. There are about 23 such classes in `tests/`. Examples:
```python
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
```
(`tests/test_macros.py`.)
- A subclass adds capabilities for a second code path rather than changing the base. `ReadableMachine(FakeMachine)` adds `read()` so macros take their real-emulator branch (`tests/test_macros.py`).
- SDK clients are faked at the message level. `FakeMessages.create(**kwargs)` records the call and returns a `SimpleNamespace` shaped like an Anthropic response with `usage` and `content` (`tests/test_llm_supervisor.py`). Pass the fake client into `Supervisor(client=...)`; the code under test never touches the network.
- Randomness is seeded: `rng=np.random.default_rng(0)` (`tests/test_ppo.py`, `src/rl_env.py`).

**`unittest.mock.MagicMock`:**
- Used only in `tests/test_battle.py`, imported inside each test method (`from unittest.mock import MagicMock`), to verify which macro was called:
```python
macros = MagicMock()
state = SimpleNamespace(battle_mode=1, num_balls=5, party_count=1, ...)
tactical_battle_step(state, macros)
macros.choose_move.assert_called_once_with(1)
macros.use_pokeball.assert_not_called()
```
- Use `MagicMock` for "was this method called with X" checks on an object you do not want to implement. Prefer a fake class when the test also needs return values or state.
- `patch` is not used. Do not patch module globals; inject the collaborator instead.

**Dependency injection (preferred over patching):**
- Side-effecting code takes its collaborators as keyword arguments, and the test passes fakes. `tools/pod_checkpoint.py` `retrieve_checkpoint(..., run=..., loader=..., clock=...)` is tested by passing a recording `run` callable, a `loader` that appends to a list, and `clock=lambda: WHEN` (`tests/test_pod_checkpoint.py`). Follow this pattern for subprocess calls, file loads, time, and randomness.
- Duck-typed hardware: `Macros(machine, reader)` accepts anything with `press`, `tick`, `frame`, `read`. Tests pass `FakeMachine` / `ScriptedReader`.

**What to Mock:**
- The emulator (`machine`), RAM reader (`reader`), LLM client, subprocess runner, filesystem loaders, and clocks.
- Anything that talks to the network or the GPU (`anthropic`, `torch`, RunPod over `rsync`/`ssh`).

**What NOT to Mock:**
- Pure game logic: type charts (`src/battle.py`), collision tables (`src/collision.py`), reward functions (`src/rl_env.py` `progress_reward`), and the `GameState` parser. Build real objects with `from_dict` or plain values and assert on the result.
- SQLite. `tests/test_nav_memory.py`, `tests/test_chain.py`, and `tests/test_run_log.py` use real databases under `logs/test/`.
- The JSONL and filesystem code. Use `tempfile` directories instead of mocking `open`.
- The real emulator, when the test is about RAM parsing or boot behavior. Use the ROM-gated tests described below.

## Skips and Environment Gates

Many tests require the ROM, saved states, or live credentials. They are skipped, not failed, when those are absent. Keep this pattern.

**ROM present:**
```python
from src.config import ROM_PATH, SYM_PATH

ROM_OK = ROM_PATH.exists() and SYM_PATH.exists()

@unittest.skipUnless(ROM_OK, "ROM not present")
class EmulatorTests(unittest.TestCase):
    ...
```
(`tests/test_emulator.py`.) The same gate appears in 16 test files, with variants `HAS_ROM = ROM_PATH.exists() and SYM_PATH.exists()` (`tests/test_live_execution.py`) and per-state checks such as `HAS_ROM and NEW_BARK.exists()` with message `"new_bark_outside.state missing"` (`tests/test_live_execution.py`).

**Live API or live emulator:**
```python
@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("POKEMIND_LIVE") == "1", "live API off")
```
Live tests require both an API key and `POKEMIND_LIVE=1`. Do not run them in CI or by default.

**Rules for new skips:**
- Always give a reason string that names the missing resource.
- Gate on file existence from `src/config.py`, not on hard-coded absolute paths.
- Offline unit tests must pass with no ROM, no GPU, and no API key. The `.planning/ROADMAP.md` acceptance checks say so explicitly (for example `pytest tests/test_llm_supervisor.py ... -q` "with no API key").

## Test Types

**Unit Tests (majority):**
- Pure functions and small classes with fakes. Examples: `tests/test_battle.py`, `tests/test_ppo.py` (reward and policy math), `tests/test_app.py` (log parsing), `tests/test_pod_checkpoint.py` (injected subprocess), `tests/test_nav_memory.py` (SQLite).
- These run in under a second each. The two modules `tests/test_macros.py` and `tests/test_battle.py` ran 49 tests in 0.37 s during this analysis.

**Integration Tests (emulator and ROM):**
- Use the real PyBoy emulator and ROM, skipped without them. Examples: `tests/test_emulator.py` (save/load round-trip, press/listener events), `tests/test_live_execution.py` (`LiveBootTests.test_stack_constructs_on_real_pyboy`), `tests/test_gym_prep.py` (walks the real Route map with `_nav()._step`), `tests/test_segment.py`.
- Keep these to behavior that cannot be faked. Use `emu.tick(..., render=False)` to avoid rendering cost.
- Always close the emulator: `try: ... finally: emu.close()` (`tests/test_emulator.py`), or `tearDownClass` for class-scoped instances.

**Smoke scripts (not in `tests/`):**
- `crystal_smoke.py` at the repo root runs a headless boot and reports status. Run it manually; it is not collected by the test runner.

**E2E Tests:** Not used. Full-game runs are recorded as demo artifacts (`scripts/run_demo_*.sh`, `recordings/`) rather than asserted in tests.

**Coverage:** None enforced. No coverage tool is configured, no `.coveragerc`, and no coverage threshold appears in the repo.

## Common Patterns

**Async Testing:**
Not applicable. The codebase is synchronous. Do not introduce `asyncio` into tests; the emulator and RL loop are single-threaded.

**Error Testing:**
```python
def test_missing_file_returns_empty(self):
    with tempfile.TemporaryDirectory() as tmp:
        self.assertEqual(load_decisions(Path(tmp) / "missing.jsonl"), [])

def test_wrong_bytes_rejected_before_boot(self):
    with tempfile.NamedTemporaryFile(suffix=".gbc") as handle:
        handle.write(b"not a rom")
        handle.flush()
        with self.assertRaises(RomMismatchError):
            Emulator(rom=handle.name, verify_rom=True)
```
- Test the documented fallback (empty list, `ok=False` result, `None`) and the malformed-input path, not just the happy path.
- Use `assertRaises` as a context manager and assert on the type; check the message only when the message is part of the contract.

**CLI Testing:**
- Tools expose `main(argv=None) -> int`, so tests call `main([...])` directly and assert on the return code. Do not spawn subprocesses for CLI tests.

**Rule-based and state-machine Testing:**
- Scenario tests are separate methods with one assertion group per scenario (`tests/test_battle.py`, `tests/test_falkner_nav.py`, 14 methods). Name each method after the scenario (`test_tactical_battle_weaken_first`).

**Determinism:**
- Seed every RNG: `np.random.default_rng(0)` or `seed=` parameters. Tests must not depend on wall-clock time. Inject `clock=lambda: WHEN` with a fixed `datetime` (`tests/test_pod_checkpoint.py`).

## Gaps and Known Weak Spots

- No pytest or unittest runner is pinned in `requirements.txt`. Add `pytest` to a dev requirements file before depending on it in CI.
- No CI configuration is present (no `.github/workflows/`). Tests are run by hand.
- Several tests write to `logs/test/` without cleaning up (`tests/test_chain.py`, `tests/test_dispatch.py`). Use `tempfile` for new tests.
- `tests/__pycache__/` exists in the working tree. It is ignored by `.gitignore`; do not commit it.
- Emulator-dependent coverage depends on the presence of `rom/pokemon_crystal.gbc`, which is committed to this repo. Confirm the ROM SHA-1 in `src/config.py` (`ROM_SHA1`) matches before trusting skip-free results.

---

*Testing analysis: 2026-11-05*
