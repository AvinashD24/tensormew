# Phase 1 — Dev A: Execution layer (H1 + H2)

**Run this spec on your machine in parallel with** [`02-dev-b-supervisor.md`](02-dev-b-supervisor.md).

**Parent:** [`SPEC.md`](../SPEC.md) · **Do not edit:** files owned by Dev B (see §4).

---

## 0. Before you start

```bash
git pull origin main
python -m venv .venv && source .venv/bin/activate   # if needed
pip install -r requirements.txt
pytest tests/ -q
python crystal_smoke.py --headless
```

Pin in Cursor: `@phase-1/01-dev-a-execution.md` `@SPEC.md` `@src/step.py` `@src/ram_parser.py` `@src/collision.py`.

---

## 1. Your mission

Build **deterministic overworld + battle execution**: movement, macros, and **when** to ask the LLM (triggers). No Anthropic API code in your files.

**Outcome:** From a loaded checkpoint, the stack can walk toward targets, detect Cut-tree blocks, run `use_cut()`, and expose hooks the supervisor will call later.

---

## 2. Deliverables (you own these paths)

| File | Purpose |
| --- | --- |
| `src/mover.py` | `go_to(x, y)`, `run_path(directions)`, stuck tracking |
| `src/macros.py` | `use_cut()`, `mash_text(frames=...)`, battle stubs: `choose_move(slot)`, `switch(slot)` |
| `src/triggers.py` | `evaluate(state) -> TriggerEvent \| None` (Cut, stuck, battle, textbox) |
| `tests/test_mover.py` | Mocked tiles / fake states |
| `tests/test_macros.py` | Mock `machine` or integration with `Emulator` when `.state` exists |
| `tests/test_triggers.py` | Synthetic `GameState` / dict fixtures |

Optional integration (after B’s schema lands): `src/tool_dispatch.py` — maps tool names → your functions (coordinate with Dev B).

**Do not create:** `src/llm_supervisor.py`, `src/tools_schema.py` (Dev B).

---

## 3. Interface contract (must match Dev B exactly)

Supervisor tools Dev B defines — **your code implements the behavior**:

| Tool name | Your implementation | Notes |
| --- | --- | --- |
| `go_to` | `mover.go_to(x: int, y: int) -> MoveResult` | Greedy using `GameState.tiles` (`up/down/left/right`) |
| `run_path` | `mover.run_path(directions: list[str]) -> PathResult` | Each dir ∈ `up, down, left, right` |
| `use_cut` | `macros.use_cut() -> MacroResult` | Face blocking tree → A → confirm (Gen 2) |
| `mash_text` | `macros.mash_text(max_frames: int = 600)` | While `TEXT_GATE` or textbox flags |
| `choose_move` | `macros.choose_move(slot: int)` | Battle; slot 1–4 |
| `switch_pokemon` | `macros.switch_pokemon(slot: int)` | Party slot 1–6 |

Shared result shape (add `src/action_results.py` **only if Dev B agrees**, or use dataclasses in `mover.py` / `macros.py` and document fields):

```python
# Minimal — supervisor reads .ok and .message
@dataclass
class ActionResult:
    ok: bool
    message: str
    frames: int = 0
```

**Trigger enum** (export from `triggers.py` for B’s prompt):

- `CUT_BLOCKED` — adjacent tile `collision.is_cut_tree(...)`
- `STUCK` — same `(map, x, y)` for N steps while not in dialogue
- `BATTLE` — `battle_mode != 0`
- `DIALOGUE` — `script_running` or `textbox_flags` and mover idle

---

## 4. Implementation notes

### 4.1 `mover.py`

- Use existing `TileStepper` (`src/step.py`) + `RamParser` on `Emulator`.
- **Greedy `go_to`:** at each step, try directions that reduce \|Δx\|+\|Δy\|; prefer walkable adjacency from `wTile*` (treat `COLL_WALL` / water as blocked; grass OK).
- **Stuck:** increment counter when position unchanged after `step`; reset on map change or successful move.
- Do not press Start/Select.

### 4.2 `macros.py`

- `use_cut()`: require overworld (`battle_mode == 0`); use facing + adjacent `is_cut_tree` to pick direction; `press("a")` with waits; verify tile change or walkability after.
- Reuse `Emulator.press` / `tick` patterns from `src/episode.py`.

### 4.3 `triggers.py`

- Pure logic where possible: input `GameState`, output optional trigger + short **context string** for LLM (Dev B embeds in prompt).
- Respect gates: don’t fire `STUCK` during `script_running` / textbox (see `TEXT_GATE` in `ram_parser.py`).

---

## 5. Verification (you run these)

```bash
pytest tests/test_mover.py tests/test_macros.py tests/test_triggers.py -v
```

**With a real save** (when available, e.g. `ilex_src` in manifest):

```bash
# Example — adjust to your CLI once wired
python tools/run_episode.py --state ilex_src --script …
```

**Acceptance:**

1. Unit tests pass without network.
2. On Ilex (or mock): trigger fires when Cut tree ahead; `use_cut()` completes without exception.
3. `go_to` moves at least one tile on `route_test` or manual checkpoint.

---

## 6. Sync with Dev B (~60 min after both start)

1. Compare tool names and argument names with `src/tools_schema.py` (B’s branch).
2. Agree on `ActionResult` location.
3. Optional: one PR merges A then B, or A adds `tool_dispatch.py` that B’s supervisor imports.

**Integration owner:** whoever finishes second wires `llm_supervisor.run_tool(name, args)` → your functions.

---

## 7. Cursor kickoff

```text
Implement phase-1/01-dev-a-execution.md only.
Add src/mover.py, src/macros.py, src/triggers.py and tests. Use TileStepper, RamParser, collision.is_cut_tree.
Match the tool interface table for Dev B. Do not implement llm_supervisor or tools_schema.
Run pytest for new tests before done.
```

---

## 8. Git discipline

- Work on branch `phase1/dev-a` (recommended) or coordinate commits to `main`.
- **Pull before push.** Only touch §2 files (+ shared `action_results.py` if both agree).
- Commit message prefix: `phase1(a): …`
