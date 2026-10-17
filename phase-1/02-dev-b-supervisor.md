# Phase 1 — Dev B: Supervisor layer (H3)

**Run this spec on your machine in parallel with** [`01-dev-a-execution.md`](01-dev-a-execution.md).

**Parent:** [`SPEC.md`](../SPEC.md) · **Do not edit:** Dev A’s mover/macros/triggers (see §4).

---

## 0. Before you start

```bash
git pull origin main
python -m venv .venv && source .venv/bin/activate   # if needed
pip install -r requirements.txt
pytest tests/ -q
```

Set `ANTHROPIC_API_KEY` in `.env` (gitignored) for one optional live test.

Pin in Cursor: `@phase-1/02-dev-b-supervisor.md` `@SPEC.md` `@src/ram_parser.py`.

---

## 1. Your mission

Build the **LLM supervisor**: tool schema, Anthropic tool-calling loop, token/cost logging, and **`always_on` baseline flag**. No greedy movement or Cut macro logic in your files — call **stubs** until Dev A lands implementations.

**Outcome:** Given `GameState.to_text()` + an optional trigger reason, the model returns **one valid tool call** (never raw buttons). Metrics append to a JSONL log for the demo chart.

---

## 2. Deliverables (you own these paths)

| File | Purpose |
| --- | --- |
| `src/tools_schema.py` | Pydantic models + Anthropic tool JSON for all tools |
| `src/llm_supervisor.py` | `Supervisor.decide(...)`, `run_tool(...)`, cost log |
| `src/supervisor_stubs.py` | No-op / fake `ActionResult` callables until A wires real code |
| `tests/test_tools_schema.py` | Schema validation, serialize/deserialize |
| `tests/test_llm_supervisor.py` | **Mocked** Anthropic client; one optional `@pytest.mark.live` |

Optional: `src/agent_loop.py` skeleton — read state → `triggers.evaluate` (import A when ready) → `supervisor.decide` → dispatch (stub).

**Do not create:** `src/mover.py`, `src/macros.py`, `src/triggers.py` (Dev A).

---

## 3. Interface contract (must match Dev A exactly)

Define these tools in `tools_schema.py` (names and types are **frozen** for Phase 1):

| Tool | JSON args | Description for model |
| --- | --- | --- |
| `go_to` | `x: int`, `y: int` | Walk toward tile on current map |
| `run_path` | `directions: list[str]` | Exact steps: up/down/left/right |
| `use_cut` | `{}` | Use HM Cut on tree blocking path |
| `mash_text` | `max_frames: int` (default 600) | Advance dialogue |
| `choose_move` | `slot: int` (1–4) | Pick battle move |
| `switch_pokemon` | `slot: int` (1–6) | Switch active Pokémon |

Every tool also takes a required `reason: str` (one sentence). Log it on `logs/decisions.jsonl` per SPEC §5.3. Do not send `reason` to the emulator.

**System prompt rules (encode in `llm_supervisor.py`):**

- You are playing Pokémon Crystal; state text is authoritative.
- Output **only** tool use — never suggest button names.
- On `CUT_BLOCKED` context, prefer `use_cut` if party has Cut.
- On `BATTLE`, prefer `choose_move` unless HP critical (then item/switch later).
- On `STUCK` / `DIALOGUE`, prefer `mash_text` or `run_path` with short path.

**Trigger context strings** (Dev A produces; you consume):

- Pass `trigger.kind` + `trigger.detail` into the user message when present.

**Dispatch:**

```python
def run_tool(self, name: str, arguments: dict) -> ActionResult:
    # Phase 1: delegate to registry dict[str, Callable]
    # Later: import tool_dispatch from Dev A
```

Use the same `ActionResult` dataclass Dev A documents (`ok`, `message`, `frames`).

---

## 4. Implementation notes

### 4.1 `tools_schema.py`

- One Pydantic model per tool args + union/`ToolCall` wrapper.
- Export `anthropic_tools() -> list[dict]` for the Messages API.
- Validate model output with Pydantic before dispatch.

### 4.2 `llm_supervisor.py`

- Model: default **`claude-haiku-4-5`** (or latest Haiku) for routine; allow env override for “hard” calls.
- Log each call: `{timestamp, model, input_tokens, output_tokens, tool_name, est_usd}` → `logs/supervisor.jsonl` (create `logs/` gitignored if needed — check `.gitignore`).
- **`always_on: bool`:** if True, call LLM every overworld step (baseline harness); if False, only when `trigger is not None` (normal mode per SPEC §1.2).

### 4.3 `supervisor_stubs.py`

```python
def stub_go_to(x, y): return ActionResult(ok=True, message=f"stub go_to {x},{y}")
# … etc.
```

Register stubs in `Supervisor(..., registry=STUB_REGISTRY)` until integration.

### 4.4 Tests

- Mock `anthropic.Anthropic.messages.create` to return a fixed `tool_use` block.
- Assert `run_tool("use_cut", {})` invoked with parsed args.
- Live test (optional, skip in CI): one real call with minimal prompt — mark `@pytest.mark.live`.

---

## 5. Verification (you run these)

```bash
pytest tests/test_tools_schema.py tests/test_llm_supervisor.py -v
```

**Acceptance:**

1. All non-live tests pass with no API key.
2. Schema exports ≥6 tools matching §3 table.
3. Mocked run: state text + `"CUT_BLOCKED: tree to the north"` → tool name `use_cut`.
4. Log file append works after a mocked decision.

---

## 6. Sync with Dev A (~60 min after both start)

1. Dev A confirms Python function signatures match your Pydantic args.
2. Replace `supervisor_stubs` registry with imports from `mover` / `macros` or `tool_dispatch`.
3. Integration test (either dev): load `ilex_src` → trigger → supervisor → real `use_cut()`.

---

## 7. Cursor kickoff

```text
Implement phase-1/02-dev-b-supervisor.md only.
Add src/tools_schema.py, src/llm_supervisor.py, src/supervisor_stubs.py and tests with mocked Anthropic.
Match tool names in 01-dev-a-execution.md §3. Do not implement mover, macros, or triggers.
Run pytest for new tests before done.
```

---

## 8. Git discipline

- Work on branch `phase1/dev-b` (recommended).
- **Pull before push.** Only touch §2 files (+ `logs/` gitignore entry if needed).
- Commit message prefix: `phase1(b): …`
