# Observability, RL reasoning levels, and the benchmark

Build plan for the decision trace, the Worker B reasoning layer, and a checkpoint benchmark. This is the hackathon plan for those three pieces. It sits on top of `SPEC.md` §5.3 and the Worker B heads in `pokemon_agent.md`.

The reasoning layer is part of the policy. It reads the heads Worker B already has (action logits, value, action-effect) and writes a structured note. It does not train a second model to narrate, and it never emits raw buttons. A one-sentence `reason` is rendered from that note by a pure function.

Everything below can be built and tested with a fake policy. The real network plugs in behind one protocol when it exists. Modes that are not built yet show up as `not_built` in the report. They do not fail the run.

---

## 1. What this adds

| Piece | Job | Depends on today |
| --- | --- | --- |
| Decision trace | One append-only JSONL row per decision, for the thought log and the cost chart | `Supervisor._log` writes `logs/supervisor.jsonl` only. `EpisodeRunner` writes `runs/<id>.jsonl` with no actor or cost. |
| Reasoning layer | Pick one of `up/down/left/right/a` at a chosen depth, and explain the pick from the heads | No `worker_b` module yet. `triggers.evaluate` and `llm_supervisor.decide` exist. |
| Benchmark | Same checkpoint, same seeds, several modes. Score progress, stalls, tokens, and dollars from the log alone | `states/manifest.json`, `EpisodeRunner`, `StateRegistry`. |

Three logs stay separate. Do not merge them.

| File | Contents | Who reads it |
| --- | --- | --- |
| `logs/supervisor.jsonl` | Cost rows already written by `Supervisor._log`, plus why that call was made | Dollar counter |
| `logs/decisions.jsonl` | Thought log: why the LLM was called, and why this step happened | Streamlit, benchmark |
| `runs/<episode_id>.jsonl` | Frame trace: button or wait, plus `GameState.to_dict()` | Replay, debugging |

`logs/` and `runs/` are gitignored. A benchmark **summary** (`benchmarks/<run_id>.json`, no screenshots, no API payloads) may be committed. Raw JSONL stays local.

---

## 2. Invariants

1. `actor=rl` actions are only `up`, `down`, `left`, `right`, `a`. No Start, Select, or B. B stays inside text macros.
2. `actor=llm` rows are tool calls from `src/tools_schema.py`. `call_cause` and `call_detail` say why the supervisor ran. `reason` says why that tool was picked. Neither string is an emulator input. An `actor=llm` row with an empty `call_cause` or empty `call_detail` is invalid and must not be written.
3. `actor=script` rows are `go_to`, `run_path`, and macros with a fixed reason, for example `segment script for ilex_cut`.
4. `reason` is one sentence, 240 characters max, produced by `render_reason`. No free-form model text on the RL path.
5. Token counts and `est_usd` are zero on `script` and `rl` rows. An escalation to the supervisor is a **second** row with `actor=llm`.
6. A failed log write must not change the action. Agent mode records `log_error` and continues. Benchmark mode treats a log write failure as a failed run.
7. Rows never contain `.env` values, API keys, or the system prompt body.
8. Two runs are comparable only when `rom_sha1`, checkpoint name, `state_sha1`, `max_steps`, RTC lock, and seed set match. The reporter refuses otherwise.

---

## 3. Decision record

Schema version `1`. One JSON object per line. Three `type` values: `run`, `decision`, `end`.

`src/trace.py` owns the Pydantic models, the writer, and `render_reason`. `src/llm_supervisor.py` keeps writing the cost log. The trace writer is an extra call, not a replacement.

### 3.1 `type=run`

Written once at start.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | `1` | Reject unknown versions |
| `run_id` | string | Ties every row in this episode |
| `episode_id` | string | Same id `EpisodeRunner` already uses |
| `mode` | enum | See §6 |
| `reasoning_level` | `0\|1\|2\|3` | Knob for this run. Ignored when mode has no RL |
| `checkpoint` | string | Manifest name |
| `state_sha1`, `rom_sha1` | string | From `StateRegistry` / `ROM_SHA1` |
| `seed` | int | RNG and PyBoy `seed_rng` |
| `max_steps` | int | Budget |
| `rtc_locked` | bool | Must be true |
| `model` | string | Supervisor model, or `""` |
| `started_at` | ISO UTC | |

### 3.2 `type=decision`

| Field | Type | Meaning |
| --- | --- | --- |
| `run_id`, `schema_version` | | |
| `timestamp` | ISO UTC | |
| `frame`, `step` | int | Emulator frame and decision index |
| `map` | `[group, number]` | `GameState.map_id` |
| `pos` | `[x, y]` | `wXCoord`, `wYCoord` |
| `facing` | string | |
| `badges` | int | Johto bit count already on `GameState` |
| `actor` | `llm` \| `rl` \| `script` | |
| `call_cause` | string | Why the LLM was invoked. Required and non-empty when `actor=llm`. `""` otherwise. See §3.2.1 |
| `call_detail` | string | Facts that justified the call, 240 characters max. Required and non-empty when `actor=llm`. `""` otherwise |
| `trigger` | string | `CUT_BLOCKED`, `STUCK`, `BATTLE`, `DIALOGUE`, or `""`. Set when `call_cause=trigger`; also copied onto the matching `rl` escalation row |
| `tool` | string | LLM/script tool name, or `""` for a raw RL step |
| `action` | string | RL button, or `""` |
| `arguments` | object | Tool args after Pydantic validation. `{}` for RL |
| `reasoning_level` | `0\|1\|2\|3` | Level that produced this row |
| `reason` | string | One sentence from `render_reason` |
| `reason_struct` | object | Machine-readable note. Required for `rl`, optional otherwise |
| `ok` | bool | `ActionResult.ok` after the action |
| `frames` | int | Frames the action consumed |
| `input_tokens`, `output_tokens`, `est_usd` | number | Copied from `Decision` for `llm`, else `0` |
| `decision_ms` | number | Wall time inside the chooser, excluding emulator ticks |
| `log_error` | string | `""` unless the append failed |

`reason_struct` for `actor=rl`:

| Field | Meaning |
| --- | --- |
| `logits` | Length 5, order `up, down, left, right, a` |
| `probs` | Softmax of logits. Same order |
| `value` | Value-head scalar |
| `effect` | Length 5. Action-effect head: P(action changes the screen) |
| `legal` | Subset of the five actions still allowed after the level's mask |
| `rejected` | List of `{action, cause}`. Causes: `predicted_bump`, `start_select`, `low_margin` |
| `chosen` | The action executed |
| `margin` | Probability gap between the best and second-best legal action |
| `lookahead` | `null`, or `{depth, frames, delta_value, rolled_back}` |
| `escalated` | bool. True when this row is a handoff, not a button |

Reject NaN, Inf, and a logit vector whose length is not 5.

### 3.2.1 Why the LLM was called

`reason` is the model's tool choice. `call_cause` is the caller, written before `messages.create` returns. The model does not get to fill `call_cause`.

| `call_cause` | When | `call_detail` must include |
| --- | --- | --- |
| `trigger` | `triggers.evaluate` returned a `TriggerEvent`, and that event is why `decide` ran | The trigger kind and `TriggerEvent.detail` (battle mode and party HP, cut-tree side, stuck count and tile, or script/textbox flags) |
| `always_on` | `Supervisor.always_on` is true and no trigger fired | The fixed text `always_on baseline; no trigger` |
| `rl_escalate` | Reasoning level 3 returned `escalated=True` and the caller then called `decide` | The policy note: margin and `stuck_count` (the same sentence as the `rl` row's `reason`) |

If a trigger fired, `call_cause` is `trigger` even when `always_on` is true. The run header already records the mode. `always_on` is only for steps where `evaluate` returned `None`.

`call_cause=rl_escalate` wins over a trigger only when the caller entered `decide` from the R3 branch. The trigger kind still goes in `trigger` when one was also present, so the log shows both the handoff and the game event.

`Supervisor.decide` returns `None` without a row when `always_on` is false and the trigger is missing. A call that did not happen is not logged.

The same `call_cause` and `call_detail` are copied onto the existing `logs/supervisor.jsonl` cost row, next to `model`, token counts, and `est_usd`.

### 3.3 `type=end`

`outcome` is `milestone`, `step_budget`, `timeout`, `stall_cap`, `error`, or `not_built`.

Also store `steps`, `frames`, `wall_s`, `badges`, `map`, `pos`, `llm_calls`, `llm_calls_by_cause` (`trigger`, `always_on`, `rl_escalate`), `input_tokens`, `output_tokens`, `est_usd`, `rl_steps`, `lookahead_frames`, `escalations`, `stalls`, `bumps`.

### 3.4 `render_reason`

Pure function. Examples (the words can change; the inputs cannot):

- R0: `policy chose up (p=0.62) value=0.41`
- R1: `rejected left as a predicted bump; chose up (effect=0.88, p=0.54)`
- R2: `lookahead 3 steps raised value by 0.07; chose right`
- R3: `margin 0.02 and stuck 8; escalate to supervisor`
- LLM `reason`: the model's own `reason` argument, truncated to 240 characters, after the tool schema requires it
- LLM `call_detail`: caller text from §3.2.1, also truncated to 240 characters. This is not `render_reason` output
- Script: the caller's fixed string

LLM tools gain a required `reason: str` on every args model in `src/tools_schema.py`. `run_tool` strips `reason` before the handler so `go_to(x, y)` stays the mover signature. The stripped string is what gets logged.

---

## 4. Who writes a row

```text
GameState
    │
    ├─ battle_mode != 0 ──────────────► battle macro or Worker C
    │                                    actor=script or actor=rl (combat later)
    ├─ path known / segment script ───► mover.go_to / run_path
    │                                    actor=script
    ├─ unmapped overworld ────────────► ReasoningLayer.choose
    │                                    actor=rl
    │                                    └─ escalated ─► Supervisor.decide
    │                                                     actor=llm
    └─ trigger (Cut, stuck, text) ────► Supervisor.decide
                                         actor=llm
```

Hook points, in order:

1. `Supervisor.decide` takes a `call_cause` and `call_detail` from the caller and writes them on both the cost row and the `decision` row before it looks at the tool the model returned. `agent_loop.step` builds those strings from `TriggerEvent` (kind plus `detail`). Today `step` only has `state_text`. The R3 caller passes `call_cause=rl_escalate` and the policy sentence.
2. `Mover.go_to`, `Mover.run_path`, and `Macros.*` append `actor=script` when the caller passes a `Trace`. A missing trace means no row (unit tests stay quiet).
3. `ReasoningLayer.choose` appends `actor=rl` before the button is pressed. `ok` is filled after `TileStepper.step` returns.
4. `EpisodeRunner` is unchanged. The benchmark runner writes `type=run` and `type=end` around whatever actor it invoked.

`agent_loop.step` grows an optional `trace: Trace | None`. When it is set, every LLM row stores `call_cause`, `call_detail`, and the trigger kind from `TriggerEvent`.

---

## 5. Reasoning layer

Module: `src/reasoning.py`.

The layer does not own the network and does not import PyBoy. Callers pass heads in. That is what lets tests run with no weights and lets Worker B arrive later without a rewrite.

### 5.1 Inputs

```text
PolicyHeads
  logits: tuple[float, float, float, float, float]   # up down left right a
  value: float
  effect: tuple[float, float, float, float, float]   # P(screen changes)
  hidden_norm: float                                  # L2 of GRU state, diagnostic only

ChooseRequest
  heads: PolicyHeads
  level: 0 | 1 | 2 | 3
  stuck_count: int
  stall_threshold: int = 8
  bump_effect_below: float = 0.15
  margin_floor: float = 0.08
  lookahead_depth: int = 3
  lookahead: Lookahead | None                         # None until a clone exists
```

`Lookahead` is a callable the benchmark or Worker B supplies:

```text
(action, depth) -> LookaheadResult(delta_value, frames, rolled_back)
```

R2 calls it at most once per decision, for the best legal action, and only when `margin < margin_floor`. If `lookahead` is `None`, R2 behaves as R1 and sets `reason_struct.lookahead = null`. It does not invent a value.

Action order is fixed: `up`, `down`, `left`, `right`, `a`. Document it next to the constant `ACTIONS`.

### 5.2 Levels

| Level | Name | Extra compute | Choice rule |
| --- | --- | --- | --- |
| 0 | Reflex | None | `argmax` of logits. Ties break in action order. |
| 1 | Affordance | Five compares | Drop actions with `effect < bump_effect_below`. Score legal actions as `logit + log(effect + 1e-6)`. If every action is dropped, fall back to R0 and record `rejected` anyway. |
| 2 | Lookahead | One cloned rollout, only on a close call | Start from the R1 ranking. If `margin < margin_floor` and a `Lookahead` is provided, roll the top action `lookahead_depth` macro-steps inside a saved PyBoy state, then `load_state` the bytes back. Keep the action when `delta_value > 0`, otherwise take the second legal action. |
| 3 | Escalate | One supervisor call when the policy is unsure | If `margin < margin_floor` or `stuck_count >= stall_threshold`, do not press a button. Return `escalated=True`, `action=""`. The caller invokes `Supervisor.decide` with `call_cause=rl_escalate` and writes a separate `actor=llm` row. Otherwise identical to R2. |

R3 is the only level that spends tokens, and only on a low-margin or stuck step. Routine walking stays at whatever level the run was configured for, with `est_usd=0`.

Worker B's action-effect head is the bump detector from `pokemon_agent.md`: P(the center of the frame changes by more than 2% L1). Until that head exists, the fake policy fills `effect` from the collision bytes (`0.0` if the facing tile is solid, else `1.0`). The layer does not care which one it got.

### 5.3 What the real policy must provide

When `src/tier3_workers/worker_b_explorer.py` exists, one method is enough:

```text
def heads(self, obs) -> PolicyHeads
```

`obs` is the Worker B observation already specified: stacked frames, visited mask, local tiles, TEDUO vector, GRU state. The reasoning layer never sees pixels. Training stays in the PPO loop (reward scale `/ 8.0`, no `[-1, 1]` clip, no Start/Select). The reasoning layer is inference-time only. Do not backprop through R2 rollouts or through the rendered sentence.

### 5.4 Fake policy

`FakePolicy` lives in `src/reasoning.py` so tests and the benchmark share it.

- Inputs: current `(x, y)`, target `(tx, ty)`, and the four adjacent collision ids.
- Logits: `+1` for a direction that reduces Manhattan distance, `-1` otherwise, `0` for `a`.
- Effect: `0` when that direction's tile is non-walkable (`collision.walkable` is false), else `1`.
- Value: negative Manhattan distance.

Tests in `tests/test_reasoning.py`:

1. R0 walks toward the target.
2. R1 refuses the greedy direction when that tile is blocked and picks the next-best open direction.
3. R2 flips to the second action when the supplied lookahead returns a negative `delta_value`.
4. R2 with `lookahead=None` matches R1 and records `lookahead: null`.
5. R3 with `stuck_count >= 8` returns `escalated=True` and an empty action.
6. Logits of the wrong length, or a NaN value, raise before a row is written.
7. `render_reason` on each level stays within 240 characters and contains the chosen action (or the word `escalate`).

No emulator in this file's tests.

---

## 6. Benchmark

Module: `src/benchmark.py`. CLI: `tools/benchmark.py`.

### 6.1 Modes

| `mode` | Actor | `reasoning_level` | What runs |
| --- | --- | --- | --- |
| `script` | script | 0 | `run_path` or `go_to` only. No model. |
| `llm_trigger` | llm | 0 | `Supervisor(always_on=False)`. Call only when `triggers.evaluate` returns an event. |
| `llm_always_on` | llm | 0 | `Supervisor(always_on=True)`. One call per overworld decision. This is the baseline in SPEC §1.2. |
| `rl_r0` … `rl_r3` | rl | 0 … 3 | `ReasoningLayer` at that level. R3 may add `llm` rows. |
| `hybrid` | script, then rl, then llm | configured, default 1 | Known path uses the mover. Unmapped steps use the reasoning layer. Cut, stuck, dialogue, and battle use the supervisor. |

`rl_*` and `hybrid` require a policy object. The CLI flag `--policy fake` uses `FakePolicy`. `--policy torch --weights PATH` imports Worker B. If the import or the file is missing, that mode's `outcome` is `not_built` and the process exit code stays 0 so a partial report still ships.

### 6.2 One cell

A cell is `(checkpoint, mode, seed)`.

1. Load the manifest entry. Refuse a checkpoint whose RAM is not ready (`map=(0,0)`, title screen).
2. `load_state`, `seed_rng(seed, seed)`, RTC already locked by `Emulator`.
3. Write `type=run`.
4. Loop until `max_steps`, timeout, stall cap, or the milestone predicate is true.
5. Write `type=end`.

Stall cap: 2,500 decisions with no milestone, matching the local stall number in `pokemon_agent.md`. On cap, `outcome=stall_cap`. Do not roll the checkpoint back inside this harness; record the failure.

Default budget when the caller does not pass one: `max_steps=300`, `timeout_s=120`. Live LLM cells should stay on a short checkpoint (`route29_entry` or `ilex_src`) so a matrix does not burn the key.

### 6.3 Milestones

Predicates over `GameState`, named in the CLI:

| Name | True when |
| --- | --- |
| `map_changed` | `map_id` differs from the start row |
| `badge_gained` | `badges` increased |
| `pos_reached` | `(x, y)` equals the target passed on the CLI |
| `cut_cleared` | No adjacent tile is a cut tree, and the start state had one |
| `step_budget` | No game predicate. The report compares distance traveled and cost at the budget. |

The report's headline number for a cell is decisions to the first true predicate, or the budget if it never fires.

### 6.4 Scores

Computed only by re-reading `logs/decisions.jsonl`. Re-aggregation must not boot PyBoy.

Per cell:

- `progress_badges`, `maps_visited`, `manhattan_gained`
- `decisions`, `frames`, `wall_s`
- `llm_calls`, `input_tokens`, `output_tokens`, `est_usd`
- `llm_calls_by_cause`: counts of `trigger`, `always_on`, and `rl_escalate`
- `rl_steps`, `lookahead_frames`, `escalations`
- `stalls` (rows whose `reason_struct` came from `stuck_count >= stall_threshold`, plus end-row `stalls`)
- `bumps` (count of `rejected.cause == predicted_bump` that were still executed, plus steps whose `pos` did not change after a move action)
- `ok_rate`
- `usd_per_milestone` (blank when the milestone did not fire)
- `calls_per_milestone`

Per mode, across seeds: mean, min, max. Print the per-seed table. Do not report a p-value.

The comparison the demo needs is `llm_trigger` against `llm_always_on` on the same checkpoint and the same seeds: calls, tokens, dollars, and whether the milestone fired. RL rows are extra columns. A missing RL mode is a blank column labeled `not_built`, not a zero.

### 6.5 Reporter rules

`compare(paths) -> Report` raises `IncompatibleRuns` when any of these differ: `schema_version`, `rom_sha1`, `checkpoint`, `state_sha1`, `max_steps`, `rtc_locked`.

Seed sets may be a subset. The report marks unpaired seeds and averages only the intersection.

### 6.6 CLI

```text
python tools/benchmark.py \
  --checkpoint route29_entry \
  --modes script,llm_trigger,llm_always_on,rl_r0,rl_r1 \
  --seeds 0,1,2 \
  --policy fake \
  --milestone step_budget \
  --max-steps 50 \
  --out benchmarks/route29.json
```

`--live` is required before any mode that constructs a real Anthropic client. Without it, `llm_*` modes use a fake client that returns `go_to` toward a fixed tile and `reason="fake"`. Tests never set `--live`.

Exit codes: `0` report written (including `not_built` modes), `2` incompatible runs or a bad checkpoint, `1` an unexpected exception.

### 6.7 Tests

`tests/test_benchmark.py` uses the fake policy and the fake client. No ROM required for the aggregator tests.

1. Two cells with different `rom_sha1` raise `IncompatibleRuns`.
2. A `not_built` mode produces an end row and a blank column, exit behavior covered by calling `compare` directly.
3. `llm_trigger` with no trigger writes zero `llm` rows. `llm_always_on` writes one per step with `call_cause=always_on` and `call_detail` `always_on baseline; no trigger`. A fired `CUT_BLOCKED` row uses `call_cause=trigger` and includes the cut-tree side from `TriggerEvent.detail`. Dollar totals follow `estimate_usd`.
4. `rl_r1` on a fixture with a blocked greedy tile records a `predicted_bump` rejection and a different `chosen` than `rl_r0`.
5. `rl_r3` with a forced `stuck_count` writes one `rl` row with `escalated=true` and one `llm` row. The `rl` row has `est_usd=0`. The `llm` row has `call_cause=rl_escalate` and a `call_detail` that includes the stuck count. An `llm` row missing `call_cause` fails validation.
6. Re-reading the JSONL reproduces the same totals.

An optional `@pytest.mark.live` test, skipped by default, runs one `llm_trigger` cell on `route29_entry` for `--max-steps 5`.

---

## 7. Build order

Each slice is one commit and leaves `pytest tests/ -q` green. Later slices are skippable. The report still runs.

| Slice | Files | Done when |
| --- | --- | --- |
| 1. Schema | `src/trace.py`, `tests/test_trace.py` | Models round-trip. `render_reason` covers all five actor/level cases. An `actor=llm` row without `call_cause` and `call_detail` is rejected. Append creates `logs/` and refuses a secret-looking key in `arguments`. |
| 2. Wire LLM and script | `src/llm_supervisor.py`, `src/agent_loop.py`, `src/tools_schema.py`, mover/macros call sites, their tests | A mocked supervisor decision appends one `decisions.jsonl` row and the existing `supervisor.jsonl` row, both with `call_cause` and `call_detail`. `reason` is required on tools and is not passed to handlers. A `run_path` with a `Trace` writes `actor=script` and leaves `call_cause` empty. |
| 3. Reasoning layer | `src/reasoning.py`, `tests/test_reasoning.py` | §5.4 tests pass with `FakePolicy`. No PyBoy import in this module. |
| 4. Benchmark | `src/benchmark.py`, `tools/benchmark.py`, `tests/test_benchmark.py` | §6.7 tests pass offline. CLI `--policy fake` writes `benchmarks/<run_id>.json`. |
| 5. Worker B adapter | only after the network module exists | `heads(obs) -> PolicyHeads`. One test with a torch module stand-in (random logits, frozen) shows R0 and R1 call `heads` once per decision. |

Slice 5 does not block slices 1–4. Do not start PPO training in this plan. The reasoning layer only consumes heads.

---

## 8. Demo mapping

| Surface | Source |
| --- | --- |
| Thought log | `decisions.jsonl` rows: `call_cause`, `call_detail`, then `tool` or `action` and `reason` |
| Dollar counter | Sum of `est_usd` on `actor=llm` |
| Cost chart | `llm_trigger` vs `llm_always_on` from the summary JSON: calls and dollars until the milestone |
| RL column | `rl_r0`–`rl_r3` mean decisions and bumps. Blank until a policy is passed |

The chart claims calls, tokens, and dollars on one named checkpoint. It does not claim a full-game cost.

---

## 9. Out of this plan

- Training Worker B, the frontier buffer, and the shared-memory env.
- Ghost scouts, SayCan's 100-boss rollouts, Metamon, and Eureka rewards. R3 escalation is a single supervisor call, not that stack.
- A learned captioner, a reasoning-token budget, or an extra network head whose only job is English.
- Streamlit itself. It should read `logs/decisions.jsonl` once slice 2 exists. The UI is a separate change.
