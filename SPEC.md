# tensormew (PokéMind) — System Specification

**Single source of truth.** Consolidates hackathon execution (`CONTEXT.md` retired) and long-term CRYSTAL-HYBRID research direction (`pokemon_agent.pdf`).

**Game:** Pokémon Crystal (GBC) on PyBoy · **ROM:** Crystal US **1.1** · **Implement with:** Cursor Agent

---

## 1. Mission & success metrics

### 1.1 Primary goal (DubHacks 2026)

**Get as far into the game as possible** before time runs out — measured by **Johto badge count**, **furthest registered checkpoint** in `states/manifest.json`, and **live progression** from the current segment.

This is **not** a full autonomous speedrun in 24 hours. Progress comes from:

1. **Checkpoint chain** — PyBoy `.state` snapshots at badge boundaries and hard puzzles (community `.sav` → continue → `tools/make_state.py` / `StateRegistry.capture`).
2. **Deterministic low level** — greedy mover, tile macros, battle macros (**0 LLM tokens** on routine steps).
3. **LLM on triggers only** — supervisor emits **tools/goals**, never raw button presses.

### 1.2 Secondary goal (thesis / demo)

Compare **LLM-on-triggers** vs **LLM-every-step baseline** on the **same checkpoints**: calls, tokens, and estimated $ per milestone. Pitch angle: *"How many steps actually needed an LLM?"* — avoid unverifiable total-game $ claims.

### 1.3 Demo surfaces (in priority order)

| Priority | Demo | Why |
| --- | --- | --- |
| P0 | **Furthest badge / map reached** | Matches primary goal |
| P1 | **Ilex Forest Cut** | Blocked mover → Cut trigger → `use_cut()` → path clears |
| P2 | **Whitney or next gym** | Battle + LLM recovery (if checkpoint exists) |
| P3 | Streamlit: frame, thought log, **$ counter**, cost chart | Judges |

### 1.4 Stretch vs unrealistic

| Target | Hackathon stance |
| --- | --- |
| New Bark → 3+ badges via agent | **Stretch** — needs many checkpoints + segment scripts |
| 8 badges + Lance | **Unlikely** unless heavily pre-seeded and scripted |
| Full `pokemon_agent.pdf` stack (PPO, scouts, Metamon CI) | **Post-hackathon** — see [Appendix A](#appendix-a-crystal-hybrid-research-roadmap-deferred) |

---

## 2. Hackathon architecture (what we build now)

Event-routed **two-layer** agent. Maps to `src/` modules — **do not** build Tier-3 RL workers during the hackathon unless H0–H4 are done.

```text
                    ┌─────────────────────────────────────┐
                    │  llm_supervisor.py (LLM, triggers)   │
                    │  tools: go_to, use_cut, battle, …   │
                    └─────────────────┬───────────────────┘
                                      │ goals / tool calls only
                    ┌─────────────────▼───────────────────┐
                    │  triggers.py                         │
                    │  Cut tree, stuck, battle, textbox    │
                    └─────────────────┬───────────────────┘
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
   mover.py + step.py          macros.py                  ram_parser.py
   greedy go_to, run_path       cut, item, mash, moves     GameState → text
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
                              emulator.py + collision.py
                                      │
                              PyBoy + rom/pokemon_crystal.*
```

**Interrupt rule (from research spec, hackathon simplification):** when `wBattleMode != 0`, run **battle macros** (and optional LLM on boss triggers). No LLM on routine wild fights if a macro suffices.

### 2.1 Module map

| Module | Status | Role |
| --- | --- | --- |
| `src/emulator.py` | Done | ROM + sym, tick, save/load, input |
| `src/ram_parser.py` | Done | Banked RAM → `GameState`, `to_text()` for LLM |
| `src/step.py` | Done | `TileStepper` — one-tile movement |
| `src/collision.py` | Done | pret collision ids, `is_cut_tree()` |
| `src/states.py` | Done | Manifest, capture, validate checkpoints |
| `src/episode.py`, `recording.py` | Done | Scripts, replay, baseline harness |
| `src/config.py`, `options.py` | Done | ROM SHA1, training options bits |
| `tools/make_state.py`, `import_sav.py`, … | Done | Checkpoint workflow |
| `src/mover.py` | **TODO** | `go_to(x,y)`, `run_path`, stuck detection |
| `src/macros.py` | **TODO** | `use_cut()`, items, battle, `mash_text()` |
| `src/triggers.py` | **TODO** | When to call supervisor |
| `src/llm_supervisor.py` | **TODO** | Anthropic tools, token/$ log, `always_on` baseline |
| `app.py` | **TODO** | Streamlit demo UI |

Legacy entrypoints: `crystal_smoke.py` (hour-0 smoke), `tests/test_*.py`.

---

## 3. Repository baseline

**Repo:** `tensormew` · **Codename:** CRYSTAL-HYBRID (research) / **PokéMind** (hackathon)

### 3.1 Environment (verified)

- Python 3.13+, venv `.venv`, `pyboy==2.8.1`
- ROM SHA1: `f2f52230b536214ef7c9924f483392993e226cfb` (Crystal US 1.1)
- Paths: `rom/pokemon_crystal.gbc`, `rom/pokemon_crystal.sym`
- Banked WRAM: `pyboy.memory[bank, addr]` for `0xD000–0xDFFF` (see `ram_parser` / `crystal_smoke.py`)
- SDL controls: arrows, `a` = A, `s` = B, Enter = Start

### 3.2 Hour-0 verification

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python crystal_smoke.py --headless
pytest tests/ -q
# In-game: python tools/make_state.py …  OR  crystal_smoke.py --make-state states/foo.state
```

### 3.3 Go/no-go checklist

1. Symbols resolve from sym
2. RAM sane in-game (`map != (0,0)` on real save)
3. Adjacent tiles (`wTile*`) change near walls
4. `step(dir)` moves one tile (`moved=True`)
5. `wScriptRunning` / `wTextboxFlags` nonzero in dialogue
6. One LLM call returns a valid tool invocation
7. At least one manifest checkpoint (`states/manifest.json`)

### 3.4 Target tree (hackathon)

```text
tensormew/
├── SPEC.md                    ← you are here
├── pokemon_agent.pdf          ← long research narrative (defer implementation)
├── crystal_smoke.py
├── requirements.txt
├── rom/pokemon_crystal.{gbc,sym}
├── states/manifest.json       ← checkpoint registry (+ gitignored *.state)
├── src/                       ← emulator, ram_parser, step, … + mover, macros, triggers, llm
├── tools/                     ← make_state, import_sav, run_episode, replay
├── tests/
├── app.py                     ← Streamlit (planned)
└── .cursor/rules/crystal-hybrid.mdc
```

---

## 4. RAM & Crystal constraints

### 4.1 Symbols (this ROM — do not hardcode addresses)

Use `pyboy.symbol_lookup(name)` or `RamParser`. PDF/pret may say `wPlayerStandingMapX`; **our sym uses `wXCoord` / `wYCoord`.**

| Symbol | Bank:addr (examples) | Use |
| --- | --- | --- |
| `wMapGroup`, `wMapNumber` | 01:dcb5, 01:dcb6 | Map id |
| `wYCoord`, `wXCoord` | 01:dcb7, 01:dcb8 | Position |
| `wFacingDirection` | 01:d044 | Facing / double-press |
| `wTileUp/Down/Left/Right` | 00:c2fa–c2fd | Adjacent collision (no Red-style grid) |
| `wPlayerTileCollision` | 01:d4e4 | Tile under player |
| `wBattleMode` | 01:d22d | 0 overworld, 1 wild, 2 trainer |
| `wScriptRunning`, `wTextboxFlags` | 01:d438, 00:cfcf | Stuck / dialogue gates |
| `wMapStatus` | 01:d432 | Map state |
| `wPartyCount`, party HP syms | 01:dcd7, … | Party |
| `wJohtoBadges` | 01:d857 | Badge bitfield |
| Bag counts | 01:d892, d8bc, d8d7 | Items |

**Why Crystal:** Gen 2 Cut is face tree → A → confirm; map id is `(group, number)`; PyBoy has no Crystal collision grid — only adjacent tile bytes.

### 4.2 Engineering traps (handle as needed)

| Issue | Hackathon approach | Full spec (defer) |
| --- | --- | --- |
| WRAM banks | Always banked read | Same |
| RTC / time of day | Note in saves; fix if encounters flip | `rtc_controller.py` lock |
| Phone / text softlocks | `wTextboxFlags` + `mash_text` macro | Async LLM |
| Move learn / evolve menus | Macro or LLM trigger | Tier 2 interrupt router |
| Nav graph races | Per-segment hardcoded paths first | SQLite WAL + ghost scouts |

---

## 5. Checkpoint campaign (progression engine)

### 5.1 Registry

- **`states/manifest.json`** — names, map, pos, badges, SHA1s (`StateRegistry` in `src/states.py`).
- Capture: `tools/make_state.py`, `register_file`, or `import_sav.py` (`.sav` → `rom/pokemon_crystal.gbc.ram` → continue → snapshot).
- **Never** commit intro/title snapshots (all-zero RAM).

### 5.2 Checkpoint priority (collect top → bottom)

| Tag | Purpose |
| --- | --- |
| `route_test` | New Bark / Route 29 — mover validation |
| `ilex_cut` | **Must-have** — HM Cut demo |
| `post_falkner` | Badge 1 |
| `post_bugsy` | Badge 2 |
| `pre_whitney` / `post_whitney` | Badge 3 — battle LLM |
| `post_morty` | Badge 4 |
| `post_chuck` / `post_jasmine` / `post_pryce` | Mid-game stretch |
| `post_clair` | Badge 8 — pre-E4 |
| `elite_four` | Stretch |

Existing example: `ilex_src` in manifest (verify tags/description match intent).

Every capture must be **labeled** so a later run can say why that snapshot exists. `StateRegistry` already stores `name`, `description`, and `tags`. Fill them on every `capture` / `register_file` / `import_sav`:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | Stable id, `^[a-z0-9_]+$` (for example `ilex_cut`) |
| `description` | yes | One sentence a human can read: where, what is about to happen |
| `tags` | yes | At least one progress tag from §5.2, plus `human` or `agent` for who saved it |
| reason (in `description` until a dedicated field exists) | yes | Why this point was saved: badge gate, Cut tree, stuck recovery, segment end |

Do not save unlabeled intro/title dumps. The agent may call a future `label_state(name, description, tags, reason)` tool when it reaches a milestone; a human uses the same fields via `tools/states.py`.

### 5.3 Decision trace (observability)

The demo must show **why** a step happened, not only the button outcome. The Streamlit thought log (§1.3) reads this trace.

On every **LLM** tool call and every later **RL** action, append one JSON line to `logs/decisions.jsonl` (gitignored with `logs/`):

| Field | Meaning |
| --- | --- |
| `timestamp`, `frame` | When |
| `map`, `pos`, `badges` | Where in the game |
| `actor` | `llm`, `rl`, or `script` |
| `trigger` | `CUT_BLOCKED`, `STUCK`, `BATTLE`, `DIALOGUE`, or empty if `always_on` |
| `tool` / `action` | What ran (`use_cut`, `go_to`, …) |
| `arguments` | Tool args, no secrets |
| `reason` | One sentence: why this action, given the state and trigger |
| `ok`, `est_usd`, token counts | Outcome and cost (LLM only) |

**LLM rule:** every tool schema includes a required `reason: str`. The model must not emit raw buttons. `reason` is logged and shown; it is not sent to the emulator.

**RL rule (post-hackathon, Appendix A):** the same row, with `actor=rl`. `reason` is a short policy note (chosen action, value, or stall cause). Do not train a separate narrator during the hackathon.

**Scripted macros** (`run_path` with no model) log `actor=script` and a fixed reason such as `segment script for ilex_cut`.

`logs/supervisor.jsonl` stays the cost log. `logs/decisions.jsonl` is the thought log. Both are append-only.

### 5.4 Segment playbook

For each checkpoint pair `(A → B)`:

1. Load state `A` via registry.
2. Run **scripted segment**: `run_path` + macros until map/badge matches `B` preconditions OR trigger fires.
3. On trigger → supervisor → tool → macro loop.
4. Capture state at segment end; append manifest.

Prefer **hardcoded paths per map** over global greedy navigation for demos (Union Cave, Sprout Tower, Ilex). Greedy `go_to` is the fallback when no script exists.

---

## 6. Hackathon implementation phases

**Parallel Phase 1 (two devs):** [`phase-1/01-dev-a-execution.md`](phase-1/01-dev-a-execution.md) · [`phase-1/02-dev-b-supervisor.md`](phase-1/02-dev-b-supervisor.md)

Work **H0 → H4** in order. One Cursor task per phase unless the user expands scope.

```text
H0 Infra ──► H1 Mover ──► H2 Macros+Triggers ──► H3 LLM ──► H4 Segments+UI
     (mostly done)              │                      │
                                 └── Ilex E2E ──────────┴── badge chain
```

| Phase | Deliverables | Verification |
| --- | --- | --- |
| **H0** | emulator, ram_parser, step, states, tools, tests | `pytest tests/ -q`, `crystal_smoke.py --headless` |
| **H1** | `mover.py`, walking on `route_test` save | `moved=True`, coords ±1, adj tiles update |
| **H2** | `macros.py`, `triggers.py`, Cut on `ilex_cut` | E2E: blocked → trigger → `use_cut()` → pass |
| **H3** | `llm_supervisor.py`, tool schema with `reason`, `always_on` baseline | Mock + one live tool call; cost log **and** decision trace |
| **H4** | Segment runners, `app.py` thought log from `logs/decisions.jsonl`, labeled manifest checkpoints | Furthest badge increased; demo laptop dry run |

**Rule:** Do not start PPO, ghost scouts, Metamon minimax, or LSTKAG build pipelines until **H4** is met or the user explicitly pivots to research mode.

### 6.1 Suggested team split (~24h)

| Hours | Focus |
| --- | --- |
| 0–4 | Checkpoints (manifest) · mover · LLM schema |
| 4–10 | Triggers · macros · Ilex E2E · baseline metrics |
| 10–16 | Next segments (Whitney, Union/Sprout) · **furthest badge** push |
| 16–22 | Streamlit · deck/video · backup runs |
| 22–24 | Demo laptop dry run |

### 6.2 Cut list (if behind — drop in this order)

1. Map overlay visualization  
2. Extra demo states beyond progression path  
3. Whitney polish → use simpler battle macro  
4. Greedy mover → segment-only `run_path`  
5. LLM baseline harness (keep triggers-only path working)

---

## 7. Cursor Agent workflow

- **Contract:** `@SPEC.md` — not `pokemon_agent.pdf` for day-to-day work.
- **One phase per task:** e.g. *"Implement H2 per SPEC §6; do not start H3."*
- **Run commands:** `pytest`, `crystal_smoke.py`, `tools/run_episode.py` — report exit codes.
- **Commits:** only when the user asks.
- **ROM:** do not rebuild from pret unless asked.

**Kickoff (H1):**

```text
Read @SPEC.md §2 and §6 H1. Implement src/mover.py using src/step.py and ram_parser GameState.
Add tests with mocked tiles. Validate on a route_test checkpoint when available.
```

---

## 8. Upstream source registry (post-hackathon / optional)

Rekursiv and research deps — **do not invent packages**. Use when building [Appendix A](#appendix-a-crystal-hybrid-research-roadmap-deferred).

| Concept | Repo |
| --- | --- |
| [rekursiv-ai](https://github.com/rekursiv-ai) org | Home |
| configgle, priml | Experiment config, ML blocks |
| trackinizer | Audit / beliefs DB |
| sagent | External phase harness (not in-emulator) |
| treekle | Checkpoint serialization |
| pret/pokecrystal | ASM reference, optional ROM regen |
| PyBoy | Emulator |
| UT-Austin-RPL/metamon | Gen2OU combat (research phase) |

Craftax patterns (frontier buffer, action-effect head) live in `pokemon_agent.pdf` + priml/configgle — **not** hackathon scope.

---

## Appendix A — CRYSTAL-HYBRID research roadmap (deferred)

Long-term **3-tier** design from `pokemon_agent.pdf` / original SPEC:

- **Tier 3:** Worker A (A* autopilot), Worker B (Recurrent PPO + Craftax curriculum), Worker C (Metamon + minimax)
- **Tier 2:** Interrupt router, SayCan gate, ghost scouts, stall caps
- **Tier 1:** LLM director, LSTKAG graph, Eureka rewards
- **Training:** `vec_env_shm`, `frontier_buffer`, 25 `states_fixtures`, Phases 1–6 CI

**Mapping to hackathon code (when continuing after DubHacks):**

| Research component | Hackathon seed |
| --- | --- |
| Worker A | `mover.py` + `run_path` |
| Tier 2 router | `triggers.py` + `wBattleMode` |
| Tier 1 | `llm_supervisor.py` |
| Worker C | `macros.py` battle tools → later Metamon |
| Worker B / scouts / PPO | Not started |

Read the PDF for equations, WRAM tables, and acceptance benchmarks when resuming research mode.

---

## Appendix B — Cursor project rules

Live copy: [`.cursor/rules/crystal-hybrid.mdc`](.cursor/rules/crystal-hybrid.mdc) (`alwaysApply: true`).

Summary:

- **Goal:** maximize game progression via checkpoints + tools-on-triggers (§1).
- **Never** hardcode hex; use sym + banked reads.
- **Never** LLM → raw buttons; **never** LLM write to nav DB (when added).
- **Battle:** macros first when `wBattleMode != 0`.
- **Phases:** H0–H4 (§6), not PDF Phases 1–6, unless user switches to research mode.

---

*Canonical spec for tensormew. Narrative detail: `pokemon_agent.pdf`. Hour-0: `crystal_smoke.py`.*
