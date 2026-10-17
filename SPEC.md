# System Specification: Project CRYSTAL-HYBRID

**Autonomous Hybrid RL-LLM Handoff Architecture for Pokémon Crystal (GBC)**

This specification defines a modular, spec-driven blueprint designed for autonomous implementation via **Cursor Agent** (Composer / Agent mode in Cursor IDE), with optional use of other phase-gated harnesses (e.g. [rekursiv-ai/sagent](https://github.com/rekursiv-ai/sagent)). Every component has strict type interfaces, deterministic state invariants, and self-verifying unit/simulation tests so the agent can build, test, and debug one phase at a time without polluting its context window.

---

## 0. Upstream Source Registry (mandatory — do not invent)

**All Rekursiv-related infrastructure in this spec must be implemented by vendoring, adapting, or explicitly depending on code from the [rekursiv-ai GitHub organization](https://github.com/rekursiv-ai). Do not hallucinate module paths, APIs, or package names that are not present in those repositories or in this spec.**

| Spec concept | Authoritative source | Repo URL |
| --- | --- | --- |
| Organization home | rekursiv.ai OSS | https://github.com/rekursiv-ai |
| Experiment / training config (pyproject, hierarchical runs, DI) | **configgle** | https://github.com/rekursiv-ai/configgle |
| Typed ML building blocks & A/B training layouts | **priml** | https://github.com/rekursiv-ai/priml |
| Beliefs, efforts, findings, scout/subgoal audit trail (optional upgrade for `nav_memory.db` semantics) | **trackinizer** | https://github.com/rekursiv-ai/trackinizer |
| Phase-gated coding-agent CLI patterns (harness only — never in-emulator control) | **sagent** | https://github.com/rekursiv-ai/sagent |
| Web research tools for Tier 1 (optional) | **wesearch** | https://github.com/rekursiv-ai/wesearch |
| Checkpoint / tree serialization (`FrontierEntry`, `.pt` sidecars) | **treekle** | https://github.com/rekursiv-ai/treekle |
| Monorepo → publishable tree sync (if splitting `tensormew/`) | **copybarista** | https://github.com/rekursiv-ai/copybarista |
| Console / agent log rendering in CI | **madcatter** | https://github.com/rekursiv-ai/madcatter |

**Craftax naming in this document** refers to Rekursiv’s published RL curriculum patterns (frontier replay buffer, proportional reward scaling, action-effect auxiliary head, 10k stall cap, anti-loop shaping) as specified in §3.1 Worker B, §4.2, and §3.2 `critique_verifier.py`. There is no separate public `craftax` repository under [rekursiv-ai](https://github.com/rekursiv-ai) at the time of this spec; implement `frontier_buffer.py` and Worker B training logic **from this spec plus patterns found in [priml](https://github.com/rekursiv-ai/priml) and [configgle](https://github.com/rekursiv-ai/configgle)**. Before adding a dependency, `grep` the org or read repo READMEs — do not invent a `rekursiv-craftax` PyPI package.

**Non-Rekursiv upstream (also mandatory):**

| Spec concept | Repo URL |
| --- | --- |
| ROM, `.sym`, LSTKAG assembly source | https://github.com/pret/pokecrystal |
| PyBoy emulator | https://github.com/PyBoy/PyBoy |
| Metamon Gen2OU combat | https://github.com/UT-Austin-RPL/metamon |

**Agent rule:** If a module is not listed in §1’s tree and not in the table above, stop and ask — do not create a placeholder package.

**ROM in this repo:** `rom/pokemon_crystal.gbc` and `rom/pokemon_crystal.sym` are **already committed**. Do not block Phase 1 on cloning or compiling [pret/pokecrystal](https://github.com/pret/pokecrystal) unless regenerating the ROM/sym pair is explicitly requested. LSTKAG / knowledge-graph work still uses pret assembly as a **read-only reference**, not as a runtime build step.

---

## 1. Repository Structure & Agentic Coding Workflow

**Repo name:** `tensormew` (project codename **CRYSTAL-HYBRID**). Paths below are relative to the repository root.

To prevent an agentic coding tool from hallucinating cross-module imports or breaking emulator timing invariants, the repository is partitioned into isolated, contract-bound subsystems. The **checked-in** `rom/pokemon_crystal.sym` plus PyBoy `symbol_lookup()` are the single source of truth for memory addresses in this codebase.

### 1.1 Current baseline (what exists today)

| Artifact | Status | Notes |
| --- | --- | --- |
| `rom/pokemon_crystal.gbc` | **Present** | Vendored Crystal ROM (~2 MiB); loaded by PyBoy |
| `rom/pokemon_crystal.sym` | **Present** | rgblink symbol table; pass to `PyBoy(..., symbols=SYM)` |
| `crystal_smoke.py` | **Present** | Hour-0 smoke: boot, symbol resolve, banked WRAM read, optional save-state walk test |
| `requirements.txt` | **Present** | Includes `pyboy==2.8.1`, `pydantic`, `anthropic`, `streamlit`, etc. |
| `src/__init__.py` | **Present** | Empty package stub only |
| `SPEC.md`, `README.md`, `pokemon_agent.pdf` | **Present** | Spec + PDF export |
| `states/` | **Runtime** | Created by smoke test; `*.state` is gitignored (see `.gitignore`) |
| `.cursor/rules/crystal-hybrid.mdc` | **Present** | Cursor Agent operating directives (§6) |
| `Makefile`, `pyproject.toml`, `tests/`, `data/`, tier modules | **Planned** | Introduce per phase; do not assume they already exist |

**Hour-0 verification (run before Phase 1 refactors):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python crystal_smoke.py --headless
# With a saved game: python crystal_smoke.py --headless --state states/your.state
```

**Canonical ROM paths (do not rename without updating the spec and smoke test):**

```python
ROM = "rom/pokemon_crystal.gbc"
SYM = "rom/pokemon_crystal.sym"
```

Refactor rule: when implementing `src/core/wram_crystal.py` and `sym_parser.py`, **port behavior from `crystal_smoke.py`** (`resolve`, `read_byte`, `read_state`, `step`) rather than rewriting from scratch.

### 1.2 Target layout (build toward this tree)

Legend: `[✓]` exists · `[ ]` not yet in repo

```text
tensormew/
├── [✓] SPEC.md
├── [✓] README.md
├── [✓] pokemon_agent.pdf
├── [✓] requirements.txt          # migrate to pyproject.toml when training stack lands
├── [✓] .cursor/rules/
│   └── [✓] crystal-hybrid.mdc  # Cursor project rules (duplicate of SPEC.md §6)
├── [ ] Makefile                  # optional: regenerate rom/ from pret; pytest; vector sims
├── [ ] pyproject.toml            # add with PyTorch / CleanRL when Phase 3 starts
├── [✓] crystal_smoke.py          # keep until Phase 1 tests supersede; then thin CLI wrapper
├── [✓] rom/
│   ├── [✓] pokemon_crystal.gbc   # vendored; primary runtime ROM (not roms/pokecrystal.gbc)
│   └── [✓] pokemon_crystal.sym
├── [ ] data/
│   ├── knowledge_graph/          # parsed pret tables (build script TBD)
│   ├── states/                   # local golden saves (gitignored *.state)
│   └── schema/                   # JSON schemas for nav_memory, subgoal_spec
├── [ ] src/
│   ├── [✓] __init__.py
│   ├── core/
│   │   ├── sym_parser.py         # parse rom/pokemon_crystal.sym + PyBoy symbol_lookup
│   │   ├── wram_crystal.py       # banked reads (from crystal_smoke.read_byte)
│   │   ├── rtc_controller.py
│   │   └── pyboy_env.py
│   ├── tier3_workers/            # worker_a_autopilot, worker_b_explorer, worker_c_combat
│   ├── tier2_broker/             # interrupt_router, saycan_gate, ghost_scouts, critique_verifier
│   ├── tier1_director/           # planner_agent, lstkag_retriever, teduo_mask, eureka_synthesizer
│   └── training/                 # vec_env_shm, frontier_buffer, ppo_trainer
└── [ ] tests/
    ├── states_fixtures/          # 25 canonical .state fixtures (Phase 4+)
    └── test_phase1_substrate.py … test_phase5_handoff.py
```

### 1.3 Cursor Agent workflow (how this repo gets built)

**Primary implementer:** Cursor Agent in this workspace (`tensormew`). Treat `SPEC.md` as the contract; pin it in chat with `@SPEC.md` (and `@crystal_smoke.py` during Phase 1).

| Practice | Why |
| --- | --- |
| **One phase per Agent task** | Ask explicitly: *"Implement Phase 2 per SPEC.md §5; do not start Phase 3."* Reduces cross-tier import hallucinations. |
| **Tests before implementation** | Match §5: add or extend `tests/test_phaseN_*.py` first, run `pytest` in the integrated terminal, then implement until green. |
| **Run smoke + pytest locally** | Agent must execute commands (not only suggest them): `python crystal_smoke.py --headless`, then phase pytest files. |
| **Project rules** | Keep §6 content in `.cursor/rules/crystal-hybrid.mdc` with `alwaysApply: true` so every Agent session inherits architectural law. |
| **Context discipline** | `@`-reference only files for the current phase (e.g. `@src/core/` during Phase 1). Avoid loading full `pokemon_agent.pdf` if `SPEC.md` suffices. |
| **No ROM rebuild by default** | Do not clone pret or replace `rom/` unless the user asks; vendored ROM/sym are authoritative (§1.1). |
| **Commits** | Commit only when the user asks; still run verification commands before reporting a phase complete. |

**Suggested kickoff prompt (Phase 1):**

```text
Read @SPEC.md §1.1, §2.1, and §5 Phase 1. Port crystal_smoke.py into src/core/ per the refactor rule.
Write tests/test_phase1_substrate.py first. Run pytest and crystal_smoke.py --headless before done.
Do not implement Phase 2 yet.
```

---

## 2. Unconsidered Engineering Factors (Critical for Pokémon Crystal)

Before writing code, six non-obvious hardware and systems traps in Pokémon Crystal (GBC / MBC3) must be engineered into the core substrate.

### 2.1 Zero-Hardcoded Hex via pokecrystal.sym Parsing

Unlike Pokémon Red, where community projects copy-paste raw hex addresses (`0xD35E`), Pokémon Crystal uses 8 switchable 4KB Work RAM banks (WRAM0 at `0xC000–0xCFFF` and switchable WRAM1–WRAM7 mapped into `0xD000–0xDFFF`, controlled by hardware register SVBK at `0xFF70`). Furthermore, addresses shift between Crystal v1.0 and v1.1.

- **In this repo:** Use `rom/pokemon_crystal.sym` with `PyBoy(ROM, symbols=SYM)` and `pyboy.symbol_lookup(name)` → `(bank, addr)`. Symbol lines look like `01:dcb8 wXCoord` (rgblink format).
- **Regenerating ROM/sym (optional):** Clone [pret/pokecrystal](https://github.com/pret/pokecrystal), build with RGBDS, and replace `rom/pokemon_crystal.{gbc,sym}` only if you intentionally rebaseline; re-run `crystal_smoke.py` and update symbol names in tests if the sym drifts.
- **PyBoy banked direct access:** For addresses in `0xD000–0xDFFF`, use `pyboy.memory[bank, addr]` with the bank from symbol lookup — see `crystal_smoke.read_byte`. Do not read switchable WRAM with a bare address.

**Symbols verified in `crystal_smoke.py` against this ROM** (use these names in new code until a parser maps pret aliases):

| Logical domain | Symbol in `rom/pokemon_crystal.sym` | Example line | Used for |
| --- | --- | --- | --- |
| Map ID | `wMapGroup`, `wMapNumber` | `01:dcb5 wMapGroup` | Position / curriculum |
| Player tile | `wXCoord`, `wYCoord` | `01:dcb8 wXCoord` | Movement, A*, scouts |
| Facing | `wFacingDirection` | `01:d044 wFacingDirection` | Double-press (Phase 1+) |
| Adjacent tiles | `wTileUp/Down/Left/Right`, `wPlayerTileCollision` | (lookup) | Greedy mover / Worker B local grid |
| Battle / UI gates | `wBattleMode`, `wScriptRunning`, `wTextboxFlags`, `wMapStatus` | `01:d22d wBattleMode` | Interrupt router |
| Progress | `wPartyCount`, `wJohtoBadges` | `01:d857 wJohtoBadges` | SayCan / badges |

Pret documentation often cites `wPlayerStandingMapX` / `wPlayerDirection`; **this vendored sym uses `wXCoord` / `wFacingDirection`**. Never hardcode either spelling’s address — always resolve from the sym file. When LSTKAG adds pret names, `sym_parser.py` should map aliases → the lookup PyBoy returns.

| Logical domain | pret / doc alias (reference only) | Bank | Function in hybrid architecture |
| --- | --- | --- | --- |
| Map & spatial | `wMapGroup`, `wMapNumber` | WRAM1 (01) | Unique 2-byte compound Map ID (group, number) |
| Player tile coords | `wPlayerStandingMapX/Y` → **`wXCoord` / `wYCoord` here** | WRAM1 (01) | Player tile position |
| Facing direction | `wPlayerDirection` → **`wFacingDirection` here** | WRAM1 (01) | Bits govern double-press |
| Off-screen NPC array | `wMapObjects` (16×16 bytes), `wObjectStructs` (13×40 bytes) | WRAM1 (01), WRAM0 (00) | Active sprites vs. map object templates (NPC collision caching) |
| Battle interrupts | `wBattleMode` | WRAM1 (01) | 0 = Overworld, 1 = Wild, 2 = Trainer (60Hz interrupt trigger) |
| Party state | `wPartyCount`, `wPartySpecies`, `wPartyMons` (6×48 bytes) | WRAM1 (01) | Species, HP, Level, Status, Moves, PP, Held Item, Experience |
| Enemy battle state | `wEnemyMon`, `wEnemyMonSpecies`, `wEnemyMonHP`, `wEnemyMonMoves` | WRAM1 (01) | Live opponent state fed to Worker C (Metamon) on battle interrupt |
| Progression gates | `wJohtoBadges`, `wKantoBadges`, `wEventFlags` (250 bytes) | WRAM1 (01) | Bitmasks for SayCan prerequisite checks and TEDUO state masking |
| Real-time clock | `wTimeOfDay`, `hHours`, `hMinutes` | WRAM1 / HRAM | 0 = Morning, 1 = Day, 2 = Night |

### 2.2 Deterministic MBC3 Real-Time Clock (RTC) & PRNG Seeding

Pokémon Crystal uses an MBC3 hardware Real-Time Clock. If 64 parallel RL workers run at real-time speed while reading the host OS system clock, wild encounter tables (`data/wild/johto_grass.asm`) will flip unpredictably between Morning, Day, and Night mid-rollout, breaking reproducibility.

- **Solution:** Implement `RTCController` in `src/core/rtc_controller.py` that overrides `hHours` and `wTimeOfDay` in RAM at the start of every episode (or locks RTC registers via PyBoy memory hooks) so time of day is a controllable environment parameter passed by Tier 1 (`target_time_of_day: Literal["MORN", "DAY", "NITE"]`). Similarly, lock the hardware divider and PRNG bytes (`hRandomAdd`, `hRandomSub`) during deterministic unit tests.

### 2.3 Concurrency-Safe Shared Navigation Graph (SQLite WAL + Shared Memory)

When 5 Ghost Scouts (Tier 2) and 1 Main Agent (Tier 3) simultaneously discover tiles and off-screen NPC positions, writing to a flat `nav_memory.json` file causes race conditions and corrupted JSON reads.

- **Solution:** Store the persistent world graph in an in-memory/disk SQLite database in WAL mode (`nav_memory.db`) with tables `maps`, `tiles`, `npcs`, and `warps` as described in the original spec. Expose a read-only connection (`?mode=ro`) to Tier 1 (`planner_agent.py`) so the LLM can never alter map state.

### 2.4 Asynchronous Non-Blocking LLM Planning & Watchdog Recovery

If Tier 1 (LLM Director) takes 4–8 seconds to reason over a sub-goal, blocking the vectorized training or scout pool wastes thousands of emulator frames.

- **Solution:** When a sub-goal finishes or stalls, the main emulator freezes on a safe overworld frame while Ghost Scouts continue mapping adjacent edges asynchronously. Every LLM output is validated against a strict Pydantic schema with AST sandbox verification for Eureka reward snippets (banning `import`, `open`, `exec`, `eval`, `__`).

### 2.5 Move-Replacement & Evolution Interrupt Handler

When a Pokémon levels up in combat and learns a 5th move, or attempts to evolve post-battle, the game enters a modal dialog that breaks standard battle or overworld policies.

- **Solution:** The Tier 2 Interrupt Router monitors `wMoveSelectionMenuType` and evolution state flags. For move replacement, Tier 2 computes the expected DPS/utility delta of the new move vs. the 4 current moves using the LSTKAG move table; if the new move strictly dominates the weakest move, a deterministic macro overwrites that slot. (HMs cannot be overwritten without the Blackthorn Move Deleter — Tier 1 assigns HMs to designated utility slots.)

---

## 3. Core Subsystem Technical Specifications

### 3.1 Tier 3 Workers (Low-Level Execution Substrate)

```text
+-----------------------------------------------------------------------------------+
|                        TIER 2: 60Hz WRAM INTERRUPT ROUTER                         |
|   Polls (01:wBattleMode), (01:wMapGroup/Number), (01:wPartyMons HP/PP), StallCap  |
+-------------------+-----------------------------------+---------------------------+
                    |                                   |
       [wBattleMode == 0]                  [wBattleMode == 1 or 2]
                    |                                   |
        +-----------+-----------+                       v
        |                       |           +---------------------------------------+
  [Path Known in DB]    [Unmapped / Puzzle] | WORKER C: COMBAT SPECIALIST           |
        |                       |           | - Wild/Grind: Metamon Gen2OU (15ms)   |
        v                       v           | - Boss/Gym: PokéChamp 3-ply Minimax   |
+---------------+     +-------------------+ +---------------------------------------+
| WORKER A: A*  |     | WORKER B: RL PPO  |
| - DoublePress |     | - 10x9 Tile CNN   |
| - Door Step   |     | - Action-Effect   |
| - 180f Settle |     | - Visited Mask    |
+---------------+     +-------------------+
```

#### Worker A: Deterministic A* & Programmatic Macro-Skills (`worker_a_autopilot.py`)

- **Input contract:** `AutopilotGoal(target_map: tuple[int,int], target_xy: tuple[int,int], action_macro: Optional[str])`
- **Movement physics wrapper (`step_tile(direction)`):** Read `wPlayerDirection`; rotate with 4+4 frames if needed; step with 8+12 frames; on map change, 180-frame warp settle and doorway sidestep reflex on warp carpets (`0x71` / `0x7A`).
- **Directed A* cost function:** Walkable = 1.0; tall grass = 1.0 if `subgoal.mode == "GRIND"` else 4.5; one-way ledges = 2.0 if allowed direction else ∞; cached NPC = high cost or 33.0 penalty buffer for wandering NPCs.

#### Worker B: Goal-Conditioned Recurrent PPO Explorer (`worker_b_explorer.py`)

Combines PokeRL's anti-loop spatial masking with Rekursiv AI's Craftax architecture (see §0 — implement from this spec + [priml](https://github.com/rekursiv-ai/priml) / [configgle](https://github.com/rekursiv-ai/configgle), not from memory):

- **Action space:** `Discrete(5)` — `['UP', 'DOWN', 'LEFT', 'RIGHT', 'A']`
- **Observation space (O_t):** `screen_stack`, `visited_mask`, `local_tile_grid`, `teduo_state_vec` as in full spec
- **Neural architecture (~8.5M params):** NatureCNN + Craftax local board encoder + GRU(512) + policy / value / action-effect heads
- **Reward scaling:** `REWARD_SCALE = 8.0`; 3-layer anti-loop penalty R_loop(t)

#### Worker C: Combat Specialist & Fast Rollout Simulator (`worker_c_combat.py`)

- **State translator:** `wram_to_gen2_state()` → [UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon) Gen2OU tensors
- **Dual-mode:** Wild/routine via Metamon; boss/gym via 3-ply PokéChamp expectiminimax with lightweight damage calculator

### 3.2 Tier 2: Affordance Handoff Broker (`src/tier2_broker/`)

- **SayCan affordance gating (`saycan_gate.py`):** Combine LLM log-prob with physical affordance; 100× headless rollouts for bosses; veto below τ_ready
- **5× parallel Ghost Scouts (`ghost_scouts.py`):** Fork PyBoy state; wall-followers + biased random + high-temperature Worker B; write discoveries to `nav_memory.db`
- **Craftax hard stall cap (`critique_verifier.py`):** 2,500 subgoal / 10,000 milestone caps; rollback and escalate to Tier 1

### 3.3 Tier 1: LLM Meta-Planner & Synthesis Engine (`src/tier1_director/`)

Tier 1 is called asynchronously via structured tool-use (`SubGoalSpec` JSON schema).

- **LSTKAG retriever:** Build graph from pret/pokecrystal assembly during `make data`
- **TEDUO mask:** Binary mask over 160-feature WRAM vector
- **Eureka synthesizer:** Sandboxed AST reward functions

Example `SubGoalSpec`:

```json
{
  "subgoal_id": "goldenrod_counter_whitney_01",
  "worker_mode": "AUTOPILOT_THEN_GRIND",
  "target_map": [11, 4],
  "target_coords": [8, 12],
  "rtc_override": "DAY",
  "combat_directive": {
    "mode": "CATCH_TARGET",
    "target_species_sym": "DROWZEE",
    "hp_retreat_threshold": 0.30
  },
  "teduo_active_wram_keys": [
    "wMapGroup", "wMapNumber", "wXCoord", "wYCoord",
    "wPartyCount", "wPartySpecies", "wItemCount"
  ],
  "termination_predicate_expr": "PARTY_HAS_SPECIES('DROWZEE')",
  "max_step_budget": 1500
}
```

---

## 4. Mass Simulation & Training Optimization Architecture

### 4.1 Zero-Copy SharedMemory Vectorized PyBoy Pool (`vec_env_shm.py`)

Pre-allocate POSIX shared memory for N=64 or 128 headless PyBoy workers; batch transfer to CUDA via `torch.frombuffer(...).pin_memory()`; macro-tick frame skip (16–20 frames per RL step).

### 4.2 Rekursiv Craftax "Frontier Practice" Curriculum Buffer (`frontier_buffer.py`)

- Maintain priority ring buffer of **Frontier Checkpoints**
- **FrontierEntry** = `(PyBoy.state bytes, h_t ∈ R^512 GRU hidden, SQLite nav_memory snapshot)`
- **20% Frontier Reset Rule:** On reset, 80% curriculum start / 20% sample from FrontierBuffer (restore RAM + recurrent state)

### 4.3 Multi-Stage Parallel Save-State Bank (`tests/states_fixtures/`)

25 canonical `.state` fixtures from Elm lab through Indigo Elite Four (see phase deliverables for filenames).

---

## 5. Phase-Gated Agentic Implementation Plan

```text
[Phase 1: Substrate & .sym] ---> [Phase 2: Worker A + Scouts] ---> [Phase 3: Worker B (Craftax PPO)]
                                                                               |
[Phase 6: Full Campaign CI] <--- [Phase 5: Tier 1 + Eureka]   <--- [Phase 4: Worker C + SayCan Gate]
```

**Rule for Cursor Agent:** Do not advance to Phase N+1 until the verification command for Phase N exits with code 0 and meets all quantitative performance gates. Report the exact terminal commands and exit codes when marking a phase complete.

### Phase 1: Deterministic Emulation Substrate & Banked .sym Reader

**Already in repo:** vendored `rom/pokemon_crystal.{gbc,sym}`; `crystal_smoke.py` proves PyBoy boot, symbol lookup, and banked reads.

**Deliverables:** Extract core logic into `src/core/` (`sym_parser`, `wram_crystal`, `rtc_controller`, `pyboy_env`), add `src/training/vec_env_shm.py`, optional `Makefile` (pret rebuild only), `tests/test_phase1_substrate.py`.

**Verification:**

1. `python crystal_smoke.py --headless` — exit 0, resolves smoke symbol set (see §2.1).
2. `pytest tests/test_phase1_substrate.py` — determinism, bank safety, ≥25,000 FPS aggregate on 64 workers (once tests exist).

### Phase 2: Worker A + Ghost Scouts

Deliverables: `ghost_scouts.py`, SQLite schema, `worker_a_autopilot.py`.

Verification: doorway trap, off-screen NPC persistence, scout discovery <15s on Sprout Tower split floor.

### Phase 3: Worker B (Craftax PPO + Frontier Buffer)

Deliverables: `worker_b_explorer.py`, `frontier_buffer.py`, `ppo_trainer.py` — wire [configgle](https://github.com/rekursiv-ai/configgle) / [priml](https://github.com/rekursiv-ai/priml) for experiment configs where applicable.

Verification: action-effect ≥95% accuracy; zero 20-step loops in 100 episodes.

### Phase 4: Worker C + SayCan Gate

Deliverables: `worker_c_combat.py`, `interrupt_router.py`, `saycan_gate.py`.

Verification: Whitney rollouts <1.5s; SayCan veto/approve; ≥85% live Whitney win rate with counter team.

### Phase 5: Tier 1 LLM Director, LSTKAG, Eureka

Deliverables: knowledge graph builder, `lstkag_retriever`, `teduo_mask`, `eureka_synthesizer`, `planner_agent`.

Verification: zero-hallucination counter retrieval; Ice Path boulder Eureka reward passes AST + behavior checks.

### Phase 6: End-to-End Campaign Regression & Telemetry CI

Deliverables: `scripts/run_campaign.py` with telemetry dashboard.

Acceptance: early-game vertical slice <15M LLM tokens; full Johto + Red validation across golden states.

---

## 6. Cursor project rules (copy into `.cursor/rules/crystal-hybrid.mdc`)

Create `.cursor/rules/crystal-hybrid.mdc` in this repository so **Cursor Agent** loads operating directives on every session. (Legacy note: other tools may use a root `AGENTS.md` with the same body; this project standardizes on Cursor rules.)

**File path:** `.cursor/rules/crystal-hybrid.mdc`

```markdown
---
description: CRYSTAL-HYBRID / tensormew — phase-gated PyBoy RL-LLM architecture
alwaysApply: true
---

# Project CRYSTAL-HYBRID — Cursor Agent Operating Directives

## 0. Source of truth
- System spec: `SPEC.md` (this repository) — use `@SPEC.md` in Agent chat
- Cursor workflow: `SPEC.md` §1.3
- Rekursiv infra: https://github.com/rekursiv-ai — see SPEC.md §0; never invent Rekursiv package names
- Runtime ROM: `rom/pokemon_crystal.gbc` + `rom/pokemon_crystal.sym` (already in repo; do not require pret build)
- pret reference (optional rebuild only): https://github.com/pret/pokecrystal

## 1. Architectural Law
You are building a 3-Tier Hybrid RL-LLM system for Pokémon Crystal (GBC) on PyBoy.
- NEVER hardcode hex memory addresses. Resolve symbols from `rom/pokemon_crystal.sym` via `PyBoy.symbol_lookup` / `src/core/sym_parser.py` and read banked memory via `pyboy.memory[bank, addr]` (see `crystal_smoke.py`).
- NEVER allow Tier 1 (LLM) to emit raw button presses (`Up`, `Down`, `A`, `B`) or gain write access to `nav_memory.db`.
- NEVER allow Worker B (Overworld RL) to press `START` or `SELECT`, or clip rewards to `[-1, 1]`. Always scale rewards by `/ 8.0`.
- ALWAYS route combat at 60Hz via hardware interrupt when `01:wBattleMode != 0` directly to Worker C (zero LLM calls during routine wild battles).

## 2. Phase-Gated Workflow (Cursor Agent)
Work strictly one phase at a time (Phase 1 -> Phase 6). Scope each Agent task to a single phase unless the user explicitly expands scope.
1. Read the current phase in `SPEC.md` §5 and baseline in §1.1.
2. Write failing tests in `tests/test_phaseX_*.py` first.
3. Implement with strict Python 3.11+ type annotations and Pydantic schemas; port from `crystal_smoke.py` when touching WRAM/symbols (§1.1 refactor rule).
4. Run verification in the integrated terminal — do not skip:
   - Hour-0 / regression: `python crystal_smoke.py --headless`
   - Phase gate: `pytest tests/test_phaseX_*.py -v`
5. Do not proceed to the next phase until all assertions and FPS/win-rate benchmarks pass; report commands run and exit codes.
6. Create git commits only when the user asks.
```

---

*Canonical markdown spec for **tensormew** (CRYSTAL-HYBRID). Narrative PDF: `pokemon_agent.pdf`. Hour-0 entrypoint: `crystal_smoke.py`. Cursor rules: `.cursor/rules/crystal-hybrid.mdc` (§6).*
