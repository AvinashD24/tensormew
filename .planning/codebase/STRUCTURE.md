# Codebase Structure

**Analysis Date:** 2026-11-05

## Directory Layout

```
map-codebase/                         # repo root (worktree)
├── src/                              # Python agent package (42 modules, imported as `src.*`)
├── tools/                            # CLI entry points and training/ops scripts (24 modules, `python -m tools.<name>`)
├── tests/                            # pytest suite (42 `test_*.py` modules, fake emulator and client)
├── states/                           # named save-states + manifest.json (checkpoints)
├── models/                           # tracked trained weights (.pt, .npz) and README
├── rom/                              # Pokémon Crystal ROM + .sym symbol file (not rebuilt)
├── recordings/                       # committed run recordings: .state milestones, run.gif, screenshots
├── data/knowledge/                   # static JSON knowledge (bosses.json)
├── scripts/                          # demo launchers (.sh) and scripts/README.md
├── docs/                             # hackathon run notes, RL benchmark doc, recordings
├── research/                         # literature survey, codebase audit, paper drafts (markdown)
├── phase-1/                          # two parallel-dev execution briefs (markdown)
├── scratch/                          # ad-hoc experiments, screenshots, route30_nav.py
├── tensormewweb/                     # Vite + React 19 static site (separate npm project)
├── .planning/                        # GSD planning docs: PROJECT, REQUIREMENTS, ROADMAP, STATE, phases/, codebase/
├── .cursor/                          # Cursor agent rules and skills
├── app.py                            # Streamlit thought-log dashboard
├── crystal_smoke.py                  # boot / RAM / walk smoke test
├── pokemon_agent.md                  # long-form research spec (CRYSTAL-HYBRID)
├── SPEC.md                           # system spec (mission, RAM, checkpoints, phases H0-H4)
├── HANDOFF.md, README.md, CONTEXT.md # handoff notes, quick start, redirect stub
├── requirements.txt                  # Python dependencies
├── run_live.sh, run_4_live.sh        # live-play launchers (hard-coded macOS python path)
└── .gitignore                        # ignores .venv, __pycache__, *.state (except recordings), logs/, runs/, *.sav, etc.
```

Generated and gitignored at runtime (created on first use, not present in the tree): `logs/` (`logs/runs.db`, `logs/decisions.jsonl`, `logs/supervisor.jsonl`, `logs/live_status/`, plots) and `runs/` (episode logs and `runs/<id>.jsonl`).

## Directory Purposes

**`src/`:**
- Purpose: All agent logic. One concern per module, flat namespace (no subpackages).
- Contains: Emulator wrapper, RAM parsing, collision tables, deterministic movement and macros, navigators, trigger evaluation, LLM supervisor, tool schemas, segment and chain orchestration, RL environment, PPO and deep and recurrent policies, state registry, run log, recording, live status.
- Key files:
  - `src/emulator.py`: `Emulator`, `Machine` Protocol, `RomMismatchError`
  - `src/config.py`: ROOT-relative paths, `ROM_SHA1`, `sha1_file`
  - `src/ram_parser.py`: `RamParser`, `GameState`, `SYMBOLS`
  - `src/step.py`: `TileStepper`, `TileStep`, `DIRS`
  - `src/mover.py`: `Mover`, `walkable`
  - `src/macros.py`: `Macros`
  - `src/triggers.py`: `evaluate`, `TriggerKind`, `STUCK_THRESHOLD`
  - `src/llm_supervisor.py`: `Supervisor`, `SYSTEM_PROMPT`, `Decision`
  - `src/tools_schema.py`: pydantic tool argument models, `anthropic_tools`
  - `src/tool_dispatch.py`: `execution_registry`
  - `src/agent_loop.py`: `act`, `append_decision`
  - `src/segment.py`: `run_segment`, `SegmentResult`
  - `src/chain.py`: `run_chain`, `discovered_name`
  - `src/rl_env.py`: `ACTIONS`, `observe`, `progress_reward`, `navigation_potential`
  - `src/ppo.py`: `Policy`, `MlpPolicy`, `ppo_update`, `load_policy`
  - `src/deep_policy.py`: `SpatialVisionPolicy` (torch-optional, stub fallback at the bottom of the file)
  - `src/recurrent_policy.py`: `RecurrentPolicy`, `gru_step`
  - `src/states.py`: `StateRegistry`, `StateEntry`
  - `src/run_log.py`: `RunLog` (SQLite)

**`tools/`:**
- Purpose: Runnable scripts. Each is a `main()` with `argparse`, invoked as `python -m tools.<name>`.
- Contains: Live play, bounded segments, chains, scripted episodes, training (CPU PPO, marathon, GPU deep), cloud sync for RunPod, state creation and import, DB and state inspection, weight sync, monitoring.
- Key files:
  - `tools/play_live.py`: main live player (769 lines, inline priority chain)
  - `tools/run_segment.py`, `tools/run_chain.py`, `tools/run_episode.py`: `src/segment.py`, `src/chain.py`, `src/episode.py` drivers
  - `tools/train_ppo.py`, `tools/train_marathon.py`, `tools/train_deep_ppo.py`: training
  - `tools/cloud_train.py`, `tools/cloud_train_deep.py`, `tools/pod_checkpoint.py`: remote training helpers
  - `tools/make_state.py`, `tools/import_sav.py`: create checkpoints
  - `tools/states.py`, `tools/runs.py`, `tools/inspect_sav.py`, `tools/replay.py`: inspect and replay
  - `tools/live_status.py`, `tools/watch_training.py`, `tools/compare_policies.py`, `tools/eval_exits.py`, `tools/compare_cost.py`: observability and evaluation
  - `tools/sync_weights.py`: copy weights into `models/`
  - `tools/run_hybrid.sh`, `tools/run_hybrid.ps1`: record a Start-to-Falkner run into `recordings/`
  - `tools/run_campaign.py`: lists checkpoints (does not script the campaign)

**`tests/`:**
- Purpose: pytest suite. Tests use a fake `Machine` and fake LLM client; they do not need the ROM.
- Contains: One `test_<module>.py` per major module, plus integration-style tests (`test_segment.py`, `test_chain.py`, `test_live_execution.py`, `test_policy_walk.py`, `test_app.py`).
- Key files: `tests/test_segment.py`, `tests/test_triggers.py`, `tests/test_llm_supervisor.py`, `tests/test_dispatch.py`, `tests/test_states.py`, `tests/test_run_log.py`, `tests/test_ram_parser.py`, `tests/test_ppo.py`, `tests/test_pod_checkpoint.py` (new, uncommitted).

**`states/`:**
- Purpose: Named save-states and their manifest. These are the unit of progress.
- Contains: `states/manifest.json` (version 1, entries keyed by name with map, pos, badges, ROM SHA1, state SHA1, tags) and 11 tracked `*.state` files: `agent_map_24_7`, `agent_map_26_1`, `agent_map_26_3`, `elm_lab_starter`, `ilex_src`, `mystery_egg_received`, `new_bark_outside`, `post_rival_cherrygrove`, `route29_entry`, `smoke_sign`, `smoke_wall`.
- Note: `.gitignore` excludes `*.state`, so these files are tracked only because they were force-added. The manifest also lists `agent_map_24_3` and `agent_map_24_6`, whose `.state` files are not in git. `states/_autosave/` is ignored.

**`models/`:**
- Purpose: Tracked trained policy weights. `play_live` falls back here when `--policy` is missing.
- Contains: `models/deep_policy.pt` (spatial-vision), `models/deep_policy_baseline_351cb55.pt` (baseline snapshot, name suffix looks like a commit prefix), `models/ppo_policy.npz` (MLP), `models/README.md`.
- Note: `tools/play_live.py` defaults to `logs/deep_policy.pt`, and if that path is missing it falls back to `models/<same basename>`.

**`rom/`:**
- Purpose: The ROM and its symbol file. The ROM hash is checked before every emulator boot.
- Contains: `rom/pokemon_crystal.gbc`, `rom/pokemon_crystal.sym`. Never regenerate or edit these.

**`recordings/`:**
- Purpose: Shareable run captures. `.gitignore` explicitly un-ignores `recordings/**/*.state`.
- Contains: `recordings/falkner_full_story/`, `recordings/falkner_run_1..4/`, `recordings/falkner_run_merged_2x/`, `recordings/falkner_showcase/`. Each holds per-milestone `.state` files, `run.gif`, and sometimes `screenshots/` with `INDEX.md` and `events.json`.

**`data/knowledge/`:**
- Purpose: Static knowledge loaded at runtime.
- Contains: `data/knowledge/bosses.json`, read by `src/knowledge.py` `lookup_boss` (default path argument is `None`, so the caller decides which file is loaded).

**`scripts/`:**
- Purpose: Demo launchers. Each `.sh` runs a `tools.*` module (for example `tools.play_live`).
- Contains: `scripts/run_demo_falkner.sh`, `scripts/run_demo_fast_headless.sh`, `scripts/run_demo_full_game.sh`, `scripts/run_demo_watch_recording.sh`, `scripts/run_4_instances.sh`, `scripts/README.md`.
- Note: some launchers (`run_live.sh`, `scripts/run_demo_falkner.sh`) hard-code `/opt/homebrew/Caskroom/miniconda/base/bin/python` as the interpreter. They need edits to run on this Windows checkout.

**`docs/`, `research/`, `phase-1/`, `.planning/`:**
- Purpose: Documentation only. Not imported by code.
- `docs/`: `docs/HYBRID_FALKNER_RUN.md`, `docs/observability-rl-benchmark.md`, `docs/recordings/*.gif`.
- `research/`: `research/INDEX.md`, `research/analysis/codebase-audit.md`, `research/prior-work/*.md`, `research/paper-draft/*.md`, `research/papers/*.md`, `research/ideas/paper-directions.md`.
- `phase-1/`: `phase-1/01-dev-a-execution.md`, `phase-1/02-dev-b-supervisor.md`.
- `.planning/`: `PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, `phases/01-infrastructure/01-01..01-07-PLAN.md`, and `codebase/` (this document set).

**`tensormewweb/`:**
- Purpose: Standalone marketing and demo site. Has its own `package.json`, `package-lock.json`, `vite.config.js`, `index.html`, and `.gitignore`.
- Contains:
  - `tensormewweb/src/main.jsx`, `tensormewweb/src/App.jsx`
  - `tensormewweb/src/components/`: `LandingTab.jsx`, `PokedexTab.jsx`, `BlogTab.jsx`, `FoundersTab.jsx`, `LiveAgentTab.jsx`, `Navbar.jsx`, `PixelBackground.jsx`, plus `ui/XpBar.jsx`
  - `tensormewweb/src/data/`: `agentTelemetryData.js`, `pokedexData.js`, `blogData.js`, `screenshotsData.js` (static content)
  - `tensormewweb/src/utils/soundEffects.js`
  - `tensormewweb/src/index.css`
  - `tensormewweb/public/`: logos, fonts, screenshots, `recordings/falkner_showcase.gif`
  - `tensormewweb/Pokemon GSC Font/`: font source and license (duplicate of the public font)

**`phase-1/` vs `.planning/phases/`:** Both hold planning briefs. `phase-1/` is the hackathon split (Dev A, Dev B). `.planning/phases/` is the GSD-tracked plan set for infrastructure.

**`scratch/`:**
- Purpose: One-off experiments. Not imported.
- Contains: `scratch/route30_nav.py`, `scratch/kyle_*.png`, `scratch/trade_step*.png`.

## Key File Locations

**Entry Points:**
- `tools/play_live.py`: main live player (policy plus optional LLM bottleneck help)
- `tools/run_segment.py`: one bounded trigger-driven segment
- `tools/run_chain.py`: chained segments with checkpoint capture
- `tools/run_episode.py`: scripted or random button play, no model
- `app.py`: Streamlit thought-log dashboard (`streamlit run app.py`)
- `crystal_smoke.py`: boot, symbol resolution, RAM read, walk test
- `tensormewweb/src/main.jsx`: web site entry

**Configuration:**
- `src/config.py`: paths, ROM hash, `RUNS_DIR`, `STATES_DIR`, `MANIFEST_PATH`
- `requirements.txt`: Python deps
- `tensormewweb/vite.config.js`: dev server (port 5173, host true)
- `tensormewweb/package.json`: web deps and scripts
- `.gitignore`: ignore rules and the recordings exception
- `.gitattributes`: line-ending and binary rules

**Core Logic:**
- `src/ram_parser.py`: game state
- `src/mover.py`, `src/macros.py`, `src/step.py`: actuation
- `src/triggers.py`, `src/llm_supervisor.py`, `src/tools_schema.py`: trigger and LLM
- `src/segment.py`, `src/agent_loop.py`, `src/chain.py`: orchestration
- `src/rl_env.py`, `src/ppo.py`, `src/deep_policy.py`: RL
- `src/states.py`, `src/run_log.py`: persistence

**Testing:**
- `tests/`: all tests (flat, `tests/__init__.py` present)
- Run: `pytest tests/ -q` from the repo root (per `README.md`)

## Naming Conventions

**Python files and modules:**
- `snake_case.py`, one concept per module: `src/local_nav.py`, `src/return_nav.py`, `src/falkner_nav.py`, `src/gym_prep.py`
- Tests: `tests/test_<module>.py` matching a module name in `src/` or `tools/`. Integration-style tests use descriptive names (`test_policy_walk.py`, `test_live_execution.py`).
- Tool scripts: verb-first names for drivers (`run_*.py`, `train_*.py`, `cloud_*.py`, `import_*.py`, `inspect_*.py`), noun names for inspectors (`states.py`, `runs.py`).

**Python identifiers:**
- Functions and methods: `snake_case` (`run_segment`, `go_to`, `mash_text`, `load_policy`, `dialogue_open`)
- Classes: `PascalCase` (`RamParser`, `GameState`, `TileStepper`, `SpatialVisionPolicy`)
- Constants: `UPPER_SNAKE` (`ROM_SHA1`, `ACTIONS`, `OBS_EVENT_FLAGS`, `STUCK_THRESHOLD`, `DEFAULT_DECISION_LOG`)
- Private helpers: leading underscore (`_cut_sides`, `_prepare_mover`, `_party_hp`, `_Rom`)
- Tool names are snake_case verbs matching the pydantic model names in `src/tools_schema.py` (`go_to` maps to `GoToArgs`, `run_path` to `RunPathArgs`, and so on)

**Game and state names:**
- Named save-states: lowercase snake_case, enforced by `_NAME = re.compile(r"^[a-z0-9_]+$")` in `src/states.py`. Examples: `mystery_egg_received`, `new_bark_outside`, `route29_entry`.
- Auto-discovered checkpoints: `agent_map_<map_group>_<map_number>` (`src/chain.py` `discovered_name`), for example `agent_map_24_3`.
- Run tags: `w<wave><kind><n>` for `logs/live_status/` files (`tools/compare_policies.py`).
- Recording directories: `recordings/<name>/`, with per-milestone state filenames such as `falkner_fight.state`, `gym_entered.state`.

**Directories:**
- Python packages: lowercase, flat (`src/`, `tools/`, `tests/`). `tools/` and `tests/` have `__init__.py`. `src/` has `__init__.py`.
- Web: `tensormewweb/src/components/` for PascalCase React components, `tensormewweb/src/data/` for `camelCaseData.js` modules, `tensormewweb/src/utils/` for helpers.

**React components:**
- `PascalCase.jsx`, one component per file, each a `*Tab.jsx` for top-level tabs (`LandingTab.jsx`, `PokedexTab.jsx`). Shared UI primitives go under `components/ui/`.

**Docs:**
- Root docs: `UPPERCASE.md` (`SPEC.md`, `HANDOFF.md`, `CONTEXT.md`)
- Planning and codebase analysis: `UPPERCASE.md` under `.planning/` and `.planning/codebase/`
- Research and dev briefs: `kebab-case.md` (`research/prior-work/rl-techniques-long-horizon.md`, `phase-1/01-dev-a-execution.md`)
- Plan files: `NN-NN-PLAN.md` under `.planning/phases/<NN-name>/`

**Run output (gitignored):**
- `logs/decisions.jsonl`, `logs/supervisor.jsonl`, `logs/runs.db`, `logs/live_status/<tag>.json`, `logs/*.png`
- `runs/<id>.jsonl`

## Where to Add New Code

**New tool handler the LLM or policy can call:**
- Implementation: add a method on `src/mover.py` `Mover` or `src/macros.py` `Macros`, returning `ActionResult` (or a subclass).
- Schema: add a `Reasoned` subclass in `src/tools_schema.py`, register it in `TOOL_MODELS` and `TOOL_DESCRIPTIONS`, and extend the `anthropic_tools()` output.
- Dispatch: map the name in `src/tool_dispatch.py` `execution_registry`.
- Tests: `tests/test_macros.py` or `tests/test_mover.py` for the handler, `tests/test_tools_schema.py` for the schema, `tests/test_dispatch.py` for the mapping.

**New trigger:**
- Implementation: add a `TriggerKind` member and a check in `src/triggers.py` `evaluate`, keeping BATTLE, DIALOGUE, CUT_BLOCKED, STUCK order unless intentionally changing priority.
- Tests: `tests/test_triggers.py`.

**New navigator for a story segment:**
- Implementation: `src/<area>_nav.py` (follow `src/falkner_nav.py` or `src/gym_prep.py`). Reuse `src/mover.py` `walkable` and `src/return_nav.py` `_Rom` for ROM-derived collision.
- Wiring: add the branch to the priority chain in `tools/play_live.py` (step 4 region), and to `src/segment.py` `_step_direction` if it should also apply to segments.
- Tests: `tests/test_<area>_nav.py` (see `tests/test_falkner_nav.py`, `tests/test_return_nav.py`).

**New reward term or observation feature:**
- Implementation: `src/rl_env.py`. Keep `OBS_DIM` consistent with any checkpoint width; `src/ppo.py` imports `MASK_START`, `N_ACTIONS`, `OBS_DIM`.
- Tests: `tests/test_ppo.py`, `tests/test_improved_training.py`.

**New policy architecture:**
- Implementation: a new module under `src/` exposing `act`, `probs`, `value`, and `load`. Register loading in `tools/play_live.py` (the `.pt` vs `.npz` dispatch around line 175) or in `src/ppo.py` `load_policy` for `.npz`.
- Tests: `tests/test_deep_policy.py` or `tests/test_recurrent_policy.py`.

**New CLI or driver:**
- Implementation: `tools/<name>.py` with a `main()` and `if __name__ == "__main__":`. Import from `src.*` using repo-root-relative imports. Run as `python -m tools.<name>`.
- Tests: add a test module under `tests/` that calls `main([...])` with a fake machine when possible.

**New checkpoint:**
- Create with `tools/make_state.py` or `tools/import_sav.py`, or use `src/chain.py` automatically. Register and validate with `tools/states.py`. Entries are written to `states/manifest.json`.

**New web page or tab:**
- Component: `tensormewweb/src/components/<Name>Tab.jsx`
- Data: `tensormewweb/src/data/<name>Data.js`
- Wiring: add the tab to `tensormewweb/src/App.jsx` `activeTab` branches and to `tensormewweb/src/components/Navbar.jsx`.

**Shared utilities:**
- Python: `src/config.py` for paths and hashes, `src/game_data.py` for name lookups, `src/collision.py` for tile logic, `src/story_flags.py` for event flags, `src/action_results.py` for result types.
- Web: `tensormewweb/src/utils/`.

**Tests:**
- All test modules go in `tests/`, flat. Prefer fakes over the ROM. The ROM is only needed by the smoke and live tests.

## Special Directories

**`states/`:**
- Purpose: Named checkpoints and their manifest.
- Generated: Mostly yes. `tools/make_state.py`, `tools/import_sav.py`, and `src/chain.py` write `.state` files and update `manifest.json`.
- Committed: Yes. The tracked `.state` files are the shipped checkpoints. `states/_autosave/` is gitignored.

**`logs/` (runtime, not in the tree):**
- Purpose: Decision, supervisor, and run logs; live status; plots.
- Generated: Yes, by `src/agent_loop.py`, `src/llm_supervisor.py`, `src/run_log.py`, `src/live_status.py`, `tools/play_live.py`.
- Committed: No (`.gitignore`).

**`runs/` (runtime, not in the tree):**
- Purpose: Episode logs and recording output. `src/config.py` `RUNS_DIR`.
- Generated: Yes, by `src/episode.py`, `src/recording.py`.
- Committed: No.

**`recordings/`:**
- Purpose: Shareable run captures with per-milestone `.state` files and `run.gif`.
- Generated: Yes, by `tools/run_hybrid.sh`, `tools/make_state.py`, and `play_live --record-dir`.
- Committed: Yes. `.gitignore` un-ignores `recordings/**/*.state`.

**`models/`:**
- Purpose: Shipped weights.
- Generated: By training (`tools/train_*.py`) into `logs/`, then copied here by `tools/sync_weights.py`.
- Committed: Yes.

**`rom/`:**
- Purpose: Game ROM and symbols. Required at runtime.
- Generated: No.
- Committed: Yes (tracked in git). `SPEC.md` pins the expected ROM SHA1, and `src/emulator.py` refuses to boot any other file.

**`tensormewweb/node_modules/` and `tensormewweb/dist/`:**
- Purpose: Installed deps and the production build.
- Generated: Yes, by `npm install` and `npm run build`.
- Committed: No (ignored by `node_modules/`, `dist/` in `.gitignore`).

**`.planning/codebase/`:**
- Purpose: Codebase analysis docs (this set) used by `/gsd:plan-phase` and `/gsd:execute-phase`.
- Generated: Yes, by the GSD codebase mapper.
- Committed: Decided by the orchestrator.

**Uncommitted in the current working tree:** `tools/pod_checkpoint.py`, `tests/test_pod_checkpoint.py`, and `.planning/PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`. These exist on disk but are not yet tracked.

---

*Structure analysis: 2026-11-05*
