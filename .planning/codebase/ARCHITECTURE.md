# Architecture

**Analysis Date:** 2026-11-05

## Pattern Overview

**Overall:** Layered, deterministic-first game agent. A Python package (`src/`) wraps a PyBoy Game Boy emulator and exposes a small set of deterministic actuation primitives (mover, macros). An LLM supervisor and learned policies (PPO, spatial-vision, GRU) sit above those primitives and only ever act by calling the same named tools. Thin CLI scripts in `tools/` and a root `app.py` compose the layers into runs. A separate Vite/React site in `tensormewweb/` is static and not connected to the Python code at runtime.

**Key Characteristics:**
- **Tools, not buttons.** The LLM and the policies never emit raw button presses to the game as a free-form action. The LLM picks one tool from `src/tools_schema.py` (`go_to`, `run_path`, `use_cut`, `mash_text`, `choose_move`, `switch_pokemon`, `use_pokeball`). The learned policies emit one of five directions or `a` (`src/rl_env.py` `ACTIONS`). Both execute through `src/mover.py` or `src/macros.py`.
- **Deterministic first.** Battle, dialogue, and tile-walking are handled without a model call. The LLM is invoked only when `src/triggers.py` `evaluate()` returns a trigger (BATTLE, DIALOGUE, CUT_BLOCKED, STUCK).
- **RAM is the source of truth.** All game state comes from `rom/pokemon_crystal.sym` symbols read through `src/ram_parser.py` `RamParser` into an immutable-ish `GameState`. The LLM is given `GameState.to_text()`, not screen pixels (the deep policy is the one exception and consumes the raw screen).
- **Protocol-based seams.** `src/emulator.py` defines a `Machine` Protocol and `src/ram_parser.py` defines a `MemoryBus` Protocol. The emulator is wrapped, not subclassed. Tests substitute fakes for these seams.
- **Checkpoint-driven progress.** Named save-states are the unit of progress. `src/states.py` keeps a hash-validated registry in `states/manifest.json`. Runs start from a checkpoint and save new ones on map change (`src/chain.py`).
- **Multiple parallel decision engines (see Concerns in Layers).** Trigger-driven segments (`src/segment.py`) and the live loop (`tools/play_live.py`) each implement their own per-step priority logic on top of the same primitives.

## Layers

**Substrate (emulator and RAM):**
- Purpose: Boot the ROM, verify it, step frames, read and write game memory, save and load states.
- Location: `src/emulator.py`, `src/config.py`, `src/ram_parser.py`, `src/collision.py`, `src/story_flags.py`, `src/options.py`, `src/game_data.py`
- Contains: `Emulator` (PyBoy wrapper with SHA1 ROM check and RTC lock), `Machine` and `MemoryBus` Protocols, `RamParser` / `GameState` (position, map, party, badges, flags, dialogue, battle mode), tile-collision tables, event-flag decoding, species/move/map name tables.
- Depends on: `pyboy`, `src/config.py` (ROM and SYM paths, SHA1 constant).
- Used by: every layer above.

**Actuation primitives (deterministic):**
- Purpose: Turn one intent into button presses and frame ticks, and report what happened.
- Location: `src/step.py` (`TileStepper`: one tile, turn-then-walk, waits out warps), `src/mover.py` (`Mover`: `go_to`, `run_path`, stuck counter, `walkable()`), `src/macros.py` (`Macros`: `mash_text`, `use_cut`, `choose_move`, `switch_pokemon`, `use_pokeball`, `run_away`, `use_battle_item`, `handle_whiteout`), `src/battle.py` (`tactical_battle_step`), `src/unstick.py` (`Unsticker`).
- Contains: Frame-counted button sequences, each returning an `ActionResult` (`src/action_results.py`: `ActionResult`, `MoveResult`, `PathResult`, `MacroResult`).
- Depends on: Substrate layer only.
- Used by: `src/tool_dispatch.py`, `src/segment.py`, `tools/play_live.py`, `src/episode.py` consumers.

**Navigation helpers (deterministic, map-specific):**
- Purpose: Choose the next direction for specific story segments.
- Location: `src/explore.py` (`Explorer`), `src/local_nav.py` (`LocalNav`), `src/falkner_nav.py` (Violet Gym path), `src/return_nav.py` (ROM-derived route to Elm's Lab, egg errand), `src/gym_prep.py` (farm and heal before Falkner), `src/nav_memory.py`, `src/frontier_buffer.py`
- Contains: BFS and A*-style grids built from ROM collision data, frontier selection, visited-tile tracking.
- Depends on: `src/mover.py` `walkable`, `src/return_nav.py` `_Rom` (reused by `falkner_nav.py` and `gym_prep.py`).
- Used by: `src/segment.py` (`_step_direction`), `tools/play_live.py` step priority, `src/rl_env.py` (`observe` consumes `LocalNav`).

**Trigger and decision (LLM on triggers):**
- Purpose: Decide when a model is needed and what it should do.
- Location: `src/triggers.py`, `src/llm_supervisor.py`, `src/tools_schema.py`, `src/supervisor_stubs.py`, `src/agent_loop.py`
- Contains:
  - `evaluate(state, stuck_count=...)` returns at most one `TriggerEvent` in priority order: BATTLE, then DIALOGUE (only when the mover is idle), then CUT_BLOCKED, then STUCK (only when no dialogue is open).
  - `Supervisor.decide()` builds the Anthropic `messages.create` call with `tools=anthropic_tools()` and `tool_choice={"type": "any"}`. It returns a `Decision` with the tool name, arguments, `reason`, and token and USD estimates.
  - `act()` in `src/agent_loop.py` glues read, trigger, decide, run tool, and log one step.
- Depends on: `src/ram_parser.py`, `src/collision.py`, `anthropic` (lazily imported), `pydantic` models in `src/tools_schema.py`.
- Used by: `src/segment.py`, `tools/play_live.py` (only when `ANTHROPIC_API_KEY` is set and the bottleneck counter trips).

**Tool dispatch:**
- Purpose: Map a tool name from the LLM or a policy onto a concrete handler.
- Location: `src/tool_dispatch.py` (`execution_registry`), `src/llm_supervisor.py` `Supervisor.run_tool`
- Contains: A dict from tool names (`go_to`, `run_path`, `use_cut`, `mash_text`, `choose_move`, `switch_pokemon`, `use_pokeball`) to bound methods of `Mover` and `Macros`. `Supervisor.run_tool` validates arguments through the pydantic models and strips `reason` before the call.
- Depends on: `src/mover.py`, `src/macros.py`, `src/tools_schema.py`.
- Used by: `src/agent_loop.py`, `tools/play_live.py`.

**Orchestration (segments, chains, episodes):**
- Purpose: Run a bounded stretch of play from a checkpoint and report the outcome.
- Location: `src/segment.py` (`run_segment`), `src/chain.py` (`run_chain`), `src/episode.py` (`EpisodeRunner`, button scripts, no model)
- Contains:
  - `run_segment` loops until the target map or tile is reached, a badge changes, a starting Cut tree is cleared, the step budget is spent, or the run is stuck. It returns a `SegmentResult` with `outcome` in {`advanced`, `budget`, `stuck`, `error`}. Each iteration either triggers the supervisor through `act()`, or takes a policy, explorer, or script step and logs it.
  - `run_chain` repeats segments. On each `advanced` outcome it captures `agent_map_<group>_<number>` through the state registry and continues from that checkpoint.
- Depends on: all lower layers; `src/rl_env.py` for `observe` and `progress_reward` when a policy is attached.
- Used by: `tools/run_segment.py`, `tools/run_chain.py`, `tools/run_episode.py`.

**Learned policies (RL):**
- Purpose: Pick the next direction from observations without a language model.
- Location: `src/rl_env.py` (`CrystalEnv`, `observe`, `progress_reward`, `navigation_potential`, reward shaping), `src/ppo.py` (`Policy`, `MlpPolicy`, `collect_rollout`, `ppo_update`, `load_policy`), `src/deep_policy.py` (`SpatialVisionPolicy`, torch-optional), `src/recurrent_policy.py` (numpy GRU `RecurrentPolicy`), `src/visual_metrics.py` (plotting)
- Contains: A 5-way discrete action space (`up`, `down`, `left`, `right`, `a`), an observation vector built from RAM and `LocalNav`, and event-flag multi-hot features. Weights are loaded by extension: `.npz` goes to `src/ppo.load_policy`, `.pt` goes to `SpatialVisionPolicy.load`.
- Depends on: `numpy`, optionally `torch`, `src/local_nav.py`, `src/story_flags.py`.
- Used by: `src/segment.py` (when `policy=` is passed), `tools/play_live.py`, training scripts `tools/train_ppo.py`, `tools/train_marathon.py`, `tools/train_deep_ppo.py`, and cloud wrappers `tools/cloud_train.py`, `tools/cloud_train_deep.py`.

**Persistence and observability:**
- Purpose: Save progress, record why each step happened, and expose it.
- Location: `src/states.py` (`StateRegistry`, `StateEntry`), `src/run_log.py` (`RunLog`, SQLite), `src/recording.py` (`InputRecorder`, `replay`), `src/live_status.py` (`write_status`, milestone tracking), `src/terminal_monitor.py`, `app.py` (Streamlit thought log)
- Contains: `states/manifest.json` (checkpoints with ROM and state SHA1s), `logs/runs.db` (SQLite `runs` and `steps` tables), `logs/decisions.jsonl` (one JSON row per decision), `logs/supervisor.jsonl` (token and cost rows), `logs/live_status/*.json`, `runs/` (recordings and episode logs).
- Depends on: `src/config.py` paths; `src/emulator.py` `InputEvent` for recordings.
- Used by: `src/segment.py`, `src/chain.py`, `src/agent_loop.py`, `tools/play_live.py`, `tools/runs.py`, `tools/states.py`, `tools/live_status.py`, `app.py`.

**Entry points (CLI and UI):**
- Purpose: Start a run, a training job, or a dashboard.
- Location: `tools/` (see Entry Points below), `crystal_smoke.py`, `app.py`, `run_live.sh`, `run_4_live.sh`, `scripts/*.sh`, `tools/run_hybrid.sh`, `tools/run_hybrid.ps1`
- Depends on: any layer.

**Web site (separate):**
- Purpose: Public demo and research site. Shows pre-recorded checkpoint telemetry.
- Location: `tensormewweb/` (Vite 6, React 19, `lucide-react`)
- Contains: `tensormewweb/src/App.jsx` tab switcher (landing, Pokédex, blog, founders). `tensormewweb/src/components/LiveAgentTab.jsx` reads `CHECKPOINTS_STREAM` from `tensormewweb/src/data/agentTelemetryData.js`.
- Depends on: nothing from the Python code. It has no `fetch` or WebSocket calls. Its data is static JS plus assets under `tensormewweb/public/`.
- Used by: deployed as a static site (`tensormewweb/vite.config.js` dev server on port 5173).

## Data Flow

**Trigger-driven segment (LLM on triggers):**

1. `tools/run_segment.py` (or `tools/run_chain.py`) builds `Emulator` (`src/emulator.py`), `RamParser` (`src/ram_parser.py`), `Supervisor` (`src/llm_supervisor.py`), and optionally `RunLog` (`src/run_log.py`).
2. `src/segment.py` `run_segment` calls `reader.parse(frame)` to get a `GameState`.
3. `evaluate(state, stuck_count=...)` in `src/triggers.py` returns a trigger or `None`.
4. If `None`, the segment takes a policy step (`policy.act(observe(...))`), an explorer step (`Explorer` / `LocalNav`), or a script step through `Mover.run_path([direction])`. It logs via `_log_script`.
5. If a trigger fires, `src/agent_loop.py` `act()` calls `Supervisor.decide(state.to_text(), trigger)`, which makes one `messages.create` call with the tool schema.
6. `Supervisor.run_tool(name, args)` validates arguments with the pydantic model, strips `reason`, and calls the handler from `src/tool_dispatch.py`.
7. `act()` appends one row to `logs/decisions.jsonl` through `append_decision`, and to `RunLog` if provided.
8. The loop re-reads state and returns `advanced`, `stuck`, or `budget`. Streak limits (25 for DIALOGUE or BATTLE, 3 otherwise) convert repeated triggers into `stuck`.
9. `run_chain` (`src/chain.py`) captures a checkpoint on `advanced` and repeats.

**Live play (policy with inline reflexes):**

1. `tools/play_live.py` `main` loads a state from `states/manifest.json` through `StateRegistry`, then `Emulator`, `RamParser`, `Macros`, `Mover`, `Unsticker`, and optionally a policy (`.pt` or `.npz`).
2. Each iteration reads `GameState` and runs an inline priority chain:
   1. Battle: `tactical_battle_step` (`src/battle.py`) with `Macros`.
   2. Dialogue: `dialogue_open(state)` then `macros.mash_text`.
   3. Same-tile recovery: `Unsticker.act` after 6 repeats of the same tile.
   4. Blocked tile or trigger handling in the play loop (Cut, NPC, lateral detour).
   5. Overworld policy step, with masks and compass overrides. Story-specific navigators take precedence when they return a direction: Mr. Pokémon's house exit, Nurse Joy healing, `return_nav.egg_return_direction`, prep farming (`gym_prep.prep_direction`), and Falkner path (`falkner_nav.falkner_nav_direction`).
3. Each step writes `logs/live_status/<tag>.json` through `src/live_status.py` and auto-saves a named state on major milestones.
4. The LLM is only consulted for the bottleneck case (`blocked_count >= 5`) and only if `ANTHROPIC_API_KEY` is set (`tools/play_live.py` around line 155).

**Emulation frame loop (shared by all paths):**

1. `Emulator.tick(n, render=...)` advances PyBoy by `n` frames and increments `frame`.
2. `Emulator.press(button, frames)` and `TileStepper.step` queue inputs and tick. Input listeners (`Emulator.add_listener`) receive `InputEvent` objects, which `src/recording.py` `InputRecorder` uses to capture replayable logs.
3. `Emulator.save_state` and `load_state` round-trip through `src/states.py` `StateRegistry.save` and `load_bytes`, which validate the ROM SHA1 and the state SHA1 against `states/manifest.json`.

**State Management:**
- Game state is re-read from RAM before every decision. No cached game model is shared between steps other than counters (`Mover.stuck`, `Unsticker.count`, `rl_bumps`, `streak`, visited-tile sets).
- Persistent state is files only: `states/manifest.json`, `states/*.state`, `logs/runs.db`, `logs/*.jsonl`, `models/*.pt|*.npz`.

## Key Abstractions

**`Machine` (Protocol):**
- Purpose: The minimum surface an actuator needs from an emulator: `frame`, `tick`, `press`, `hold`, `release`, `save_state`, `load_state`, `read`, `symbol`, `screen_image`, `add_listener`.
- Examples: `src/emulator.py` (`Machine`, `Emulator`), `src/mover.py` and `src/macros.py` constructors.
- Pattern: Structural typing. Tests pass fakes.

**`GameState` and `RamParser`:**
- Purpose: One consistent snapshot of the game, with derived helpers (`in_battle`, `in_dialogue`, `party_can_cut`, `to_text`, `to_dict`).
- Examples: `src/ram_parser.py` (`GameState`, `RamParser.parse`, `SYMBOLS` table), `src/ram_parser.py` `dialogue_open`.
- Pattern: Parse-once, read-only snapshot. Symbols resolve from `rom/pokemon_crystal.sym`, not hard-coded addresses. Banked WRAM reads use `pyboy.memory[bank, addr]`.

**`ActionResult` family:**
- Purpose: Uniform return from every actuator: `ok`, `message`, `frames`, plus `MoveResult`, `PathResult`, `MacroResult`.
- Examples: `src/action_results.py`, returned by `src/mover.py` and `src/macros.py` and by stubs in `src/supervisor_stubs.py`.

**`TriggerEvent` and `TriggerContext`:**
- Purpose: `TriggerEvent` (in `src/triggers.py`) is the deterministic reason to ask for help. `TriggerContext` (in `src/llm_supervisor.py`) is the wire form passed to `Supervisor.decide`. `Supervisor.decide` accepts either.
- Pattern: Priority-ordered, first match wins.

**`Decision`:**
- Purpose: The result of one model call: tool name, arguments, `reason`, model id, token counts, and estimated USD. It is what gets logged.
- Examples: `src/llm_supervisor.py` (`Decision`), `src/agent_loop.py` (`append_decision`).

**`StateRegistry` / `StateEntry`:**
- Purpose: Named, hash-validated checkpoints. Names must match `^[a-z0-9_]+$`.
- Examples: `src/states.py`, `states/manifest.json`, `tools/states.py`.

**Policy objects (duck-typed):**
- Purpose: Anything with `act(obs) -> (action, logp)`, optional `probs(obs)`, optional `value(obs)`, optional `probs_recurrent`.
- Examples: `src/ppo.py` `Policy` and `MlpPolicy`, `src/deep_policy.py` `SpatialVisionPolicy`, `src/recurrent_policy.py` `RecurrentPolicy`.
- Pattern: `load_policy` and `SpatialVisionPolicy.load` dispatch by file type. `run_segment` checks `hasattr` for `probs` and `value`.

**`SegmentResult` and `ChainResult`:**
- Purpose: Outcome records for a bounded run and a sequence of runs.
- Examples: `src/segment.py`, `src/chain.py`.

## Entry Points

**`tools/play_live.py` (main live player):**
- Location: `tools/play_live.py`, `main()`. Also reached through `run_live.sh`, `run_4_live.sh`, `scripts/run_demo_falkner.sh`, `tools/run_hybrid.sh`.
- Triggers: Manual CLI run. Flags include `--state`, `--policy`, `--post-egg-policy`, `--headless`, `--speed`, `--watch`, `--record-dir`, `--balls`.
- Responsibilities: Full inline control loop with optional policy and optional LLM bottleneck help. Writes live status and recordings.

**`tools/run_segment.py` (bounded trigger-driven run):**
- Location: `tools/run_segment.py`
- Triggers: Manual CLI. README and the `EMPTY_HINT` in `app.py` both use `python -m tools.run_segment --state ilex_src`.
- Responsibilities: Build the stack and call `src/segment.py` `run_segment`.

**`tools/run_chain.py` (checkpoint chaining):**
- Location: `tools/run_chain.py`
- Responsibilities: Repeats `run_segment` and saves a new checkpoint on each map change.

**`tools/run_episode.py` (scripted play, no model):**
- Location: `tools/run_episode.py`, `src/episode.py`
- Responsibilities: Plays a named snapshot forward with a `ButtonScript` or `RandomScript`, and writes a plain log under `runs/`.

**Training entry points:**
- `tools/train_ppo.py` (MLP PPO, `.npz` output), `tools/train_marathon.py` (wall-clock headless PPO), `tools/train_deep_ppo.py` (spatial-vision PPO, `.pt` output, GPU), `tools/cloud_train.py` and `tools/cloud_train_deep.py` (sync code to a remote pod and run the above), `tools/sync_weights.py` (copy trained weights to `models/`).

**`app.py` (thought-log dashboard):**
- Location: `app.py`
- Triggers: `streamlit run app.py`
- Responsibilities: Read `logs/decisions.jsonl` (default path, relative to the current working directory) and render steps, cost, tokens, badges, and reward or entropy curves. It never touches the emulator.

**`crystal_smoke.py` (boot check):**
- Location: `crystal_smoke.py`
- Responsibilities: Boot PyBoy, resolve symbols, read RAM, walk a few tiles, save a state. It has its own minimal RAM reader and does not use `src/ram_parser.py`.

**`tensormewweb` dev server:**
- Location: `tensormewweb/src/main.jsx` mounts `App.jsx`. Dev via `npm run dev` in `tensormewweb/`.

## Error Handling

**Strategy:** Fail soft in the decision layer, fail loud at the boundaries. The LLM path never raises into the game loop. Tool and macro failures return `ok=False` results. ROM and state mismatches raise.

**Patterns:**
- `Supervisor.decide` wraps the whole model call in `try/except Exception` and returns an offline fallback `Decision` (model `"offline_fallback"`, zero cost). The fallback picks `choose_move` for BATTLE, `run_path` right for CUT, `mash_text` for DIALOGUE, otherwise `mash_text`. Any API error, missing key, or malformed tool call silently takes this path. The only signal is the logged model name.
- `Supervisor.run_tool` returns `ActionResult(ok=False, message="no handler ...")` for unknown tools rather than raising.
- `run_segment` catches all exceptions at the top level and returns `outcome="error"` with `error` text, which `RunLog` stores.
- `Emulator.__init__` raises `RomMismatchError` when the ROM SHA1 differs from `src/config.py` `ROM_SHA1`. `StateRegistry` validates state SHA1s and raises `StateNotFound` (a `KeyError`) for unknown names.
- `RamParser` construction raises `SymbolError` when required symbols are missing from the `.sym` file (`src/ram_parser.py`).
- `tools/play_live.py` falls back from a missing policy path to `models/<basename>` when the given path does not exist.

## Cross-Cutting Concerns

**Logging:** Structured JSONL and SQLite, not the `logging` module. `src/agent_loop.py` `append_decision` writes `logs/decisions.jsonl`. `src/llm_supervisor.py` writes `logs/supervisor.jsonl`. `src/run_log.py` writes `logs/runs.db`. Console output in `tools/` uses `print` with `flush=True`.

**Validation:** Tool arguments are validated by pydantic models in `src/tools_schema.py` (`GoToArgs`, `RunPathArgs` with direction literals, `MashTextArgs` with `max_frames` 1 to 10000, `ChooseMoveArgs` slot 1 to 4, `SwitchPokemonArgs` slot 1 to 6). Every tool requires a non-empty `reason` up to 400 characters. Config and ROM integrity are checked by SHA1 in `src/config.py`, `src/emulator.py`, `src/states.py`.

**Authentication:** Only the Anthropic API key, read from the environment by the `anthropic` SDK (`anthropic.Anthropic()` in `src/llm_supervisor.py`). No other credentials are used. The web site has no auth.

**Configuration:** Environment variables. `POKEMIND_MODEL` (default `claude-haiku-4-5`), `POKEMIND_HARD_MODEL`, `POKEMIND_INPUT_USD_PER_MTOK` and `POKEMIND_OUTPUT_USD_PER_MTOK` (cost estimation), `PYBOY_VISUAL`, `PYBOY_SOUND`, `PYBOY_SPEED`, `ANTHROPIC_API_KEY`. Paths come from `src/config.py` (ROOT-relative) except for the decision and supervisor log paths and `app.py`, which use CWD-relative `logs/...` defaults.

**Coupling and Duplication Notes:**
- Two decision engines exist. `src/segment.py` (used by `run_segment` and `run_chain`) and the inline chain in `tools/play_live.py` implement overlapping priorities (battle, dialogue, cut, stuck, policy). New behavior that should apply to both needs to be added in both places, or moved into a shared module.
- `src/supervisor_stubs.py` is the default registry for `Supervisor` (`Supervisor.__init__` copies `STUB_REGISTRY` when no `registry` is passed). Production callers must pass `execution_registry(...)` from `src/tool_dispatch.py`. `tools/play_live.py` does this; a caller that forgets gets no-op results.
- `src/supervisor_stubs.py` docstring still describes the stubs as temporary. `src/tool_dispatch.py` now supplies the real handlers.
- `src/llm_supervisor.py` `SYSTEM_PROMPT` encodes Route 29 and Cherrygrove coordinates and says "You do NOT have HM01 Cut". The tool list still includes `use_cut`. Route knowledge lives in a prompt string, not in the navigation modules.

---

*Architecture analysis: 2026-11-05*
