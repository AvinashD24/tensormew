# Codebase Concerns

**Analysis Date:** 2026-11-05

## Tech Debt

**Battle logic hardcoded to Falkner's types:**
- Issue: `tactical_battle_step` and `pick_best_move` default `enemy_types` to `(2, 0)` (Flying/Normal). The enemy's real types are never read from RAM, and `play_live` never passes them in, so every battle is scored as if the opponent were Falkner's Pidgey/Hoothoot pair.
- Files: `src/battle.py` (lines ~78-92 `score_move`/`pick_best_move`, ~212 `tactical_battle_step`), `tools/play_live.py` (battle branch, ~line 285 call `tactical_battle_step(state, macros, grind=...)`)
- Impact: Move choice is wrong for any non-Flying/non-Normal opponent. The `TYPE_CHART` has only about 17 entries (`src/battle.py:8-60`), and there is no STAB, damage calculation, status, or switching logic beyond heuristics.
- Fix approach: Read enemy species/types from RAM (the Crystal enemy-mon struct), pass them into `tactical_battle_step`, and complete the Gen 2 type chart. Add a damage estimate before relying on `score_move`.

**Hardcoded map and coordinate logic in the policy loop:**
- Issue: `tools/play_live.py` (~lines 530-600) is a long `elif` chain of hardcoded `map_id` and `(x, y)` checks for the Pokémon Center heal, Mr. Pokémon's house exit, the Route 29 eastbound path, Elm's lab desk, Falkner's gym, and building exits. `src/rl_env.py` `navigation_potential` (~lines 261-408) does the same with per-map constants (for example, `base_pc = 320.0`, `380.0 + ...` for Violet PC).
- Files: `tools/play_live.py`, `src/rl_env.py`, `src/falkner_nav.py` (hardcoded step list in its module docstring and `MAP_*` constants at lines 19-29), `src/return_nav.py`, `src/gym_prep.py`
- Impact: The "learned" policy is largely overridden by scripts in the live path. Adding a new region means editing a new `elif` branch and the reward shaping together. Coordinates depend on exact ROM map IDs and are fragile to any change in warp layout or save state.
- Fix approach: Move per-map constants into one data file keyed by map ID (for example `data/knowledge/`), and replace the `elif` ladder with a table of scripted stages. Keep a per-stage counter so overrides can be attributed.

**Two divergent decision-log formats:**
- Issue: `src/llm_supervisor.py` `Supervisor._log` writes `logs/supervisor.jsonl` with `model`, `input_tokens`, `output_tokens`, `tool_name`, `est_usd`. `app.py` (the Streamlit thought log) and `src/agent_loop.py`, `src/segment.py`, `src/chain.py`, `tools/run_segment.py` read or write `logs/decisions.jsonl`, which expects `reason`, `map`, `pos`, `badges`, `actor`, `trigger`, `tool`, `ok`, `est_usd`. The live loop in `tools/play_live.py` never writes `decisions.jsonl`.
- Files: `src/llm_supervisor.py` (default `log_path` line ~86, `_log` at line 188), `app.py` (line 9), `src/agent_loop.py` (line 13), `tools/play_live.py`
- Impact: The dashboard's "Running cost" and token totals would show zero for the live LLM director path, even though calls are billed. The cost-per-milestone reporting described in `.planning/PROJECT.md` cannot be produced from the current logs.
- Fix approach: Have one writer for LLM calls that records the full schema, and point `app.py` at it. Add a schema test.

**Silent exception swallowing in hot paths:**
- Issue: About two dozen bare `except Exception: pass` blocks hide real errors. Examples: `src/emulator.py:89` (`_lock_rtc` failure is ignored, so the RTC lock can silently not apply), `tools/play_live.py:57` (`equip_balls` swallows RAM write failures), `tools/play_live.py:327`, `tools/play_live.py:513`, `tools/play_live.py:729`, `src/macros.py:46`, `:69`, `:168`, `:217`, `src/visual_metrics.py:521-600`, `src/terminal_monitor.py:234`.
- Files: as listed above
- Impact: Failures appear as wrong behavior several steps later with no log line. This is why failures such as "RTC not locked" or "party read failed" are hard to diagnose.
- Fix approach: Narrow exception types and log through the existing `TerminalMonitor` or `run_log`. At minimum, log once per distinct exception on the RTC lock and RAM-read paths.

**Dead or unwired code:**
- Issue: `src/explore.py`, `src/segment.py`, `src/chain.py`, `src/nav_memory.py` (older checkpoint-chain path), `src/frontier_buffer.py` and `src/recurrent_policy.py` (not used by the trainers), `src/saycan.py` (not wired into the loop), `src/supervisor_stubs.py` (default registry in `Supervisor`, so a `Supervisor()` built without a registry only returns stub results).
- Impact: Readers and planners cannot tell which path is live. `Supervisor.__init__` silently defaults to `STUB_REGISTRY` (`src/llm_supervisor.py:92`).
- Fix approach: Mark or remove unused modules. Make the registry a required argument.

**Stale LLM system prompt:**
- Issue: `SYSTEM_PROMPT` in `src/llm_supervisor.py` (lines ~23-35) hardcodes a Route 29 walkthrough and says "NEVER call use_cut" and "You cannot cut ANY trees before the first gym". It does not cover anything after Falkner.
- Files: `src/llm_supervisor.py`
- Impact: Any LLM call after the Zephyr Badge is steered by obsolete instructions. The offline fallback in `_offline_fallback` also hardcodes `run_path(["right"])` for CUT triggers.
- Fix approach: Generate the prompt from the current progress (`src/story_flags.py`, `src/live_status.py`) instead of a fixed string.

**Committed binaries and ROM in the repository:**
- Issue: `rom/pokemon_crystal.gbc` (2 MB), `rom/pokemon_crystal.sym`, two ~53 MB PyTorch checkpoints (`models/deep_policy.pt`, `models/deep_policy_baseline_351cb55.pt`), and `states/*.state` are tracked in git. There is no Git LFS (`.gitattributes` only sets `*.sh` line endings).
- Files: `rom/`, `models/`, `states/`, `.gitattributes`, `.gitignore`
- Impact: Each retrained checkpoint adds roughly 50 MB of history. The repository is public (`origin` is on GitHub), and the Pokémon Crystal ROM is copyrighted; redistributing it may not be permitted. `.gitignore` lists `*.state` but these files are tracked anyway, so the ignore rule gives false comfort.
- Fix approach: Remove the ROM from version control (keep the SHA-1 check in `src/config.py` and document where the user supplies it). Move checkpoints to Git LFS or to release artifacts. Track `models/` through a manifest with hashes.

**Dependency manifest incomplete:**
- Issue: `requirements.txt` has no `torch`, but `src/deep_policy.py`, `tools/train_deep_ppo.py`, `tools/pod_checkpoint.py` and the deep policy path in `tools/play_live.py` need it. `httpx2==2.13.1` and `httpcore2==2.13.1` are listed in place of the standard `httpx`/`httpcore` names. These two names need to be checked against PyPI before anyone installs from this file.
- Files: `requirements.txt`
- Impact: A fresh install cannot run the deep policy. The unusual package names are a supply-chain risk if they were typed by mistake.
- Fix approach: Add an explicit torch pin (or an extras group), verify `httpx2`/`httpcore2` and replace them with the intended packages if needed.

## Known Bugs

**Pod checkpoint test fails on Windows path separators:**
- Symptoms: `tests/test_pod_checkpoint.py::test_load_succeeds_without_a_snapshot` fails with `['logs\\ppo_policy.npz'] != ['logs/ppo_policy.npz']`.
- Files: `tests/test_pod_checkpoint.py` (line ~34), `tools/pod_checkpoint.py`
- Trigger: Run the unit tests on Windows (this checkout is Windows 11).
- Workaround: None yet. Compare with `Path(...)` or normalize separators in the test.

**Windows test failures from temp-file handling and missing torch:**
- Symptoms (from `python -m unittest discover -s tests -t .` on Python 3.14, Windows): 252 tests run, 1 failure, 5 errors, 1 skipped.
  - `tests/test_ppo.py::MlpPolicyTests::test_mlp_policy_act_and_learn`: `PermissionError` when `np.savez` reopens a `NamedTemporaryFile` on Windows.
  - `tests/test_deep_policy.py::DeepPolicyTests::test_load_feedforward_checkpoint_into_recurrent` and `test_save_and_load_recurrent_policy`: errors from `torch.save` writing to a temp file that is still open (`open file failed with error code: 32`).
  - `tests/test_improved_training.py::ImprovedTrainingTests::test_load_17_dim_checkpoint_into_21_dim_policy`: error in the same torch save/load path.
  - `tests/test_emulator.py::RomCheckTests::test_wrong_bytes_rejected_before_boot`: error on the ROM check path.
  - `tests/test_pod_checkpoint.py`: see above.
- Files: as listed above
- Trigger: Run on Windows. The NamedTemporaryFile issue is a test-code problem, not a library bug.
- Workaround: Use `tempfile.TemporaryDirectory()` and close handles before reopening.

**Supervisor swallows API errors and counts them as successful fallbacks:**
- Symptoms: `Supervisor.decide` catches every `Exception` and returns `_offline_fallback`, logged with `model="offline_fallback"` and `est_usd=0.0`. An API timeout, a bad key, or a malformed tool call looks identical to a deliberate fallback. If the API call succeeds but the response has no tool call, tokens are billed but `est_usd` is recorded as 0.
- Files: `src/llm_supervisor.py` (`decide`, `_offline_fallback`)
- Trigger: Any API or parsing error, or missing `ANTHROPIC_API_KEY` (the `anthropic.Anthropic()` constructor raises).
- Workaround: None. Check `logs/supervisor.jsonl` for `offline_fallback` rows.

**RTC lock is best-effort and unverified:**
- Symptoms: `Emulator._lock_rtc` (`src/emulator.py:86-89`) swallows any exception. If `rtc_lock_experimental` fails, time-of-day wild encounter tables change mid-run (`pokemon_agent.md` documents this as a reproducibility issue), and nothing reports it.
- Files: `src/emulator.py`, `src/recording.py:46` (`"rtc_locked": True` is written into recording metadata regardless of result)
- Trigger: Any pyboy version where `rtc_lock_experimental` is missing or raises.
- Workaround: None. Verify the lock by reading the RTC registers after boot.

**Cheat applied to the player's bag in live play:**
- Symptoms: `equip_balls` writes ten Poké Balls directly into WRAM (`wNumBalls`, `wBalls`) when the bag is empty.
- Files: `tools/play_live.py` (lines 48-58), default `--balls 10` (line ~101)
- Trigger: Any live run where the bag has no balls.
- Workaround: Pass a flag or disable it. This is a RAM edit, so any run that reports itself as "agent-only" is not comparable to an unassisted run. The audit in `research/analysis/codebase-audit.md` already flags this.

**Live loop can overwrite a tracked checkpoint state:**
- Symptoms: `tools/play_live.py` (lines ~733-736) saves `states/mystery_egg_received.state` when the egg is first obtained and the file does not exist. That path is inside the tracked `states/` directory, and `states/manifest.json` is not updated by this code.
- Files: `tools/play_live.py`, `states/manifest.json`
- Trigger: A run that reaches the egg event on a checkout where the file was removed.
- Workaround: Pass a different `--record-dir` or move the file. The manifest then drifts from disk.

**Documentation contradicts the recorded progress:**
- Symptoms: `HANDOFF.md` says no badge has been earned. The recordings in `recordings/falkner_full_story/` (21 screenshots, `falkner_beaten.state`, `run.gif`) and `states/manifest.json` show the Zephyr Badge was earned. `HANDOFF.md` also cites `logs/` data that is not in the repository.
- Files: `HANDOFF.md`, `recordings/falkner_full_story/`, `states/manifest.json`, `docs/HYBRID_FALKNER_RUN.md`
- Trigger: Any reader using HANDOFF.md to plan the next phase.
- Workaround: Trust the recordings and `docs/HYBRID_FALKNER_RUN.md`.

## Security Considerations

**`StrictHostKeyChecking=no` on all remote-pod SSH and rsync calls:**
- Risk: The SSH host key is never verified, so a man-in-the-middle could impersonate the RunPod host. The pod receives the codebase via rsync and returns model checkpoints that are then loaded with `torch.load`, so a spoofed host could feed a malicious pickle.
- Files: `tools/cloud_train.py` (lines ~39, 72), `tools/cloud_train_deep.py` (lines ~45, 96), `tools/pod_checkpoint.py` (lines ~90, 109)
- Current mitigation: The key file is read from `~/.ssh/id_runpod` by default. Nothing else.
- Recommendations: Pin the pod's host key in a `known_hosts` file, drop `StrictHostKeyChecking=no`, and prefer `-o StrictHostKeyChecking=accept-new` only for first contact.

**Hardcoded pod host and port in source:**
- Risk: The public IP address and SSH port of a RunPod instance are committed as defaults in two training scripts. Pods are ephemeral, so the value goes stale, and anyone with the repo knows where to probe.
- Files: `tools/cloud_train.py` (lines 14-15), `tools/cloud_train_deep.py` (lines 14-15)
- Current mitigation: Both are overridable with `--host` and `--port`.
- Recommendations: Remove the defaults and require `--host`/`--port` or read them from an environment variable that is not committed.

**Unpickling checkpoints without `weights_only`:**
- Risk: `torch.load` is called without `weights_only=True` in `src/deep_policy.py` (line ~374, `SpatialVisionPolicy.load`) and `tools/pod_checkpoint.py` (line ~25, `load_checkpoint`). Checkpoints pulled from the pod are arbitrary pickles unless the user relies on the library default. Some older torch versions default to full unpickling.
- Files: `src/deep_policy.py`, `tools/pod_checkpoint.py`
- Current mitigation: `np.load` in `src/ppo.py` and the other NumPy loaders use the default `allow_pickle=False`, which is safe.
- Recommendations: Pass `weights_only=True` explicitly. Checkpoint dicts here only hold tensors and scalars, so this should load without changes.

**Public Streamlit and Vite dev servers:**
- Risk: `tensormewweb/vite.config.js` sets `server.host: true`, which exposes the dev server on all network interfaces. `app.py` is a Streamlit app that reads `logs/decisions.jsonl` with a user-editable path (`st.text_input`), so it can read any JSONL file the process can access.
- Files: `tensormewweb/vite.config.js`, `app.py` (line 132)
- Current mitigation: Local-only use is assumed.
- Recommendations: Use `host: false` (the default) unless LAN access is needed. Restrict the Streamlit path to the `logs/` directory.

**Secrets handling is by convention only:**
- Risk: `tools/play_live.py` calls `load_dotenv()` at startup, and `src/llm_supervisor.py` reads `ANTHROPIC_API_KEY` from the environment through the `anthropic` client. `.env` is in `.gitignore`. No code path logs the key, but nothing enforces that either.
- Files: `tools/play_live.py` (lines 10-13), `src/llm_supervisor.py`, `.gitignore`
- Current mitigation: `.env` is ignored. This document records only the variable name.
- Recommendations: Keep the key out of logs and tracebacks, and add a pre-commit check for `sk-ant-` strings.

## Performance Bottlenecks

**Per-step navigation potential evaluated for every candidate direction:**
- Problem: In `tools/play_live.py` (section D, ~lines 490-515), each overworld step builds a copied `state` object for each of four directions and calls `navigation_potential` four times, with a `try/except` that discards errors.
- Files: `tools/play_live.py`, `src/rl_env.py` (`navigation_potential`, `progress_reward`)
- Cause: Shaping is recomputed from scratch, and the `type(state)(**{**state.__dict__, ...})` copy is costly.
- Improvement path: Compute the potential for the four neighbor tiles from cached grid data, and only reconstruct the state once per step.

**Torch model in the live loop:**
- Problem: `SpatialVisionPolicy` (about 13.2M parameters including a GRU, `src/deep_policy.py`) is used per live step with a 144x160 RGB frame copy (`emu._pb.screen.ndarray[:, :, :3].copy()`, `tools/play_live.py` ~line 437).
- Files: `src/deep_policy.py`, `tools/play_live.py`
- Cause: One CPU forward pass per emulated step; the `--speed` default is 4 (`tools/play_live.py:98`).
- Improvement path: Run inference on GPU when available and skip inference on frames where only dialogue or battle text is shown (already partly done for battle and dialogue branches).

**Training throughput far below the spec:**
- Problem: `HANDOFF.md` records 4 emulators in training. `.planning/ROADMAP.md` and `pokemon_agent.md` specify 64-128 workers and 25,000 aggregate FPS. No vectorized shared-memory runner exists (`pokemon_agent.md` references `vec_env_shm.py`, which is not in `src/`).
- Files: `src/rl_env.py`, `tools/train_ppo.py`, `tools/train_deep_ppo.py`
- Cause: Only multi-process Python workers are implemented.
- Improvement path: Implement the shared-memory vec env or use the existing `tools/cloud_train_deep.py` path with the planned worker count.

## Fragile Areas

**Hardcoded WRAM addresses and species IDs:**
- Files: `src/macros.py` (lines 165, 178, 185, 214 read `0xCFA9`/`0xcfa9` directly for the party cursor; the trade flow hardcodes species `69` at line 55), `src/battle.py`, `src/options.py` (which says options are "resolved by symbol, not by address", but `macros.py` does not follow that rule)
- Why fragile: A change to the ROM build or to `rom/pokemon_crystal.sym` breaks these reads silently. The `except Exception: return 0` in `_party_cursor` makes that failure look like a closed menu.
- Safe modification: Look up addresses with `emu.symbol(...)` as `options.py` and `equip_balls` do. Add a test that asserts each symbol resolves in `rom/pokemon_crystal.sym`.
- Test coverage: `tests/test_macros.py` and `tests/test_emulator.py` do not verify addresses against the symbol file.

**Story flags and milestone detection:**
- Files: `src/story_flags.py`, `src/live_status.py` (`MILESTONES`), `src/triggers.py`
- Why fragile: Milestones depend on exact event-flag bits and on `has_event` string names. `src/ram_parser.py` reads these bit flags, and `src/rl_env.py` and `tools/play_live.py` refer to them by name in several places, so one renamed flag breaks several branches.
- Safe modification: Keep flag names in one module, and add a test that checks each name exists in the symbol file.
- Test coverage: Unit tests exist, but there are no integration checks against the real ROM symbols.

**Save-state dependence:**
- Files: `states/manifest.json`, `states/*.state`, `recordings/*/`, `tools/play_live.py` (`--state` default `route29_entry`)
- Why fragile: Many tests and runs start from a specific `.state` file. Each state encodes the RAM snapshot, the RTC, and the options at capture time. `states/manifest.json` lists 9 entries, but the milestone states in `recordings/` are not registered in it.
- Safe modification: Always load states through `StateRegistry` and verify `state_sha1` and `rom_sha1` (the manifest stores both).
- Test coverage: Tests that need a state skip or fail when the file is missing.

**Unstick and blocked-count reflexes:**
- Files: `src/unstick.py` (`count` and `CYCLE`), `tools/play_live.py` (`blocked_count` branch ~lines 380-520)
- Why fragile: Escalation depends on a bump counter. Long NPC dialogue, a moving NPC, or a wrong `walkable` tile can push the agent into a cycle of detours that the counter does not catch.
- Safe modification: Change thresholds only together with `tests/test_unstick.py`, and rerun the Falkner route tests.

**Single-person LLM director path:**
- Files: `tools/play_live.py` (LLM director branch, ~lines 390-435), `src/llm_supervisor.py`
- Why fragile: The branch only runs at `blocked_count >= 5`, so it rarely fires. When it does, it can call `use_cut` (vetoed when `party_can_cut` is false) and `run_tool` with LLM-provided arguments.
- Safe modification: Keep the veto and offline fallback. Add a test for each tool name the LLM can return.

## Scaling Limits

**Single-machine emulation:**
- Current capacity: `tools/play_live.py` runs one emulator. Training scripts default to 4 (`HANDOFF.md`) or 12-16 workers on a cloud pod (`tools/cloud_train_deep.py --envs 16`).
- Limit: Each PyBoy process holds full emulator state, and the training loop writes checkpoints to `logs/` on every update. Disk and RAM grow with worker count.
- Scaling path: Use the shared-memory vec env described in `pokemon_agent.md`, and move checkpoint writes to a background thread.

**Recording size:**
- Current capacity: `recordings/` is about 29 MB and `models/` about 102 MB (measured on this checkout).
- Limit: Git history grows with each committed GIF, state, and checkpoint.
- Scaling path: See the committed-binaries item under Tech Debt.

## Dependencies at Risk

**`httpx2` and `httpcore2`:**
- Risk: Non-standard names in `requirements.txt` (`httpx2==2.13.1`, `httpcore2==2.13.1`). The standard packages are `httpx` and `httpcore`, which the `anthropic` client depends on.
- Impact: If these are typos or unofficial packages, installing the requirements could pull unintended code.
- Migration plan: Verify on PyPI, then replace with the intended packages or remove them if `anthropic` already pulls in `httpx`.

**`torch` not declared:**
- Risk: Required by `src/deep_policy.py` and the trainers, but absent from `requirements.txt`.
- Impact: The deep policy and `tools/pod_checkpoint.py` fail on a clean install. `tests/test_deep_policy.py` and `tests/test_improved_training.py` error when torch is missing.
- Migration plan: Add a pinned `torch` line, or an optional `requirements-gpu.txt`.

**PyBoy version pin:**
- Risk: `pyboy==2.8.1` is pinned, and the emulator code uses `rtc_lock_experimental` (`src/emulator.py:88`). The method name is marked experimental.
- Impact: A PyBoy upgrade could silently turn the RTC lock into a no-op (see Known Bugs).
- Migration plan: Test the RTC lock against each new PyBoy release, and fail loudly if it is missing.

## Missing Critical Features

**Progression beyond the Zephyr Badge:**
- Problem: No badge other than Zephyr is earned. No HM other than Cut is handled, and `use_cut` is untested on a real tree (`research/analysis/codebase-audit.md`). Surf, Strength, Flash, Whirlpool, and Waterfall are not implemented, and no water collision exists in `src/collision.py`.
- Blocks: Azalea, Ilex Forest, Bugsy, and every later gym. The project's stated goal of a full run cannot proceed.

**Progression planner:**
- Problem: `src/story_flags.py` covers about 40 flags, none for badges after Falkner, the Elite Four, or Red.
- Blocks: Any automatic route through the game without per-map scripting.

**Party building and item logistics:**
- Problem: No Poké Mart, PC, or money handling. The live loop only equips balls by RAM write (`equip_balls`).
- Blocks: Sustainable play beyond the current scripted segments.

**Real-time clock handling:**
- Problem: The RTC is locked (`rtc_lock=True` in `tools/play_live.py:143`), so time-of-day events and the Bug Catching Contest are untested.
- Blocks: Time-dependent encounters and any event that needs the real clock.

## Test Coverage Gaps

**Live emulator integration:**
- What's not tested: The full `tools/play_live.py` loop, the LLM director branch, and the egg-to-Falkner route run against a real ROM. The unit tests in `tests/` stub most of the emulator.
- Files: `tools/play_live.py`, `tests/test_live_execution.py`, `tests/test_dispatch.py`
- Risk: Changes to the `elif` chain or the macros can break the story path with no failing test.
- Priority: High

**Battle type handling:**
- What's not tested: Enemy types read from RAM, and move choice against a non-Falkner opponent.
- Files: `src/battle.py`, `tests/test_battle.py`
- Risk: The default `(2, 0)` types are the only case that is exercised.
- Priority: High

**LLM cost logging and failure paths:**
- What's not tested: `Supervisor.decide` on API exceptions, malformed tool calls, and the `decisions.jsonl` vs `supervisor.jsonl` schemas.
- Files: `src/llm_supervisor.py`, `app.py`, `tests/test_llm_supervisor.py`, `tests/test_app.py`
- Risk: Cost totals in the dashboard and the milestone cost report can be wrong without any test failing.
- Priority: High

**Windows-specific paths and temp files:**
- What's not tested: Under Windows, the suite has the six failures listed in Known Bugs. These are not caught in CI because no Windows runner is configured (no workflow files are present in the repository).
- Files: `tests/test_ppo.py`, `tests/test_deep_policy.py`, `tests/test_improved_training.py`, `tests/test_emulator.py`, `tests/test_pod_checkpoint.py`
- Risk: Regressions on Windows go unnoticed.
- Priority: Medium

**Web dashboard:**
- What's not tested: `tensormewweb/` has no test script in `package.json`. The telemetry in `tensormewweb/src/data/agentTelemetryData.js` is hand-authored, and `LiveAgentTab.jsx` shows it under a "live agent" label (`live-pill`).
- Files: `tensormewweb/src/components/LiveAgentTab.jsx`, `tensormewweb/src/data/agentTelemetryData.js`
- Risk: Visitors can read invented numbers as measured results.
- Priority: Medium

---

*Concerns audit: 2026-11-05*
