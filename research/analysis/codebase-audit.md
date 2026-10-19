# tensormew / PokéMind codebase audit

Read-only audit by a research subagent (code, docs, manifest, recordings read; nothing executed).
There is no `logs/` directory in the repo, so training/eval evidence comes from commit messages, recordings and docs.
Tests live in `tests/` (~252 test functions).

## 1. Architecture as implemented

The real system is a **scripted pipeline with learned pieces on top**, not the "RL + LLM on triggers" design in SPEC.md.

- Entry point `tools/play_live.py:247-746`, a priority ladder:
  1. Battle: `tactical_battle_step` (rules)
  2. Dialogue: `macros.mash_text`
  3. Unsticker, then a "blocked 3x" reflex
  4. Optional LLM call, only with `ANTHROPIC_API_KEY` and `blocked_count>=5` (`:388`)
  5. Policy: CNN+GRU samples an action, re-weighted by hand-coded masks (`:474-514`: walkability, visit-count penalty, anti-backtrack, `navigation_potential` compass)
  6. Long `elif` chain (`:529-591`) overriding the policy with fixed rules for Poké Center, egg return, rival trigger, Elm, Falkner
- Scripted navigators: `src/return_nav.py`, `src/falkner_nav.py`, `src/gym_prep.py` run BFS over ROM-derived map grids (`return_nav.py:55`). Gym interiors use hardcoded step tables. `falkner_nav.py:315-359` scripts the Kyle trade. The map/subgoal sequence is hardcoded, so the README's "no route table" claim is only half true.
- Unused/side code: `explore.py`, `segment.py`, `chain.py`, `nav_memory.py` (older checkpoint-chain path); `frontier_buffer.py`, `recurrent_policy.py` (unused by trainers).
- Not built: `src/trace.py`, `src/reasoning.py`, the benchmark runner from `docs/observability-rl-benchmark.md`, Metamon, Worker C. `saycan.py` is a 31-line party check not wired into the loop.
- `app.py` is a Streamlit thought log; `tensormewweb/` is a marketing/dashboard site with hand-authored telemetry (`src/data/agentTelemetryData.js`), not real logs.

## 2. Furthest progression

- **Verified furthest point: Zephyr Badge (Falkner, Violet City).** Evidence: `recordings/falkner_full_story/` (21 screenshots, `INDEX.md`, per-milestone `.state`, `run.gif`), runs 1-4, `falkner_showcase`. About 3,150 steps New Bark -> badge; lead farmed to Lv 15, Quilava Lv 16 won.
- **Not genuine end-to-end learned play.** `docs/HYBRID_FALKNER_RUN.md` tabulates RL vs hardcoded: only New Bark -> Mr. Pokémon is RL (and masked/shaped). Egg return, Elm hand-over, walk to Violet, training to Lv 15, healing and all battles are scripted.
- `states/manifest.json` has 9 entries, none past New Bark / Route 29, plus `ilex_src` (a human 8-badge save). Milestone states exist on disk but are unregistered.
- Never achieved: Azalea, Bugsy, agent-used Cut, any second badge.

## 3. RL setup

- Observation (`rl_env.py:18-35`): 17 base RAM features (+ goal one-hot, LocalNav block, 10 event flags); deep policy also gets 144x160 RGB.
- Actions: 5 (U/D/L/R/A). No B/Start/Select. Battle, menus, switching, items are not RL actions (`rl_env.py:796-802`).
- Reward (`progress_reward`, `rl_env.py:411-609`): heavily hand-shaped (step -0.01, new tile +1/+2, new map +10, badge +30, event flags, catches, level-ups, whiteout -25, frontier/LocalNav potentials). `navigation_potential` (`rl_env.py:261-408`) hardcodes coordinates for Routes 29/30 and Violet. Deep trainer adds curiosity, loop penalty, momentum, +50 for egg delivery.
- Algorithms: NumPy PPO (`ppo.py`); PyTorch `SpatialVisionPolicy` (ResNet-18 + RAM MLP + GRU(512), ~13.2M params) in `deep_policy.py`, trained by `tools/train_deep_ppo.py` (GAMMA .99, lambda .95, clip .2).
- Scale: 12-16 headless workers on one RTX 4090; ~1.5-2e5 steps per run. Curriculum covers only Route 29 -> Cherrygrove -> Route 30 -> egg -> Elm return; deep policy never trained past the egg.

## 4. LLM usage

- Optional, Haiku 4.5 default (`llm_supervisor.py:16`), triggers in `triggers.py`, six tools in `tools_schema.py`, offline fallback.
- In `play_live` only reached when stuck >= 5 steps; recorded Falkner runs did not need it.
- System prompt hardcodes a Route 29 walkthrough and "NEVER use Cut" (`llm_supervisor.py:23-35`), which breaks after Falkner.
- No real LLM-vs-baseline cost comparison (`tools/compare_cost.py` uses a fake trace).

## 5. Blockers for a full-game run

- **Battle is Falkner-specific:** `battle.py:~100` defaults enemy types to (Flying, Normal), never reads enemy type from RAM; 16 moves, ~17 type-chart entries; no STAB/damage estimate/status/team building; flee for anything uncatchable.
- **HM/field moves:** `use_cut` untested on a real tree; no Surf/Strength/Flash/Whirlpool/Waterfall/Fly; no water collision; nothing obtains an HM; no curriculum past Violet.
- **No progression planner:** `story_flags.py` has ~40 flags, none for badges beyond Falkner, E4 or Kanto.
- **Cheats/RTC:** `equip_balls` writes 10 Poké Balls into RAM (`play_live.py:48`); RTC locked (`rtc_lock=True`) so time-of-day events untested; text speed/battle scene/style forced via `options.py`.
- **Known failures:** fragile switching/faint macros, bump-count unstick, "7/8 exits, flat" for linear policy, long NPC dialogues stall, some Windows test failures, hardcoded Route 29/30/31/Violet maps.
- **Docs drift:** HANDOFF says "No badge has been earned" (contradicted by recordings); README/SPEC/web disagree with code; `pokemon_agent.pdf` referenced but file is `pokemon_agent.md`.

## 6. Novel vs standard

- Standard: PPO, GAE, CNN+GRU, frontier/novelty rewards, event-flag rewards (pokemonred_puffer lineage), save-state curriculum, ROM-derived pathfinding.
- Mildly distinctive: Crystal RAM/event-flag stack (banked reads, 2048-bit event reader), trigger-gated LLM with cost/decision logging (not yet measured), hybrid RL + scripted overrides with per-stage attribution. Nothing research-novel yet.

## 7. Top 10 engineering gaps for a full clear (priority order)

1. Real battle engine: enemy species/type/moves from RAM, full Gen 2 chart, damage calc, status, switching, items (or a learned battle policy).
2. Generalised navigator + HM/field-move system (Cut, Surf, Strength, Flash, Whirlpool, Waterfall; water collision; obtaining/teaching HMs).
3. Progression planner / goal DAG over all 16 badges, E4, Red; replace hardcoded Falkner stages.
4. Replace per-map potentials and `elif` overrides with generic shaping so the policy, not the script, learns.
5. Party building and levelling strategy; Poké Center/Mart/PC logistics; money and items.
6. Unlock RTC; handle time-of-day, phone calls, Bug Catching Contest.
7. Robust dialogue/menu layer (shops, trades, PC, move relearner); give the policy B/Start.
8. Checkpoint registry + seeded long-horizon eval harness with variance; regression tests without missing states.
9. Scale training past the egg with per-region curriculum states; remove hardcoded RunPod host.
10. Remove or document RAM cheats and wrong prompt text; reconcile docs with code.

## 8. Existing metrics / eval tooling

- Logging: `logs/training_metrics.jsonl`, `TerminalMonitor`, `src/visual_metrics.py`, `tools/watch_training.py`.
- Eval: `tools/eval_exits.py`, `tools/compare_policies.py`, `src/live_status.py` (`MILESTONES`), `src/run_log.py` (SQLite), `recordings/`.
- Gaps: seeded multi-run benchmark, decision-trace schema and LLM-vs-always-on cost comparison are specified but unimplemented. No committed logs, so no reportable numbers exist yet.
