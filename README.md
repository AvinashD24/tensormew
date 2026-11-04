# tensormew (PokéMind)

Autonomous agent for **Pokémon Crystal** (GBC) on PyBoy — **goal: progress as far into the game as possible** during DubHacks, using checkpoint chains + LLM-on-triggers.

## Docs

- **[`SPEC.md`](SPEC.md)** — mission, architecture, RAM, checkpoints, phases H0–H4

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m tools.play_live --help
pytest tests/ -q
```

- **ROM:** `rom/pokemon_crystal.gbc` + `rom/pokemon_crystal.sym` (US 1.1)
- **Checkpoints:** `states/manifest.json` + `tools/make_state.py`, `tools/import_sav.py`

## Copy-paste commands

Run these from the repo root. Windows PowerShell shown first; the cloud commands run in WSL (they need `rsync`).

**0. Activate the environment (PowerShell)**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**1. Open the dashboard** (live view of a run, plus replay of any saved run). It opens http://127.0.0.1:8765/ ; pick a
saved run under **Source** to replay it. Leave it running in its own terminal.

```powershell
python -m tools.dashboard
python -m tools.dashboard --demo        # no training needed: a synthetic run to look at the UI
```

**2. Train on this machine** (CPU, slow; good for a quick check). Starts every agent at New Bark with fresh weights and
saves to `logs/<run-id>.pt`; watch it in the dashboard from step 1.

```powershell
python -m tools.train_deep_ppo --config long_v3 --updates 3 --envs 4 --rollout 16 --max-steps 48 --no-curriculum --state new_bark_outside --run-id local-check
```

**3. Train on a GPU pod** (the real thing). First create a pod in the Runpod console, copy its SSH host and port from the
Connect tab (they change every time a pod starts), and keep your SSH key at `~/.ssh/id_runpod` inside WSL
(`chmod 600`). Then, from WSL in the repo folder (adjust host, port and run id):

```bash
python3 -m tools.cloud_train_deep --host <POD_IP> --port <POD_PORT> --config long_v2 \
  --run-id my-run --updates 300 --envs 32 --no-curriculum --state new_bark_outside \
  --detach --watch --no-dashboard --sync-every 5
```

This uploads the code, installs `pyboy` on the pod, trains there, and copies telemetry into `logs/` every 5 seconds so the
dashboard from step 1 shows it live. The checkpoint comes back as `logs/my-run.pt` when the run ends.
Use `--envs` up to the pod's core count (a 32 vCPU pod: 32). Each update takes roughly 25 s at 32 envs.
`--config long_v4` = `long_v3` plus a **dense path-progress reward** (`path_progress`, `path_progress_scale`, `src/path_progress.py`): +0.08 per tile of NEW best progress along the 309-step story path (progress = 309 - BFS steps to Falkner over the ROM map graph, warps included and ledges handled as in the game: walk onto the edge tile from above or the side, press its direction to jump two tiles, never climb; the route was 319 steps before this ledge fix), best-ever per episode, never negative, nothing for stepping back or re-walking, free for the start state's own progress (archive starts), a new best resets the dwell window; reported as the `path` reward component (about 24.7 for the whole path). Reward only: the observation is unchanged. Ablate with `{"preset": "long_v4", "path_progress": false}`.
`--config long_v4` additionally turns on (each field can be switched off, e.g. `{"preset": "long_v4", "aux_loss": false}`): **gamma 0.999** (horizon 1000 instead of 333 steps; `gae_lambda` stays 0.95; the whole 256-step rollout is one BPTT sequence and the GRU state is carried across rollouts), **entropy 0.01 -> 0.001** (was 0.04 -> 0.01), an **exploration budget** (`explore_budget` 12 per episode instead of 30, intrinsic weight `intrinsic_weight_end` 0.4 reached after `intrinsic_decay_updates` 800 updates), a **heal reward for HP actually restored** (`heal_reward`, `heal_reward_scale` 4, `heal_budget` 8, `src/heal_reward.py`: concave in the HP fraction so low-HP heals pay most, nothing for level-ups or white-out recovery), **low-HP battle practice** (`battle_lowhp_weight` 3: battle-archive states with the lead at <= `battle_lowhp_threshold` 40% HP and a heal item in the bag are 4x as likely to be loaded), **auxiliary prediction heads** (`aux_loss`, `aux_coef` 0.05, `aux_horizon` 8, `src/aux_tasks.py`: map id, lead-HP change and battle-start k steps ahead, targets read from the stored RAM observations; old checkpoints still load, the heads are not needed at inference) and **milestone success rates** (`ms_reach*`, `ms_envs_at`, `path_best_*` in the metrics, history and `state.json`; dashboard panel "Milestone reach rate" and the "Reaching further milestones?" check). Literature and status of each: `research/analysis/design_hardcoded_vs_learned.md`.
`--config long_v9` = `long_v8` made **progress-first** (reward and policy only, nothing scripted, no direction or distance in the observation; every new field defaults off in long_v3..v8, e.g. `{"preset": "long_v9", "anti_farm": false}`). Evidence: run v8-1 (`long_v8`, 107-195 updates, 12 envs) earned its positive reward mostly from battles (+2208, 846 wild battles, 0 trainer battles) and curiosity (+1842) while the path reward paid +1176; best path progress ~51 of 695 tiles, 92 % of visits on Route 29, 131 white-outs, entropy 1.58 -> 0.99, A pressed 39 %+. (1) `path_progress_scale` 0.3 -> **1.0** (the whole egg-errand route pays 695 instead of 208), `milestone_reward_scale` **1.5** (story payments 195 -> 292) and `milestone_speed_bonus` **1.5** (every milestone / key-event / badge payment is multiplied by 1.5 at step 0 of the episode falling linearly to 1.0 at the step limit, `src/anti_farm.py`), `explore_budget` 12 -> 8, `retreat_budget` 15 -> 30. (2) `anti_farm`: wild-battle win / KO / damage x`wild_battle_scale` 0.4 and then x0.6^(n-2) from the 3rd wild battle of an episode (floor 10 %), all under a `wild_battle_budget` of 6 per episode (v8: 20 shared by every battle); trainer battles keep full size under their own `trainer_battle_budget` 20; the level-up reward is paid only while the party's highest level is below the current stage's `dwell_level_targets` entry (levelling for Falkner stays rewarded, grinding above the target pays nothing); a catch pays `catch_reward` 2 only for a species not yet caught in the episode (`catch_budget` 4). (3) `time_feature`: one extra observation input, steps_in_episode / episode_limit in [0, 1] (Pardo et al. 2018, "Time Limits in Reinforcement Learning"); 80 -> 81 inputs, older checkpoints raise `CheckpointIncompatibleError`. (4) `entropy_boost_threshold` 0.8 -> 1.0. Per-term before / after budgets: `research/analysis/design_hardcoded_vs_learned.md`. Tests: `tests/test_long_v9.py`.
`--config long_v8` = `long_v7` plus **valid-action masking in battles, a periodic from-start evaluation and the egg-errand route** (three flags, default off in every other preset, e.g. `{"preset": "long_v8", "battle_action_mask": false}`). (1) `battle_action_mask` (`src/battle_mask.py`; Huang & Ontanon 2020, "A closer look at invalid action masking in policy gradient algorithms"): in a battle the buttons that provably do nothing in the visible screen get a logit of -1e9 before the softmax, so exploration is not spent on them. The context is read from the background tilemap (what the player sees), not from the cursor RAM (stale outside its own menu). Rules, each checked on the real emulator by pressing the button the way the worker does and comparing screen, tilemap and menu/battle WRAM against a no-button run: main FIGHT/PKMN/PACK/RUN grid: edge directions and B are no-ops (the grid does not wrap); move list: left/right are no-ops, up/down only with one move (with 2-4 moves the list wraps, so they are NOT masked); party list: left/right; a plain battle text box (full-width box, text printed, no cursor anywhere): all four directions. A and B are never both masked, the mask is never all False, unknown screens (bag list, yes/no prompts, sub-menus, animations, transition frames) and everything outside a battle are unmasked. The worker publishes the mask of the current observation, the trainer stores it per step in the rollout (`RolloutBatch.masks`), and the policy applies the same masked logits in `act`, `forward*`, `evaluate_actions` and `evaluate_actions_sequence`, so log-probs, entropy and the PPO ratio all describe the masked distribution. The mask is not an observation feature, checkpoints are unchanged, and a policy trained with masks should also be evaluated with them (the logits of masked buttons are never trained). Logged per update: `mask_valid_frac` (mean share of allowed actions on battle steps), `mask_active_frac`, `mask_battle_steps`. Not done: overworld blocked-direction masking (needs per-tile collision/NPC knowledge; skipped). (2) `periodic_eval` (`eval_every` 50, `eval_episodes` 4, `eval_max_steps` 1024; `src/periodic_eval.py`, `PeriodicEvaluator` in `tools/train_deep_ppo.py`): after every `eval_every`-th update the current policy plays the episodes from `new_bark_outside` only, on one extra in-process env (no archive / checkpoint / swarm resets, no stagger offset, no start jitter), stochastic like training with its own sampling generator, GRU state and emulator, the early battle A-bias off; it reads and writes no training state and no archive. It logs `eval_update`, `eval_best_progress_mean/max`, `eval_ms_reach` (fraction of episodes reaching each milestone), `eval_ms_steps` (mean steps to it), `eval_whiteout_frac`, `eval_return_mean`, `eval_seconds` on that update only (metrics.jsonl, history.jsonl, `state.json["eval"]`, an `eval` event) and the dashboard shows a "From-start evaluation" card (hidden for runs without it). CLI: `--eval-every/--eval-episodes/--eval-max-steps`.
`--config long_v8` additionally sets **`egg_errand_route`** (the third v8 change, off in every other preset; `src/story_path.py` EGG profile, the phases in `src/route_graph.py`). (3) The route has three PHASES read from event flags in RAM (30 = egg received, 31 / 1812 = egg given to Elm / Route 30 open), each with its own BFS distance field and goal tile: phase 0 New Bark -> Mr. Pokemon's house, tile (3, 6) where he hands over the egg (the Route 30 choke tiles are blocked in the graph): 185 steps; phase 1 back to Elm's lab, tile (5, 3) in front of Elm: 183 steps; phase 2 the opened graph to the tile in front of Falkner (the 309-step route plus the walk from Elm's lab to the New Bark start tile): 327 steps; 695 in all. The path reward is `path_progress_scale` per tile of NEW best `phase_offset + phase_length - distance to the phase goal` (continuous at the phase borders, so no jump; best-ever per episode as before; the retreat cost is measured in the same combined tiles; unknown positions pay nothing). The story milestones grow from 8 to 11: Route 29 +5, Cherrygrove +10, Route 30 +10, **Mr. Pokemon's house +10, egg received +15, egg delivered +30**, Route 31 +10, Violet gatehouse +5, Violet City +15, Violet Gym +15, Zephyr Badge +60 and `EVENT_BEAT_FALKNER` +10 (total 195, the base list is 140). The observation gets the 3 extra "done" bits (story block 10 -> 13, `DEEP_OBS_DIM` 77 -> 80; nothing about direction or distance is added; checkpoints of earlier presets raise `CheckpointIncompatibleError`), the milestone panels, records, source-split stats and route-checkpoint buckets (keyed by the combined progress, file name suffix `_p<phase>`) follow the milestone count, the swarm archive saves a state when the egg is received and when it is delivered (stage entries 5 and 6), a loaded checkpoint seeds the maps implied by its event flags so nothing is repaid, and `dwell_level_targets` has 12 entries (`[8, 8, 10, 10, 10, 10, 10, 10, 11, 12, 12, 0]`). The dashboard draws the three phase legs (different colour and dash) and the 11 story chips for runs whose `meta.json` has `egg_errand_route`; older runs keep the 309-step layout.
`--config long_v7` = `long_v6` plus **learned-vs-head-start metrics and healing care** (every new field defaults off in long_v3..v6, e.g. `{"preset": "long_v7", "lowhp_exposure_penalty": false}`). Evidence: run v6-1 (`long_v6`, ~400 updates): 43-63 % of the finished episodes were white-outs, the showcase lead was below 30 % HP in about half of the update snapshots, the heal reward paid ~3 per update in total, and the overall reach rates mix base-start episodes with checkpoint resets that begin further along. (1) `source_split_metrics` (`src.milestone_stats.SourceSplitStats`): per reset source (`base` = plain New Bark start, `route` checkpoint, `swarm`, `battle`) the number of episodes, the mean path-progress GAIN (best minus the progress at the episode start) and the milestone reach rates counting only milestones not already passed at the start; metrics keys `src_split` (this update) and `src_split_cum` (whole run); dashboard panel "Learned progress vs head start" (the base-start row is the learned-from-scratch measure; older runs show a quiet "not recorded" note). (2) Healing (`src/heal_care.py`, reward only, nothing scripted): `heal_reward_scale` 4 -> 6 and `heal_budget` 8 -> 10 (still capped per episode); `lowhp_exposure_penalty` (`lowhp_exposure_threshold` 0.3, cost `lowhp_exposure_min` 0.002 -> `lowhp_exposure_max` 0.01 per overworld step over `lowhp_exposure_ramp` 128 exposed steps, `lowhp_exposure_budget` 6 per episode; free in Pokemon Center / Mart maps, in battle, in text and on a healing step; component `lowhp`); `battle_survive_bonus` (`battle_survive_amount` 0.5 for a battle WON with the lead above `battle_survive_threshold` 0.5 HP, `battle_survive_budget` 3; fleeing pays nothing; component `survive`); `potion_use_reward` (`potion_use_amount` 1.0 when a healing item is used, HP is restored and the lead was below `potion_use_threshold` 0.5, `potion_use_budget` 3; component `potion`); `heal_trip_exempt` (retreat, stagnation and dwell costs are zero while the lead is below `lowhp_exposure_threshold` HP or in a Center / Mart, so walking back to heal is never punished); `battle_lowhp_weight` 3 -> 6 and `battle_lowhp_threshold` 0.4 -> 0.5 (more low-HP-with-Potions battle-archive starts). Offline "learned vs head start" check of any checkpoint: `python -m tools.evaluate_policy --sampled --checkpoint logs/X.pt --config long_v6 --episodes 12 --max-steps 2048 --seed 1000 --name X` (stochastic actions, base start, the real training env; add `--uniform-random` for the no-learning baseline).
`--config long_v6` = `long_v5` plus **three changes for the end-of-Route-29 plateau** (each a config field, default off in every other preset). Evidence: run v5-1 (`long_v5`, update 317, 24 envs): best path progress ever 97 of 99 Route 29 steps (once), typical per-episode best ~30, only Route 29 reached (0 for Cherrygrove), 19 % of visits on one tile of the Route 29 gate building (a dialogue loop), net reward ~0. (1) `route_ckpt` (`src/route_ckpt.py`): a self-generated reverse curriculum / Go-Explore archive. A worker saves an emulator state when its current path progress enters a bucket (every `route_ckpt_every` 12 tiles) it has not saved in this episode (calm state only: no battle / text / white-out; one save per 64 steps per worker); files `rc_b<bucket>_<slot>_s<stage>.state` under `logs/swarm/<run>/route/`, `route_ckpt_per_bucket` 4 per bucket (the oldest is replaced), `route_ckpt_max` 60 in all (the fullest bucket loses its oldest). `route_ckpt_prob` 0.4 of the resets load one, the bucket drawn with weight `floor + (1 - floor) * bucket / max_bucket` (floor 0.2, so the furthest bucket is most likely and every bucket keeps a chance). Reset order, one draw per reset (the first episode is always a start state): battle archive 0.15, swarm archive 0.25 (was 0.5, the swarm archive only holds map-entry states), route checkpoint 0.40, the rest (0.20) the start state; validation requires the three shares to sum to <= 1 and `path_progress`. On a checkpoint reset the best-progress baseline is the state's own progress (no free reward), budgets / seen sets reset normally, and the seen-map set is seeded with the milestone maps the saving episode had passed (`_s<stage>` in the file name), so a passed milestone is never paid twice and the milestone bits in the observation are consistent. Metrics: `route_ckpt_size`, `route_ckpt_max_bucket`, `route_ckpt_buckets` (states per bucket), `route_ckpt_resets`, `route_ckpt_reset_share`. (2) `dialogue_loop_penalty` (`src/dialogue_loop.py`, component `dialogue_loop`): 0.02 per step (budget 10 per episode) while an A-driven dialogue repeats without effect: more than 60 such dialogue steps in the last 200 steps with no progress signal (map, event flag, bag, party, money or Pokedex change), or one (map, tile, facing) talked to more than 6 times without effect. Not charged: the first talk to a tile, text not opened by the agent's A press, battle steps, Pokemon Center maps. (3) `warmup_rollout` 1: one rollout before update 0 that only feeds the running reward scaler (no gradient step, not logged, update counter and archives untouched, episodes continue; skipped when resuming a scaler). Why: every tile is novel at the start of a run, so update 0 pays curiosity + new-tile bonuses on almost every step (mean reward 12.4 vs ~0 later); the scaler's first statistics would otherwise come from that burst alone. Literature: Go-Explore (Ecoffet et al. 2019, Nature 2021), reverse curriculum (Florensa et al. 2017), the backward algorithm from one demonstration (Salimans & Chen 2018; here the "demonstration" is the agents' own best progress), RND initialises its normalisation statistics with random-agent steps (Burda et al. 2018). Guesses: bucket size 12, 4 per bucket, 60 files, floor 0.2, the reset shares, and all dialogue-loop thresholds. Also in `long_v6` (reward only, nothing added to the observation; evidence: 36-76 % of the Route 29 visits sat in the west dead-end corridor, x 14-16, y 14-15, 18 tiles deep, path progress 41 at its junction down to 23): (4) `retreat_penalty` (`src/path_progress.py`, component `retreat`): every walked step that lowers the BFS path progress is charged `retreat_cost_frac` 0.35 x `path_progress_scale` 0.3 = 0.105 per tile lost, re-approach is free and new bests pay in full, at most `retreat_budget` 15 per episode; never in battle / text, Pokemon Center maps, on a map-changing step (warp, door, teleport), after a white-out or off the route graph. The real route is monotone (+1 progress per step), so it costs exactly 0 there (tested on all 309 steps); walking the 18-tile corridor in and out costs 1.89, camping in its far end by oscillating hits the 15 cap after ~143 outward steps. It is NOT potential-based shaping in the sense of Ng, Harada & Russell (1999) (that would pay for both directions and keep the optimal policy); this asymmetric version is not policy-invariant on purpose. (5) `stagnation_penalty` (`src/stagnation.py`, component `stagnation`): after `stagnation_after` 250 counted steps (not battle, text or Pokemon Center) without a new best path progress, 0.004 per step rising linearly to 0.02 over 300 steps, at most 10 per episode (about 3.6 for the first ramp, then ~0.02 per step, so a camping episode pays -10 at roughly step 870 of its stagnation); free, with no ramp banking, while the dwell level exemption holds (party below the stage target and a fight or level-up in the last 512 steps); needs `dwell_penalty`. Composition with dwell: per step the agent pays max(dwell, stagnation), never the sum; the stagnation component is only the excess over that step's dwell charge and only the excess counts against its budget. Thresholds are guesses. (6) `terminate_on_whiteout`: the step on which the whole party faints ends the episode as a TRUE termination (`done`, `terminated`, not `truncated`: no value bootstrap, the target is the -15 white-out penalty alone); the env then resets through the normal path (start / swarm / battle / route-checkpoint) with all per-episode state fresh. Info carries `end_reason` (`whiteout` / `time_limit`); the metric `whiteout_episodes` (and `episodes_ended`) is logged per update.

`--config long_v5` = `long_v4` plus **anti-collapse changes** (each a config field, default off in every other preset, e.g. `{"preset": "long_v5", "idle_press_penalty": false}`). Evidence: run v4-1 (`long_v4`, 75 updates, 24 envs) collapsed to pressing A (58 % of the recent actions; entropy 1.78 -> 1.12 at update 38 -> 0.40 at update 75; mean best path progress 8-22 of 319 tiles, the route length at that time; 309 after the ledge fix). Design note: an A press does not move the character, so it escapes the movement-linked `bump` and `loop` costs while the path reward (0.08/tile) was tiny next to them; with entropy coefficient 0.01 -> 0.001 and gamma 0.999, PPO locked the cheap behaviour in (part of the A share is legitimate battle-menu pressing, so the cost also has to be checked per state). Changes: (1) `disable_bump_penalty` / `disable_loop_penalty` (need `path_progress`): wall-bump and tile-revisit costs off, dwell + path progress cover idling; step, damage and whiteout stay. (2) `idle_press_penalty` (`src/idle_press.py`, reward component `idle_press`): `idle_press_cost` 0.015 per A / B press in the plain overworld after which nothing observable changed (position, facing, map, dialogue / script, battle, party, bag, money, event flags), at most `idle_press_budget` 10 per episode; battle, menu and dialogue steps are never charged. Known hole: talking to the same NPC again is not "no effect". (3) `path_progress_scale` 0.08 -> 0.3: the whole path pays ~96 next to ~140 of milestones and the exploration cap of 12 (a walker earns ~+0.3 per new best tile against -0.01 per step; A-spam now costs ~-0.025 per step until the dwell and idle budgets run out). (4) Entropy: start 0.02, floor 0.005, `entropy_anneal_updates` 1200 (the coefficient cannot anneal faster than update / 1200, on top of the route-progress gating), and the safety net `entropy_boost` (`src/entropy_guard.py`): while update < `entropy_boost_until` 300 and the previous update's mean entropy is below `entropy_boost_threshold` 0.8, the coefficient is raised to `entropy_boost_coef` 0.05 until the entropy is back at 1.0; logged as `entropy_boost` (0/1). Only the loss coefficient changes, so the PPO update stays valid. Literature: entropy regularization is standard in PPO (OpenAI Five annealed it with much larger batches) but Andrychowicz et al. 2020 ("What Matters in On-Policy RL") found it did not help on continuous control, and Ahmed et al. 2019 show its effect depends on the reward landscape; the specific values here are guesses.
`--config long_v3` is the earlier fresh from-scratch preset: `long_v2` (32 envs x 256 steps, 800 updates, New Bark
start, no route oracle, the agent plays its own battles) plus staggered episodes (resets no longer happen in lockstep;
`phase_spread` and `reward_8` are logged), a self-generated battle archive with an early A-bias inside battle menus, and
furthest-map weighted archive resets. Each is a config flag (`stagger_episodes`, `battle_archive`, `battle_a_bias`,
`swarm_furthest_weighting`), so any of them can be ablated. `long_v3` also has a **dwell penalty** (`dwell_penalty`, `src/dwell.py`): staying in at most 16 tiles for 192 non-battle steps with nothing new (tile, map, milestone, new species) costs 0.005 -> 0.03 per step (reported as the `dwell` reward component, at most 15 per episode, reset by leaving the area). It is zero while the party's highest level is below `dwell_level_targets[furthest story stage]` (default `[8, 8, 10, 10, 10, 11, 12, 12, 0]`, stages New Bark ... Violet Gym, Falkner beaten) and the agent fought or leveled within `dwell_grind_grace` (512) steps, so leveling for Falkner is not punished; the older per-tile `loop` term is switched off in that case too (`dwell_exempt_loop`). Pokecenter maps are exempt. Example ablation: `{"preset": "long_v3", "dwell_penalty": false}`. Other ablations, e.g. `--config path/to/json` with `{"preset": "long_v3", "battle_a_bias": 0}`. **Checkpoints from before this change cannot be loaded** (the observation
lost the route direction/distance features, gained battle features, and the policy gained a B button); loading one
raises `CheckpointIncompatibleError` telling you to retrain.

**4. Resume a run from its checkpoint** (needs `logs/my-run.pt` and `logs/my-run.optim.pt` locally; the script uploads them):

```bash
python3 -m tools.cloud_train_deep --host <POD_IP> --port <POD_PORT> --config long_v2 \
  --resume --save logs/my-run.pt --run-id my-run-2 --updates 200 --envs 32 --no-curriculum --state new_bark_outside \
  --detach --watch --no-dashboard
```

**5. Reconnect to a run that is still going on the pod**, resume syncing and pull the checkpoint when it ends:

```bash
python3 -m tools.cloud_train_deep --host <POD_IP> --port <POD_PORT> --attach --watch --no-dashboard --save logs/my-run.pt
```

**6. Evaluate a checkpoint** (greedy, from the New Bark save) and replay it:

```powershell
python -m tools.evaluate_policy --checkpoint logs/my-run.pt --episodes 3
```

**7. Stop paying when you are done.** A pod bills until you stop or terminate it, even if training has ended or crashed.
Terminate it in the Runpod console (Pods -> your pod -> Terminate) after the checkpoint is back in `logs/`.
A stopped pod often cannot restart (its GPU gets taken), so copy results first and make a new pod next time.

Troubleshooting: if the dashboard shows nothing, the run id folder under `logs/runs/` is the source of truth (pick it under
**Source**). If a long cloud run dies, its latest checkpoint is on the pod in `/workspace/tensormew/logs/`; pull it with
`rsync -az -e "ssh -i ~/.ssh/id_runpod -p <POD_PORT>" "root@<POD_IP>:/workspace/tensormew/logs/my-run*" logs/`.

## What is hard-coded and what is learned

The agent starts from `states/new_bark_outside.state` (Cyndaquil, outside New Bark Town). The policy gets the screen plus
a RAM vector (position, map id, party, bag, which story milestones are done, battle facts) and presses six buttons
(up/down/left/right/A/B). It is **not** told where to go and it plays **its own battles** (fight, move choice, switching,
items, running). The goal is the story path to the Zephyr Badge. **The Mystery Egg errand is required**: Route 30 is cut in
half at the one-tile choke (5, 25) by NPC sprites (Joey's event, two Rattatas, trainer Mikey) that only disappear once Elm has the egg
(hide flag 1812 is set together with `EVENT_GAVE_MYSTERY_EGG_TO_ELM`), so the real order is Route 29 -> Cherrygrove -> Route 30
-> Mr. Pokemon's house (the egg) -> rival fight at Cherrygrove -> back to Elm's lab in New Bark (give the egg, Route 30 opens)
-> Cherrygrove -> Route 30 -> Route 31 -> Violet City -> Violet Gym -> Falkner. The older 309-step "shortcut" route
(presets up to `long_v7`) skips the errand and can only be finished from states saved after it. Still hard-coded by design:
the start state, 10 Poke Balls + 5 Potions at every reset, auto-advanced *overworld* dialogue, and the reward terms.
Details and reasoning: [`research/analysis/design_hardcoded_vs_learned.md`](research/analysis/design_hardcoded_vs_learned.md).
The old BFS goal potential survives only as the opt-in `goal_potential` config flag (off in every preset).

## Training from scratch

This branch carries the infrastructure only: no trained weights, recordings or web app.

```bash
python -m tools.train_deep_ppo --updates 100 --save logs/deep_policy.pt   # fresh policy
python -m tools.train_deep_ppo --resume ...                               # opt in to warm-start
python -m tools.train_deep_ppo --push-weights ...                         # opt in to commit/push weights to THIS branch
```

Resume and weight-push are both off by default. `tools/sync_weights.py` refuses to push from `main`.

## Live training dashboard

Training streams telemetry to `logs/live/` (disable with `--no-live`). Watch it in a browser while it runs:

```bash
python -m tools.dashboard          # http://127.0.0.1:8765, reads logs/live + logs/training_metrics.jsonl
python -m tools.dashboard --demo   # preview the UI with a synthetic run, no training or ROM needed
```

It shows live game screens from every parallel emulator, reward / entropy / KL / loss charts with hover,
a training-phase timeline, per-map exploration heatmaps with agent trails, the action mix, recent episodes,
milestones (new maps, badges, records) and automatic health warnings (entropy collapse, KL spikes, stuck exploration, stale feed).
### Replaying a finished run

Every run also saves an append-only history under `logs/runs/<run-id>/` (`history.jsonl` with each update's
metrics, per-agent positions, tile-visit deltas, episodes and milestones; `frames/` with a JPEG montage of all
agents every `--frame-every` updates, default 5; `meta.json`). Open the dashboard and pick the run under **Source**
to scrub or play it back: charts, heatmap, milestones and the agent montage all rewind together.
Name a run with `--run-id`; disable history with `--runs-dir ""`.

### Cloud (GPU pod) training

Telemetry is written on the pod, so bring it home while the run is going:

```bash
python -m tools.cloud_train_deep --updates 200 --run-id baseline-1 --watch
```

`--watch` rsyncs `logs/live/`, `logs/runs/` and `logs/training_metrics.jsonl` from the pod every `--sync-every` seconds
(default 5), starts the dashboard locally, and stays up after the run so you can replay it. Telemetry is always
pulled at the end, even if training fails or you Ctrl+C, so a destroyed pod no longer takes the data with it.
Manual alternative: run `python3 -m tools.dashboard --no-open` on the pod and tunnel with
`ssh -L 8765:127.0.0.1:8765 root@<pod>`.

**Pod address.** A pod gets a new SSH host/port every time it starts, so set `RUNPOD_SSH_HOST` / `RUNPOD_SSH_PORT`
(or pass `--host/--port`) per session. Community Cloud stock for 4090-class GPUs is often empty; if a pod will not
start, try Secure Cloud or another GPU type rather than waiting.

**Preflight.** Before training, the script SSHes in and checks the GPU (`nvidia-smi`), core count (`nproc`),
free disk and python deps. The stock PyTorch image lacks `pyboy` and Ubuntu 24.04 blocks plain pip, so it runs
`pip install --break-system-packages pyboy==2.8.1` automatically (`tools/pod_bootstrap.sh` does the same by hand).
`--envs` above the core count is capped with a warning. Skip with `--skip-preflight`.

**Config.** `--config long_v9|long_v8|long_v7|long_v6|long_v5|long_v4|long_v3|long_v2|blueprint_v1|long_v1|legacy|path/to/config.json` validates the config locally (`src/train_config.py`),
uploads a JSON file to the pod, and passes `--config` to the remote trainer.

**Detached runs.** For multi-hour runs use `--detach`: the job runs under tmux (or `setsid nohup`) on the pod, logging
to `/workspace/tensormew_job/train.log` and writing an exit code to `/workspace/tensormew_job/done`, so a dropped SSH
session or a sleeping laptop does not kill it. The script then polls and syncs telemetry. If it is interrupted or you
close the terminal, reconnect with the same pod address:

```bash
python -m tools.cloud_train_deep --updates 400 --config long_v2 --run-id v1 --detach --watch
python -m tools.cloud_train_deep --attach --watch --save logs/v1.pt   # later: resume syncing, pull the checkpoint (each run saves to logs/<run-id>.pt)
```

Exit codes: 0 ok, 1 sync failure, 2 preflight/config error, 3 remote training failed, 4 job lost (died without a
done marker, or pod unreachable), 130 interrupted. Telemetry and the checkpoint are always pulled on completion.
If the pod itself is deleted, the checkpoint and `/workspace` go with it unless you use a network volume.
