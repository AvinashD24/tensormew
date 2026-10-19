# Community RL / bot work on Pokemon gameplay (Red/Blue/Yellow, Gold/Silver/Crystal)

Compiled 2026-10-19. Scope: agents that play the single-player game through an emulator (not battle simulators such as Pokemon Showdown). Stars/licenses/dates were read live from the GitHub API on the compile date. Claims I could not confirm from a primary source are tagged **[UNVERIFIED]**. Items I verified by reading code or symbol files myself are described as "read".

Key takeaway up front: **there is no published RL agent that has meaningfully progressed through Gold/Silver/Crystal.** All Gen-2 RL repos found are small, early-stage, and report no results. The only full-game RL result is on Red (Rubinstein et al., scripted HM handling, 10B-100B-step scale). The only Gen-2 completion found is by an LLM harness (Gemini on Crystal), not RL.

---

## 1. Comparison table

| # | Project | Game | Algo | Obs | Actions | Reward style | How far | Compute | License | Stars (2026-10-19) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) (v1 `baselines/`, v2 `v2/`) | Red | PPO (SB3), CnnPolicy / MultiInputPolicy | 3-frame stack 72x80 gray + HP/level/badge/event bits + (v2) coord visit map | 7 buttons, 24 frames/step | events + level + heal + badges + exploration (v1 frame-KNN, v2 coordinate count) | Pewter/Cerulean region; v2 README: "Reaches Cerulean" | v1: 16 envs; v2: 64 envs (sec 2.1); step totals not published | MIT | 7,923 |
| 2 | [PufferAI/pokegym](https://github.com/PufferAI/pokegym) + PufferLib 0.5/0.7 (J. Suarez, xinpw8) | Red | PPO (CleanRL via PufferLib) | Whidden-style, heavily modified | 7 buttons | Whidden-style + Bill/Cut/etc. flags | Badge 3 (Lt. Surge) at 404M steps; 86% of envs got HM01; Celadon reached | hundreds of millions of steps; WandB run | MIT | 53 |
| 3 | [drubinstein/pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer) (a.k.a. thatguy11325) "Pokemon Red (RL Edition)" | Red | PPO + LSTM (PufferLib/CleanRL), 288 envs / 24 workers, config targets 10B-100B steps | dict: 2-bit screen, visited mask, direction, map id, bag, full party stats, 320 event bits, etc. | 7 buttons + heavy scripted helpers | event-flag count + exploration + badge + HM/Cut/menu shaping | **Claims to beat the game (E4/champion)** with scripts (March 2025); not stable with all scripts off | not stated; policy <10M params | **none (no license file)** | 211 |
| 4 | Pleines et al., [arXiv 2502.19920](https://arxiv.org/abs/2502.19920) "Pokemon Red via Reinforcement Learning" | Red | PPO, 32 workers, 65,536 batch; GRU variant | gray 72x80 x3 frames, 48x48 visited crop, party HP/level, event flags | 7 buttons, 8 held + 16 released | +2/event, +0.005/new coord, 2.5x heal, level term | Cerulean City (~20% of game); **no config solved Cut** | ~36 h/run (non-recurrent); GRU ~24 days CPU | CC / arXiv (code: MarcoMeter/neroRL `poke_red` branch, MIT) | neroRL 43 |
| 5 | [reddheeraj/PokemonRL](https://github.com/reddheeraj/PokemonRL) = "PokeRL" [arXiv 2604.10812](https://arxiv.org/abs/2604.10812) | Red | PPO (SB3) with curriculum of 3 sequences | 4 gray frames + per-map visited mask | 7 buttons | per-sequence custom rewards | exits house, reaches grass, wins first rival fight only | small (single machine) | NOASSERTION (paper CC BY 4.0) | 4 |
| 6 | [xjViper/PokemonCrystalAI](https://github.com/xjViper/PokemonCrystalAI) | **Crystal 1.1** | PPO (SB3 CnnPolicy), gamma 0.998 | 128x40x3 composite: 3 frames + reward bars + KNN novelty | 6 buttons (8 with start) | Whidden v1 port: events (flag popcount), level, heal, -dead, badge x5, seen-poke, frame-KNN | **No result reported** (README = a GIF) | script default is toy: 3 envs x 20,480 steps x 3 iterations | MIT | 2 |
| 7 | [ThirdStepper/Pokemon_Gold](https://github.com/ThirdStepper/Pokemon_Gold) | **Gold** (pokegold layout) | PPO (SB3), 7 envs, 7M-step cfg / 8 h overnight | 80x72 gray, frame stack 4, frame skip 12 | D-pad + A/B/Start/Select | dense shaping + one-shot plot-flag table incl. HM01-06, escalating badge rewards | **WIP; "dont expect much"**; no results | 7 envs on one desktop | BSD-2-Clause | 0 |
| 8 | [gk331251-byte/pokemon-gold-rl](https://github.com/gk331251-byte/pokemon-gold-rl) | **Gold** | PPO (SB3), 8 envs | RAM-only 26 values | 8 buttons | exploration/badge/team/penalties | no results; **address table partly wrong** (sec 3.3) | claims 50-60 steps/s (README also says ~180) | none | 0 |
| 9 | [matiaszanolli/PokemonCrystalRL](https://github.com/matiaszanolli/PokemonCrystalRL) | **Crystal** | Hybrid: small local LLM (Ollama) + RL curriculum | screen + 25 RAM values | buttons | +100 first mon, +500/badge, +50/level, etc. | no results shown | local | MIT (file) / NOASSERTION (GitHub) | 3 |
| 10 | [DhananjayAshok/GameBoyWorlds](https://github.com/DhananjayAshok/GameBoyWorlds) | Red, **Crystal**, Brown, Prism, Fool's Gold, Star Beasts | framework (Gymnasium); targets LLM/VLM and RL agents | screen-region parsers + optional WRAM parser | low-level buttons or abstract "openmenu()" controllers | task/metric based | framework, no leaderboard claims | n/a | MIT | 14 |
| 11 | Edelaar thesis, Leiden 2025 (LIACS) | **Gold** (abstracted) | PPO (SB3) + curriculum | team composition vectors | pick Pokemon (action-masked) | gym-leader battle HP ratio | **Not gameplay**: 6-step team-builder episodes, heuristic battle sim | 100k steps/gym | thesis | n/a |
| 12 | PokeAgent Challenge speedrun track, [arXiv 2603.15563](https://arxiv.org/abs/2603.15563) | Emerald (GBA, mGBA) | winners: LLM-scripted policies distilled + RL; Hamburg: recurrent PPO | frames + limited state | buttons | +1/milestone (15 milestones to Roxanne) | 6/22 valid submissions finished; best 40:13 | n/a | paper CC BY-NC-SA 4.0 | heatz123 MIT, DEEPEST MIT |
| 13 | Gemini plays Crystal (J. Zhang / ARISE) | **Crystal** | LLM agent harness (not RL) | screen + harness state | tool calls | none | **Beat Crystal incl. Red** (Dec 2025) per secondary reposts | ~1.9B tokens est. by a commenter | n/a | n/a |
| 14 | ThePokeBot (2015) | Red | hand-coded scripted bot, reads RAM, TAS-style route | RAM | buttons | n/a | 1:50:14 any% glitchless-ish vs human WR 1:49 (April 2015) | n/a | not found | n/a |

Table caveat: rows 12-14 are included for design/safety lessons; they are not Gen-1/2 RL on the same footing.

---

## 2. Detailed project notes

### 2.1 PWhiddy/PokemonRedExperiments (the origin of almost everything)
- Link: https://github.com/PWhiddy/PokemonRedExperiments. Author: Peter Whidden. Repo created 2019-12-25; last push 2026-09-11. MIT. 7.9k stars, 786 forks. YouTube video `DcYLT37ImBY` made it famous (2023) **[UNVERIFIED exact date]**.
- Game: Pokemon Red, PyBoy emulator, start from post-intro save state. Algo: PPO from Stable-Baselines3.
- Read from code, v1 (`baselines/run_baseline_parallel_fast.py`): 16 SubprocVecEnv, ep length 2048*10, `n_steps=ep_length//8`, batch 128, 3 epochs, gamma 0.998, CnnPolicy, `action_freq` 24 emulator frames per step.
- Read from code, v2 (`v2/baseline_fast_v2.py`): 64 envs, ep length 2048*80, batch 512, 1 epoch, gamma 0.997, ent_coef 0.01, MultiInputPolicy. Replaces frame-KNN novelty with coordinate-count exploration. README: "Trains faster... Reaches Cerulean... Streams to map by default".
- Observation v2 (read): Dict of `screens` 72x80x3 (frame stack), `health`, `level` (Fourier-encoded), `badges` (8 bits), `events` (bit vector of the event-flag RAM region), `map` (explored-coordinate crop), `recent_actions`.
- Actions: 7 (4 d-pad, A, B, Start). Hold ~8 frames then release, 24 frames/step.
- Reward v2 (read, `get_game_state_reward`): `event` = 4 x (max event-flag popcount above baseline), `heal` x10, `badge` x10, `explore` = 0.1 x unique coords seen x explore_weight, `stuck` = -0.05 x repeat-coordinate penalty. Level and opponent-level terms are commented out.
- Key tricks: reward = RAM bit counting (event flags at 0xD747-0xD886 in Red); heal reward counts HP *gains* only if party size unchanged; max-event reward is monotonic; fast-text save state; map-visualization/streaming wrapper (`pokerl-map-viz`, 57 stars, MIT).
- Limits: stalls around Cut/HM01 and later; hand-engineered per-run luck; compute heavy for single machine.

### 2.2 PufferAI/pokegym + PufferLib (Joseph Suarez, xinpw8/Daniel Addis, Keelan Donovan)
- Link: https://github.com/PufferAI/pokegym (53 stars, MIT, last push 2024-06-08). Docs say "clone PufferLib branch 0.5 ... `pip install -e '.[cleanrl,pokemon_red]'`", train with `demo.py --train`.
- **No Pokemon env exists in current PufferLib main**: I listed the repo tree at HEAD (2026-10-19) and found no pokemon path. It lived in the 0.5/0.7 era (branch names `BET_pufferlib_0.7_badge_3_400m` in the pokegym README). PufferLib itself: 6,540 stars, MIT.
- Reported results (from the pokegym README, i.e. self-reported): most envs get Badges 1-2; Badge 1 at 9.6M steps; Badge 2 at 35M; Bill saved at 53M; 86% of envs obtained HM01 (first at 60M); 68% taught Cut; some reached Celadon and entered Gym 3 via Cut; **Badge 3 at 404M steps**. WandB run is linked from the README.
- Goal stated: "beat Pokemon Red using as pure RL as possible."
- I found no separate paper or blog by Suarez on Pokemon results (puffer.ai blog page returned nothing relevant). Suarez appears as co-author of the pokemonred_puffer README and is thanked in the Pleines paper.
- Lesson: the PufferLib port is mostly a throughput story (vectorized emulators; many hundreds of millions of steps).

### 2.3 drubinstein/pokemonred_puffer ("Pokemon Red (RL Edition)") - the strongest Red result
- Link: https://github.com/drubinstein/pokemonred_puffer (211 stars, 33 forks, created 2024-01-21, last push 2025-12-04). Mirror/original owner thatguy11325. **No license file** (GitHub shows none) - all rights reserved by default, so do not copy code verbatim. Authors: David Rubinstein, Keelan Donovan, Daniel Addis, Kyoung Whan Choe, Joseph Suarez, Peter Whidden.
- Write-up: https://drubinstein.github.io/pokerl/ (v1 changelog 2025-03-04): "As of February 2025, we are able to beat Pokemon Red with Reinforcement Learning", policy <10M params ("60,500x smaller than DeepSeek-V3"). Gigazine (2025-03-10) adds: agent beats the game **with scripts in place**; it can clear with any single script removed but "not yet stable with all scripts disabled". Safari Zone was hardest: required a script plus a reward proportional to steps remaining.
- Algorithm (read config): PPO with LSTM policy, `use_rnn: True`, `batch_size 65536`, `minibatch 2048`, 3 epochs, gamma 0.998, gae 0.95, ent 0.01, clip 0.1, `bptt_horizon 16`, `num_envs 288`, `num_workers 24`, `total_timesteps 10_000_000_000` (comment: 100B for full game). CARBS sweep support. Hardware not stated. torch.compile on CUDA.
- Env tricks (read `config.yaml` and `environment.py`):
  - **Scripted HM/field-move helpers**: `auto_use_cut`, `auto_teach_cut`, `auto_use_surf`, `auto_teach_surf`, `auto_teach_strength`, `auto_use_strength`, `auto_solve_strength_puzzles`, `auto_pokeflute` (Snorlax), `auto_flash`, `auto_next_elevator_floor`, `auto_remove_all_nonuseful_items` (bag management), `infinite_money`, `skip_safari_zone` (gives Gold Teeth + HM03), `insert_saffron_guard_drinks`, `disable_wild_encounters` (repel), `perfect_ivs`. Cut implementation (`cut_if_next`): read the 20x18 tile map from `wTileMap`, find a cuttable-tree tile (0x3D overworld / 0x50 gym) adjacent to the player, face it, then drive Start -> Pokemon -> party mon -> CUT menu with button macros. Hooks (`pyboy.hook_register`-style, `cut_hook`, `surf_hook`, `pokeflute_hook`) record valid/invalid cut coordinates for reward.
  - **Swarm / state archive**: `swarm: True`, `archive_states`, `early_stop` per event (EVENT_BEAT_BROCK 30, MISTY 300, HM01 600, LT_SURGE 1200 -- thresholds in training iterations), `one_epoch` on EVENT_BEAT_CHAMPION_RIVAL, `required_tolerance`: when an env reaches a new milestone its save state is archived and other envs are reset to it.
  - Exploration reward decays (`DecayWrapper` with per-category forgetting factors: coords 0.9995, npc 0.995, menus 0.998) so revisiting stays worth something.
  - Obs (read `observation_space`): `screen` (2-bit compressed, half res), `visited_mask`, `direction`, `blackout_map_id`, `battle_type`, `map_id`, bag items/quantities, full party species/HP/status/types/level/stats/moves, 320 `events` bits, flags `rival_3`, `game_corner_rocket`, `saffron_guard`, `lapras`.
  - Reward (read `rewards/baseline.py`): 4 x event-flag count (excluding museum ticket), 0.012 per new (tileset, coord), 0.02 per new NPC interaction and per hidden object, 5 per badge, 5 each for Bill-chain/SS-ticket/HM01 events, 4 for Cut taught, 1.0 per valid cut coord/tile, small menu-visit rewards (start/pokemon/stats/bag 0.01-0.1), `rival3` when `wSSAnne2FCurScript == 4`.
- Uses pret `pokered.sym` (shipped in repo) with `pyboy.symbol_lookup("wTileMap")` etc. - **symbol-driven RAM access is the pattern to copy** (for Crystal use pokecrystal.sym).
- Limits: heavy scripting (the authors call the output "a technique for producing solutions to Pokemon", not a general policy); no license.

### 2.4 Pleines et al. 2025 (arXiv 2502.19920) - the only real paper on Whidden-style RL
- Authors: Marco Pleines, Daniel Addis, David Rubinstein, Frank Zimmer, Mike Preuss, Peter Whidden. v1 2025-02-27, v2 2025-03-11, cs.LG. Code: MarcoMeter/neroRL, branch `poke_red` (MIT).
- PPO details read from the paper: gamma 0.997, GAE 0.95, clip 0.2, vf coef 0.5, lr 3e-4 AdamW, 3 epochs x 8 minibatches, no entropy bonus, 32 workers x 2048 steps = 65,536 batch. Obs: gray screen 72x80 x3 frames, 48x48 binary visited crop, party HP/level, event flags (species/stats/moves hidden). Actions: 7 buttons; 8 frames pressed + 16 released = 1 decision / 24 frames (~392 env steps/s on a Ryzen 7 2700X vs ~9,403 raw frames/s).
- Reward: event +2, new coordinate +0.005, heal 2.5 x summed fractional HP gained, level 0.5 x min(sum, (sum-22)/4+22). Step budget starts at 10,240 and grows 2,048 per event completed.
- Scope: starts after Oak's Parcel, ends at Cerulean (~20% of the game). Reached Cerulean ~97-98%; Bill's quest 19% (Fast) / 48% (GRU). Humans: ~11,200 steps to Cerulean; best agent ~18,500.
- Ablation findings (very useful): no navigation reward => zero milestones; nav x10 => agent explores and rarely beats Brock; **Bulbasaur agent farmed Zubat in Mt. Moon via the heal reward and never reached Cerulean (reward hacking)**; dropping the level reward scored best (4.60 vs 4.19 milestones); **no configuration solved the Cut/HM01 barrier**. Compute: one run ~36 h; GRU runs CPU-only ~24 days each; total env steps not reported.

### 2.5 PokeRL (reddheeraj / Mudireddy & Patibandla), arXiv 2604.10812 (April 2026)
- https://github.com/reddheeraj/PokemonRL (4 stars). PPO + SB3 + PyBoy, three curriculum sequences (house exit, reach grass, beat rival), each with its own save state and reward. RAM used for events/termination but not fed to policy (per a secondary summary). Not a serious progress result; relevant for env-hardening patterns (action-loop and button-spam mitigations) **[UNVERIFIED: read via secondary summary; the PDF text was not extractable]**.

### 2.6 xjViper/PokemonCrystalAI - the only Crystal fork of Whidden v1 (read in full)
- https://github.com/xjViper/PokemonCrystalAI; MIT; created 2024-01-08, last push 2024-08-21; 2 stars; Python. "Adapting [PokemonRedExperiments] to play Pokemon Crystal". ROM sha1 `f2f52230b536214ef7c9924f483392993e226cfb` (Crystal 1.1). Ships `pokecrystal11.sym` (58k lines, the full pret symbol file), `flags.md`, `map.md` (maps) and `extra_data.py`.
- Algo/obs/actions identical to Whidden v1 (hnswlib frame-KNN novelty, 36x40 downsample, reward-bar "memory" rows appended to the image). Training script default: 3 envs, 20,480-step episodes, only 3 learn() iterations - a smoke test, not a serious run.
- **Crystal RAM addresses it uses** (all confirmed against the pret symbols, see sec 3.1): badges 0xD857/0xD858 (Johto/Kanto), event flag range 0xDA72-0xDB71, party count 0xDCD7, party species 0xDCD8.., levels 0xDCFE + 0x30*i, HP 0xDD01 + 0x30*i (max HP at +2), x/y 0xDCB8/0xDCB7, map group/number 0xDCB5/0xDCB6, seen-pokemon region.
- Reward quirks: event reward subtracts a hard-coded **125** baseline flag count; `get_badges` sums bit counts of both badge bytes; heal reward only if party size unchanged.
- Limits: no HM handling, no navigation help, no reported progress, the event reward is a raw popcount over all 2048 flag bits minus a hard-coded 125, so noise flags (trainer-seen, sprite toggles) count as progress. The address range itself is correct (NUM_EVENTS = 2048 bits = 256 bytes).
- `flags.md` is a note on adding battle/PP/move rewards (wBattleReward, wPartyMon1PP, etc.).

### 2.7 ThirdStepper/Pokemon_Gold (Gold, pokegold layout) - best-engineered Gen-2 env found
- https://github.com/ThirdStepper/Pokemon_Gold; BSD-2-Clause; created/pushed Nov 2025; 0 stars; "very much a work in progress, dont expect much". PPO via SB3, PyBoy, Python 3.12. Requires `pokegold.sym` (pret `symbols` branch) and pret `event_flags.asm` (parsed by `event_flags.py` so rewards can be keyed by event *name*).
- Config (read): 80x72 gray obs, `FRAME_STACK 4`, `FRAME_SKIP 12` (explicitly replaced a "sticky d-pad" because it created 12 PPO samples per decision), 7 envs, N_STEPS 2048, batch 2048, 4 epochs, lr 3e-4, `TOTAL_TRAINING_STEPS 7M`, optional 8-hour time-boxed training in chunks. Episode cap 10,000 steps.
- **PyBoy symbol hooks** (read `pokemon_gold_env.py`): `hook_register(None, "OpenText"/"CloseText"/"StartBattle"/"ExitBattle"/"IsNPCAtCoord.yes"/"DoPlayerMovement.bump"/"EnterMapWarp"/"EnterMapConnection")` give exact, frame-accurate signals for dialog, battles, bumps into walls/NPCs, and door/warp transitions. Used for collision penalties, door-spam penalties, "A near NPC" rewards, and pausing movement penalties during dialog/battle.
- Reward (read `config.py`): step -0.0001; new tile +0.001; repeated tile -0.0001; bump -0.01 (escalating, capped); new map bank 1.5 / new map number 1.0 / new world tile 0.1; door enter +0.05; A near NPC 0.01; A in dialog 0.005; new species seen 0.1 / caught 1.0; level-up 0.05; **escalating badge rewards [5, 6.5, 8, 9.5, 11, 13, 15, 18]**; one-shot `PLOT_FLAG_REWARDS` dict of Crystal/Gold event names (see sec 3.2): starter from Elm 1.1, bicycle 2.7, **HM01 Cut 3.0, HM02 Fly 3.0, HM03 Surf 3.4, HM04 Strength 3.0, HM05 Flash 2.5, HM06 Whirlpool 3.0**, Slowpoke Well 2.4, Radio Tower 3.3, rival encounters ~0.4-1.5, gym beats 1.1-1.8 as token amounts. Reward clip 20. Anti-stuck modules exist but are disabled by default (`NOVELTY_BONUS_ENABLED=False`, `STUCK_DETECTION_ENABLED=False`).
- HM handling: **none beyond rewarding the HM item events**; there is no scripted Cut/Surf use, no badge gating.
- Results: none published.

### 2.8 gk331251-byte/pokemon-gold-rl
- https://github.com/gk331251-byte/pokemon-gold-rl; no license; created 2025-09-30; 0 stars. PPO via SB3, PyBoy, 8 envs, RAM-only 26-value observation (no pixels), 32 emulator frames/step. README claims ~180 steps/s; `PERFORMANCE.md` says 50-60 steps/s (internally inconsistent). "1M steps in 40-50 min" claim **[UNVERIFIED]**.
- **Address table partially wrong for Gold** (checked against `pret/pokegold` `symbols/pokegold.sym`, see sec 3.3): player x/y, map group/number, badges (0xD57C/D), money, party count, Pokedex ranges are right, but battle flag `0xD057`, `BATTLE_TYPE 0xD05A`, `ENEMY_MON_SPECIES 0xD0F0`, and per-mon HP/stat offsets (`0xDA2C`, `0xDA34`...) do not match the Gold symbols (`wBattleMode` is 0xD116, `wEnemyMonSpecies` 0xD0EF, `wPartyMon1HP` 0xDA4C). Do not reuse.

### 2.9 matiaszanolli/PokemonCrystalRL (Crystal, LLM+RL hybrid)
- https://github.com/matiaszanolli/PokemonCrystalRL; MIT file; 3 stars; Aug-Oct 2025. Hybrid: small Ollama LLM (smollm2:1.7b etc.) gives strategic guidance, curriculum shifts from LLM-heavy to RL; screen analysis (overworld/battle/dialog/menu), 25+ mapped addresses (`core/memory_map.py` - file path not retrievable from API when I checked, so addresses not audited), SQLite decision history, web dashboard. Reward: +100 first Pokemon, +500/badge, +50/level, +10 new map, +0.1 movement. Looks AI-assisted and large relative to its evidence; **no results**.

### 2.10 DhananjayAshok/GameBoyWorlds
- https://github.com/DhananjayAshok/GameBoyWorlds; MIT; 14 stars; created 2025-11-10; active to 2026-09. Gymnasium wrappers + PyBoy for Red, **Crystal**, and fan hacks (Brown, Prism, Fool's Gold, Star Beasts) to test generalization/contamination. Screen-region (pixel) state parsers primarily; an optional `MemoryBasedPokemonCrystalStateParser` for **Crystal v1.1** reads WRAM: Johto badges at 0xD857 (bit order zephyr, hive, plain, fog, mineral, storm, glacier, rising), `wStatusFlags` 0xD84C with **Hall of Fame bit 6**, and `wHallOfFameCount` 0xD95E. Abstract controllers (`openmenu(player)`) hide button sequences from LLM agents.
- Caveat in its docs: Gen-2 screen parsing depends on a specific text *frame* setting; agents must not change options (cursor-on-options check exists for this).
- Useful as a reference for menu macros and for "episode done" = Hall of Fame detection in Crystal.

### 2.11 Edelaar (Leiden, 2025) - excluded as gameplay RL
- "Testing Game Experience Using Deep Reinforcement Learning and Curriculum Learning", Philip Edelaar, supervisors Mike Preuss and Giulio Barbero, dated 2025-08-20 (a 2025-2026 version is also listed). I read the PDF text. It is **not** gameplay: each episode is six discrete steps choosing team members for the eight Johto gyms; battles use fixed heuristics; leveling standardized; PPO (SB3 defaults), 100k steps per gym, curriculum = reuse of previous-gym policy, action masking. Only relevant for gym-leader team data and the observation that curriculum policies show bias toward earlier-gym picks.

### 2.12 PokeAgent Challenge (NeurIPS 2025), Emerald speedrun track
- [arXiv 2603.15563](https://arxiv.org/abs/2603.15563) (v2 2026-03-17; Karten, Grigsby et al.; CC BY-NC-SA 4.0). Game: Pokemon Emerald on mGBA (python bindings per the starter kit). Task: 15 milestones to beat Roxanne; reward +1 per milestone, gamma 1. 22 valid submissions, 6 reached 100%. Winners: 1st Heatz 40:13 (LLM writes subgoal scripts -> imitation distillation -> RL refine), 2nd Hamburg PokeRunners (pure recurrent PPO with milestone-conditioned rewards), 3rd Anthonys 1:29:17 (pure LLM harness), Deepest (fewest steps, 649). Raw frontier VLMs without a harness ~0%.
- Code: https://github.com/heatz123/pokeagent-solution (10 stars, MIT), https://github.com/lastdefiance20/pokeagent-speedrun-DEEPEST (4 stars, MIT), https://github.com/sethkarten/continual-harness (338 stars, MIT). Starter kit is VLM-based.
- Lesson for a Gen-2 project: **LLM-generated scripted policies plus RL refinement beat pure RL and pure LLM on a short speedrun**.

### 2.13 LLM harness runs (not RL, but the only Gen-2 completion)
- Gemini on Crystal: Joel Zhang (ARISE) streamed Gemini 3 Pro vs 2.5 Pro on Pokemon Crystal (~2025-12-15 per reposts); Gemini 3 Pro reportedly earned all 16 badges and beat Elite Four, Champion and Red. Sources are aggregator reposts and an HN thread pointing to blog.jcz.dev; **[UNVERIFIED: original blog not read]**. Token/cost figures (1.88B tokens, ~$22.5k) are a commenter's estimate.
- Gemini 2.5 Pro beat Pokemon Blue (~37 days, May 2025). Claude Plays Pokemon (Red) with memory files and a navigation guide; starter harness at [davidhershey/ClaudePlaysPokemonStarter](https://github.com/davidhershey/ClaudePlaysPokemonStarter) (205 stars, no license).
- Other LLM-on-GB repos (not audited): benchflow-ai/pokemon-gym (100 stars, no license) **[UNVERIFIED content]**; SeanClay10/local-vlm-plays-pokemon (Unlicense; Qwen3-VL + PyBoy, Red); CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark; Orak benchmark ([arXiv 2506.03610](https://arxiv.org/pdf/2506.03610)) scores a Pokemon Red segment by 12 storyline flags.

### 2.14 TAS / speedrun bots
- **ThePokeBot** (Kotaku, 2015-04-21): a bot that reads game RAM, follows an optimized any% glitchless-style route, takes risks and auto-resets when behind pace; best 1:50:14 vs human record 1:49 (Exarion). Author/code not stated in the article **[UNVERIFIED: no repo found]**. Streamed on Twitch.
- TASVideos: Gen-1/2 TAS submissions use BizHawk (Gambatte core) and the pret WRAM labels (Hungarian `w` prefix); a Glitch City Crystal any% route note mentions manipulating `wEventFlags + 236`. I found **no public BizHawk/mGBA bot** for Crystal that reads wEventFlags. BizHawk repos found: 40Cakes/pokebot-bizhawk (126 stars, Gen 3 shiny hunting, not Gen 1/2), pierreportal2/pkred-bot (0 stars, Red BizHawk Lua), lucaslingle/pokemon_red_bots (2019).
- Simonvt1412/pokemon-red-tas-rl (MIT, 0 stars): PPO+PyBoy scaffold for TAS-style research on Red; no results.
- Other Crystal tooling: minormending/crystal-pilot (MIT, 2026): headless auto-pilot for grinding/hunting/catching at ~470x real time (not RL); sergiomele97/Linked_crystal_monorepo (MIT): "Pokemon Crystal MMO using PyBoy".

### 2.15 Kaggle and similar
- I found **no Kaggle notebook/competition for Pokemon Red/GSC gameplay RL**. Kaggle-linked Pokemon efforts are the 2026 Pokemon TCG AI Battle Challenge (card game) and Kaggle Game Arena (chess etc.). The NeurIPS 2025 PokeAgent listing mentions Kaggle Game Arena but I could not confirm a Gen-1/2 task. Other single-person forks (davidpaulius/PokemonRedRL, lilleswing/pokemonredexperiments, JoeTheChode/Pokemon-Yellow-RL, Hrishirt/pokegym for FireRed) add nothing documented.

---

## 3. Pokemon Crystal / Gold / Silver reference data

### 3.1 Crystal WRAM addresses (verified against pret/pokecrystal symbols, both 1.0 `pokecrystal.sym` and 1.1 `pokecrystal11.sym`; the WRAM addresses below are identical in both)
Read by me from the pret `symbols` branch; cross-checked with xjViper's code and GameBoyWorlds' parser. `bank:addr` as in the sym file (WRAM bank 1 for D000+ addresses; PyBoy `memory[addr]` works unbanked for these in normal play).

| Symbol | Address | Note |
|---|---|---|
| wTextboxFlags | 0xCFCF | nonzero when textbox active |
| wOptions | 0xCFCC | text speed / frame (frame matters for pixel parsers) |
| wBattleMode | 0xD22D | 0 none, 1 wild, 2 trainer |
| wBattleType | 0xD230 | |
| wEnemyMonSpecies / Level / HP | 0xD206 / 0xD213 / 0xD216 | |
| wBattleResult | 0xD0EE | |
| wScriptMode / wScriptRunning | 0xD437 / 0xD438 | script (dialog/cutscene) running |
| wPlayerDirection | 0xD4DE | |
| wPlayerMapX / wPlayerMapY | 0xD4E6 / 0xD4E7 | |
| wStatusFlags | 0xD84C | bit 6 = Hall of Fame done (per GameBoyWorlds) |
| wMoney | 0xD84E (3 bytes BE) | |
| wJohtoBadges / wKantoBadges | 0xD857 / 0xD858 | bit0 Zephyr, 1 Hive, 2 Plain, 3 Fog, 4 Mineral, 5 Storm, 6 Glacier, 7 Rising |
| wTMsHMs | 0xD859 | TM/HM ownership |
| wNumKeyItems / wKeyItems | 0xD8BC / 0xD8BD | |
| wPlayerState | 0xD95D | walk/bike/surf |
| wHallOfFameCount | 0xD95E | |
| **wEventFlags** | **0xDA72 .. 0xDB71** (256 bytes, NUM_EVENTS = 2048 bits) | next symbol wCurBox 0xDB72 |
| wBikeFlags | 0xDBF5 | |
| wMapGroup / wMapNumber | 0xDCB5 / 0xDCB6 | |
| wYCoord / wXCoord | 0xDCB7 / 0xDCB8 | |
| wLastSpawnMapGroup / wWarpNumber | 0xDCB2 / 0xDCB4 | |
| wPartyCount | 0xDCD7 | species list at 0xDCD8..DCDD |
| wPartyMon1 (struct stride 0x30) | Species 0xDCDF, Moves 0xDCE1, Level 0xDCFE, Status 0xDCFF, HP 0xDD01 (max HP +2) | |
| wPokedexCaught / Seen | 0xDE99 / 0xDEB9 | 32 bytes each |

xjViper's Crystal code uses exactly these values (0xD857/8, 0xDA72, 0xDCD7, 0xDCB5-8, 0xDCFE, 0xDD01), which is independent corroboration. Not yet validated in a live emulator by me: **[UNVERIFIED in-emulator]**.

### 3.2 Crystal event-flag indices (derived by me from pret/pokecrystal `constants/event_flags.asm`, counting `const` lines and honoring `const_skip`/`const_next`; total comes out to exactly 2048, which matches wEventFlags size, so the counting is internally consistent)
Byte = `0xDA72 + (idx >> 3)`, bit = `idx & 7`.
- EVENT_GOT_A_POKEMON_FROM_ELM idx 26 (byte +3, bit 2). EVENT_GOT_BICYCLE 91 (+11, bit 3).
- **HMs**: HM01 CUT 16 (+2, bit 0), HM02 FLY 17 (bit 1), HM03 SURF 18 (bit 2), HM04 STRENGTH 19 (bit 3), HM05 FLASH 20 (bit 4), HM06 WHIRLPOOL 21 (bit 5), HM07 WATERFALL 1672 (+209, bit 0).
- Gym leaders: BEAT_FALKNER 1213 (+151, bit 5), BUGSY 1214, WHITNEY 1215, MORTY 1216 (+152, bit 0), JASMINE 1217, CHUCK 1218, PRYCE 1219, CLAIR 1220 (+152, bit 4) - note flag order Jasmine before Chuck.
- EVENT_BEAT_ELITE_FOUR 68 (+8, bit 4) (this one is a Kanto/early flag name, check usage before relying on it); EVENT_BEAT_ELITE_4_WILL 1464, KOGA 1465, BRUNO 1466, KAREN 1467, EVENT_BEAT_CHAMPION_LANCE 1468 (+183, bit 4). Easier "game complete" detector: Hall-of-Fame bit/count above.
- EVENT_CLEARED_RADIO_TOWER 33 (+4, bit 1), EVENT_CLEARED_SLOWPOKE_WELL 43 (+5, bit 3). Rival: EVENT_RIVAL_NEW_BARK_TOWN 1097 ... SPROUT_TOWER 1104, BURNED_TOWER 1105.
- Gold/Silver have the same names but different layout/WRAM; `pret/pokegold` `constants/event_flags.asm` is the source (ThirdStepper parses it by name).
- **No public project found that does HM handling in Gen 2.** Game rule (my knowledge, not from the sources above **[UNVERIFIED here]**): in Johto using a field move needs the badge (Cut: Hive, Fly: Storm, Surf: Fog, Strength: Plain, Flash: Zephyr, Whirlpool: Glacier, Waterfall: Rising) *and* a party Pokemon that knows it; HMs are taught from the item. The Red solution (sec 2.3) is the template: macro through Start -> Pokemon -> mon -> field move when the facing tile is cuttable/water, gate on the badge bit.

### 3.3 Gold/Silver WRAM (pokegold layout, checked against pret/pokegold `symbols/pokegold.sym`)
wEventFlags 0xD7B7 (256 bytes), wJohtoBadges 0xD57C, wKantoBadges 0xD57D, wTMsHMs 0xD57E, wStatusFlags 0xD571, wMoney 0xD573, wPlayerState 0xD682, wHallOfFameCount 0xD683, wBikeFlags 0xD93F, wMapGroup 0xDA00, wMapNumber 0xDA01, wYCoord 0xDA02, wXCoord 0xDA03, wPartyCount 0xDA22, wPartyMon1Species 0xDA2A, Level 0xDA49, HP 0xDA4C, wPokedexCaught 0xDBE4, wPokedexSeen 0xDC04, wPlayerMapX/Y 0xD20D/0xD20E, wScriptRunning 0xD15F, wTextboxFlags 0xD19C, wBattleMode 0xD116, wBattleType 0xD119, wEnemyMonSpecies 0xD0EF, Level 0xD0FC, HP 0xD0FF. **These differ from Crystal's addresses**; do not mix them. ThirdStepper's table matches these.
Data Crystal wiki (datacrystal.tcrf.net) has "Pokemon Gold and Silver: RAM map"; I did not read it, the pret symbols are the authoritative source.

---

## 4. Cross-cutting lessons from the Red results

1. **Event-flag popcount is the backbone reward** (Whidden, Rubinstein, xjViper): cheap, dense in the early game, but monotonic and with no sense of order, so the community added explicit milestone events (Bill chain, SS Ticket, HM01) and "required event" gating.
2. **Exploration rewards are mandatory and must be scaled carefully** (Pleines: none => zero progress; x10 => never fights).
3. **Reward hacking is the dominant failure**: heal-reward farming (Zubat in Mt. Moon), level reward distorting priorities, menu-visit rewards, door-spam, Safari Zone step starvation.
4. **HMs/field moves are the wall.** Pure RL did not pass Cut within the paper's budget; Rubinstein's full-game result relies on scripts (auto-Cut/Surf/Strength/Flute/Flash, bag management, item injection). PufferLib's 404M-step Badge-3 run shows the compute cost of learning HM use implicitly.
5. **Swarm/archive training** (reset envs to the furthest-progress save state) and per-event `early_stop` thresholds are what get long-horizon progress.
6. **Throughput design**: 24-frame action repeat, 2-bit/half-res screens, 288 envs, tuned PufferLib vectorization; Gen-2 projects run 3-8 envs (hundreds of steps/s), orders of magnitude below that.
7. Compute scale for Red full-game: config comment "10B steps; 100B for full games" **[reported in config, not measured]**.

---

## 5. Reusable for tensormew

Highest-value, lowest-risk reuses (license-checked):

1. **Crystal address + event-flag tables** (sec 3.1/3.2). Generate from `pret/pokecrystal` `symbols/pokecrystal11.sym` + `constants/event_flags.asm` instead of hand-typing; use bank-1 WRAM addresses as listed. Corroborated by xjViper (MIT) and GameBoyWorlds (MIT). Add an assert at env init that the ROM hash is Crystal 1.0 vs 1.1 (the WRAM table is identical in both, ROM symbols are not, so hooks by symbol name must load the right `.sym`).
2. **PyBoy symbol hooks** (ThirdStepper, BSD-2; pattern also in drubinstein): `hook_register(None, "DoPlayerMovement.bump" / "EnterMapWarp" / "IsNPCAtCoord.yes" / "OpenText" / "StartBattle")` for exact collision/dialog/warp signals instead of pixel heuristics. Confirm each label exists in `pokecrystal11.sym` (names are from pokegold; most are shared with Crystal but verify).
3. **Scripted HM helpers** (pattern from drubinstein `cut_if_next`/`surf_if_attempt`/`teach_hm`; reimplement, **no license so do not copy code**): when facing a cuttable tile / water and the party can use the move and the badge bit (wJohtoBadges) is set, run the Start-menu macro; make each helper a config flag so you can ablate them (their `auto_*` flags) and report with/without scripts honestly.
4. **Reward skeleton**: monotonic max-event count (excluding noise flags), coordinate-count exploration with decay (DecayWrapper style), escalating badge rewards, named one-shot milestone dict keyed by pret event names (ThirdStepper), gym-leader `EVENT_BEAT_*` token rewards, and a "milestone per HM obtained + HM used" ladder. Avoid or cap the heal and level rewards (documented exploit); pause movement penalties during dialog/battle.
5. **Swarm + early-stop training scaffold** (drubinstein): archive a PyBoy save state when a new event flag is reached and restart laggards from it. Re-implement; LSTM policy; ~65k batch.
6. **Evaluation protocol**: Pleines' milestone-completion-rate and steps-to-milestone vs human baseline; GameBoyWorlds' Hall-of-Fame (wStatusFlags bit 6 / wHallOfFameCount) for "game completed" in Crystal.
7. **Credible expectation setting**: only Red has a full-game RL claim; nothing comparable exists for Gen 2. Treat the Falkner/Bugsy/Whitney region as the first realistic RL target; Cut (Ilex Forest, needs Hive badge) and Surf/Strength/Whirlpool/Waterfall are the HM walls.
8. **Do not reuse**: gk331251's address table (wrong battle/HP offsets), xjViper's training script defaults (toy), matiaszanolli's hybrid (unproven), drubinstein code verbatim (no license). PyBoy itself is LGPL-3.0 (GitHub reports NOASSERTION because the LICENSE file is LGPL text plus project notes; **[UNVERIFIED]** whether there are extra terms - check before shipping a derived emulator wrapper).

License summary: Whidden MIT; PufferAI/pokegym MIT; PufferLib MIT; neroRL MIT; xjViper MIT; ThirdStepper BSD-2-Clause; GameBoyWorlds MIT; heatz/DEEPEST/continual-harness MIT; drubinstein/pokemonred_puffer **none**; gk331251 **none**; pret disassemblies show no license in the GitHub API (they are fan disassemblies of copyrighted ROMs - use the symbol files and constants as reference data, do not redistribute ROM-derived assets) **[UNVERIFIED legal status]**; ROMs must be user-supplied everywhere.

---

## 6. Sources
- PWhiddy/PokemonRedExperiments README, `v2/red_gym_env_v2.py`, `v2/baseline_fast_v2.py`, `baselines/run_baseline_parallel_fast.py` (GitHub API, read 2026-10-19)
- drubinstein/pokemonred_puffer README, `config.yaml`, `rewards/baseline.py`, `environment.py`; https://drubinstein.github.io/pokerl/; https://gigazine.net/gsc_news/en/20250310-pokemon-rl-edition
- PufferAI/pokegym README; PufferAI/PufferLib tree listing
- https://arxiv.org/abs/2502.19920 (Pleines et al.); https://arxiv.org/abs/2604.10812 (PokeRL); https://arxiv.org/abs/2603.15563 (PokeAgent)
- xjViper/PokemonCrystalAI (all src files, `pokecrystal11.sym`); ThirdStepper/Pokemon_Gold (`config.py`, `rewards.py`, `pokemon_gold_env.py`); gk331251-byte/pokemon-gold-rl; matiaszanolli/PokemonCrystalRL; DhananjayAshok/GameBoyWorlds (`parsers.py`)
- pret/pokecrystal and pret/pokegold `symbols` branch (`pokecrystal.sym`, `pokecrystal11.sym`, `pokegold.sym`), `constants/event_flags.asm`, `constants/engine_flags.asm`, `ram/wram.asm`
- https://theses.liacs.nl/pdf/2024-2025-EdelaarPPhilip.pdf; https://kotaku.com/the-robots-are-now-coming-for-our-pokemon-world-records-1699225352; HN thread on Gemini 3 Pro vs 2.5 Pro in Crystal (via search, original blog not read); TASVideos / Glitch City pages via search.
