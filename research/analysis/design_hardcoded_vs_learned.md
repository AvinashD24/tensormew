# Design: what is and is not hard-coded now

Context: the previous audit found the agent was hand-fed (route oracle, scripted battles, scripted goals), so reward
curves said little about learning. This note records the state after the "no oracle, learned battles" change
(`long_v2`, observation schema `v2-no-oracle-learned-battles`).

## Removed

| Was | Now |
| --- | --- |
| `route_features`: stage one-hot + first key on a BFS shortest path to the current goal + distance to it (9 floats in the observation) | gone. `route_graph` has no `route_features`, `current_stage` or `next_direction` |
| `quest_goal_vector` (4 goal tokens), `story_objective_maps` (2x tile bonus on the "right" maps), `navigation_potential` (hand-coded per-map potentials), gym waypoints | gone |
| goals egg -> Elm -> Falkner (743 steps), bonuses for the egg, Elm, the rival, Togepi, HM01, 2.0 per unnamed event flag | minimal story path (319 steps), see below; those bonuses are gone; unnamed flags pay 0.05 from the explore budget |
| `tactical_battle_step` in the trainer / evaluator / play_live (the policy never fought) | the policy presses every button in battle |
| sticky actions everywhere | never in battle (a repeated direction would corrupt menu navigation) |
| 5 actions (no B) | 6 actions: up, down, left, right, A, B |
| `play_live` scripted navigators (egg return, Falkner nav, gym prep, compass, post-egg policy) | removed; only the reflexive unstick / Cut / LLM-director helpers remain |

## The story path (what is truly required)

Only taking a starter from Elm is a hard gate, and the start state already has Cyndaquil. From the New Bark start tile
the shortest walk to the tile in front of Falkner is 319 steps (`route_graph`, BFS over ROM collision data) with no egg,
Elm or rival detour. Milestones (`src/story_path.py`), each paid once per episode on first arrival:

| Milestone | Reward |
| --- | --- |
| Route 29 (24,3) | 5 |
| Cherrygrove City (26,3) | 10 |
| Route 30 (26,1) | 10 |
| Route 31 (26,2) | 10 |
| Violet gatehouse (26,11) | 5 |
| Violet City (10,5) | 15 |
| Violet Gym (10,7) | 15 |
| Zephyr Badge | 60, plus 10 for `EVENT_BEAT_FALKNER` |

Total 140. Extending is one `Milestone(...)` line (next gate: Route 32 -> Azalea); Sprout Tower is optional and unrewarded.

## Observation (`DEEP_OBS_DIM` = 77)

17 base (4 neighbour tiles, position, badges, battle and dialogue flags, visited neighbours, party size, max level, dex
count, fraction of milestones done) + 10 story (8 milestone-done bits, map group, map number) + 50 battle features
(`src/battle_obs.py`): in-battle, wild / trainer, enemy species / level / HP / status, active mon species / level / HP /
status / slot, party HP and level (6 slots), the 4 move ids, PP and usable flags, battle-menu cursor, move cursor, raw menu
cursor x / y, number of healing items and Poke Balls. Nothing in it is directional or a distance. Addresses come from
`rom/pokemon_crystal.sym` and were checked on the live emulator (menu cursor reads 1..4 for FIGHT / PKMN / PACK / RUN, PP
with the PP-Up bits masked, enemy HP / level / species). Battle WRAM holds junk outside a battle, so those fields are
zeroed unless `wBattleMode != 0`. The legacy single-process `CrystalEnv` observation (`OBS_DIM` = 121) carries the same
story and battle blocks.

## Rewards (all generic)

- every step -0.01; wall bump -0.05 x streak; low-HP pressure -0.01 per step below 25% lead HP.
- exploration (one shared 30-point per-episode budget): new tile 0.1, new off-path map 0.5, first warp 1.0, first A-press
  script 0.5, count novelty 0.08 / sqrt(visits), tiny straight-line momentum, unnamed event flag 0.05.
- battle (budget 20): damage dealt 1.5 x fraction of the enemy's HP, KO 1.0, win 5 (trainer) / 3 (first three wild) then 0.2.
- survival: HP lost -2 x fraction of party HP, ally faint -1, white-out -15.
- growth: level-ups (budget 15), catches 3 then halving (budget 6, deliberately small), Pokecenter heal (budget 15), potion heal 3 x fraction.
- no term mentions an egg, Elm, the rival or a direction. `components` keys: step, bump, route (only with `goal_potential`),
  tile, neighbor, map, milestone, warp, interaction, event, badge, catch, level, battle, damage, heal, whiteout, behind_map,
  other (+ curiosity, loop, momentum from the worker). `milestone` is new; the dashboard shows unknown keys by name.

## Still hard-coded, on purpose

- Start state outside New Bark with Cyndaquil (the user's choice).
- 10 Poke Balls and 5 Potions are written into the bag at every reset (`equip_supplies`), so catching and item use can be learned from episode one.
- Overworld dialogue is auto-advanced with `mash_text` (80 frames per step): signs, NPC chatter, cutscenes, Nurse Joy's heal prompt. Battle text and every battle menu are policy-controlled.
- The reward terms and their budgets; the 6-button action set (no START / SELECT: the START menu is never needed for the badge, and Potions are used from the in-battle PACK menu; healing outside battle is Nurse Joy via A).
- The reflexive "escape" after 4 blocked moves (a stuck-wall reflex, not directional) and 0-6 random start moves (jitter).
- The go-explore style swarm archive (the first env to enter a new story map saves a state; 30% of resets load one).
- `long_v3` adds three self-generated training aids (each behind its own config flag; no hand-made states, the only start is still `new_bark_outside`):
  - staggered episodes (`stagger_episodes`, `episode_len_jitter`): a random per-env initial step offset and +-25% episode length, so the 24-32 envs no longer reset in the same update (that lockstep produced an exact 8-update reward cycle); logged as `phase_spread`, with `reward_8` (mean over 8 updates) as a cycle-proof trend;
  - a battle archive (`battle_archive*`): states saved when any env's battle flag goes 0 -> non-zero (rate limited, at most 32, keyed by kind / map / enemy species), loaded on 15% of resets, plus an A-button logit bias (0.5 -> 0 over 150 updates) applied only while the in-battle flag is set. The bias lives inside the policy (`SpatialVisionPolicy.battle_a_bias`, used by `forward`, `act` and both `evaluate_actions*`), so rollout and PPO update see the same distribution; evaluation loads with the bias at 0;
  - furthest-map resets (`swarm_prob` 0.5, `swarm_furthest_weighting`): swarm-archive states are sampled with a weight that grows with the story stage (furthest stage reached = 1, floor 0.15 for earlier stages).
  - a dwell penalty (`dwell_penalty`, `src/dwell.py`) against farming one bush: live data showed 56% of visits on Route 29, thousands of wild battles and no trainer battles. A sliding window of the last 192 non-battle, non-dialogue steps is "stuck" when it covers at most 16 tiles and nothing new happened in it (new tile / map / story milestone / new species); the cost ramps 0.005 -> 0.03 per step, is capped at 15 per episode and resets when the agent leaves the area. Battle steps neither count nor reset it, so walk-battle-walk loops keep the counter. It is not a ban on grinding: `dwell_level_targets` gives the party level wanted per furthest story stage (8 before Cherrygrove, 10 on Cherrygrove / Routes 30-31, 11 at the Violet gate, 12 in Violet City / the Gym; Falkner has Pidgey L7 and Pidgeotto L9), and while the highest party level is below it and the agent is really fighting (a battle ended or a level-up within 512 steps) the dwell cost is zero. The older per-tile `loop` term (episode-wide revisit counts of single tiles, battle steps included in the counts) did punish grinding, so it is switched off under the same exemption; outside it, `loop` and `dwell` both apply (different signals: tile revisit counts vs. area without progress).
- `long_v4` adds a dense path-progress reward (`path_progress`, `src/path_progress.py`). This is reward shaping, not an observation oracle: the policy is told nothing (no direction, no distance, no new input), it only gets paid when it has actually walked further along the story path than ever before in the episode. `progress(position) = 319 - BFS steps to Falkner` over the ROM map graph (`route_graph`, warps and ledges included), so off-path tiles, dead-end corridors and buildings count as the path tile they hang off minus the walk back. The reward is `0.08 * max(0, progress - best_progress)` with `best_progress` rising monotonically per episode, initialised from the real start state (an archive state mid-route pays nothing for what it already has). Because only a new best pays, the payments telescope to `0.08 * (final best - start progress)` <= 25.5: stepping back costs nothing and pays nothing, re-walking pays nothing until the old best is passed, so oscillating, circling or entering and leaving buildings cannot be farmed, and there is no sign flip to create an oscillation incentive (the older `goal_potential` potential difference was symmetric, so back-steps cost and the agent could be punished for exploring a side path). White-out teleports pay nothing. A new best also resets the dwell window. Cost of the design: the reward still encodes the route's geometry (it is the same BFS distance), so it is a hand-made shaping signal, kept separate and ablatable; milestone bonuses are unchanged and still ~5x the path total.
- Optional, OFF by default in every preset: `goal_potential` (BFS potential toward Falkner). It is an oracle, kept for ablations only.

## long_v4 learning-speed techniques: literature support and status

None of these adds a scripted path, a demonstration or a direction / distance feature to the observation. "Guess" means the
literature does not settle the value for this game; the flag exists so it can be ablated.

| Technique | Literature support | Status in `long_v4` |
|---|---|---|
| Milestone success rates (diagnostic) | Pleines et al. 2025 (arXiv 2502.19920) judge Pokemon Red agents by the fraction of milestones completed, not by return | Added. Window and cumulative reach rate per milestone among episodes that started before it, share of envs at or above, best path progress |
| gamma 0.997 -> 0.999 | RND (Burda et al. 2018, arXiv 1810.12894): raising gamma_E from 0.99 to 0.999 "greatly improves performance"; OpenAI Five (2019, arXiv 1912.06680): gamma 0.9993 (180 s horizon) and longer horizons kept helping; Andrychowicz et al. 2020 (arXiv 2006.05990): gamma is one of the most important hyperparameters, tune it. Against: Pleines 2025 tuned gamma over {0.99, 0.997, 0.999, 0.9995} for Pokemon Red and used 0.997 | Set to 0.999 on the numbers: the path is 319 steps but episodes are 2048 steps, and 0.997**1000 = 0.05 hides the last milestones. **Partly a guess** for this game |
| gae_lambda 0.95, rollout 256, GRU state carried, BPTT = whole rollout | OpenAI Five and Pleines 2025 both use lambda 0.95 (Andrychowicz recommends 0.9 for continuous control). Pleines found 2048-step worker rollouts better than 512 / 1024; OpenAI Five unrolls the LSTM 16 steps with the state carried | Unchanged. The trainer stores `initial_hx` per rollout, carries it across rollouts and resets it at episode ends; a 256-step BPTT window is far above the usual 16-128. A longer rollout is the lever Pleines supports but it changes the sample budget, so it is not changed here |
| Exploration budget 30 -> 12, intrinsic weight 1 -> 0.4 | RND: intrinsic reward normalized by a running std and mixed with a fixed extrinsic:intrinsic coefficient of 2:1, gamma_I 0.99; Pleines 2025: the navigation reward is essential but scaled up 10x the agent "rarely defeats" Brock. Live v3-1: 30 of 42.5 positive reward per episode was exploration, milestones 1.8 | Budget and linear decay are engineering choices, **no paper gives these numbers (guess)**; no episodic / non-episodic mixing or reward normalization of the intrinsic term was added |
| Heal reward for HP restored | Pleines 2025 pay `2.5 * sum(delta HP / max HP)` for level-ups, Pokecenter and potions and report that agents exploit it (repeated healing at the mother or Pokecenter, 8-9 blackouts) and that removing it reduces performance | Concave in the HP fraction (u(f) = 1 - (1 - f)^2), mean over party slots, none for level-ups / white-out recovery / map changes, budget 8 per episode. The weighting toward low HP is **a guess** |
| Low-HP battle practice | Self-generated state archives are Go-Explore-style restarts (Ecoffet et al. 2019, not re-read for this change); weighting by "needs healing" has no direct citation | Weight 3 on archive states with lead HP <= 40% and a heal item. **Guess** |
| Auxiliary prediction heads | UNREAL (Jaderberg et al. 2016, arXiv 1611.05397): auxiliary tasks gave 880% human-normalized Atari score and a 10x mean learning speed-up on Labyrinth; Lample and Chaplot 2016 (arXiv 1609.05521): predicting game features "dramatically improves" training speed | Map id, lead-HP change and battle start 8 steps ahead, weight 0.05. Evidence is for A3C / DQN agents, not PPO + GRU; the weight and horizon are **guesses** |
| Entropy 0.04 -> 0.01 becomes 0.01 -> 0.001 | OpenAI Five: 0.01 default, annealed to 0.001 in the long Rerun run (Figure 28: 0 still learns, 0.01 best early); Pleines 2025 use no entropy bonus (1e-3 .. 1e-6 gave no gain); Andrychowicz 2020: no evidence that entropy regularization helps when the action head starts small (it does here, gain 0.01). PPO paper Atari value 0.01 (recalled, not re-fetched) | Live v3-1 entropy 1.70 of 1.79 with approx_kl 0.006 and clip 0.06 means the policy is still near uniform, so the 0.04 start is not buying useful exploration |

## Open risks

- Menu navigation in battle has to be discovered from 6 buttons. A random policy does reach FIGHT -> first move by pressing A, but switching, PACK and items are rarer. `long_v3` adds self-generated battle practice (battle archive states plus a decaying A-bias in battle); `long_v2` has neither.
- Battle steps cost frames: a random policy spends about 40% of its steps in battle on Route 30.
- A level-5 Cyndaquil against Falkner needs grinding or healing; damage-dealt and level-up rewards are what has to teach that.
- The dashboard story panel (`tools/dashboard_static`, `tools/build_route_map.py`) still draws the old five egg-errand goals.
