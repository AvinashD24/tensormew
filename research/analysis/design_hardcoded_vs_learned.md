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
- Optional, OFF by default in every preset: `goal_potential` (BFS potential toward Falkner). It is an oracle, kept for ablations only.

## Open risks

- Menu navigation in battle has to be discovered from 6 buttons. A random policy does reach FIGHT -> first move by pressing A, but switching, PACK and items are rarer. Not implemented: start-in-battle curriculum states, or an A-bias in battle early on.
- Battle steps cost frames: a random policy spends about 40% of its steps in battle on Route 30.
- A level-5 Cyndaquil against Falkner needs grinding or healing; damage-dealt and level-up rewards are what has to teach that.
- The dashboard story panel (`tools/dashboard_static`, `tools/build_route_map.py`) still draws the old five egg-errand goals.
