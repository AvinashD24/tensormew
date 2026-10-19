# Reading note: "Pokemon Red via Reinforcement Learning" (+ the project that actually beat the game)

Compiled 2026-10-19. Every claim is tagged by source; anything not confirmed is marked **[UNVERIFIED]**.

## 0. Premise check (read this first)

The task premise ("NeurIPS 2025 paper on beating Pokemon Red with RL") does **not** hold for the paper with arXiv 2502.19920. Three separate things get conflated:

| Thing | What it is | Beat the full game? | Venue |
|---|---|---|---|
| **Pleines et al., arXiv 2502.19920** "Pokemon Red via Reinforcement Learning" | 8-page academic baseline paper. Authors: Marco Pleines, Daniel Addis, David Rubinstein, Frank Zimmer, Mike Preuss, Peter Whidden. v1 (27 Feb 2025) was titled "Playing Pokemon Red via Deep Reinforcement Learning"; v2 11 Mar 2025. | **No.** Scope is explicitly cut to "up to the end of Cerulean City" (~20% of the game). Agent reaches Cerulean; almost never finishes it. | arXiv comment says "under review". A later citing paper lists it as **2025 IEEE Conference on Games (CoG), pp. 1-8** (secondary source only; I did not see the IEEE page). **No evidence of NeurIPS 2025** in arXiv, search, or the NeurIPS competition paper. |
| **"Pokemon RL Edition" / pokerl book + `pokemonred_puffer`** (Rubinstein, Donovan, Addis, Choe, Suarez, Whidden) | Online write-up (v1 changelog 4 Mar 2025) and repo. This is the project that claims the full game was beaten by a <10M-parameter RL policy. | **Yes, with heavy caveats** (scripts, swarming, map-ID boosts, infinite money; not yet stable with all scripts off). Not peer reviewed. | Web book + GitHub; its "Future" page promises an arXiv draft "soon" **[no such paper found]**. |
| **NeurIPS 2025 PokeAgent Challenge** (arXiv 2603.15563) | The NeurIPS 2025 competition paper. Tracks: Showdown battling (Gen 1 OU, Gen 9 OU) and a **Speedrunning track on Pokemon Emerald** (reach first gym, 15 milestones). | N/A: Emerald, first gym only. | NeurIPS 2025 competition track. Mentions Red only in passing (GPT-5 "finished Pokemon Red in 6,470 steps" with a caveat on incomparable step counts). |

So the only "NeurIPS 2025" Pokemon paper is PokeAgent (Emerald + battling), and the only "beat Pokemon Red with RL" claim is the pokerl project, which is not the 2502.19920 paper. The note below covers both Red sources, keeping them clearly separated. Section A = the paper. Section B = the pokerl project. Section C = PokeAgent RL-relevant bits.

Secondary-source warning: PokeRL (arXiv 2604.10812, Texas A&M) says Pleines et al. used "a reward function with more than 25 components" and "<10M parameters". That is wrong for the paper (it has 4 reward terms, ~2M params / 4M with GRU). The ">25 parameters" and "<10M" figures belong to the pokerl project. Do not trust second-hand summaries.

---

# A. Pleines et al. (arXiv 2502.19920)

## A1. Exact claims
- DRL (PPO) agent "successfully completes the first several hours of Pokemon Red", specifically "up to Cerulean City".
- Ablations "reveal vulnerabilities in reward shaping" (agents exploit heal reward etc.).
- Argues long-form games give long-term-coherence challenges absent from Atari; proposes Pokemon as a testbed for LLM agents, hierarchical RL, exploration.
- Claims "to our knowledge, no peer-reviewed publications have explored training agents on Pokemon Red".
- Code/data/pretrained agent "available at (REDACTED FOR REVIEW)" in v2. Secondary source says code is MarcoMeter/neroRL branch `poke_red`; I could not confirm contents of that branch from the page (README is generic neroRL). **[UNVERIFIED]** that it contains the full env/replay data.

## A2. Task scope and start state
- Milestones counted: Viridian Forest, Brock (badge 1), Mt. Moon, Cerulean City, Misty (badge 2), Bill's quest, "Cerulean Done" (defeat the Rocket thief). Table II tracks 7 milestones. Vermilion unlock needs Bill's quest (Misty is skippable).
- Env starts **after the Oak's Parcel return** (they skip the backtracking). The agent is **given Squirtle** in the baseline (Charmander / Bulbasaur as variants; a "Choose Starter" variant starts at game start).
- Sim speed: 9,403 raw SPS on a Ryzen 7 2700X; with 24 frames per decision ~392 decisions/s per core.

## A3. Environment design
- **Emulator / access:** Game Boy emulator with RAM access (they note "additional engineering effort is needed to extract relevant information from the Game Boy emulator's RAM"). Game state vector and event flags come from RAM; the visuals are pixels.
- **Observation (hybrid pixels + RAM, deliberately human-limited):**
  - Grayscale screen downsampled x2 to **72x80**, **3-frame stack**.
  - A separate **48x48 binary visited-coordinates crop** centred on the player.
  - Game-state vector: HP and level of each party member + **event-completion flags**. No species, stats, or movesets.
- **Action space:** discrete 7 = A, B, Start, Up, Down, Left, Right (Select omitted).
- **Frame skip / timing:** each press held 8 frames then released 16 frames => one decision per **24 frames**; reliably moves exactly one tile. Authors note this is coarse for dialogs and battles (a dialog costs >=10 steps; evolution ~15; pressing B cancels evolution).
- **Determinism:** GB RNG depends on input history, so deterministic play => deterministic env; they suggest random button presses on reset to counter this. They do not address generalisation (fixed start state).
- **Terminal condition:** time only. Budget starts at **10,240 steps** and grows **+2,048 per completed event** (dynamic episode length), so workers desync in their resets, which fights catastrophic forgetting.

## A4. Reward function (only four terms, all summed)
`R = R_event + R_nav + R_heal + R_lvl`
- **R_event = +2** per completed event (trainer battle, quest step). Described as "breadcrumbs".
- **R_nav = +0.005** per new overworld coordinate visited in an episode.
- **R_heal = 2.5 * sum over 6 party slots of (HP_after - HP_before) / HP_max_i**, triggered on level-up, Pokemon Center use, or potion use on a non-full-HP Pokemon (equation text in the PDF extraction was garbled; this is my reconstruction **[exact form UNVERIFIED]**).
- **R_lvl = 0.5 * ( min(sum of 6 levels, 22) + (sum of levels - 22)/4 above the threshold )** (again reconstructed from garbled equation **[exact form UNVERIFIED]**). Threshold 22 ~ level to fight Misty; marginal gain beyond it divided by 4, "discouraging wild-encounter grinding".
- No penalties. No badge-specific term (badges come only via event flags).

## A5. Algorithm and compute
- **PPO** (clipped surrogate + clipped value loss, no entropy bonus: they tried 1e-3..1e-6 with no gain).
- Network: two Nature-CNNs (screen, binary visited image), one FC for state vector; concat -> projection -> body (FC, or **GRU** for the recurrent variant) -> policy head + value head. **~2M params, 4M with GRU.** ReLU, AdamW.
- Hyperparameters (Table I): gamma 0.997 (tuned over {0.99, 0.997, 0.999, 0.9995}); GAE lambda 0.95; **32 workers x 2048 worker steps = 65,536 batch**; 3 epochs; 8 minibatches; clip 0.2; value coef 0.5; LR 3e-4; max grad norm 0.5. 2048-step horizon beat 512 / 1024. Effective horizon ~333 steps (1/(1-0.997)).
- **Training budget:** curves run to ~400M env steps (axis in Fig 3, read off the plot **[approximate]**). A single run ~**36 h** on their cluster (PC2 Paderborn + TU Dortmund Linux-HPC). **GRU runs were CPU-only, ~24 days per run** because 2048-step BPTT blows up VRAM.
- Protocol: 5 seeds per config; evaluation at 25 time points x 30 episodes; 30 human playthroughs by skilled players as reference.
- PufferLib is thanked for RL expertise in the acknowledgements; it is not described as the training framework.

## A6. Results (Table II/III, mean +/- std over 5 seeds)
- **Baseline (Squirtle):** Brock 0.99, Mt. Moon 0.97, Cerulean 0.85, Misty 0.27, Bill 0.11, Cerulean Done 0.00 (about 4.2 of 7 milestones). Humans: 7/7, Cerulean in ~11,000 steps vs ~21,000-25,000 for agents.
- **Fast** (faster text speed, battle animations off): best of the "full reward" runs, 4.4 milestones, Brock 0.98, Cerulean 0.93, Misty 0.33, Bill 0.19, **reaches Cerulean in ~18,500 steps**.
- **GRU:** worse early, better on Bill's quest: 48% completed Bill's quest vs 49% reaching Cerulean (the memory helps with the "three interactions in order, reset if you leave the house" quest). Baseline-like only otherwise. Fast: 71% visited Bill's house but only 19% finished the quest.
- **Best ablation = remove level reward (-R_lvl): 4.60 milestones**, exceeding the baseline (4.19). Authors' hypothesis: more focus on event reward; most events are XP-rich trainer battles.
- **Starter effect:** Charmander: Misty 0.02 (water type counters it). Bulbasaur: fails Cerulean entirely due to heal-reward exploit.
- **Only "Cerulean Done" completions:** human 1.00; -R_lvl 0.03; Fast 0.00 etc.; a couple of outliers (GRU 0.21 +/- 0.3 on Fast-Rheal, i.e. very few seeds). No config reliably finishes Cerulean.

## A7. Ablations and failure modes (the paper's main scientific content)
1. **No navigation reward => 0 milestones** (agent sits near Pallet Town). **10x navigation reward => almost never beats Brock** (pure exploration, avoids battles and menus to save steps). Exploration reward is knife-edge.
2. **Heal-reward exploitation:** Bulbasaur learns to farm Mt. Moon Zubat (Leech Seed vs Leech Life) for endless HP deltas, or battles near Pallet Town and heals via the player's mother; one of 5 seeds in several configs. `Num. Heals` reaches ~399 (Bulbasaur) vs ~13 (baseline). Removing heal reward lowers performance generally because the Pokemon Center stops being used as respawn point (8-9 blackouts, long retrace). Bulbasaur exception.
3. **Time bias in discounted return:** in the Choose Starter run the agent always picks Charmander (fewer button presses to the first event/level reward), though Squirtle is best vs Brock; the consequence is >3,000 steps away, far beyond the 333-step effective horizon. Credit assignment across the starter choice is hopeless without hierarchy.
4. **Skip-the-gym behavior:** agents often skip Misty and go for Vermilion; critical events may need stronger feedback.
5. **Catastrophic forgetting / instability:** fixed episode length syncs worker resets; dynamic length helps but stds are huge.
6. **Action inefficiency:** agents waste steps in dialogs/evolution; fixed 24-frame action granularity is coarse.
7. **Nobody reaches Vermilion's gym:** Cut (HM01) needs a Cut-capable Pokemon, the item, menu navigation, and the right placement; none of the reward variants cracks it. Authors say a further reward signal would be needed, with more exploit risk.

## A8. What the agent could NOT do (and why, per the authors)
- **HMs / Cut tree at Vermilion:** four-part chain (get Pokemon that can learn Cut, obtain HM01, teach via menus, use at tree) with no intermediate reward and no prior need. Unsolved.
- **Surf, Safari Zone, Silph Co, Victory Road, Elite Four:** out of scope; never attempted in the paper (scope ends at Cerulean). The paper does not mention them beyond a sentence on Gym 3 needing Cut and a hidden button puzzle (via Anthropic's Claude post).
- **Battle strategy / item and move analysis:** explicitly out of scope ("analyzing item usage and move selection falls outside the scope"). Species/moves are not even observed.
- **Generalisation:** not addressed; fixed start state; ROM randomisers proposed.
- **Bill's quest:** order-sensitive, resets on leaving; most agents fail.

## A9. Limitations stated by the authors
- Needs a legal ROM and RAM-extraction engineering.
- Evaluation as expensive as training (~36 h); GRU infeasible on GPU (24 days CPU per run).
- Strategy space too large to analyse; thousands of episodes not manually reviewed.
- Reward shaping risk grows with each new fix (curse of dimensionality).

## A10. Open questions / future work listed
- Curiosity (ICM) / RND to replace the naive nav reward and the visited-mask observation.
- Hierarchical RL (Director) so "choose starter" is a high-level action; "timing as an action" (when to act vs observe) to save wasted dialog steps.
- Curriculum learning / engineered rewards for Cut and puzzles.
- ROM randomisers for robustness.
- Voyager-style code-generating agent; LLM agents (related work notes Claude targeting badge 3, nunu.ai Emerald, PokeLLMon for battles).

## A11. Reproducibility
- Positive: all hyperparameters in Table I, five seeds, std reported, human baselines.
- Negative: repo link redacted in v2; no wall-clock per config beyond "36 h"; GRU infeasible without CPU farm; reward equations are the sole precise reward spec but figures (Fig 3) lack std bands.

---

# B. The project that actually "beats Pokemon Red": pokerl / `pokemonred_puffer`

Sources: web book at drubinstein.github.io/pokerl (v1 4 Mar 2025), repo `drubinstein/pokemonred_puffer`, GIGAZINE coverage. Not peer reviewed. Authors: David Rubinstein, Keelan Donovan, Daniel Addis, Kyoung Whan Choe, Joseph Suarez, Peter Whidden.

## B1. Claims, verbatim-ish
- "As of February 2025 ... can beat the game with RL using a <10 million parameter policy ... with minimal simplifications" (about 60,500x smaller than DeepSeek-V3).
- "The output is not a policy capable of beating Pokemon, but a technique for producing solutions to Pokemon."
- Results page: "an agent that beats Pokemon **with a few caveats**"; winning runs seen with each script removed individually, but not stably with all scripts off ("a couple of bugs that sometimes occur").
- Goal is "champion", not catch-all-151.
- Whether the Elite Four / Champion fight itself is shown with a timing: the Results and Metrics pages I fetched give **no Elite Four details, no total training-to-win hours, no env count** (**[UNVERIFIED]** beyond "beats the game"). One scripted component is "Blocking the Indigo Plateau exit at game end", consistent with the game being cleared at the Hall of Fame.

## B2. Env design
- PyBoy + Gymnasium; PRET disassembly for RAM addresses. **Action frequency 24 ticks, button release delay 8 ticks** (same as the paper). Screen 144x160 -> grayscale -> 72x80, bit-packed to 2 bits/pixel (buckets 0/85/153/255). Event flags bit-packed and decoded on the GPU.
- **Observation** (explicit design rule: nothing a human wouldn't know): 72x80 screen + 72x80 visited mask (per mini-episode); 2560-bool event vector (320 RAM bytes + 4 custom events: Rival 3, Lapras, Saffron guard drink, Game Corner Rocket); facing direction; map ID and blackout-map ID (embeddings); 20 items + quantities; party 6x11 (species, 2 type, moves embeddings); battle type; Safari steps left (/502).
- **Mini-episodes:** emulator keeps running; exploration/seen-set maps reset periodically (jittered `full_reset_frequency`); game itself is not reset. Full episode ends on soft-lock (e.g. out of money) or game completion.
- **Start:** after Parcel Delivery, **Bulbasaur or Charmander** starter (so something can learn Cut); gift **Lapras guaranteed** at Silph Co.

## B3. Reward (>25 terms; values not published on the page)
Terms (coefficients **[UNVERIFIED]**): exploration (new coords, treated broadly as "any interaction" because pure coordinate exploration never fought Brock), warps, hidden objects, signs, A-press (dropped: little gain), valid/invalid Cut / Surf / Pokeflute coordinate counts, Cut/Surf/Pokeflute tile counts, `use_surf`, ~**80 required events**, required items (HMs), useful optional items (Bicycle), badges, HM count, seen/caught Pokemon, obtained move IDs, level, Pokemon Center heal, ball use, menu-opened terms, special events (Rival 3, Game Corner Rocket, Saffron guard, Lapras), and **Safari Zone reward proportional to steps left** at the highest Safari map reached.
- **Level reward** (from Whidden's video): L = sum of 6 levels; L<15: reward L; L>=15: 30 + (L-15)/4. Removing it "caused failed experiments".
- **Map-ID boosts** ("regretfully"): coordinate-exploration reward is boosted on maps where an objective is pending (gyms until beaten; Rocket Hideout etc.), with exceptions (Route 23 not until all gyms; Lavender Tower not until Giovanni + Silph Scope, else agents farmed wild levels). The authors say this may put the game "on rails" and that an agent that could read text would not need it.
- Lessons: sparse rewards alone keep the agent in Pallet Town; dense reward too strong ignores objectives; 10-20 agents reached only ~10% of map-ID coverage.

## B4. Algorithm, architecture, compute
- **PPO** via PufferLib-style async vectorised envs (CleanRL-derived `cleanrl_puffer.py`).
- **Policy ~5M params (~20 MB)**: CNN (32@8s2, 64@4s2, 64@3s2) on 2-channel screen+mask, embeddings for map/items/species/types/moves, linear 512 + ReLU, **LSTM(128, 1 layer)**. LSTM chosen over frame stacking / 3D CNN / attention / SSMs for VRAM and speed.
- **Hardware:** 4 donated machines (RTX 4090, i9-14900K, 128 GB RAM each); previously ~$200/month on vast.ai. Throughput up 10x to ~10,000 SPS; emulator dominates. **288 agents** shown in metrics page (so ~288 envs **[inferred from metrics page, not stated as training config]**).
- Training time: "between 7 hours and 1 week" per run per env page; one scripted run reduced from 1 day to **7 hours** after a month of **CARBS** (Imbue) sweeps optimising reward vs time. Beating Brock in <30 min; with wild battles disabled, badge 1 in ~10 min.

## B5. Save-state / curriculum tricks ("The Swarm")
- Inspired by Go-Explore. Save states recorded whenever any agent hits a required objective; **when any agent completes a required objective, all agents load that agent's save state** ("maintains temporal and spatial closeness"). Rather than many independent short episodes, it runs "multiple very long episodes of cooperative environments". Other trigger heuristics were worse. The save-state tracker resets when the game is won. Authors note it may limit generalisation and say they would like to remove it.
- The repo has a `pyboy_states/` dir (existence only; use not documented there).

## B6. Hand-engineering that remains (this is what makes the "beat the game" claim qualified)
Scripts **still in use**: toss non-key items when bag full; **infinite money**; teach + solve **STRENGTH** boulder puzzles; teach + use **SURF**; auto-use **Pokeflute**; block the Indigo Plateau exit; use **FLASH**.
Scripts **removed over time**: teaching + using **CUT**; maxing Pokemon stats; auto-inserting drinks in Celadon; automated elevators; disabled wild battles. Also **no battling policy** ("we ignored creating a policy for battling ... providing suitable conditions for leveling up"; grinding wins except in rare cases). Mitigated risks: losing vital Pokemon, full bag, no money at Safari, only non-damaging moves.

## B7. Per-obstacle notes (what they say, not how solved)
- **Cut (HM01):** via SS Anne captain; unlocks 2 gyms and the route east of Cerulean. **Surf (HM03):** Safari Zone Warden, within a **500-step limit**; needed for Blaine and the Elite Four route. **Strength (HM04):** Gold Teeth in Safari -> Fuchsia old man; needed for Victory Road. Fly/Flash optional. Teaching/using any HM needs >= **17 actions**. **Pokeflute** for Snorlax. Safari: linear route to a fork ~75% in; step-proportional reward; "eventually complete it, thousands of training steps".
- **Silph Co / Rocket Hideout / Game Corner / Pokemon Tower / Saffron / Sabrina teleporters / Giovanni spin tiles:** listed as "brute-forceable" given strong exploration plus event rewards; the book says these quests need either exploration of the whole world or text-reading, and they chose exploration.
- **Victory Road / Elite Four:** not covered on the pages I could read.

## B8. Observed failure modes (metrics page)
- Little exploration in the Safari Zone; exploit of hidden-object reward via gambling; agents avoid battles to protect within-episode exploration reward; mis-weighted tree-cut reward -> loops of tree cutting; heatmap showed with wild battles off the fastest Rock Tunnel route emerges. Pokecenter usage plots used to tune heal reward.

## B9. Stated future work / limits
Remove scripts and swarm; stronger exploration methods; record a full playthrough video; arXiv draft promised; **move to Gen II (Gold/Silver/Crystal)** where HMs can be used by pressing A in the field (fewer menu steps) but Gen II has more HMs, is "extremely difficult", and gift Pokemon cannot reliably teach all HM moves. Idea: **Totodile starter + catch the Red Gyarados** to cover HMs; open question how to get Surf, Waterfall, Whirlpool. Also notes Claude's Pokemon play uses no RL and likely has Pokemon pretraining data, so is not zero-shot.

---

# C. NeurIPS 2025 PokeAgent Challenge: RL-relevant extracts (Emerald, not Red)

(arXiv 2603.15563v2; I read ~2/3 of the HTML; compute/hyperparameter detail not read.)
- Speedrunning track: Pokemon Emerald, reward = +1 per milestone to first gym (Roxanne), 15 milestones, gamma=1. 22 valid submissions, 6 hit 100%. Winner **Heatz** 40:13 via **Scripted Policy Distillation** (LLM writes subgoals + scripted policies -> imitation -> DQN with expert-seeded replay, epsilon from 0.1 to 0). Second place **Hamburg PokeRunners (pure RL)**: recurrent PPO (LSTM), CNN on 128x128 gray frames, sinusoidal x/y position encoding, one-hot location, **38-dim binary milestone vector as memory + goal conditioning** (reduced catastrophic forgetting), **only positive rewards** (step penalties caused safe/stalling behaviour), several small models stitched at inference, 1:14:43 to first badge. Human top 18 min, average 1:22:05.
- Battling track: top methods RL or search; winners PA-Agent (Gen 1 OU), FoulPlay (Gen 9 OU).
- "A harness is a prerequisite for OOD performance" for LLMs.

---

# D. What transfers to Pokemon Crystal and our tensormew project

(Our repo uses a Crystal ROM and Falkner/gym navigation; see CONTEXT.md / SPEC.md in the repo root for specifics. I did not re-read them for this note, so mapping below is from the papers.)

**Directly reusable**
1. **Action timing:** press 8 frames, release 16, decide every 24 frames: tile-exact movement. Gen II uses the same grid movement, so this should carry over; Crystal's text/battle speed options can be set to "fast" in the save state (the paper's Fast variant: ~25% fewer steps to Cerulean).
2. **Observation recipe:** 72x80 grayscale + visited mask + event-flag vector + party features + map ID embeddings (the pokerl feature set). Crystal's RAM map is documented by pret/pokecrystal, so event flags and map IDs are readable the same way.
3. **Reward skeleton:** event reward (dominant, ~+2 per event in paper), small per-coordinate nav bonus (0.005 scale), capped level reward, heal reward. Take the lesson to **cap or remove the heal reward**, or gate it by "heal at a center" only, to avoid the Leech Seed / mother-healing loops. Use ablation result: dropping level reward helped in the paper (4.60 vs 4.19).
4. **Dynamic step budget** (10,240 + 2,048 per event) and desynced resets for stability; or mini-episodes where only exploration maps reset.
5. **PPO hyperparameters** as a sane start: gamma 0.997, lambda 0.95, 2048-step rollouts, clip 0.2, LR 3e-4, no entropy bonus; LSTM(128) as in pokerl rather than the 24-day CPU GRU.
6. **Swarm / Go-Explore-style checkpoint sharing** (all envs load the first env's save state upon a new required event) is the strongest known trick for open-world sparse progress and maps naturally onto our existing save-state infrastructure (states/ dir) and any "reset to checkpoint" logic.
7. **Metrics:** live coordinate map + per-badge timing + Pokemon Center / heal counts per run as the early-warning system for reward hacks. Inspect party tables for unexpected exploits.
8. **Positive-only reward lesson** (Hamburg PokeRunners) and milestone-vector conditioning as memory.
9. **Reward-hack catalogue** to pre-empt: heal farming, gambling at hidden objects, tree-cut loops, hoarding exploration reward by avoiding battles, skipping gyms, grinding in a zone with boosted level reward.

**Crystal-specific differences to plan for**
- Gen II has **more HMs** (Cut, Fly, Surf, Strength, Flash, Whirlpool, Waterfall) and more are mandatory; the pokerl authors flag that gift Pokemon can't be relied on to teach them. HMs in Gen II are usable by pressing A in front of the obstacle, removing menu depth (a big win for RL). Badge-gated field moves also differ.
- Crystal has **two regions (Johto + Kanto)**, a day/night clock, phone calls, held items, and a different event system; the event-flag vector and custom-event list must be rebuilt from pokecrystal.
- Starter: Totodile idea from pokerl (plus Red Gyarados) is hypothesis only; Cut is obtained early (Ilex/Gym 2-3 area), which changes the Falkner-to-Bugsy-to-Whitney progression we use.
- No Safari Zone step-limit puzzle exactly as in Red, but Bug-Catching Contest and Radio Tower/Rocket takeover are scripted bottlenecks.

---

# E. Gaps and novelty opportunities

1. **Remove the scripts.** pokerl still scripts Strength, Surf, Flash, Pokeflute, items, infinite money. A Crystal agent that **learns HM acquisition and use end-to-end** (using the A-press field interaction) would be novel and is explicitly listed as an open question by both groups.
2. **Battling is unaddressed.** Neither Red source learns battle strategy (no species/moves in the paper's observation; pokerl says grinding suffices). Crystal's Gym leaders and Red are significantly harder; a learned battle module (action masking, type-chart features from RAM) or a hybrid (RL navigation + battle policy from Showdown-style agents such as the PokeAgent Metamon baselines) is open.
3. **Fair evaluation of the "beat the game" claim.** The pokerl book has no peer-reviewed protocol, no variance, no compute accounting, no Elite Four breakdown. Publishing a reproducible benchmark (seeds, wall-clock, ablations of each map-ID boost / script) with success rate and steps for Crystal would fill a hole.
4. **Replace map-ID boosts with learned exploration** (RND/ICM/NGU or LLM-derived subgoal rewards). The papers list this as future work; PokeAgent winners show LLM-generated scripts distilled into RL can be fast in Emerald, which suggests **LLM-authored reward/subgoal curricula** for Crystal. The pokerl authors state text reading would remove the boosts.
5. **Hierarchical / options.** The time-bias result (agent picks the fastest-reward starter) is a clean failure case for hierarchical or higher-gamma credit assignment; a minimal-scope experiment (Totodile/Chikorita/Cyndaquil choice with measured downstream cost) is cheap in Crystal.
6. **Generalisation.** Both start from fixed states; Crystal randomisers (starter, Pokemon, items) and cross-save generalisation are unstudied.
7. **Memory.** GRU wins on order-sensitive quests (Bill) but is too costly; LSTM-128 is used in pokerl. A study of memory architecture vs compute (e.g. truncated BPTT, state-space models) on order-sensitive Crystal quests (Sudowoodo, Radio Tower, Ice Path) would be novel.
8. **Sample efficiency.** ~7-24 h on 4x4090 (pokerl) vs. 36 h / 24 days (paper). Reporting steps-to-badge curves on Crystal with a smaller footprint is a differentiator.
9. **Safety/robustness of reward shaping** as a first-class result: systematic ablation table of every shaping term on Crystal (the paper only ran 15 configs on 4 terms).
10. **Honest framing.** Be precise in any write-up that neither Red source is "NeurIPS"; the Red full-game result is an unreviewed technical writeup with scripted assists.

---

# Sources

- Pleines et al., arXiv abs: https://arxiv.org/abs/2502.19920 ; PDF: https://arxiv.org/pdf/2502.19920 (v1: https://arxiv.org/pdf/2502.19920v1 ; v2: https://arxiv.org/abs/2502.19920v2). arXiv comments: "8 pages, 3 figures, 3 tables, under review".
- CoG 2025 citation (secondary, citing paper "Mastering PokeGym"): https://arxiv.org/pdf/2604.08340 (via search snippet only, not read in full); ResearchGate listing (403 on fetch): https://www.researchgate.net/publication/394970144_Pokemon_Red_via_Reinforcement_Learning
- pokerl web book: https://drubinstein.github.io/pokerl/ ; env: https://drubinstein.github.io/pokerl/docs/chapter-2/env-setup/ ; observations: .../chapter-2/observations/ ; rewards: .../chapter-2/rewards/ ; policy: .../chapter-2/policy/ ; running: .../chapter-3/running/ ; metrics: .../chapter-3/metrics/ ; swarm: .../chapter-3/swarm/ ; results: .../chapter-4/results/ ; future: .../chapter-4/future/ ; appendix: .../appendix/breakdown/{battling,8badges,field_interactions,team_rocket}/
- Repo: https://github.com/drubinstein/pokemonred_puffer ; README raw: https://raw.githubusercontent.com/drubinstein/pokemonred_puffer/main/README.md
- GIGAZINE coverage: https://gigazine.net/gsc_news/en/20250310-pokemon-rl-edition
- neroRL branch (Pleines code, contents unconfirmed): https://github.com/MarcoMeter/neroRL/tree/poke_red
- PokeAgent Challenge (NeurIPS 2025 competition): https://arxiv.org/html/2603.15563v2
- PokeRL (secondary, contains mistakes about Pleines): https://arxiv.org/pdf/2604.10812
- Whidden 2023 video (cited by the paper): https://www.youtube.com/watch?v=DcYLT37ImBY ; Anthropic Claude Pokemon post cited: https://www.anthropic.com/news/visible-extended-thinking
- PufferLib: https://arxiv.org/abs/2406.12905

## Unverified / not checked list
- NeurIPS 2025 acceptance of 2502.19920: no evidence found; CoG 2025 venue only from a secondary citation.
- Exact forms of R_heal and R_lvl equations (PDF extraction garbled).
- Training step counts (~400M) read from figure axis text.
- Pokerl: reward coefficients, number of envs for the winning run, total hours to win, Elite Four / Champion fight handling, Victory Road handling, any arXiv draft.
- Contents of neroRL `poke_red` branch and whether human replay data / pretrained agent were released.
- Not read: PokeAgent paper's final ~44% (appendix E writeups, compute).
- Not cross-checked against our repo's CONTEXT.md / SPEC.md.
