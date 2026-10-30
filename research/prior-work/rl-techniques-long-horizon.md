# RL / Agent Techniques for Very Long-Horizon, Sparse-Reward, Partially Observable Games (target: Pokémon Crystal)

Compiled 2026-10-19. Scope: techniques that could help tensormew clear all of Pokémon Crystal (16 badges, Elite Four, Red; ~1M+ steps, hundreds of subgoals) and that could support a NeurIPS paper.

**Verification legend.**
- Unmarked title, venue, and year: confirmed by a web search or fetch in this session. Search snippets are sometimes secondary, so the venue is only as good as the listing found.
- `[UNVERIFIED]`: recalled from memory or seen only in a secondary source, and not re-checked. Treat it as a lead.
- Numeric results quoted here come from abstracts or listing pages, not full-text reads. Re-check them before citing.

**tensormew baseline context** (from `SPEC.md`, `README.md`). The repo already has these pieces:
- PyBoy and RAM parsing (`ram_parser.py`)
- A checkpoint manifest (`states/manifest.json`, `StateRegistry.capture`)
- A deterministic mover and macros
- An LLM supervisor that fires on triggers only
- A small PPO (`rl_env.py`, `ppo.py`)

Its stated metric is Johto badge count and the furthest registered checkpoint. Its secondary thesis is "how many steps actually need an LLM." That agent is a hybrid, not a pure end-to-end RL agent. Several directions below exploit that fact.

---

## 0. Directly relevant Pokémon prior art (read first)

| Item | Venue / year | Note |
|---|---|---|
| Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden, "Pokemon Red via Reinforcement Learning," arXiv 2502.19920 | arXiv 2025; `research/README.md` calls it the NeurIPS 2025 paper. I did not confirm the venue [UNVERIFIED]. | Simplified Red env plus PPO baseline. Gets through Cerulean City. Ablations show agents exploit specific reward signals. Authors pitch Pokémon as a testbed for hierarchical RL, LLM agents, and exploration. |
| Mudireddy and Patibandla, "PokeRL: Reinforcement Learning for Pokemon Red," arXiv 2604.10812 | arXiv, Apr 2026 | Loop-aware wrapper, anti-loop and anti-spam layers, dense hierarchical reward. Covers only early-game tasks (leave the house, reach grass, win the first rival battle). Documents "action loops, menu spam, wandering." |
| Karten et al., "The PokeAgent Challenge: Competitive and Long-Context Learning at Scale," arXiv 2603.15563 | NeurIPS 2025 competition; arXiv Mar 2026 | Battling Track plus Speedrunning Track (a standardized RPG speedrun framework with multi-agent LLM orchestration). It reports "considerable gaps between generalist (LLM), specialist (RL), and elite human performance." Which RPG the speedrun track uses was not confirmed [UNVERIFIED]. |
| Grigsby et al., "Human-Level Competitive Pokémon via Scalable Offline RL with Transformers" (Metamon), arXiv 2504.04395 | arXiv 2025; venue [UNVERIFIED] | Battle-only. Offline RL on large human battle data. Relevant for the battle layer, not for navigation. |
| Orak benchmark, arXiv 2506.03610 | arXiv 2025 | Includes Pokémon Red. Reports Gemini-2.5-pro 83.3, DeepSeek-R1 75.0, Claude-3.7 63.9 (abstract-level figures). |
| PokéAI, arXiv 2506.23689 | arXiv 2025 | Text-only multi-agent system for Red. |
| Zhang (ARISE Foundation), Gemini 3 Pro vs 2.5 Pro race in Pokémon Crystal | Dec 2025 blog and stream; secondary reposts only | Reports Gemini 3 Pro finishing Crystal (16 badges, E4, Red) with a shared harness. I did not open the primary source. [UNVERIFIED details] |
| Universidad Complutense de Madrid thesis, Go-Explore on Pokémon Red with exploration-only reward and no RAM access | Thesis, not arXiv | Reported human-like navigation. Not a peer-reviewed result. [UNVERIFIED rigor] |
| PufferLib, arXiv 2406.12905 | 2024 | Cites Pokémon Red RL (Whidden's project) as a showcase. Useful for throughput. |

**Gap.** I found no peer-reviewed pure-RL or hybrid-RL agent that clears Crystal. LLM-harness agents reportedly have (Gemini, secondary source). The 2025 RL baselines stop around Cerulean City or the first gyms. That gap is the opening for a paper.

---

## 1. Exploration

**RND, Random Network Distillation** (Burda et al., ICLR 2019 [UNVERIFIED venue]). The bonus is the prediction error against a fixed random network. It is simple and scalable, and it fades as states get familiar.
*Crystal:* A cheap baseline on screens or RAM features. It fades on revisits, so it does little to push through long dead ends. Expect it to chase text boxes, menus, and animations unless the observation is masked.

**NGU and Agent57** (Badia et al., NGU arXiv 2002.06038, ICLR 2020 [UNVERIFIED venue]; Agent57, arXiv 2003.13350, ICML 2020). Episodic novelty from nearest neighbors in a learned controllable-state embedding, modulated by lifetime RND novelty. Agent57 adds a bandit meta-controller over exploration rates and separate extrinsic and intrinsic value heads.
*Crystal:* The episodic term suits "visit all rooms in this dungeon this episode." The lifetime term stops re-exploring Route 29. A controllable-state embedding (an inverse-dynamics model) can ignore NPC animation and text.

**E3B, Exploration via Elliptical Episodic Bonuses** (Henaff, Raileanu, Jiang, Rocktäschel, NeurIPS 2022). Extends count-based episodic bonuses to continuous embeddings via an ellipse fit over the episode's embeddings. State of the art on 16 MiniHack tasks.
*Crystal:* A natural episodic novelty for a procedure like "explore a new map section from a checkpoint," where episodes are short and resettable.

**Count and hash-based bonuses** (Bellemare et al. 2016; Tang et al. "#Exploration," NeurIPS 2017) [UNVERIFIED]. Hash RAM-derived features (map id, x, y, event flags) and reward 1/√n.
*Crystal:* This is probably the strongest cheap bonus. RAM gives an exact, low-noise state key, as in the community Red work. Good to anchor against before anything fancier.

**Go-Explore** (Ecoffet et al., "First return, then explore," Nature 2021; arXiv 1901.10995 as "Go-Explore: a New Approach for Hard-Exploration Problems"). Keep an archive of visited cells, return to promising cells without exploring, then explore from them. Robustify afterward by imitation. Reported mean over 43k on Montezuma's Revenge, about 4x the earlier state of the art. Critics note it needs determinism or resettable states.
*Crystal:* PyBoy save states give exact resettability, and `states/manifest.json` is already a hand-built archive. Direct fit. The open design question is the cell representation, such as (map, coarse position, event-flag subset) versus learned.

**Intelligent Go-Explore** (Lu, Hu, Clune, arXiv 2405.15143, ICLR 2025). A foundation model replaces hand-coded heuristics for which archive states are "interesting" and which to expand. Tested on language-based tasks (Game of 24, text worlds, etc.), not on pixel games.
*Crystal:* Tells us where to put the LLM. It judges interestingness at archive-selection time and not every step, which matches the repo's "LLM on triggers" design. Not yet shown on a GBC-scale game, so this is a gap.

**Backward-from-demonstration reset curricula** (Salimans and Chen, "Learning Montezuma's Revenge from a Single Demonstration," arXiv 1812.03381, 2018). Train RL with episodes reset to states along one demonstration, moving the start back as competence rises. This turned an exponential exploration problem into a roughly quadratic one in a toy analysis, and scored 74.5k on Montezuma. It needed 128 GPUs for 2 weeks. It is a 2018 preprint that I did not confirm was published.
*Crystal:* The same trick works on any trajectory recovered from the manifest, or on a trajectory Go-Explore discovered. It is the most direct bridge from "one successful playthrough" to a robust policy.

**Jump-Start RL (JSRL)** (Uchendu et al., ICML 2023). A guide policy rolls out the first h steps and the learner finishes. h is annealed. There is a polynomial-in-horizon sample complexity bound for the setting.
*Crystal:* A scripted or LLM harness can be the guide policy. Its first N steps feed an RL learner, which is exactly the hybrid regime tensormew is in.

**Other exploration methods** (BYOL-Explore, NeurIPS 2022; Plan2Explore, ICML 2020) [UNVERIFIED]. World-model-based curiosity.
*Crystal:* These need a latent dynamics model that handles text and menus. See section 6.

---

## 2. Hierarchical RL, options, skill discovery, goal-conditioned RL

**Director** (Hafner, Lee, Fischer, Abbeel, "Deep Hierarchical Planning from Pixels," NeurIPS 2022, arXiv 2206.04114). A manager picks latent goals in the world-model space and a worker reaches them. Goals can be decoded to images for inspection. Strong on sparse-reward mazes.
*Crystal:* The decodable-goal idea could power an inspectable "go to the gym leader's door" manager. Whether latent goals stay stable across 1M steps is untested.

**METRA** (Park, Rybkin, Levine, ICLR 2024 oral, arXiv 2310.08887). Unsupervised skills built from a temporal-distance-aware latent, which scales to pixel quadruped and humanoid locomotion.
*Crystal:* Could in principle learn "directional navigation" skills reusable across maps, giving a skill basis for an overworld controller. Skills may not map onto discrete semantic acts like "talk" or "use Cut."

**Contrastive RL / goal-conditioned value as representation** (Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL," NeurIPS 2022, arXiv 2206.07568). Hindsight relabeling plus a contrastive critic gives a goal-conditioned value without TD learning or hand rewards.
*Crystal:* RAM predicates (map reached, flag set, item obtained) provide a free goal space. Hindsight relabeling over all past trajectories gives plenty of training signal for reaching those goals.

**HER, Option-Critic, DIAYN** (2017 to 2018 classics) [UNVERIFIED]. Not re-verified here.
*Crystal:* Standard components for a goal-conditioned or option-based low level.

**Recent LLM-planner / RL-executor hierarchies** (all 2026 arXiv; I did not check peer review except where noted):
- HiPER (arXiv 2602.16165, listed as an ICML 2026 poster). Planner proposes subgoals, executor carries them out, with explicit credit assignment.
- HiMAC (arXiv 2603.00977). Macro/micro split. Tested on ALFWorld, WebShop, and Sokoban.
- HIPIF (arXiv 2606.10507). Discards the history of finished subgoals to keep context short.
- STO-RL (arXiv 2601.08107). LLM-ordered subgoals for offline RL under sparse rewards.

*Crystal:* Context folding after each completed subgoal maps onto Crystal's badge-by-badge structure. These are mostly text-environment results, not GBC games.

---

## 3. LLM-guided reward design and goal generation

**Eureka** (Ma et al., ICLR 2024). An LLM writes reward code, evolutionary search iterates on RL training feedback. Reported beating human-written rewards on 83% of 29 tasks. Robotics-style tasks with fast simulators.
*Crystal:* Fast, resettable segment rollouts could be the fitness oracle (cost: a rollout per candidate). The risk is reward hacking against a proxy. Fitness must be measured on RAM-defined milestones that the LLM-written reward cannot see or edit.

**Text2Reward** (Xie et al., ICLR 2024 spotlight, arXiv 2309.11489). Produces dense reward code from a compact environment description, refined with human feedback. Matched expert rewards on 13 of 17 manipulation tasks.
*Crystal:* The compact environment description is a RAM schema. Our `ram_parser.GameState` is already that.

**ELLM** (Du et al., ICML 2023, arXiv 2302.06692). Prompts an LLM with the state description for plausible next goals, then rewards the agent for achieving them. Tested in Crafter and Housekeep.
*Crystal:* A direct template, with the text from `GameState.to_text()`. Common-sense goals ("buy Poké Balls," "heal at the Pokémon Center") come for free. Crafter is much shorter than Crystal, so the scaling question is open.

**Motif** (Klissarov, D'Oro et al., ICLR 2024, arXiv 2310.00166). An LLM gives preferences over pairs of game-event captions, which are distilled into an intrinsic reward. In NetHack, intrinsic-only training beat score-only training. The authors also report misalignment (wireheading) when combining two individually-aligned rewards.
*Crystal:* Captions can be written from RAM diffs ("learned HM Cut," "defeated Falkner"). Labeling can be done offline and in bulk, so cost is bounded. Needs an explicit hacking audit.

**MaestroMotif** (Klissarov et al., ICLR 2025, arXiv 2412.08542). An LLM turns skill descriptions into AI-feedback rewards, writes initiation/termination code, and writes the policy-over-skills as code. Evaluated on NetHack.
*Crystal:* Very close to the design we want: named skills (navigate, fight trainer, shop, heal, solve puzzle), each trained with its own reward, composed by LLM-written code. The paper is NetHack, which is a roguelike with a different structure from a story-gated RPG.

**OMNI and OMNI-EPIC** (Zhang, Lehman, Stanley, Clune; OMNI ICLR 2024, arXiv 2306.01711; OMNI-EPIC ICLR 2025). Uses a model of human "interestingness" to pick tasks that are learnable and worthwhile.
*Crystal:* Could rank subgoals or curriculum states. Open-ended in nature, so it needs adapting to a finite game with a fixed ending.

**LLM-as-subgoal-planner**: Voyager and SayCan below, plus the hierarchies in section 2.

**Caveat.** I found no paper using LLM-written rewards for a sparse-reward story-gated RPG. The nearest are NetHack and Crafter results.

---

## 4. LLM + RL hybrids in games, and benchmarks

**Voyager** (Wang et al., TMLR 2024; arXiv 2305.16291). GPT-4 driven Minecraft agent with an automatic curriculum, an executable skill library, and iterative self-verification. Reports 3.3x more unique items and up to 15.3x faster tech-tree milestones than prior state of the art (arXiv figures; the TMLR figures differ).
*Crystal:* The skill library idea fits macros (`macros.py`). Skills would be verified by RAM post-conditions. Voyager relies on a code API, and Crystal has no such API unless we build one. The repo already has a partial one in `tool_dispatch`.

**SayCan** (Ahn et al., CoRL 2022, arXiv 2204.01691). The LLM proposes candidate actions and learned value functions score feasibility.
*Crystal:* Affordance value functions could be learned from reset rollouts, such as "P(reach Goldenrod from here within 5k steps)." The LLM would be gated by competence estimates instead of trusting itself.

**JARVIS-1** (Wang et al., arXiv 2311.05997, NeurIPS 2023 workshop). Multimodal LLM planning with memory of past experience, goal-conditioned controllers. Reports 12.5% on the diamond-pickaxe task.
*Crystal:* The multimodal experience memory is a pattern for remembering how a previous gym or puzzle was solved.

**Minecraft RL/IL stack:**
- MineDojo (NeurIPS 2022, datasets and benchmarks outstanding paper per press coverage): internet-scale video, wiki, and Reddit data.
- VPT (Baker et al., NeurIPS 2022, arXiv 2206.11795): an inverse dynamics model labels unlabeled video, then behavioral cloning, then RL fine-tuning. First agent to craft diamond tools.
- STEVE-1 (Lifshitz et al., NeurIPS 2023): text-to-behavior on top of VPT.
- Plan4MC [UNVERIFIED].

*Crystal:* VPT is the template for offline priors (section 7).

**NetHack and NLE.**
- NetPlay (arXiv 2403.00690): zero-shot GPT-4 agent.
- Tuyls et al., "Scaling Laws for Imitation Learning in Single-Agent Games" (TMLR 2024; arXiv 2307.09423, originally titled for NetHack): BC loss and return scale smoothly with compute.
- "Benchmarking Long-Horizon Agents with Checkpointed Exploration in NetHack" (Liu, Zhang, Karten; NeurIPS 2026 workshop poster): save and restore, BAL_min and BAL_max metrics. Their search-based Prime-agent-explore reports 11.38% BAL_min versus about 2% for harness baselines. The numbers come from a single workshop paper.

*Crystal:* The checkpointed-exploration paper is the closest 2026 relative of our setup. It combines save/restore with an LLM, benchmarked on a procedurally generated game. It does not do a story-gated RPG.

**BALROG** (Paglieri et al., ICLR 2025, arXiv 2411.13543). Benchmark of LLM/VLM agents on games. Finds weak performance on hard games and especially poor vision-based decision-making.
*Crystal:* Supports using text/RAM observations instead of raw pixels for any LLM component.

**Crafter / Craftax** (Craftax: Matthews et al., ICML 2024 spotlight, arXiv 2402.16801). A JAX rewrite of Crafter that is up to 250x faster, plus a harder NetHack-flavored variant where existing exploration methods make little progress.
*Crystal:* A cheap proxy domain for prototyping the methods before spending GBC emulator hours. A Craftax win does not guarantee a Crystal win.

**AlphaStar** (Vinyals et al., Nature 2019). Supervised learning on human replays to initialize a league, then RL. Above 99.8% of ranked players.
*Crystal:* The pattern "BC prior first, RL after" applies, but there are no replay corpora for Crystal comparable to StarCraft's.

**Surveys.** "A Survey on LLM-Based Game Agents" (arXiv 2404.02039).

**Orak** (above): a benchmark covering Pokémon Red among other games.

**Note on harness fairness.** Community posts say the Claude and Gemini Pokémon runs used different harnesses (including overlay help for navigation) and are not controlled benchmarks. I only saw these in secondary and blog sources [UNVERIFIED].

---

## 5. State abstraction, frontier, and archive methods

No single paper to cite here beyond Go-Explore, Intelligent Go-Explore, and E3B above. The design choices that matter for Crystal:
- **Cell definition.** Candidates: (map_id, tile_x, tile_y) coarse-grained, plus a hash of the event-flag bitset, party summary, and badge count. Too fine and the archive explodes. Too coarse and Go-Explore returns to the wrong place.
- **Selection.** Count-based, learning-progress, LLM-judged interestingness (Intelligent Go-Explore), or value/frontier distance.
- **Returning.** In Crystal, "return" is just `load_state`, but PyBoy states are large (a snapshot per cell costs memory). A trajectory-replay fallback is cheaper but needs determinism, including RNG. Whether Crystal's RNG makes replay unreliable here is not established [UNVERIFIED].
- **Gating.** Many Crystal advances are gated by items or flags (Cut, Surf, badges), so the archive needs the gating state in the cell, or the frontier looks permanently stuck.

---

## 6. World models and memory architectures

**DreamerV3** (Hafner, Pasukonis, Ba, Lillicrap, "Mastering diverse control tasks through world models," Nature 640, 2025, arXiv 2301.04104 as "Mastering Diverse Domains through World Models"). Single config over 150+ tasks. First to collect diamonds in Minecraft from scratch without human data or curricula, though press reports about 9 days of play before the first diamond and a Minecraft-specific low success rate.
*Crystal:* Could learn a latent model of Crystal for imagined exploration. Text boxes and menus are open questions for pixel prediction.

**Dreamer memory limits.** R2I ("Mastering Memory Tasks with World Models," arXiv 2403.04253; ICLR 2024 venue [UNVERIFIED]) replaces the recurrent core with S4-style SSMs to improve memory-intensive tasks. TransDreamerV3 (arXiv 2506.17103) argues that DreamerV3's RSSM has weak long-term memory. Memory Maze (arXiv 2210.13383) quantifies memory as a bottleneck.
*Crystal:* Which tasks are actually memory-bound is unknown (see direction 7).

**Sequence-model backbones.**
- RLBenchNet (arXiv 2505.15040): LSTM/GRU struggled on the hardest memory tasks, while Transformer-XL, Gated TrXL, and Mamba-2 solved them. Mamba-2 used about 8x less memory than TrXL. One preprint.
- Decision Mamba-Hybrid (NeurIPS 2024). Mamba produces subgoals from long-term memory and a transformer predicts actions.
- Drama (arXiv 2410.08893). Mamba in model-based RL.
- xLSTM (Beck et al., NeurIPS 2024 [UNVERIFIED venue; seen on an author index]). Large Recurrent Action Model (Schmied et al., ICML 2025, arXiv 2410.22391): xLSTM matches transformers on 432 tasks with linear-time inference.
- I found no head-to-head xLSTM vs Mamba POMDP memory benchmark in RL.

*Crystal:* Throughput matters for 1M+ step rollouts, so recurrent/SSM backbones with linear-time inference are attractive. The gain over an explicit RAM-derived "what have I done" vector is an empirical question.

**Practical note.** In tensormew, RAM provides event flags, party, items, and map. Much of the long-term memory a pixel agent needs is directly readable. This weakens the case for memory architectures as the main contribution, but it makes a clean ablation (see direction 7).

---

## 7. Offline RL, imitation, human data, speedruns

**VPT** (above). Pipeline: small labeled set to inverse dynamics model, IDM labels large unlabeled video, BC, RL fine-tune.
*Crystal:* Speedrun/let's-play videos are plentiful. A GBC IDM is much simpler than Minecraft's because the action space is 8 buttons. Domain mismatch: runs use different saves, glitches, and RNG manipulation.

**MineRL dataset** (IJCAI 2019). 60M+ state-action pairs. A reminder that human demos at scale are an enabling asset. Nothing equivalent exists for Crystal that I found.

**TAS / speedrun datasets.** I searched for tool-assisted speedrun input files used for imitation learning and found none. TASVideos input files exist for many games, but whether Crystal movies exist and are usable is [UNVERIFIED].

**Scaling laws for IL** (Tuyls et al., TMLR 2024). BC performance follows power laws in compute. Single-agent game datasets are a plausible lever.

**Metamon** (battle only, section 0). Evidence that offline RL on large human data is effective for Pokémon decisions.

**Backward algorithm and JSRL** (section 1) are the main ways a single demonstration or a guide policy is turned into a learned policy. The repo's manifest-based checkpoint chain amounts to a hand-made demonstration.

---

## 8. Curriculum from save states, automatic checkpoint discovery, reset-based learning

- **Reverse curriculum** (Salimans and Chen; Florensa et al. 2017 [UNVERIFIED]): start near the goal, move the start back.
- **Start-state curricula via a guide** (JSRL).
- **Archive growth as automatic checkpoint discovery** (Go-Explore). Cells reached for the first time become checkpoints. Nothing standard exists for choosing which cells deserve to be kept as "milestone" checkpoints in an RPG.
- **Checkpointed exploration for LLM agents** (Liu, Zhang, Karten, NeurIPS 2026 workshop), above.
- **Prioritized level replay and unsupervised environment design** (Jiang et al. 2021; Dennis et al. 2020) [UNVERIFIED]. Learning-progress-weighted choice of start states. Craftax reports these make little progress there.

*Crystal:* tensormew currently hand-registers checkpoints (`tools/make_state.py`, `StateRegistry.capture`). Making checkpoint discovery automatic, and measuring how many hand-made checkpoints are needed, is both a practical and a research gap.

---

## 9. Reward hacking and shaping pitfalls

- **Observed in Pokémon:** Pleines et al. (2502.19920) show agents exploiting specific reward signals. PokeRL (2604.10812) lists loops, menu spam, and wandering, and answers with explicit anti-loop machinery.
- **Wireheading by composition:** Motif authors' post (Alignment Forum, "Wireheading and misalignment by composition on NetHack") reports that two rewards that are individually aligned can produce misaligned behavior when summed, depending on prompt wording.
- **LLM reward generators:** Eureka-style loops risk "in-context reward hacking" when the same model writes and judges (2025 survey "Sailing by the Stars," arXiv 2505.02686). Other reward-hacking surveys I saw are from 2026 and I did not check them [UNVERIFIED].
- **Classic results to keep in mind** [UNVERIFIED, from memory]: potential-based shaping preserves optimal policies (Ng, Harada, Russell 1999). Specification-gaming catalogs (Krakovna et al.).
- **Crystal-specific exploit surfaces** (my analysis, not literature): shopping and PC box menus, repeated healing, grinding in a safe spot, daily-event timers, trainer-rematch phone calls, and any reward tied to Pokédex counts or party level.
- **Determinism critique of Go-Explore:** it presumes a resettable or deterministic simulator. Fine for our setting, but any claim about "exploration" must say resets were used.

---

## 10. Evaluation protocols for "beating a whole game"

- **Stat practice:** Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" (NeurIPS 2021 outstanding paper, arXiv 2108.13264). Interval estimates, performance profiles, IQM. With one expensive full-game run per seed, few seeds is the norm, so use bootstrap intervals and report all seeds.
- **Progress metrics from the game side:** BALROG (ICLR 2025) uses a progression-based score. Checkpointed-NetHack introduces BAL_min and BAL_max relative to human percentiles. PokeAgent provides a standardized speedrun track.
- **What should be reported for Crystal (a proposal, based on the literature above and the tensormew SPEC):**
  - Milestones reached (badges plus finer RAM event flags), with area under the milestone-vs-steps curve.
  - Resource accounting: emulator steps, resets used, LLM calls and tokens, human-authored checkpoints, hand-coded macros.
  - A "no-reset, from new game" track versus a "reset-assisted" track.
  - Which information the agent saw: pixels only, RAM, or RAM-derived text.
  - Several seeds and independent save files.
  - Hold-out evaluation on segments not used for tuning.
- **Comparability warning:** the PokeAgent paper states that earlier Pokémon demonstrations used different games, harnesses, and criteria and so were not comparable. Any Crystal paper should align with PokeAgent where possible.

---

## 11. Ranked shortlist: 8 candidate novel research directions (NeurIPS)

Ranking weighs novelty against the literature above, fit to tensormew's existing assets, feasibility within a submission cycle, and risk. All 8 assume Crystal with resettable PyBoy states.

### 1. Stall-triggered LLM "unblocking" for Go-Explore, with counterfactual verification on resets
- **Hypothesis.** Go-Explore-style archives stall at gating bottlenecks (missing HM, key item, story flag), and a stall detector (archive-frontier growth below a threshold) plus an LLM that proposes *why* and *what* (from RAM text) beats both unguided Go-Explore and LLM-every-step. Resets let us verify each LLM hypothesis cheaply: load the cell, apply the proposed unblocking plan, check whether new event flags appear.
- **Experiment.** Compare against count-based and RND archives, Intelligent-Go-Explore-style per-step interestingness, and an LLM-every-step harness. Measure bottlenecks passed (ground truth from the pret disassembly or a walkthrough) per emulator step and per token, over 5 or more seeds. Ablate: no verification, no stall detector, small versus large LLM.
- **Risk.** A strong LLM may already know Crystal's progression, which would make the stall detector look unnecessary. Needs a contamination control, such as scrambled item/map names or a ROM hack. Go-Explore on Red exists informally (the thesis), so novelty rests on the stall-and-verify loop and the evaluation.

### 2. Learning-progress start-state curriculum over the archive DAG, distilled into one end-to-end policy
- **Hypothesis.** Extending the backward-from-demonstration idea from a single trajectory to a whole discovered archive, with a bandit or learning-progress sampler over start cells, yields one goal-conditioned (or unconditioned) policy that chains many segments and survives a final no-reset test from New Game, which Go-Explore alone does not provide (it robustifies single trajectories).
- **Experiment.** Train on archive cells from direction 1 (or from manifest checkpoints). Compare uniform, backward-only, JSRL-annealed, and learning-progress sampling. Evaluate end-to-end from New Game with no resets, reporting milestone-AUC. Add a distribution-shift test using perturbed RNG or a different save.
- **Risk.** Compute (an RL policy over a 1M-step game is expensive). The no-reset final test may simply fail, which would still be a publishable negative result only if the failure analysis is sharp. PPO plus a recurrent policy may be too weak, so a fallback is to distill into a behavior-cloned policy.

### 3. RAM-caption preference rewards (Motif-style) with an adversarial reward-hacking audit
- **Hypothesis.** AI-feedback ordinal rewards over RAM-diff captions avoid the exploit patterns documented for hand-shaped rewards in Pokémon (Pleines et al.), and an automated adversary that searches for high-reward but non-progressing behavior (loops, menu spam, grinding) can quantify hacking and be used to repair the reward.
- **Experiment.** Label captions offline with an LLM, train the reward model, then train agents on segments. Compare against hand shaping, Eureka-style evolved reward code, and sparse milestone reward. Report a hacking metric (reward gained per milestone), not only success rate. Test composition effects (Motif's wireheading finding) explicitly.
- **Risk.** Motif and Eureka are established, so the novelty must come from the audit and from the long-horizon story-gated setting. The measured differences could be small. Label cost is manageable but not zero.

### 4. A milestone-lattice evaluation protocol with resource and reset accounting for "beating a whole game"
- **Hypothesis.** Badge count and "furthest map" hide differences between methods. A milestone lattice built from RAM event flags (hundreds of nodes with dependency edges), reported as AUC over steps and with explicit accounting of resets, LLM tokens, human checkpoints, and observation type, separates methods that badge count cannot.
- **Experiment.** Build the lattice from the disassembly, implement metrics with rliable-style intervals, and run 4 to 6 baselines (RL, Go-Explore variants, an LLM harness). Show rank changes under the new metric versus badges. Fits a datasets and benchmarks track.
- **Risk.** PokeAgent already offers a speedrun track and a standardized framework, so this must be differentiated (Crystal, RL-oriented, reset accounting). Evaluation papers need broad adoption signals. Verify exactly which game and metrics PokeAgent uses before committing.

### 5. SayCan-style affordance-gated LLM planner over a typed RAM-predicate goal language, with a tokens-per-milestone frontier
- **Hypothesis.** An LLM that proposes only typed goals (reach map X, defeat trainer Y, obtain item Z, set flag W), gated by learned competence estimates from resets, matches an LLM-every-step harness on milestone progress at a fraction of tokens, and the cost-versus-progress frontier is the paper's contribution.
- **Experiment.** Train a hindsight-relabeled goal-conditioned low level on RAM predicates, learn competence estimates for gating, then compare the cost frontier against LLM-every-step and scripted baselines on the same checkpoints (the repo's stated secondary thesis). Sweep LLM size.
- **Risk.** Might be seen as engineering that LLM-harness results (including the reported Gemini 3 Pro Crystal clear) already overshadow. Novelty is in the cost frontier and competence gating. Goal-conditioned low-level training on long maps may be hard.

### 6. VPT-style inverse-dynamics labeling of speedrun and agent-stream video as a prior for Crystal
- **Hypothesis.** A small labeled set (our own play, with button logs) trains an inverse dynamics model good enough to label hours of public Crystal footage, giving a behavior prior that, followed by backward-curriculum RL, beats from-scratch RL on segment-solving sample efficiency.
- **Experiment.** Train the IDM on emulator-generated data (unlimited and perfectly labeled), test transfer to real video (compression, overlays, different save states). Compare BC prior plus RL, RL alone, and the prior from scripted demonstrations. Measure sample efficiency on held-out segments.
- **Risk.** Video domain gap (overlays, scaling, filters), legal and data-use questions for scraped footage, and the speedrun states may not match our checkpoints. Value of the prior may be small relative to resets, which already supply strong exploration.

### 7. Memory-architecture study with RAM-labeled memory dependencies in Crystal (LSTM vs Transformer-XL vs Mamba vs xLSTM vs explicit RAM memory)
- **Hypothesis.** Most of Crystal's "memory" demands are readable from RAM (flags, party, items), so an explicit RAM-derived progress vector matches or beats long recurrent memory, but in pixel-only settings state-space/xLSTM backbones beat LSTM on the subset of tasks that truly need history.
- **Experiment.** Annotate memory-dependent decision points (using the disassembly) and make a segment suite. Train PPO with each backbone and each observation variant. Report success on memory-dependent versus memory-free subsets.
- **Risk.** Engineering heavy, with results likely to be incremental. It becomes a useful section or workshop paper more than a main-track one, unless the annotated memory-dependency suite is released and adopted.

### 8. Latent-space Go-Explore with a learned world model and imagined frontier scoring
- **Hypothesis.** Replacing hand-designed cells with clusters in a learned (Dreamer/R2I-style) latent space, and ranking frontier candidates by imagined novelty, removes the need for RAM-based cell design and transfers across games.
- **Experiment.** Train a world model on GBC frames (plus optional RAM auxiliary targets), define cells by latent clustering, compare against RAM-hash cells on milestone discovery rate. Test transfer to a second game (Red or Emerald).
- **Risk.** Highest. Pixel world models on text-heavy GBC screens are untested, latent cells may merge distinct story states, and compute is large. The upside is generality, which reviewers like, but the odds of a clean win are lowest.

**Suggested sequencing.** Directions 1 and 4 share infrastructure (the archive and the lattice) and could form one paper. Direction 2 follows from 1. Direction 3 can be developed in parallel as a standalone study. Directions 5 to 8 are alternates if 1 and 2 hit dead ends.

---

## Source list (checked via search or fetch in this session unless marked)

- PokeRL: arxiv.org/abs/2604.10812 · Pokemon Red via RL: arxiv.org/abs/2502.19920 · PokeAgent Challenge: arxiv.org/abs/2603.15563 · Orak: arxiv.org/pdf/2506.03610 · PokéAI: arxiv.org/abs/2506.23689 · Metamon: arxiv.org/pdf/2504.04395 · PufferLib: arxiv.org/pdf/2406.12905
- Go-Explore: arxiv.org/abs/1901.10995 · Intelligent Go-Explore: arxiv.org/abs/2405.15143 · Salimans and Chen: arxiv.org/abs/1812.03381 · JSRL: arxiv.org/abs/2204.02372 · Agent57: arxiv.org/abs/2003.13350 · E3B: arxiv.org/abs/2210.05805
- Director: arxiv.org/abs/2206.04114 · METRA: arxiv.org/abs/2310.08887 · Contrastive RL: arxiv.org/abs/2206.07568 · HiPER: arxiv.org/abs/2602.16165 · HiMAC: arxiv.org/abs/2603.00977 · HIPIF: arxiv.org/pdf/2606.10507 · STO-RL: arxiv.org/pdf/2601.08107
- Eureka (ICLR 2024 poster page) · Text2Reward: arxiv.org/abs/2309.11489 · ELLM: arxiv.org/abs/2302.06692 · Motif: arxiv.org/abs/2310.00166 · MaestroMotif: arxiv.org/abs/2412.08542 · OMNI: arxiv.org/abs/2306.01711
- Voyager: arxiv.org/abs/2305.16291 · SayCan: arxiv.org/abs/2204.01691 · JARVIS-1: arxiv.org/abs/2311.05997 · VPT: arxiv.org/abs/2206.11795 · STEVE-1 (NeurIPS 2023) · BALROG: arxiv.org/abs/2411.13543 · NetPlay: arxiv.org/abs/2403.00690 · Craftax: arxiv.org/abs/2402.16801 · NetHack IL scaling: arxiv.org/abs/2307.09423 · Checkpointed NetHack: neurips.cc/virtual/2026/169831 · Game-agent survey: arxiv.org/abs/2404.02039
- DreamerV3 (Nature 640, 2025; arxiv.org/abs/2301.04104) · R2I: arxiv.org/abs/2403.04253 · TransDreamerV3: arxiv.org/abs/2506.17103 · RLBenchNet: arxiv.org/abs/2505.15040 · Decision Mamba (NeurIPS 2024) · Drama: arxiv.org/abs/2410.08893 · LRAM: arxiv.org/abs/2410.22391
- rliable: arxiv.org/abs/2108.13264 · Sailing by the Stars survey: arxiv.org/abs/2505.02686
