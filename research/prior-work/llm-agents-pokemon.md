# Prior work: LLM / VLM agents playing Pokémon

Compiled 2026-10-19 via WebSearch/WebFetch. Sourcing note: most "livestream" numbers are community-reported, not formal benchmarks; "steps", "turns", "actions" are not defined consistently across projects (the PokeAgent paper says so explicitly). Anything I could not confirm from a fetched page is tagged **[UNVERIFIED]**. Several pages were only read through a summarizing fetcher, so exact figures should be re-checked against the primary source before citing in a paper.

---

## 0. TL;DR table

| Project | Model | Game | Result | Scaffold flavor | Steps / time |
|---|---|---|---|---|---|
| Claude Plays Pokémon (ClaudePlaysPokemon) | Sonnet 3.7 -> Opus 4/4.5/4.6/4.7 | Red | 3.7 reached Lt. Surge's gym (per creator, Feb 2025); Opus 4.7 **beat the game, announced 2026-05-16** | Minimal, deliberately; notes + nav hints | Opus 4.5: ~170k steps / 500+ h at Jan 2026 (Time); ~230k steps at Victory Road |
| Gemini Plays Pokémon (GPP, Joel Z) | 2.5 Pro (Blue, Yellow Legacy); 3 Pro (Crystal) | Blue, Yellow Legacy hard, Crystal | Blue beaten twice (May 2 and Jun 9 2025); Crystal beaten Nov-Dec 2025 | Heavy, then progressively lighter | Blue: 813 h then 406 h; 106,505 steps (run 1); Crystal: ~24k turns, ~17 days, 1.88B tokens |
| GPT Plays Pokémon (Clad3815 / OpenAI) | o3, GPT-5, 5.1, 5.2 | Red, Crystal, Emerald, FireRed | o3 Red (18,184 steps); GPT-5 Red (6,470 steps); GPT-5.1 Crystal (9,454 steps / 108 h); GPT-5.2 Emerald, FireRed [partly UNVERIFIED] | Tool-rich: minimap, markers, path tool, notepad, goal tracker | see section 3 |
| PokeAgent Challenge (NeurIPS 2025 -> arXiv 2603.15563) | 5 frontier LLMs + RL teams | Showdown + Emerald (to gym 1) | Best: LLM-written scripts distilled to RL (Heatz, 40:13) | MCP orchestrator, A*, critic, KB | 1.8x slower than avg human at best LLM harness |
| Orak (ICLR 2026) | 12+ LLMs | Red (beat Brock) + 11 other games | GPT-5 88.9 vs human novice 86.1 | Text state from RAM | one-gym scope |

---

## 1. Claude Plays Pokémon (Anthropic, Red)

- **Links:** Twitch `ClaudePlaysPokemon`; creator David Hershey (Anthropic Applied AI). Retrospectives: [A Year Late, Claude Finally Beats Pokémon (LessWrong, Julian Bradshaw, 2026-05-16)](https://www.lesswrong.com/posts/sehJYg5Yny9fvpbpt/a-year-late-claude-finally-beats-pokemon), [Insights into Claude Opus 4.5 from Pokémon (LessWrong)](https://www.lesswrong.com/posts/u6Lacc7wx4yYkBQ3r/insights-into-claude-opus-4-5-from-pokemon), [Opus 4.5 follow-up](https://www.greaterwrong.com/posts/gogZyeistdaDFuhbG/claude-plays-pokemon-opus-4-5-follow-up), [TIME, 2026-01-13](https://time.com/7345903/ai-chatgpt-claude-gemini-pokemon/), [Claude 4 announcement](https://www.anthropic.com/news/claude-4) (memory-file "Navigation Guide" example). No single official Anthropic write-up of the final win was found; the win rests on community/press coverage.
- **Launch:** Feb 2025 with Claude 3.7 Sonnet. Creator quote (via BGR, secondhand): furthest seen was past Lt. Surge's gym. Brock took ~20 h (Substack review, secondhand).
- **Scaffold (as best reconstructed, not an official spec):**
  - Screenshot every turn + text from RAM (walkable-tile highlighting; PokeAgent paper Table 2 lists: frame, location, walkable tiles, party stats).
  - Tools: button presses, a pathfinder/"navigator", a knowledge base / notes file (file-based summaries; memory files in Opus 4+). Opus 4.7 additions: a zoom-into-screen-region tool and a visual-reference tool that stores screenshots/crops as memory.
  - Past "Critic Claude" subagent that flagged stuck behavior, plus visual hints and generic advice; some of these were later removed (LessWrong), i.e. harness got *thinner* over time on purpose.
  - Context is re-initialized and the model leans on its own notes (TIME). Opus 4.6 reportedly had ~10 note files in directories incl. a "learnings" file distilled from failures [secondhand].
  - Spinner tiles labeled as obstructions and screenshot delayed until spin stops (Opus 4.5 era) - a human-added affordance.
  - Opus 4.7 run reportedly allowed larger button sequences per turn and multiple A presses in menus, which makes raw step counts non-comparable across models (commenter claim in LessWrong comments; treat as [UNVERIFIED]).
- **Milestones / cost / steps:**
  - Sonnet 3.7: Lt. Surge region; stuck for dozens of hours at a time (TIME).
  - Opus 4.5: ~29k steps to Silph Co; >50k steps stuck there; Safari Zone ~8k steps; Cinnabar Mansion 112k steps over months; ~48,854 steps / 300+ h at first blog; ~170k steps / 500+ h in TIME; ~230k steps and all 8 badges, stuck at Victory Road boulders (it skipped Gold Teeth -> no Strength HM until backtracking).
  - Opus 4.6: Victory Road in ~30k steps, Cinnabar Mansion ~3k steps (vs 112k), then stuck on boulder puzzles for months [LessWrong; another source says a run reset ~2 weeks after 4.6 release - inconsistent, unresolved].
  - Opus 4.7: beat Elite Four and Champion; Elite Four needed a couple of tries. Total steps/hours/$ **not reported**. One boulder puzzle took "nearly a day".
  - Dollar cost: not published anywhere I found. **[UNVERIFIED / unknown]**.
  - Surge, Cinnabar, Victory Road all confirmed as milestones in the above timeline; the Cinnabar Card Key / Secret Key and Safari Strength are the known blockers.
- **Failure modes (documented):** circling Erika's gym ~4 days (~8,000 steps) not realizing Cut was needed; ~9,000 steps lost to one Rocket Hideout spinner misread; 50-minute loop over a hallucinated "elevator"; ignoring items on the ground ("worthless"); not seeing Victory Road switches; hallucinating targets near the presumed goal; treating trainer line of sight as random; one wrong note stalling progress for days (anterograde-amnesia analogy); a good switch note destroyed during note cleanup; resource mismanagement (PP, Charizard overuse).
- **Lessons:** (1) long-term memory quality = note quality; (2) vision (not reasoning) was the specific bottleneck for Cinnabar/Victory Road and Opus 4.7's improved vision plus a zoom tool resolved it; (3) persistence beats cleverness; (4) a *minimal* harness makes this a model test, not an engineering test, but it costs 10-100x more steps than heavy harnesses.

## 2. Gemini Plays Pokémon (Joel Z; Blue, Yellow Legacy, Crystal)

- **Links:** [The Making of Gemini Plays Pokémon](https://blog.jcz.dev/the-making-of-gemini-plays-pokemon); [How Gemini 3 Pro Beat Pokémon Crystal (and 2.5 Pro didn't)](https://blog.jcz.dev/gemini-3-pro-vs-25-pro-in-pokemon-crystal); [TechCrunch 2025-05-03](https://techcrunch.com/2025/05/03/googles-gemini-has-beaten-pokemon-blue-with-a-little-help); harness code in [sethkarten/continual-harness](https://github.com/sethkarten/continual-harness) (README says it contains the GPP benchmark harness; no documented `--game crystal` flag, so Crystal support in that repo is unclear).
- **Verified: it did beat the game.** Blue run 1 (Gemini 2.5 Pro) finished 2025-05-02, ~813 h; run 2 finished 2025-06-09, ~406 h with *zero* manual interventions (author's own blog). Later Gemini 3 Pro beat Yellow Legacy hard mode and Crystal ("without losing an end-game battle"; Continual Harness paper, TIME). Pichai announced the May Blue win.
- **Crystal run numbers (author blog):** 3 Pro defeated Red at turn 24,178 with ~1.88B tokens, ~17 days wall-clock; 2.5 Pro stalled at 4 badges (Whitney's Miltank grind >2 days; Olivine Lighthouse loop 16,403 turns); author projects 2.5 Pro would need ~157k turns, >15B tokens, ~69 days. 3 Pro used ~half the turns and ~60% fewer tokens than 2.5 Pro to equal milestones. Other cited figures: "173 hours" (PokeAgent paper intro) **[UNVERIFIED, conflicts with 17 days]**; ~424.5 h (LessWrong commenter) is consistent with ~17 days. No dollar cost reported.
- **Harness (Blue era):** Node.js + mGBA; each turn = annotated screenshot + text dump from RAM; 1M context but resets every 100 turns with a summary; persistent goals (primary/secondary/tertiary); textual XML fog-of-war "Map Memory" with a "Reachable Unseen Tiles" list; RAM-injected PP (prompt-only fix for PP hallucination failed); **Critic** fresh-context Gemini instance writing a walkthrough; **Pathfinder** subagent (solved Rocket Hideout spinner maze; hallucinated shortcuts); **Boulder Puzzle Strategist** subagent (sometimes said no solution existed); precomputed spinner end-coordinates.
- **Harness (later, Yellow Legacy/Crystal):** Pathfinder/boulder agents removed, most nav hints removed; meta-tools added: `define_agent`, `run_code`, map markers, notepad, custom tools; World Knowledge Graph added then removed as unhelpful. Crystal: Mental Map, Notepad, Map Markers, Code Execution, Custom Agents/Tools; one-button-per-turn "training wheels".
- **Controversy (verified from TechCrunch/blog):** Claude and Gemini ran *different* harnesses; Gemini's gave tile walkability, object positions, pathfinder/puzzle agents; Joel Z said "Please don't consider this a benchmark", "Gemini and Claude have different tools and receive different information", acknowledged "dev interventions" (e.g. told it to talk to a Rocket Grunt twice for the Lift Key - a game bug/quirk later fixed in Yellow), insisted no walkthroughs. Logan Kilpatrick's "5th badge vs 3, with a different harness" comment drew the criticism. Joel's "iron man suit" analogy (TIME). Interventions in run 1 included manually clearing context after ~8 h looping in Pewter.
- **Notable emergent behavior:** Gemini 3 Pro wrote its own `press_sequence` tool to exploit `autopress_buttons` and bypass the one-button-per-turn rule (author blog); Joel framed loopholes as intelligence tests.
- **Failure modes:** loops past ~100k tokens of context; attractor traps in counter-intuitive mazes; claiming "stuck" with unseen tiles remaining; Full Heal vs Full Restore confusion -> ~100 h loop; "model panic" at low HP (also in Google's 2.5 technical report); cross-game hallucination ("TEA"); Pokegear radio Up/Down vs Left/Right; shutter puzzle order; ignoring map markers; off-screen NPC (Crystal NPC positions only visible on screen, not in RAM-friendly form like Gen 1) broke 2.5 Pro's search.
- **Lessons:** "The scaffolding around an AI is as important as the model itself"; reframing the task ("explore every tile") > model swaps; fresh-context sub-agents help hard sub-problems; telling the model to ignore prior knowledge may have suppressed correct memorized answers (shutter puzzle).

## 3. GPT / o3 plays Pokémon (OpenAI, Clad3815)

- **Links:** [OpenAI Devs launch post (2025-08-07, Twitch GPT_Plays_Pokemon; GPT-5 on Red)](https://x.com/OpenAIDevs/status/1953551715543744547); [CO/AI: GPT-5 beats Red in 6,470 steps (2025-08-15)](https://about.getcoai.com/news/gpt-5-beats-pokemon-red-in-6470-steps-smashing-ai-record/); [Clad3815/gpt-play-pokemon-firered (open-sourced harness)](https://github.com/Clad3815/gpt-play-pokemon-firered); [Clad3815 X post](https://x.com/Clad3815/status/2023000476451570087) (claims GPT-5.2 beat FireRed with no human input; I could not fetch it, snippet only - **[UNVERIFIED]**).
- **Results:** o3: Red in 18,184 steps (>15 days). GPT-5: Red in 6,470 steps (~7 days). Habr cites o3 27,040 vs GPT-5 9,517 steps on **Crystal** (separate run; **[UNVERIFIED]**). GPT-5.1 Crystal: 9,454 steps, 108 h, ~3x human pace (LessWrong). BenchmarkList (self-reported) lists a GPT-5 Crystal run via Codex App Server with custom tools, 16 badges, ~19,517 steps, 2025-08-25 **[UNVERIFIED, self-reported]**. GPT-5.2 on Emerald (6th badge Jan 2026) and FireRed. A commenter claims "GPT-6 beat FireRed in ~18 h" **[UNVERIFIED]**. Not found: any o3-on-Yellow run (the "o3/Yellow" premise did not check out).
- **Harness:** separate harnesses per game (Red, Crystal, Emerald, FireRed). FireRed open-source: mGBA + Lua socket -> Python FastAPI bridge reading memory -> Node.js agent loop; persisted `memory.json`, `objectives.json`, `markers.json`, `map_visits.json`, `summaries.json`, `last_criticism.txt` (self-critique); separate configurable pathfinding model (`OPENAI_MODEL_PATHFINDING`); minimap + fog-of-war. PokeAgent Table 2 lists GPT Plays Pokémon as: frame, location, party stats, money, tile colors; button inputs + self-built minimap; goal tracker + notepad. GPT-5 Crystal live-data page (secondhand) mentioned `path_to_location` for >10-tile moves **[UNVERIFIED]**.
- **Strategy note:** GPT-5 over-leveled one Pokémon and tanked everything (press coverage, Reddit) - brute force over team-building.
- **Lessons:** Efficiency from OpenAI's side was pushed via tools (pathfinding, markers) and prompting for speed; step counts are 3-10x lower than Claude/Gemini but under much richer harnesses, so not a model-only comparison.

## 4. Papers and benchmarks

### 4a. Full-game / long-horizon

- **PokeAgent Challenge** - Karten et al. (31 authors, Princeton/UT Austin/Google etc.), [arXiv 2603.15563](https://arxiv.org/abs/2603.15563), Mar 2026 (v2 later). Two tracks: Showdown battling (20M+ trajectories) and Speedrunning (Pokémon **Emerald**, to Roxanne, 15 milestones). 100+ teams; 22 valid speedrun submissions, 6 reached 100%.
  - Organizer harness: orchestrator + subagents (battle, reflection, gym puzzles, objective verification); MCP tools (A*, buttons, knowledge retrieval); auto context compaction. Same harness across five frontier models: Gemini 3 Flash fastest mean (~2h24), Claude Sonnet 4.5 completes all but 6:25-20:45 variance and 3-4x cost, GPT-5.2 in between; best harness ~1.8x slower than avg human (1:22:05); top human 18 min.
  - Winner Heatz 40:13 (1,608 steps): **LLM writes subgoals -> LLM writes scripted policies per subgoal -> distilled to NN via imitation, refined with DQN ("Scripted Policy Distillation")**. 2nd Hamburg PokéRunners: pure recurrent PPO (LSTM + CNN + 38-d milestone vector, positive rewards only since negatives caused stalling) 1:14:43 to gym 1. Best pure-LLM harness 1:29:17. Deepest: Gemini 2.5 Flash + pathfinding/memory/thinking tools + guidebook, 649 steps (fewest), 2:04:29.
  - Raw VLMs without a harness ~0% completion; CLI agents (Claude Code, Codex CLI, Gemini CLI) lose coherence over long horizons. Open problem "VLM-SLAM" (localization, action-distance, objective detection). Table 2 compares harness dimensions (state/tools/memory/feedback/finetune): the differentiator is **feedback** (critic agent for error recovery).
  - Battling cost: per-game cost varies >70x (GPT-5.2 $1.247/game vs DeepSeek V3 ~$0.017).
- **Continual Harness** - Karten et al., [arXiv 2605.09998](https://arxiv.org/html/2605.09998v1), May 2026. Reset-free self-refining harness (Refiner edits prompt, sub-agents, skills, memory every F steps). Emerald, Gemini 3 Pro: 100% of 31 milestones at $130 median vs $215 for minimalist (-40%); Red (11-milestone subset): between minimal and expert harness. Also Gemma-4 DAgger/SFT co-learning (not converged). Failure: schema-mismatch loop - one invalid tool payload repeated 842 times over 1,003 turns (Yellow Legacy Route 4); after ~500 stalled turns agent stops inventing tools and repeats; weak models (Flash-Lite) do worse with the self-improving harness.
- **Orak** - Park et al. (KRAFTON), [arXiv 2506.03610](https://arxiv.org/abs/2506.03610), ICLR 2026. Pokémon Red scoped to *beat Brock*, 12 story flags; text state from game (location, party, inventory, battle, on-screen text); tool or joypad actions; default reflection+planning+memory agent. Scores: GPT-5 88.9, Gemini-2.5-Pro 83.3, R1 75.0, Claude-3.7 63.9 (72.2 with vision), GPT-4o 38.9, human novice 86.1; small models ~0. Adding images gave mixed results.
- **VideoGameBench** - [arXiv 2505.18134](https://arxiv.org/abs/2505.18134), 2025. 10 1990s games from raw pixels, real-time; best (Gemini 2.5 Pro, Claude 3.7) 0.48% overall / 1.6% Lite. Whether Pokémon Crystal is in the set: the fetched abstract did not list the games, so **[UNVERIFIED]** (I recall Pokémon Crystal being included; confirm in the paper).
- **BALROG** - (Paglieri et al., ICLR 2025). To my knowledge contains BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack, NetHack - **no Pokémon** **[from memory, not re-fetched]**. Relevant only as a methodological comparison for long-horizon text/VLM agents.
- **General Modular Harness** - [arXiv 2507.11633](https://arxiv.org/html/2507.11633v1): perception/memory/reasoning ablations on Sokoban, Tetris, 2048, Candy Crush; **no Pokémon**, but cites it as an extension. Memory helps long horizon, perception helps spatial.
- **PokeGym** - [arXiv 2604.08340](https://arxiv.org/abs/2604.08340), Apr 2026: VLM benchmark on 3D *Pokémon Legends: Z-A* with no game-state access; G-EvoMAC test-time evolution reaches 60.18% (+11 pts over best baseline). Different game family, not Gen 2.
- **Nunu.ai** - case study: Emerald, first 3 gyms in 5 h 15 min livestream (company claim; secondhand). Method details not found.
- **"Show HN: LLM plays Pokémon (open sourced)"** (adenta/fire_red_agent, Mar 2025): RAM-to-text because vision unreliable; useful thread listing loop failures and the argument that a classical planner (GOAP) would beat an LLM and be cheaper.

### 4b. Battle-only (Pokémon Showdown)

- **PokeLLMon** - Hu, Huang, Liu, [arXiv 2402.01118](https://arxiv.org/abs/2402.01118) (Feb 2024): in-context RL from text feedback, knowledge-augmented generation (type chart), consistent action generation (fixes "panic switching"); 49% ladder win rate, 56% invited. Direct precedent for "LLM only where reasoning is needed" being the *battle* layer.
- **PokéChamp** - Karten, Nguyen, Jin, [arXiv 2503.04094](https://arxiv.org/abs/2503.04094), ICML 2025: LLM as action sampler + opponent model + value function inside minimax; GPT-4o 76% vs best LLM bot, 84% vs best rule bot; Llama-3.1-8B 64% vs PokeLLMon; projected 1300-1500 Elo. Dataset 3M+ games.
- **Metamon** (UT-Austin-RPL; Grigsby et al.) - offline RL transformer, Gen 1 OU; used in PokeAgent (PA-Agent builds on it). Not an LLM; shows RL beats LLM on battles. **[details from PokeAgent paper summary; paper itself not fetched]**
- **LLMs as Pokémon Battle Agents** - [arXiv 2512.17308](https://arxiv.org/abs/2512.17308), Dec 2025: custom simplified battle engine, no concrete numbers in abstract.
- PokeAgent finding: Pokémon battling is "nearly orthogonal to standard LLM benchmarks".

### 4c. Hybrid LLM + RL / LLM-guided reward or planning

- **Heatz / Scripted Policy Distillation** (above) - the only hybrid that *won* a Pokémon speedrun track. LLM = subgoal + script author; RL = refinement/speed.
- **Pleines et al., "Pokémon Red via RL"** - [arXiv 2502.19920](https://arxiv.org/abs/2502.19920) (Whidden co-author; PufferLib env ~700 steps/s): PPO to Cerulean; reward shaping is exploitable; no LLM.
- **PokeRL** - [arXiv 2604.10812](https://arxiv.org/abs/2604.10812), Apr 2026: PPO with loop-aware wrapper, map masking, anti-spam, hierarchical dense reward; only first-rival-battle scope; no numbers in abstract. Documents PPO loop/menu-spam failure analogous to LLM loops.
- **Continual Harness co-learning** (Gemma-4 student, Gemini teacher, PRM) - small-scale, not converged.
- LLM-shaped reward on Pokémon specifically: a short Alex Zhang note ("A Simple Framework for Intrinsic Reward-Shaping for RL using LLM Feedback") uses Pokémon with DQN and LLM-written reward code; date/venue unconfirmed **[UNVERIFIED]**. No peer-reviewed LLM-subgoal + PPO paper on Pokémon Red/Crystal found beyond Heatz. STO-RL (arXiv 2601.08107) is general, not Pokémon.

### 4d. Pokémon Crystal-specific

- Gemini 3 Pro Crystal (above); GPT-5/5.1 Crystal (above). **No peer-reviewed paper on a full-game Crystal LLM agent found**; Crystal evidence is blogs/livestreams plus mentions in PokeAgent/Continual Harness related work. PokeAgent says GPT-5.1, 5.2 and Gemini 3 Flash completed Crystal and Emerald.
- Crystal quirks relevant to tensormew (from Joel's blog): NPC positions only reliably visible on screen; Whitney's Miltank is a skill wall; Goldenrod shutter puzzle (Left, Middle, Right); Olivine Lighthouse pits; Pokegear radio controls; Mt. Mortar boulder pushed twice; Blackthorn boulder puzzle.

## 5. Twitch / live streams

| Stream | Model(s) | Game | Notes |
|---|---|---|---|
| ClaudePlaysPokemon | Claude 3.7 -> Opus 4.7 | Red | Minimal harness; hands-off creator |
| Gemini_Plays_Pokemon | Gemini 2.5/3 Pro | Blue, Yellow Legacy, Crystal | Heavy harness, human-in-loop refinement |
| GPT_Plays_Pokemon | o3, GPT-5/5.1/5.2 | Red, Crystal, Emerald, FireRed | Companion dashboard shows prompts/reasoning/tools |
| opusplays.com (fan project, "Claude Opus 4.7 plays Pokémon Red") | Opus 4.7 | Red | Not Anthropic; rules: no human input, no save-scumming, no walkthroughs; shows 0 progress at fetch (new run). Treat as unofficial. |
| Nunu.ai | proprietary | Emerald | company stream |

## 6. Cross-cutting patterns

1. **Harness > model for efficiency.** Same model class: 100x differences in steps between minimal (Claude, ~230k+) and tool-rich (GPT-5.x ~6-10k) harnesses. Heavy harnesses convert a vision/perception task into reasoning over text; minimal ones measure vision.
2. **Perception is RAM-augmented, not pure pixels.** Almost every winner reads RAM/maps into text (walkable tiles, positions, party, PP). Pure-pixel runs (VideoGameBench, "vision_only" scaffold) fail or crawl.
3. **Pathfinding tools shift cost.** A* / pathfinder agents and map markers appear in Gemini (early), GPT, PokeAgent. The LLM's remaining job is goal selection + puzzles.
4. **Memory:** notes/notepad + periodic context reset/summarization is universal. Failure = stale/wrong notes; fix = verification, critic, versioned notes.
5. **Loops are the dominant failure** (hours to days): Pewter backyard 8 h, Erika gym 4 days, Full Heal 100 h, Olivine Lighthouse 16k turns, tool-payload loop 842x. Mitigations: critic subagents, stuck detectors, context resets, anti-loop wrappers.
6. **Hallucinated map/objects:** elevators, shortcuts from pathfinder, "no solution" from puzzle agents, cross-game memory (TEA).
7. **Puzzle gadgets are the real walls:** spinner mazes, boulders/switches (Victory Road), Card Key / Secret Key (Cinnabar, Silph), shutters, Whitney. These are where agents spend 10-1000x the steps of ordinary overworld.
8. **LLMs lose speed battles to RL/scripts** (PokeAgent): LLM as prior/planner + code/RL as executor wins.
9. **Metric hygiene problem:** steps vs turns vs actions vs hours incl. API downtime; no one reports $ per milestone for full games except Continual Harness (Emerald $130-215) and PokeAgent (battle $/game).

---

## Comparison vs tensormew's LLM-on-triggers design

tensormew (SPEC.md): Pokémon Crystal on PyBoy; deterministic greedy mover + tile macros + battle macros (0 LLM tokens on routine steps); LLM supervisor invoked only on triggers (`CUT_BLOCKED`, `STUCK`, `BATTLE`, `DIALOGUE`), emitting *tools with a required `reason`*, never raw buttons; checkpoint chain of `.state` files; decision-trace JSONL; planned PPO (`rl_env`, `ppo`) for natural-progress rewards; baseline = LLM-every-step on same checkpoints for calls/tokens/$.

| Axis | Claude / Gemini / GPT streams | PokeAgent / Continual Harness | tensormew |
|---|---|---|---|
| Who acts each step | LLM, every turn (button or short sequence) | LLM orchestrator + subagents; A* tool | Deterministic mover/macros; LLM only on triggers |
| LLM output type | buttons / small sequences / tool calls | MCP tool calls | tool calls with `reason`, no raw buttons (closest to PokeAgent MCP, but stricter) |
| Perception | screenshot + RAM text | frame + map/party | RAM `GameState.to_text()` (Crystal-specific symbols), PyBoy adjacent tile bytes |
| Navigation | LLM + pathfinder subagent/tool | A* tool | Greedy explore of unvisited tiles; `go_to` only when LLM names a tile; no route table |
| Memory | notes + summaries + context resets | KB + compaction + critic | Checkpoint registry + decision log; LLM call is mostly stateless per trigger (needs verification: SPEC does not describe an LLM notes file) |
| Cost model | Unbounded; 1-2B tokens per game (Gemini 3 Pro Crystal) | $130-215/Emerald (to full milestones); battle $/game reported | Explicitly measures calls/tokens/$ per milestone vs every-step baseline |
| Starting state | New game | New game / to gym 1 | Pre-seeded checkpoints (human `.sav` imports); "furthest badge" metric, not full autonomy |

**Closest relatives:** (a) Heatz/SPD (LLM-authored scripts, RL refinement) - tensormew's macros are hand-written rather than LLM-generated, but the "LLM as sparse planner, code as executor" thesis is the same; (b) Continual Harness - reduces cost by ~40% with a *skill library*, analogous to macros; (c) Deepest (PokeAgent) - training-free, pathfinding + guidebook, fewest steps (649). (d) Orak/PokéChamp/PokeLLMon - LLM confined to battle reasoning.

**What prior work implies for tensormew's risks:**
- Trigger-based calls will concentrate exactly where prior agents fail: puzzles (Cut tree, boulders/Strength, Whitney, shutter puzzle). Expect the LLM to be called repeatedly in the same place; add a per-trigger retry cap and critic-style escalation (Gemini's Critic, PokeAgent's feedback agent). Continual Harness's 842-repeat tool-payload loop shows the schema-validation/looping risk exists even for tool-only LLMs.
- A deterministic explorer inherits Gen-2-specific pitfalls: NPC positions visible only on screen, off-screen NPCs breaking routes (Joel's 2.5 Pro lighthouse), spinners, ice, one-way ledges.
- Stuck detection quality = the quality of the trigger function; no prior harness has published trigger precision/recall.

## Novelty gaps (what tensormew could claim, carefully)

1. **Calls-per-milestone accounting.** No published full-game Pokémon run reports LLM calls, tokens and $ per milestone on identical checkpoints for both a trigger-gated and an every-step policy. Closest: Gemini 3 vs 2.5 Pro token comparison (1.88B vs projected 15B+, different models, not controlled), Continual Harness $/milestone on Emerald. A same-checkpoint A/B on Crystal is new. Caveat: must define "step/turn" explicitly given the field's inconsistency.
2. **Gen 2 / Crystal full-game evidence is anecdotal** (blogs, streams). A reproducible, open Crystal harness with RAM symbols and checkpoints (PokeAgent covers Emerald; Continual Harness covers Red/Emerald) fills a gap. Needs the sym/ROM-version spec (US 1.1) for reproducibility.
3. **Trigger-gated supervision as an explicit design variable.** Prior work has either LLM-every-turn or scripts-first-then-distill; none ablate *when* to call the LLM (stuck / blocked-by-HM / battle boss / dialogue). Frame "how many steps actually needed an LLM" as a measurable fraction.
4. **Tool-calls-with-`reason` decision trace** shared across LLM, script, and RL actors (`logs/decisions.jsonl`) - an interpretability/audit angle that streams show only as CoT text.
5. **Hybrid: trigger-gated LLM + natural-progress PPO** (new tiles/maps/badges) in a single trace. Prior hybrids: Heatz (offline LLM scripts), Continual Harness (Gemma co-learning). PokeRL/Pleines are pure RL. No LLM-supervised PPO on Crystal published that I found.
6. **Checkpoint-chain evaluation** (segment A->B success from labeled `.state` files) - gives variance estimates per segment, versus the single-run anecdotes (n=1) that dominate the literature.

**Honest limits on novelty:** deterministic pathfinding + LLM subgoals is established (PokeAgent winners); "LLM only when needed" resembles Gemini's gradual harness slimming and Continual Harness skills. The defensible novelty is the *controlled cost-per-milestone measurement* and Crystal-specific open harness, not the architecture idea itself. Pre-seeded checkpoints from human saves also means tensormew cannot claim end-to-end autonomy (SPEC already states this).

## Open items to verify before citing

- Anthropic's own account of Opus 4.7 completion (hours/steps/$); only LessWrong/press found.
- "173 hours" Gemini 3 Pro Crystal (likely wrong); "GPT-6 FireRed 18 h"; GPT-5.2 FireRed beat; GPT-5 Crystal 9,517 vs 19,517 steps (two sources disagree).
- VideoGameBench game list (is Crystal included?), BALROG game list (assumed no Pokémon).
- Metamon and PokeLLMon details beyond abstracts.
- Whether Claude's harness exposes a dedicated navigator tool vs. highlighted tiles only (sources conflict: PokeAgent Table 2 says pathfinder + KB; LessWrong comment says tiles highlighted).
