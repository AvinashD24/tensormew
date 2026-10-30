# Venue and publication plan: full Pokémon Crystal clear agent

Prepared 2026-10-19. Facts are tagged with where they came from. `[UNVERIFIED]` means a secondary source only, an inference, or something I could not open. Official NeurIPS pages were fetched for 2026 policy. Nothing for 2027 is official yet.

## 0. Bottom line

1. **ICLR 2027 (Sep 25, 2026) and AAAI-27 (Jul 28, 2026) are closed.** NeurIPS 2026 was decided on Sep 24 and its workshop deadlines (late Aug to early Sep) have also passed. The next realistic cycle is **NeurIPS 2027, expected abstract May 14 and full paper May 21, 2027 [UNVERIFIED: third-party projection, official call not yet published]**. That gives about 30 weeks.
2. **The premise "a NeurIPS 2025 Pokémon Red paper" did not verify.** `research/README.md` says the "NeurIPS 2025 Pokémon Red RL paper". I could not find a Pokémon Red paper in either the Main track or the Datasets & Benchmarks track. Details in section 2. Do not cite it as evidence of what reviewers accept until someone finds the actual OpenReview link.
3. **A bare "we beat Crystal" is incremental.** A streamed Gemini 3 Pro run reportedly cleared Crystal in Dec 2025 (blog and X posts, not peer reviewed). Claude and GPT agents have reportedly cleared Red/Blue. The paper needs a measured, reproducible, cost-or-evaluation-science contribution on top of the clear.
4. **Our verified progress is far from a full clear.** `research/analysis/codebase-audit.md` shows the furthest verified point is the Zephyr Badge (1 badge), mostly scripted. A full clear by April 2027 is a large engineering risk. The plan below has go/no-go gates and a fallback that does not require a full clear.
5. **Recommended path:** target NeurIPS 2027. Prefer the Evaluations & Datasets (E&D) track with a benchmark/audit framing (Framing A) that uses our agent as the flagship reference agent. Keep the Main track efficiency framing (Framing B) as the alternative if the clear is clean and cheap. Decide the track by March 31, 2027. A paper cannot go to both.

---

## 1. Venue status and calendar

### 1.1 NeurIPS 2026 (verified, official)

| Item | Value | Source |
|---|---|---|
| Abstract / full paper deadline | May 4 / May 6, 2026 AoE, same for Main, E&D and Position tracks | neurips.cc CallForPapers, newsletter Apr 2026 |
| Author notification | Sep 24, 2026 | neurips.cc Dates |
| Location assignment notices | Oct 6, 2026 | neurips.cc Dates |
| Locations | Sydney (main). Atlanta and Paris are official satellites. Main conference in Sydney is Dec 11-12 and Dec 6-12 overall. The dates page and the newsletter differ by about a day for satellites (Atlanta Dec 8/9-13, Paris Dec 9-13). | neurips.cc Dates, newsletter Apr 2026 |
| Workshop paper deadlines | "Suggested submission date" Aug 29, mandatory notification Sep 29. Individual workshops extended to Sep 5-7. All closed. | neurips.cc Dates, workshop sites |
| Competition track | Submission deadline May 15, 2026, results due Oct 31 | neurips.cc Dates |
| Track rename | "Datasets & Benchmarks" is now **"Evaluations & Datasets" (E&D)** in 2026 | NeurIPS blog 2026-03-23 |

Workshops seen for NeurIPS 2026, all closed:
- CL4FM Agents: deadline Sep 7, non-archival, Sydney Dec 11-12.
- TTCL Agents: deadline Sep 5, non-archival, Atlanta Dec 13.
- PTA, AIWILD, SEA: all closed.

I found no 2026 edition of "Foundation Models for Decision Making", "Language Gamification" or "Wordplay". Language Gamification existed at NeurIPS 2024 (verified). Whether these workshops recur in 2026 or 2027 is `[UNVERIFIED]`.

### 1.2 NeurIPS 2027 (expected, not official)

- A third-party tracker (opencurious.com, last checked Oct 8, 2026) projects **abstract May 14, 2027 and paper May 21, 2027 AoE, location "Europe", city and dates not announced**. The tracker says these dates are extrapolated and not official. `[UNVERIFIED]`
- NeurIPS 2026 used a 2-day gap between abstract and full paper. The 2027 projection used a 7-day gap. Plan against **May 14** as the effective deadline, because a missed abstract registration usually blocks the full paper (same tracker caveat).
- Assume the 2026 rules carry over (9 pages, checklist, double-blind, etc.) until the 2027 call is published. `[UNVERIFIED for 2027]`
- Expect workshop deadlines about late Aug to early Sep 2027 and decisions about late Sep 2027, by analogy. `[UNVERIFIED]`

### 1.3 Alternative venues (what is open from today)

| Venue | Status as of 2026-10-19 | Fit for us |
|---|---|---|
| ICLR 2027 | Abstract Sep 18, paper Sep 25, 2026. **Closed.** Decisions Dec 16, 2026. Conference Apr 26-30, 2027, California per most sources, official dates page says TBA. | Missed. Note it introduced reciprocal reviewing and AI-use policies. |
| ICML 2027 | Official call not found. Third-party projection: abstract Jan 16, paper Jan 22, 2027, South America, dates unannounced. `[UNVERIFIED]` | About 13 weeks away. Too soon for a full clear. Only viable if we submit an analysis or benchmark paper without a full clear. |
| AAAI-27 | Main track closed (full paper Jul 28, 2026). Journal track deadline Oct 30 or 31, 2026 (pages disagree) and needs an already-published journal paper, so not applicable. Conference Feb 16-23, 2027, Montréal. | Closed. AAAI-28 deadline is probably about July 2027 by analogy. `[UNVERIFIED]` Good fallback. |
| RLC 2027 | No 2027 call found. RLC 2026 had abstract about Mar 2 and paper about Mar 6, 2026, with notification May 8. Two sources disagree by a day on the dates. RLC 2025 acceptance was 115 of 295 (39%). `[UNVERIFIED for 2027]` | Strong early venue if timed (Mar 2027). RL-native audience, higher acceptance, appreciates honest empirical work. Probably too early for a complete clear. |
| IEEE CoG 2027 | No 2027 call found. CoG 2026 (Madrid, Sep 1-4) had full papers Mar 17, 2026, auxiliary papers May 14, 2026, notification May 1. 2025 was Mar 15. `[UNVERIFIED for 2027]` | Good fallback for a complete game-AI paper. Auxiliary and demo tracks give extra late options. |
| AIIDE 2027 | No 2027 call found. AIIDE 2026: abstract about Jun 19, full about Jun 26, 2026, notification Aug 7, held in Belo Horizonte, Brazil on Nov 9, 2026. `[UNVERIFIED for 2027]` | Good fallback, game-AI audience, lower bar for engineering-heavy systems. |
| AAMAS 2027 / ALA workshop | AAMAS 2027 is May 3-7, 2027, Hanoi (workshops May 3-4). Workshop proposals were due Oct 29, 2026. Recommended workshop paper date Jan 24, 2027. ALA's own 2027 call not found. | ALA is a cheap early venue if it runs. `[UNVERIFIED]` |
| NeurIPS 2027 workshops | Not announced. | Plan a non-archival workshop version of any early results (Aug-Sep 2027). |
| TMLR / arXiv | Rolling. | TMLR is a legitimate no-deadline fallback for a solid empirical paper. `[UNVERIFIED: not checked this session]` |

### 1.4 Acceptance context

- NeurIPS 2025: 21,575 valid main-track submissions and 5,290 accepted (about 24.5%). D&B had 1,995 submissions. Source is a secondary summary (intuitionlabs, via search). `[UNVERIFIED]`
- Do not rely on any particular acceptance rate for the E&D track. NeurIPS says it "aims for an equally stringent review as the main conference track".

---

## 2. What reviewers have accepted for Pokémon: the "NeurIPS 2025 Red paper" check

What I could find:

| Item | What it is | Venue status |
|---|---|---|
| Pleines et al., "Pokemon Red via Reinforcement Learning" (arXiv 2502.19920; v1 titled "Playing Pokémon Red via Deep Reinforcement Learning") | PPO baseline on Red with the PokemonRedExperiments/Puffer lineage. Paper is about reward shaping and its exploits. Reaches an early-game point (Cerulean City per the search summary). | arXiv comment says "under review". **No venue listed. I found no NeurIPS 2025 acceptance.** |
| Karten et al., "The PokéAgent Challenge: Competitive and Long-Context Learning at Scale" (arXiv 2603.15563) | Benchmark and competition paper. Battling track and Speedrunning track (Emerald, standardized milestones, open-source orchestration for LLM harnesses). Over 100 teams. Reports a gap between generalist LLMs, specialist RL and elite humans. | Tied to the **NeurIPS 2025 Competition track**. Also appears as a poster at NeurIPS 2026 (Dec 9, 2026 Sydney). The track is not stated on the poster page. `[UNVERIFIED]` |
| Orak (arXiv 2506.03610) | 12-game LLM-agent benchmark, includes Pokémon Red per a search summary. | No venue listed on arXiv. |
| "Stop Comparing LLM Agents Without Disclosing the Harness" (arXiv 2605.23950) | Position paper. Harness variance can exceed model variance on long-horizon tasks. Proposes a disclosure standard. | arXiv only. |
| OpenReview search "Pokemon Red" | Only a DEMO 2026 (ICML workshop) poster on JEPA pretraining for Pokémon Red, plus rejected or withdrawn ICLR 2024/2025 submissions on Pokémon LLM work. | None in NeurIPS 2025 main or D&B. |

Conclusion for the venue decision:
- **Verified:** the closest NeurIPS-endorsed Pokémon artifact is a competition/benchmark paper, not a method paper on Red.
- **Verified:** several Pokémon papers were rejected or withdrawn at ICLR 2024/2025 (OpenReview). That is a warning that "agent plays Pokémon" with a thin contribution does not clear a top-venue bar.
- **Not verified:** that any Red agent paper appeared in NeurIPS 2025 main or D&B. Ask whoever wrote `research/README.md` for the exact title or link. If it is the Pleines paper, correct the note to say it is a preprint with no confirmed venue.
- The literature gap that the field itself names: earlier Pokémon demos used different games (Red, Blue, Crystal, Emerald), different harnesses and different success criteria, so they cannot be compared (PokeAgent paper, related work). A standardized, harness-disclosed Crystal result is therefore a real contribution.

Other Pokémon-adjacent context (not peer-reviewed unless stated):
- Gemini 3 Pro reportedly earned 16 badges, beat the Elite Four, Champion and Red in Crystal, in a streamed run by Joel Zhang (Dec 15, 2025). It reportedly used about half the tokens and turns Gemini 2.5 Pro needed to get 4 badges. `[UNVERIFIED]` Only reposts and a Hacker News thread were read, not the original blog.
- Claude Plays Pokémon on Red: Opus 4.5 reportedly stalled at Victory Road (LessWrong follow-up). A Manifold market says Opus 4.7 later beat Red and the Elite Four. `[UNVERIFIED]`
- The PokeAgent paper's related work reports Gemini 2.5 Pro finishing Blue in 406 hours and a 173-hour Gemini 3 Pro completion, and GPT-5 finishing in 6,470 steps. It cites these via press links. `[UNVERIFIED]`

---

## 3. NeurIPS formatting and policy checklist (2026 rules, assume 2027 is similar)

All of the following is from the NeurIPS 2026 Main Track Handbook or the E&D call, fetched 2026-10-19, unless tagged.

| Topic | Rule |
|---|---|
| Page limit | 9 content pages for the main text, including figures and tables. References, optional appendices and the checklist do not count. +1 page at camera-ready. Max file 50 MB. Over-limit or modified margins and fonts can be desk-rejected. LaTeX style only (Word template discontinued). |
| E&D page limit | The E&D page states none and defers to the main call for formatting. So 9 pages is the working assumption. |
| Contribution types (Main) | Choose one: General, Theory, Use-Inspired, Concept & Feasibility, Negative Results. Reviewers apply type-specific criteria. "The bar for [Negative Results] is expected to be high." |
| Double blind | No identifying info in paper, supplement or code. No acknowledgments at submission. Cite own work in third person. Violations are desk-rejected. |
| E&D blinding | Default double-blind. Single-blind is allowed if the artifact cannot be anonymized (collection, hosting, or infrastructure reveals affiliation); select it in the form. If a benchmark builds on an existing codebase, make a best-effort anonymization. |
| arXiv-first | Non-anonymous preprints do **not** cause rejection. Use the `preprint` option of the style file. Do not write "Under review at NeurIPS". Reviewers are told not to search for preprints. "Aggressive advertising" of a submission may be treated as a violation. |
| Reproducibility checklist | Mandatory paper checklist in the template. Submissions should include information to reproduce results (code, data, weights, compute description). |
| Code | Anonymized ZIP up to 100 MB. Encouraged. Should be self-contained and executable, otherwise explain why (e.g. non-free libraries). De-anonymized links for accepted papers. |
| E&D code and data | Hosted, accessible and documented at submission, **mandatory** when the main contribution is a reusable executable artifact such as an environment, benchmark suite or tool. Materials must be available to reviewers without a personal request to the authors, and non-compliance can justify desk rejection. Croissant metadata (core plus Responsible AI fields) is required for dataset contributions. Datasets over 4 GB need a small sample. For analysis or empirical work without such an artifact, code is not mandatory if there is a code-submission justification and the claims can be verified. |
| LLM use by authors | Describe agent or LLM use in the experimental setup if it is an important, original or non-standard part of the method. Spell check, grammar and basic code assistance need not be documented. Authors are responsible for all content, including fabricated references. LLMs cannot be authors. Prompt injection aimed at reviewers is prohibited. Our LLM supervisor is part of the method, so disclose it fully. |
| LLM detectors | The 2026 **Position Paper track** desk-rejected 178 of 969 submissions (18.4%) using Pangram AI-detector scores and required papers to be "substantially human-written". Backlash followed over the detector's validity. Whether this extends to Main or E&D in 2027 is `[UNVERIFIED]`. The prudent assumption is that heavily LLM-written prose is a risk. Write the paper text by hand. |
| Reviewer LLM use | E&D bans reviewer LLM use. Main track ran a separate experiment with sanctioned LLM support. |
| Dual submission | Overlapping archival submissions count as prior work. Main and E&D at once is prohibited. A paper under review cannot go to another archival venue. **Non-archival workshops are allowed.** Thin slicing is prohibited. |
| Reciprocal reviewing | Authors who are assigned as reviewers lose access to reviews of their own papers until they finish reviewing. Every author needs an OpenReview profile, and profiles without an institutional email can take up to two weeks to moderate. Create profiles by April 2027. |
| Optional | Google "Paper Assistant Tool" gave authors free private pre-submission feedback in 2026. Whether it returns in 2027 is `[UNVERIFIED]`. |

---

## 4. Code, ROM and save-state legal issues

This is not legal advice. Check with the university or employer counsel before release.

### 4.1 How prior work handled it

- **PokemonRedExperiments / Puffer lineage (verified from README):** the repo does not ship a ROM. The user must supply a legally obtained ROM and rename it. The README gives a SHA-1 (`ea9bcae617fdf159b045185467ae58b2e4a48b9a`) so users can verify they have the right file. This is the community standard.
- **Third-party Hermes/Nous pokemon-agent skill (search summary):** states the user must provide their own ROM and that the agent cannot download or distribute ROMs.
- **PokeAgent Challenge (verified):** "all datasets, baselines, and infrastructure are publicly released", hosted on GitHub and Hugging Face, with a self-contained local emulator and milestone framework. The visible text had no ROM statement. How they handle the Emerald ROM is `[UNVERIFIED]`. Read their repo README before copying their approach.
- **Gemini and Claude stream runs:** no ROM is distributed. They are videos or logs.

### 4.2 Legal landscape (secondary sources)

- Distributing a commercial ROM is copyright infringement; owning the cartridge does not authorize distribution. Several law-oriented explainers agree. `[UNVERIFIED as legal opinion]`
- Nintendo has litigated hard against ROM-distribution sites and emulator makers. A secondary blog lists LoveROMs/LoveRetro ($12.23M, 2018) and Yuzu ($2.4M, 2024). `[UNVERIFIED: figures not checked against court records]`
- A Nintendo attorney reportedly conceded in 2025 that emulation software alone is not illegal. `[UNVERIFIED]`
- I found no case law or guidance specific to save states or research datasets. `[UNVERIFIED]`

### 4.3 Problems in the current repo (verified by `git ls-files`)

- **`rom/pokemon_crystal.gbc` and `rom/pokemon_crystal.sym` are tracked in git.** If this repo is or ever becomes public, that is a ROM distribution. Remove them from the tree and from history (e.g. `git filter-repo`) before any release, and add `rom/` to `.gitignore`.
- **`.gitignore` explicitly un-ignores `recordings/**/*.state`**, and many `.state` files are tracked. A PyBoy save state is a snapshot of emulated memory and may contain copyrighted game assets (graphics tiles, text). `[UNVERIFIED: not inspected]` Treat states as potentially derivative and do not publish them. Publish instead:
  - action traces (button sequences plus RNG and RTC seeds),
  - RAM-derived event logs and milestone timestamps,
  - a replay script that verifies the user's ROM SHA-1 and emulator version.
- **Downloaded save files:** the SPEC mentions importing community `.sav` files. Do not redistribute them without checking terms.
- **`.sym` file:** derived from the community pokecrystal disassembly (pret). Whether publishing symbol addresses is acceptable is `[UNVERIFIED]`. Safer to publish a table of RAM addresses with our own labels and cite the disassembly.
- **`tensormewweb/` telemetry is hand-authored** (audit finding). Do not link it from the paper, because reviewers would treat fabricated-looking telemetry as an integrity problem.

### 4.4 Recommended release design

1. Public repo: code, configs, prompts, trained policy weights (check that weights were not trained on states that embed assets, and keep to model files only), evaluation harness, replay logs.
2. Bring-your-own-ROM with SHA-1 check and a clear README line. No ROM, no states, no sav files.
3. Report a determinism guarantee: same ROM SHA-1, emulator version (PyBoy), seed and action trace reproduce the milestone log.
4. For reviewers: they must obtain the ROM themselves. Put this in the checklist and the E&D "Code Submission Justification", and provide replay logs, videos and milestone tables so claims can be checked without the ROM. For E&D this is the largest compliance risk because the rules say artifacts must be accessible without personal requests. A benchmark that needs a ROM the reviewer must find is a known weak spot. Raise it explicitly in the paper's limitations and ethics statement.
5. Screenshots in the paper: normal in the literature, but keep them small and few. `[UNVERIFIED]`
6. Add a "game IP" paragraph in the ethics section: non-commercial research, no ROM redistribution, trademarks belong to Nintendo, Game Freak and The Pokémon Company.

---

## 5. What contribution types win for game-playing papers

| Type | Fits | Strength | Weakness | Evidence |
|---|---|---|---|---|
| New method | Main (General / Use-Inspired) | Highest ceiling | Needs a clear gain over strong baselines and ablations. A hand-scripted agent is not seen as a method. | ICLR 2024/25 Pokémon LLM submissions were rejected or withdrawn (OpenReview). |
| Benchmark / environment / protocol | E&D | No SOTA needed. "No new model is required, and the work need not outperform prior methods." Must state evaluative claims, assumptions and limits. | Needs accessible artifacts. ROM licensing complicates this. | NeurIPS E&D call, 2026-03-23 blog. |
| Analysis / audit | E&D, or Main (Concept & Feasibility) | Explicitly in scope: failure modes, reproductions, audits, stress tests, harness effects. | Needs a sharp, testable claim. | E&D call. |
| Negative results | E&D, or Main (Negative Results) | Explicitly welcome. | Main-track bar is stated as high. | Main Track Handbook, E&D blog. |
| Competition paper | E&D | Prior competition papers are in scope. | We did not run a competition. | E&D blog. |
| Demo / system | CoG, AIIDE | Lower bar. | Lower visibility. | Venue norms. `[UNVERIFIED]` |

---

## 6. What makes a Pokémon paper novel beyond "we beat the game"

The honest status: LLM-harness agents reportedly clear Crystal and Red already, and RL work covers Red. So "full clear" is a table-stakes result, not the contribution. Candidate novelty axes, ranked by fit with our codebase and the SPEC thesis ("How many steps actually needed an LLM?"):

1. **Cost-controlled progress.** Full-game clear with a hybrid of scripted macros, learned policies and event-triggered LLM calls. Report tokens, dollars and wall-clock per milestone against LLM-every-step baselines under the *same* harness. The SPEC already warns against unverifiable total-game dollar claims, and `tools/compare_cost.py` currently uses a fake trace, so this is unmeasured today.
2. **Harness-controlled evaluation.** The field's own gap (PokeAgent paper, arXiv 2605.23950): comparisons mix models and harnesses. A Crystal milestone suite with an explicit harness-disclosure card, and a variance decomposition (model vs harness vs seed), is publishable on its own.
3. **Memorization/contamination probes.** Crystal walkthroughs are all over the web. Test the same agent on modified games (randomized or relabeled maps, shuffled item locations, renamed locations) and measure the drop. Whether suitable Crystal randomizer or hack tooling exists and is legally usable is `[UNVERIFIED]`. Note that this needs a patched ROM, which raises the same distribution problem (distribute the patch, not the ROM).
4. **Autonomy audit ("what counts as beating the game?").** Define assistance levels: RAM reads, RAM writes, RTC lock, save states, scripted macros, hardcoded route tables, and LLM prompts carrying walkthrough text. Re-run the same agent at each level. Our codebase currently uses several (`equip_balls` writes Poké Balls into RAM, `rtc_lock=True`, a Route 29 walkthrough in the system prompt, hardcoded gym step tables), so this has to be reported honestly anyway. Turning that into a measured "assistance ladder" makes a weakness into a contribution.
5. **Component attribution.** Per-stage attribution of which component (rule, RL, LLM) solved each segment, with counterfactual replacement. The audit already flags this as mildly distinctive.
6. **Crystal-specific long-horizon structure.** Day/night and RTC-dependent events, phone calls, the Bug-Catching Contest, Kanto post-game, and Red at Mt. Silver. These give a harder, less-contaminated long-horizon test than Red. Weak as a standalone reason, good as supporting detail.
7. **Reward-hacking taxonomy.** Pleines et al. already document reward exploits on Red. A Crystal replication with new failure modes (event-flag rewards, whiteout loops) is incremental unless paired with a mitigation that is shown to work.

---

## 7. Three recommended framings

### Framing A (recommended): Evaluation and audit, E&D track
**Title:** *CrystalBench: Harness-Disclosed, Contamination-Aware Evaluation of Long-Horizon Agents on Pokémon Crystal*
- Contributions: (i) milestone suite derived from RAM event flags covering all 16 badges, Elite Four, Champion and Red; (ii) autonomy-level protocol (assistance ladder); (iii) harness-disclosure card and variance decomposition; (iv) reference agents including our hybrid full-clear agent, a pure-LLM harness and a PPO baseline; (v) replay-based, ROM-free verification of reported results.
- Why: E&D explicitly welcomes audits, harness analyses, RL environments and negative results, and does not require SOTA. A full clear is a reference point, not the sole claim.
- Main risk: the ROM accessibility requirement (section 4) and overlap with PokeAgent Challenge (position as complementary: Crystal, harness-disclosed, contamination probes, cost).

### Framing B: Method and efficiency, Main track (Use-Inspired or General)
**Title:** *How Many Steps Need a Language Model? Event-Triggered Supervision for Full-Game Pokémon Crystal at a Fraction of the Cost*
- Contributions: an event-triggered LLM supervisor over scripted and learned low-level control that clears Crystal; Pareto frontier of progress vs tokens, dollars and wall-clock against LLM-every-step baselines and pure RL; trigger and component ablations; per-segment attribution.
- Why: matches the SPEC thesis; a quantitative cost claim is the strongest "beyond the clear" result available.
- Main risk: reviewers call it "engineering plus scripting" unless the scripted share is small, measured and honestly reported, and unless baselines are strong. Needs the clear to be real (not segment-by-segment from loaded states).

### Framing C: Analysis / negative results, E&D or Main (Negative Results / Concept & Feasibility)
**Title:** *What Does "Beating Pokémon" Mean? Assistance, Memorization and Reward Hacking in Long-Horizon Game Agents*
- Contributions: taxonomy of assistance, replication across agent families (LLM harness, RL, hybrid) at each assistance level, memorization probes, reward-exploit catalog, and a recommended reporting standard.
- Why: lowest dependence on a flawless clear; works even if only 4 to 8 badges are achieved autonomously. Complements arXiv 2605.23950.
- Main risk: Negative Results bar is stated as high; needs a crisp, falsifiable thesis and clean experiments, otherwise reads as an opinion piece.

**Recommendation:** write Framing A as the paper, borrow B's cost analysis as a section, and keep C's audit as the core of A's protocol. If the clear is cheap and clean by Mar 31, 2027, consider switching the lead to B for the Main track.

---

## 8. Backward-planned timeline (today Mon 2026-10-19 to deadline)

Planning deadline: **abstract Fri 2027-05-14, paper Fri 2027-05-21** (expected, unofficial). Re-check neurips.cc monthly; the real call will usually appear by about March/April. If official dates shift, slide the plan.

| Date | Milestone | Gate / decision |
|---|---|---|
| Oct 19 - Nov 15, 2026 | Fix the premise: locate the "NeurIPS 2025 Red paper" or correct notes. Read PokeAgent repo and harness. Remove ROM and states from git history. Create OpenReview profiles for all authors. Write the autonomy/assistance ladder definitions and log which assists the code uses today. | Legal cleanup done before any public push. |
| Nov 16 - Dec 31 | Engineering: battle system (type data from RAM, STAB, status), HM/field moves (Cut, Surf, Strength, Flash, Whirlpool, Waterfall, Fly), progression planner with flags for all 16 badges. Build milestone logger and cost/token logging that writes real traces. Replace the fake trace in `tools/compare_cost.py`. | **Gate 1 (Dec 31):** 4 Johto badges autonomously with real logs. If not met, drop Framing B and re-scope to A/C on partial progress. |
| Jan 1 - Feb 28, 2027 | Reach 8 Johto badges plus Elite Four. Stand up baselines (LLM-every-step, PPO event-flag baseline, scripted-only). Draft evaluation protocol and harness card. Early partial results to an RL or games venue if one is open (ALA at AAMAS 2027, workshop paper date about Jan 24 `[UNVERIFIED]`; RLC 2027 around early March `[UNVERIFIED]`). | **Gate 2 (Feb 28):** 8 badges and Elite Four in at least one end-to-end run. ICML 2027 (about Jan 22 `[UNVERIFIED]`) is not realistic for a full clear; skip it unless a pure analysis paper is ready. |
| Mar 1 - Mar 31 | Full clear including Kanto badges and Red. Freeze the agent. Post an arXiv tech report in preprint style if a competing result appears; otherwise after submission. | **Gate 3 (Mar 31):** at least 1 full clear in a logged run. Decide track (A/E&D vs B/Main). If no clear: submit A or C without a clear claim, or aim fallback venues (CoG about Mar 2027, then AAAI-28 / AIIDE 2027 around June to July `[UNVERIFIED]`). |
| Apr 1 - Apr 20 | Run the full experiment matrix (section 9): 5 seeds main config, ablations, baselines, contamination probes. Budget LLM spend now; API models change, so pin versions and log all prompts and responses. | Freeze experiments by **Apr 20**. |
| Apr 21 - May 4 | Write. Human-written prose, figures, tables, related work, limitations. Fill the paper checklist. Anonymize repo and supplement, build the ≤100 MB code ZIP, Croissant file if E&D has a dataset. Run internal mock review. | Full draft by **May 4**. |
| May 5 - May 13 | Revise, proof, remove identifying info, verify page count (9 pages), verify references exist (hallucinated citations are an author responsibility). | Abstract and title registered by **May 12**, two days of margin. |
| **May 14** | **Abstract deadline (expected).** | |
| May 15 - May 20 | Final polish, upload supplement, check PDF in OpenReview preview. | |
| **May 21** | **Full paper deadline (expected).** Aim to submit one full day early. | |
| After May 21 | Post arXiv (preprint mode, no "under review at NeurIPS"). Do not market aggressively during review. Prepare a non-archival NeurIPS 2027 workshop version if the main submission is rejected (Aug to Sep 2027, `[UNVERIFIED]`). | |
| About late Sep 2027 | Decision (by analogy with Sep 24, 2026). `[UNVERIFIED]` | |

---

## 9. Required experiments, ablations, seeds and baselines

### 9.1 Primary evaluation
- Success metric: fraction of a fixed list of RAM-event milestones reached (16 badges, Elite Four, Champion, Red). Time-to-milestone in emulator steps, wall-clock, tokens and dollars.
- **Seeds:** at least 5 independent full-game runs per main configuration, with different RNG and different LLM sampling seeds. At least 3 per ablation at full-game level, and 10 or more per segment where segments are evaluated from checkpoints. Report completion rate, median and IQR, and bootstrap confidence intervals (stratified bootstrap as in `rliable`). The NeurIPS checklist asks about statistical significance and error bars.
- Report **failures**: where runs stall, how often a human or script had to intervene (count interventions; target zero).
- Fix and report the emulator version, ROM version (Crystal US 1.1, SHA-1), RTC policy and save-state policy. The final results must **not** use locked RTC, RAM writes (the current `equip_balls` cheat) or loaded intermediate states. If they do, say so and put them on the assistance ladder.

### 9.2 Baselines
1. LLM-every-step harness (same tools, same observation, same model) with at least two model sizes, one frontier and one cheaper, with model and version pinned.
2. LLM-on-triggers (ours) vs. varied trigger thresholds.
3. PPO with event-flag reward adapted to Crystal (pokemonred_puffer lineage), and a pure scripted agent without any LLM.
4. Our hybrid without RL (scripts + LLM) and without LLM (scripts + RL).
5. Reference points from literature, labeled as non-comparable unless re-run: the Gemini Crystal streamed run, Claude Red runs, PokeAgent speedrun baselines. `[UNVERIFIED]` for streamed results.
6. Human reference: step count for a standard casual and a speedrun-style playthrough. `[UNVERIFIED]`

### 9.3 Ablations
- Assistance ladder: toggle RAM reads, RAM writes, RTC lock, hardcoded route tables, walkthrough text in prompts, scripted gym step tables.
- Observation modality: RAM only vs pixels vs both.
- Trigger design: drop each trigger; vary stuck thresholds; LLM frequency vs progress.
- Memory design for the LLM (notes, summaries) and context length.
- Model swap and harness swap (to produce the model-vs-harness variance decomposition).
- Component replacement per segment (scripted vs RL vs LLM).

### 9.4 Contamination and robustness
- Modified-world variants (patched map or text layout) vs original, measuring progress drop. Distribute patches, not ROMs.
- Prompt leakage test: with and without hardcoded Route 29 walkthrough text.
- Repeat on a second game as a transfer check if budget allows (Red via PyBoy). Optional, but it pre-empts the "one game, one ROM" criticism.

### 9.5 Reproducibility package
- Pinned requirements, seeds, full prompt and response logs, replay traces with ROM SHA-1 verification, compute description (the audit mentions 12-16 workers on one RTX 4090), cost table, and an honest list of scripted components.

---

## 10. Reviewer-risk list

| # | Risk | Likely review comment | Mitigation |
|---|---|---|---|
| 1 | Hand-scripting dominates | "This is a scripted bot, not an agent." Current code has hardcoded gym step tables, Falkner-specific navigation, coordinates in reward shaping. | Measure and report the scripted share per segment. Prefer general mechanisms. Put scripting on the assistance ladder. |
| 2 | Cheats and privileged access | RAM writes (`equip_balls`), RTC lock, forced options, RAM-derived observation. | Remove cheats from headline results. Report each assist's effect. |
| 3 | Incremental over existing clears | Gemini 3 Pro reportedly cleared Crystal; Claude cleared Red; PokeAgent exists. | Lead with cost, harness-controlled evaluation and audit, not the clear itself. |
| 4 | Not reproducible | ROM cannot be shared; LLM APIs change or are deprecated; non-deterministic sampling. | ROM SHA-1 and replay traces; pin model versions; log all calls; report variance. |
| 5 | E&D accessibility rule | Reviewers must access artifacts without contacting authors; a ROM-dependent benchmark may be judged inaccessible. | Release everything except the ROM, supply videos and logs, and explain in the code-submission justification. |
| 6 | Small n | Single game, few seeds, one run. | Section 9.1 seed counts; add Red transfer if possible. |
| 7 | Prompt leakage and memorization | The system prompt contains a Route 29 walkthrough and "NEVER use Cut". Models may recall walkthroughs. | Remove walkthrough text; run contamination probes. |
| 8 | Unmeasured or fabricated-looking numbers | `compare_cost.py` uses a fake trace; `tensormewweb` telemetry is hand-authored; docs contradict code (HANDOFF says no badge earned, recordings show one). | Regenerate every number from real logs before drafting. Never cite the web dashboard. Clean docs. |
| 9 | Overclaiming in the title/abstract | "Beats Pokémon" when it needs checkpoints or interventions. | Define "clear" precisely (milestones, assist level, interventions = 0) and use that wording. |
| 10 | Wrong track or fit | "Games are not ML." Or E&D reviewers see a method paper, Main reviewers see a benchmark. | Pick one contribution type and write to its criteria (section 3 and 5). |
| 11 | Dual-submission or anonymity violation | Repo name `tensormew`, personal paths in code, public arXiv with affiliation before the anonymous ZIP. | Scrub identifiers. Use anonymous repo mirror. Keep preprint wording compliant. |
| 12 | LLM-writing flags | 2026 Position track desk-rejected 18.4% on detector scores. `[UNVERIFIED for Main/E&D 2027]` | Write prose by hand. Disclose LLM use in method and in the checklist. Verify all references. |
| 13 | Legal/ethics | ROM or save-state redistribution; Nintendo IP. | Section 4. Remove ROM from git history. Add game-IP statement. |
| 14 | Compute and cost | 7-hour LLM boss fights and multi-hundred-hour runs are expensive; budget overruns before Apr 20. | Budget the experiment matrix by Jan 31. Use cheaper models for ablations. |
| 15 | Scooped | Fast-moving area (PokeAgent living benchmark, lab-run streams). | Post arXiv early if a competing Crystal result appears. The NeurIPS preprint policy allows it. |

---

## 11. Immediate actions (next 2 weeks)

1. Resolve the "NeurIPS 2025 Red paper" reference in `research/README.md`.
2. Purge `rom/` and ROM-derived `.state` files from git history; add `rom/`, `*.gbc`, `*.state` to `.gitignore` and replace with a SHA-1 verification script.
3. Create OpenReview profiles for every author (institutional email avoids moderation delays).
4. Decide the assistance-ladder definitions and make the code log which assists are on.
5. Read the PokeAgent speedrun repo to align milestones and harness card format.
6. Set a calendar reminder to check neurips.cc for the 2027 call and Dates page on the 1st of each month.
7. Set the Gate 1/2/3 dates above in the project tracker.

## Sources consulted (all fetched or searched 2026-10-19)

- NeurIPS 2026: neurips.cc/Conferences/2026/CallForPapers, /MainTrackHandbook, /CallForEvaluationsDatasets, /EvaluationsDatasetsReviewerGuidelines, neurips.cc/Conferences/current/Dates, blog.neurips.cc newsletters (Apr, Jun 2026), blog.neurips.cc/2026/03/23 (E&D track).
- NeurIPS 2027 projection: opencurious.com/ai-conference-deadlines/neurips-2027 (unofficial).
- Workshops: neurips26-cl4fmagents.github.io, ttcl-agents.github.io.
- Other venues: iclr.cc/Conferences/2027/CallForPapers, aaai.org AAAI-27 pages, warwick.ac.uk AAMAS 2027 calls, cog2026.org, opencurious.com ICML 2027 page, AIIDE-26 call (substack), RLC 2026 deadline listings.
- Pokémon literature: arXiv 2502.19920, 2603.15563, 2506.03610, 2605.23950, neurips.cc/virtual/2026/poster/139132, OpenReview search, Gemini Crystal run reposts (youmind.com, Hacker News), LessWrong and Manifold on Claude Plays Pokémon.
- ROM/legal: PokemonRedExperiments README (via search), emulation legal explainers (secondary).
- Repo: `README.md`, `SPEC.md`, `.gitignore`, `research/analysis/codebase-audit.md`, `git ls-files`.
