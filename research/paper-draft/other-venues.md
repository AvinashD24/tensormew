# Non-NeurIPS publication venues for the Pokémon Crystal full-clear agent

Research date: 2026-10-19. Scope: every venue worth considering besides NeurIPS (main, D&B and workshops, covered elsewhere).
Method: web search plus fetches of official pages where reachable. Anything not confirmed on an official or primary page is tagged `[UNVERIFIED]`.
"Est." means extrapolated from the previous cycle, not announced. Do not plan around an estimated date without re-checking the official call.
All deadlines are AoE (UTC-12) unless noted.

## 0. Bottom line

- Three big venues have already closed for this cycle: ICLR 2027 (Sep 25, 2026), AAAI-27 (Jul 28, 2026) and AAMAS 2027 (Oct 8, 2026). Neither the AAAI-27 nor the AAMAS 2027 paper deadline is recoverable. The only side door is the AAMAS fast track, which needs an AAAI-27 submission.
- The best realistic primary is **RLC 2027** (Reinforcement Learning Conference). Its call is not out yet. The previous cycle's full-paper deadline was Mar 5, so expect about early March 2027 `[UNVERIFIED]`. It is RL-first and published in the Reinforcement Learning Journal. It has the closest Pokémon precedent (Metamon). Acceptance was 39% in 2025. 2027 will be a two-site event (Brown University, plus Odense, Denmark jointly with EWRL), which is official.
- **ICML 2027** is the prestige stretch. The deadline is est. Jan 22, 2027 and the official call is not published. It only makes sense if the full clear and ablations are frozen by early January.
- **TMLR** (rolling, no deadline) is the safest fallback and fits a long, system-heavy paper well.
- **Workshop-first** options with real dates: ICLR 2027 workshops (Apr 29-30, 2027; the workshop-paper deadline is suggested as Feb 1, 2027, with notification by Feb 26) and ALA at AAMAS 2027 (est. Feb 2027 `[UNVERIFIED]`).
- Be careful with precedent claims. I could **not** verify that the "Pokémon Red via Reinforcement Learning" paper (Pleines et al., arXiv 2502.19920) appeared at NeurIPS 2025. The arXiv comment field says only "8 pages, 3 figures, 3 tables, under review". Treat "NeurIPS 2025" as `[UNVERIFIED]` until someone checks the NeurIPS 2025 proceedings or OpenReview.
- The closest direct competitor and citation is **The PokeAgent Challenge** (arXiv 2603.15563). It was a NeurIPS 2025 Competition Track entry, is listed as a NeurIPS 2026 poster (Sydney, Dec 9, 2026), and has a Speedrunning track on the Pokémon RPG with a multi-agent harness. Reviewers at any venue will ask how we compare, so our evaluation protocol should map onto it.

## 1. Precedents: where Pokémon and game-agent work has been published

| Paper | Venue (status) | Source / notes |
|---|---|---|
| Metamon: Human-Level Competitive Pokémon via Scalable Offline RL with Transformers (Grigsby et al.) | RLC 2025, published in RLJ vol. 6; Edmonton, Aug 5-9, 2025 (verified) | rlj.cs.umass.edu/2025/papers/Paper340.html |
| PokéChamp: an Expert-level Minimax Language Agent (Karten, Nguyen, Jin) | ICML 2025, PMLR v267 (verified). An earlier version is also listed at NeurIPS 2024 under a different title; track not checked `[UNVERIFIED]` | proceedings.mlr.press/v267/karten25a.html |
| BALROG: Benchmarking Agentic LLM and VLM Reasoning on Games | ICLR 2025 (verified) | iclr.cc/virtual/2025/poster/28856 |
| Orak: A Foundational Benchmark for LLM Agents on Diverse Video Games | ICLR 2026 (verified) | proceedings.iclr.cc/paper_files/paper/2026 |
| The PokeAgent Challenge: Competitive and Long-Context Learning at Scale | NeurIPS 2025 Competition Track; NeurIPS 2026 poster, Sydney, Dec 9, 2026 (verified). Main vs D&B track not stated `[UNVERIFIED]` | arXiv 2603.15563; neurips.cc/virtual/2026/poster/139132 |
| VideoGameBench: Can VLMs complete popular video games? (Zhang et al.) | arXiv only as far as I could see (v3 May 2026); no venue listed. Peer-reviewed venue `[UNVERIFIED]` | arXiv 2505.18134 |
| Pokemon Red via Reinforcement Learning (Pleines et al.) | arXiv 2502.19920, "under review". NeurIPS 2025 acceptance `[UNVERIFIED]` (not found in any source) | arXiv listing |
| PokeLLMon (Hu, Huang, Liu) | arXiv 2402.01118; ICML-style template, but no accepted venue found `[UNVERIFIED]` | arXiv |
| PokéAI (Red, multi-agent), PokeRL (Red, early game), VGC-Bench | arXiv only as far as I could see | arXiv 2506.23689, 2604.10812, 2506.10326 |
| Gemini / Claude "plays Pokémon" runs | Blogs, tweets and lab reports, not peer reviewed | The runs differ in harness help, so they are not comparable (LessWrong analysis) |

I found no Pokémon RL paper confirmed at IEEE CoG, AIIDE, FDG, AAAI or IJCAI. That is a gap, not necessarily a negative: it means no one has tried, or the search tools missed it. Searching DBLP for "Pokémon" at those venues is a cheap follow-up.

Takeaway: the confirmed top-tier precedents are in the ML/RL/LLM-agent venues (ICML, ICLR, RLC, NeurIPS), not the games venues. Game-AI venues are lower-prestige but lower-competition and a natural home for the full-game-clear story.

## 2. Venue by venue

Scoring: fit 1-5 = how well a "hybrid RL + LLM agent, deterministic macros, checkpoint curricula, full Pokémon Crystal clear" paper matches the venue's audience, scope and reviewer expectations. It is not prestige alone. Acceptance rates are for the most recent cycle I could verify.

### 2.1 Main ML conferences

**RLC 2027 (Reinforcement Learning Conference)**
- Next deadline: not announced. RLC 2026 had its full-paper deadline Mar 5 (AoE) with abstract deadline Mar 1 or Mar 5 (the page is ambiguous), notification May 5/8 (page is contradictory), and camera-ready Jul 18. So est. early March 2027 `[UNVERIFIED]`.
- Location/dates: RLC 2027 is announced on the RLC 2026 page as a two-site event, Brown University (early August) and Odense, Denmark (late August, joint with EWRL). Authors pick a site. Exact dates `[UNVERIFIED]`. (RLC 2026: Montréal, Aug 15-18.)
- Page limit (RLC 2026 call): recommended 8 pages main text, hard max 12. References and supplement are excluded. A cover page with a contributions box is required, and reviewers judge mostly against it. Overclaiming can cause rejection.
- Topics include deep RL, hierarchical RL, exploration, benchmarks and applied RL, evaluation methodology, imitation learning, and multi-agent RL. There are no separate tracks. Double-blind on OpenReview, at least two reviewers plus an AC.
- Prestige/acceptance: young, RL-specialist, respected in the RL community. 2025: 115 of 295 accepted (39.0%), from a single source (RIKEN AIP). Published in RLJ (peer-reviewed journal).
- Precedent: Metamon (RLC 2025).
- Fit: **5**. Checkpoint curricula, reward and macro design, exploration, and a full-game RL clear are squarely in scope. The LLM component is fine as long as the RL story (curriculum, credit assignment, sample efficiency) carries the paper. Weakness: if reviewers see mostly engineering, the contributions box has to be sharp.

**ICML 2027**
- Next deadline: not officially published. One tracker estimates abstract Jan 16 and paper Jan 22, 2027 AoE, explicitly projected from the prior cycle `[UNVERIFIED]`.
- Location: official Future Meetings page lists only "South America" for 2027 (and Eastern US for 2028). City and dates are not announced. A July 11-17 window is a tracker projection `[UNVERIFIED]`.
- Page limit: ICML 2026 used 8 pages for the main text at submission `[UNVERIFIED for 2027]`.
- Prestige/acceptance: top tier. ICML 2026 (Seoul): 6,352 of 23,918 accepted (26.6%), per aggregated coverage. 2025: 26.9% (12,107 submissions). Spotlights are about 2.2% and orals 0.7%.
- Precedent: PokéChamp (ICML 2025).
- Fit: **4**. ICML needs a methodological or empirical insight that generalizes beyond Pokémon, not just a game-completion result. The risk is "this is an engineering system". Strong if framed as lessons on hybrid planning, macro abstraction, or curriculum design with controlled ablations.
- Timing is the constraint. Roughly 12 weeks from now, with the agent still in development.

**ICLR 2027 (main track): closed**
- Abstract Sep 18, paper Sep 25, reviews Nov 5, decisions Dec 16, 2026 (all verified, official). Conference Apr 26-28, 2027, workshops Apr 29-30. Location is listed as "to be announced" on the official dates page. Third-party sites disagree (California vs Brazil), so treat location as `[UNVERIFIED]`.
- Page limit: 9 pages main text at submission, 10 for rebuttal/camera-ready, references and appendix uncapped (official author guidelines per search snippet). New policies on AI use, co-authorship and reciprocal reviewing apply.
- Acceptance: ICLR 2026 was 27.4% (5,355 of 13,763 decided papers; 19,525 valid submissions). 2025 was about 32%.
- Precedents: BALROG (ICLR 2025), Orak (ICLR 2026).
- Fit: **4** in content, but unavailable now. The next main-track shot is ICLR 2028, est. late September 2027 `[UNVERIFIED]`. ICLR is the venue with the best benchmark/agent precedents if we pivot to a benchmark framing.

**AAAI-27: closed; AAAI-28 is the next shot**
- AAAI-27 main track: abstracts Jul 21, full papers Jul 28, 2026 (verified, official). Montréal, Feb 16-23, 2027. Phase 1 rejections went out Sep 24; final notification Nov 30; camera-ready Dec 14. 7 pages main content plus references (9 total).
- Acceptance: AAAI-25 was 4,167 of 23,680 (17.6%). AAAI-26 had about 29,000 submissions with about 23,000 reviewed; the final rate is not confirmed in my sources.
- AAAI-28 deadline: est. late July 2027 `[UNVERIFIED]`.
- Journal track: AAAI-27 journal track electronic submissions due Oct 30, 2026. This is for work already published in selective journals, so it is not useful now.
- Fit: **3**. Broad AI audience, an agent-systems paper is acceptable, but the very large pool (and now AI-assisted first-phase review) makes it a lottery for system papers.

**IJCAI-27 (Kyoto, Japan, Aug 7-13, 2027; satellite in Hengqin, China, Aug 15-17)**
- Location/dates: confirmed on ijcai.org.
- Deadline: not published. Recent main-track deadlines were Jan 23, 2025, Jan 17, 2024 and Jan 18, 2023, so est. mid-January 2027 `[UNVERIFIED]`. One tracker says Jan 15, 2027 (projected).
- Page limit: IJCAI-25 allowed 7 pages plus references `[UNVERIFIED for 2027]`.
- Acceptance: IJCAI-ECAI 2026 reported 858 accepted of 6,342 submissions (13.5%), but another source says 713 in the main track alone. Treat as roughly 13-14% `[UNVERIFIED]`.
- Fit: **3**. Selective, agent/planning friendly, no Pokémon or game-agent precedent that I found. A lower expected value than RLC or ICML for this topic.

**AAMAS 2027 (Hanoi, Vietnam, May 3-7, 2027): closed for the main track**
- Abstract Oct 1, paper Oct 8, rebuttal Nov 20-24, notification Dec 21, 2026, camera-ready Jan 25, 2027 (verified, official).
- Findings of AAMAS: papers not selected for the Proceedings are automatically considered for Findings (poster only) unless authors opt out.
- AAAI fast track (the only open door): papers rejected from AAAI-27 main track whose every human reviewer score is at least 5 can resubmit. Deadline Dec 11, 2026, notification Feb 1, 2027, camera-ready Feb 11, 2027. We did not submit to AAAI-27, so this is not available.
- Topic areas include Learning and Adaptation, Generative and Agentic AI (including benchmarks for agentic systems), and Robotics and Control.
- Acceptance: AAMAS 2026 full papers were 338 of about 1,455 submissions (25.5%), with another 193 accepted as extended abstracts. The GAAI area was 18% for full papers.
- Page limit: not stated on the call page (see the separate submission instructions) `[UNVERIFIED]`.
- Fit: **3**. The MAS community is receptive to agent architectures, but the single-agent Pokémon story is not a natural match. AAMAS 2028 (est. autumn 2027) is the next real chance.

### 2.2 Language/agents venues

**COLM 2027 (Conference on Language Modeling)**
- Next deadline: not announced. A tracker lists est. Mar 31, 2027 (paper) with the conference in October 2027, venue TBD `[UNVERIFIED]`. COLM 2026 is Oct 6-9, 2026, at the Hilton Union Square, San Francisco (verified on colm.cc).
- Page limit: I could not retrieve the call. COLM used about 9 pages main text in prior years `[UNVERIFIED]`.
- Acceptance: COLM 2024 about 29% (299 of 1,036, from Papercopilot, `[UNVERIFIED]`); I found no reliable COLM 2025 rate.
- Precedent: no confirmed Pokémon paper. LLM-agent and game-agent work is on topic for COLM generally.
- Fit: **4 if the contribution is primarily about the LLM component** (e.g. LLM as planner or critic, what must stay deterministic, how to stop the LLM from hallucinating state). If the paper is mostly RL, reviewers will push it to RLC/ICML.

**ACL 2027 / EMNLP**
- ACL 2027: Kyoto, Aug 17-22, 2027 (official). ARR submission deadline Jan 4, 2027 (stated on the official site). Main-conference call, tracks and page limits not yet available. Whether there is a "language agents" track is `[UNVERIFIED]`.
- EMNLP 2026 is in Budapest, Oct 24-29, 2026 (aggregators; verify). Its ARR deadline was May 25, 2026, so it has passed. EMNLP 2025 acceptance was about 22.2% (1,811 of 8,174, one tracker). EMNLP 2027 deadlines `[UNVERIFIED]`.
- Fit: **2**. NLP venues want a language contribution. A Pokémon agent where the LLM is one component is likely off-scope unless we isolate a language-agent finding. Use only for a spin-off paper.

### 2.3 Games and game-AI venues

**IEEE Conference on Games (CoG) 2027**
- Next deadline: not announced. CoG 2026 (Madrid, Sep 1-4, 2026) had full papers due Mar 17, 2026, notification May 1 and camera-ready May 28. CoG 2025 (Lisbon, Aug 26-29) had full papers due Mar 15, 2025, with an 8-page limit including references and appendices. So est. mid-March 2027 `[UNVERIFIED]`. 2027 location and dates not found.
- Auxiliary papers (CoG 2026): short papers 4 pages, competition and vision papers 8 pages, due May 14, 2026. Useful for a short follow-up.
- Acceptance: not found. `[UNVERIFIED]`
- Fit: **4**. This is the community that cares about "an AI that actually finishes the game". Moderate prestige, lower variance than ICML. Pokémon-specific precedent at CoG not found.

**AIIDE 2027**
- AIIDE 2026: Belo Horizonte, Brazil, Nov 9-13, 2026 (first edition outside US/Canada). Full-paper deadline was Jun 26, 2026 per one source (another says Jun 19), notification Aug 7, camera-ready Aug 28. Full papers 9 pages excluding references.
- AIIDE 2027: no call found. Est. around June 2027 `[UNVERIFIED]`. Acceptance rate not found.
- Fit: **3**. Good for the game-AI audience and AAAI Press proceedings, but smaller and with a games-design flavor.

**FDG 2027 (Foundations of Digital Games)**
- Melbourne, Australia, Jun 21-24, 2027. Deadlines per a co-chair's LinkedIn post: workshops/panels Nov 16, 2026, full papers and abstract-only Jan 11, 2027, late-breaking/demos/doctoral consortium Mar 8, 2027 (secondary source; fdg2027.org was not reachable, so `[UNVERIFIED]`).
- Length (FDG 2026 rule, may carry over): full papers up to 8,000 words, short and late-breaking 6,000 words, excluding references.
- Fit: **2**. FDG is design, HCI and humanities leaning. A technical RL agent paper is unusual there. Late-breaking work (Mar 8) is a cheap option for exposure only.

**CHI PLAY 2027, SIGGRAPH**
- CHI PLAY 2027: no call found. The 2026 edition (York, Nov 2, 2026) had a Feb 18, 2026 deadline.
- Fit: **1**. Human-computer interaction or graphics venues. Only relevant if we add a human study (e.g. players watching or co-playing with the agent). SIGGRAPH is not relevant.

### 2.4 Journals

**TMLR (rolling)**
- No deadline. Target decision about 9 weeks after submission (no guarantee). Action editor checks scope in one week, reviews due within 2 weeks of assignment, then 2 weeks of discussion. Double-blind, mandatory template. Submissions may be any length, but over 12 pages of main content gets a slower, unenforced timeline.
- Acceptance criteria are correctness and clear claims, not novelty or impact. That suits a large engineering and evaluation paper. Acceptance rate: not verified `[UNVERIFIED]`.
- TMLR papers have been eligible for the CoLLAs journal track in past years (e.g. 2024). Whether that continues for 2027 is `[UNVERIFIED]`.
- Fit: **4**. The best fallback after a conference rejection, and also a good first choice if the contribution is a thorough system paper rather than a headline method.

**IEEE Transactions on Games (ToG)**
- Rolling. Full papers 10 to 14 pages (extra fees above 10), letters 4, short 6-8, survey 15-20. Over 12 pages must be justified. Double-anonymous from Jan 1, 2025. Special issues cannot be tied to a conference.
- Fit: **3**. Right audience (game AI), slower, lower visibility than ML venues. Reasonable as an extended journal version after a CoG paper.

**JAIR**
- Rolling. About three months to a first decision; JAIR's own transparency data suggests a 23.1% average acceptance estimate for a Jan 1, 2025 submission (not a classic acceptance rate). Impact factor sources disagree (about 4 to 4.7) `[UNVERIFIED]`.
- Length and format rules were not retrieved.
- Fit: **3**. Suitable for a long, careful system paper, but a slow route with less visibility than TMLR for this community. A reasonable choice if a thorough "lessons learned" paper is wanted.

### 2.5 Lifelong, adaptive and workshop venues

**CoLLAs 2027 (Conference on Lifelong Learning Agents)**
- 2026 edition: Bucharest, Romania, Sep 14-18, 2026, abstract Apr 10, paper Apr 15, 2026. PMLR proceedings. No 2027 call found, so est. April 2027 `[UNVERIFIED]`. Page limits not retrieved.
- Fit: **3**. A checkpoint curriculum over a long sequence of badges and areas is a decent lifelong-learning framing, but the story must be about continual learning or forgetting, not game completion.

**ALA 2027 (Adaptive and Learning Agents workshop, at AAMAS 2027, Hanoi)**
- Not found. The 2025 edition had a Feb 4, 2025 deadline and took original research, work in progress, outlook papers and journal-paper presentations. So est. February 2027 `[UNVERIFIED]`.
- Fit: **4** as a workshop. AAMAS also has a Workshop Outreach Pipeline for eligible non-archival workshop papers (details not retrieved).

**ICLR 2027 workshops (Apr 29-30, 2027)**
- Official: workshop acceptance notification Nov 29, 2026; suggested workshop paper deadline Feb 1, 2027; mandatory accepted-paper notification Feb 26, 2027. The proposal window closed Oct 9, 2026 (from an organizer post, `[UNVERIFIED]`). The accepted list is not public until after Nov 29.
- Fit: **4**. Typical themes likely to appear (agents, long-horizon, lifelong, games) are `[UNVERIFIED]` until the list is out.

**ICML 2027 workshops**
- Not announced. Expect paper deadlines in May-June 2027 `[UNVERIFIED]`. ICML 2026 had LLM4Plan (Planning in the Era of LLMs) as an example in this space. Fit: **4** as a workshop.

**NeurIPS 2026 workshops (information only, since NeurIPS is covered elsewhere)**
- 102 accepted workshops across Sydney (Dec 11-12, 2026), Paris and Atlanta (Dec 12-13). Relevant titles, from the official announcement blog: "Workshop on Evaluation of Interactive Agents" (Atlanta), "Continual Learning in the Era of Foundation Models and Embodied Agents" (Sydney), "PTA: From Pretrained Representations to Acting Agents" (Sydney), "Physical Understanding for Decision-Making: Bridging Foundation Models and Reliable Agents" (Sydney), "Who Verifies the Agents?" (Sydney), "Foundations of LLM Post-Training in Changing Environments" (Paris), "Towards Test-Time Continual Learning Agents" (Atlanta).
- Deadlines I saw for other NeurIPS 2026 workshops were Aug 29-Sep 10, 2026. Assume most are closed. `[UNVERIFIED]` per workshop; check each OpenReview page.
- Other cycle's relevance: a workshop talk at Sydney does not help a December 2026 timeline, but it shows which communities exist for 2027 submissions.

**Open-endedness / foundation models for decision making / game-AI workshops**
- I found no confirmed 2026-27 workshop titled "Open-Endedness" or "Foundation Models for Decision Making" at ICLR/ICML/NeurIPS. The NeurIPS 2026 list has only borderline matches ("Self-Evolving Diversity-Driven Search for Robust AI Systems"). Watch the ICLR 2027 list (Nov 29) and the ICML 2027 workshop announcements.

### 2.6 arXiv-first

- Always possible, with no review. For this community (Pokémon, speedrunning, Twitch-style demos) visibility on arXiv, plus a repo and a video of the full clear, matters as much as the venue. Every precedent above was on arXiv before or at acceptance (Metamon, PokéChamp, BALROG, Orak, VideoGameBench, PokeAgent Challenge).
- Most ML venues in this list allow arXiv preprints while under double-blind review, but each 2027 policy must be re-read `[UNVERIFIED]`. ICLR 2027 enforces desk rejection if author identity is revealed in the paper or supplement. The safe way is to post the preprint with the venue's allowed wording and not to link the repo in the submission itself.
- Fit: **5** as a complement, **1** as a standalone credential.

## 3. Ranking

Ranking combines fit, prestige, whether the deadline is reachable given likely agent readiness, and precedent. "Deadline" is the next one that matters, not an already-closed one.

| Rank | Venue | Fit | Prestige | Next deadline | Notes |
|---|---|---|---|---|---|
| 1 | RLC 2027 | 5 | High in RL | est. early Mar 2027 `[UNVERIFIED]` | Metamon precedent, 39% acceptance, RLJ, Brown/Odense |
| 2 | ICML 2027 | 4 | Top | est. Jan 22, 2027 `[UNVERIFIED]` | Stretch only; about 26.6% acceptance; PokéChamp precedent |
| 3 | TMLR | 4 | Good, rolling | any time | Safest fallback; about 9 weeks |
| 4 | COLM 2027 | 4 (LLM-heavy framing) | High in LM | est. Mar 31, 2027 `[UNVERIFIED]` | Only if LLM contribution dominates |
| 5 | IEEE CoG 2027 | 4 | Medium | est. mid-Mar 2027 `[UNVERIFIED]` | Game-AI home; 8 pages |
| 6 | ICLR 2027 workshops | 4 | Workshop | Feb 1, 2027 (suggested) | Workshop-first; notifications by Feb 26 |
| 7 | ALA @ AAMAS 2027 | 4 | Workshop | est. Feb 2027 `[UNVERIFIED]` | Workshop-first; Hanoi May 3-7 |
| 8 | ICLR 2028 | 4 | Top | est. late Sep 2027 `[UNVERIFIED]` | Next main-track chance for the BALROG/Orak audience |
| 9 | IEEE ToG | 3 | Medium | rolling | Extended version after CoG |
| 10 | AAAI-28 | 3 | High, lottery | est. Jul 2027 `[UNVERIFIED]` | About 17.6% (AAAI-25); 7 pages |
| 11 | IJCAI-27 | 3 | High, selective | est. mid-Jan 2027 `[UNVERIFIED]` | About 13.5% `[UNVERIFIED]` |
| 12 | CoLLAs 2027 | 3 | Niche | est. Apr 2027 `[UNVERIFIED]` | Needs lifelong-learning framing |
| 13 | AIIDE 2027 | 3 | Medium | est. Jun 2027 `[UNVERIFIED]` | 9 pages |
| 14 | JAIR | 3 | Good, slow | rolling | Long careful paper |
| 15 | AAMAS 2028 / Findings | 3 | High | est. autumn 2027 `[UNVERIFIED]` | AAMAS 2027 is closed |
| 16 | ACL 2027 / EMNLP | 2 | High | ARR Jan 4, 2027 (ACL, official) | Off-scope unless language-centric |
| 17 | FDG 2027 | 2 | Niche | Jan 11, 2027 (secondary) `[UNVERIFIED]` | Late-breaking Mar 8 for exposure |
| 18 | CHI PLAY / SIGGRAPH | 1 | n/a | n/a | Only with a human study |

## 4. Recommended strategy

**Decision rule.** What the headline claim is decides the venue.
- Full-game clear is the headline, and the method is RL with curricula, macros and an LLM helper: **RLC 2027**.
- A generalizable finding about hybrid LLM-plus-RL control with strong ablations across games or settings: **ICML 2027**, otherwise NeurIPS 2027 (other agent).
- LLM is the main novelty (planner, critic, memory, state grounding): **COLM 2027**.
- Mostly system/evaluation depth, no clean "method": **TMLR**, or **CoG 2027** then ToG.
- Do not submit the same work to two archival venues at once. RLC, ICML, COLM and CoG all prohibit dual submission, and RLC publishes in a journal. Workshops marked non-archival are the exception, and each workshop's policy must be checked.

**Primary: RLC 2027.**
- Submit about early March 2027 (est.).
- It is RL-native and has the only Pokémon precedent among the RL/agent venues (Metamon). Acceptance is about 39%, higher than ICML/ICLR/AAAI/IJCAI. Notification around May 2027 leaves time to fall back before COLM (if open), CoG notifications, and NeurIPS 2027 deadline discussions with the other agent.
- A two-site conference (Brown, Odense) gives flexibility for travel.

**Stretch (only if ready): ICML 2027.**
- Go/no-go on Jan 4, 2027. Go only if the full Crystal clear is reproducible across several seeds, ablations of macros, curriculum and LLM are complete, and PokeAgent Challenge-style baselines are in.
- If no-go, redirect the same draft to RLC and do not split effort.

**Fallbacks, in order.**
1. TMLR (any time after a rejection, 9-week loop; keep the draft in TMLR format from the start for easy transfer).
2. CoG 2027 (mid-March est.). If RLC is rejected in May, the CoG deadline will have passed, so submit to CoG only in parallel if the paper is split (a short paper on one component) or choose TMLR instead.
3. IEEE ToG as an extended version of any CoG/workshop paper.
4. AIIDE 2027 (est. June), then ICLR 2028 / AAAI-28 (autumn 2027) with improved results.

**Workshop-first options (optional, low cost, non-archival only).**
- ICLR 2027 workshops: submit a short version Feb 1, 2027 (suggested deadline), after the list appears Nov 29, 2026. Presentation Apr 29-30, 2027. This is non-archival, so it does not block RLC or ICML, but check each workshop's policy.
- ALA at AAMAS 2027 (est. Feb 2027): good for the adaptive-agents audience and feedback from MARL researchers.
- Never post the workshop version as the main paper if the workshop is archival.

**arXiv-first.** Post a preprint (paper plus repo plus full-clear video) at the moment of the primary submission, after re-reading the venue's anonymity policy. For ICLR-style desk-reject rules, keep the repo link out of the submission itself. A public result of "full Crystal clear" will also be something Pokémon-agent competitors (e.g. PokeAgent Challenge participants) can benchmark against, which improves citations.

## 5. Dated timeline

| Date | Action | Confidence |
|---|---|---|
| 2026-10-19 to 2026-11-15 | Freeze the evaluation protocol (what counts as "beat the full game"; resets, saves, step budget, macro and LLM call accounting), aligned with the PokeAgent Challenge Speedrunning track. | Plan |
| 2026-11-29 | ICLR 2027 workshop list released; pick 1-2 workshops with agents, long-horizon or lifelong themes. | Official date |
| 2026-12-09 | PokeAgent Challenge poster at NeurIPS 2026 in Sydney. Read the final version, align baselines. | Official |
| 2026-12-16 | ICLR 2027 decisions. Look at which game-agent papers were accepted, for positioning. | Official |
| 2027-01-04 | ACL 2027 ARR deadline (official). Skip unless we pivot to a language contribution. Also the internal go/no-go for ICML 2027. | Official / internal |
| 2027-01-11 | FDG 2027 full papers (secondary source). Skip unless we do a design-oriented paper. | `[UNVERIFIED]` |
| ~2027-01-15 | IJCAI-27 deadline (est.). Optional, low priority. | `[UNVERIFIED]` |
| ~2027-01-22 | ICML 2027 paper deadline (est.); abstract about Jan 16. Stretch. | `[UNVERIFIED]` |
| 2027-02-01 | ICLR 2027 workshop paper deadline (suggested, per official workshop guidelines); notification by 2027-02-26. | Official guideline |
| ~2027-02 | ALA 2027 deadline (est., by 2025 pattern). | `[UNVERIFIED]` |
| ~2027-03-05 | **RLC 2027 full-paper deadline (est. from RLC 2026).** Primary. | `[UNVERIFIED]` |
| ~2027-03-15 | IEEE CoG 2027 full papers (est.). | `[UNVERIFIED]` |
| ~2027-03-31 | COLM 2027 (est.). Only in place of RLC, never both. | `[UNVERIFIED]` |
| ~2027-04-15 | CoLLAs 2027 (est.). | `[UNVERIFIED]` |
| 2027-04-26 to 04-30 | ICLR 2027 conference and workshops. | Official dates; location TBA |
| 2027-05-03 to 05-07 | AAMAS 2027, Hanoi (ALA workshop). | Official |
| ~2027-05-08 | RLC 2027 notification (est.). If rejected, revise per reviews, then go to TMLR or the autumn deadlines. | `[UNVERIFIED]` |
| 2027-06-21 to 06-24 | FDG 2027, Melbourne. | Secondary |
| ~2027-07 | ICML 2027, South America (city not announced). AAAI-28 deadline (est. late July). | `[UNVERIFIED]` |
| 2027-08-07 to 08-13 | IJCAI-27, Kyoto. | Official |
| 2027-08 | RLC 2027 at Brown (early Aug) and Odense (late Aug). | Official (exact dates `[UNVERIFIED]`) |
| ~2027-09 | ICLR 2028 deadline (est.). | `[UNVERIFIED]` |

## 6. Caveats and items to verify next

1. **RLC 2027 call**: check rl-conference.cc for the real deadline and page limit when posted. Everything above is carried over from RLC 2026.
2. **ICML 2027 call and location**: only "South America" is official.
3. **COLM, CoG, AIIDE, CoLLAs, ALA 2027 calls**: none were found; dates are extrapolations.
4. **Pleines et al. "NeurIPS 2025" claim**: unconfirmed; the arXiv entry says "under review". Do not cite it as a NeurIPS paper without checking the proceedings.
5. **Acceptance rates**: RLC (single source), COLM, CoG, AIIDE, TMLR are unverified or missing. IJCAI-2026 numbers conflict (858/6,342 vs 713 main track).
6. **Page limits for 2027** (ICML, AAAI-28, IJCAI, AAMAS, COLM, CoG): taken from prior cycles; re-check each call.
7. **arXiv and anonymity policies** differ per venue. Re-read before posting the preprint.
8. **PokeAgent Challenge** (NeurIPS 2025 competition, NeurIPS 2026 poster): confirm which Pokémon RPG its Speedrunning track uses; the arXiv abstract does not name the game.

## Sources (primary)

- ICLR 2027: iclr.cc/Conferences/2027/CallForPapers, /Dates, /AuthorGuidelines; ICLR 2026 retrospective (blog.iclr.cc)
- AAAI-27: aaai.org/conference/aaai/aaai-27/main-technical-track-call/
- AAMAS 2027: warwick.ac.uk/fac/sci/dcs/aamas2027/calls/ (main track, AAAI fast track), AAMAS 2026 welcome letter (ifaamas.org)
- RLC: rl-conference.cc/2026/index.html and /2026/callforpapers.html (includes the RLC 2027 two-site announcement); RIKEN AIP RLC 2025 acceptance post
- ACL 2027: 2027.aclweb.org
- IJCAI: ijcai.org (IJCAI-27 location)
- ICML: icml.cc/Conferences/FutureMeetings
- CoLLAs 2026 CFP (ContinualAI Google Group); COLM: colm.cc
- TMLR FAQ and guides: jmlr.org/tmlr; IEEE ToG: transactions.games; JAIR: jair.org
- NeurIPS 2026 workshops: blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/; PokeAgent Challenge: arxiv.org/abs/2603.15563, neurips.cc/virtual/2026/poster/139132
- Precedents: rlj.cs.umass.edu/2025/papers/Paper340.html (Metamon), proceedings.mlr.press/v267/karten25a.html (PokéChamp), iclr.cc/virtual/2025/poster/28856 (BALROG), proceedings.iclr.cc 2026 (Orak), arxiv.org/abs/2505.18134 (VideoGameBench), arxiv.org/abs/2502.19920 (Pokémon Red RL)
