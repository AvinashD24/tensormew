# Candidate paper directions (synthesis)

Built from all reports in `../prior-work/`, `../papers/`, `../analysis/`, `../paper-draft/`.
Detail, hypotheses, experiments and risks for each direction live in
`../prior-work/rl-techniques-long-horizon.md` section 11. This file is the decision layer.

## Facts that constrain the paper

1. "We beat Crystal" is **not novel**: Gemini 3 Pro and GPT-5.1 have reported Crystal clears (livestream/blog evidence, no peer-reviewed paper; some numbers `[UNVERIFIED]`).
2. No peer-reviewed RL agent clears Crystal or Red end-to-end. Pleines et al. (arXiv 2502.19920) reaches Cerulean only and is **not** a NeurIPS paper. The only full-game RL result is the non-peer-reviewed `pokemonred_puffer`, which scripts HMs.
3. The closest competitor is the PokeAgent Challenge (arXiv 2603.15563, Emerald + Showdown). Its winner was LLM-written scripts distilled to RL.
4. tensormew today: one verified badge (Zephyr), mostly scripted; no logged runs; ROM and `.state` files are tracked in git.
5. NeurIPS 2026 is closed. Realistic targets: workshops (early 2027), RLC 2027 (est. early Mar), ICML 2027 (est. Jan), NeurIPS 2027 (est. mid-May). Most 2027 dates are estimates.

## Recommended plan

**Primary framing:** a measured study of *when an LLM is needed* in long-horizon game play, on Crystal, with open tooling.
Core claim: controlled calls/tokens/$ per milestone, trigger-gated vs LLM-every-step, on identical checkpoints, with a milestone-lattice metric.
This combines directions 1, 4 and 5 below and is the only gap all the surveys agree is open.

| # | Direction | Novelty | Feasibility | Notes |
| --- | --- | --- | --- | --- |
| 1 | Stall-triggered LLM unblocking for Go-Explore with verification on resets | High | Medium | Needs contamination control (LLM knows Crystal) |
| 4 | Milestone-lattice evaluation with reset/token accounting | Medium | High | Differentiate from PokeAgent; E&D-track fit |
| 5 | Affordance-gated LLM planner, tokens-per-milestone frontier | Medium | Medium | The repo's own stated thesis |
| 2 | Learning-progress start-state curriculum over the archive | High | Low-Med | Compute heavy; negative result must be sharp |
| 3 | RAM-caption preference rewards plus reward-hacking audit | Medium | Medium | Motif/Eureka exist; the audit is the new part |
| 6 | Inverse-dynamics labeling of speedrun video | Medium | Low | Domain-gap and data-use risk |
| 7 | Memory-architecture study | Low-Med | Medium | Workshop-scale |
| 8 | Latent-space Go-Explore | High | Low | Highest risk |

## Engineering that must happen first (from the audit)

1. Real battle engine reading enemy species/type/moves from RAM (current code assumes Falkner's types).
2. Field moves: Cut, Surf, Strength, Flash, Whirlpool, Waterfall, Fly, plus water collision.
3. Progression graph over 16 badges, E4, Red (`research/prior-work/battle-ai-and-crystal-progression.md` has a 58-milestone checklist).
4. Fix `src/story_flags.py`: HM flag indices are 14-19 but the disassembly gives 16-21; Chuck/Jasmine flags are swapped (per the battle/Crystal report, verify before changing).
5. Unlock the RTC and test time-of-day events; remove RAM cheats (`equip_balls`).
6. Register milestone states, build a seeded eval harness with logged runs. Without logs there are no results to report.
7. Resolve the ROM/state-in-git issue before any public release.

## Honest risks

- Reviewers may see an LLM-gated scripted pipeline as engineering. The contribution must be the measurement, not the architecture.
- Pre-seeded human checkpoints mean we cannot claim end-to-end autonomy; say so up front.
- Frontier LLMs may already know Crystal; control for it (scrambled names or a ROM hack).
- Several surveyed numbers are community-reported or `[UNVERIFIED]`; re-check every figure against a primary source before citing.
