# Research: beating all of Pokémon Crystal + a NeurIPS paper

Goal: (1) understand the NeurIPS 2025 Pokémon Red RL paper and the rest of the Pokémon-RL/LLM literature,
(2) extend tensormew from a hackathon checkpoint-chain agent to a full-game (8+8 badges, Elite Four, Red) clear,
(3) turn the result into a paper with a genuinely novel contribution.

Note: tensormew targets **Pokémon Crystal** (GBC, PyBoy). The NeurIPS 2025 paper is on **Red**, so every
finding needs a "does it transfer to Crystal?" check (Crystal adds time-of-day, phone/calendar events, Kanto post-game, Red at Mt. Silver).

Claims that a research agent could not verify are tagged `[UNVERIFIED]`. Treat them as leads, not facts.

## Layout

| Path | Contents |
| --- | --- |
| `papers/` | Deep reading notes on individual papers (start with the NeurIPS 2025 Red paper) |
| `prior-work/` | Surveys: community RL codebases, LLM agents, battle AI + Crystal progression, long-horizon RL techniques |
| `analysis/` | Audit of our own codebase: what exists, what blocks a full clear |
| `ideas/` | Candidate novel contributions, ranked; experiment designs |
| `paper-draft/` | Venue/timeline plan, outline, eventually the draft |

## Index (filled in as research lands)

See `INDEX.md`.
