# PWhiddy/PokemonRedExperiments (read directly via `gh api`, 2026-10-20)

Repo: https://github.com/PWhiddy/PokemonRedExperiments. MIT license, ~7.9k stars, ~786 forks, created 2019-12-25, last push 2026-09-11 ("map render, finishes", "regenerate state from replay, and resume from saved state"). So it is still maintained.

## What it is

The origin project of most community Pokémon Red RL work (a viral YouTube video, a Discord, a live "training broadcast" map). It is a hobby/engineering repo, not a paper. The README points to three follow-ups:
- Pleines et al., arXiv 2502.19920 (the academic baseline; see `../papers/neurips2025-pokemon-red-rl.md`)
- Pokémon RL Edition (`pokemonred_puffer`; the full-game attempt)
- PufferAI `pokegym`

## Setup (V2, the recommended version)

- **Stack:** PyBoy + Gymnasium + Stable-Baselines3 PPO. `baseline_fast_v2.py` uses `SubprocVecEnv` with **64 CPU envs**. Episode length `2048 * 80` steps. Runs are resumable from checkpoints.
- **Action space:** 7 discrete buttons (4 directions, A, B, Start). `action_freq = 24` frames per decision.
- **Observation:** a Dict of
  - 72×80 screen with 3 frame stacks
  - HP fraction
  - Fourier-encoded level sum
  - 8 badge bits
  - event-flag bits (RAM `0xD747`–`0xD87E`)
  - a local explore map
  - the last 3 actions
- **Reward (V2, `get_game_state_reward`):** a sum of
  - events: `4 * max_event_reward`
  - heal: `10 * Σ(heal_amount²)` (squared, to damp farming)
  - badges: `10 * badges`
  - exploration: `0.1 * explore_weight * unique_coords`
  - stuck: `-0.05 * revisit_count_penalty`
  - Level, opponent-level and death terms are present but commented out. Reward scale 0.5, explore weight 0.25.
- **V1 → V2 change:** V2 replaces the frame-KNN novelty reward with coordinate-based exploration. V2 "reaches Cerulean".
- **Start state:** a saved `init.state` (after the starter/parcel), plus a few other `.state` files. Ships a pretrained model and interactive runner.

## Takeaways for tensormew

1. The design is very close to Pleines et al. and is deliberately simple. Pleines used 4 terms; this V2 uses 5.
2. **Squared heal reward** is a concrete anti-exploit fix for the heal-farming failure that Pleines documents.
3. **Event-flag reward from RAM bits** is the core progress signal. Our Crystal equivalent needs the flag indices from the pret disassembly (see the battle/Crystal report; note the HM-index bug in our `src/story_flags.py`).
4. **A negative "stuck" term** works here, in contrast to the PokeAgent finding that negative rewards caused stalling in one PPO team. Treat this as an open question to test.
5. Still **Red only, early game only**. There is no Crystal support, no HM handling, no battle logic. It is a starting point, not a solution.
6. MIT-licensed: code and design can be reused (unlike `pokemonred_puffer`, which has no license).
7. Rewards are not normalised per milestone, and there is no published full-game result from this repo itself. The full-game claim belongs to `pokemonred_puffer`.

## Open items

- I did not read V1 (`baselines/`) reward code line by line, nor the video/Discord claims. The reward values above are from the V2 source and will drift if the repo changes.
- Compute and wall-clock for a Cerulean-reaching run are not stated in the README `[UNVERIFIED]`.
