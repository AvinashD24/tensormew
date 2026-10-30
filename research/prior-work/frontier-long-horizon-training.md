# How long-horizon games get beaten, and what it means for tensormew's training

Compiled 2026-10-30. Builds on `rl-techniques-long-horizon.md` and `rl-for-games-history.md` (October 19); this note adds
primary-source numbers on **batch size, optimizer steps, horizon and reward shaping**, and compares them with this repo's
actual training code and its first from-scratch run (`nb-fresh-1`, 200 updates).

**Verification tags.** Unmarked = read in a primary source this session. `[summary]` = from a model-extracted summary of a
primary source (re-check before quoting). `[UNVERIFIED]` = secondary source or not re-checked. `[computed]` = my arithmetic.

## 1. Bottom line

1. Every system that beat a long-horizon game did it with **orders of magnitude more samples per update and in total**
   than tensormew has used. Pokémon RL practice is ~65k samples per update and 10^8-10^10 total env steps; our runs were
   256-2,048 per update and 5x10^4-2x10^5 total.
2. Our recurrent PPO update takes **4 optimizer steps per update** (whole batch per epoch). The community takes 24-96.
   More samples per update would not fix that; **minibatching over envs** would (and it cuts GPU memory).
3. `progress_reward` pays a ~+300 one-time bonus at the start of every episode (verified in the `nb-fresh-1` history).
   It is constant w.r.t. the agent's actions, but it dominates advantage normalization and the critic loss.
4. Frontier labs relied on stable, mostly fixed rewards, long discount horizons, huge batches, and (when the setup changed)
   function-preserving "surgery" instead of restarting. LLM labs that "beat Pokémon" used harnesses, not RL.

## 2. What was done elsewhere

| System | Algorithm / setup | Scale and key settings | Source |
|---|---|---|---|
| OpenAI Five (Dota 2) | PPO + 4096-unit LSTM, ~159M params, GAE lambda 0.95 | Batch up to 2,949,120 timesteps (1536 GPUs); abstract: "~2 million frames every 2 seconds"; 10 months, 770 PFlops/s-days; Adam; entropy 0.01 default | arXiv 1912.06680 Sec 3.1-3.2, abstract |
| OpenAI Five horizon | Horizon H = T/(1-gamma), T = 0.133 s per step | Base agent 180 s horizon (gamma 0.9993); resuming at longer horizons helped up to 6-12 min | Sec 4.5, Fig 6 |
| OpenAI Five batch scaling | 983k-step batch vs 123k baseline | ~2.5x speedup for ~8x batch: "sublinear" | Sec 4.3 |
| OpenAI Five sample reuse | PPO epochs on the same data | Reusing data "even 2-3 times can cause a factor of two slowdown"; target ~1 | Sec 4.4 |
| OpenAI Five reward | Hand-written once, then mostly left alone; ~20 terms (win +5, tower/barracks, gold, xp, kills, small per-step terms) | "We constructed the reward function once at the start of the project"; shaping "loosely potential-based" [summary]; non-win terms decayed over game time [summary]; team spirit annealed 0.3 -> 1.0 | Sec 3.2, App G [summary] |
| OpenAI Five "surgery" | Net2Net-style function-preserving edits when obs/model/env change | 20+ surgeries; new weights initialised so the old function is unchanged; but "plateaued at a weaker skill level than the from-scratch model" | Sec 3.3, 4.2 |
| Pleines et al. 2025 (Pokemon Red) | PPO, ~2M params (4M with GRU), 4 reward terms, no entropy bonus | 32 workers x 2048 = **65,536 batch**, 8 minibatches, 3 epochs, gamma 0.997, ~400M env steps [approx, read off a plot], ~36 h; reaches Cerulean City only | arXiv 2502.19920 (`papers/neurips2025-pokemon-red-rl.md`) |
| pokemonred_puffer (beats Red with scripted help) | PPO + LSTM(512), swarm/save-state sharing, >25 reward terms | `config.yaml`: num_envs 288, **batch 65,536**, minibatch 2,048, bptt 16, 3 epochs, gamma 0.998, lambda 0.95, clip 0.1, ent 0.01, lr 2e-4, total_timesteps 10B (comment: 100B for full games). Book: ~10k steps/s on 4x RTX 4090 boxes, "beat Brock in less than 30 minutes" | github.com/drubinstein/pokemonred_puffer `config.yaml`; docs chapter 3 (config values are current defaults, not necessarily the published run's) |
| Go-Explore | Archive of states, return then explore, then robustify with imitation | Solved Montezuma/Pitfall; needs resettable states (we have them) | Nature 2021 (see `rl-techniques-long-horizon.md`) |
| AlphaStar | Supervised on human replays, then RL (TD(lambda), V-trace, UPGO) with KL to the supervised policy, plus league training | Imitation prevents forgetting; pseudo-rewards tied to human statistics [UNVERIFIED, blog summary] | DeepMind blog / Nature 2019 |
| Gemini Plays Pokémon | Not RL: LLM + harness (memory, sub-agents, skills) | Completed Blue (May 2025), Yellow Legacy hard (Aug 2025), Crystal (Nov 2025). Authors: "the harness rather than the episode is the transferable unit" [summary] | arXiv 2605.09998 |

Critical batch size theory (`arXiv 1812.06162`, McCandlish, Kaplan, Amodei, OpenAI Dota team): a measurable "gradient noise
scale" estimates the largest useful batch size within about an order of magnitude; critical batch sizes range from ~20 (a
small autoencoder) to over 10 million (Dota); it rises as the loss falls. **Only the abstract-level claims were confirmed**
(the PDF text could not be extracted this session) `[UNVERIFIED details]`.

## 3. Reward-shaping lessons that recur

1. **Few, well-separated terms.** Pleines: event +2, new-tile +0.005, plus capped level and heal terms. Ten times the
   exploration reward made agents avoid battles and never beat the first gym; no exploration reward left them idle.
2. **Events dominate, exploration is a tie-breaker.** Ratio of a new tile to a story event: Pleines 1:400; pokemonred_puffer
   ~1:40 (exploration 0.019 vs event 0.75) up to ~1:370 (required_event 7.0) [computed from config.yaml].
3. **Every added term is an exploit surface.** Documented: heal farming (Leech Seed loops), gambling at hidden objects,
   tree-cutting loops, avoiding battles to protect exploration reward, skipping gyms. Squared heal reward and per-center
   caps are the fixes in use.
4. **Potential-based shaping** (Ng et al. 1999) keeps the optimal policy; OpenAI Five's shaping was "loosely" of this form.
   ICML 2025 work (ADOPS, Forbes et al.) extends it to intrinsic-motivation bonuses in very sparse games.
5. **Positive-only rewards** worked better in one PokeAgent entry (step penalties caused stalling); pokemonred_puffer's
   V2 sibling uses a small "stuck" penalty. Treat negative terms as an experiment, not a default.
6. **Do not keep re-tuning mid-run.** OpenAI Five kept its reward fixed for months and used surgery for model changes; a
   manual hyperparameter change before their final event "lost both games" [summary, App Q].
7. **Long discount horizon.** Five used ~3 minutes of game time and then grew it; Pleines tuned gamma over
   {0.99, 0.997, 0.999, 0.9995} and kept 0.997.

## 4. tensormew today versus that practice

| | tensormew (this repo) | Pleines 2025 | pokemonred_puffer | OpenAI Five |
|---|---|---|---|---|
| Samples per update | 256 (nb-fresh-1), 2,048 (earlier 4090 runs) | 65,536 | 65,536 | ~1M-2.9M |
| Optimizer steps per update | **4** (recurrent path: whole batch x 4 epochs) | 24 (8 x 3) | ~96 (32 minibatches x 3) [computed] | "32 PPO gradient steps" per batch (Sec 4.4) |
| Total env steps so far | 51,200 (nb-fresh-1); ~205k per earlier 100-update run [computed] | ~400M | 10B configured | very large |
| gamma / lambda | 0.99 / 0.95 (`compute_gae` defaults) | 0.997 / 0.95 | 0.998 / 0.95 | 0.9993 / 0.95 |
| clip / entropy / lr | 0.2 / 0.02 / 3e-4 | 0.2 / 0 / 3e-4 | 0.1 / 0.01 / 2e-4 | - / 0.01 / 5e-5 [summary] |
| Episode length | 1,024 | 10,240 + 2,048 per event | 19,816 mini-episodes | n/a |

Observed throughput on the 8-env Community RTX 3080 pod: ~250 env steps/s `[computed: 51,200 steps in ~200 s]`.

## 5. Findings in this repo (verified against the code and the `nb-fresh-1` history)

1. **Episode-start event bonus.** `tools/train_deep_ppo.py` clears `seen_events` on every reset (line 120);
   `src/rl_env.py:progress_reward` pays `KEY_EVENT_REWARDS` (5-50) or +2 for every set flag not in `seen_events`. The first
   step of each episode therefore pays for all flags already set in the start state. Evidence: mean reward at updates 0, 32,
   64, 96, 128, 160, 192 is 308, 307, 331, 316, 350, 352, 352 (all 8 envs reset together every 1,024 steps = 32 updates) versus
   about -8 to +35 elsewhere; critic loss at those updates is 1,385-1,814 versus 0.1-60. It is independent of the action
   taken, but one outlier per env per batch dominates `adv = (adv - mean) / std` and the value loss. *Fix to consider:* seed
   `seen_events` with the flags set after each reset.
2. **Too few gradient steps.** The recurrent branch of the PPO update does `for epoch in range(4)` over the entire batch,
   i.e. 4 optimizer steps per update; `nb-fresh-1` made ~800 in total. Env-wise minibatching (split the batch by env, carry
   each env's initial hidden state, step the optimizer per minibatch) is standard recurrent PPO. It would also cut peak GPU
   memory, which is what overflowed a 10 GB card at 1,024 frames (about 9 MB of activations per frame `[computed]`).
3. **Discount horizon is short.** gamma 0.99 gives an effective horizon of ~100 steps; the Pokémon baselines use 0.997-0.998
   (~333-500 steps).
4. **Exploration is probably over-weighted.** A new tile pays 1.0 (2.0 on objective maps), a new map +10, plus a 0.2
   neighbor bonus and a 0.08/sqrt(n) curiosity term, against events of 5-50. The Pokémon baselines keep a tile at 1/40-1/400
   of an event. The `nb-fresh-1` status was "looping/stuck" for much of the run. This is a hypothesis to test, not a result.
5. **Sample budget.** `nb-fresh-1` used 51,200 env steps, ~0.01% of Pleines' budget. At ~250 steps/s, 10M steps is ~11 hours
   on that pod; pokemonred_puffer reports ~10,000 steps/s across four 4090 machines `[computed estimates]`.

## 6. Suggested next run (ordered; each is a reversible change)

1. Re-run for a longer budget before changing the algorithm, so results are not noise: aim for 5-20M env steps.
2. Fix the episode-start bonus (finding 1) and re-check that mean reward is smooth across updates 0, 32, 64.
3. Add env-wise minibatching to the recurrent update; target 16+ envs x 128 steps, 4-8 minibatches, 3 epochs.
   Expect this to fit on a 24 GB card with room to spare and possibly on the 10 GB 3080 `[untested]`.
4. Raise gamma to 0.997 and test removing the per-step -0.01 and capping tile rewards; one change at a time.
5. Keep the reward fixed within a run (Five's lesson). If the reward or policy must change mid-project, use a
   function-preserving transfer (keep trunk, re-initialise the value head) and compare against a from-scratch control.
6. Longer term: swarm/Go-Explore sharing of new-event save states (pokemonred_puffer; `src/frontier_buffer.py` is the
   start), and LLM-written subgoals via the harness approach (`llm-agents-pokemon.md`).
7. Measure the gradient noise scale to choose batch size empirically instead of guessing (cheap to log per update).

## 7. Open questions and unverified items

- The large-batch paper's exact critical-batch figures and formulas (PDF unreadable here).
- OpenAI Five's learning-rate and exact entropy schedule are from a model-extracted appendix summary.
- Whether the ~300 start bonus actually harms learning, versus only inflating logged reward, is not isolated; the pattern and
  the mechanism are verified, the causal effect is not.
- Gemini step counts and token costs per completed game were not found.
- pokemonred_puffer `config.yaml` is the current default, not necessarily the published run's setting.

## Sources

- OpenAI Five: https://arxiv.org/abs/1912.06680 (HTML v1 read: https://arxiv.org/html/1912.06680v1)
- Large-batch training: https://arxiv.org/abs/1812.06162
- Pleines et al.: https://arxiv.org/abs/2502.19920
- pokemonred_puffer: https://github.com/drubinstein/pokemonred_puffer (`config.yaml`), https://drubinstein.github.io/pokerl/
- Continual Harness / Gemini Plays Pokémon: https://arxiv.org/abs/2605.09998
- Go-Explore: https://arxiv.org/abs/1901.10995; AlphaStar: https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- Reward shaping: https://arxiv.org/abs/2505.12611 (ADOPS, ICML 2025); https://arxiv.org/abs/2402.07411
- PokeAgent Challenge: https://arxiv.org/abs/2603.15563
