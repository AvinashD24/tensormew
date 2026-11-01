# Implementation plan: blueprint + research -> training code

Shared brief for the parallel implementation agents. Worktree: `C:\Users\arcku\tensormew\.claude\worktrees\training-viz`
(branch `feat/training-viz`). Read these first:

- `research/analysis/policy_reward_blueprint.md`: the design (route-progress reward, stage in the observation, smaller
  exploration bonuses, entropy schedule, stage-balanced starts, sticky actions, start jitter, greedy eval).
- `research/prior-work/frontier-long-horizon-training.md`: the evidence (batch size, optimizer steps, horizon, shaping).

## Goals (in priority order)

1. **Reward**: pay for progress toward the current goal, not for the commute or for exploration; remove the episode-start
   event-flag bonus (`seen_events` is cleared on reset, so every episode pays for flags already set, about +300).
2. **Policy inputs**: the policy must see the story stage, the first step of the shortest path to the current goal and the
   remaining distance (`route_features`), so one tile can map to different directions at different stages.
3. **PPO update**: recurrent PPO with env-wise minibatching (many optimizer steps per update, lower GPU memory), gamma 0.997,
   correct bootstrapping on truncation, entropy schedule, noise-scale logging. Currently 4 optimizer steps per update.
4. **Infrastructure**: sticky actions and start jitter in training, stage-balanced start states, a shared frontier archive,
   config presets, robust cloud runs (detached, bootstrapped, resumable), per-component reward logging, a greedy evaluator.
5. **Visibility**: the dashboard must show where reward comes from, so reward hacking is visible during a run.

## Rules for every agent

- Work only in the worktree above. **Do not run `git commit`, `git push`, `git stash`, or switch branches**; the
  orchestrator commits. Do not touch files you do not own (table below). If you need a change in someone else's file, say
  so in your final report instead of making it.
- Windows host; shell is Git Bash or PowerShell. Run tests with `python -m pytest <files> -q`. Python is the conda
  interpreter on PATH. No network access is needed; do not install packages. The ROM is `rom/pokemon_crystal.gbc`,
  start states are in `states/`.
- Known failures that already exist on `main` and are NOT yours to fix: `tests/test_deep_policy.py` (2 tests),
  `tests/test_emulator.py::RomCheckTests::test_wrong_bytes_rejected_before_boot`,
  `tests/test_improved_training.py::...test_load_17_dim_checkpoint_into_21_dim_policy`,
  `tests/test_ppo.py::MlpPolicyTests::test_mlp_policy_act_and_learn` (Windows temp-file / torch-load errors).
- Keep the rest of the suite green. If an existing test asserts an old value you intentionally changed, update that test
  and say so in your report.
- Match the surrounding code's style and comment density. Add tests for new behavior in the new test file you own.
- Do not claim anything works without running it. Final report: what you built, files changed, test results (counts), anything
  you could not finish, and any change you need from another owner. Keep it under 400 words.

## File ownership

| Agent | Owns (create or edit) | Must not touch |
|---|---|---|
| A1 route graph | `src/route_graph.py`, `tests/test_route_graph.py` | everything else |
| A2 reward + obs | `src/rl_env.py`, `tests/test_reward_blueprint.py`, any existing test that asserts old reward/obs values | `tools/train_deep_ppo.py` |
| A3 PPO algorithm | `src/ppo_update.py`, `tests/test_ppo_update.py` | `tools/train_deep_ppo.py`, `src/deep_policy.py` (read only) |
| A5 dashboard + eval | `tools/dashboard_static/index.html`, `tools/dashboard.py`, `src/live_feed.py`, `tests/test_live_feed.py`, `tools/evaluate_policy.py`, `tests/test_evaluate_policy.py` | `tools/train_deep_ppo.py` |
| A6 infra scripts + config | `src/train_config.py`, `tests/test_train_config.py`, `tools/cloud_train_deep.py`, `tools/pod_bootstrap.sh`, `tests/test_cloud_train.py`, `tools/sync_weights.py` | `tools/train_deep_ppo.py` |
| A4 trainer integration (wave 2) | `tools/train_deep_ppo.py`, `tests/test_curriculum_and_recurrent_training.py` | modules above except to fix integration bugs reported by their owners |

## Contracts (code against these exact names; they are being written in parallel)

### `src/route_graph.py` (A1)

```python
STAGE_COUNT = 4          # 0 GET_EGG (to Mr. Pokemon's house), 1 RETURN_EGG (deliver to Elm),
                         # 2 TO_GYM (walk to Falkner), 3 POST_GYM (Togepi egg at the Violet Pokecenter)
ROUTE_FEATURE_DIM = 9    # stage one-hot (4) + next-step one-hot over up/down/left/right (4) + remaining distance/100 clipped to [0,1] (1)

def current_stage(state) -> int                      # from event flags via state.has_event(name)
def covers_map(map_id: tuple[int, int]) -> bool      # True if the route graph has this map (exploration bonuses shrink there)
def route_potential(state) -> float                  # Phi = -(steps remaining along the WHOLE route to the final goal);
                                                     # continuous across goals; telescopes around a full circle
def route_features(state) -> numpy.ndarray           # float32, shape (ROUTE_FEATURE_DIM,)
def goal_remaining(state) -> float | None            # steps to the CURRENT goal, None if unknown
```

`state` is the parsed game state used everywhere in `src/rl_env.py` (`map_group`, `map_number`, `x`, `y`, `has_event(name)`,
`event_flags`, `party_hp`, ...). The graph is built from ROM-derived map grids using the existing helpers in
`src/return_nav.py`, `src/falkner_nav.py`, `src/collision.py`; it merges both navigators into one graph with warps.
Must be cheap enough to call every env step (cache the BFS distance fields per stage).

### `src/rl_env.py` additions (A2)

```python
ROUTE_PROGRESS_WEIGHT = 0.08                         # reward = 0.08 * (Phi(s') - Phi(s)), gamma = 1 so it telescopes
def initial_seen_events(state) -> set[int]           # flag ids already set in state.event_flags (seed seen_events after reset)
def progress_reward(..., components: dict[str, float] | None = None)   # same positional args as today; when a dict is
                                                     # passed, add each term's contribution under a stable key:
                                                     # step, bump, route, tile, neighbor, map, warp, interaction, event,
                                                     # badge, catch, level, battle, damage, heal, whiteout, behind_map, other
OBS_DIM                                              # updated to include ROUTE_FEATURE_DIM; train script imports it
```

### `src/ppo_update.py` (A3)

```python
@dataclass
class PPOHParams:   # defaults: gamma=0.997, gae_lambda=0.95, clip_ratio=0.2, vf_coef=0.5, entropy_coef=0.02,
                    # max_grad_norm=0.5, epochs=3, num_minibatches=4, target_kl=0.05 (None disables), norm_adv=True,
                    # vf_clip=True, adv_clip=None
def compute_gae(rewards, values, dones, last_value, gamma, lam, truncated=None, bootstrap_values=None)
        # tensors (T, B); returns (returns, advantages) of shape (T, B). `truncated[t, b]` marks time-limit ends:
        # bootstrap with bootstrap_values[t, b] (value of the final observation) instead of treating them as terminal.
@dataclass
class RolloutBatch:  # screens (B, T, 3, H, W), rams (B, T, D), actions (B, T), old_logps (B, T),
                     # returns (B, T), advantages (B, T), initial_hx (1, B, H) or None
def ppo_update(policy, optimizer, batch: RolloutBatch, hp: PPOHParams, entropy_coef: float | None = None) -> dict
        # env-wise minibatching for recurrent policies (each minibatch = a subset of envs with their full T-step
        # sequences and their own initial hidden state); one optimizer step per minibatch; returns stats:
        # policy_loss, value_loss, entropy, approx_kl, clip_frac, grad_norm, n_optimizer_steps, explained_variance,
        # noise_scale (simple gradient-noise-scale estimate, NaN if unavailable), early_stopped (bool)
class EntropySchedule:  # EntropySchedule(start=0.03, end=0.005).value(progress in [0, 1]) -> float (linear)
```

### `src/live_feed.py` (A5)

`LiveFeed.observe(infos=...)` accepts `info["reward_components"]` (dict of the keys above, per-step values) and accumulates
them per update and per run; they appear in `state.json` and each history record under `"components"`.

### `src/train_config.py` (A6)

```python
@dataclass
class TrainConfig:  # every training knob: envs, rollout_steps, updates, max_steps, lr, ppo (PPOHParams fields), entropy
                    # schedule start/end, sticky_prob, jitter_max, start_states, stage_balanced, swarm, seed, ...
def load_config(path_or_preset: str) -> TrainConfig  # JSON file or preset name ("blueprint_v1", "legacy")
def apply_overrides(cfg, overrides: dict) -> TrainConfig
```

## Waves

1. A1, A2, A3, A5, A6 run in parallel (disjoint files).
2. A4 integrates everything into `tools/train_deep_ppo.py` and runs an end-to-end smoke test (2 envs, 2 updates, local).
3. Review: full test suite, spec check against the blueprint, reward-component sanity run, final summary.
