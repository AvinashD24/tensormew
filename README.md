# tensormew (PokéMind)

Autonomous agent for **Pokémon Crystal** (GBC) on PyBoy — **goal: progress as far into the game as possible** during DubHacks, using checkpoint chains + LLM-on-triggers.

## Docs

- **[`SPEC.md`](SPEC.md)** — mission, architecture, RAM, checkpoints, phases H0–H4

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m tools.play_live --help
pytest tests/ -q
```

- **ROM:** `rom/pokemon_crystal.gbc` + `rom/pokemon_crystal.sym` (US 1.1)
- **Checkpoints:** `states/manifest.json` + `tools/make_state.py`, `tools/import_sav.py`

## Copy-paste commands

Run these from the repo root. Windows PowerShell shown first; the cloud commands run in WSL (they need `rsync`).

**0. Activate the environment (PowerShell)**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**1. Open the dashboard** (live view of a run, plus replay of any saved run). It opens http://127.0.0.1:8765/ ; pick a
saved run under **Source** to replay it. Leave it running in its own terminal.

```powershell
python -m tools.dashboard
python -m tools.dashboard --demo        # no training needed: a synthetic run to look at the UI
```

**2. Train on this machine** (CPU, slow; good for a quick check). Starts every agent at New Bark with fresh weights and
saves to `logs/<run-id>.pt`; watch it in the dashboard from step 1.

```powershell
python -m tools.train_deep_ppo --config long_v3 --updates 3 --envs 4 --rollout 16 --max-steps 48 --no-curriculum --state new_bark_outside --run-id local-check
```

**3. Train on a GPU pod** (the real thing). First create a pod in the Runpod console, copy its SSH host and port from the
Connect tab (they change every time a pod starts), and keep your SSH key at `~/.ssh/id_runpod` inside WSL
(`chmod 600`). Then, from WSL in the repo folder (adjust host, port and run id):

```bash
python3 -m tools.cloud_train_deep --host <POD_IP> --port <POD_PORT> --config long_v2 \
  --run-id my-run --updates 300 --envs 32 --no-curriculum --state new_bark_outside \
  --detach --watch --no-dashboard --sync-every 5
```

This uploads the code, installs `pyboy` on the pod, trains there, and copies telemetry into `logs/` every 5 seconds so the
dashboard from step 1 shows it live. The checkpoint comes back as `logs/my-run.pt` when the run ends.
Use `--envs` up to the pod's core count (a 32 vCPU pod: 32). Each update takes roughly 25 s at 32 envs.
`--config long_v4` = `long_v3` plus a **dense path-progress reward** (`path_progress`, `path_progress_scale`, `src/path_progress.py`): +0.08 per tile of NEW best progress along the 319-step story path (progress = 319 - BFS steps to Falkner over the ROM map graph, warps included), best-ever per episode, never negative, nothing for stepping back or re-walking, free for the start state's own progress (archive starts), a new best resets the dwell window; reported as the `path` reward component (about 25.5 for the whole path). Reward only: the observation is unchanged. Ablate with `{"preset": "long_v4", "path_progress": false}`.
`--config long_v4` additionally turns on (each field can be switched off, e.g. `{"preset": "long_v4", "aux_loss": false}`): **gamma 0.999** (horizon 1000 instead of 333 steps; `gae_lambda` stays 0.95; the whole 256-step rollout is one BPTT sequence and the GRU state is carried across rollouts), **entropy 0.01 -> 0.001** (was 0.04 -> 0.01), an **exploration budget** (`explore_budget` 12 per episode instead of 30, intrinsic weight `intrinsic_weight_end` 0.4 reached after `intrinsic_decay_updates` 800 updates), a **heal reward for HP actually restored** (`heal_reward`, `heal_reward_scale` 4, `heal_budget` 8, `src/heal_reward.py`: concave in the HP fraction so low-HP heals pay most, nothing for level-ups or white-out recovery), **low-HP battle practice** (`battle_lowhp_weight` 3: battle-archive states with the lead at <= `battle_lowhp_threshold` 40% HP and a heal item in the bag are 4x as likely to be loaded), **auxiliary prediction heads** (`aux_loss`, `aux_coef` 0.05, `aux_horizon` 8, `src/aux_tasks.py`: map id, lead-HP change and battle-start k steps ahead, targets read from the stored RAM observations; old checkpoints still load, the heads are not needed at inference) and **milestone success rates** (`ms_reach*`, `ms_envs_at`, `path_best_*` in the metrics, history and `state.json`; dashboard panel "Milestone reach rate" and the "Reaching further milestones?" check). Literature and status of each: `research/analysis/design_hardcoded_vs_learned.md`.
`--config long_v3` is the earlier fresh from-scratch preset: `long_v2` (32 envs x 256 steps, 800 updates, New Bark
start, no route oracle, the agent plays its own battles) plus staggered episodes (resets no longer happen in lockstep;
`phase_spread` and `reward_8` are logged), a self-generated battle archive with an early A-bias inside battle menus, and
furthest-map weighted archive resets. Each is a config flag (`stagger_episodes`, `battle_archive`, `battle_a_bias`,
`swarm_furthest_weighting`), so any of them can be ablated. `long_v3` also has a **dwell penalty** (`dwell_penalty`, `src/dwell.py`): staying in at most 16 tiles for 192 non-battle steps with nothing new (tile, map, milestone, new species) costs 0.005 -> 0.03 per step (reported as the `dwell` reward component, at most 15 per episode, reset by leaving the area). It is zero while the party's highest level is below `dwell_level_targets[furthest story stage]` (default `[8, 8, 10, 10, 10, 11, 12, 12, 0]`, stages New Bark ... Violet Gym, Falkner beaten) and the agent fought or leveled within `dwell_grind_grace` (512) steps, so leveling for Falkner is not punished; the older per-tile `loop` term is switched off in that case too (`dwell_exempt_loop`). Pokecenter maps are exempt. Example ablation: `{"preset": "long_v3", "dwell_penalty": false}`. Other ablations, e.g. `--config path/to/json` with `{"preset": "long_v3", "battle_a_bias": 0}`. **Checkpoints from before this change cannot be loaded** (the observation
lost the route direction/distance features, gained battle features, and the policy gained a B button); loading one
raises `CheckpointIncompatibleError` telling you to retrain.

**4. Resume a run from its checkpoint** (needs `logs/my-run.pt` and `logs/my-run.optim.pt` locally; the script uploads them):

```bash
python3 -m tools.cloud_train_deep --host <POD_IP> --port <POD_PORT> --config long_v2 \
  --resume --save logs/my-run.pt --run-id my-run-2 --updates 200 --envs 32 --no-curriculum --state new_bark_outside \
  --detach --watch --no-dashboard
```

**5. Reconnect to a run that is still going on the pod**, resume syncing and pull the checkpoint when it ends:

```bash
python3 -m tools.cloud_train_deep --host <POD_IP> --port <POD_PORT> --attach --watch --no-dashboard --save logs/my-run.pt
```

**6. Evaluate a checkpoint** (greedy, from the New Bark save) and replay it:

```powershell
python -m tools.evaluate_policy --checkpoint logs/my-run.pt --episodes 3
```

**7. Stop paying when you are done.** A pod bills until you stop or terminate it, even if training has ended or crashed.
Terminate it in the Runpod console (Pods -> your pod -> Terminate) after the checkpoint is back in `logs/`.
A stopped pod often cannot restart (its GPU gets taken), so copy results first and make a new pod next time.

Troubleshooting: if the dashboard shows nothing, the run id folder under `logs/runs/` is the source of truth (pick it under
**Source**). If a long cloud run dies, its latest checkpoint is on the pod in `/workspace/tensormew/logs/`; pull it with
`rsync -az -e "ssh -i ~/.ssh/id_runpod -p <POD_PORT>" "root@<POD_IP>:/workspace/tensormew/logs/my-run*" logs/`.

## What is hard-coded and what is learned

The agent starts from `states/new_bark_outside.state` (Cyndaquil, outside New Bark Town). The policy gets the screen plus
a RAM vector (position, map id, party, bag, which story milestones are done, battle facts) and presses six buttons
(up/down/left/right/A/B). It is **not** told where to go and it plays **its own battles** (fight, move choice, switching,
items, running). The goal is the minimal story path to the Zephyr Badge (Route 29 -> Cherrygrove -> Route 30 ->
Route 31 -> Violet City -> Violet Gym -> Falkner); the Mystery Egg errand is not required. Still hard-coded by design:
the start state, 10 Poke Balls + 5 Potions at every reset, auto-advanced *overworld* dialogue, and the reward terms.
Details and reasoning: [`research/analysis/design_hardcoded_vs_learned.md`](research/analysis/design_hardcoded_vs_learned.md).
The old BFS goal potential survives only as the opt-in `goal_potential` config flag (off in every preset).

## Training from scratch

This branch carries the infrastructure only: no trained weights, recordings or web app.

```bash
python -m tools.train_deep_ppo --updates 100 --save logs/deep_policy.pt   # fresh policy
python -m tools.train_deep_ppo --resume ...                               # opt in to warm-start
python -m tools.train_deep_ppo --push-weights ...                         # opt in to commit/push weights to THIS branch
```

Resume and weight-push are both off by default. `tools/sync_weights.py` refuses to push from `main`.

## Live training dashboard

Training streams telemetry to `logs/live/` (disable with `--no-live`). Watch it in a browser while it runs:

```bash
python -m tools.dashboard          # http://127.0.0.1:8765, reads logs/live + logs/training_metrics.jsonl
python -m tools.dashboard --demo   # preview the UI with a synthetic run, no training or ROM needed
```

It shows live game screens from every parallel emulator, reward / entropy / KL / loss charts with hover,
a training-phase timeline, per-map exploration heatmaps with agent trails, the action mix, recent episodes,
milestones (new maps, badges, records) and automatic health warnings (entropy collapse, KL spikes, stuck exploration, stale feed).
### Replaying a finished run

Every run also saves an append-only history under `logs/runs/<run-id>/` (`history.jsonl` with each update's
metrics, per-agent positions, tile-visit deltas, episodes and milestones; `frames/` with a JPEG montage of all
agents every `--frame-every` updates, default 5; `meta.json`). Open the dashboard and pick the run under **Source**
to scrub or play it back: charts, heatmap, milestones and the agent montage all rewind together.
Name a run with `--run-id`; disable history with `--runs-dir ""`.

### Cloud (GPU pod) training

Telemetry is written on the pod, so bring it home while the run is going:

```bash
python -m tools.cloud_train_deep --updates 200 --run-id baseline-1 --watch
```

`--watch` rsyncs `logs/live/`, `logs/runs/` and `logs/training_metrics.jsonl` from the pod every `--sync-every` seconds
(default 5), starts the dashboard locally, and stays up after the run so you can replay it. Telemetry is always
pulled at the end, even if training fails or you Ctrl+C, so a destroyed pod no longer takes the data with it.
Manual alternative: run `python3 -m tools.dashboard --no-open` on the pod and tunnel with
`ssh -L 8765:127.0.0.1:8765 root@<pod>`.

**Pod address.** A pod gets a new SSH host/port every time it starts, so set `RUNPOD_SSH_HOST` / `RUNPOD_SSH_PORT`
(or pass `--host/--port`) per session. Community Cloud stock for 4090-class GPUs is often empty; if a pod will not
start, try Secure Cloud or another GPU type rather than waiting.

**Preflight.** Before training, the script SSHes in and checks the GPU (`nvidia-smi`), core count (`nproc`),
free disk and python deps. The stock PyTorch image lacks `pyboy` and Ubuntu 24.04 blocks plain pip, so it runs
`pip install --break-system-packages pyboy==2.8.1` automatically (`tools/pod_bootstrap.sh` does the same by hand).
`--envs` above the core count is capped with a warning. Skip with `--skip-preflight`.

**Config.** `--config long_v4|long_v3|long_v2|blueprint_v1|long_v1|legacy|path/to/config.json` validates the config locally (`src/train_config.py`),
uploads a JSON file to the pod, and passes `--config` to the remote trainer.

**Detached runs.** For multi-hour runs use `--detach`: the job runs under tmux (or `setsid nohup`) on the pod, logging
to `/workspace/tensormew_job/train.log` and writing an exit code to `/workspace/tensormew_job/done`, so a dropped SSH
session or a sleeping laptop does not kill it. The script then polls and syncs telemetry. If it is interrupted or you
close the terminal, reconnect with the same pod address:

```bash
python -m tools.cloud_train_deep --updates 400 --config long_v2 --run-id v1 --detach --watch
python -m tools.cloud_train_deep --attach --watch --save logs/v1.pt   # later: resume syncing, pull the checkpoint (each run saves to logs/<run-id>.pt)
```

Exit codes: 0 ok, 1 sync failure, 2 preflight/config error, 3 remote training failed, 4 job lost (died without a
done marker, or pod unreachable), 130 interrupted. Telemetry and the checkpoint are always pulled on completion.
If the pod itself is deleted, the checkpoint and `/workspace` go with it unless you use a network volume.
