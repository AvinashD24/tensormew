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

**Config.** `--config blueprint_v1|legacy|path/to/config.json` validates the config locally (`src/train_config.py`),
uploads a JSON file to the pod, and passes `--config` to the remote trainer.

**Detached runs.** For multi-hour runs use `--detach`: the job runs under tmux (or `setsid nohup`) on the pod, logging
to `/workspace/tensormew_job/train.log` and writing an exit code to `/workspace/tensormew_job/done`, so a dropped SSH
session or a sleeping laptop does not kill it. The script then polls and syncs telemetry. If it is interrupted or you
close the terminal, reconnect with the same pod address:

```bash
python -m tools.cloud_train_deep --updates 400 --config blueprint_v1 --run-id v1 --detach --watch
python -m tools.cloud_train_deep --attach --watch --save logs/v1.pt   # later: resume syncing, pull the checkpoint (each run saves to logs/<run-id>.pt)
```

Exit codes: 0 ok, 1 sync failure, 2 preflight/config error, 3 remote training failed, 4 job lost (died without a
done marker, or pod unreachable), 130 interrupted. Telemetry and the checkpoint are always pulled on completion.
If the pod itself is deleted, the checkpoint and `/workspace` go with it unless you use a network volume.
