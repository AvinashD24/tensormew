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
