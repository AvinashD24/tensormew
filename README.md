# tensormew (PokéMind)

Autonomous agent for **Pokémon Crystal** (GBC) on PyBoy — **goal: progress as far into the game as possible** during DubHacks, using checkpoint chains + LLM-on-triggers.

## Docs

- **[`SPEC.md`](SPEC.md)** — mission, architecture, RAM, checkpoints, phases H0–H4
- [`pokemon_agent.pdf`](pokemon_agent.pdf) — long-term CRYSTAL-HYBRID research (defer during hackathon)
- [`CONTEXT.md`](CONTEXT.md) — redirect stub (content merged into SPEC)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m tools.play_live --help
pytest tests/ -q
```

- **ROM:** `rom/pokemon_crystal.gbc` + `rom/pokemon_crystal.sym` (US 1.1)
- **Checkpoints:** `states/manifest.json` + `tools/make_state.py`, `tools/import_sav.py`
- **Models & Weights:** Pre-trained weights tracked under [`models/`](models/) (`models/deep_policy.pt`, `models/ppo_policy.npz`)
