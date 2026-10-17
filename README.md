# tensormew (PokéMind)

Autonomous agent for **Pokémon Crystal** (GBC) on PyBoy — **goal: progress as far into the game as possible** during DubHacks, using checkpoint chains + LLM-on-triggers.

## Docs

- **[`SPEC.md`](SPEC.md)** — **only spec you need** (mission, architecture, RAM, checkpoints, phases H0–H4, Cursor workflow)
- [`pokemon_agent.pdf`](pokemon_agent.pdf) — long-term CRYSTAL-HYBRID research (defer during hackathon)
- [`CONTEXT.md`](CONTEXT.md) — redirect stub (content merged into SPEC)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python crystal_smoke.py --headless
pytest tests/ -q
```

- **ROM:** `rom/pokemon_crystal.gbc` + `rom/pokemon_crystal.sym` (US 1.1)
- **Checkpoints:** `states/manifest.json` + `tools/make_state.py`, `tools/import_sav.py`
- **Cursor rules:** [`.cursor/rules/crystal-hybrid.mdc`](.cursor/rules/crystal-hybrid.mdc)
