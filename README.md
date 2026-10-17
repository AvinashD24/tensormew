# tensormew

Autonomous hybrid RL–LLM agent for Pokémon Crystal (GBC).

- **Spec:** [`SPEC.md`](SPEC.md) (§1 baseline, §1.3 Cursor Agent workflow)
- **Cursor rules:** [`.cursor/rules/crystal-hybrid.mdc`](.cursor/rules/crystal-hybrid.mdc) (`alwaysApply` — phase-gated Agent directives)
- **ROM:** `rom/pokemon_crystal.gbc` + `rom/pokemon_crystal.sym` (vendored)
- **Smoke test:** `python crystal_smoke.py --headless` after `pip install -r requirements.txt`
