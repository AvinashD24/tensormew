# PokéMind: project context (paste this into Claude)

## What we're building (DubHacks 2026, 24h, Chef's Choice track)
An event-routed hierarchical agent that plays **Pokémon Crystal** in PyBoy.
- **Low level (free, 0 tokens):** deterministic Python, a greedy mover plus hardcoded macros (Cut, use item, battle move, switch).
- **High level (LLM, only on triggers):** a Claude Haiku-class model for routine triggers, a Sonnet-class model for hard calls. It outputs **goals/tool calls**, never raw button presses.
- **Thesis / metric:** LLM calls and $ per milestone vs an **LLM-every-step baseline**, both measured by us on the same save states.
- **Pitch hook:** "Gemini beat Pokémon Blue in ~37k LLM actions / 400+ hrs. How many of those needed an LLM?" Avoid unverifiable $ claims.
- **Must-have demo:** the Ilex Forest Cut tree. The mover is blocked → the trigger sees the Cut-tree collision → the LLM calls `use_cut()` → the path clears. Show a live $ counter vs the baseline.

## Why Crystal, not Red
- **No collision grid.** PyBoy's `game_area_collision()` only exists for Red/Blue. On Crystal we read the **4 adjacent tiles** from RAM (`wTileUp/Down/Left/Right`) and use a greedy step-toward-target mover, with hardcoded paths for mazes.
- **Easier Cut.** In Gen 2, pressing A facing a Cut tree prompts "use CUT?", so the macro is face → A → A.
- **Map identity** is `(wMapGroup, wMapNumber)`.

## Environment (verified working)
- macOS, Python 3.13, venv at `.venv`, `pyboy 2.8.1`, numpy.
- ROM: `rom/pokemon_crystal.gbc`, **sha1 `f2f52230b536214ef7c9924f483392993e226cfb` = Crystal US 1.1**. Everyone needs this exact hash.
- Symbols: `rom/pokemon_crystal.sym` from the pret/pokecrystal `symbols` branch (`pokecrystal11.sym` for 1.1). All 16 needed symbols resolve.
- `pyboy.symbol_lookup(name)` returns `(bank, addr)`. **Addresses in 0xD000–0xDFFF are WRAM bank 1, so read them as `pyboy.memory[bank, addr]`.** Plain `memory[addr]` returned junk (badges=8).
- Controls in the SDL window: arrows, `a` = A, `s` = B, Enter = Start.

## Key RAM symbols (from our .sym)
| Symbol | Addr | Use |
|---|---|---|
| wMapGroup / wMapNumber | 01:dcb5 / 01:dcb6 | current map |
| wYCoord / wXCoord | 01:dcb7 / 01:dcb8 | player position |
| wFacingDirection | 01:d044 | facing |
| wTileDown/Up/Left/Right | 00:c2fa–c2fd | adjacent collision (mover + Cut-tree trigger) |
| wPlayerTileCollision | 01:d4e4 | tile under the player (`wPlayerStandingTile` does NOT exist) |
| wBattleMode | 01:d22d | 0 = overworld |
| wScriptRunning / wTextboxFlags | 01:d438 / 00:cfcf | gates for the stuck detector (dialogue, phone calls) |
| wMapStatus | 01:d432 | map state |
| wPartyCount / wPartyMon1HP | 01:dcd7 / 01:dd01 | party |
| wJohtoBadges | 01:d857 | badges bitfield |
| wNumItems / wNumKeyItems / wNumBalls | 01:d892 / d8bc / d8d7 | bag |

## Status (as of ~1pm Sat)
- [x] Repo, venv, pyboy installed; ROM verified 1.1; .sym resolves 16/16.
- [x] Banked WRAM read fix.
- [ ] **In progress:** a valid save state. The first `states/test.state` was taken during the intro (all zeros). Remake it with `--make-state`, which now prints live position and says READY.
- [ ] Walking test: `moved=True`, coords change by 1, adjacent tiles change near walls.
- [ ] Find the Cut-tree collision constant (pokecrystal `constants/collision_constants.asm`).

## Hour-0 go/no-go
1. Symbols resolve ✅
2. RAM reads are sane in-game (map != 0,0)
3. Adjacent tiles change next to walls/trees
4. `step(dir)` moves exactly 1 tile
5. Text/script flag goes nonzero in dialogue
6. One Haiku-class call returns a valid tool call
7. At least one demo save state

## Planned modules (src/)
| File | Does |
|---|---|
| emulator.py | load ROM + .sym, tick, save/load state, frame grab |
| ram_parser.py | banked reads of all symbols above → state dict + text summary for the LLM |
| mover.py | `step(dir)`, greedy `go_to(x,y)` via adjacent tiles, `run_path([...])` |
| macros.py | `use_cut()`, `use_item(id)`, `choose_move(slot)`, `switch(slot)`, `mash_text()` |
| triggers.py | facing a Cut tree; mover stuck N tries; battle start; position variance gated on battle/script/textbox |
| llm_supervisor.py | Anthropic tool-calling loop, per-call token + $ log, `always_on` flag for the baseline |
| app.py | Streamlit: frame ~10fps, thought log, live $ counter, cost chart |

## Save states needed
1. **Ilex Forest Cut tree (must-have):** after the Hive Badge + HM01, Cut taught.
2. **Whitney / Miltank fight:** type + potion choices.
3. **Union Cave or Sprout Tower:** navigation + LLM recovery.
4. 1–2 easy states (New Bark → Route 29) for mover testing.

How: a community `.sav` renamed to `rom/pokemon_crystal.gbc.ram` → continue → `--make-state`. Otherwise speed-play (Ilex is ~1.5–2 hrs in). Gotchas: phone calls pop text boxes; the RTC clock warning is harmless.

## Time split (4 people)
- **0–4h:** P1 emulator + ram_parser · P2 step + adjacent tiles · P3 tool schema + prompt + token logging · P4 save states
- **4–10h:** P1 triggers · P2 mover + macros · P3 end-to-end on the Ilex state · P4 baseline harness (always_on, ~200 steps/state)
- **10–16h:** all 3 states working. **Hour-14 check: Ilex must work end to end.** P4 collects the cost numbers + chart.
- **16–22h:** P1+P2 Streamlit · P3+P4 deck, script, backup videos
- **22–24h:** dry runs on the demo laptop

## Cut list (in order)
Map overlay → Union Cave/Sprout demo → Whitney battle loop → greedy mover replaced by hardcoded `run_path` per demo.

## Team preferences
Python only. Terse, actionable answers. Brief high-level explanation, then code.
