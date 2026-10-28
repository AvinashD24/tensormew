# Hybrid run: start → egg → Elm → Falkner (with recording)

One command plays Crystal from the New Bark start to Falkner's badge, in a window, and records it.

```bash
# Linux / macOS / Git Bash
tools/run_hybrid.sh my_run
# Windows PowerShell
tools\run_hybrid.ps1 my_run
```

Output goes to `recordings/my_run/`:

| File | What |
|---|---|
| `run.gif` | the run, one frame every 6 steps (`--record-every`), rewritten every 100 frames and again when Falkner is beaten |
| `<milestone>.state` | an emulator save state per story milestone (`got_egg`, `gave_egg`, `gym_entered`, `falkner_beaten`, ...). Resume from one with `--state recordings/my_run/gave_egg.state` |

Recordings can be committed (`*.gif` is no longer ignored, and `recordings/**/*.state` is un-ignored). Samples:

- `recordings/falkner_showcase/run.gif`: the paced showcase run described below (Violet City → gym → Falkner, 0 blackouts).

Recording needs a window (no `--headless`): headless runs render blank frames, so the GIF would be blank.
It reads PyBoy's own frame buffer, so the window can sit behind others.

## Showcase mode (paced for viewers)

```bash
python -m tools.play_live --state recordings/falkner_run_4/violet.state \
  --policy models/deep_policy_baseline_351cb55.pt --post-egg-policy models/deep_policy.pt \
  --speed 2 --watch --record-dir recordings/falkner_showcase --record-frames-every 6
```

`--speed 2` is twice game speed. `--watch` draws every frame (the macros normally advance the emulator without drawing, which
freezes the window during menus and battles) and gives battle text 70 frames before the next skip press.
`--record-frames-every 6` adds a smooth ~10 fps recording while the run is in Violet City / the gym; elsewhere it records one frame
every `--record-every` steps.

## What is RL and what is hardcoded

| Stage | Who decides |
|---|---|
| New Bark → Mr. Pokémon, up to the egg | **RL**: `models/deep_policy_baseline_351cb55.pt` (`--policy`), with safety layers (blocked-tile mask, anti-backtrack, `src/unstick.py`) |
| Egg back to Elm (Cherrygrove rival, Route 29, lab hand-over) | hardcoded: `src/return_nav.py` (ROM shortest path) + steps in `tools/play_live.py` |
| Lab → Violet City → gym | hardcoded: `src/falkner_nav.py` (ROM shortest path, S-curve bridge) |
| Training to Lv 15 on Route 30, healing at Cherrygrove / Violet, healing between gym trainers | hardcoded: `src/gym_prep.py` |
| Every battle (moves, items, switching, fleeing, fainted Pokémon) | hardcoded rules: `src/battle.py`, `src/macros.py` |
| Optional LLM "supervisor" on bottlenecks | on only if `ANTHROPIC_API_KEY` is set (`tools/play_live.py`) |

`--post-egg-policy models/deep_policy.pt` hands control to the second RL policy once `EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON`
is set, but the hardcoded stages above cover almost all of that part, so it rarely acts. Both weight files are in the repo.

## Why the lead is trained first

Falkner (Crystal): Pidgey Lv 7 + Pidgeotto Lv 9-10, both with Mud-Slap (super effective on Cyndaquil, lowers accuracy).
Guides say bring the starter to about Lv 12+ (Cyndaquil learns Ember at Lv 12). `gym_prep` farms wild battles in Route 30's tall grass
until the lead is Lv 15, walks to the Cherrygrove Pokémon Center under 50% HP, arrives in Violet City at full HP, and leaves the gym
to heal under 60% between the Bird Keepers and Falkner (beaten trainers stay beaten). In trainer battles the bot never switches
out voluntarily to a much weaker Pokémon, and at most once per battle.

## Things worth knowing

- Faint handling (`Macros.switch_fainted_pokemon`): the party menu opens by itself with the cursor ON the fainted Pokémon, so
  pressing A selects it ("no will to battle"). The macro waits for the menu, lets it settle, then cycles with DOWN + A, stepping
  over fainted rows. `wMenuCursorY` (0xCFA9) is the live cursor; `wPartyMenuCursor` (0xD0D8) is only the last selection.
- The ROM tile table (`TilesetJohtoColl`) is only right for outdoor maps; the gym interior is walked with hardcoded steps.
- The collision table marks a hop-ledge row as not enterable sideways, but you can walk along it (Route 30, beside Mikey):
  `Unsticker` tries both sidesteps and lets the stepper report whether it moved.

## Tests

```bash
python -m pytest -q tests -k "gym_prep or battle or unstick or macro or falkner or live_status or collision or egg_rival"
```

(Some unrelated tests fail on Windows from temp-file locks, and one needs a save state that is not in the repo.)
