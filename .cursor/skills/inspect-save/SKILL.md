---
name: inspect-save
description: >-
  Load a Pokémon Crystal .sav, .srm, or .state and print badges, party moves,
  whether Cut (0x0F) is known, the current map, and wOptions. Use when the user
  asks to inspect, check, or load a save, a snapshot, or a cartridge file.
---

# Inspect a Crystal save

Run this from the repo root. Do not register the file and do not rewrite options.

```bash
.venv/bin/python -m tools.inspect_sav PATH_OR_NAME
```

`PATH_OR_NAME` is a `.sav`, a `.srm`, a `.state`, or a name already in `states/manifest.json`.

The command prints two lines. Repeat them to the user unchanged:

```text
badges=N map=(group,number) party=[(species, (move, move, move, move)), ...] cut=yes|NO
wOptions 0xNN options text=... style=... scene=...
```

`cut=NO` means no party move is `0x0F`. Say that this save cannot Cut the Ilex tree.

If the command exits 1, repeat its error. Offer `--window` only when it says the game never reached the overworld.

A cartridge file longer than 32768 bytes is truncated to that. That is expected for an RTC footer.
