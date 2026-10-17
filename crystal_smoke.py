"""
Crystal hour-0 smoke test: PyBoy boots, .sym resolves, banked RAM reads work, stepping works.

  python crystal_smoke.py                            # boot + RAM read (pre-game values = junk, that's fine)
  python crystal_smoke.py --make-state states/test.state   # play manually, close window -> saves state
  python crystal_smoke.py --state states/test.state  # load state, read RAM, walk 5 steps
"""
import argparse
import os
import sys

from pyboy import PyBoy

ROM = "rom/pokemon_crystal.gbc"
SYM = "rom/pokemon_crystal.sym"

SYMBOLS = [
    # position
    "wMapGroup", "wMapNumber", "wXCoord", "wYCoord", "wFacingDirection",
    # adjacent collision (what the greedy mover uses)
    "wTileUp", "wTileDown", "wTileLeft", "wTileRight", "wPlayerTileCollision",
    # gates for the stuck detector
    "wBattleMode", "wScriptRunning", "wTextboxFlags", "wMapStatus",
    # progress
    "wPartyCount", "wJohtoBadges",
]


def resolve(pyboy):
    addrs, missing = {}, []
    for s in SYMBOLS:
        try:
            addrs[s] = pyboy.symbol_lookup(s)  # (bank, addr)
        except Exception:
            missing.append(s)
    return addrs, missing


def read_byte(pyboy, bank, a):
    # 0xD000-0xDFFF is the switchable WRAM bank on GBC -> read the bank explicitly
    if 0xD000 <= a < 0xE000:
        try:
            return pyboy.memory[bank, a]
        except Exception:
            pass
    return pyboy.memory[a]


def read_state(pyboy, addrs):
    out = {k: read_byte(pyboy, b, a) for k, (b, a) in addrs.items()}
    if "wJohtoBadges" in out:
        out["wJohtoBadges"] = bin(out["wJohtoBadges"]).count("1")
    return out


def pos(s):
    return (s.get("wMapGroup"), s.get("wMapNumber"), s.get("wXCoord"), s.get("wYCoord"))


def step(pyboy, addrs, direction, timeout=40):
    before = pos(read_state(pyboy, addrs))
    pyboy.button_press(direction)
    moved = False
    for _ in range(timeout):
        pyboy.tick()
        if pos(read_state(pyboy, addrs)) != before:
            moved = True
            break
    pyboy.button_release(direction)
    pyboy.tick(10)
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--make-state", help="play manually, close window to save state here")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    for p in (ROM, SYM):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    os.makedirs("states", exist_ok=True)

    pyboy = PyBoy(ROM, symbols=SYM, window="null" if args.headless else "SDL2")
    addrs, missing = resolve(pyboy)
    print(f"resolved {len(addrs)}/{len(SYMBOLS)} symbols")
    if missing:
        print("  MISSING:", missing)

    if args.make_state:
        print("Play in the window (arrows, A=a, B=s, Start=Enter). Close the window to save.")
        while pyboy.tick():
            pass
        with open(args.make_state, "wb") as f:
            pyboy.save_state(f)
        print(f"saved {args.make_state}")
        print("RAM at save:", read_state(pyboy, addrs))
        pyboy.stop(save=False)
        return

    if args.state:
        with open(args.state, "rb") as f:
            pyboy.load_state(f)
        pyboy.tick(30)
    else:
        pyboy.tick(900)
        print("no --state: values are pre-game junk, walking skipped")

    print("RAM:", read_state(pyboy, addrs))

    if args.state:
        print("\nwalking test:")
        for d in ["down", "down", "right", "up", "left"]:
            moved = step(pyboy, addrs, d)
            s = read_state(pyboy, addrs)
            adj = {k[5:]: hex(s[k]) for k in ("wTileUp", "wTileDown", "wTileLeft", "wTileRight") if k in s}
            print(f"  {d:5s} moved={moved} pos={pos(s)} adj={adj} script={s.get('wScriptRunning')}")
        with open("states/smoke_after.state", "wb") as f:
            pyboy.save_state(f)
        print("save/load state OK")

    pyboy.stop(save=False)
    print("\nSMOKE TEST DONE")


if __name__ == "__main__":
    main()