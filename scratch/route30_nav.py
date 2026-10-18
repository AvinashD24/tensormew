from pathlib import Path
from collections import deque
from src.emulator import Emulator
from src.ram_parser import RamParser, dialogue_open
from src.mover import walkable, _DELTA
from src.macros import Macros
from src.battle import tactical_battle_step

emu = Emulator(headless=True, speed=0, rtc_lock=True)
emu.load_state(Path("states/mystery_egg_received.state").read_bytes())
reader = RamParser(emu)
macros = Macros(emu, reader)

st = reader.parse(emu.frame)
print("Start map:", st.map_id, "pos:", (st.x, st.y), "egg:", st.has_event("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"))

# Exit Mr. Pokemon house
for _ in range(4):
    emu.press("down", frames=8)
    emu.tick(16)
st = reader.parse(emu.frame)
print("After exit house map:", st.map_id, "pos:", (st.x, st.y))

# Route 30 to Cherrygrove: walk south to (10, 53)
# In Route 30, right-hand path stays at x >= 12 until y >= 40, then goes to (10, 53)
def navigate_step(target_x, target_y):
    global st
    # Priority order to target
    dx = target_x - st.x
    dy = target_y - st.y
    candidates = []
    if abs(dy) >= abs(dx):
        if dy > 0: candidates.append("down")
        elif dy < 0: candidates.append("up")
        if dx > 0: candidates.append("right")
        elif dx < 0: candidates.append("left")
    else:
        if dx > 0: candidates.append("right")
        elif dx < 0: candidates.append("left")
        if dy > 0: candidates.append("down")
        elif dy < 0: candidates.append("up")
    
    for d in ("down", "left", "right", "up"):
        if d not in candidates:
            candidates.append(d)
            
    for d in candidates:
        t = st.tiles.get(d, 0xFF)
        if walkable(t, d):
            # check constraint: if on Route 30 and y < 40, keep x >= 12
            target_nx = st.x + _DELTA[d][0]
            target_ny = st.y + _DELTA[d][1]
            if st.map_id == (26, 1) and target_ny < 40 and target_nx < 12:
                continue
            emu.press(d, frames=8)
            emu.tick(16)
            st = reader.parse(emu.frame)
            return True
    return False

# Step towards (15, 42) then (10, 53)
step = 0
while step < 500:
    st = reader.parse(emu.frame)
    if st.battle_mode != 0:
        tactical_battle_step(st, macros)
        continue
    if dialogue_open(st):
        macros.mash_text(max_frames=120)
        continue
    if st.map_id == (26, 3): # Cherrygrove City reached!
        print(f"Reached Cherrygrove City at step {step}! Pos: ({st.x}, {st.y})")
        break
    if st.y < 42:
        navigate_step(15, 42)
    else:
        navigate_step(10, 53)
    step += 1

print("Final map:", st.map_id, "pos:", (st.x, st.y))
