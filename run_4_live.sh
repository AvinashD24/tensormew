#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

STATE="${1:-route29_entry}"
SPEED="${2:-0}"

echo "Starting 4 parallel PyBoy instances from '${STATE}' at speed ${SPEED}..."

# Kill any existing play_live instances to avoid conflicts
pkill -f "tools.play_live" 2>/dev/null || true
sleep 1

mkdir -p logs

# Instance 1: Top-Left (x=50, y=50)
SDL_VIDEO_WINDOW_POS="50,50" /opt/homebrew/Caskroom/miniconda/base/bin/python -m tools.play_live --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE}" --tag "inst1" > logs/inst1.log 2>&1 &
PID1=$!

# Instance 2: Top-Right (x=550, y=50)
SDL_VIDEO_WINDOW_POS="550,50" /opt/homebrew/Caskroom/miniconda/base/bin/python -m tools.play_live --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE}" --tag "inst2" > logs/inst2.log 2>&1 &
PID2=$!

# Instance 3: Bottom-Left (x=50, y=520)
SDL_VIDEO_WINDOW_POS="50,520" /opt/homebrew/Caskroom/miniconda/base/bin/python -m tools.play_live --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE}" --tag "inst3" > logs/inst3.log 2>&1 &
PID3=$!

# Instance 4: Bottom-Right (x=550, y=520)
SDL_VIDEO_WINDOW_POS="550,520" /opt/homebrew/Caskroom/miniconda/base/bin/python -m tools.play_live --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE}" --tag "inst4" > logs/inst4.log 2>&1 &
PID4=$!

echo "Successfully started 4 live PyBoy instances:"
echo "  [Instance 1] PID $PID1 (Top-Left)     -> logs/inst1.log"
echo "  [Instance 2] PID $PID2 (Top-Right)    -> logs/inst2.log"
echo "  [Instance 3] PID $PID3 (Bottom-Left)  -> logs/inst3.log"
echo "  [Instance 4] PID $PID4 (Bottom-Right) -> logs/inst4.log"
