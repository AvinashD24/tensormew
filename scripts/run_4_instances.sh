#!/usr/bin/env bash
# ==============================================================================
# TensorMew Live Demo: 4 Tiled PyBoy Windows
# ==============================================================================
# Launches 4 simultaneous PyBoy instances positioned neatly in a 2x2 grid
# across the display for multi-agent demonstrations and visual comparisons.
# ==============================================================================

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Select Python binary
PYTHON="/opt/homebrew/Caskroom/miniconda/base/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

STATE="${1:-mystery_egg_received}"
SPEED="${2:-3}"

echo "==============================================================================="
echo "🪟 Starting 4 Tiled PyBoy Demo Instances"
echo "==============================================================================="
echo "State:  ${STATE}"
echo "Speed:  ${SPEED}x"
echo "Python: ${PYTHON}"
echo "==============================================================================="

# Terminate any previously running play_live instances to avoid window overlap
pkill -f "tools.play_live" 2>/dev/null || true
sleep 1

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="."

# Instance 1: Top-Left (x=50, y=50)
SDL_VIDEO_WINDOW_POS="50,50" "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE}" --tag "inst1" > logs/inst1.log 2>&1 &
PID1=$!

# Instance 2: Top-Right (x=550, y=50)
SDL_VIDEO_WINDOW_POS="550,50" "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE}" --tag "inst2" > logs/inst2.log 2>&1 &
PID2=$!

# Instance 3: Bottom-Left (x=50, y=520)
SDL_VIDEO_WINDOW_POS="50,520" "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE}" --tag "inst3" > logs/inst3.log 2>&1 &
PID3=$!

# Instance 4: Bottom-Right (x=550, y=520)
SDL_VIDEO_WINDOW_POS="550,520" "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE}" --tag "inst4" > logs/inst4.log 2>&1 &
PID4=$!

echo "Successfully started 4 live PyBoy instances:"
echo "  [Instance 1] PID $PID1 (Top-Left)     -> logs/inst1.log"
echo "  [Instance 2] PID $PID2 (Top-Right)    -> logs/inst2.log"
echo "  [Instance 3] PID $PID3 (Bottom-Left)  -> logs/inst3.log"
echo "  [Instance 4] PID $PID4 (Bottom-Right) -> logs/inst4.log"
echo ""
echo "To tail logs: tail -f logs/inst1.log"
echo "To stop all:  pkill -f 'tools.play_live'"
