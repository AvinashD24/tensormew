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

# Parse arguments: supports 4 stages mode, 4 distinct states, or single state
if [[ "${1:-stages}" == "stages" || "${1:-stages}" == "story" ]]; then
    STATE1="mystery_egg_received"
    STATE2="post_rival_cherrygrove"
    STATE3="mystery_egg_delivered"
    STATE4="agent_map_10_5"
    SPEED="${2:-3}"
    MODE="4 Progressive Story Stages (Egg -> Rival -> Elm -> Falkner)"
elif [[ "$#" -ge 4 ]]; then
    STATE1="$1"
    STATE2="$2"
    STATE3="$3"
    STATE4="$4"
    SPEED="${5:-3}"
    MODE="4 Custom States"
else
    STATE1="$1"
    STATE2="$1"
    STATE3="$1"
    STATE4="$1"
    SPEED="${2:-3}"
    MODE="Single State (${1}) with 4 Divergent Policies"
fi

echo "==============================================================================="
echo "🪟 Starting 4 Tiled PyBoy Demo Instances"
echo "==============================================================================="
echo "Mode:   ${MODE}"
echo "Speed:  ${SPEED}x"
echo "Python: ${PYTHON}"
echo "  [1] Top-Left:     ${STATE1}"
echo "  [2] Top-Right:    ${STATE2}"
echo "  [3] Bottom-Left:  ${STATE3}"
echo "  [4] Bottom-Right: ${STATE4}"
echo "==============================================================================="

# Terminate any previously running play_live instances to avoid window overlap
pkill -f "tools.play_live" 2>/dev/null || true
sleep 1

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="."

# Instance 1: Top-Left (x=50, y=50)
SDL_VIDEO_WINDOW_POS="50,50" "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE1}" --temperature 0.3 --tag "inst1" > logs/inst1.log 2>&1 &
PID1=$!

# Instance 2: Top-Right (x=550, y=50)
SDL_VIDEO_WINDOW_POS="550,50" "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE2}" --temperature 0.5 --tag "inst2" > logs/inst2.log 2>&1 &
PID2=$!

# Instance 3: Bottom-Left (x=50, y=520)
SDL_VIDEO_WINDOW_POS="50,520" "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE3}" --temperature 0.7 --tag "inst3" > logs/inst3.log 2>&1 &
PID3=$!

# Instance 4: Bottom-Right (x=550, y=520)
SDL_VIDEO_WINDOW_POS="550,520" "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt --speed "${SPEED}" --state "${STATE4}" --temperature 1.0 --tag "inst4" > logs/inst4.log 2>&1 &
PID4=$!

echo "Successfully started 4 live PyBoy instances:"
echo "  [Instance 1] PID $PID1 (Top-Left)     [${STATE1}] -> logs/inst1.log"
echo "  [Instance 2] PID $PID2 (Top-Right)    [${STATE2}] -> logs/inst2.log"
echo "  [Instance 3] PID $PID3 (Bottom-Left)  [${STATE3}] -> logs/inst3.log"
echo "  [Instance 4] PID $PID4 (Bottom-Right) [${STATE4}] -> logs/inst4.log"
echo ""
echo "To tail logs: tail -f logs/inst1.log"
echo "To stop all:  pkill -f 'tools.play_live'"
