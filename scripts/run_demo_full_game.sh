#!/usr/bin/env bash
# ==============================================================================
# TensorMew Live Demo: Full End-to-End Story Run
# ==============================================================================
# Runs autonomously from the very start (New Bark Town / Route 29) through
# Cherrygrove City, Mr. Pokémon's House (Egg), Rival Battle, Elm's Lab,
# Route 30/31, Violet City, and Falkner's Gym.
# ==============================================================================

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Select Python binary
PYTHON="/opt/homebrew/Caskroom/miniconda/base/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

SPEED="${1:-3}"
STATE="${2:-route29_entry}"

echo "==============================================================================="
echo "🌟 Launching TensorMew End-to-End Story Demo"
echo "==============================================================================="
echo "State:  ${STATE}"
echo "Speed:  ${SPEED}x (Visible GUI)"
echo "Python: ${PYTHON}"
echo "==============================================================================="

export PYTHONUNBUFFERED=1
export PYTHONPATH="."

exec "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt \
    --state "$STATE" \
    --speed "$SPEED" \
    "$@"
