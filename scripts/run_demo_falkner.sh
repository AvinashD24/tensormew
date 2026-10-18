#!/usr/bin/env bash
# ==============================================================================
# TensorMew Live Demo: Falkner Gym Run
# ==============================================================================
# Runs from delivery of the Mystery Egg to Professor Elm, through Route 30/31
# tall grass training (learning Ember + catching Bellsprout), Violet City Kyle trade
# (receiving Rocky the Onix), Violet Poké Center healing, and defeating Gym Leader Falkner.
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
STATE="${2:-mystery_egg_received}"

echo "==============================================================================="
echo "🎮 Launching TensorMew Live Falkner Gym Demo"
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
