#!/usr/bin/env bash
# ==============================================================================
# TensorMew Live Demo: Paced Presentation & GIF Recording Mode
# ==============================================================================
# Runs at 2x emulation speed with battle text paused so human spectators can
# follow dialogue and combat, while simultaneously recording high-quality GIF
# frames and saving state checkpoints per story milestone to recordings/demo_run/.
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
REC_DIR="${2:-recordings/demo_run}"

echo "==============================================================================="
echo "🎥 Launching TensorMew Presentation & Recording Demo"
echo "==============================================================================="
echo "State:         ${STATE}"
echo "Speed:         2x (Paced text)"
echo "Recording Dir: ${REC_DIR}"
echo "Python:        ${PYTHON}"
echo "==============================================================================="

export PYTHONUNBUFFERED=1
export PYTHONPATH="."

mkdir -p "$REC_DIR"

exec "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt \
    --state "$STATE" \
    --speed 2 \
    --watch \
    --record-dir "$REC_DIR" \
    --record-every 6 \
    --record-frames-every 4 \
    "$@"
