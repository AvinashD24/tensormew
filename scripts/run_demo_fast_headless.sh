#!/usr/bin/env bash
# ==============================================================================
# TensorMew Live Demo: Uncapped Headless Evaluation / Benchmark
# ==============================================================================
# Executes headless at max emulation speed (uncapped speed 0) without opening
# an SDL window. Ideal for quick verification, testing, and continuous benchmarks.
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

echo "==============================================================================="
echo "⚡ Launching TensorMew Fast Headless Evaluation"
echo "==============================================================================="
echo "State:  ${STATE}"
echo "Speed:  Uncapped (Speed 0, Headless)"
echo "Python: ${PYTHON}"
echo "==============================================================================="

export PYTHONUNBUFFERED=1
export PYTHONPATH="."

exec "$PYTHON" -m tools.play_live \
    --policy logs/deep_policy.pt \
    --state "$STATE" \
    --speed 0 \
    --headless \
    "$@"
