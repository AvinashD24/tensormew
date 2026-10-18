#!/usr/bin/env bash
# Play start -> egg -> Elm -> Falkner in a window and record it to recordings/<name>/.
#   tools/run_hybrid.sh [name]
# Env: STATE (default new_bark_outside, or a path to a .state file), SPEED (default 4), PYTHON (default python).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
exec "${PYTHON:-python}" -m tools.play_live \
  --state "${STATE:-new_bark_outside}" \
  --policy models/deep_policy_baseline_351cb55.pt \
  --post-egg-policy models/deep_policy.pt \
  --speed "${SPEED:-4}" \
  --record-dir "recordings/${NAME}" \
  --tag "${NAME}"
