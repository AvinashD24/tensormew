#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
exec /usr/bin/env PYTHONUNBUFFERED=1 /opt/homebrew/Caskroom/miniconda/base/bin/python -m tools.play_live --policy logs/deep_policy.pt "$@"

