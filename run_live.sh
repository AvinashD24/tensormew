#!/usr/bin/env bash
exec /opt/homebrew/Caskroom/miniconda/base/bin/python -m tools.play_live --policy logs/deep_policy.pt --speed 4 "$@"
