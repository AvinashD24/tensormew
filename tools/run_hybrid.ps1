# Play start -> egg -> Elm -> Falkner in a window and record it to recordings\<name>\.
#   tools\run_hybrid.ps1 [name]
# Env: STATE (default new_bark_outside, or a path to a .state file), SPEED (default 4), PYTHON (default python).
param([string]$Name = ("run_" + (Get-Date -Format "yyyyMMdd_HHmmss")))
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$state = if ($env:STATE) { $env:STATE } else { "new_bark_outside" }
$speed = if ($env:SPEED) { $env:SPEED } else { "4" }
$env:PYTHONIOENCODING = "utf-8"
& $py -m tools.play_live --state $state `
  --policy models/deep_policy_baseline_351cb55.pt `
  --post-egg-policy models/deep_policy.pt `
  --speed $speed `
  --record-dir "recordings/$Name" `
  --tag $Name
