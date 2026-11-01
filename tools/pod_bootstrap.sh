#!/usr/bin/env bash
# Manual dependency setup for a fresh RunPod GPU pod (the stock PyTorch image lacks pyboy, and Ubuntu 24.04
# blocks plain pip). Run on the pod:  bash tools/pod_bootstrap.sh
set -u

echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
echo "Cores: $(nproc)  (keep --envs <= this)"
echo "Disk:  $(df -h /workspace 2>/dev/null | tail -1 | awk '{print $4" free"}')"

missing=0
for mod in torch numpy PIL; do
  python3 -c "import $mod" 2>/dev/null || { echo "MISSING: $mod (use a PyTorch image)"; missing=1; }
done

if ! python3 -c "import pyboy" 2>/dev/null; then
  echo "Installing pyboy==2.8.1 ..."
  pip install --break-system-packages "pyboy==2.8.1" || { echo "pyboy install failed"; exit 1; }
fi
python3 -c "import pyboy; print('pyboy ok')" || exit 1

command -v tmux >/dev/null 2>&1 || echo "note: tmux not found; --detach falls back to setsid nohup"
exit $missing
