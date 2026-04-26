#!/usr/bin/env bash
# Localhost browser demo for presentation.
#
# Usage:
#   ./serve_rl_demo.sh
#   ./serve_rl_demo.sh --speed 3 --port 8080

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_PY="/Users/ethan/Documents/Anaconda/anaconda3/envs/edgar_project/bin/python"
if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1 && [[ -x "$CONDA_PY" ]]; then
  PYTHON_BIN="$CONDA_PY"
fi

exec "$PYTHON_BIN" -m deep_dynamics.rl.scripts.drive_server \
  --config deep_dynamics/rl/configs/default.yaml \
  --checkpoint deep_dynamics/rl_runs/final_models/final_model_v3.pt \
  --speed 2 \
  --port 8000 \
  --open \
  "$@"
