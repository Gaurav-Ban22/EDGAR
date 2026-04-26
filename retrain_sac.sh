#!/usr/bin/env bash
# Retrain SAC on RacingEnv with your central YAML (formula_rl reward, raceline, etc.).
#
# Run from anywhere:
#   ./retrain_sac.sh
#   ./retrain_sac.sh --max-episodes 500
#
# Uses the same Python as your shell (activate conda first, e.g. conda activate edgar_project).

set -euo pipefail
cd "$(dirname "$0")"
exec python -u -m deep_dynamics.rl.scripts.train \
  --config deep_dynamics/rl/configs/default.yaml \
  "$@"
