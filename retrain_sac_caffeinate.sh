#!/usr/bin/env bash
# Retrain SAC while preventing macOS idle sleep.
#
# Usage:
#   ./retrain_sac_caffeinate.sh --max-episodes 3000
#
# Note: caffeinate cannot reliably keep a Mac running if the lid is closed
# unless you are in a supported clamshell/power setup.

set -euo pipefail
cd "$(dirname "$0")"

exec caffeinate -dimsu python -u -m deep_dynamics.rl.scripts.train \
  --config deep_dynamics/rl/configs/default.yaml \
  "$@"
