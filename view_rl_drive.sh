#!/usr/bin/env bash
# Interactive map: SAC policy + random spawns; click "Restart" for a new start.
set -euo pipefail
cd "$(dirname "$0")"
exec python -m deep_dynamics.rl.scripts.drive_viewer \
  --config deep_dynamics/rl/configs/default.yaml \
  "$@"
