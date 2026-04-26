#!/usr/bin/env bash
# Evaluate a trained SAC policy on RacingEnv.
#
# Checkpoint: default is training.sac_policy_checkpoint in the YAML
# (e.g. deep_dynamics/rl_runs/checkpoints/sac_latest.pt). Override with --checkpoint.
#
# New policies (27-dim obs with current default.yaml): omit --legacy-policy.
# Old centerline-only checkpoints (20-dim): add --legacy-policy.
#
# Examples:
#   ./eval_sac.sh
#   ./eval_sac.sh --episodes 50 -v
#   ./eval_sac.sh --checkpoint deep_dynamics/rl_runs/checkpoints/sac_latest.pt

set -euo pipefail
cd "$(dirname "$0")"
exec python -m deep_dynamics.rl.scripts.evaluate \
  --config deep_dynamics/rl/configs/default.yaml \
  "$@"
