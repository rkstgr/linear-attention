#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_wandb_agent_cpu.sh <entity/project/sweep_id> [wandb agent args...]"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export REQUIRE_JAX_GPU="${REQUIRE_JAX_GPU:-0}"

uv sync --group experiment
exec uv run --no-sync wandb agent "$@"
