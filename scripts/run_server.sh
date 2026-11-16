#!/usr/bin/env bash
# Start the policy server for one model:  scripts/run_server.sh xr1|rldx [port] [reb.pt]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../setup/env.sh"
MODEL="${1:?usage: run_server.sh xr1|rldx [port] [reb.pt]}"
PORT="${2:-10086}"
REB="${3:-}"
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" NO_ALBUMENTATIONS_UPDATE=1
source scripts/model_env.sh "$MODEL"
exec "$MODEL_PY" -u -m failpatch.server --model "$MODEL" "${MODEL_ARGS[@]}" --port "$PORT" ${REB:+--reb "$REB"}
