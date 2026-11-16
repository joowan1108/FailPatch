#!/usr/bin/env bash
# Annotate the deployment buffers on THIS machine, so the API key never leaves it.
#
#   scripts/annotate_local.sh "<ssh options and host>" <remote results dir> [model ...]
#   e.g. scripts/annotate_local.sh "-p 49526 root@38.117.87.49" /workspace/failpatch_results xr1
#
# Downloads <results>/<model>/<task>/{buffer,buffer_meta.json}, runs failpatch.annotate locally with
# the key from $FAILPATCH_API_ENV (default ~/.failpatch_api_env), uploads labels.json and marks the
# task as annotated. Re-run scripts/run_all.sh on the instance afterwards.
# PYTHON must point to a Python with numpy and opencv (default: python3).
set -euo pipefail
SSH_TARGET="${1:?usage: annotate_local.sh \"<ssh options and host>\" <remote results dir> [model ...]}"
REMOTE="${2:?remote results dir}"
shift 2
MODELS=("${@:-xr1 rldx}")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL="${LOCAL_BUFFER_DIR:-$HOME/failpatch_buffers}"
PYTHON="${PYTHON:-python3}"
API_ENV="${FAILPATCH_API_ENV:-$HOME/.failpatch_api_env}"
[[ -f $API_ENV ]] || { echo "[x] $API_ENV not found" >&2; exit 1; }
# shellcheck disable=SC2086
ssh_() { ssh -o BatchMode=yes $SSH_TARGET "$@"; }

# shellcheck disable=SC2068
for model in ${MODELS[@]}; do
    for task in $(ssh_ "ls -d $REMOTE/$model/*/buffer 2>/dev/null | xargs -n1 dirname | xargs -n1 basename"); do
        if ssh_ "test -f $REMOTE/$model/$task/annotate.done"; then
            echo "[skip] $model/$task already annotated"; continue
        fi
        dest="$LOCAL/$model/$task"; mkdir -p "$dest"
        echo "[get] $model/$task"
        ssh_ "tar cf - -C $REMOTE/$model/$task buffer buffer_meta.json" | tar xf - -C "$dest"
        ( set +x; source "$API_ENV"; cd "$HERE"; PYTHONPATH="$HERE" "$PYTHON" -u -m failpatch.annotate \
            --out-dir "$dest" --annotator "${ANNOTATOR:-auto}" )
        ssh_ "cat > $REMOTE/$model/$task/labels.json && touch $REMOTE/$model/$task/annotate.done" < "$dest/labels.json"
        echo "[put] $model/$task labels uploaded"
    done
done
