#!/usr/bin/env bash
# [instance] RoboCasa assets (~10 GB), human demonstration datasets of the tasks (small),
# then the XR-1 (~10 GB) and RLDX-1 (~14 GB) checkpoints. Shares its markers with the DSRL
# repo, so anything already downloaded on this instance is skipped. Safe to re-run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$CKPT" "$HF_HOME" "$DOWNLOAD_STATE"
rm -f "$DOWNLOAD_STATE/failed"
echo $$ > "$DOWNLOAD_STATE/pid"
step=""
trap 'echo "$step" > "$DOWNLOAD_STATE/failed"; echo "[x] download of $step failed"' ERR

step=assets
if [[ ! -f $DOWNLOAD_STATE/assets.done ]]; then
    (yes y || true) | "$VENVS/sim/bin/python" -m robocasa.scripts.download_kitchen_assets
    touch "$DOWNLOAD_STATE/assets.done"
fi
echo "[ok] RoboCasa assets"

step=demos
if [[ ! -f $DOWNLOAD_STATE/demos.done ]]; then
    # shellcheck disable=SC2086
    (yes y || true) | "$VENVS/sim/bin/python" -m robocasa.scripts.download_datasets \
        --tasks $TASKS --split pretrain --source human
    touch "$DOWNLOAD_STATE/demos.done"
fi
echo "[ok] demonstration datasets"

step=xr1
if [[ ! -f $DOWNLOAD_STATE/xr1.done ]]; then
    "$VENVS/xr1/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365', local_dir='$XR1_MODEL')"
    touch "$DOWNLOAD_STATE/xr1.done"
fi
echo "[ok] XR-1 checkpoint"

step=rldx
if [[ ! -f $DOWNLOAD_STATE/rldx.done ]]; then
    "$SRC/RLDX-1/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('RLWRLD/RLDX-1-FT-RC365', local_dir='$RLDX_MODEL')"
    touch "$DOWNLOAD_STATE/rldx.done"
fi
echo "[ok] RLDX-1 checkpoint"
echo "[ok] downloads done"
rm -f "$DOWNLOAD_STATE/pid"
