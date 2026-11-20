#!/usr/bin/env bash
# Wait on bw2 (CPU only) until START (KST) and then until our GPU has had no compute process for
# FREE_MIN consecutive minutes; then run scripts/bw2_live.sh on it. Run inside tmux:
#   START="2026-11-21 06:30" FREE_MIN=20 bash scripts/bw2_autostart.sh [out-name]
# Log: $RESULTS/autostart.log. Nothing touches the GPU before the free window has been observed.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/bw2_env.sh
START="${START:?START=\"YYYY-MM-DD HH:MM\" (KST)}"
FREE_MIN="${FREE_MIN:-20}"
mkdir -p "$RESULTS"
LOG="$RESULTS/autostart.log"
say() { echo "[autostart $(TZ=Asia/Seoul date '+%F %T')] $*" | tee -a "$LOG"; }
start_s=$(TZ=Asia/Seoul date -d "$START" +%s) || exit 1
say "waiting until $START KST, then for $FREE_MIN free minutes on GPU $GPU"
while (( $(date +%s) < start_s )); do sleep 60; done
free=0
while (( free < FREE_MIN )); do
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if gpu_free && [[ -n $used ]] && (( used < 1000 )); then
        free=$((free + 1))
    else
        (( free > 0 )) && say "GPU $GPU busy again after $free free minutes (memory used ${used:-?} MiB)"
        free=0
    fi
    (( free > 0 && free % 5 == 0 )) && say "GPU $GPU free for $free minutes"
    (( free < FREE_MIN )) && sleep 60
done
say "GPU $GPU free for $FREE_MIN minutes: starting scripts/bw2_live.sh"
bash scripts/bw2_live.sh "$@" 2>&1 | tee -a "$LOG"
say "bw2_live.sh finished"
