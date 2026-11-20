#!/usr/bin/env bash
# FailPatch with the paper's live demonstration branch on bw2 (GPU 5 only):
#   bash scripts/bw2_live.sh [out-name]          (run inside tmux; logs to $RESULTS/xr1_fixed_ft/<out>/log.txt)
# Fixed kitchen 11/34, CoffeeSetupMug, XR-1 with the fine-tuned action expert (2k). Configuration of the
# best cached-feature run (lambda_rd 1.0, noise floor q75, centering, class balance) with the demonstration
# branch live as in the paper: every step, 32 distinct demonstration samples with XR-1's colour
# augmentation and fresh (noise, time) draws, preprocessed by CPU workers and passed through the frozen
# VLA in one batched forward. Converge schedule (at most 100k steps; train_state.pt every 10k, resumable
# with RESUME=<path>). Then the 30k, 60k and final checkpoints are evaluated on the 100 final scenes.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/bw2_env.sh
source setup/env.sh >/dev/null
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" NO_ALBUMENTATIONS_UPDATE=1
OUT="${1:-reb_ftmg_center_q75_rd10_live_bw2}"
F=$RESULTS/xr1_fixed_ft; T=CoffeeSetupMug; D=$F/$OUT
PORT="${PORT:-10586}"   # not the default 10086: the server is shared
WORKERS="${WORKERS:-24}"
export XR1_ACTION_EXPERT=$F/finetune_nn/checkpoints/action_expert_step002000.pt
source scripts/model_env.sh xr1
for f in "$XR1_ACTION_EXPERT" "$RESULTS/demos_k11s34_nn/demos_$T.npz" "$F/$T/labels.json" "$RESULTS/xr1_fixed/layout_style.txt"; do
    [[ -f $f ]] || { echo "[bw2] missing $f"; exit 1; }
done
gpu_free || { echo "[bw2] GPU $GPU is in use by another process:"; nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv; exit 1; }
read L S < "$RESULTS/xr1_fixed/layout_style.txt"
mkdir -p "$D"
echo "[bw2] start $(date -Is) GPU $GPU, out $D, workers $WORKERS${RESUME:+, resume $RESUME}"
"$MODEL_PY" -u -m failpatch.train --model xr1 "${MODEL_ARGS[@]}" --tasks $T --results "$F" --demos "$RESULTS/demos_k11s34_nn" \
    --draws 16 --action-loss sum --energy-scale auto --margins noise-floor --noise-floor-quantile 0.75 \
    --fail-temperature-scaling margin --center-features task --redirect-margin-weight 1.0 \
    --redirect-balance null-vs-expert --lambda-rd 1.0 --demo-live --live-workers "$WORKERS" --max-steps 100000 \
    ${RESUME:+--resume "$RESUME"} --out "$D" >> "$D/log.txt" 2>&1
grep -q "action expert loaded" "$D/log.txt" || { echo "[bw2] training did NOT use the fine-tuned action expert"; exit 1; }
[[ -f $D/reb.pt ]] || { echo "[bw2] training failed"; tail -5 "$D/log.txt"; exit 1; }
echo "[bw2] training finished $(date -Is)"

evaluate() {  # $1 checkpoint, $2 tag
    bash scripts/run_server.sh xr1 "$PORT" "$1" > "$F/server_$2.log" 2>&1 & local SP=$!
    until grep -q "ready:" "$F/server_$2.log"; do kill -0 $SP 2>/dev/null || { echo "[bw2] server died"; tail "$F/server_$2.log"; exit 1; }; sleep 2; done
    "$VENVS/sim/bin/python" -u -m failpatch.eval --model xr1 --task $T --port "$PORT" --split pretrain --n-envs 6 --seed 7 \
        --fixed-noise --layout "$L" --style "$S" --out-dir "$F/$T" --first-episode 0 --episodes 100 --tag "$2" 2>&1 | grep -E "^\[eval|Traceback|Error"
    kill $SP; wait $SP 2>/dev/null; sleep 10
    [[ -f $F/$T/eval_base_ft2k.json ]] || return 0
    python3 - "$F/$T" "$2" <<'PY'
import json, math, sys
d, tag = sys.argv[1], sys.argv[2]
B = {e["seed"]: e["success"] for e in json.load(open(f"{d}/eval_base_ft2k.json"))["episodes"]}
E = {e["seed"]: e for e in json.load(open(f"{d}/eval_{tag}.json"))["episodes"]}
fb = sum(E[k]["success"] and not B[k] for k in E); bf = sum(B[k] and not E[k]["success"] for k in E); n = fb + bf
p = min(1, 2 * sum(math.comb(n, i) for i in range(min(fb, bf) + 1)) / 2 ** n) if n else 1
pf = lambda ok: [E[k].get("patched_fraction", 0) for k in E if B[k] == ok]
print(f"[bw2] {tag}: ft base {sum(B.values())} | FailPatch {sum(e['success'] for e in E.values())}/100 | "
      f"fail->success {fb} success->fail {bf} p {p:.4f} | patched: base-failed {sum(pf(False)) / max(1, len(pf(False))):.3f} "
      f"base-succeeded {sum(pf(True)) / max(1, len(pf(True))):.3f}")
PY
}
evaluate "$D/checkpoints/reb_step030000.pt" "${OUT#reb_}_30k"
[[ -f $D/checkpoints/reb_step060000.pt ]] && evaluate "$D/checkpoints/reb_step060000.pt" "${OUT#reb_}_60k"
evaluate "$D/reb.pt" "${OUT#reb_}_final"
echo "[bw2] done $(date -Is)"
