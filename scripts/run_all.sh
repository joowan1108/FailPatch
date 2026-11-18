#!/usr/bin/env bash
# Full FailPatch experiment for each model (all tasks trained jointly, as in the paper):
#   1) base evaluation            -> <model>/<task>/eval_base.json
#   2) deployment buffer          -> <model>/<task>/buffer/        (base-policy rollouts)
#   3) offline annotation         -> <model>/<task>/labels.json
#   4) FailPatch training (SFT)   -> <model>/reb_<variant>/reb.pt, train_summary.json
#   5) FailPatch evaluation       -> <model>/<task>/eval_failpatch_$VARIANT.json
# Finished stages are skipped, so the script can be re-run after an interruption.
#
# Two GPUs: run one model per GPU in two shells, e.g.
#   GPU=0 PORT=10086 MODELS=xr1  bash scripts/run_all.sh
#   GPU=1 PORT=10087 MODELS=rldx bash scripts/run_all.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../setup/env.sh"
cd "$PROJECT"

MODELS="${MODELS:-xr1 rldx}"
GPU="${GPU:-0}"
SPLIT="${SPLIT:-pretrain}"
N_ENVS="${N_ENVS:-4}"
SEED="${SEED:-7}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
N_SUCCESS="${N_SUCCESS:-50}"
N_FAIL="${N_FAIL:-50}"
MAX_EPISODES="${MAX_EPISODES:-300}"
NUM_DEMOS="${NUM_DEMOS:-50}"
DEMO_SAMPLES="${DEMO_SAMPLES:-1000}"
# Demonstration samples and their cached frozen features. The defaults are the first runs (1,000
# anchor steps per task); DEMO_DIR=demos_5k / FEATURE_DIR=features_5k select the larger set that
# keeps the experts from memorising a small fixed set of rows (see README).
DEMO_DIR="${DEMO_DIR:-demos}"
FEATURE_DIR="${FEATURE_DIR:-features}"
# auto | openai | anthropic | heuristic | external (annotate on another machine so that the API key
# never touches this instance; see scripts/annotate_local.sh)
ANNOTATOR="${ANNOTATOR:-auto}"
PORT="${PORT:-10086}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
# VARIANT selects the training objective; each variant gets its own <model>/reb_<variant> directory
# and eval_failpatch_<variant>.json, so several can be compared on the same buffers:
#   paper    hyper-parameters exactly as in the paper (table 5)
#   adapted_sum  the paper's objective (summed action loss, Redirect push only on states where some
#            expert clears the Null threshold) with Redirect margins = half the failure/success gap
#            measured per task on the first 5 chunks after the failure onset, and the failure routing
#            temperature scaled with it (T = 0.2 x margin / 0.1). Redirect margin weight 0.01 as in the paper. The archived old-buffer adapted_sum results used one margin for
#            all tasks, measured on every H- chunk, and pushed on every H- state (--redirect-bootstrap)
#   adapted  Redirect margins relative to the measured failure/success gap, experts pushed towards
#            the success prototype from the start, and a stronger push (see README)
VARIANT="${VARIANT:-adapted}"
case "$VARIANT" in
    paper)       VARIANT_ARGS="--action-loss sum" ;;
    adapted)     VARIANT_ARGS="--action-loss mean --margins relative --redirect-bootstrap --redirect-margin-weight 1.0" ;;
    adapted_sum) VARIANT_ARGS="--action-loss sum --margins task --margin-onset-chunks 5 --fail-temperature-scaling margin" ;;
    # same objective as adapted_sum, trained on the larger demonstration set (run with
    # DEMO_DIR=demos_5k FEATURE_DIR=features_5k); a separate name keeps the earlier results
    adapted_sum_d5k) VARIANT_ARGS="--action-loss sum --margins task --margin-onset-chunks 5 --fail-temperature-scaling margin" ;;
    # adapted_sum_d5k with responsibilities from the loss averaged over the draws of each scene
    adapted_sum_d5k_scene) VARIANT_ARGS="--action-loss sum --margins task --margin-onset-chunks 5 --fail-temperature-scaling margin --demo-target scene" ;;
    # ... plus prototype retrieval and Redirect distances measured after removing the shared mean direction
    adapted_sum_d5k_center) VARIANT_ARGS="--action-loss sum --margins task --margin-onset-chunks 5 --fail-temperature-scaling margin --demo-target scene --center-features task" ;;
    *) echo "unknown VARIANT $VARIANT" >&2; exit 1 ;;
esac

# API key for the offline annotator: keep it out of templates and shell history by putting
#   export ANTHROPIC_API_KEY=...   (or OPENAI_API_KEY=...)
# into $WORK/.api_env (chmod 600). Without a key the heuristic annotator is used.
if [[ -f $WORK/.api_env ]]; then source "$WORK/.api_env"; fi

export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" NO_ALBUMENTATIONS_UPDATE=1
PY="$VENVS/sim/bin/python"
SERVER_PID=""; MON_PID=""; MODEL_T0=0; model=""

stop_server() {
    if [[ -n $SERVER_PID ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
finish_model() {
    stop_server
    if [[ -n $MON_PID ]]; then
        printf '%s\t%s\n' "$(date -Is)" "$((SECONDS - MODEL_T0))" >> "$RESULTS/$model/model_wall_time.tsv"
        kill "$MON_PID" 2>/dev/null || true
        wait "$MON_PID" 2>/dev/null || true
    fi
    MON_PID=""
    return 0
}
trap finish_model EXIT

start_server() {   # start_server [reb.pt]; waits for the "ready:" line
    local log="$RESULTS/$model/server.log" t0=$SECONDS
    bash scripts/run_server.sh "$model" "$PORT" "${1:-}" >"$log" 2>&1 &
    SERVER_PID=$!
    until grep -q "ready:" "$log" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[x] $model server exited while loading; see $log" >&2; tail -20 "$log" >&2; exit 1
        fi
        (( SECONDS - t0 > 3600 )) && { echo "[x] $model server not ready after 1 h" >&2; exit 1; }
        sleep 2
    done
    printf '%s\t%s\n' "server" "$((SECONDS - t0))" >> "$RESULTS/$model/model_load_time.tsv"
    echo "[run] $model server ready after $((SECONDS - t0)) s${1:+ (with REB)}"
}

# stage <dir> <name> <command...>: run, log, and append "<name>\t<seconds>" to <dir>/stage_times.tsv.
# Never call it as the condition of if/||/&& (bash would then ignore errors inside it).
stage() {
    local dir=$1 name=$2 t0=$SECONDS
    shift 2
    "$@" 2>&1 | tee -a "$dir/log.txt"
    printf '%s\t%s\n' "$name" "$((SECONDS - t0))" >> "$dir/stage_times.tsv"
}

mkdir -p "$RESULTS/$DEMO_DIR"
nvidia-smi -i "$GPU" --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$RESULTS/gpu.txt"
echo "[run] GPU $GPU: $(cat "$RESULTS/gpu.txt")"
wait_for_download assets
wait_for_download demos
for task in $TASKS; do
    if [[ ! -f $RESULTS/$DEMO_DIR/demos_$task.npz ]]; then
        stage "$RESULTS/$DEMO_DIR" "prepare_$task" "$PY" -u -m failpatch.prepare_demos --task "$task" --split "$SPLIT" \
            --out "$RESULTS/$DEMO_DIR" --num-demos "$NUM_DEMOS" --samples "$DEMO_SAMPLES"
    fi
done

for model in $MODELS; do
    wait_for_download "$model"
    mkdir -p "$RESULTS/$model/reb_$VARIANT"
    MODEL_T0=$SECONDS
    nvidia-smi -i "$GPU" --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 \
        >> "$RESULTS/$model/gpu_usage.csv" &
    MON_PID=$!
    source scripts/model_env.sh "$model"

    # 1-3: base evaluation, deployment buffer, annotation
    need_base=0
    pending_labels=0
    for task in $TASKS; do
        [[ -f $RESULTS/$model/$task/eval_base.json && -f $RESULTS/$model/$task/buffer_meta.json ]] || need_base=1
    done
    (( need_base )) && start_server
    for task in $TASKS; do
        out="$RESULTS/$model/$task"; mkdir -p "$out"
        common=(--model "$model" --task "$task" --port "$PORT" --split "$SPLIT" --n-envs "$N_ENVS" --seed "$SEED" --out-dir "$out")
        if [[ ! -f $out/eval_base.json ]]; then
            stage "$out" base_eval "$PY" -u -m failpatch.eval "${common[@]}" --episodes "$EVAL_EPISODES"
        fi
        if [[ ! -f $out/buffer_meta.json ]]; then
            stage "$out" collect "$PY" -u -m failpatch.collect "${common[@]}" \
                --n-success "$N_SUCCESS" --n-fail "$N_FAIL" --max-episodes "$MAX_EPISODES"
        fi
        if [[ ! -f $out/annotate.done ]]; then
            if [[ $ANNOTATOR == external ]]; then
                pending_labels=1   # labels are produced elsewhere (scripts/annotate_local.sh)
            else
                stage "$out" annotate "$PY" -u -m failpatch.annotate --out-dir "$out" --annotator "$ANNOTATOR"
                touch "$out/annotate.done"
            fi
        fi
    done
    stop_server   # free the GPU memory for training
    if (( pending_labels )); then
        echo "[run] $model: deployment buffers are ready; waiting for labels."
        echo "      On the machine that holds the API key run scripts/annotate_local.sh, then re-run this script."
        finish_model
        continue
    fi

    # 4: FailPatch training (all tasks jointly)
    if [[ ! -f $RESULTS/$model/reb_$VARIANT/reb.pt ]]; then
        # shellcheck disable=SC2086
        stage "$RESULTS/$model/reb_$VARIANT" train "$MODEL_PY" -u -m failpatch.train --model "$model" "${MODEL_ARGS[@]}" \
            --tasks $TASKS --results "$RESULTS/$model" --demos "$RESULTS/$DEMO_DIR" --out "$RESULTS/$model/reb_$VARIANT" \
            --feature-cache "$RESULTS/$model/$FEATURE_DIR" $VARIANT_ARGS $EXTRA_TRAIN_ARGS
    fi

    # 5: evaluation with the REB attached
    need_eval=0
    for task in $TASKS; do [[ -f $RESULTS/$model/$task/eval_failpatch_$VARIANT.json ]] || need_eval=1; done
    (( need_eval )) && start_server "$RESULTS/$model/reb_$VARIANT/reb.pt"
    for task in $TASKS; do
        out="$RESULTS/$model/$task"
        common=(--model "$model" --task "$task" --port "$PORT" --split "$SPLIT" --n-envs "$N_ENVS" --seed "$SEED" --out-dir "$out")
        if [[ ! -f $out/eval_failpatch_$VARIANT.json ]]; then
            stage "$out" "failpatch_eval_$VARIANT" "$PY" -u -m failpatch.eval "${common[@]}" --episodes "$EVAL_EPISODES" \
                --tag "failpatch_$VARIANT"
        fi
    done
    finish_model
done

if ls "$RESULTS"/*/reb_*/train_summary.json >/dev/null 2>&1; then
    "$PY" scripts/summarize.py "$RESULTS"
fi
