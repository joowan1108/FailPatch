# source scripts/model_env.sh xr1|rldx  ->  MODEL_PY (python of that model's env), MODEL_ARGS
case "$1" in
    xr1)  MODEL_PY="$VENVS/xr1/bin/python"
          MODEL_ARGS=(--model-path "$XR1_MODEL" --xr1-repo "$SRC/Xiaomi-Robotics-1")
          # XR1_ACTION_EXPERT=<file>: serve / extract features with a fine-tuned action expert (failpatch.finetune)
          if [ -n "${XR1_ACTION_EXPERT:-}" ]; then MODEL_ARGS+=(--action-expert "$XR1_ACTION_EXPERT"); fi ;;
    rldx) MODEL_PY="$SRC/RLDX-1/.venv/bin/python"
          MODEL_ARGS=(--model-path "$RLDX_MODEL") ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
esac
