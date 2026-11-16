# source scripts/model_env.sh xr1|rldx  ->  MODEL_PY (python of that model's env), MODEL_ARGS
case "$1" in
    xr1)  MODEL_PY="$VENVS/xr1/bin/python"
          MODEL_ARGS=(--model-path "$XR1_MODEL" --xr1-repo "$SRC/Xiaomi-Robotics-1") ;;
    rldx) MODEL_PY="$SRC/RLDX-1/.venv/bin/python"
          MODEL_ARGS=(--model-path "$RLDX_MODEL") ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
esac
