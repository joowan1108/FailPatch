# Paths and GPU pinning on the shared B200 server (bw2, user guest1). Source BEFORE setup/env.sh:
#   source scripts/bw2_env.sh && source setup/env.sh
# Everything lives under ~/jwjw; the conda environments fp-xr1 / fp-sim are reached through
# $VENVS/xr1 and $VENVS/sim (symlinks created below), so the rest of the code is unchanged.
# Only GPU 5 is ours: every process sees it as its single device (cuda:0), and MuJoCo renders on it.
export JW="${JW:-$HOME/jwjw}"
export WORK="$JW" SRC="$JW/src" VENVS="$JW/venvs" CKPT="$JW/checkpoints" HF_HOME="$JW/hf_cache" RESULTS="$JW/results"
export GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
mkdir -p "$VENVS"
[[ -e $VENVS/xr1 ]] || ln -s "$JW/miniconda3/envs/fp-xr1" "$VENVS/xr1"
[[ -e $VENVS/sim ]] || ln -s "$JW/miniconda3/envs/fp-sim" "$VENVS/sim"

# gpu_free: true when no compute process runs on our GPU (checked before anything starts on it)
gpu_free() {
    [[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]
}
