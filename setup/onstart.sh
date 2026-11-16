# Vast.ai template "On-start Script": clone the code, then fetch assets, demo datasets and
# checkpoints in the background (progress: tail -f /workspace/failpatch_download.log).
cd /workspace
if [ ! -d FailPatch ]; then
  git clone "https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}github.com/joowan1108/FailPatch.git"
fi
cd FailPatch && git pull --ff-only || true
nohup bash setup/download.sh > /workspace/failpatch_download.log 2>&1 &
