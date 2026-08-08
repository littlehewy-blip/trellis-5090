#!/bin/bash
# Container entrypoint. Runs the DEFERRED GPU checks at first-run (the GPU-free
# `docker build` skipped them), then serves pod_worker.py on :8000.
set -e
export HF_TOKEN=$(cat /workspace/.hf_token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TRELLIS_DIR=/workspace/TRELLIS.2
export WEIGHTS_DIR=/workspace/weights/TRELLIS.2-4B
export WORKER_PORT=8000
export PYTHONPATH=/workspace/TRELLIS.2
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace

# --- deferred GPU gate (moved out of the build) ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL [first-run]: nvidia-smi not found — container must run on an NVIDIA GPU host (--gpus all)." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# --- deferred REAL ops probe (money-safety gate before weights + serve) ---
echo "[first-run] running deferred CUDA ops probe ..."
python3 /workspace/ops_probe.py || {
  echo "FATAL [first-run]: OPS_PROBE_FAIL — CUDA ops did not import/forward on this GPU. Not serving." >&2
  exit 1
}

# --- weights: download on first run if the mounted volume is empty ---
if [[ ! -f "$WEIGHTS_DIR/.download_complete" ]]; then
  echo "[first-run] weights absent — downloading microsoft/TRELLIS.2-4B ..."
  WEIGHTS_DIR="$WEIGHTS_DIR" python3 - <<'PY'
from huggingface_hub import snapshot_download
import os
out = os.environ.get("WEIGHTS_DIR", "/workspace/weights/TRELLIS.2-4B")
snapshot_download(repo_id="microsoft/TRELLIS.2-4B", local_dir=out, local_dir_use_symlinks=False)
open(os.path.join(out, ".download_complete"), "w").write("ok\n")
print("weights ready at", out)
PY
else
  echo "[first-run] weights present at $WEIGHTS_DIR"
fi

echo "[first-run] OPS_PROBE_OK — starting worker on :$WORKER_PORT"
exec python3 pod_worker.py
