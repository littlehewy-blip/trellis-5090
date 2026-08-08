#!/usr/bin/env bash
# Install Microsoft TRELLIS.2 + TRELLIS.2-4B weights on a RunPod GPU pod.
# Invoked by submit_batch.py after the pod is RUNNING (or baked into dockerStartCmd).
# Idempotent: skips clone/download when already present.
#
# MONEY-SAFETY (ported from forge/install.sh patterns — do not remove):
#   1) Pin torch==2.6.0 + torchvision==0.21.0 + torchaudio==2.6.0 (cu124 triplet) —
#      unpinned torch = guaranteed fail against TRELLIS.2 custom CUDA ops on CUDA
#      12.4 devel images. Leaving the image's stock torchaudio (e.g. 2.4.0) after
#      upgrading torch causes undefined-symbol import failures in the TRELLIS chain.
#   2) REAL ops compile-and-import probe (flash_attn / o_voxel / flex_gemm / cumesh)
#      MUST pass BEFORE any multi-GB weights download. Broken env aborts at ~$0.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORK="${WORKSPACE:-/workspace}"
TRELLIS_DIR="${TRELLIS_DIR:-$WORK/TRELLIS.2}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$WORK/weights/TRELLIS.2-4B}"
WORKER_PORT="${WORKER_PORT:-8000}"
# TRELLIS.2 official pin (see microsoft/TRELLIS.2 setup.sh --new-env / forge install.sh)
# Triplet must stay aligned: torch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0 (cu124).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_PIN="${TORCH_PIN:-torch==2.7.1}"
TORCHVISION_PIN="${TORCHVISION_PIN:-torchvision==0.22.1}"
TORCHAUDIO_PIN="${TORCHAUDIO_PIN:-torchaudio==2.7.1}"
SKIP_WEIGHTS="${SKIP_WEIGHTS:-0}"
# BUILD_ONLY=1 => GPU-free `docker build`: compile/install the CUDA ops (nvcc
# cross-compiles for sm_120 via TORCH_CUDA_ARCH_LIST, no device needed) but SKIP
# every runtime GPU touch (nvidia-smi gate, torch.cuda device tensors, the ops
# forward-probe, weights, worker exec) — those defer to container first-run
# (start_worker_hf.sh + ops_probe.py). Default 0 => live-pod behavior UNCHANGED.
BUILD_ONLY="${BUILD_ONLY:-0}"
export BUILD_ONLY

mkdir -p "$WORK" "$WEIGHTS_DIR"

# Full provision transcript to $WORK/setup.log (the exec'd worker inherits these
# fds, so its stderr lands here too). During setup a throwaway http.server on
# $WORKER_PORT serves it (killed before the real worker binds) so the client
# can save it as pod_setup_remote.log even when setup aborts.
# Skipped under BUILD_ONLY: the tee process-substitution + backgrounded
# http.server can wedge a `docker build` RUN layer at exit — let build output
# flow straight to docker's own log instead.
if [[ "$BUILD_ONLY" != "1" ]]; then
  exec > >(tee -a "$WORK/setup.log") 2>&1
  ( cd "$WORK" && nohup python3 -m http.server "$WORKER_PORT" --bind 0.0.0.0 >/dev/null 2>&1 & echo $! > "$WORK/.logsrv.pid" )
fi

die() {
  echo "ABORT [pod_setup]: $*" >&2
  echo "ABORT [pod_setup]: fix the environment and re-run. Multi-GB weights were NOT downloaded." >&2
  # Grace window: keep the setup-log server reachable so the client can save
  # setup.log and tear the pod down itself (happens within seconds). Hard
  # exit after 10 min regardless — the pod never idles unbounded.
  sleep 600
  exit 1
}

echo "[pod_setup] host=$(hostname) starting money-safe TRELLIS.2 provision..."

# ---------------------------------------------------------------------------
# 1) CUDA gate
# ---------------------------------------------------------------------------
if [[ "$BUILD_ONLY" == "1" ]]; then
  echo "[pod_setup] BUILD_ONLY=1 — GPU-free build: skipping nvidia-smi gate (deferred to first-run). nvcc-only compile follows."
elif ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found — this script is for a Linux NVIDIA GPU pod (RunPod), not a CPU box."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
  GPU_VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' \r')"
  echo "[pod_setup] GPU: $GPU_NAME | VRAM_MiB: $GPU_VRAM"
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda; do
    if [[ -d "$c" ]]; then
      export CUDA_HOME="$c"
      export PATH="$CUDA_HOME/bin:${PATH:-}"
      echo "[pod_setup] CUDA_HOME=$CUDA_HOME"
      break
    fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "WARNING [pod_setup]: nvcc not on PATH. Prefer a RunPod *devel* PyTorch image (CUDA toolkit)."
  echo "  Recommended: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  echo "  Without nvcc, TRELLIS.2 CUDA ops compile will FAIL (probe aborts before weights — \$0 waste)."
fi

# ---------------------------------------------------------------------------
# 2) PINNED torch triplet (cu124) — unpinned = guaranteed ops/compile mismatch
# ---------------------------------------------------------------------------
echo "[pod_setup] installing PINNED $TORCH_PIN + $TORCHVISION_PIN + $TORCHAUDIO_PIN from $TORCH_INDEX"
python3 -m pip install --upgrade pip wheel setuptools >/dev/null 2>&1 || true
python3 -m pip install "$TORCH_PIN" "$TORCHVISION_PIN" "$TORCHAUDIO_PIN" --index-url "$TORCH_INDEX" \
  || die "failed to install pinned torch ($TORCH_PIN / $TORCHVISION_PIN / $TORCHAUDIO_PIN). Do NOT proceed with unpinned torch."

python3 - <<'PY' || die "torch CUDA sanity failed — aborting BEFORE clone/weights"
import torch, os
import torchaudio
build_only = os.environ.get("BUILD_ONLY", "0") == "1"
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("torchaudio", torchaudio.__version__)
if not build_only and not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
ver = torch.__version__.split("+")[0]
if not ver.startswith("2.6."):
    print(f"WARNING: expected torch 2.6.x for TRELLIS.2, got {torch.__version__}")
ta_ver = torchaudio.__version__.split("+")[0]
if not ta_ver.startswith(("2.6.","2.7.")):
    raise SystemExit(f"expected torchaudio 2.6.x matching torch, got {torchaudio.__version__}")
if build_only:
    print("BUILD_ONLY: torch import + version OK; skipping device props + cuda tensor (deferred to first-run)")
else:
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} VRAM_GB={props.total_memory/(1024**3):.1f}")
    x = torch.zeros(1, device="cuda")
    print("SANITY_OK", x.device)
PY

# ---------------------------------------------------------------------------
# 3) Clone TRELLIS.2 + compile CUDA ops
# ---------------------------------------------------------------------------
if [[ ! -d "$TRELLIS_DIR/.git" ]]; then
  echo "[pod_setup] cloning microsoft/TRELLIS.2 ..."
  git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR" \
    || die "git clone microsoft/TRELLIS.2 failed"
else
  echo "[pod_setup] TRELLIS.2 already cloned at $TRELLIS_DIR"
  (cd "$TRELLIS_DIR" && git submodule update --init --recursive || true)
fi

cd "$TRELLIS_DIR"
export PYTHONPATH="${TRELLIS_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# v6 recipe fixes: pillow-simd needs zlib/jpeg headers; official setup.sh's
# --flash-attn hits the build-isolation trap (no torch in the wheel env) —
# our fallback below installs it correctly with --no-build-isolation.
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq zlib1g-dev libjpeg-dev >/dev/null 2>&1 || true

# Prefer project setup scripts when present; fall back to pip -e .
if [[ -f setup.sh ]]; then
  echo "[pod_setup] running official TRELLIS.2 setup.sh (CUDA ops compile — expect 20–40 min)..."
  set +e
  # shellcheck disable=SC1091
  . ./setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
  SETUP_RC=$?
  set -e
  if [[ $SETUP_RC -ne 0 ]]; then
    echo "WARNING [pod_setup]: setup.sh exited $SETUP_RC — will rely on ops probe + pip fallbacks"
  fi
elif [[ -f install.sh ]]; then
  bash install.sh || true
fi

if [[ -f setup.py ]] || [[ -f pyproject.toml ]]; then
  pip install -e . --no-build-isolation || pip install -e . || \
    die "TRELLIS.2 editable install (pip install -e .) failed — 'trellis2' would be missing at runtime."
fi

# --- transformers 5.x compat patch (7/16) ------------------------------------
# TRELLIS.2's DinoV3FeatureExtractor was written against transformers where
# DINOv3ViTModel exposed its transformer blocks at `.layer`. transformers 5.x
# nests them one level deeper at `.model.layer` (verified: forward returns a
# plain tensor, embeddings/rope_embeddings stay top-level). Without this the
# pipeline loads fine but every /generate 500s with
# "'DINOv3ViTModel' object has no attribute 'layer'". Idempotent sed.
for _f in trellis2/modules/image_feature_extractor.py \
          trellis2/trainers/flow_matching/mixins/image_conditioned.py; do
  if [[ -f "$_f" ]]; then
    sed -i 's/enumerate(self\.model\.layer)/enumerate(self.model.model.layer)/' "$_f"
  fi
done
echo "[pod_setup] applied DINOv3 transformers-5.x .model.layer compat patch"

# Common runtime deps for image-to-3D export path
pip install -q pillow trimesh huggingface_hub einops opencv-python-headless || true

# Solidify-then-bake geometry stack (geo_relay_2026-07-24): occupancy union +
# exterior flood + marching cubes + quadric decimate. Non-fatal — if any of
# these is missing the worker's solidify import fails and it falls back to the
# proven band=1 export path (stamped in X-Geometry-Path).
# blinker pre-fix (attempt-4 lesson): Ubuntu ships a distutils blinker 1.4 that
# pip cannot uninstall; open3d's dash->flask->blinker chain then aborts the
# WHOLE combined install atomically (scipy/skimage lost too). Overwrite it
# first, and split the installs so open3d flakiness can't sink the core pair.
pip install -q --ignore-installed blinker || true
pip install -q "scipy==1.14.1" "scikit-image==0.24.0" fast-simplification || \
  echo "[pod_setup] WARN solidify CORE deps missing — worker will use band=1 fallback"
pip install -q "open3d==0.18.0" || \
  echo "[pod_setup] WARN open3d missing — decimation falls back to trimesh/fast-simplification"

# BiRefNet (MIT) background-removal deps: its Hugging Face trust_remote_code
# modeling imports timm + kornia (einops covered above; torch/torchvision/
# transformers/numpy already present). Without these, pod_worker's first
# /generate 500s with an ImportError while loading ZhengPeng7/BiRefNet. Kept
# non-fatal so setup still completes; a failure surfaces clearly at first
# generate. BiRefNet replaced TRELLIS.2's built-in RMBG-2.0 (non-commercial).
pip install -q timm kornia || true

# Explicit fallbacks for critical CUDA ops (idempotent; die if still missing after attempt)
echo "[pod_setup] ensuring critical CUDA ops packages (fallback path)..."
mkdir -p /tmp/extensions

python3 -c "import flash_attn" 2>/dev/null || \
  pip install flash-attn==2.7.3 --no-build-isolation || \
  die "flash-attn install failed. Need CUDA toolkit (nvcc) on a *devel* image. Aborting BEFORE weights."

python3 -c "import o_voxel" 2>/dev/null || {
  if [[ -d "$TRELLIS_DIR/o-voxel" ]]; then
    rm -rf /tmp/extensions/o-voxel
    cp -a "$TRELLIS_DIR/o-voxel" /tmp/extensions/o-voxel
    pip install /tmp/extensions/o-voxel --no-build-isolation || \
      die "o-voxel install failed. Aborting BEFORE weights (required for GLB export)."
  else
    die "o-voxel source missing at $TRELLIS_DIR/o-voxel (clone --recursive failed?)."
  fi
}

python3 -c "import cumesh" 2>/dev/null || {
  if [[ ! -d /tmp/extensions/CuMesh ]]; then
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
  fi
  pip install /tmp/extensions/CuMesh --no-build-isolation || \
    die "CuMesh install failed. Aborting BEFORE weights."
}

python3 -c "import flex_gemm" 2>/dev/null || {
  if [[ ! -d /tmp/extensions/FlexGEMM ]]; then
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
  fi
  pip install /tmp/extensions/FlexGEMM --no-build-isolation || \
    die "FlexGEMM install failed. Aborting BEFORE weights."
}

# ---------------------------------------------------------------------------
# 4) REAL ops smoke-probe — MUST pass before any multi-GB weights pull
# ---------------------------------------------------------------------------
if [[ "$BUILD_ONLY" == "1" ]]; then
  echo "[pod_setup] BUILD_ONLY=1 — skipping REAL ops device-probe (deferred to container first-run: ops_probe.py + HEALTHCHECK)."
else
echo "[pod_setup] REAL CUDA ops smoke-probe (BEFORE weights — money-safety gate)..."
set +e
PYTHONPATH="${TRELLIS_DIR}${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
"""Ops probe: flash_attn CUDA forward + o_voxel + flex_gemm + cumesh + trellis pipeline pkg."""
import sys

def fail(msg: str) -> None:
    print(f"OPS_PROBE_FAIL: {msg}", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except Exception as e:
    fail(f"torch import: {e}")

if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is False")

# --- flash-attn: real tiny forward (not just import) ---
try:
    from flash_attn import flash_attn_func
except Exception as e:
    fail(f"flash_attn import failed (ops not built?): {e}")

try:
    q = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, k, v)
    assert out is not None and out.shape == q.shape, out.shape
    print("flash_attn tiny forward OK:", tuple(out.shape))
except Exception as e:
    fail(f"flash_attn CUDA forward failed: {e}")

# --- o_voxel: required for GLB export ---
try:
    import o_voxel
except Exception as e:
    fail(f"o_voxel import failed (ops not built?): {e}")

if not hasattr(o_voxel, "postprocess"):
    fail("o_voxel has no postprocess module")
if not hasattr(o_voxel.postprocess, "to_glb"):
    fail("o_voxel.postprocess.to_glb missing — GLB export will crash")

try:
    t = torch.zeros(1, device="cuda")
    _ = t + 1
    print("o_voxel import + CUDA tensor OK; to_glb callable:", callable(o_voxel.postprocess.to_glb))
except Exception as e:
    fail(f"CUDA tensor after o_voxel import failed: {e}")

# flex_gemm / cumesh hard checks
for mod in ("flex_gemm", "cumesh"):
    try:
        __import__(mod)
        print(f"{mod} import OK")
    except Exception as e:
        fail(f"{mod} import failed: {e}")

# --- the actual pipeline package (what the worker imports at runtime) ---
_errs = []
for _mod in ("trellis2.pipelines", "trellis.pipelines"):
    try:
        __import__(_mod)
        print(f"{_mod} import OK")
        break
    except Exception as e:
        _errs.append(f"{_mod}: {e}")
else:
    fail("trellis pipeline package import failed: " + " | ".join(_errs))

print("OPS_PROBE_OK")
sys.exit(0)
PY
PROBE_RC=$?
set -e
if [[ $PROBE_RC -ne 0 ]]; then
  # die() (not bare exit) so the grace window + log server let the client
  # catch the ABORT marker and fail fast — this is the most likely abort path.
  die "OPS_PROBE_FAIL — CUDA ops / trellis package did not compile/import cleanly. Weights download SKIPPED."
fi

echo "[pod_setup] OPS_PROBE_OK — safe to consider weights download"
fi  # end BUILD_ONLY ops-probe skip

# RECIPE CAPTURE: freeze the proven stack into /workspace so the client can
# fetch it (log server / SSH) — feeds the network-volume + Docker-image bake.
pip freeze > "$WORK/proven_requirements.txt" 2>/dev/null || true
{ nvcc --version 2>/dev/null | tail -1; python3 --version; dpkg -l | grep -E "zlib1g-dev|libjpeg-dev" | awk '{print $2, $3}'; } > "$WORK/proven_env.txt" 2>/dev/null || true
echo "[pod_setup] recipe captured: proven_requirements.txt + proven_env.txt"

# ---------------------------------------------------------------------------
# 5) Weights ONLY after ops probe passed
# ---------------------------------------------------------------------------
if [[ "$SKIP_WEIGHTS" == "1" ]]; then
  echo "[pod_setup] SKIP_WEIGHTS=1 — not downloading microsoft/TRELLIS.2-4B"
elif [[ ! -f "$WEIGHTS_DIR/.download_complete" ]]; then
  echo "[pod_setup] ops probe passed — downloading microsoft/TRELLIS.2-4B weights (large) ..."
  WEIGHTS_DIR="$WEIGHTS_DIR" python3 - <<'PY' || die "weights download failed after successful ops probe"
from huggingface_hub import snapshot_download
import os
out = os.environ.get("WEIGHTS_DIR", "/workspace/weights/TRELLIS.2-4B")
snapshot_download(
    repo_id="microsoft/TRELLIS.2-4B",
    local_dir=out,
    local_dir_use_symlinks=False,
)
open(os.path.join(out, ".download_complete"), "w").write("ok\n")
print("weights ready at", out)
PY
else
  echo "[pod_setup] weights already present at $WEIGHTS_DIR"
fi

# Under a GPU-free build the image layer is now baked (ops compiled, deps pinned,
# recipe captured). Do NOT pull weights or start the worker at build time.
if [[ "$BUILD_ONLY" == "1" ]]; then
  echo "[pod_setup] BUILD_ONLY=1 — compile+install complete, image layer baked. Weights + worker deferred to runtime."
  exit 0
fi

# ---------------------------------------------------------------------------
# 6) Start worker or idle
# ---------------------------------------------------------------------------
WORKER_SRC="${WORKER_SRC:-$WORK/pod_worker.py}"
if [[ -f "$WORKER_SRC" ]]; then
  echo "[pod_setup] starting worker: $WORKER_SRC on :$WORKER_PORT"
  # Free the port: stop the throwaway setup log server before the worker binds.
  # pkill by pattern: the pidfile lives on the (persistent) volume and can go
  # stale across container restarts — a survivor holds the port and the worker
  # crashes on bind (hit this on the 7/16 volume build).
  pkill -f "http.server ${WORKER_PORT}" 2>/dev/null || true
  [[ -f "$WORK/.logsrv.pid" ]] && kill "$(cat "$WORK/.logsrv.pid")" 2>/dev/null || true
  sleep 1
  export TRELLIS_DIR WEIGHTS_DIR WORKER_PORT
  export PYTHONPATH="${TRELLIS_DIR}${PYTHONPATH:+:$PYTHONPATH}"
  exec python3 "$WORKER_SRC"
else
  echo "[pod_setup] no pod_worker.py at $WORKER_SRC — setup complete, idle."
  exec sleep infinity
fi
