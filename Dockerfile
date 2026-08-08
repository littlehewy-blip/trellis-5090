# TRELLIS.2 image for RTX 5090 (Blackwell / sm_120, cu128).
# Bakes the PROVEN VN env once (compile flash-attn/spconv/kaolin/nvdiffrast/cumesh)
# so pods start in minutes instead of a ~1hr per-pod compile.
# Faithful to /workspace/pod_setup.sh on the VN pod; weights are NOT baked
# (mounted/downloaded at runtime via HF_TOKEN) to keep the image lean.
#
# GPU-FREE BUILD (2026-08-08 refactor): pod_setup.sh runs with BUILD_ONLY=1, which
# nvcc-cross-compiles the CUDA ops for sm_120 (TORCH_CUDA_ARCH_LIST=12.0) with NO
# GPU present, and DEFERS every runtime GPU touch (nvidia-smi, torch.cuda forwards,
# the ops probe) to container first-run (start_worker_hf.sh -> ops_probe.py) and the
# HEALTHCHECK. So `docker build` needs Docker + CPU/RAM/disk only — no GPU. Build it
# anywhere (GitHub Actions -> ghcr.io, a CPU VM, etc.), then run on a 5090 pod.

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Compile parallelism. On the 16GB free GitHub runner, flash-attn's sm_120
# (Blackwell) kernels OOM-kill the runner at MAX_JOBS 8, 4, AND 2 — a single
# ptxas invocation can peak >10GB. Fully SERIALIZE the compile (1 job, 1 nvcc
# thread) + rely on the 20GB host swapfile (workflow step) for headroom.
# MAX_JOBS covers the PyTorch cpp_extension builds (flash-attn); CMAKE/MAKEFLAGS
# below cover the cmake/make sub-builds (o-voxel, nvdiffrast) that ignore MAX_JOBS.
ARG MAX_JOBS=2
ARG NVCC_THREADS=1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    MAKEFLAGS=-j2 \
    PIP_NO_CACHE_DIR=1 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRELLIS_DIR=/workspace/TRELLIS.2 \
    WEIGHTS_DIR=/workspace/weights/TRELLIS.2-4B \
    WORKER_PORT=8000 \
    PYTHONPATH=/workspace/TRELLIS.2

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ninja-build build-essential libgl1 libglib2.0-0 wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Proven build script + worker + pin list + first-run probe, from the VN pod.
COPY pod_setup.sh pod_worker.py proven_requirements.txt start_worker_hf.sh start_worker.sh ops_probe.py /workspace/

# Compile CUDA ops + install pinned deps ONCE, GPU-FREE. BUILD_ONLY=1 => nvcc
# cross-compiles the ops for sm_120 with no device, and skips the runtime GPU
# checks (nvidia-smi / cuda forwards / ops-probe / tee log-server / worker exec) —
# those defer to first-run. SKIP_WEIGHTS=1 => the multi-GB TRELLIS.2-4B weights are
# NOT baked (they mount/download at /workspace/weights at runtime).
RUN BUILD_ONLY=1 SKIP_WEIGHTS=1 FORCE_CUDA=1 MAX_JOBS=2 NVCC_THREADS=1 CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 TORCH_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES= bash /workspace/pod_setup.sh

EXPOSE 8000

# Deferred GPU gate: the container is healthy only once the real ops probe passes
# on the actual GPU host. start_worker_hf.sh runs the same probe before serving.
HEALTHCHECK --interval=30s --timeout=30s --start-period=20m --retries=10 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Entry = the SUPERVISOR (circ review 2026-08-08, A5 P0 crash fix): it runs start_worker_hf.sh
# in a restart loop so a hard-CUDA-fault exit(77) reloads the model instead of killing the pod
# (flock single-instance + pkill; backoff resets on the /workspace/.last_success marker, not
# wall-time; worker stdout/stderr -> /workspace/worker.log). start_worker_hf.sh still reads
# /workspace/.hf_token, runs the ops probe, downloads weights on first run, serves :8000.
CMD ["bash", "/workspace/start_worker.sh"]
