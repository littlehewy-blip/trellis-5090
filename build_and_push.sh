#!/bin/bash
# Build + push the TRELLIS.2 5090 image. GPU-FREE BUILD (2026-08-08): the Dockerfile
# runs pod_setup.sh with BUILD_ONLY=1, so this builds on ANY Docker host — CPU VM,
# GitHub Actions, laptop — NO GPU needed. (The mini still can't run it: AMD/ROCm has
# no Docker/NVIDIA runtime — but any Docker+CPU box works. GH Actions is cheapest, see
# .github/workflows/build-trellis.yml.)
#
# Prereqs on the build host:
#   - docker (CPU is fine; nvidia runtime NOT required to build)
#   - registry login done (e.g. `docker login ghcr.io`)
#   - the proven files present in this dir (pulled from VN /workspace/):
#       pod_setup.sh pod_worker.py proven_requirements.txt start_worker_hf.sh ops_probe.py
set -euo pipefail
REG="${REG:-ghcr.io/PLACEHOLDER_USER}"      # <-- set to your GHCR/DockerHub namespace
IMG="${IMG:-trellis-5090}"
TAG="${TAG:-v1}"
REF="$REG/$IMG:$TAG"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

for f in Dockerfile pod_setup.sh pod_worker.py proven_requirements.txt start_worker_hf.sh ops_probe.py; do
  [ -e "$f" ] || { echo "MISSING build-context file: $f (pull from VN /workspace/)"; exit 1; }
done

echo "[build] $REF (GPU-free: nvcc cross-compiles flash-attn/spconv/kaolin/nvdiffrast/cumesh for sm_120 — ~30-45min first time)"
docker build --progress=plain -t "$REF" .

echo "[smoke] import-only check inside the image (no GPU: cuda will read False on a CPU builder, expected)"
docker run --rm "$REF" python -c "import torch; print('torch',torch.__version__,'cuda_build',torch.version.cuda)" || \
  echo "WARN: import smoke failed — inspect build log"

echo "[push] $REF"
docker push "$REF"
echo "DONE -> launch pods with image=$REF, mount weights volume at /workspace/weights, mount .hf_token"
