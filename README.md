# TRELLIS.2 5090 Docker bake — ready for circ review

**Goal:** bake the proven VN TRELLIS.2 env into one image so every future pod starts in
minutes (docker pull + run) instead of the ~1hr per-pod source compile.

**Status:** Dockerfile + build script + full build context assembled (2026-08-08).
NOT YET BUILT — the mini is AMD/ROCm with no Docker, so it can't build a CUDA/sm_120 image.

## Contents (build context)
- `Dockerfile` — bakes env via the proven `pod_setup.sh` (SKIP_WEIGHTS=1); weights mount at runtime.
- `build_and_push.sh` — run on a Docker-capable NVIDIA host; set `REG`.
- `pod_setup.sh`, `pod_worker.py`, `proven_requirements.txt`, `start_worker_hf.sh` — pulled from VN `/workspace/`.

## To build (on an NVIDIA + Docker host, NOT the mini)
```bash
REG=ghcr.io/<you> bash build_and_push.sh
```
Then launch pods: image `ghcr.io/<you>/trellis-5090:v1`, mount a weights volume at
`/workspace/weights`, mount `.hf_token`. Point `pod_lane_mf.sh <host> <port>` at it as a new lane.

## Open questions for circ review (GPT + Grok)
1. **Build host** — where? Vast pods are containers; docker-in-docker often unavailable. Need a
   host that supports `docker build` with NVIDIA runtime, or a dedicated builder.
2. **Base image** — `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` vs VN's cu13.0.3 template.
   pod_setup.sh pins torch 2.7.1+cu128, so cu128 devel should match — confirm sm_120 kernels build.
3. **pod_setup.sh in `docker build`** — it self-tees to setup.log and briefly starts an http.server
   on WORKER_PORT; `die()` sleeps 600s on failure. Harmless in build but consider a build-only path.
4. **Weights** — runtime volume-mount (chosen) vs baking the 4B into the image (huge). Confirm.
5. **Registry + push creds** — GHCR vs Docker Hub.

## Circ-review note
The approach is already covered by `tools/TRELLIS_SPINUP_CIRC_REVIEW_2026-08-07.md`
(GPT-5.2 + Grok-4.5, one day old). A fresh review of THIS Dockerfile needs XAI/OPENAI keys,
which are not on the mini (they live on A5 `C:\Users\Littl\config\.env`).
