#!/usr/bin/env python3
"""Ops probe: flash_attn CUDA forward + o_voxel + flex_gemm + cumesh + trellis pipeline pkg.

Faithful extract of the pod_setup.sh smoke-probe. In the GPU-free Docker build this
device probe is DEFERRED to container first-run (start_worker_hf.sh runs it before
serving) and to the HEALTHCHECK — the build itself never touches a GPU.
Exit 0 = OPS_PROBE_OK; non-zero = OPS_PROBE_FAIL (worker must NOT serve).
"""
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
