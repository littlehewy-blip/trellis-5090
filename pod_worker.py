#!/usr/bin/env python3
"""Minimal HTTP worker for TRELLIS.2-4B image-to-3D on the RunPod GPU.

Runs inside the pod after pod_setup.sh. Host deps stay stdlib+requests+Pillow;
this file may use torch/trellis only inside the container.

Endpoints:
  GET  /health  -> {"ok": true, "model": "...", "busy": false, ...}
  POST /generate  multipart: file=<png>  -> application/octet-stream GLB
  POST /generate  JSON: {"image_b64": "...", "name": "blue"} -> GLB bytes
  POST /generate while a render is in flight -> 429 + Retry-After (SERIAL GUARD)

SERIAL-RENDER GUARD (2026-07-26, POD_THROUGHPUT_BRIEF): this is a
ThreadingHTTPServer, so before this patch a second POST /generate simply started
a SECOND render on the same GPU. That is exactly how the measured OOM cascades
began: the client's 600s timeout gave up, the GPU kept rendering the abandoned
job, and the next POST stacked on top of it (21:13/21:23 -> tunnel dead -> 501s
-> 28.9 min at zero; 01:38 -> 40 min at zero). A module-level lock now serialises
/generate; a concurrent POST is answered 429 + Retry-After instead of stacking.
The request body is always read to completion BEFORE the 429 is written, so the
client sees a clean 429 rather than a connection reset.
"""

from __future__ import annotations

import base64
import cgi  # removed from stdlib in py3.13 — container image pins py3.11 (see config docker_image)
import errno
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# THREAD CAP (circular must-fix P0, 2026-07-30 H200 review): on a 192-vCPU pod,
# uncapped OpenMP/MKL/OpenBLAS fan-out turns CPU-side stages (solidify: trimesh/
# scipy/skimage/Open3D) into a thread-storm — seconds become minutes. Must be set
# before the first numpy/scipy import anywhere in the process.
_CPU_THREADS = os.environ.get("CPU_THREADS", "8")
for _tv in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS", "GOTO_NUM_THREADS"):
    os.environ.setdefault(_tv, _CPU_THREADS)

PORT = int(os.environ.get("WORKER_PORT", "8000"))
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "/workspace/weights/TRELLIS.2-4B"))
MODEL_ID = os.environ.get("MODEL_ID", "microsoft/TRELLIS.2-4B")
TEX_SIZE = int(os.environ.get("TEXTURE_RES", "2048"))
DECIMATE = int(os.environ.get("DECIMATE_TARGET", "200000"))
# Background-removal model: BiRefNet (MIT) replaces TRELLIS.2's bundled RMBG-2.0
# (non-commercial) so the model whose output shapes a sold asset is MIT-licensed.
BIREFNET_MODEL_ID = os.environ.get("BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet")

_pipeline = None
_birefnet_model = None
# #7 readiness (A5 fast-follow): lifecycle so callers back off during the MINUTES-long model
# load / reload instead of hammering HTTP 000. "loading" -> "ready" (pipeline up) -> "dead"
# (load raised - supervisor will restart the process). Guarded by a lock (circ 2026-08-08): a
# plain str assignment is GIL-atomic, but the lock makes the set/read explicit and future-proof.
_LOAD_STATE = "loading"
_LOAD_STATE_LOCK = threading.Lock()


def _set_load_state(state: str) -> None:
    global _LOAD_STATE
    with _LOAD_STATE_LOCK:
        _LOAD_STATE = state


def _get_load_state() -> str:
    with _LOAD_STATE_LOCK:
        return _LOAD_STATE

# --- SERIAL-RENDER GUARD state ---------------------------------------------
# One render at a time, process-wide. Held for the GPU section only (the
# multipart parse happens outside it so a busy worker still drains the body).
_GEN_LOCK = threading.Lock()
_GEN_BUSY_SINCE = 0.0   # monotonic-ish wall start of the in-flight render, 0 = idle
_GEN_BUSY_NAME = ""     # slot name of the in-flight render, for diagnostics
_GEN_PROGRESS_TS = 0.0  # wall time of the last observed FORWARD PROGRESS
_GEN_PROGRESS_WHAT = "" # what that progress was, for the watchdog's log line


def mark_progress(what: str) -> None:
    """Record that the in-flight render advanced a stage.

    Pete 2026-08-04: the watchdog must watch WORK, not the clock. A pure elapsed
    timer cannot tell a slow-but-healthy render from a wedged one, so it will
    eventually kill good work. Stage boundaries call this; the watchdog refuses
    to kill anything that has advanced recently."""
    global _GEN_PROGRESS_TS, _GEN_PROGRESS_WHAT
    _GEN_PROGRESS_TS, _GEN_PROGRESS_WHAT = time.time(), what
# Optional per-pod GPU telemetry (utilization + memory.used + memory.reserved).
# Namespaced by hostname because two pods can share one network volume at
# /workspace and would otherwise interleave into one CSV.
GPU_TELEMETRY = os.environ.get("GPU_TELEMETRY", "1") == "1"
GPU_TELEMETRY_S = int(os.environ.get("GPU_TELEMETRY_S", "60"))
# Build pin, reported by /health (circular review must-fix P0-1). dockerStartCmd
# rewrites /workspace/pod_worker.py from a base64 blob captured at pod-CREATE
# time on every container start, so a pod can silently revert to an older worker
# and nothing downstream would notice. The driver logs this string at startup so
# "I thought the serial guard was live" can never be a guess.
WORKER_BUILD = "2026-08-08-fastfollow"

# --- P0 pod-stability knobs (circular_pod_stability VERDICT, 2026-08-04) ------
# IO_TIMEOUT_S: per-request socket timeout so a half-dead ssh -L forward cannot
# block wfile.write of the GLB forever while _GEN_LOCK is held (P0-3).
# BUSY_HARD_LIMIT_S: if a render holds the serial guard longer than this the
# render thread is wedged (CPU/native hang or a blocked send); the watchdog
# os._exit(77)s so start_worker.sh's restart loop reloads the model (P0-2).
# Default 600 (NOT 1200 - a soft limit strands the pod ~20 min). Recovery is a
# full model reload (MINUTES), not seconds. Both env-overridable.
IO_TIMEOUT_S = int(os.environ.get("IO_TIMEOUT_S", "90"))
BUSY_HARD_LIMIT_S = int(os.environ.get("BUSY_HARD_LIMIT_S", "600"))
BUSY_WATCHDOG_TICK_S = int(os.environ.get("BUSY_WATCHDOG_TICK_S", "15"))
# PROGRESS-AWARE WATCHDOG (Pete 2026-08-04). BUSY_HARD_LIMIT_S alone killed a
# healthy-but-slow render on render-b. Elapsed time is now only a PRECONDITION:
# the watchdog additionally requires that the render has made no stage progress
# for BUSY_STALL_LIMIT_S *and* that this process is burning no CPU. That last
# test matters because the solidify/bake stage is CPU-side with the GPU at 0%,
# so GPU-idle on its own is a false-positive kill signal. BUSY_ABS_CEILING_S is
# the backstop for a wedge that still spins CPU (infinite loop).
BUSY_STALL_LIMIT_S = int(os.environ.get("BUSY_STALL_LIMIT_S", "300"))
# Ceiling 3600 was rejected by both reviewers as far too soft on a metered GPU:
# measured H100 job is ~307s, so 900 is ~3x headroom and caps the blast radius
# of a spinning wedge at ~$0.65 on the H100 instead of ~$2.59.
BUSY_ABS_CEILING_S = int(os.environ.get("BUSY_ABS_CEILING_S", "900"))
# Jiffies of CPU this process must burn per tick to count as "working". Linux
# USER_HZ is 100, so 15 jiffies over a 15s tick is ~1% of one core - low enough
# that a genuinely stuck process never clears it, high enough that timer noise
# does not keep a corpse alive.
BUSY_CPU_MIN_JIFFIES = int(os.environ.get("BUSY_CPU_MIN_JIFFIES", "15"))


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    _set_load_state("loading")

    # Official TRELLIS.2 import path (microsoft/TRELLIS.2)
    Pipeline = None
    errs: list = []
    for mod_name, cls_name in (
        ("trellis2.pipelines", "Trellis2ImageTo3DPipeline"),
        ("trellis2.pipelines", "TrellisImageTo3DPipeline"),
        ("trellis.pipelines", "TrellisImageTo3DPipeline"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            Pipeline = getattr(mod, cls_name)
            break
        except Exception as exc:
            errs.append(f"{mod_name}.{cls_name}: {exc}")
            continue
    if Pipeline is None:
        raise RuntimeError("TRELLIS pipeline import failed: " + " | ".join(errs))

    if WEIGHTS_DIR.is_dir() and any(WEIGHTS_DIR.iterdir()):
        pipe = Pipeline.from_pretrained(str(WEIGHTS_DIR))
    else:
        pipe = Pipeline.from_pretrained(MODEL_ID)
    pipe.cuda()
    # DTYPE-MISMATCH ROOT FIX v2 (2026-07-19). A dependency drift made
    # TRELLIS.2-4B's weights load in fp16 (Half) while the pipeline's own
    # preprocessing still produces an fp32 (float) input tensor -> every
    # /generate 500'd at the first conv/bias with
    #   "Input type (float) and bias type (c10::Half) should be the same".
    # Round 1 wrapped pipe.run() in an OUTER torch.autocast("cuda", fp16); it
    # did NOT help because TRELLIS runs numerically-sensitive blocks inside an
    # INNER torch.autocast(enabled=False) that an outer context cannot override
    # (failure reproduced at ~2.7s, i.e. before autocast could ever act). The
    # pre-regression numeric path was fp32 end-to-end, so restore it: force the
    # entire pipeline to fp32 so weights match the fp32 input EVERYWHERE,
    # autocast-disabled regions included. Version/direction-agnostic and it
    # removes the clash at the source instead of masking it at the boundary.
    # FIX3 (2026-07-19) — STAGED, NOT YET SMOKED. The fp16/fp32 "regression"
    # premise above was WRONG: rounds 1 & 2 never reached pipe.run() at all — the
    # true first failure was BiRefNet (a SEPARATE model, now fixed with
    # model.float() in load_birefnet). With BiRefNet fixed, pipe.run() ran for the
    # first time and _force_pipeline_fp32() itself CAUSED a new clash:
    #   "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"
    #   (sparse_structure_flow.py:240 -> attention/modules.py:69 -> linear.py:125).
    # TRELLIS.2 natively runs bf16: it creates bf16 activations that the
    # fp32-forced weights no longer match. Correct action is to leave the pipeline
    # in its native dtype and drop the fp32 force entirely. Disabled below; the
    # BiRefNet fix stands on its own. Verify in an interactive/human-eyes smoke.
    # DTYPE-FIX (2026-07-29): unify each leaf module's params+buffers to its
    # own weight dtype and to the CUDA device. Kills the intra-layer split
    # 'Input type (c10::Half) and bias type (float) should be the same' (and
    # the cpu/cuda straggler variant) WITHOUT forcing a single global dtype
    # (fp32-force was proven wrong: it clashed with the native bf16 attention).
    _unify_pipeline_dtypes(pipe)
    _pipeline = pipe
    _set_load_state("ready")  # #7: callers may now stop backing off
    return pipe


def _force_pipeline_fp32(pipe) -> None:
    """Cast every torch sub-module the pipeline owns to fp32, non-fatally.

    TRELLIS pipelines are plain container objects (not a single nn.Module), so
    weights hang off attributes / a `.models` dict. Cast each nn.Module we can
    reach; a miss just means that branch stays as-loaded (the traceback capture
    in do_POST will pinpoint any surviving fp16 op). Best-effort, never raises.
    """
    import torch.nn as nn

    def _cast(obj) -> None:
        try:
            fn = getattr(obj, "float", None)
            if callable(fn):
                fn()
        except Exception:
            pass

    _cast(pipe)
    for attr in ("models", "modules", "modules_dict", "_models"):
        d = getattr(pipe, attr, None)
        if isinstance(d, dict):
            for m in d.values():
                if isinstance(m, nn.Module):
                    _cast(m)
    for name in dir(pipe):
        try:
            v = getattr(pipe, name)
        except Exception:
            continue
        if isinstance(v, nn.Module):
            _cast(v)


def _iter_pipeline_modules(pipe):
    """Yield every nn.Module the TRELLIS pipeline owns (it is a plain container,
    not one nn.Module). Mirrors _force_pipeline_fp32's reachability walk."""
    import torch.nn as nn
    seen = set()

    def _emit(m):
        if isinstance(m, nn.Module) and id(m) not in seen:
            seen.add(id(m))
            return True
        return False

    roots = []
    for attr in ("models", "modules", "modules_dict", "_models"):
        d = getattr(pipe, attr, None)
        if isinstance(d, dict):
            roots.extend(d.values())
    for name in dir(pipe):
        try:
            roots.append(getattr(pipe, name))
        except Exception:
            continue
    for r in roots:
        if isinstance(r, nn.Module) and _emit(r):
            # .modules() yields the root plus every descendant leaf/branch
            for sub in r.modules():
                if _emit(sub):
                    yield sub
                elif sub is r:
                    yield r


def _unify_pipeline_dtypes(pipe) -> None:
    """Make every leaf module internally dtype- and device-consistent.

    ROOT CAUSE (2026-07-29): after from_pretrained + pipe.cuda(), a handful of
    modules end up with their *weight* in fp16/bf16 but their *bias* (or a
    lazily-created / non-persistent float buffer) left in fp32 -- or left on
    CPU. The forward then throws, per layer, one of:
      * "Input type (c10::Half) and bias type (float) should be the same"
      * "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)"
      * "Expected all tensors to be on the same device, cuda:0 and cpu!"
    This poisons entire input classes (sb_*, tr_*, ww2_*, ssc_*, ...).

    FIX: for each leaf module, take its `weight` dtype as the local truth and
    cast that module's *floating* params + buffers to it, and move ALL of its
    params + buffers to CUDA. We deliberately do NOT impose one global dtype
    (the disabled _force_pipeline_fp32 proved that fights TRELLIS.2's native
    bf16 attention -> 'mat1 and mat2 ... BFloat16 and Float'). Non-float buffers
    (int index tensors, bool masks) keep their dtype; only device is unified.
    Best-effort and never raises -- a miss just leaves that branch as-loaded.
    """
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_dtype = 0
    fixed_dev = 0

    for mod in _iter_pipeline_modules(pipe):
        # Local float target = this module's own weight dtype, else its first
        # floating parameter. None => module has no float params (skip dtype).
        target = None
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.is_floating_point():
            target = w.dtype
        if target is None:
            for p in mod.parameters(recurse=False):
                if p is not None and p.is_floating_point():
                    target = p.dtype
                    break

        for pname, p in list(mod.named_parameters(recurse=False)):
            if p is None:
                continue
            new_dtype = target if (p.is_floating_point() and target is not None) else p.dtype
            if p.dtype != new_dtype or p.device != dev:
                if p.dtype != new_dtype:
                    fixed_dtype += 1
                if p.device != dev:
                    fixed_dev += 1
                with torch.no_grad():
                    p.data = p.data.to(device=dev, dtype=new_dtype)

        for bname, b in list(mod.named_buffers(recurse=False)):
            if b is None:
                continue
            new_dtype = target if (b.is_floating_point() and target is not None) else b.dtype
            if b.dtype != new_dtype or b.device != dev:
                if b.dtype != new_dtype:
                    fixed_dtype += 1
                if b.device != dev:
                    fixed_dev += 1
                mod._buffers[bname] = b.to(device=dev, dtype=new_dtype)

    print(
        f"[worker] _unify_pipeline_dtypes: recast {fixed_dtype} float tensor(s), "
        f"moved {fixed_dev} tensor(s) to {dev}",
        flush=True,
    )


def load_birefnet():
    """Lazy-load BiRefNet (ZhengPeng7/BiRefNet, MIT) for background removal.

    Replaces TRELLIS.2's built-in RMBG-2.0 (non-commercial) as the model whose
    output shapes the sold asset. Weights auto-download from Hugging Face on the
    pod at first call, exactly as RMBG's did (public repo; HF_TOKEN is read from
    the environment only if present and is never logged).
    """
    global _birefnet_model
    if _birefnet_model is not None:
        return _birefnet_model
    import torch
    from transformers import AutoModelForImageSegmentation

    token = os.environ.get("HF_TOKEN") or None
    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_MODEL_ID, trust_remote_code=True, token=token
    )
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # FIX3 (2026-07-19): ZhengPeng7/BiRefNet's remote code loads its weights in
    # fp16 by default. transform_image(...ToTensor) produces an fp32 input, so
    # the very first patch_embed conv threw "Input type (float) and bias type
    # (c10::Half) should be the same" (birefnet.py:1045 -> conv.py:549). This is
    # a SEPARATE model from the TRELLIS `pipe` that _force_pipeline_fp32() casts,
    # so fixes 1 & 2 never touched it. Force the whole cutout model to fp32 to
    # match the fp32 input everywhere (mask quality unaffected; H100 has headroom).
    model.float()
    model.to(device)
    model.eval()
    _birefnet_model = model
    return model


def birefnet_cutout(image_path: Path):
    """Open an image and return an RGBA PIL image whose alpha is BiRefNet's
    foreground mask.

    Same input/output role as the previous RMBG-2.0 step (image in ->
    foreground-only image out); only the removal model changed. The RGBA result
    is handed to TRELLIS.2's pipeline, whose preprocess_image detects the
    non-uniform alpha channel and therefore skips its own RMBG-2.0 removal while
    still performing the geometric foreground normalization it always does.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    model = load_birefnet()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size)
    # LICENSE GUARD: TRELLIS.2 bypasses its RMBG-2.0 step only when alpha is
    # NON-uniform (not np.all(alpha == 255)). If BiRefNet ever returns an
    # all-foreground mask, pin one corner pixel to 254 so the bypass always
    # triggers and RMBG-2.0 can never silently run on a sold asset.
    if mask.getextrema()[0] == 255:
        mask.putpixel((0, 0), 254)
    image.putalpha(mask)
    return image


# === Geometry solidify helpers (geo_relay_2026-07-24: Grok diagnosis + GPT code).
# Root cause of multi-shell / non-watertight output: TRELLIS's extract is an
# isosurface of a noisy multi-sheet field (onion skins, hairline seams at thin
# junctions — measured gaps 0.5–2 voxels at articulation zones). Fix: union all
# shells in a dense occupancy grid, bridge sub-voxel seams (binary closing),
# exterior flood-fill, marching cubes -> ONE closed manifold, then bake — the
# 2048 texture is sampled from attr_volume AFTER geometry, so nothing is lost.
# band>1 remesh is BANNED: measured 214->28,320 comps + 16% degens + OOM death.
import gc
import time
import numpy as np

# Tunables (env-overridable)
SOLIDIFY_RES = int(os.environ.get("SOLIDIFY_RES", "288"))      # 256-320 sweet spot
SOLIDIFY_CLOSE = int(os.environ.get("SOLIDIFY_CLOSE", "2"))    # closing iterations (voxels)
SOLIDIFY_SMOOTH = int(os.environ.get("SOLIDIFY_SMOOTH", "3"))  # Taubin iterations post-MC
DEBRIS_FACE_FRAC = float(os.environ.get("DEBRIS_FACE_FRAC", "0.002"))
FALLBACK_BAND = int(os.environ.get("REMESH_BAND", "1"))        # proven-safe fallback
FALLBACK_PROJECT = int(os.environ.get("REMESH_PROJECT", "0"))


def _torch_vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    return -1


def _exterior_from_border(mask_bool: np.ndarray) -> np.ndarray:
    """Compute the 'outside' region by flooding the complement from the padded border."""
    from scipy import ndimage
    inv = ~mask_bool
    seed = np.zeros_like(inv, dtype=bool)
    seed[0, :, :] = inv[0, :, :]
    seed[-1, :, :] = inv[-1, :, :]
    seed[:, 0, :] = inv[:, 0, :]
    seed[:, -1, :] = inv[:, -1, :]
    seed[:, :, 0] = inv[:, :, 0]
    seed[:, :, -1] = inv[:, :, -1]
    outside = ndimage.binary_propagation(seed, mask=inv)
    return outside


def _rasterize_fixed_grid(tm, aabb_min, aabb_max, res):
    """Conservative fallback voxelizer: mark voxels for vertices; dilate once.
    Returns (grid_bool, 4x4 transform index->world)."""
    from scipy import ndimage
    aabb_min = np.asarray(aabb_min, dtype=np.float64)
    aabb_max = np.asarray(aabb_max, dtype=np.float64)
    grid = np.zeros((res, res, res), dtype=bool)
    rel = (tm.vertices - aabb_min) / (aabb_max - aabb_min + 1e-12)
    ijk = np.clip((rel * (res - 1)).astype(np.int32), 0, res - 1)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    grid = ndimage.binary_dilation(grid, iterations=1)
    scale = (aabb_max - aabb_min) / (res - 1)
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0], transform[1, 1], transform[2, 2] = scale
    transform[:3, 3] = aabb_min
    return grid, transform


def solidify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    aabb_min=(-0.5, -0.5, -0.5),
    aabb_max=(0.5, 0.5, 0.5),
    res: int = SOLIDIFY_RES,
    close_rad: int = SOLIDIFY_CLOSE,
    smooth_iters: int = SOLIDIFY_SMOOTH,
    target_faces: int = 200_000,
) -> tuple:
    """Union all shells via dense occupancy -> seam closing -> exterior flood -> MC.
    Returns (verts_float32, faces_int32, stats). Raises on failure."""
    import trimesh
    from skimage.measure import marching_cubes
    from scipy import ndimage

    t0 = time.time()
    stats = {"res": int(res)}

    verts = vertices
    faces_in = faces
    if hasattr(verts, "detach"):
        verts = verts.detach().cpu().numpy()
    if hasattr(faces_in, "detach"):
        faces_in = faces_in.detach().cpu().numpy()
    verts = np.asarray(verts, dtype=np.float64, order="C")
    faces_in = np.asarray(faces_in, dtype=np.int64, order="C")

    stats["input_verts"] = int(len(verts))
    stats["input_faces"] = int(len(faces_in))
    if len(faces_in) < 10 or len(verts) < 3:
        raise RuntimeError("raw mesh too small to solidify")

    finite = np.isfinite(verts).all(axis=1)
    if not finite.all():
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[finite] = np.arange(finite.sum())
        faces_in = remap[faces_in]
        faces_in = faces_in[(faces_in >= 0).all(axis=1)]
        verts = verts[finite]
        stats["dropped_nonfinite_verts"] = int((~finite).sum())

    tm = trimesh.Trimesh(vertices=verts, faces=faces_in, process=False)

    # Pre-voxel debris drop: cheap and prevents exploding grids on confetti
    try:
        parts = tm.split(only_watertight=False)
    except Exception:
        parts = [tm]
    if len(parts) > 1:
        parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
        min_faces = max(50, int(DEBRIS_FACE_FRAC * len(tm.faces)))
        kept = [p for p in parts if len(p.faces) >= min_faces]
        if not kept:
            kept = parts[:1]
        tm = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
        stats["pre_comps"] = int(len(parts))
        stats["pre_kept"] = int(len(kept))
    else:
        stats["pre_comps"] = 1
        stats["pre_kept"] = 1

    # Dense occupancy in the fixed world AABB
    extent = np.asarray(aabb_max, dtype=np.float64) - np.asarray(aabb_min, dtype=np.float64)
    pitch = float(np.max(extent) / res)
    # Heartbeats through solidify (circular review P0-2). This is the historical
    # 0%-GPU hang site AND the longest CPU stage, so with CPU removed from the
    # kill predicate these marks are the only thing distinguishing "slow" from
    # "wedged". Each step below can run for minutes on a dense mesh.
    mark_progress("solidify:voxelize")
    try:
        vox = tm.voxelized(pitch=pitch, method="subdivide")
        grid = np.asarray(vox.matrix, dtype=bool, order="C")
        transform = np.asarray(vox.transform, dtype=np.float64)
    except Exception as exc:
        stats["voxelize_fallback"] = str(exc)
        grid, transform = _rasterize_fixed_grid(tm, aabb_min, aabb_max, res)
    stats["grid_shape_raw"] = list(np.asarray(grid).shape)

    # Pad a voxel so the border is a guaranteed seed for outside
    grid = np.pad(grid, 1, constant_values=False)

    # Bridge hairline seams (the measured 0.5-2 voxel junction gaps)
    if close_rad > 0:
        structure = ndimage.generate_binary_structure(3, 3)
        mark_progress("solidify:closing")
        grid = ndimage.binary_closing(grid, structure=structure, iterations=int(close_rad))

    # Exterior flood on complement, then solid = ~outside
    mark_progress("solidify:flood")
    outside = _exterior_from_border(grid)
    solid = ~outside

    # Keep only the largest solid component (removes closed interior debris blobs)
    mark_progress("solidify:label")
    labeled, nlab = ndimage.label(solid)
    if nlab == 0:
        raise RuntimeError("no solid after flood/closing")
    if nlab > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        keep_id = int(counts.argmax())
        solid = (labeled == keep_id)
        stats["voxel_comps"] = int(nlab)
    else:
        stats["voxel_comps"] = 1

    # Un-pad to match the original transform
    solid = solid[1:-1, 1:-1, 1:-1]
    occ = int(solid.sum())
    stats["occupied_voxels"] = occ
    if occ < 10:
        raise RuntimeError("occupied voxel count too small")

    # Marching cubes at 0.5; verts are in grid index space -> world via transform
    mark_progress("solidify:marching_cubes")
    mc_verts, mc_faces, _, _ = marching_cubes(solid.astype(np.float32, order="C"), level=0.5)
    homo = np.concatenate([mc_verts.astype(np.float64), np.ones((len(mc_verts), 1))], axis=1)
    world = (transform @ homo.T).T[:, :3]

    tm_out = trimesh.Trimesh(vertices=world, faces=mc_faces.astype(np.int64), process=True)

    # Largest surface component (belt-and-suspenders)
    try:
        parts = tm_out.split(only_watertight=False)
        if len(parts) > 1:
            parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
            tm_out = parts[0]
            stats["post_mc_comps"] = int(len(parts))
        else:
            stats["post_mc_comps"] = 1
    except Exception:
        stats["post_mc_comps"] = -1

    # Light smoothing of MC stair-steps (Taubin: non-shrinking)
    if smooth_iters > 0 and len(tm_out.vertices) > 0:
        try:
            mark_progress("solidify:smooth")
            trimesh.smoothing.filter_taubin(tm_out, iterations=int(smooth_iters))
            stats["smooth"] = int(smooth_iters)
        except Exception as sm_exc:
            stats["smooth"] = f"skip:{sm_exc}"

    # Decimate to target
    if target_faces and len(tm_out.faces) > target_faces:
        try:
            import open3d as o3d
            o3 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(tm_out.vertices.astype(np.float64, order="C")),
                o3d.utility.Vector3iVector(tm_out.faces.astype(np.int32, order="C")),
            )
            o3 = o3.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
            o3.remove_degenerate_triangles()
            o3.remove_duplicated_triangles()
            o3.remove_duplicated_vertices()
            o3.remove_non_manifold_edges()
            tm_out = trimesh.Trimesh(
                np.asarray(o3.vertices), np.asarray(o3.triangles), process=True
            )
            stats["decimate"] = "open3d_quadric"
        except Exception as dec_exc:
            try:
                tm_out = tm_out.simplify_quadric_decimation(face_count=int(target_faces))
                stats["decimate"] = "trimesh_quadric"
            except Exception as dec2:
                stats["decimate"] = f"skip:{dec_exc}|{dec2}"

    v_out = np.asarray(tm_out.vertices, dtype=np.float32, order="C")
    f_out = np.asarray(tm_out.faces, dtype=np.int32, order="C")
    stats.update(
        {
            "output_verts": int(len(v_out)),
            "output_faces": int(len(f_out)),
            "watertight": bool(getattr(tm_out, "is_watertight", False)),
            "elapsed_s": round(time.time() - t0, 3),
        }
    )
    if len(f_out) < 10:
        raise RuntimeError("solidify produced too few faces")
    return v_out, f_out, stats


def _to_like(arr, ref, is_float: bool):
    """Match arr to ref's container/dtype/device. o_voxel's to_glb calls .cuda()
    on its inputs — numpy in = AttributeError ('numpy.ndarray' has no 'cuda',
    attempt-3 0/4). The pipeline's own mesh.vertices/faces are torch tensors, so
    mirror them exactly; plain numpy only if ref itself is numpy."""
    if hasattr(ref, "device"):
        import torch
        t = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(
            np.ascontiguousarray(arr))
        return t.to(device=ref.device, dtype=ref.dtype)
    return np.asarray(arr, dtype=(np.float32 if is_float else np.int32), order="C")


def _export_to_glb(mesh, vertices, faces, out_path: Path, remesh: bool,
                   remesh_band: int = 1, remesh_project: int = 0) -> None:
    """Delegate to o_voxel.postprocess.to_glb with our geometry and the original
    attr volume (bake happens here, AFTER geometry)."""
    import o_voxel  # type: ignore
    v = _to_like(vertices, mesh.vertices, is_float=True)
    f = _to_like(faces, mesh.faces, is_float=False)

    # CIRCULAR must-fix P0 (2026-07-30 H200 review): attrs/coords passed raw let
    # the bake dispatch off-GPU (H200: 15+ min at 0% GPU util with weights in
    # VRAM). Align device to the geometry but PRESERVE dtype — coords are integer
    # voxel indices; casting them to vertices' float dtype corrupts them.
    def _on_geo_device(x):
        ref = mesh.vertices
        if not hasattr(ref, "device"):
            return x
        import torch
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(
            np.ascontiguousarray(x))
        return t.to(device=ref.device)  # dtype untouched
    av = _on_geo_device(mesh.attrs)
    cd = _on_geo_device(mesh.coords)
    try:
        print(f"[worker] export inputs: verts={getattr(v, 'device', 'np')} "
              f"attrs={getattr(av, 'device', 'np')}/{getattr(av, 'dtype', '?')} "
              f"coords={getattr(cd, 'device', 'np')}/{getattr(cd, 'dtype', '?')}",
              flush=True)
    except Exception:
        pass

    kwargs = dict(
        vertices=v,
        faces=f,
        attr_volume=av,
        coords=cd,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(os.environ.get("DECIMATE_TARGET", "200000")),
        texture_size=int(os.environ.get("TEXTURE_RES", "2048")),
        remesh=remesh,
        verbose=False,
    )
    if remesh:
        kwargs["remesh_band"] = int(remesh_band)
        kwargs["remesh_project"] = int(remesh_project)

    glb_obj = o_voxel.postprocess.to_glb(**kwargs)
    # PNG textures, not WebP: some Pillow builds lack _webp.HAVE_WEBPANIM
    # (raises AttributeError, not TypeError) and WebP round-trips fail in
    # many glTF importers anyway (circular-loop finding #5).
    for _kw in ({"extension_webp": False}, {}):
        try:
            glb_obj.export(str(out_path), **_kw)
            break
        except (TypeError, AttributeError):
            continue
    else:
        glb_obj.export(str(out_path))


def _write_geo_meta(stem: str, meta: dict, tmp_dir: Path) -> None:
    try:
        (tmp_dir / f"{stem}.geo.json").write_text(
            json.dumps(meta, default=str), encoding="utf-8"
        )
        ws = Path(os.environ.get("WORKSPACE", "/workspace")) / "geo_meta"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / f"{stem}.geo.json").write_text(
            json.dumps(meta, default=str), encoding="utf-8"
        )
    except Exception as wexc:
        print(f"[worker] geo meta write failed: {wexc}", flush=True)


def image_to_glb_bytes(image_path: Path) -> tuple:
    """TRELLIS forward + solidify-then-bake. Returns (glb_bytes, meta). On
    solidify failure falls back to the proven band=1 remesh. Meta is also
    written to /workspace/geo_meta for batch pull before teardown."""
    pipe = load_pipeline()
    # STAGE TIMERS (circular must-fix P0, 2026-07-30 H200 review): without
    # per-stage clocks the 0%-GPU hang was unattributable (solidify vs bake vs
    # forward). Printed per render; abandoned jobs still leave these on stdout
    # even though geo_meta is only written after export.
    _t0 = time.time()
    # Background removal via BiRefNet (MIT) — see birefnet_cutout for the
    # license guard; the RGBA cutout makes TRELLIS skip its RMBG-2.0 step.
    image = birefnet_cutout(image_path)
    _t_biref = time.time()
    mark_progress("birefnet")

    import torch
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        try:
            torch.set_num_threads(int(os.environ.get("CPU_THREADS", "8")))
            torch.set_num_interop_threads(1)
        except Exception:
            pass  # interop can only be set once per process

    with torch.no_grad():
        try:
            outputs = pipe.run(image)
        except TypeError:
            # pipeline API variants differ: some expose .run(), others __call__
            outputs = pipe(image)
    _t_fwd = time.time()
    mark_progress("forward")

    mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    try:
        _dev = lambda x: (f"{type(x).__name__}:"
                          f"{getattr(x, 'device', 'np')}/{getattr(x, 'dtype', '?')}")
        print(f"[worker] stages: birefnet={_t_biref - _t0:.1f}s "
              f"forward={_t_fwd - _t_biref:.1f}s | mesh devices: "
              f"verts={_dev(mesh.vertices)} attrs={_dev(mesh.attrs)} "
              f"coords={_dev(mesh.coords)}"
              if hasattr(mesh, "attrs") else
              f"[worker] stages: birefnet={_t_biref - _t0:.1f}s "
              f"forward={_t_fwd - _t_biref:.1f}s | mesh=legacy", flush=True)
    except Exception:
        pass

    tmp_dir = Path(tempfile.mkdtemp())
    out_path = tmp_dir / f"{image_path.stem}.glb"
    meta = {
        "geometry_path": None,
        "solidify_stats": None,
        "fallback_reason": None,
        "vram_peak_mb_after_forward": _torch_vram_mb(),
        "texture_size": TEX_SIZE,
        "decimate_target": DECIMATE,
    }

    # If no o_voxel-style mesh, use legacy export and return
    has_ovoxel = mesh is not None and all(
        hasattr(mesh, a)
        for a in ("vertices", "faces", "attrs", "coords", "layout", "voxel_size")
    )
    if not has_ovoxel:
        for attr in ("export_glb", "save_glb", "export", "save"):
            fn = getattr(mesh, attr, None)
            if callable(fn):
                try:
                    fn(str(out_path))
                    break
                except TypeError:
                    try:
                        fn(str(out_path), format="glb")
                        break
                    except Exception:
                        continue
        if not out_path.is_file():
            raise RuntimeError("mesh export produced no GLB")
        meta["geometry_path"] = "legacy_mesh_export"
        _write_geo_meta(image_path.stem, meta, tmp_dir)
        data = out_path.read_bytes()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return data, meta

    # Pre-cap raw extract (cheap; neutral per circular — leave in place)
    if hasattr(mesh, "simplify") and callable(mesh.simplify):
        try:
            mesh.simplify(16_777_216)
        except Exception:
            pass

    # --------- PRIMARY: solidify then bake (remesh=False) --------------------
    solidify_ok = False
    try:
        v_raw = mesh.vertices
        f_raw = mesh.faces
        if hasattr(v_raw, "detach"):
            v_raw = v_raw.detach().cpu().numpy()
        if hasattr(f_raw, "detach"):
            f_raw = f_raw.detach().cpu().numpy()

        v_sol, f_sol, s_stats = solidify_mesh(
            np.asarray(v_raw),
            np.asarray(f_raw),
            aabb_min=(-0.5, -0.5, -0.5),
            aabb_max=(0.5, 0.5, 0.5),
            res=SOLIDIFY_RES,
            close_rad=SOLIDIFY_CLOSE,
            smooth_iters=SOLIDIFY_SMOOTH,
            target_faces=int(os.environ.get("DECIMATE_TARGET", "200000")),
        )
        meta["solidify_stats"] = s_stats

        del v_raw, f_raw
        gc.collect()

        _t_exp0 = time.time()
        # Reclaim VRAM right before export (circular P1): export is the peak and
        # the 24GB cards OOM here. Cheap, best-effort, never fatal.
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
        mark_progress("pre-export-solid")
        _export_to_glb(
            mesh,
            vertices=v_sol,
            faces=f_sol,
            out_path=out_path,
            remesh=False,  # critical: do NOT re-run o_voxel narrow-band remesh
        )
        meta["geometry_path"] = "solidify_v1"
        meta["export_s"] = round(time.time() - _t_exp0, 1)
        solidify_ok = True
        mark_progress("solidify+export")
        print(
            f"[worker] solidify_v1 OK faces={s_stats.get('output_faces')} "
            f"wt={s_stats.get('watertight')} t={s_stats.get('elapsed_s')}s "
            f"vram_mb={_torch_vram_mb()}",
            flush=True,
        )
    except Exception as exc:
        meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        meta["solidify_traceback"] = traceback.format_exc()
        print(f"[worker] solidify FAILED -> band={FALLBACK_BAND} fallback: {exc}", flush=True)
        traceback.print_exc()

    # --------- FALLBACK: proven band=1 o_voxel remesh ------------------------
    if not solidify_ok:
        # Pass the pipeline's own tensors UNTOUCHED — byte-identical to the
        # pre-patch working call (attempt-3 lesson: numpy-ifying poisoned the
        # fallback with the same .cuda() crash as the primary).
        _t_exp0 = time.time()
        # Fallback export had NO progress mark at all (circular review P0-1,
        # both reviewers): a slow fallback export was silently unprotected.
        mark_progress("pre-export-fallback")
        _export_to_glb(
            mesh,
            vertices=mesh.vertices,
            faces=mesh.faces,
            out_path=out_path,
            remesh=True,
            remesh_band=FALLBACK_BAND,
            remesh_project=FALLBACK_PROJECT,
        )
        mark_progress("export-done-fallback")
        meta["geometry_path"] = f"fallback_band{FALLBACK_BAND}"
        meta["export_s"] = round(time.time() - _t_exp0, 1)
        print(f"[worker] fallback band={FALLBACK_BAND} export OK", flush=True)

    meta["vram_peak_mb_end"] = _torch_vram_mb()
    _write_geo_meta(image_path.stem, meta, tmp_dir)

    if not out_path.is_file():
        raise RuntimeError("export produced no GLB")
    # Read the bytes, then DROP the temp dir. Circular review, both reviewers:
    # every render leaked its mkdtemp() directory - a multi-MB GLB plus textures
    # per asset, ~700/day, never cleaned. The geo meta also lives in
    # /workspace/geo_meta, so nothing is lost by removing the temp copy.
    data = out_path.read_bytes()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return data, meta


class Handler(BaseHTTPRequestHandler):
    server_version = "TrellisSpikeWorker/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[worker] " + (fmt % args) + "\n")

    def _json(self, code: int, obj: dict, extra_headers: dict = None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                try:
                    self.send_header(str(k), str(v))
                except Exception:
                    continue
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, code: int, data: bytes, content_type: str, filename: str,
               extra_headers: dict = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        # Connection: close on the success path too (circular P0-3): the 429 path
        # already closed; long-lived keep-alive sockets complicate failure
        # detection and let a half-dead ssh forward hide.
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                try:
                    self.send_header(str(k), str(v))
                except Exception:
                    continue
        self.end_headers()
        # Stream in chunks and flush instead of one giant wfile.write. With a
        # per-request socket timeout set (IO_TIMEOUT_S), a dead forward now
        # surfaces as a socket.timeout on a small write - caught in do_POST and
        # treated like BrokenPipe - rather than blocking forever with _GEN_LOCK
        # held ("busy forever, GPU 0%").
        mv = memoryview(data)
        chunk = 262144
        for off in range(0, len(mv), chunk):
            self.wfile.write(mv[off:off + chunk])
        self.wfile.flush()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/health"):
            ready = _pipeline is not None
            busy_since = _GEN_BUSY_SINCE
            # "busy" lets the driver poll cheaply after a 429 instead of
            # re-POSTing a multi-MB matte just to be refused again. The
            # "pipeline_loaded" key is retained verbatim: the driver's
            # healthy() asserts the substring "pipeline" in the body.
            self._json(
                200,
                {
                    "ok": True,
                    "model": MODEL_ID,
                    "pipeline_loaded": ready,
                    "load_state": _get_load_state(),  # #7: loading | ready | dead
                    "weights": str(WEIGHTS_DIR),
                    "busy": bool(busy_since),
                    "busy_slot": _GEN_BUSY_NAME if busy_since else "",
                    "busy_elapsed_s": (round(time.time() - busy_since, 1)
                                       if busy_since else 0.0),
                    "host": socket.gethostname(),
                    "build": WORKER_BUILD,
                },
            )
            return
        if path == "/ready":
            # #7 readiness endpoint (A5 fast-follow): a dedicated, cheap signal so a caller can
            # distinguish LOADING (model reloading, back off) from READY (feed now) from DEAD
            # (load failed; supervisor is restarting). 200 only when READY, else 503 so a plain
            # HTTP status check suffices without parsing the body.
            _state = _get_load_state()
            code = 200 if (_state == "ready" and _pipeline is not None) else 503
            self._json(code, {
                "ok": code == 200,
                "load_state": _state,
                "busy": bool(_GEN_BUSY_SINCE),
                "build": WORKER_BUILD,
            })
            return
        if path.startswith("/geo_meta/"):
            # The driver's try_fetch_geo() has always GET'd this path to recover
            # a lost X-Geometry-Path, but no route existed, so it could only
            # ever return "unrecorded" (circular review, additional finding).
            # Read-only, basename-only: no traversal out of the geo_meta dir.
            ws = Path(os.environ.get("WORKSPACE", "/workspace")) / "geo_meta"
            target = ws / Path(path[len("/geo_meta/"):]).name
            try:
                data = target.read_bytes()
            except OSError as exc:
                self._json(404, {"ok": False, "error": f"no geo meta: {exc}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/setuplog":
            log_path = Path(os.environ.get("WORKSPACE", "/workspace")) / "setup.log"
            try:
                data = log_path.read_bytes()[-16384:]
            except OSError as exc:
                self._json(404, {"ok": False, "error": f"setup.log unavailable: {exc}"})
                return
            self._bytes(200, data, "text/plain", "setup.log")
            return
        self._json(404, {"ok": False, "error": "not found"})

    def _error_500(self, name: str, exc: Exception):
        # BLINDNESS FIX (2026-07-19): round 1 guessed the layer because the
        # error handler returned only str(exc) (a one-line message) and the
        # full traceback lived only in the pod's worker log, which teardown
        # discarded. Now capture the FULL traceback three ways so it always
        # survives:  (1) worker stderr,  (2) a per-asset .err file under
        # /workspace/worker_errs/ (pulled before teardown),  (3) the JSON
        # 500 body itself -> the pod-side `curl -o on_out/$n.glb` writes it
        # INTO the stub .glb, so the traceback rides home with the batch pull
        # even if nothing else is fetched. Never let a diagnosis be lost.
        tb = traceback.format_exc()
        traceback.print_exc()
        try:
            errdir = Path(os.environ.get("WORKSPACE", "/workspace")) / "worker_errs"
            errdir.mkdir(parents=True, exist_ok=True)
            (errdir / f"{name}.err").write_text(tb, encoding="utf-8")
        except Exception as werr:
            sys.stderr.write(f"[worker] could not write .err file: {werr}\n")
        self._json(500, {"ok": False, "error": str(exc), "traceback": tb})

    def do_POST(self):
        global _GEN_BUSY_SINCE, _GEN_BUSY_NAME
        # Per-request socket timeout (circular P0-3): bounds BOTH the body read
        # and every wfile.write below. A half-dead ssh -L forward can no longer
        # stall the handler indefinitely while it holds _GEN_LOCK.
        try:
            self.connection.settimeout(IO_TIMEOUT_S)
        except Exception:
            pass
        path = urlparse(self.path).path
        if path != "/generate":
            self._json(404, {"ok": False, "error": "not found"})
            return
        name = "asset"
        # --- phase 1: drain + parse the request body ------------------------
        # ALWAYS completed before the serial guard is consulted. Answering 429
        # without reading the body would leave the client's multipart upload
        # half-sent and it would surface as a connection reset, not a 429.
        try:
            ctype = self.headers.get("Content-Type", "")
            tmp = Path(tempfile.mkdtemp())
            img_path = tmp / "input.png"

            if "multipart/form-data" in ctype:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": ctype,
                    },
                )
                file_item = form["file"] if "file" in form else None
                if file_item is None:
                    self._json(400, {"ok": False, "error": "missing file field"})
                    return
                if getattr(file_item, "filename", None):
                    name = Path(file_item.filename).stem or name
                img_path.write_bytes(file_item.file.read())
            else:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                name = str(payload.get("name") or name)
                b64 = payload.get("image_b64") or payload.get("image")
                if not b64:
                    self._json(400, {"ok": False, "error": "missing image_b64"})
                    return
                img_path.write_bytes(base64.b64decode(b64))
        except Exception as exc:
            self._error_500(name, exc)
            return

        # --- poison-input quarantine (circ review 2026-08-08, A5 P0-6) -------
        # An input that previously HARD-faulted is refused fast (422) before the GPU
        # guard, so a re-feed cannot restart-storm the pod. Cleared by deleting
        # /workspace/quarantine.json. OOM inputs are NOT quarantined (transient).
        if _is_quarantined(img_path):
            sys.stderr.write(f"[worker] 422 QUARANTINED input refused: {name}\n")
            self._json(422, {"ok": False,
                             "error": "input quarantined (previously hard-faulted)",
                             "name": name},
                       extra_headers={"X-Worker-Build": WORKER_BUILD})
            shutil.rmtree(img_path.parent, ignore_errors=True)
            return

        # --- (b) TTL SOFT-quarantine (A5 fast-follow) -----------------------
        # An input that has repeatedly hit RESOURCE PRESSURE (too big to fit right now) is
        # a GOOD asset, so it is NOT permanently quarantined - but re-feeding it in a tight
        # loop just restart-storms nothing and burns cycles. Refuse it with 429 + Retry-After
        # for a TTL window so the feeder backs off, then it auto-clears. Never permanent.
        _soft_until = _soft_quarantined_until(img_path)
        if _soft_until:
            _wait = max(1, int(_soft_until - time.time()))
            sys.stderr.write(f"[worker] 429 SOFT-QUARANTINE {name}: repeated resource-pressure, "
                             f"backoff {_wait}s\n")
            self._json(429, {"ok": False,
                             "error": "input soft-quarantined (repeated resource-pressure); retry later",
                             "name": name, "retry_after_s": _wait},
                       extra_headers={"X-Worker-Build": WORKER_BUILD, "Retry-After": str(_wait)})
            shutil.rmtree(img_path.parent, ignore_errors=True)
            return

        # --- phase 2: SERIAL-RENDER GUARD -----------------------------------
        # Non-blocking: a busy worker refuses immediately with 429 so the driver
        # can wait and re-offer, instead of a second render stacking on the GPU.
        if not _GEN_LOCK.acquire(False):
            elapsed = (time.time() - _GEN_BUSY_SINCE) if _GEN_BUSY_SINCE else 0.0
            retry_after = 60 if elapsed < 60 else 30
            sys.stderr.write(
                f"[worker] 429 BUSY: refusing {name}; {_GEN_BUSY_NAME or '?'} "
                f"has been rendering {elapsed:.0f}s\n")
            self._json(
                429,
                {"ok": False, "error": "busy: a render is already in flight",
                 "busy_slot": _GEN_BUSY_NAME, "busy_elapsed_s": round(elapsed, 1)},
                extra_headers={"Retry-After": str(retry_after),
                               "X-Busy-Elapsed-S": f"{elapsed:.1f}",
                               "X-Busy-Slot": _GEN_BUSY_NAME or "",
                               "X-Worker-Build": WORKER_BUILD,
                               "Connection": "close"},
            )
            shutil.rmtree(img_path.parent, ignore_errors=True)
            return

        # --- phase 3: the GPU section (guard held) --------------------------
        _GEN_BUSY_SINCE, _GEN_BUSY_NAME = time.time(), name
        mark_progress("accepted")
        sent = False
        _need_hard_restart = False
        # VRAM ledger (A5 fast-follow): per-job telemetry, best-effort, NEVER breaks a render.
        _oom_retries = 0
        _outcome = "ok"
        _asset_class = _classify_asset(name)
        _ledger = _vram_ledger_start()  # resets torch peak stats + samples free_at_start
        try:
            try:
                glb, geo_meta = image_to_glb_bytes(img_path)
            except Exception as _oom_exc:
                # circ review 2026-08-08 (Grok+GPT+A5 P0-1): a VAE-decoder OOM usually
                # does NOT corrupt the CUDA context. Free VRAM and retry ONCE in-process
                # (staying alive) rather than escalating to process death. A 2nd OOM
                # propagates to the outer handler -> 500, and we STILL stay alive.
                # (a) A5 fast-follow: retry on ANY transient RESOURCE-PRESSURE fault - plain
                # OOM OR a cuBLAS/cuDNN ALLOC_FAILED. Both are a good input losing a VRAM race,
                # not context corruption, so free+retry once instead of a MINUTES-long restart.
                if _is_resource_pressure(_oom_exc):
                    sys.stderr.write(
                        f"[worker] resource-pressure on {name}: freeing VRAM + retrying once "
                        f"({type(_oom_exc).__name__})\n")
                    sys.stderr.flush()
                    _oom_retries += 1
                    _free_vram()
                    mark_progress("oom-retry")  # reset watchdog stall timer for attempt 2
                    glb, geo_meta = image_to_glb_bytes(img_path)
                else:
                    raise
            headers = {
                "X-Geometry-Path": geo_meta.get("geometry_path", "unknown"),
                "X-VRAM-Peak-MB": str(geo_meta.get("vram_peak_mb_end", -1)),
                "X-Worker-Build": WORKER_BUILD,
            }
            sent = True
            self._bytes(200, glb, "model/gltf-binary", f"{name}.glb",
                        extra_headers=headers)
            _mark_success()  # supervisor resets restart-backoff only on a real render (P0-3)
            _outcome = "oom_retry_ok" if _oom_retries else "ok"
        except (BrokenPipeError, ConnectionResetError, socket.timeout,
                TimeoutError) as exc:
            # The client (or the ssh tunnel) went away, or the send timed out on
            # a half-dead forward (circular P0-3). The render itself succeeded
            # and was paid for; do NOT run _error_500, which would write a second
            # response onto a dead socket and file a bogus worker_errs entry for
            # a good asset (circular review P1-2). Treat a send-path timeout the
            # SAME as BrokenPipe: log and return, never a 500.
            sys.stderr.write(
                f"[worker] client disconnected/timed out "
                f"{'while sending' if sent else 'during'} "
                f"{name}: {type(exc).__name__} - render completed, response lost\n")
        except OSError as exc:
            # EPIPE / ECONNRESET / ETIMEDOUT arriving as a bare OSError - same
            # class of dead-socket failure as above (circular P0-3). Anything
            # else is a genuine 500.
            if exc.errno in (errno.EPIPE, errno.ECONNRESET, errno.ETIMEDOUT):
                sys.stderr.write(
                    f"[worker] socket error {'while sending' if sent else 'during'} "
                    f"{name}: errno={exc.errno} {exc} - render completed, response lost\n")
            else:
                self._error_500(name, exc)
                _outcome = "error_500"
        except Exception as exc:
            self._error_500(name, exc)
            # circ review 2026-08-08 (Grok+GPT+A5 P0-1/P0-2/P0-6): OOM is non-fatal
            # (retried above; a 2nd OOM lands here and we STAY ALIVE - usually transient
            # VRAM contention with the char/money-line, not poison). A HARD/sticky CUDA
            # fault corrupts the context and would crash the process uncatchably on the
            # NEXT job (the 000 cascade). Quarantine the poison input and flag a clean
            # restart to run AFTER the finally cleanup (NOT os._exit here - that skips
            # finally and leaks the temp dir + serial lock).
            if _is_hard_cuda_error(exc):
                # circ review 2026-08-08 (Grok+GPT, A5 P1): a hard fault ALWAYS restarts
                # (context may be corrupt), but only DETERMINISTIC poison quarantines the
                # input. NOTE: after (a), a cuBLAS/cuDNN ALLOC_FAILED is classed as resource
                # pressure (below), NOT hard - so this branch is genuine corruption only.
                _need_hard_restart = True
                _outcome = "hard_fault"
                if _is_poison(exc):
                    _quarantine_add(img_path, name, str(exc)[:200])
                    _fault_reason = "deterministic poison - quarantined"
                else:
                    # Non-deterministic hard fault (e.g. cuBLAS/cuDNN alloc failure that we now
                    # restart on rather than retry, per circ): the input is likely GOOD, so do NOT
                    # permanently quarantine - but strike the TTL soft-quarantine so a re-feed that
                    # keeps faulting backs off instead of restart-storming the pod (circ (b)).
                    _soft_quarantine_note(img_path, name, str(exc)[:160])
                    _fault_reason = ("non-deterministic hard fault - NOT quarantined "
                                     "(good input preserved); soft-quarantine struck")
                sys.stderr.write(
                    f"[worker] HARD CUDA fault on {name}: {type(exc).__name__}: "
                    f"{str(exc)[:160]} - {_fault_reason}; will exit(77) after cleanup\n")
                sys.stderr.flush()
            elif _is_resource_pressure(exc):
                # (a)+(b) A5 fast-follow: the in-process retry above still lost the VRAM race
                # -> 500 but STAY ALIVE (context is fine, no restart). (b) bound repeats: note
                # it in the TTL soft-quarantine so a persistently-too-big input backs off for a
                # while, WITHOUT a permanent poison quarantine (it is still a GOOD asset).
                _outcome = "oom_500"
                _soft_quarantine_note(img_path, name, str(exc)[:160])
            else:
                _outcome = "error_500"
        finally:
            _GEN_BUSY_SINCE, _GEN_BUSY_NAME = 0.0, ""
            _GEN_LOCK.release()
            # Post-render VRAM hygiene (circular P0-4): free the cache and
            # surface any deferred CUDA fault NOW rather than poisoning the next
            # render. Runs after the lock is released so the next job/watchdog is
            # never blocked. Best-effort; only where torch is importable.
            try:
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.synchronize()
                    except Exception as sync_exc:
                        sys.stderr.write(
                            f"[worker] post-render cuda.synchronize surfaced a "
                            f"fault: {type(sync_exc).__name__}: {sync_exc}\n")
            except Exception:
                pass
            # Phase-1 upload temp dir (circular review P1-2/P3 leak).
            try:
                shutil.rmtree(img_path.parent, ignore_errors=True)
            except Exception:
                pass
            # VRAM ledger row (A5 fast-follow): recorded for EVERY outcome incl. OOM/hard-fault
            # (in finally so it logs even on exit-after-cleanup), so "do we need more VRAM?" is
            # answered from data, not crashes. Best-effort; never raises.
            _vram_ledger_write(name, _asset_class, _ledger, _oom_retries, _outcome)
            # Operator-requested post-render settle. synchronize() above already blocks
            # until the free completes, so this is belt-and-suspenders for async
            # allocator settling; default 0 (off), env POST_RENDER_SETTLE_S.
            if POST_RENDER_SETTLE_S > 0:
                time.sleep(POST_RENDER_SETTLE_S)
            # A5 P0-2: hard-CUDA restart runs ONLY after full cleanup above (lock
            # released, temp removed, cache freed, logs flushed) - never a bare
            # os._exit mid-except that would leak them.
            if _need_hard_restart:
                sys.stderr.write(
                    "[worker] cleanup complete - exiting(77) for supervisor restart\n")
                sys.stderr.flush()
                os._exit(77)


# --- crash-resilience helpers (circ review 2026-08-08: Grok + GPT + A5) --------------
POST_RENDER_SETTLE_S = float(os.environ.get("POST_RENDER_SETTLE_S", "0") or "0")
_SUCCESS_MARKER = Path("/workspace/.last_success")
_QUARANTINE_PATH = Path("/workspace/quarantine.json")


def _mark_success() -> None:
    """Touch a marker after a completed render so the supervisor resets its restart
    backoff on real SUCCESS, not on elapsed wall-time (A5 P0-3)."""
    try:
        _SUCCESS_MARKER.write_text(str(time.time()))
    except Exception:
        pass


def _input_hash(img_path) -> str:
    try:
        import hashlib
        return hashlib.sha1(Path(img_path).read_bytes()).hexdigest()
    except Exception:
        return ""


def _atomic_write_json(path, obj) -> None:
    """Write JSON atomically (temp in the same dir + fsync + os.replace) so a crash or os._exit
    mid-write can never leave a truncated file that _load_* then fails to parse (circ 2026-08-08).
    Falls back to a plain write only if the atomic path itself errors. Best-effort; never raises."""
    try:
        with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False,
                                         encoding="utf-8") as tf:
            json.dump(obj, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
            tmp = tf.name
        os.replace(tmp, str(path))
    except Exception:
        try:
            path.write_text(json.dumps(obj, indent=2))
        except Exception:
            pass


def _load_quarantine() -> dict:
    try:
        return json.loads(_QUARANTINE_PATH.read_text())
    except Exception:
        return {}


def _is_quarantined(img_path) -> bool:
    h = _input_hash(img_path)
    return bool(h) and h in _load_quarantine()


def _quarantine_add(img_path, name, reason) -> None:
    """Record a HARD-faulting input by content hash so a re-feed is refused (A5 P0-6).
    Only hard CUDA faults quarantine; transient OOMs do not."""
    try:
        h = _input_hash(img_path)
        if not h:
            return
        q = _load_quarantine()
        q[h] = {"name": name, "reason": reason, "ts": time.strftime("%Y-%m-%dT%H%M%SZ")}
        _atomic_write_json(_QUARANTINE_PATH, q)
    except Exception:
        pass


def _is_oom(exc) -> bool:
    """True if exc is a CUDA out-of-memory — recoverable in-process via empty_cache +
    retry (A5 P0-1). Does NOT require a process restart."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    s = str(exc).lower()
    return "out of memory" in s or "cuda oom" in s


def _is_resource_pressure(exc) -> bool:
    """True ONLY for a clean CUDA OOM that is safe to retry in-process.

    A5 fast-follow (a) originally asked to also treat cuBLAS/cuDNN ALLOC_FAILED as
    retry-in-process. The Grok+GPT circ (2026-08-08) REJECTED that: an alloc failure raised from
    inside a cuBLAS/cuDNN kernel can leave the CUDA context sticky/corrupt, so retrying re-uses a
    bad context (deterministic re-fail or silent garbage). A plain torch.OutOfMemoryError is raised
    at the Python allocation boundary with the context INTACT, so it alone is retry-safe. cuBLAS/
    cuDNN alloc failures therefore fall through to _is_hard_cuda_error -> restart (fresh context),
    and the restart-storm they could cause is bounded by the soft-quarantine on the non-poison
    hard-fault path. (A5 to confirm this safety override of item (a).)"""
    return _is_oom(exc)


def _is_hard_cuda_error(exc) -> bool:
    """True if exc is a sticky/unrecoverable CUDA fault (NOT transient resource pressure).
    Such a fault corrupts the context; the only reliable fix is a process restart. Kept narrow
    so a recoverable OOM / cuBLAS-alloc failure is never misclassified as fatal (A5 P0-1, and
    A5 fast-follow (a): resource-pressure allocations are excluded here so they free+retry, not
    restart)."""
    if _is_resource_pressure(exc):
        return False
    s = str(exc).lower()
    return any(k in s for k in (
        "cuda error", "device-side assert", "illegal memory access",
        "cublas", "cudnn", "[cumesh] cuda", "misaligned address", "an illegal"))


def _is_poison(exc) -> bool:
    """True ONLY for DETERMINISTIC, input-triggered CUDA faults that recur on the SAME input
    every time, so the input must be quarantined (A5 P0-6). Poison is a POSITIVE allow-list:
    a transient resource-pressure fault (plain OOM, cuBLAS/cuDNN ALLOC_FAILED) never matches,
    so a good input is never permanently quarantined (A5 P1 decouple). Because it is a positive
    match, a poison message that ALSO contains 'alloc' is still correctly quarantined - closing
    the precedence hole the circ review (Grok+GPT 2026-08-08) flagged in an early-exclusion design.
    NOTE: _is_hard_cuda_error still restarts on non-alloc cublas/cudnn/generic 'cuda error';
    this only governs QUARANTINE, never restart."""
    if _is_resource_pressure(exc):
        return False
    s = str(exc).lower()
    return any(k in s for k in (
        "device-side assert", "illegal memory access", "[cumesh] cuda",
        "misaligned address", "an illegal"))


# --- (b) TTL soft-quarantine + VRAM ledger + asset classifier (A5 fast-follow) -------
_SOFT_QUARANTINE_PATH = Path("/workspace/soft_quarantine.json")
SOFT_QUARANTINE_THRESHOLD = int(os.environ.get("SOFT_QUARANTINE_THRESHOLD", "3"))
SOFT_QUARANTINE_TTL_S = int(os.environ.get("SOFT_QUARANTINE_TTL_S", "1800"))
_VRAM_LEDGER_PATH = Path(os.environ.get("VRAM_LEDGER_PATH", "/workspace/vram_ledger.csv"))
_VRAM_LEDGER_HEADER = ("ts_utc,slug,asset_class,torch_peak_reserved_gib,smi_used_peak_gib,"
                       "free_at_start_gib,oom_retries,outcome")


def _load_soft_quarantine() -> dict:
    try:
        return json.loads(_SOFT_QUARANTINE_PATH.read_text())
    except Exception:
        return {}


def _soft_quarantine_note(img_path, name, reason) -> None:
    """(b) Record a RESOURCE-PRESSURE 500 for this input. After SOFT_QUARANTINE_THRESHOLD hits
    inside one TTL window, stamp an `until` so re-feeds get a 429 backoff for SOFT_QUARANTINE_TTL_S.
    This is TEMPORARY (a good asset that is momentarily too big), never a permanent poison
    quarantine. Best-effort; never raises."""
    try:
        h = _input_hash(img_path)
        if not h:
            return
        q = _load_soft_quarantine()
        now = time.time()
        e = q.get(h) or {"count": 0, "first_ts": now, "name": name}
        # roll the window: if the last strike is older than the TTL, start counting fresh
        if now - e.get("first_ts", now) > SOFT_QUARANTINE_TTL_S:
            e = {"count": 0, "first_ts": now, "name": name}
        e["count"] = int(e.get("count", 0)) + 1
        e["last_reason"] = reason
        e["last_ts"] = now
        if e["count"] >= SOFT_QUARANTINE_THRESHOLD:
            e["until"] = now + SOFT_QUARANTINE_TTL_S
        q[h] = e
        _atomic_write_json(_SOFT_QUARANTINE_PATH, q)
    except Exception:
        pass


def _soft_quarantined_until(img_path) -> float:
    """Return the epoch until which this input is soft-quarantined, or 0 if not (or expired)."""
    try:
        h = _input_hash(img_path)
        if not h:
            return 0.0
        e = _load_soft_quarantine().get(h)
        if not e:
            return 0.0
        until = float(e.get("until", 0) or 0)
        return until if until > time.time() else 0.0
    except Exception:
        return 0.0


def _classify_asset(name) -> str:
    """Coarse asset class from the slug, for per-class VRAM profiling in the ledger. Best-effort;
    'unknown' when nothing matches. (Data-driven off-pod routing later keys off this.)"""
    try:
        s = str(name).lower()
        if any(k in s for k in ("char", "npc", "clown", "creature", "figure", "person", "mf_")):
            return "character"
        if any(k in s for k in ("canopy", "fern", "bush", "shrub", "foliage", "tree", "willow",
                                "conifer", "cedar", "frond", "leaf")):
            return "foliage"
        if any(k in s for k in ("rock", "granite", "basalt", "boulder", "stone", "cliff",
                                "terrain", "log", "mound", "mushroom", "chanterelle")):
            return "nature"
        if any(k in s for k in ("prop", "crate", "barrel", "sign", "tool", "weapon")):
            return "prop"
        return "unknown"
    except Exception:
        return "unknown"


def _smi_used_gib() -> float:
    """Best-effort whole-card memory.used in GiB via nvidia-smi. -1.0 if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        return round(int(out.stdout.strip().splitlines()[0]) / 1024.0, 2)
    except Exception:
        return -1.0


def _vram_ledger_start() -> dict:
    """START hook: reset torch peak stats and sample free_at_start. Returns a small dict passed
    to _vram_ledger_write. Best-effort; never raises."""
    d = {"free_at_start_gib": -1.0}
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            free_b, _total_b = torch.cuda.mem_get_info()
            d["free_at_start_gib"] = round(free_b / (1024 ** 3), 2)
    except Exception:
        pass
    return d


def _vram_ledger_write(name, asset_class, start, oom_retries, outcome) -> None:
    """END hook (called in finally): append one row. torch_peak = was the JOB big; smi_used_peak
    = was the CARD crowded (co-residency). MINIMAL smi variant: sampled once at job end (noted to
    A5). Telemetry MUST NEVER break a render -> fully swallowed. Writes header once."""
    try:
        torch_peak = -1.0
        try:
            import torch
            if torch.cuda.is_available():
                torch_peak = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)
        except Exception:
            pass
        row = "{ts},{slug},{cls},{tp},{smi},{fs},{r},{o}".format(
            ts=time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()),
            slug=str(name).replace(",", "_"),
            cls=str(asset_class).replace(",", "_"),
            tp=torch_peak, smi=_smi_used_gib(),
            fs=start.get("free_at_start_gib", -1.0),
            r=int(oom_retries), o=str(outcome).replace(",", "_"))
        new = not _VRAM_LEDGER_PATH.exists()
        with _VRAM_LEDGER_PATH.open("a", encoding="utf-8") as fh:
            if new:
                fh.write(_VRAM_LEDGER_HEADER + "\n")
            fh.write(row + "\n")
    except Exception:
        pass


def _free_vram() -> None:
    """Best-effort VRAM reclaim (gc + empty_cache + synchronize)."""
    try:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _busy_watchdog() -> None:
    """Daemon: hard-kill the process if a render wedges while holding the guard.

    Circular P0-2. A render can get stuck with the GPU idle but _GEN_LOCK held
    forever (CPU/native solidify hang, or a blocked send on a half-dead tunnel);
    every subsequent /generate then 429s indefinitely. If _GEN_BUSY_SINCE has
    been set longer than BUSY_HARD_LIMIT_S we os._exit(77) so start_worker.sh's
    restart loop relaunches the worker (a full model reload = MINUTES, not
    seconds). NEVER fires at idle (_GEN_BUSY_SINCE == 0) nor during
    load_pipeline() - busy is only ever set inside do_POST phase 3.
    """
    prev_cpu = _proc_cpu_jiffies()
    while True:
        time.sleep(BUSY_WATCHDOG_TICK_S)
        cpu_now = _proc_cpu_jiffies()
        cpu_delta = (cpu_now - prev_cpu) if (cpu_now >= 0 and prev_cpu >= 0) else -1.0
        prev_cpu = cpu_now

        since = _GEN_BUSY_SINCE  # read once; set/cleared under do_POST
        if not since:
            continue
        now = time.time()
        elapsed = now - since
        # Nothing is even a candidate until it has held the guard too long.
        if elapsed <= BUSY_HARD_LIMIT_S:
            continue

        # Signal: has the render advanced a stage recently? This is the ONLY
        # kill gate. Circular review 2026-08-04 (both reviewers, NO-GO): process
        # CPU must NOT be AND-ed into the predicate. It is measured process-wide
        # via /proc/self/stat, so draining a multi-MB multipart POST body before
        # returning 429 - which happens constantly under driver retry - burns
        # CPU while the render thread is dead. That fails OPEN: `wedged` never
        # becomes true and Mode-2 recovery silently degrades to the ceiling,
        # strictly worse than the pure timer this replaced. CPU is now logged as
        # a DIAGNOSTIC only. Safety therefore rests on dense mark_progress()
        # coverage, including inside solidify and on every export path.
        last_prog = _GEN_PROGRESS_TS if _GEN_PROGRESS_TS > since else since
        stalled_s = now - last_prog

        over_ceiling = elapsed > BUSY_ABS_CEILING_S
        wedged = stalled_s > BUSY_STALL_LIMIT_S

        if not (wedged or over_ceiling):
            continue

        # TOCTOU re-check (circular review P0-4, both reviewers; I missed it).
        # `since` was sampled up to a full tick ago and the predicates took time
        # to evaluate. If the render completed in that window, _GEN_BUSY_SINCE
        # has been cleared or replaced by the NEXT job - killing now would take
        # down an idle worker or a healthy successor mid-render.
        if _GEN_BUSY_SINCE != since:
            continue

        why = ("absolute ceiling" if over_ceiling and not wedged
               else "no stage progress")
        sys.stderr.write(
            f"[worker] BUSY WATCHDOG: render '{_GEN_BUSY_NAME or '?'}' held the "
            f"serial guard {elapsed:.0f}s ({why}); last progress "
            f"'{_GEN_PROGRESS_WHAT or 'none'}' {stalled_s:.0f}s ago, cpu_delta="
            f"{cpu_delta:.0f} jiffies/tick - exiting(77) for start_worker.sh "
            f"restart (model reload takes minutes)\n")
        sys.stderr.flush()
        os._exit(77)


def _proc_cpu_jiffies() -> float:
    """User+system CPU jiffies consumed by THIS process (Linux /proc/self/stat).

    Used by the watchdog as a liveness signal: a render wedged on a native
    deadlock or a blocked socket burns no CPU, while a legitimately slow
    CPU-side solidify/bake burns plenty. Returns -1.0 if unreadable, which the
    watchdog treats as 'working' so a /proc hiccup can never kill a good render.
    Fields 14/15 (1-indexed) of /proc/self/stat are utime/stime; the comm field
    can contain spaces and parentheses, so split AFTER the final ')'."""
    try:
        raw = Path("/proc/self/stat").read_text()
        parts = raw[raw.rfind(")") + 1:].split()
        # after comm: state is parts[0], so utime/stime are parts[11]/parts[12]
        return float(parts[11]) + float(parts[12])
    except Exception:
        return -1.0


def _pid_is_nvidia_smi(pid: int) -> bool:
    """True if <pid> is a live nvidia-smi (Linux /proc). Used so the telemetry
    single-instance lock survives PID recycling."""
    try:
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
        return comm.startswith("nvidia-smi")
    except Exception:
        return False


def start_gpu_telemetry() -> None:
    """Best-effort nvidia-smi sampler -> /workspace/gpu_telemetry_<host>.csv.

    Closes the estate's biggest measurement blind spot (no GPU-utilization data
    anywhere; memory.reserved never logged, which is exactly the number needed
    to settle the two-workers-per-GPU question). Strictly non-fatal: any failure
    here must never stop the worker from binding its port. Hostname in the
    filename because two pods can share one network volume at /workspace.

    SINGLE-INSTANCE LOCK (circular P1): a Popen'd `nvidia-smi -l` survives the
    worker's os._exit / crash, so start_worker.sh's restart loop would otherwise
    stack a new sampler every restart. A /tmp lockfile (pod-local, NOT the shared
    /workspace volume) records the sampler PID; if it is still a live nvidia-smi
    we reuse it instead of spawning another.
    """
    if not GPU_TELEMETRY:
        return
    try:
        lock = Path("/tmp") / f"gpu_telemetry_{socket.gethostname()}.lock"
        if lock.exists():
            try:
                old_pid = int((lock.read_text().strip() or "0"))
            except Exception:
                old_pid = 0
            if old_pid > 0 and _pid_is_nvidia_smi(old_pid):
                print(f"[worker] gpu telemetry already running pid={old_pid} - "
                      f"not stacking a second sampler", flush=True)
                return
        out = Path(os.environ.get("WORKSPACE", "/workspace")) / (
            f"gpu_telemetry_{socket.gethostname()}.csv")
        fh = open(out, "a", buffering=1)
        proc = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=timestamp,utilization.gpu,utilization.memory,"
             "memory.used,memory.reserved,memory.total",
             "--format=csv,noheader,nounits", "-l", str(GPU_TELEMETRY_S)],
            stdout=fh, stderr=subprocess.DEVNULL, close_fds=True)
        try:
            lock.write_text(str(proc.pid))
        except Exception:
            pass
        print(f"[worker] gpu telemetry -> {out} every {GPU_TELEMETRY_S}s "
              f"(pid={proc.pid})", flush=True)
    except Exception as exc:
        print(f"[worker] gpu telemetry unavailable (non-fatal): {exc}", flush=True)


def main() -> int:
    start_gpu_telemetry()
    # Busy-elapsed watchdog (circular P0-2): daemon so it dies with the process.
    # Started before load_pipeline() is harmless - it only acts once a render has
    # set _GEN_BUSY_SINCE, which never happens during model load.
    threading.Thread(target=_busy_watchdog, name="busy-watchdog",
                     daemon=True).start()
    print(f"[worker] busy watchdog armed: hard limit {BUSY_HARD_LIMIT_S}s, "
          f"stall limit {BUSY_STALL_LIMIT_S}s, ceiling {BUSY_ABS_CEILING_S}s, "
          f"io timeout {IO_TIMEOUT_S}s, cpu=DIAGNOSTIC-ONLY (not a kill gate), "
          f"build {WORKER_BUILD}", flush=True)
    # Eager-load on start so /health reflects readiness after first load attempt
    try:
        load_pipeline()
        print(f"[worker] pipeline loaded; listening on 0.0.0.0:{PORT}", flush=True)
    except Exception as exc:
        # #7: eager load failed. Mark DEAD so /ready returns 503 and callers back off; the next
        # /generate re-enters load_pipeline() (which flips loading->ready on success).
        _set_load_state("dead")
        print(f"[worker] pipeline load deferred (will retry on first request): {exc}", flush=True)

    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
