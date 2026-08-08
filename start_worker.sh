#!/bin/bash
# TRELLIS worker SUPERVISOR — baked into the image so every pod inherits crash-resilience
# (A5 P0, circ review 2026-08-08 Grok+GPT+A5). The worker os._exit(77)s only on a HARD/
# unrecoverable CUDA fault (context corruption) after cleaning up; this loop reloads the
# model in a fresh process. OOM is handled in-process (free+retry) and does NOT reach here.
cd /workspace
SUP_LOG=/workspace/supervisor.log
WORKER_LOG=/workspace/worker.log
SUCCESS=/workspace/.last_success
LOCK=/workspace/.supervisor.lock
backoff=2; MAX=60

# P0-5: single supervisor instance (never two workers contending one GPU).
exec 9>"$LOCK"
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || { echo "[supervisor] another supervisor holds the lock; exiting" >> "$SUP_LOG"; exit 0; }
fi
echo "[supervisor] start $(date -u +%FT%TZ) pid=$$" >> "$SUP_LOG"

while true; do
  # P0-5: guarantee no stale worker survives before we launch a new one.
  pkill -f pod_worker.py 2>/dev/null && sleep 2
  before=$(stat -c %Y "$SUCCESS" 2>/dev/null || echo 0)
  start=$(date +%s)
  # P0-4: capture worker stdout/stderr (else a detached worker logs to /dev/null).
  bash /workspace/start_worker_hf.sh >> "$WORKER_LOG" 2>&1
  code=$?
  ran=$(( $(date +%s) - start ))
  after=$(stat -c %Y "$SUCCESS" 2>/dev/null || echo 0)
  # P0-3: reset backoff ONLY when a render actually completed this run (success marker
  # advanced) - NOT on elapsed wall-time, which a load-then-fast-death cycle would game.
  if [ "$after" -gt "$before" ]; then
    echo "[supervisor] worker exited code=$code after ${ran}s (completed >=1 render) - backoff reset" >> "$SUP_LOG"
    backoff=2
  else
    echo "[supervisor] worker exited code=$code after ${ran}s (NO render completed) - backoff ${backoff}s" >> "$SUP_LOG"
    backoff=$(( backoff * 2 )); [ "$backoff" -gt "$MAX" ] && backoff=$MAX
  fi
  sleep "$backoff"
done
