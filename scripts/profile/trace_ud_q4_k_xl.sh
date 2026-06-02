#!/usr/bin/env bash
# Kernel trace UD-Q4_K_XL decode on a single MI50 (pp, devices 0).
# Phase 3 diagnosis of the K-quant gap surfaced in Phase 1
# (UD-Q4_K_XL at 0.42x Q4_0). Reports top decode kernels by wall.
set -euo pipefail

OUTDIR="${1:-/tmp/trace_ud_q4_k_xl}"
MODEL=/artefact/models/Qwen3.6-27B-UD-Q4_K_XL.gguf
source /artefact/flambeau/.env
BIN=/artefact/flambeau/target/release/flambeau
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')

mkdir -p "$OUTDIR"; rm -f "$OUTDIR"/* 2>/dev/null || true
LOG="$OUTDIR/server.log"

"$ROCPROFV3" --kernel-trace --output-directory "$OUTDIR" --output-format csv -- \
  "$BIN" serve --model "$MODEL" \
  --devices 0,2 --mesh-mode pp \
  --port "$PORT" --inflight-slots 1 --ctx-cap 2048 \
  >"$LOG" 2>&1 &
PID=$!
trap "kill -TERM $PID 2>/dev/null; wait $PID 2>/dev/null || true" EXIT

# Wait for ready.
for _ in $(seq 1 240); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "died" >&2; tail -30 "$LOG" >&2; exit 1; }
  sleep 0.5
done
echo "server up :$PORT"

# Warmup so JIT/first-call doesn't dominate.
REQ='{"model":"x","messages":[{"role":"user","content":"Continue: cat, dog, tree,"}],"max_tokens":4,"temperature":0,"seed":0}'
curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null
# Measured run: 1 prefill (8 tok) + 3 decode steps.
curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null

# Graceful SIGTERM so rocprofv3 atexit flushes the SQLite output.
kill -TERM $PID
wait $PID 2>/dev/null || true

echo
echo "=== trace files ==="
ls -la "$OUTDIR"
