#!/usr/bin/env bash
# Kernel trace UD-Q2_K_XL decode on a single MI50 (pp, devices 0).
# Probes the IQ-MoE codebook-cache hypothesis: is
# indexed_moe_mmvq_iq{2_xs,3_xxs}_r2_dp4a HBM-bound on weight reads, or
# compute-bound waiting on IQ*_GRID constant-cache lookups?
set -euo pipefail

OUTDIR="${1:-/tmp/trace_ud_q2_k_xl}"
MODEL=/artefact/models/Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf
source /artefact/flambeau/.env
ROCPROFV3="${ROCPROFV3:-/opt/rocm-host/bin/rocprofv3}"
BIN=/artefact/flambeau/target/release/flambeau
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')

mkdir -p "$OUTDIR"; rm -f "$OUTDIR"/* 2>/dev/null || true
LOG="$OUTDIR/server.log"

"$ROCPROFV3" --kernel-trace --output-directory "$OUTDIR" --output-format csv -- \
  "$BIN" serve --model "$MODEL" \
  --devices 0,2,1,3 --mesh-mode pp+tp --pp-size 2 --tp-size 2 \
  --port "$PORT" --inflight-slots 1 --ctx-cap 2048 \
  >"$LOG" 2>&1 &
PID=$!
trap "kill -TERM $PID 2>/dev/null; wait $PID 2>/dev/null || true" EXIT

# 122B-A10B mmap load is several minutes.
for _ in $(seq 1 600); do
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "died" >&2; tail -30 "$LOG" >&2; exit 1; }
  sleep 1
done
echo "server up :$PORT"

REQ='{"model":"x","messages":[{"role":"user","content":"Continue: cat, dog, tree,"}],"max_tokens":4,"temperature":0,"seed":0}'
curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null
curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null

kill -TERM $PID
wait $PID 2>/dev/null || true

echo
echo "=== trace files ==="
ls -la "$OUTDIR"
