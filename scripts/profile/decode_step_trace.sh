#!/usr/bin/env bash
# Capture a single-request decode-step trace under rocprofv3 for one
# stack (v2 or legacy), then dump kernel-summary CSV.
#
# Usage: decode_step_trace.sh <v2|legacy> <output-dir>
#
# Boots serve in foreground under rocprofv3 --kernel-trace, fires a
# K=4 request (1 prefill + 3 decode), shuts the server, emits the
# captured kernel CSV.

set -euo pipefail
MODE="${1:-v2}"
OUTDIR="${2:-/tmp/decode_trace_$MODE}"

if [[ -z "${ROCPROFV3:-}" ]]; then
  source /artefact/flambeau/.env
fi
[[ -x "$ROCPROFV3" ]] || { echo "ROCPROFV3 not executable: $ROCPROFV3" >&2; exit 2; }

GGUF=/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
BIN=/artefact/flambeau/target/release/flambeau
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/* 2>/dev/null || true

ENV_PREFIX=""
if [[ "$MODE" == "v2" ]]; then
  ENV_PREFIX="FLAMBEAU_V2=1"
fi

LOG="$OUTDIR/server.log"
env $ENV_PREFIX "$ROCPROFV3" --kernel-trace --hip-trace --output-directory "$OUTDIR" --output-format csv -- \
  "$BIN" serve \
    --model "$GGUF" \
    --devices 0,2 --mesh-mode pp --port "$PORT" \
    --inflight-slots 1 --ctx-cap 4096 \
    >"$LOG" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true" EXIT

# Wait for health.
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "server died, log tail:" >&2; tail -40 "$LOG" >&2; exit 1
  fi
  sleep 0.5
done
echo "$MODE serve up on :$PORT (pid $SERVER_PID)"

# Fire one decode-only run (K=4 tokens). Run warmup first so the
# captured trace isn't dominated by first-call JIT/alloc.
REQ='{"model":"x","messages":[{"role":"user","content":"Continue: cat, dog, tree,"}],"max_tokens":4,"temperature":0}'
echo "warmup..."
curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null
echo "measurement..."
RESP=$(curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ")
echo "response: $RESP"

# Stop server (this flushes rocprof trace).
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo
echo "=== trace files ==="
ls -la "$OUTDIR"
