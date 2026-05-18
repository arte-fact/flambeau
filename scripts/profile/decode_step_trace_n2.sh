#!/usr/bin/env bash
# Variant of decode_step_trace.sh that fires N=2 concurrent requests
# so the multi-slot branch of v2 standard_attn / moe_ffn runs and
# trace captures the batched-attn / batched-MoE kernels.

set -euo pipefail
MODE="${1:-v2}"
OUTDIR="${2:-/tmp/decode_trace_n2_$MODE}"

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
[[ "$MODE" == "v2" ]] && ENV_PREFIX="FLAMBEAU_V2=1"

LOG="$OUTDIR/server.log"
env $ENV_PREFIX "$ROCPROFV3" --kernel-trace --hip-trace --output-directory "$OUTDIR" --output-format csv -- \
  "$BIN" serve \
    --model "$GGUF" \
    --devices 0,2 --mesh-mode pp --port "$PORT" \
    --inflight-slots 4 --decode-batch-window-us 1500 --ctx-cap 4096 \
    >"$LOG" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true" EXIT

for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died" >&2; tail -40 "$LOG" >&2; exit 1; }
  sleep 0.5
done
echo "$MODE serve up on :$PORT"

REQ='{"model":"x","messages":[{"role":"user","content":"Continue: cat, dog, tree, bird, fish, mountain, star, moon, sun,"}],"max_tokens":4,"temperature":0}'

# warmup pair
curl -fsS -X POST -H 'content-type: application/json' "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null &
P1=$!
curl -fsS -X POST -H 'content-type: application/json' "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null &
P2=$!
wait $P1 $P2

# measurement pair
curl -fsS -X POST -H 'content-type: application/json' "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >"$OUTDIR/resp_a.json" &
P1=$!
curl -fsS -X POST -H 'content-type: application/json' "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >"$OUTDIR/resp_b.json" &
P2=$!
wait $P1 $P2

cat "$OUTDIR/resp_a.json"; echo; cat "$OUTDIR/resp_b.json"; echo

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
ls "$OUTDIR/threadreaper/" 2>&1
