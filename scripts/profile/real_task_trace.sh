#!/usr/bin/env bash
# Capture kernel trace for a real-task prompt (pp~700 / tg=128). Wraps
# flambeau serve in rocprofv3, fires one warmup + one measurement
# request, then shuts the server (which flushes the trace).
#
# Usage:
#   real_task_trace.sh <v2|legacy> <outdir> <gguf_path> <devices_csv> <mesh> [tp_size]

set -euo pipefail
MODE="$1"; OUTDIR="$2"; MODEL="$3"; DEVICES="$4"; MESH="$5"; TP_SIZE="${6:-}"

source /artefact/flambeau/.env
BIN=/artefact/flambeau/target/release/flambeau
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/* 2>/dev/null || true

ENV_PREFIX=""
[[ "$MODE" == "v2" ]] && ENV_PREFIX="FLAMBEAU_V2=1"

MESH_FLAGS=(--mesh-mode "$MESH")
if [[ "$MESH" == "tp" && -n "$TP_SIZE" ]]; then
    MESH_FLAGS+=(--tp-size "$TP_SIZE")
elif [[ "$MESH" == "pp+tp" ]]; then
    MESH_FLAGS+=(--pp-size 2 --tp-size 2)
fi

LOG="$OUTDIR/server.log"
env $ENV_PREFIX "$ROCPROFV3" --kernel-trace --output-directory "$OUTDIR" --output-format csv -- \
  "$BIN" serve --model "$MODEL" --devices "$DEVICES" "${MESH_FLAGS[@]}" \
  --port "$PORT" --inflight-slots 1 --ctx-cap 4096 \
  >"$LOG" 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null; wait $PID 2>/dev/null || true" EXIT

for _ in $(seq 1 240); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "died" >&2; tail -30 "$LOG" >&2; exit 1; }
  sleep 0.5
done

# Real-task prompt — ~700 tokens, asks for ~128 tokens out.
PROMPT='You are a senior compiler engineer. Write an exhaustive technical explanation in 100 words covering: (1) AMD CDNA-1 gfx906 V_DOT4_I32_I8 instruction encoding and sign-extension semantics across int8 lanes, (2) per-VGPR register pressure when fused into Q4_0 / Q8_0 MMVQ kernels, including the 32-VGPR threshold for wave64 occupancy, (3) the role of dp4a in quantized matmul, with the per-block scale folding and the ~4x FMA reduction it provides, (4) NVIDIA __dp4a comparison on sm_61+ Pascal, (5) LDS double-buffering interactions with the wave scheduler for HBM2 latency hiding, (6) a worked numerical example tracing 8 dp4a invocations across a single Q8_0 / int8 K block of 32 elements, including the int32 partial accumulation and the K.d x Q.d scalar dequant at block boundary, (7) when dp4a wins vs loses on gfx906 (arithmetic intensity, large K, fused quant) vs decode-attention single-row mmvq, and (8) typical rocprofv3 PMC signatures to look for. Be exhaustive with concrete numbers.'

REQ=$(python3 -c "import json,sys; print(json.dumps({'model':'x','messages':[{'role':'user','content':'$PROMPT'}],'max_tokens':128,'temperature':0}))")

# Warmup (tiny request)
curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' >/dev/null
echo "warmup done"

# Measurement
START=$(date +%s%N)
RESP=$(curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ")
END=$(date +%s%N)
WALL_MS=$(( (END - START) / 1000000 ))
PT=$(python3 -c 'import json,sys;print(json.load(sys.stdin)["usage"]["prompt_tokens"])' <<<"$RESP")
CT=$(python3 -c 'import json,sys;print(json.load(sys.stdin)["usage"]["completion_tokens"])' <<<"$RESP")
echo "MODE=$MODE pp_tok=$PT tg_tok=$CT wall_ms=$WALL_MS"

kill $PID
wait $PID 2>/dev/null || true

echo "trace files in $OUTDIR:"
ls -la "$OUTDIR/threadreaper/" 2>&1
