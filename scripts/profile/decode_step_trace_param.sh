#!/usr/bin/env bash
# Param trace harness. Usage:
#   decode_step_trace_param.sh <v2|legacy> <outdir> <model_path> <devices_csv> <mesh>
set -euo pipefail
MODE="$1"; OUTDIR="$2"; MODEL="$3"; DEVICES="$4"; MESH="$5"
source /artefact/flambeau/.env
BIN=/artefact/flambeau/target/release/flambeau
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
mkdir -p "$OUTDIR"; rm -f "$OUTDIR"/* 2>/dev/null || true
ENV_PREFIX=""
[[ "$MODE" == "v2" ]] && ENV_PREFIX="FLAMBEAU_V2=1"
LOG="$OUTDIR/server.log"
env $ENV_PREFIX "$ROCPROFV3" --kernel-trace --output-directory "$OUTDIR" --output-format csv -- \
  "$BIN" serve --model "$MODEL" --devices "$DEVICES" --mesh-mode "$MESH" \
  --port "$PORT" --inflight-slots 1 --ctx-cap 4096 \
  >"$LOG" 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null; wait $PID 2>/dev/null || true" EXIT
for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "died" >&2; tail -30 "$LOG" >&2; exit 1; }
  sleep 0.5
done
REQ='{"model":"x","messages":[{"role":"user","content":"Continue: cat, dog, tree,"}],"max_tokens":4,"temperature":0}'
curl -fsS -X POST -H 'content-type: application/json' "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null
curl -fsS -X POST -H 'content-type: application/json' "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ" >/dev/null
kill $PID; wait $PID 2>/dev/null || true
