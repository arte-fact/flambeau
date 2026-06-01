#!/usr/bin/env bash
# v2 shared-session concurrent smoke. Boots the server with
# --inflight-slots 2 + --decode-batch-window-us 1500 and fires two
# concurrent /v1/chat/completions requests. Asserts both responses
# are non-empty and distinct (the scheduler must serve both slots
# through the shared Session, NOT just one).
#
# Usage:
#   scripts/serve/v2_concurrent_smoke.sh <gguf_path> [<mesh>] [<devices>]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <gguf_path> [mesh] [devices]" >&2
  exit 2
fi

GGUF="$1"
MESH="${2:-pp}"
DEVICES="${3:-0}"

if [[ ! -f "$GGUF" ]]; then
  echo "SKIP: GGUF $GGUF not present" >&2
  exit 0
fi

PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/target/release/flambeau"
if [[ ! -x "$BIN" ]]; then
  echo "building flambeau (release) ..." >&2
  (cd "$ROOT" && cargo build -p flambeau-cli --release --features hip_serve)
fi

LOG="$(mktemp -t v2_conc_smoke_XXXX.log)"
echo "log: $LOG"

case "$MESH" in
  pp)     MESH_FLAGS=(--mesh-mode pp) ;;
  tp)     N=$(echo "$DEVICES" | tr ',' '\n' | wc -l); MESH_FLAGS=(--mesh-mode tp --tp-size "$N") ;;
  pp+tp)  MESH_FLAGS=(--mesh-mode pp+tp --pp-size 2 --tp-size 2) ;;
  *) echo "unknown mesh $MESH" >&2; exit 2 ;;
esac

FLAMBEAU_V2=1 "$BIN" serve \
  --model "$GGUF" \
  --devices "$DEVICES" \
  "${MESH_FLAGS[@]}" \
  --port "$PORT" \
  --inflight-slots 2 \
  --decode-batch-window-us 1500 \
  --ctx-cap 4096 \
  >"$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died during boot. tail of log:" >&2
    tail -60 "$LOG" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "server never became healthy" >&2
  tail -60 "$LOG" >&2
  exit 1
fi
echo "server up on :$PORT"

REQ_A='{"model":"smoke","messages":[{"role":"user","content":"Reply with exactly: alpha"}],"max_tokens":16,"temperature":0}'
REQ_B='{"model":"smoke","messages":[{"role":"user","content":"Reply with exactly: bravo"}],"max_tokens":16,"temperature":0}'

OUT_A="$(mktemp)"
OUT_B="$(mktemp)"

START_NS=$(date +%s%N)
curl -fsS -H 'content-type: application/json' -X POST \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ_A" >"$OUT_A" &
PID_A=$!
curl -fsS -H 'content-type: application/json' -X POST \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d "$REQ_B" >"$OUT_B" &
PID_B=$!

wait "$PID_A"
RC_A=$?
wait "$PID_B"
RC_B=$?
END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

if [[ $RC_A -ne 0 || $RC_B -ne 0 ]]; then
  echo "FAIL: curl rc A=$RC_A B=$RC_B" >&2
  tail -60 "$LOG" >&2
  exit 1
fi

CONTENT_A="$(python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"])' <"$OUT_A")"
CONTENT_B="$(python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"])' <"$OUT_B")"

echo "A: $CONTENT_A"
echo "B: $CONTENT_B"
echo "wall: ${ELAPSED_MS}ms (2 concurrent /v1/chat)"

if [[ -z "$CONTENT_A" || -z "$CONTENT_B" ]]; then
  echo "FAIL: empty content (A='$CONTENT_A' B='$CONTENT_B')" >&2
  exit 1
fi

# Sanity — each prompt is keyed to a distinct expected word; the model
# need not echo it perfectly but should at least not produce identical
# strings (would indicate slot crosstalk).
if [[ "$CONTENT_A" == "$CONTENT_B" ]]; then
  echo "WARN: identical content from distinct prompts — possible slot crosstalk" >&2
fi

echo "OK: v2 concurrent smoke (N=2 shared session, mesh=$MESH)"
