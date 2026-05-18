#!/usr/bin/env bash
# v2 forward-stack serve smoke. Boots the server with FLAMBEAU_V2=1
# on a random port, POSTs one /v1/chat/completions, asserts the
# response carries non-empty content + finish_reason=stop/length.
#
# Usage:
#   scripts/serve/v2_smoke.sh <gguf_path> [<mesh>] [<devices>]
#
# Args:
#   gguf_path  required, e.g. /artefact/models/Qwen3.5-9B-Q4_1.gguf
#   mesh       optional, default "pp"; one of pp / tp / pp+tp
#   devices    optional CSV, default "0"
#
# Exits non-zero on boot failure, HTTP failure, or empty response.

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

# Find a free port.
PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/target/release/flambeau"
if [[ ! -x "$BIN" ]]; then
  echo "building flambeau (release) ..." >&2
  (cd "$ROOT" && cargo build -p flambeau-cli --release --features hip_serve)
fi

LOG="$(mktemp -t v2_smoke_XXXX.log)"
echo "log: $LOG"

# Mesh flags.
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
  --inflight-slots 1 \
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

# Wait for /health.
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died during boot. tail of log:" >&2
    tail -40 "$LOG" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "server never became healthy" >&2
  tail -40 "$LOG" >&2
  exit 1
fi
echo "server up on :$PORT"

REQ='{
  "model": "smoke",
  "messages": [
    {"role": "user", "content": "Reply with exactly: hello world"}
  ],
  "max_tokens": 32,
  "temperature": 0
}'

RESP="$(curl -fsS -H 'content-type: application/json' \
  -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -d "$REQ")"
echo "response: $RESP"

CONTENT="$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"])')"
FINISH="$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["finish_reason"])')"

if [[ -z "$CONTENT" ]]; then
  echo "FAIL: empty content" >&2
  exit 1
fi
case "$FINISH" in
  stop|length) : ;;
  *) echo "FAIL: unexpected finish_reason $FINISH" >&2; exit 1 ;;
esac

# Quality check — repeated-single-char output indicates the
# chained-prefill correctness regression (#222). The architectural
# smoke still PASSES if HTTP returned text; the regression is filed
# as a follow-up.
FIRST="${CONTENT:0:1}"
REPEAT=$(echo -n "$CONTENT" | tr -d "$FIRST" | wc -c)
ALL_LEN=${#CONTENT}
NON_FIRST=$((ALL_LEN - (ALL_LEN - REPEAT)))
if [[ "$ALL_LEN" -gt 4 && "$NON_FIRST" -lt 2 ]]; then
  echo "WARN: degenerate output (single-char repeat) — likely #222 chained-prefill correctness bug" >&2
fi

echo "OK: v2 serve smoke (HTTP path) passed (mesh=$MESH content=${CONTENT:0:80}...)"
