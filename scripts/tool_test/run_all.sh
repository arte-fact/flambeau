#!/usr/bin/env bash
# One-command API-compliance gate: boot a flambeau server, run the official
# OpenAI- and Anthropic-SDK harnesses against it, then shut it down cleanly.
#
# Covers: tool calling (S1-S5), no-tools (S6), thinking + reasoning_content
# (S7), streaming reasoning (S8), logit_bias (S9), error envelope (S10), and
# the Anthropic /v1/messages surface (A1 text, A2 tool_use, A3 round-trip,
# A4 streaming, A5/A6 thinking, A7 error envelope).
#
# Usage:
#   scripts/tool_test/run_all.sh [MODEL_GGUF]
#
# Env overrides:
#   FLAMBEAU_DEVICES   default "hip:0,2,1,3" (pp2tp2)
#   FLAMBEAU_PORT      default 8080
#   FLAMBEAU_CTX_CAP   default 8192
#
# Exit code 0 iff every asserted scenario passed.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:-/artefact/models/Qwen3.5-27B-Q4_0.gguf}"
PORT="${FLAMBEAU_PORT:-8080}"
DEVICES="${FLAMBEAU_DEVICES:-hip:0,2,1,3}"
CTX_CAP="${FLAMBEAU_CTX_CAP:-8192}"
BIN="$ROOT/target/release/flambeau"
LOG="$(mktemp /tmp/flambeau_run_all.XXXXXX.log)"
MODEL_ID="$(basename "$MODEL" .gguf)"
BASE="http://localhost:${PORT}"

fail() { echo "ERROR: $*" >&2; exit 2; }
[ -x "$BIN" ] || fail "missing $BIN — build with: cargo build --release --bin flambeau --features hip_serve"
[ -f "$MODEL" ] || fail "missing model $MODEL"

echo "== booting flambeau ($MODEL_ID, $DEVICES, ctx-cap $CTX_CAP) =="
"$BIN" serve --model "$MODEL" \
  --mesh-mode pp+tp --pp-size 2 --tp-size 2 --devices "$DEVICES" \
  --port "$PORT" --ctx-cap "$CTX_CAP" --inflight-slots 2 >"$LOG" 2>&1 &
SERVER_PID=$!

# Graceful SIGTERM teardown (HIP context cleanup avoids the VRAM-leak that a
# kill -9 leaves behind).
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    timeout 60 tail --pid="$SERVER_PID" -f /dev/null 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
  curl -sf "$BASE/health" >/dev/null 2>&1 && { echo "== healthy =="; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "== server died =="; tail -15 "$LOG"; exit 2; }
  sleep 2
done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "== health timeout =="; tail -15 "$LOG"; exit 2; }

rc=0
echo
echo "== OpenAI SDK harness (S1-S10) =="
python3 "$ROOT/scripts/tool_test/run.py" --model "$MODEL_ID" --assert --base-url "$BASE/v1" || rc=1
echo
echo "== Anthropic SDK harness (A1-A7) =="
python3 "$ROOT/scripts/tool_test/anthropic_smoke.py" --base-url "$BASE" --model "$MODEL_ID" || rc=1

echo
if [ "$rc" -eq 0 ]; then echo "== ALL GREEN =="; else echo "== FAILURES PRESENT =="; fi
exit "$rc"
