#!/usr/bin/env bash
# V1.8.C server smoke — starts the flambeau server, hits every endpoint with
# curl + the official openai Python client, asserts on outputs, kills the
# server, reports pass/fail.
#
# Prereqs:
#   - flambeau CLI built with `--features hip_serve`
#   - FLAMBEAU_QWEN3_GGUF env (or first arg) points at a real GGUF
#   - Python 3 with `openai` installed (pip install --break-system-packages openai)
#
# Usage: bench/server_smoke.sh [gguf_path] [port]

set -euo pipefail

GGUF="${1:-${FLAMBEAU_QWEN3_GGUF:-/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}}"
PORT="${2:-18099}"
BIND="127.0.0.1:${PORT}"
BASE_URL="http://${BIND}"

if [ ! -f "$GGUF" ]; then
    echo "GGUF not found: $GGUF" >&2
    exit 1
fi

BIN="$(dirname "$0")/../target/release/flambeau"
if [ ! -x "$BIN" ]; then
    echo "flambeau binary not built at $BIN — run: cargo build --release -p flambeau-cli --features hip_serve" >&2
    exit 1
fi

LOG=$(mktemp -t flambeau_smoke.XXXXXX.log)
echo "[smoke] starting server on ${BIND} (log: $LOG)"
"$BIN" serve --model "$GGUF" --devices 0,1 --port "$PORT" > "$LOG" 2>&1 &
SRV_PID=$!
trap 'echo "[smoke] cleanup"; kill -9 $SRV_PID 2>/dev/null || true; rm -f "$LOG"' EXIT

# Wait for /health.
echo "[smoke] waiting for /health …"
for i in $(seq 1 60); do
    if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
        echo "[smoke] up after ${i}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 60 ]; then
        echo "[smoke] server never came up. Log:"
        tail -30 "$LOG"
        exit 1
    fi
done

pass() { echo "[smoke] PASS  $1"; }
fail() { echo "[smoke] FAIL  $1"; exit 1; }

# ---- curl: /health ----
out=$(curl -sf "${BASE_URL}/health")
echo "$out" | grep -q '"status":"ok"' || fail "/health payload: $out"
pass "/health"

# ---- curl: /v1/models ----
out=$(curl -sf "${BASE_URL}/v1/models")
echo "$out" | grep -q '"object":"list"' || fail "/v1/models payload: $out"
echo "$out" | grep -q '"owned_by":"flambeau"' || fail "/v1/models owner: $out"
pass "/v1/models"

# ---- curl: /v1/chat/completions (non-streaming) ----
req='{"messages":[{"role":"user","content":"Reply exactly: Hello world."}],"max_tokens":16,"temperature":0}'
out=$(curl -sf -X POST "${BASE_URL}/v1/chat/completions" -H 'content-type: application/json' -d "$req")
echo "$out" | grep -q '"object":"chat.completion"' || fail "chat.completions object: $out"
echo "$out" | grep -q '"finish_reason":"stop"' || fail "chat.completions finish_reason: $out"
# Content should be non-empty + not contain leaked special tokens.
content=$(echo "$out" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(j["choices"][0]["message"]["content"])')
[ -n "$content" ] || fail "chat.completions empty content"
echo "$content" | grep -q "<|im_" && fail "chat.completions leaked <|im_…|> tokens: $content"
pass "/v1/chat/completions  (content: $(echo "$content" | head -c 60))"

# ---- curl: /v1/completions ----
out=$(curl -sf -X POST "${BASE_URL}/v1/completions" -H 'content-type: application/json' -d '{"prompt":"The capital of France is","max_tokens":4,"temperature":0}')
echo "$out" | grep -q '"object":"text_completion"' || fail "completions object: $out"
# Model should mention Paris (conservative check; completion might be " Paris" etc.)
echo "$out" | grep -qi paris || fail "completions content didn't mention Paris: $out"
pass "/v1/completions  (Paris)"

# ---- real OpenAI Python client ----
python3 - <<PY
import sys
from openai import OpenAI
client = OpenAI(base_url="${BASE_URL}/v1", api_key="not-used")
# Models list
models = client.models.list()
assert any(m.id for m in models.data), f"empty models list: {models}"
# Chat completion
resp = client.chat.completions.create(
    model="flambeau",
    messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
    max_tokens=8,
    temperature=0,
)
text = resp.choices[0].message.content or ""
assert "<|im_" not in text, f"leaked special token: {text!r}"
assert resp.choices[0].finish_reason in {"stop", "length"}, f"bad finish: {resp}"
print(f"[smoke] openai-client chat content: {text!r}", file=sys.stdout)
PY
pass "openai-python-client  (chat.completions)"

echo "[smoke] ALL PASS"
