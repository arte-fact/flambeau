#!/usr/bin/env bash
# Attach samply to a running flambeau server and record while we
# fire one streaming chat. Output: /tmp/flam_<arm>.json.gz.
set -euo pipefail
ARM="${1:-q8}"
PORT="${2:-22210}"
DUR="${3:-30}"
OUT="/tmp/flam_${ARM}.json.gz"
PID=$(pgrep -f "target/release/flambeau" | head -1)
if [[ -z "${PID}" ]]; then
  echo "no flambeau pid" >&2
  exit 1
fi
echo "samply attach pid=${PID} arm=${ARM} dur=${DUR}s"
samply record -p "${PID}" -r 999 -d "${DUR}" \
              --save-only --no-open -o "${OUT}" &
SAMPLY_PID=$!
sleep 2  # let samply prime
# fire one streaming chat
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
     -H 'content-type: application/json' \
     --data '{"model":"qwen36-27b-q4_0",
              "messages":[{"role":"user","content":"Write a long detailed essay on the history of operating-system schedulers."}],
              "stream":true,"temperature":0.0,"seed":0,"max_tokens":80}' \
     > /tmp/chat_${ARM}.sse 2>&1 &
CURL_PID=$!
wait "${SAMPLY_PID}" || true
kill "${CURL_PID}" 2>/dev/null || true
echo "wrote ${OUT}"
