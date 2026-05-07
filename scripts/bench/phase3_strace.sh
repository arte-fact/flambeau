#!/usr/bin/env bash
# Boot flambeau server with given KV layout, fire one warm-up chat,
# then strace -c attach during a measured chat. Output: syscall-time
# summary that reveals where host wall goes (futex contention,
# epoll_wait idle, hipMemcpy via ioctl, etc).
set -euo pipefail
ARM="${1:-q8}"
LOG=/tmp/strace_${ARM}.txt
SERVER_LOG=/tmp/serve_${ARM}.log
PORT=22210
MODEL_PATH=/artefact/models/Qwen3.6-27B-Q4_0.gguf

# Profile env from optimized.toml — extract via python
PROFILE_ENV=$(python3 - <<'PY'
import sys, pathlib
sys.path.insert(0, "scripts/bench")
import run_env_impact as rei
e = rei.load_profile_env(pathlib.Path("bench/profiles/optimized.toml"))
for k, v in e.items():
    print(f"{k}={v}")
PY
)

echo "=== boot ${ARM} ==="
env_args=()
while IFS= read -r kv; do
  env_args+=("${kv}")
done <<<"${PROFILE_ENV}"
env_args+=("FLAMBEAU_KV=${ARM}" "FLAMBEAU_INFLIGHT_SLOTS=8")

env "${env_args[@]}" target/release/flambeau serve \
    --model "${MODEL_PATH}" \
    --devices 0,2,1,3 --mesh-mode pp+tp \
    --pp-size 2 --tp-size 2 --port ${PORT} \
    --ctx-cap 4096 --kv "${ARM}" --inflight-slots 8 \
    > "${SERVER_LOG}" 2>&1 &
SRV_PID=$!
echo "server pid=${SRV_PID}"

# wait for ready and discover model id
MODEL_ID=""
for i in {1..240}; do
  RESP=$(curl -sS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || true)
  if [[ -n "${RESP}" ]]; then
    MODEL_ID=$(echo "${RESP}" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || true)
    if [[ -n "${MODEL_ID}" ]]; then
      echo "ready after ${i}s, model_id=${MODEL_ID}"
      break
    fi
  fi
  sleep 1
done
if [[ -z "${MODEL_ID}" ]]; then
  echo "FAIL: server never became ready"
  kill "${SRV_PID}" 2>/dev/null || true
  exit 1
fi

# warm-up chat (kernels JIT'd, weights paged in)
echo "=== warmup chat ==="
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
     -H 'content-type: application/json' \
     --data '{"model":"'"${MODEL_ID}"'","messages":[{"role":"user","content":"Hello."}],"stream":false,"max_tokens":16,"temperature":0}' \
     >/dev/null
sleep 0.5

echo "=== strace attach + measured chat ==="
sudo -n -- strace -c -f -p "${SRV_PID}" -o "${LOG}" 2>&1 &
STRACE_PID=$!
sleep 1

# measured chat (80 tokens)
TIME_OUT=$(curl -sS -o /tmp/chat_${ARM}.sse -w '%{time_total}' \
     -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
     -H 'content-type: application/json' \
     --data '{"model":"'"${MODEL_ID}"'","messages":[{"role":"user","content":"Write a long detailed essay on the history of operating-system schedulers."}],"stream":true,"temperature":0,"seed":0,"max_tokens":80}')
echo "chat wall: ${TIME_OUT}s"

# stop strace
sudo -n -- kill -INT "${STRACE_PID}" 2>/dev/null || true
wait "${STRACE_PID}" 2>/dev/null || true

# stop server
kill "${SRV_PID}" 2>/dev/null || true
wait "${SRV_PID}" 2>/dev/null || true

echo "=== ${LOG} ==="
cat "${LOG}"
