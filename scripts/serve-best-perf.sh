#!/usr/bin/env bash
# scripts/serve-best-perf.sh — launch flambeau on the user's daily-driver
# config: Qwen3.5-9B-Q4_1 / pp2tp2 / GPU sampler / scheduler-aggregated
# concurrent decode / max-fitting context. Listens on port 8081.
#
# Override defaults via env:
#   MODEL=/path/to/other.gguf  ./scripts/serve-best-perf.sh
#   PORT=8080                  ./scripts/serve-best-perf.sh
#   CTX=16384                  ./scripts/serve-best-perf.sh
#   SLOTS=4                    ./scripts/serve-best-perf.sh
set -euo pipefail

# ---- knobs --------------------------------------------------------------
MODEL="${MODEL:-/artefact/models/Qwen3.6-27B-Q4_1.gguf}"
PORT="${PORT:-8081}"
# 32k stays comfortably inside 16 GB / rank for 9B-Q4_1 on pp2tp2 KV.
# Bump to 65536 / 131072 if you have headroom; reduce to 8192 if loading
# 27B/35B on the same rig.
CTX="${CTX:-32768}"
SLOTS="${SLOTS:-2}"
# Device order: pp2tp2 stages = (hip:0, hip:2) and (hip:1, hip:3).
# Per the rig topology memory: GPUs 0,1 on die 0; GPUs 2,3 on die 1; the
# {0,2}/{1,3} cross-die pairing avoids the {2,3} intra-die-1 fault.
DEVICES="${DEVICES:-hip:0,2,1,3}"

# ---- env -----------------------------------------------------------------
# Cap context above the model's architectural max (qwen3.5/3.6: 131072).
export FLAMBEAU_CTX_CAP="${CTX}"

# Multi-slot inflight pool. Even without batched decode, having N>1
# slots lets concurrent /v1/chat requests overlap host-side work
# (tokenize, sampler, response build) — that alone gave 1.30x at
# N=2 in P2.9b-i1's live test. Default 1 = no concurrency.
export FLAMBEAU_INFLIGHT_SLOTS="${SLOTS}"

# Scheduler-aggregated batched decode is OFF by default — it's the
# P2.9b-i2-D-wire path. Set FLAMBEAU_BATCHED_DECODE=1 in the env
# explicitly when testing the scheduler. The legacy decode path is
# the proven daily-driver default.

# GPU-side topk + penalty sampler (TP/Hybrid only; engages on this hybrid).
# Skips the 600 KB host-logits DtoH per token.
export FLAMBEAU_GPU_SAMPLER=1

# Chunked prefill default ubatch — keeps the per-layer attention scratch
# bounded (~10 MB at L=512). 9B/Q4_1 has plenty of headroom; 35B-A3B
# users may want to drop this to 256 to leave room for KV at higher ctx.
export FLAMBEAU_PREFILL_UBATCH="${FLAMBEAU_PREFILL_UBATCH:-512}"

# Server tracing.
export RUST_LOG="${RUST_LOG:-info}"

# ---- launch --------------------------------------------------------------
BIN="$(dirname "$0")/../target/release/flambeau"
if [[ ! -x "$BIN" ]]; then
    echo "missing $BIN — run: cargo build -p flambeau-cli --release --features hip_serve" >&2
    exit 1
fi

echo "MODEL=${MODEL}"
echo "PORT=${PORT}  CTX_CAP=${CTX}  SLOTS=${SLOTS}  DEVICES=${DEVICES}"
echo "GPU_SAMPLER=1  BATCHED_DECODE=${FLAMBEAU_BATCHED_DECODE:-0}"
echo

exec "$BIN" serve \
    --model "${MODEL}" \
    --devices "${DEVICES}" \
    --mesh-mode pp+tp \
    --pp-size 2 \
    --tp-size 2 \
    --port "${PORT}"
