# Host-side decode profile — Lever B null + revised diagnosis — 2026-05-04

`FLAMBEAU_HOST_PROFILE=1` instruments `run_completion_blocking_streaming`
to break per-token wall into the chat-handler's host sections:
`decode_logits` (GPU forward + sync), `stop-mask`, `sampler.sample`,
`push_and_emit` (parser + SSE channel), `stopstr_check`.

## Measurements (Qwen3.6-27B-Q4_1, 82-token prompt + 128-token decode, single-stream, greedy)

### TP2 — 26 t/s = 38.5 ms/token

| section          | ms/tok | % of wall |
|------------------|--------|-----------|
| decode_logits    | 38.361 | **99.52%** |
| sampler.sample   |  0.134 |   0.35%   |
| push_and_emit    |  0.049 |   0.13%   |
| stop-mask        |  0.000 |   0.00%   |
| stopstr_check    |  0.000 |   0.00%   |
| **TOTAL_per_step** | 38.546 | 100.00%   |

### PP4 — 20 t/s = 49.4 ms/token

| section          | ms/tok | % of wall |
|------------------|--------|-----------|
| decode_logits    | 49.171 | **99.59%** |
| sampler.sample   |  0.145 |   0.29%   |
| push_and_emit    |  0.053 |   0.11%   |
| stop-mask        |  0.000 |   0.00%   |
| stopstr_check    |  0.000 |   0.00%   |
| **TOTAL_per_step** | 49.371 | 100.00%   |

## Findings

### 1. The host-side chat loop is at-floor

Sampler + parser + channel + stop-string detection together consume
**~0.5% of per-token wall**. The earlier "48% host coordination gap"
diagnosis (`certs/perf/topology_diagnosis_2026_05_04/cert.md`) was
wrong — that estimate was derived by subtracting the existing
2026-04-29 35B-A3B/TP2 kernel-trace cert's GPU active time from wall,
but the cert's `1273 ms across 64 decode steps` includes prefill (one
L=512 prefill burns ~600 ms by itself), so the per-token decode GPU
time was dramatically under-estimated.

Decode-only on 27B / TP2 / single-stream: GPU time ≈ wall ≈ 38 ms/token.
**There is no host-side gap to close.**

### 2. Lever B (skip scheduler channel for N=1) is null

The proposal was to bypass the scheduler `mpsc::channel` round-trip
on single-stream paths. Concrete impact ceiling: `push_and_emit` is
already only **0.05 ms/token** (0.13%). Even a perfect bypass that
eliminates this entirely would be lost in measurement noise.

### 3. Lever C (async forward — overlap sample with next dispatch) is null

The proposal was to overlap token N's host sampler with token N+1's
GPU dispatch. `sampler.sample` is **0.13 ms/token** (0.35%). Same
ceiling argument as Lever B.

### 4. The actual decode bottleneck lives inside `decode_logits`

`decode_logits` calls `forward_one_token_*` which is the per-token
GPU forward pass (64 layers of attention + FFN + MoE + AR), followed
by `download_logits_host` (DtoH of the [vocab] logit vector for host
sampling).

To move the needle on single-stream decode tok/s, the lever must be
inside the GPU forward path itself:
- **Per-rank kernel launch overhead** (~100 launches × ~10 µs = ~1 ms;
  HIP graph capture would address this — documented as null on PP per
  `feedback_*` notes but worth re-checking on TP2 specifically since
  TP2 launches MORE kernels per token than PP per stage)
- **AR sync points** (TP2 does 2-4 AllReduces per layer × 64 layers =
  128-256 sync points; per the 35B-A3B/TP2 cert AR is ~6% of GPU time)
- **Skip the per-token logits DtoH** when GPU sampler is engaged
  (D3 Phase B) — already shipped for TP/Hybrid greedy; the chat path
  in `decode_logits` here is the LEGACY host-sampler path which always
  DtoHs. Forcing the keep-on-device variant in the legacy chat handler
  would save 4 KB DtoH × per-token sync = ~0.5 ms estimated. Modest.

### 5. Per-call alloc/dispose churn (Lever A) won't help decode either

`ShardedForwardOneTokenScratchTp` (the DECODE scratch, not prefill)
is already pre-allocated on `Inflight::Tp` per the existing pool
design. Lever A's per-slot pre-allocated PREFILL scratch only helps
the prefill phase — fixes the #321 OOM-under-concurrency bug, but
does not change single-stream decode tok/s.

## Lever B + C: closing as null

Closing #323 (Lever B) and #325 (Lever C) as **measured null**, not
just predicted-null. The host-side profile rules out both with
direct ms-per-section data, not estimation.

## Lever A: still worth doing

Lever A's value is correctness/scalability, not single-stream perf.
Currently `prefill_serialiser` (the #321 fix) caps concurrent prefill
at one-at-a-time on TP/Hybrid. Per-slot pre-allocated scratch would
let concurrent prefills proceed in parallel without OOM, restoring
the small per-stream prefill TTFT gain that the matrix's 9B/pp4 path
benefits from (6183 → 6707 ms at N=2 = 1.05× — basically free).

## Reproducer

```sh
RUST_LOG=info FLAMBEAU_HOST_PROFILE=1 FLAMBEAU_INFLIGHT_SLOTS=1 \
FLAMBEAU_GPU_SAMPLER=1 FLAMBEAU_CTX_CAP=4096 FLAMBEAU_PREFILL_UBATCH=512 \
target/release/flambeau serve \
  --model /artefact/models/Qwen3.6-27B-Q4_1.gguf \
  --devices hip:0,1 --mesh-mode tp --tp-size 2 --port 18181

curl -s -N http://127.0.0.1:18181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"…200-word prompt…"}],
       "max_tokens":128,"temperature":0,"stream":true}' > /dev/null

# Server stderr prints:
# === HOST decode profile (n=119, post-warmup) ===
#   decode_logits  : 38.361 ms/tok ...
```
