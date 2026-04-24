# V2.24.b — Lever-2 prefill microbatching AB report

Goal: unlock prefill PP scaling by pipelining ubatches across ranks so
rank k+1 can process ubatch i while rank k processes ubatch i+1.
Reference: llama.cpp PR #6017 + discussion #20252 (Mar 2026) — PP
scales near-linearly when `n_batch > n_ubatch`.

## Iterations

### Iter 1 — external chunking at bench level

Added `FLAMBEAU_UBATCH` env to `perf_baseline_qwen35_9b.rs`. Outer loop
splits `tokens[0..L]` into chunks of size U and calls
`forward_prefill_pp` once per chunk (serial). No code change in
`forward_prefill_pp` itself — the existing function already handles
`start_position` correctly for resumption, so chunking is structurally
safe. **Verifies correctness of the ubatch-split structure.**

### Iter 2 — ubatch=512 (2 chunks at L=1024)

Runs two full PP sweeps — rank 0..3 on first 512 tokens, then rank 0..3
on next 512. Serial both in ubatch order AND across ranks per ubatch.

### Iter 3 — ubatch=128 (8 chunks at L=1024)

Finer-grained ubatches. Explores whether smaller chunks could amortise
differently. Expected to be strictly worse for serial-only chunking.

## Results (Qwen3.5-9B-Q4_1 Mesh<1> prefill)

| UBATCH | L=128 | L=512 | **L=1024** | Δ L=1024 vs baseline |
|---:|---:|---:|---:|---:|
| (none, baseline) | 543.61 | 757.45 | **772.55** | — |
| 1024 (Iter 2 sanity) | 542.27 | 756.44 | 771.90 | noise |
| 512 (Iter 2) | 541.79 | 757.03 | 727.41 | **−5.8 %** |
| 256 (Iter 1) | 542.56 | 650.23 | 624.20 | **−19.2 %** |
| 128 (Iter 3) | 540.84 | 526.64 | 505.55 | **−34.5 %** |

Mesh<4> L=1024 UBATCH=512: 721.70 vs baseline 766.07 = **−5.8 %**.

**All iterations regress.** None win. Regression scales with ubatch
count: more chunks → more PP boundary overhead (peer_copy, per-chunk
output head call, etc.) with no compensating parallelism.

## Root cause

Current `forward_prefill_pp` is a pure sequential pipeline: rank 0 ⇒
rank 1 ⇒ ... ⇒ rank N−1. Each rank uses the *single* `default_stream`
and calls `hipStreamSynchronize` at every peer-copy boundary. Serial
chunking inside this architecture cannot produce overlap — each ubatch
walks the full rank chain front-to-back before the next starts.

**For real microbatch PP overlap you need:**

1. Per-rank independent stream(s) so rank 0 can keep issuing work for
   ubatch i+1 while rank 1+'s stream consumes ubatch i.
2. `hipEventRecord` + `hipStreamWaitEvent` instead of
   `hipStreamSynchronize` at peer-copy boundaries — converts blocking
   syncs to DAG edges the driver can schedule.
3. Double-buffered scratch (hidden_a/hidden_b per rank per ubatch
   lane) so two ubatches can live in per-rank memory simultaneously.
4. Restructured `forward_prefill_pp` with an outer ubatch loop and an
   inner per-ubatch driver that fires work across ranks without
   blocking.

Each of those items is multi-session scope. Confirmed by the literature
scan (V2.23 documentation cycle):

- *llama.cpp PR #6017 (f30ea47):* adds `n_ubatch` + `LLAMA_SCHED_MAX_COPIES`
  (graph duplication so each ubatch-lane has its own scheduler slot).
  Substantial refactor of `ggml_backend_sched`.
- *TD-Pipe paper (arXiv 2506.10470 §2.3):* PP microbatching with complex
  data dependencies and workload imbalance requires careful scheduling;
  naive chunking regresses.

## Next cycle

**V2.25 prefill microbatching — full async.** Not a session-sized
chunk. Minimal implementation plan:

1. Per-rank `HipStream` array in `HipCluster` (N streams per rank for
   N-deep ubatch pipeline); keep `default_stream` as a fallback.
2. `peer_copy_via_host_async` variant that records a `hipEvent_t` on
   src's DtoH stream and lets dst wait on it (no CPU sync).
3. `ShardedForwardPrefillScratch` doubled: `hidden_a_ubatch[u_lane]`
   and `hidden_b_ubatch[u_lane]`.
4. `forward_prefill_pp` restructured as `for ubatch in ubatches { for
   rank in ranks { async_chain(rank, ubatch, u_lane=ubatch % U_LANES) } }`
   with events gating each rank's work on its predecessor's.
5. Output head runs on the last rank after the FINAL ubatch finishes.

Expected: L=1024 M<4> 766 → ≥ 1100 tok/s (per the turbo reference at
1569 tok/s = ideal; our target 73 % closes the scaling gap).

## Gate

- cert-check hip/gfx906: 48 rows, 0 failures (no kernel changes).
- UD-Q4_K_S 8-tok parity preserved (Iter 1 only touches bench code).
- All 3 iterations archived — Iter 1 `FLAMBEAU_UBATCH` env stays in
  `perf_baseline_qwen35_9b.rs` for future regression testing.

## Regeneration

```
for u in 128 256 512 1024; do
  FLAMBEAU_UBATCH=$u FLAMBEAU_MESH_RANKS=1 \
    FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
    ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture
done
```
