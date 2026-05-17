# GPU sampler rewire: null result on chat path (2026-05-17)

Attempted to re-enable the GPU sampler in the scheduler chat-decode
path, per the comment in `decode_loop.rs:664` that Phase 12.5
disabled it pending a buffer-rewire follow-up.

## What was wired

1. `resolve_head_logits` (PP arm) — switched from
   `inflight.decode.per_rank[head].output_head.logits_f32` (legacy
   single-token decode scratch, dormant after Phase 12.5) to
   `inflight.prefill.per_rank[head].output_head.logits_f32` (the
   prefill scratch, which is where `forward_decode_batched_pp` lands
   its logits — the same scratch is used for batched decode too).

2. `run_completion_scheduler_pp_blocking` decode loop — after
   `decode_via_scheduler_into` (which still does the full-vocab DtoH
   host download internally), call `gpu_sampler::run_gpu_topk` on
   the device-resident logits to get top-K (id, prob) pairs, then
   use `Sampler::sample_from_topk` for the multinomial draw instead
   of the CPU full-vocab `Sampler::sample`. Gated on:
   - `state.gpu_sampler` flag (--no-gpu-sampler off path)
   - `!sampling.has_penalties()` (penalties need full-vocab access)
   - `!params.json_mode` (JSON mask covers top-2048 host slice)
   - `model.topology() == "pp"` (TP/Hybrid arms route through shared
     batched scratch; wire as follow-up)
3. Stop-token mask in K-space instead of full-vocab (was 80 stop_ids ×
   full-vocab indexed write; now 80 × K-iterate, K=2048).

## Bench result — qwen3.6-35B-A3B-Q4_0 / pp2 / pp4

|  | baseline (CPU sampler) | with GPU sampler | delta |
|---|---:|---:|---:|
| pp2 decode | 34.59 tok/s | **33.87 tok/s** | **−2.1%** |
| pp4 decode | 32.54 tok/s | **32.35 tok/s** | **−0.6%** |

Both topologies show a **small net loss**. Coherence preserved
("The capital of France is **Paris**. …" on both engines).

## Why it didn't help

The GPU sampler's win in MEMORY's Sampler-D3 lineage (35.6 → 58 t/s
on this same model+topology) had three contributors:

1. **CPU full-vocab sort eliminated** — was ~12 ms/token. The chat
   path's CPU sampler ALREADY does partial-sort (Sampler-A's
   `TOP_K_AUTO_CAP = 2048`) when `top_p` is set, which drops the
   cost to ~1-2 ms. Sampler-A captured most of the 35.6 → 53
   improvement.
2. **600 KB host DtoH download eliminated** — Phase D3-B's
   `LogitsSink::KeepOnDevice`. This skip is what gave the final
   ~5 t/s from 53 → 58.
3. **GPU topk_softmax_f32** — 0.83 ms/token vs CPU ~1-2 ms.

This attempt re-wired (3) but NOT (2). The host download still
happens inside `dispatch_decode_one` →
`forward_decode_batched_pp::download_logits_host`. So we pay both
the host download AND the GPU topk launch + sync + 16 KB tuple
download — a net loss because the saving over partial-sort CPU is
sub-millisecond and the GPU kernel's launch/sync overhead exceeds
it.

## Why the original 35 → 58 went through this lever

Phase D3 was wired on top of a DIFFERENT decode hot path —
`forward_one_token_tp_inner` / `decode_keep_logits_on_device`
(see MEMORY `Sampler-D3`). That path had a `LogitsSink::KeepOnDevice`
variant that physically skipped the host download. Phase 12.5
collapsed decode onto `forward_decode_batched_*`, which doesn't
have a `KeepOnDevice` variant — and the comment in
`decode_loop.rs:664` notes this explicitly:

> The batched output head writes to a different scratch buffer than
> the keep-on-device kernels read from. Re-wiring `resolve_head_logits`
> to the batched scratch is a follow-up.

What the comment elided: **rewiring resolve_head_logits is necessary
but not sufficient.** The download is what made the GPU sampler net-
positive in the D3 measurement. Without skipping it, GPU topk is just
a launch + sync round-trip that loses to CPU partial-sort.

## Lever-3 revised — what would actually work

To recover the 53 → 58 (or better) gain we need **either**:

A. **`KeepOnDeviceLogits` variant on `forward_decode_batched_pp`**
   — add a `skip_host_download: bool` flag (or a sibling fn), wire
   it to skip the `download_logits_host` step at the head rank for
   slot[0] when the caller will pull logits via GPU sampler. Then
   the GPU sampler reads the device-resident buffer directly; the
   16 KB top-K DtoH is the only host transfer. ~1-2 sessions of
   work + parity.

B. **Replace the bench's chat path with the simpler `infer` path**
   for benchmarking. The `infer` measurement showed 69 tok/s; the
   chat bench shows 34. The remaining gap (29 ms/token wall vs
   14.5 ms/token infer = +14.5 ms/token overhead) lives in
   `run_completion_scheduler_pp_blocking` and `dispatch_decode_one`
   beyond what GPU sampler addresses. Examples: dispatch's
   per-step `inflight_pool[i].blocking_lock()` (tokio mutex,
   not std), Vec capacity guards, SSE not used (non-stream), tracing
   spans. These are <ms each but per-token costs add up.

Neither is a quick win. Reverting this attempt and leaving the
existing CPU partial-sort sampler in place.

## Reverted

- `crates/server/src/gpu_sampler.rs` (PP arm pointer fix) — reverted
- `crates/server/src/routes/decode_loop.rs` (GPU sampler hook) — reverted

The async peer-copy change (commit `bab194f`, +1%) stays in place; the
`peer_copy_via_host_event` orchestrator change is the right shape
for any future async / overlapped PP decode work.

## Open question for next session

What accounts for the **14.5 ms/token overhead** between
`flambeau infer` (69 tok/s) and the chat-completion bench (34 tok/s)?

Candidates from looking at `run_completion_scheduler_pp_blocking`
and `dispatch_decode_one`:
- Per-step `tokio::sync::Mutex::blocking_lock` on `inflight_pool[slot]`
- Per-step `state.slot_in_use` atomics
- The "n_others_active" probe (iterates all slots' atomics)
- `Vec<f32>` capacity guard on `logits_out`
- BatchSlot allocation
- Tracing spans / `info!` calls (debug level off at info)

A `samply` CPU profile of the chat path (now that we have sudo) would
disambiguate. Worth doing before any more kernel-level work.
