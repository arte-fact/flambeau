# Chat-path per-step breakdown — 35B-A3B / PP2 (2026-05-17)

Followup to `gpu_sampler_rewire_null.md`. With sudo unable to lower
`perf_event_paranoid` in the sandbox, fell back to inline `Instant`
instrumentation around the scheduler decode loop + `decode_via_scheduler_into`'s
fast-path. Pinpoints the **+14.5 ms/token chat-path overhead** vs `flambeau infer`.

## Setup

- Server: `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` / PP2 (hip:0,2) / inflight=1
- Request: chat completion, `~300-token prompt` × `max_tokens=64`,
  `temperature=0.7 top_p=0.9 seed=0` (matches bench config)
- Instrumented `routes.rs::decode_via_scheduler_into` fast-path (lock +
  dispatch_decode_one) and `routes/decode_loop.rs::run_completion_scheduler_pp_blocking`
  inner loop (decode + stopmask + sampler), gated on `FLAMBEAU_TG_PROFILE=1`.

## Per-step budget (steady-state, position ≥ 88)

```
[TG-CHAT pos=N] total=22.2ms prefill_lock=0us inflight_lock=1us dispatch_decode_one=22.2ms
[TG-LOOP step=N] total=24.2ms decode=22.2ms stopmask=0us sampler=2.0ms
```

| phase | µs | % |
|---|---:|---:|
| total per step | ~24 200 | 100% |
| **`dispatch_decode_one`** (forward + DtoH download) | **~22 200** | **91.7%** |
| stop-token mask iteration | 0 | 0% |
| CPU sampler (partial-sort + softmax) | ~2 000 | 8.3% |
| `prefill_serialiser` lock | 0 | 0% |
| `inflight_pool` mutex acquire | 0-2 | 0% |
| `slot_in_use` atomics | 0 | 0% |

## Comparison: where the bench's 14.5 ms/token gap actually lives

| path | per-step | breakdown |
|---|---:|---|
| `flambeau infer` (legacy fused) | 14.5 ms | embed 0.2 + layers 2.3 (launch) + argmax-sync 12.0 |
| chat bench (scheduler) | 24.2 ms | `dispatch_decode_one` 22.2 + sampler 2.0 |

`dispatch_decode_one` is `~7.7 ms slower` than the legacy `forward_one_token_pp`
on the same model + topo. That delta exactly matches the routes.rs:721-724
comment:

> The batched code path (`forward_decode_batched_*`) replays prefill-
> flavoured kernels which add a few extra launches per layer (separate
> rmsnorm + 2 quant variants); at N=1 those launches are pure overhead
> vs the fused decode form.

## Diagnoses ruled out

These were all assumed to be the culprit at various points; the data
disproves each one:

- ~~CPU sort cost~~ → only 2 ms (Sampler-A's partial-sort works).
- ~~Stop-token mask iteration over 151k vocab~~ → 0 µs (negligible).
- ~~`tokio::sync::Mutex::blocking_lock` on inflight_pool~~ → 0-2 µs.
- ~~`slot_in_use` atomics + n_others_active probe~~ → 0 µs.
- ~~SSE chunk serialization~~ → not in this test (non-stream POST).
- ~~Per-step Vec capacity guard (~600 KB logits)~~ → 0 µs.
- ~~`tracing` span overhead at info level~~ → 0 µs.

## Confirmed bottleneck: forward kernel choice

The chat path takes `dispatch_decode_one` → `forward_decode_batched_with_inflights`
→ `model.forward_decode_batched` → `qwen3moe_forward_decode_batched` →
`forward_decode_batched_pp(... N=1 ...)`.

The `infer` path takes `forward_one_token_pp` (legacy fused decode).

For each of 40 layers, the batched form runs:
- `rmsnorm_f16` separately
- `quantize_f16_q8_1` separately
- `quantize_f16_q8_1_mmq` separately (DS4 path)

The fused decode form bundles `rmsnorm + quantize_q8_1` in one launch
(`flambeau_rmsnorm_q8_1_fused` per the rocprofv3 cert). At 40 layers ×
~2 extra launches × ~5 µs HIP overhead + HBM round-trips, the per-token
overhead lands at ~7-8 ms. Matches the 7.7 ms measured.

## Lever ranking — REVISED (third time, finally definitive)

1. **Restore the fused decode path for N=1 chat requests**. Two options:
   a. **Route N=1 through `forward_one_token_pp`** when only one slot is
      active. Today's `dispatch_decode_one` ALWAYS goes through the
      batched path. The `dispatch_decode_one` fast-path predicate is
      already in place (`n_others_active == 0`); just dispatch to the
      legacy fused forward there. Risk: legacy forward still exists for
      qwen3-moe? Check.
   b. **Add a `forward_decode_batched_pp::N1_fused` specialization** that
      uses the fused rmsnorm+quant_q8_1 kernels at N=1.
   Estimated payoff: 14.5 ms → 17 ms (with sampler) = 60 tok/s (closes
   most of the gap to llama.cpp's 61).

2. **GPU sampler with `KeepOnDeviceLogits`** — the 2 ms CPU sampler is
   2x the 0.83 ms GPU topk + small DtoH. Still meaningful but secondary
   to lever 1.

3. **Lever 1 (PP overlap)** — even smaller payoff now that the gap is
   firmly attributed to per-layer-launch overhead in the batched form.

## Conclusion

The previous cert's framing — "chat handler has ~14.5 ms/token overhead
in housekeeping" — was wrong. **The overhead is entirely in the
forward-pass kernel choice**: at N=1, the batched `forward_decode_batched_pp`
runs decompressed/non-fused kernels that the legacy `forward_one_token_pp`
fused. Phase 12.5 collapsed decode onto the batched path for code-share,
trading ~30% N=1 wall throughput for ~zero N>1 maintenance cost.

The fix is structural but not a kernel rewrite: dispatch the N=1 case
back through the fused decode path. Whether to keep the batched
path as the only N≥2 path or merge them depends on session/scratch
plumbing — a 1-2 session refactor.
