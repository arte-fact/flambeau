# Per-token host phase breakdown — flambeau qwen3.6-35B-A3B PP2 (2026-05-17)

Follow-up to `cert.md` after `forward_one_token_pp` inline `Instant`
instrumentation pinpointed the per-phase host budget. Reverts the
"GPU-idle is the dominant cost" framing in the earlier cert.

## Setup

- Patched `crates/blocks/src/topology.rs::forward_one_token_pp` to
  print `[TG] pos=N total=Xus embed=...us peer=...us layers=[...]us
  head=...us argmax=...us` per token when `FLAMBEAU_TG_PROFILE=1`.
- Run: `flambeau infer --model Qwen_Qwen3.6-35B-A3B-Q4_0.gguf --prompt "Hi"
  --max-tokens 64 --devices hip:0,2 --mesh-mode pp`.
- 64 tokens at steady state (pos ≥ 8).

## Per-token budget (steady-state, pos ≥ 8)

| phase | µs | % | what it is |
|---|---:|---:|---|
| total | ~14 500 | 100% | wall per token (= 69 tok/s) |
| `embed_token` | ~220 | 1.5% | rank-0 host-bounce token-embedding lookup |
| `peer_copy_via_host_event` (rank 0→1) | ~20 | 0.1% | async event-bridged, host returns immediately |
| rank 0 layers launch loop (20 layers) | ~1 150 | 8% | host time issuing async kernels |
| rank 1 layers launch loop (20 layers) | ~1 120 | 7.7% | host time issuing async kernels |
| `output_head` | ~4 | 0% | host launch only |
| **`argmax` (DtoH logits + sync)** | **~12 000** | **82%** | host blocks until GPU finishes ALL queued kernels |
| other | ~4 | 0% | bookkeeping |

## Interpretation

The `argmax` 12 ms is **NOT 12 ms of argmax CPU compute**. It is the
host blocking on the first `hipStreamSynchronize` of the per-token
forward — kernel launches above are all async, GPU runs in parallel,
and the DtoH logits download forces the sync that waits for the entire
forward to land. **The 12 ms IS the GPU per-token compute budget.**

This means flambeau's per-token GPU work is ~12 ms = **83 tok/s
GPU ceiling on this model+topology**. The earlier rocprofv3 cert
measured 1565 ms / 128 tokens = 12.2 ms/token average — they match.

## The 2× chat-bench discrepancy

`flambeau infer` (this measurement): **~69 tok/s**.
`certs/perf/bench_matrix_2026_05_16` chat completion bench: **34.49 tok/s**
on the same model+topology.

The chat-completion path adds ~14.5 ms/token on top of the per-token
GPU forward. This overhead is **outside `forward_one_token_pp`** —
it's in the chat-completion handler:

```
[handler] tokenize chat template + apply
  → run_completion_blocking_ids
    → scheduler path (qwen3-moe)
      → loop:
        decode_via_scheduler_into  ─┐
          dispatch_decode_one       │ ~14.5 ms / token (GPU forward)
            forward_one_token_pp    ┘
        sampler.sample(logits, ...) ─── ?? ms (CPU sort / penalties)
        stop-mask iteration over 151k vocab
        SSE chunk serialisation
        stop-string substring scan over decoded text
```

The likely culprit from MEMORY:
- `feedback_qmatmul_small_m_no_amortize`, `Sampler-A+E+F shipped` notes:
  CPU sampler at vocab=151k was 12 ms/token full sort → reduced to ~1-2 ms with
  partial-sort top_k=2048 default.
- `Sampler-D4`: GPU sampler kernel was 8.4× faster than CPU partial-sort
  (0.83 ms vs 6.94 ms at V=151424). Chat sampled 35.6 → 58 tok/s
  (1.63×) when wired.
- **But the GPU sampler is currently DISABLED** in the chat path —
  `crates/server/src/routes/decode_loop.rs:664`:
  ```rust
  // GPU sampler scratch alloc is gated on the keep-logits-on-device
  // optimisation, which was disabled when decode collapsed onto
  // `forward_decode_batched_*` (Phase 12.5)
  let _ = state.gpu_sampler;
  let use_gpu_sampler = false;
  ```
  Phase 12.5 regressed this path by changing the output-head scratch
  buffer; rewiring `resolve_head_logits` to the batched scratch is the
  outstanding follow-up noted in the comment.

## Lever ranking — REVISED (definitive)

Based on the per-phase budget:

1. **Re-wire the GPU sampler in the chat path** — single biggest lever.
   Code comment explicitly identifies the regression. Estimated payoff
   from MEMORY (Sampler-D4): chat sampled **+63%** on qwen3.6-35B-A3B
   (35.6 → 58 tok/s when last measured). For the bench
   `qwen36-35b-a3b-q4_0 / pp2 / 34.49 tok/s`, target after fix would be
   **~55-60 tok/s**, approaching llama.cpp's 61.37 (1.06× of llama).

2. **Skip the legacy CPU sampler entirely on greedy chat** — already
   the case for `temperature=0` per the sampler flow, but the bench
   uses `temp=0.7 top_p=0.9` so non-greedy is the realistic path.

3. **Reduce GPU per-token budget** (the 12 ms ceiling) — slow lever,
   requires structural kernel work (MoE expert fusion, etc.). Only
   worth chasing AFTER (1) lands and we're at the actual GPU ceiling.

## Conclusion

The earlier `cert.md` claim of "host idle dominates 58% of wall on
PP" was wrong — that was measuring the chat-completion handler's
extra overhead, not the per-token forward. The per-token forward
itself is GPU-bound at ~83 tok/s; we get ~69 tok/s on `infer`
(close to the ceiling, 17% overhead). The bench number of 34 tok/s
loses an additional 50% in the chat handler — and the documented
culprit is the disabled GPU sampler from Phase 12.5.

**Fix it: re-wire the keep-on-device output head + GPU sampler in
`run_completion_scheduler_pp_blocking` / `dispatch_decode_one`.**
