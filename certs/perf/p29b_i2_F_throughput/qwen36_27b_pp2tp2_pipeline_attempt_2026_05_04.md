# P2.9b-i2-F throughput cert RE-RUN — post-#290+#291 pipelined attempt

**Date**: 2026-05-04
**Model**: Qwen3.6-27B-Q4_1
**Topology**: pp2tp2 — `--devices hip:0,2,1,3 --pp-size 2 --tp-size 2`
**Build**: post-#290 (`forward_decode_pipelined_hybrid` written), post-#291
(scheduler dispatches to it under `FLAMBEAU_DECODE_PIPELINE=1` + `n_stages==2`
+ `n>=2`).

## Result: pipelined-N=2 produces DIVERGENT output AND wall regression

### Output divergence

Same prompt ("Write a short poem about the sea (8 lines).", temp=0.0,
max_tokens=64) across three modes:

| mode                                 | md5      | first 60 chars |
|--------------------------------------|----------|----------------|
| N=1 single                           | 59f23a62 | "The waves crash down with thunderous sound,\nAcross the shore" |
| N=2 batched (no pipeline)            | 59f23a62 | "The waves crash down with thunderous sound,\nAcross the shore" |
| N=2 pipelined (async peer_copy)      | c2547432 | "The waves crash down on the rocky shore,\nSalt spray dances i" |
| N=2 pipelined (blocking peer_copy)   | c2547432 | "The waves crash down on the rocky shore,\nSalt spray dances i" |

- Both pipelined slots produce md5-identical-within-batch output
  (scheduler + slot-pool work).
- Pipelined diverges from batched/single at the **first decode token**
  (poem text branches starting from token 1 of the response).
- The divergence is NOT in the async event coordination —
  `FLAMBEAU_PIPELINE_BLOCKING_COPY=1` produces the same divergent output.

### Wall-clock regression

| mode                         | concurrent wall | speedup vs sequential |
|------------------------------|-----------------|-----------------------|
| N=2 batched (no pipeline)    | 6.04 s          | 0.96× (sequential = 5.80 s) |
| N=2 pipelined                | 6.82 s          | 0.85× (regression)    |
| N=2 pipelined-blocking       | 6.92 s          | 0.84× (regression)    |

Even ignoring correctness, pipelined adds ~0.8s wall over batched at N=2.
At PP=2/N=2 the design ceiling is 1.33× — pipelined is delivering 0.88× of
batched, so something is eating ~50% of the would-be win.

## Root cause hypothesis (not yet bisected)

`pipelined_run_slot_through_stage` runs each slot at `n_tokens=1` through
the SINGLE-slot decode helpers:

- `forward_full_attn_layer_decode_batched_tp(slots=&[single])` — same as
  what the batched driver calls at N=2 row 0 (should be bit-identical).
- `forward_gdn_decode_tp` — the **single-slot** GDN path
  (FLAMBEAU_GDN_NO_BATCHED=1 equivalent). Differs from
  `forward_gdn_decode_batched_tp` at N=1 in the matmul wrapping. The
  `feedback_qmatmul_small_m_no_amortize` memory note documents that
  `qmatmul(m=1)` and `qmatmul(m=N)` at small m both dispatch to per-row
  MMVQ but through different code paths — potential numerical difference
  from kernel ordering inside `dispatch_qmatmul`.
- `forward_dense_ffn_prefill_tp(n=1)` / `forward_moe_ffn_prefill_tp(n=1)`
  — same kernels at n=N row 0 should produce identical output, but the
  router / expert dispatch path may have n-dependent buffer aliasing.

Since blocking-copy variant has the SAME divergent output, the bug is in
the per-slot loop body, NOT in the async stream coordination.

## Most likely fix path

Replace `forward_gdn_decode_tp` in `pipelined_run_slot_through_stage`
with `forward_gdn_decode_batched_tp` at N=1 (using the
`gdn_decode_batched` scratch field, which is already present on the
prefill scratch). If output then matches `forward_decode_batched_hybrid`
at N=2, the bug is the GDN single-slot path at n=1 (and the design
should explicitly note "use batched-GDN at N=1 in pipelined mode").

If still divergent, bisect by:
1. Disabling pipelining in the scheduler (`FLAMBEAU_DECODE_PIPELINE=0`)
   and verifying batched N=2 reproduces md5 59f23a62 — confirms the
   reference output.
2. Modifying `pipelined_run_slot_through_stage` to skip the GDN/Attn
   alternation by overriding `cfg.is_recurrent(il)` to a fixed value
   per A/B run, narrowing which layer kind diverges.

## Path forward

The function is gated by `FLAMBEAU_DECODE_PIPELINE=1` + `n_stages==2` +
`n>=2`, so the production path (`forward_decode_batched_hybrid`) is
unaffected. The 3× cert gate **stays open**; #292 is reopened pending
the bisect above.

## Reproduce

```bash
# Server (pipelined):
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=2 FLAMBEAU_DECODE_PIPELINE=1 \
  ./target/release/flambeau serve \
  --model /artefact/models/Qwen3.6-27B-Q4_1.gguf \
  --devices hip:0,2,1,3 --mesh-mode pp+tp --pp-size 2 --tp-size 2 --port 8080

# Server (no pipeline, reference):
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=2 \
  ./target/release/flambeau serve [...]

# Server (pipelined with blocking-copy bisect):
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=2 FLAMBEAU_DECODE_PIPELINE=1 \
  FLAMBEAU_PIPELINE_BLOCKING_COPY=1 \
  ./target/release/flambeau serve [...]

# Client (sends N=1, then 2 concurrent at temp=0):
python3 /tmp/test_compare.py
```
