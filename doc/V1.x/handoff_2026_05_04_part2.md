# Handoff — 2026-05-04 (part 2): pipelined-decode arc

Continuation of `handoff_2026_05_04.md`. The previous session closed
batched-decode #275/#276/#266c/#286 and queued #290 (PP-pipelined decode)
as the next-highest-leverage task.

## What this session shipped

| commit  | task | result |
|---------|------|--------|
| 6d2c9a7 | **#290** | `forward_decode_pipelined_hybrid` — initial implementation (two-phase enqueue) |
| 5484949 | **#291** | wire pipelined dispatch into scheduler + correctness fixes (interleaved per-slot, sub_cluster streams, async logits DtoH) |

(plus uncommitted FIXME doc-comment + cert + handoff)

The function is wired and dispatches under
`FLAMBEAU_DECODE_PIPELINE=1` + `n_stages==2` + `n>=2`. Default OFF.

## Live test result — fixed correctness, perf regression at N=2 only

> **UPDATE (end of session):** the GDN-batched-at-N=1 fix landed and
> restored bit-identical output. Pipelined N=2 now md5-matches batched
> N=2 (= matches N=1). Wall regression remains 0.82× at N=2; the
> scheduler gate is raised to N≥4 where the pipelining ceiling (1.6×)
> can plausibly beat the per-slot host overhead. 27B Q4_1 OOMs at
> INFLIGHT_SLOTS=4 so the N≥4 case is unvalidated. The text below
> captures the original buggy state for context.

See `certs/perf/p29b_i2_F_throughput/qwen36_27b_pp2tp2_pipeline_attempt_2026_05_04.md`.

- **Output divergence**: pipelined-N=2 produces md5 c2547432 ("rocky
  shore...") vs batched-N=2 md5 59f23a62 ("thunderous sound..." which
  matches N=1). Coherent text but different token sequence from the
  first decode token.
- **Bit-identical-within-batch**: both pipelined slots produce the same
  output → scheduler + slot-pool are correct, divergence is per-call.
- **Wall regression**: 6.82 s pipelined vs 6.04 s batched at N=2 = 0.85×
  speedup (worse than batched). Blocking-peer_copy variant is 6.92 s
  (same regression).

## Bisect: bug is NOT in async stream coordination

I added `FLAMBEAU_PIPELINE_BLOCKING_COPY=1` debug switch that swaps
`peer_copy_via_host_async_laned` for the blocking `peer_copy_via_host`
inside the per-slot loop, keeping everything else (per-slot
single-token kernel calls, interleaved structure) identical.

Result: blocking variant produces **the same divergent output** as the
async variant. Therefore the bug is in the per-slot loop body, not in
the async event/bridge coordination.

## Where to look (in order of likelihood)

1. **`forward_gdn_decode_tp` at N=1 vs `forward_gdn_decode_batched_tp` at
   N=1**. The pipelined function uses the SINGLE-SLOT `forward_gdn_decode_tp`
   path (which uses the `gdn_decode` field — single-token GDN scratch).
   The batched driver at N=2 uses `forward_gdn_decode_batched_tp` with
   the `gdn_decode_batched` scratch. Per the
   `feedback_qmatmul_small_m_no_amortize` memory note these dispatch
   through different `qmatmul` code paths at small m — potential numerical
   difference from per-row MMVQ kernel ordering. Quick fix: replace the
   call in `pipelined_run_slot_through_stage` (`hybrid.rs` ~line 1759)
   with `forward_gdn_decode_batched_tp(layer_states=&mut [layer_state], n=1)`
   and re-test.

2. **`forward_dense_ffn_prefill_tp(n=1)` / `forward_moe_ffn_prefill_tp(n=1)`
   row-0 output differs from N=2 row-0**. The router / expert-pick path
   may not be n-independent. Less likely than (1) since these are pure
   compute kernels that should produce identical row-0 output regardless
   of batching dimension.

3. **`forward_full_attn_layer_decode_batched_tp` at N=1 with single
   slot**. Same call pattern the batched driver uses at N=2 row 0 —
   should be bit-identical. Lowest probability culprit; worth ruling
   out by pinning N=1 attention output bit-by-bit against N=2 row 0.

## How to bisect (step-by-step)

```bash
# 1. Establish reference: batched N=2 = N=1 (already verified)
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=2 ./target/release/flambeau serve [...]
python3 /tmp/test_compare.py
# Expected: N=1 md5 = N=2 md5 = 59f23a62

# 2. Reproduce the bug: pipelined N=2 != reference
pkill -f flambeau\\ serve
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=2 FLAMBEAU_DECODE_PIPELINE=1 \
  ./target/release/flambeau serve [...]
python3 /tmp/test_compare.py
# Observed: N=2 md5 = c2547432

# 3. Confirm async is not the bug:
pkill -f flambeau\\ serve
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=2 FLAMBEAU_DECODE_PIPELINE=1 \
  FLAMBEAU_PIPELINE_BLOCKING_COPY=1 ./target/release/flambeau serve [...]
python3 /tmp/test_compare.py
# Observed: same c2547432 → async is not the bug.

# 4. Try the GDN-batched fix:
# Edit crates/models/qwen3-moe/src/forward/hybrid.rs:
#   In pipelined_run_slot_through_stage, replace
#       super::gdn_tp::forward_gdn_decode_tp(...)
#   with the equivalent batched-N=1 call:
#       let gdn_batched = stage_scratch.per_rank[r].gdn_decode_batched.as_mut().unwrap();
#       super::gdn_tp::forward_gdn_decode_batched_tp(
#           ops, stream, device, cfg,
#           attn_norm, attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_a,
#           ssm_dt_bias, ssm_conv1d, ssm_norm, ssm_out,
#           &mut [layer_state],
#           gdn_batched, hidden_a, partial_attn_out,
#           1, world, kq_replicated,
#       )?;
# Rebuild + re-run /tmp/test_compare.py.
# If md5 matches 59f23a62 → bug found, ship the change.
```

## After correctness is fixed: the perf question stays open

Even if (1) above fixes correctness, the wall-time regression suggests
that per-slot host-side launch overhead at N=2 is eating the
1.33×-ceiling pipelining win. Analysis:

- Pipelined per-slot loop: host enqueues ~64 layers × ~8 kernels/layer
  × 2 slots ≈ 1000 launches. At ~5 µs/launch ≈ 5 ms host time per slot.
- Batched at N=2: same number of layers × kernels but at n=2; 64 × ~8 =
  ~512 launches total. ~2.5 ms host time.

So pipelined doubles host launch overhead. The 1.33× pipelining ceiling
can't beat the 2× launch-overhead cost at N=2. The win curve crosses
zero somewhere between N=2 and N=4 (where ceiling rises to 1.6×).

**Recommendation**: even after fixing correctness, skip pipelining for
PP=2/N=2 and only enable for PP=2/N≥4 (or PP=4/N≥2). The scheduler
gate should be `n_stages==2 && n>=4 && pipeline_enabled`. PP=2/N=2 stays
on the existing batched driver.

But: with `INFLIGHT_SLOTS=4` we hit OOM during slot-pool boot on the 27B
Q4_1 model (KV cache × 4 slots over 64 layers × 16 attention heads × 128
head_dim × 2 stages × pp=2 ≈ exceeds gfx906 16GB limit). So validating
N=4 needs either:
- A smaller KV context (default 4096 → 2048 or 1024).
- A smaller model (e.g., Qwen3.5-9B-Q4_1 in pp2 layout — but 9B is
  dense FFN, no GDN, so the test wouldn't exercise the divergent path).
- The 27B Q4_0 (smaller weights, more KV headroom).

## Open tasks queue

- **#292** [in_progress, blocked]: re-cert throughput. Blocked on
  correctness bisect (the GDN-batched-at-N=1 fix above).
- **#288** [pending]: Batched-MMVQ kernels (Q4_0/Q4_1/Q5_K/Q6_K) —
  remains the highest-leverage path to the 3× gate per the prior
  session's analysis (post-#287 cert).
- **Newly identified follow-up**: if pipelining stays sub-linear at N=2,
  evaluate whether the design should target a 1F1B-style cross-slot
  per-layer interleave instead of per-slot full-stage interleave. The
  V2.25.h prefill template uses 1F1B per layer; we could mirror it at
  decode for finer-grained overlap. Higher complexity, more code.

## Diagnostic toggles added

- `FLAMBEAU_DECODE_PIPELINE=1` — enable pipelined dispatch in scheduler
  (defaults OFF; gates by `n_stages==2 && n>=2`).
- `FLAMBEAU_PIPELINE_BLOCKING_COPY=1` — debug variant: swap
  `peer_copy_via_host_async_laned` for blocking `peer_copy_via_host`.
  Used to bisect async vs per-slot-loop bug. Slower (host-syncs every
  hand-off) but reveals if async coordination is the issue.

(Plus all toggles inherited from `handoff_2026_05_04.md`:
`FLAMBEAU_BATCHED_DECODE`, `FLAMBEAU_INFLIGHT_SLOTS`,
`FLAMBEAU_BATCH_WINDOW_US`, `FLAMBEAU_FORCE_BATCH_WINDOW`,
`FLAMBEAU_NO_FAST_PATH`, `FLAMBEAU_GDN_NO_BATCHED`,
`FLAMBEAU_TRACE_BATCH`.)
