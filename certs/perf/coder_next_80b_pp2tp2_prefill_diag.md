# CN-80B-10 — pp2tp2 prefill 3× gap on Coder-Next-80B (diagnosis) — **SUPERSEDED**

**This cert was wrong.** The original claim that "PP uses tile8 MMQ
while TP uses MMVQ" doesn't survive contact with the code: Coder-Next's
`ffn_down_exps` is Q4_1 and `q4_0_use_tile8` (moe.rs:1056-1058) requires
`down_dt ∈ {Q4_0, Q8_0}`, so PP also falls through to MMVQ on this
model. The correct diagnosis (84 % of pp2tp2 prefill wall in
`forward_moe_ffn_prefill_tp`'s 49 ms-per-layer MoE chain — different
root cause) lives in `coder_next_80b_pp2tp2_prefill_diag_v2.md`.

The text below is preserved for the audit trail.

---



## Numbers

|                | pp4 prefill pp512 | pp2tp2 prefill pp512 | ratio |
|----------------|------------------:|---------------------:|------:|
| 35B-A3B-UD-Q4_K_S | 605                |  458                 | 0.76× |
| Coder-Next-80B  | 568                |  183                 | **0.32×** |

Per-layer-stage prefill cost at L=512:

| model | layers/stage | per-stage ms | per-layer ms |
|-------|-------------:|-------------:|-------------:|
| 35B-A3B    | 20 | 280  | 14.0 |
| Coder-Next | 24 | 1399 | **58.3** |

**Coder-Next per-layer prefill on pp2tp2 = 4× slower than 35B-A3B** —
matches the 4× more experts (512 vs 128). MoE TP prefill scales
linearly with expert count under the current path.

## Root cause

Documented in `forward_moe_ffn_prefill_tp`'s own docstring
(`crates/models/qwen3-moe/src/forward/moe_tp.rs:302`):

> "this is the bandwidth-stable indexed-MoE MMVQ path: no sort+pad
> MMQ tile8. The PP non-TP `forward_moe_ffn_prefill` auto-routes to
> tile8 at `n_tokens >= 32`; the TP MMVQ-per-token fallback is
> correct at any L. **tile8 + sort-by-expert TP integration is a
> V2.x perf lever** (would require per-rank sort scratch +
> expert-id replication invariant)."

Concretely:

- **PP path** at L=512: tokens routed through `forward_moe_ffn_prefill`
  which sorts tokens by expert + batches into the
  `indexed_moe_mmq_q4_K_*_tile8_dp4a` family (V2.22.b productisation
  for Q4_K, V2.31.a for Q5_K, equivalents for Q4_0/Q4_1). One MMQ
  tile8 launch processes all tokens routed to a given expert in one
  shot. Compute is amortised; HBM weight reads happen ~once per
  active expert.
- **TP path** today: `forward_moe_ffn_prefill_tp` falls back to
  per-token indexed-MoE MMVQ. For each prompt token × top-k=8 active
  experts × 48 layers × 512 prompt tokens = ~200K MMVQ launches per
  pass. Weight reads happen per-token-per-expert; no amortisation.

The fall-back was deliberate (correct + V1-shippable) but never
upgraded to MMQ tile8 because:
  1. Sort-by-expert needs per-rank sort scratch (not yet allocated
     in `MoePrefillScratch`-TP).
  2. The MMQ tile8 kernel reads expert-id arrays produced by
     `topk_f32`. In TP with sharded MoE experts (each rank holds
     `n_experts / world` experts), the expert-id space differs per
     rank, complicating the dispatch.
  3. The non-TP tile8 was authored with a single-rank invariant on
     the sort scratch + expert-id table. TP integration needs
     replication of the full top-k table across ranks plus per-rank
     filter for the experts owned locally.

## Why 35B-A3B doesn't show the gap as severely

- **128 experts (vs 512)**: 4× fewer per-layer indexed-MoE calls.
  At pp512: 128 × 8 × 40 × 512 = ~21M MMVQ unit-ops vs
  Coder-Next's ~85M. Even though MMVQ is per-token, the absolute
  count is small enough on 35B that other kernels dominate.
- **40 layers (vs 48)**: 1.2× fewer.

Combined: 35B's TP MoE wall is ~4.8× smaller than Coder-Next's.
That puts it below the threshold where the kernel-shape difference
(MMVQ vs MMQ tile8) shows up as a topology-level gap.

## What's NOT the issue

- **Hybrid batched prefill driver**: confirmed engaged via trace
  (`FLAMBEAU_HYBRID_PREFILL_TRACE=1` showed
  `batched_opt_in=true L=512 → BATCHED`). Same numbers with explicit
  fallback test.
- **F16 router (CN-80B-9)**: contributed ~+1-3 % only. Negligible
  vs the 3× gap.
- **GDN TP path**: forward_gdn_prefill_tp / forward_full_attn_prefill_tp
  exist and the 35B perf via the same path (458) shows GDN/full-attn
  TP themselves work well. The Coder-Next-specific gap is in MoE.

## Fix scope (CN-80B-11)

Port the `indexed_moe_mmq_*_tile8_dp4a` family to TP. Two pieces of
infrastructure needed:

1. **Per-rank sort scratch in `MoePrefillScratch`-TP** — store
   sorted-by-expert token list + per-expert ranges. ~50 lines of
   alloc + dispose.
2. **Expert-id replication invariant on the topk output** — the TP
   driver needs every rank to know the full top-k assignment so each
   can filter for its local expert subset. Today the topk runs
   replicated on every rank's router output (router weight is
   Replicated per `tp_layout::for_tensor`), so all ranks already
   compute the same expert-id table. The invariant is already met;
   just needs to be documented + relied on in the new TP tile8
   driver.
3. **Per-rank MMQ tile8 dispatch** — call the existing
   `indexed_moe_mmq_q4_0_*_tile8_dp4a` (and Q4_1, Q5_K, Q4_K
   variants) for the experts owned by this rank. Output is
   `partial_ffn_out` summed across experts.

Estimated effort: 1-2 sessions. The kernels exist; this is wiring
them through the TP forward path with appropriate sort scratch.

## Expected gain

If TP MoE prefill speedup matches PP's MMQ vs MMVQ ratio (~4× on
this rig per V2.22.b memory), Coder-Next pp2tp2 pp512 should jump
from 183 → ~700, exceeding pp4's 568. Combined with pp2tp2's
existing decode advantage (46 vs 41), pp2tp2 would become the
clear best topology.

## Closes

CN-80B-10 #136 — root cause identified. CN-80B-11 #137 ports the
fix.
