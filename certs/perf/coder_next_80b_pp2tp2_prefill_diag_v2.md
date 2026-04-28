# CN-80B-11a — pp2tp2 prefill diagnosis (v2, replaces CN-80B-10)

## CN-80B-10 was wrong

CN-80B-10 claimed PP uses tile8 MMQ while TP uses MMVQ — but
`forward_moe_ffn_prefill`'s `q4_0_use_tile8` gate (moe.rs:1056-1058)
requires `down_dt ∈ {Q4_0, Q8_0}`. Coder-Next's `ffn_down_exps` is
**Q4_1**, so PP also falls through to the MMVQ branch on this model.
Both topologies use indexed-MoE MMVQ — the gap is elsewhere.

## v2 diagnosis (HipEvent marks in `forward_prefill_tp_batched_layers`)

Coder-Next-Q4_0 pp2tp2 prefill L=512, post-warmup:

| section            | total ms | mean ms/call | % wall |
|--------------------|---------:|-------------:|-------:|
| **ptp_moe_ffn**    |  2350.5  |    49.0      | **84.3 %** |
| ptp_attn_gdn       |   165.4  |     4.6      |   5.9 % |
| ptp_router         |    68.8  |     1.4      |   2.5 % |
| ptp_shared         |    47.3  |     1.0      |   1.7 % |
| ptp_attn_full      |    35.2  |     2.9      |   1.3 % |
| ptp_ffn_ar         |    32.3  |     0.67     |   1.2 % |
| ptp_attn_ar        |    25.4  |     0.53     |   0.9 % |
| ptp_ffn_norm       |     3.0  |     0.06     |   0.1 % |
| total attributed   |  2727.8  |              |  97.8 % |

Wall: 2789 ms (183 tok/s).

## Reading

- **Hot section: `ptp_moe_ffn` at 49 ms / layer × 48 layers**.
  This wraps `forward_moe_ffn_prefill_tp` (gate+up MMVQ → swiglu →
  quantize → down MMVQ → optional shared-add) on each rank. 84 % of
  wall.
- AR is ~2 % combined (`ptp_attn_ar` + `ptp_ffn_ar`). Not the issue.
- Router runs 4× per layer (replicated across all 4 ranks) but only
  totals 2.5 % wall. F16 router (CN-80B-9) helped a bit; not the
  story.
- GDN attn TP at 5.9 % is reasonable — pp4's GDN attn (full-attn
  layers excluded) runs in roughly similar wall.

## Comparison

- **pp4 per-layer wall** (all kernels): 18.8 ms
- **pp2tp2 per-layer MoE alone**: 49 ms
- **pp2tp2 MoE > pp4 per-layer total** by 2.6×

So the MoE FFN TP path is doing 2.6× MORE work than the entire PP
forward-per-layer (and that's just one of pp2tp2's per-layer
sections). MoE FFN TP at half the per-rank `inter` SHOULD be ≤ PP's
MoE FFN time, not 5× more. Something is structurally wrong with
the TP MoE prefill kernel shape on Coder-Next.

## Hypotheses for CN-80B-11b

1. **`indexed_moe_mmvq_*_gate_up` kernel inefficient at
   `local_inter = 256`** (Coder-Next inter=512 / world=2). The grid
   shape may be launch-overhead-bound at this dim, or per-block work
   too small to amortise overhead. PP at full `inter = 512` doesn't
   hit this regime.
2. **TP per-rank kernel fans out over n_experts × top_k × n_tokens
   in a way that doesn't shrink with `local_inter`** — the
   per-(token, slot) block does work proportional to local_inter,
   but a fixed per-launch overhead per (token, slot) may dominate
   when local_inter is small.
3. **Q4_1 ffn_down on the down-projection MMVQ** has a slower kernel
   path than Q4_0 down. Same shape on PP (PP also has Q4_1 down)
   but maybe the Q4_1 MMVQ-per-token kernel scales worse than Q4_0
   at the TP per-rank shape.

## Fix options (CN-80B-11b)

**A. Profile inside `forward_moe_ffn_prefill_tp`**: split
`ptp_moe_ffn` into `gate_up`, `swiglu_quant`, `down`, `combine`
sub-marks. Pick the heaviest sub-section.

**B. Q4_1 down kernel inspection**: if (3) is the issue, port the
existing `indexed_moe_mmq_q4_0_down_tile8_dp4a` to Q4_1
(CN-80B-11c). Same kernel structure with the (q·d + m)
reconstruction. Would lift PP prefill too (Coder-Next pp4 currently
uses MMVQ fallback for the same Q4_1-down reason; tile8 with the
new Q4_1 down kernel would unlock the PP fast path).

**C. n_tokens-batched indexed MMVQ kernel rework**: collapse the
per-(token, slot) launch into one larger kernel that processes an
entire (n_tokens × top_k) batch with proper occupancy at small
local_inter.

Option B (Q4_1 indexed-MoE MMQ tile8) is the highest-confidence
lever — it's a documented gap (the `q4_0_use_tile8` dispatch
literally rejects Q4_1 down) AND fixes both pp4 and pp2tp2 at once.
Estimated 1-2 sessions: Q4_1 down tile8 kernel + dispatch wire +
parity cert. Q4_0 gate_up already has tile8 + sort scratch infra.

## Closes

- CN-80B-11a #139 — root cause re-identified with measured data.
  CN-80B-10 cert filed as DIAGNOSIS-WRONG; this v2 supersedes.
- Sets up CN-80B-11b #140 (Option A, in-MoE marks) and CN-80B-11c
  #141 (Option B, Q4_1 tile8 kernel — strongly recommended).
