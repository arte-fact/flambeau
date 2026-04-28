# CN-80B-7 (perf iter 4) — intra-GDN profile + Q4_0 qkv+gate fusion

## Profile (intra-GDN marks added in iter-4)

`forward_gdn_decode` instrumented with 11 section markers
(`gdn_norm_quant`, `gdn_proj_qkv_gate`, `gdn_proj_alpha_beta`,
`gdn_conv1d`, `gdn_silu`, `gdn_l2norm_qk`, `gdn_state_step`,
`gdn_ssm_norm`, `gdn_swiglu_quant`, `gdn_ssm_out`, `gdn_cast_f16`).
Coder-Next-80B pp4 decode tg=32:

| GDN sub-section      | total_ms | mean_ms | % of GDN |
|----------------------|---------:|--------:|---------:|
| gdn_ssm_out          |   109.11 |   0.095 |   30.5 % |
| gdn_proj_qkv_gate    |    66.26 |   0.058 |   18.5 % |
| gdn_state_step       |    33.94 |   0.029 |    9.5 % |
| gdn_norm_quant       |    25.13 |   0.022 |    7.0 % |
| gdn_l2norm_qk        |    18.87 |   0.016 |    5.3 % |
| gdn_conv1d           |    18.56 |   0.016 |    5.2 % |
| gdn_proj_alpha_beta  |    14.65 |   0.013 |    4.1 % |
| gdn_swiglu_quant     |    14.47 |   0.013 |    4.0 % |
| gdn_ssm_norm         |    11.86 |   0.010 |    3.3 % |
| gdn_silu             |    10.40 |   0.009 |    2.9 % |
| gdn_cast_f16         |     9.77 |   0.008 |    2.7 % |
| (sync/start)         |    13.45 |       — |    3.7 % |

#1 = `gdn_ssm_out` (Q5_K MMVQ over `[hidden=2048, d_inner=4096]`,
5.5 MiB weight read per call). Already on the optimal kernel variant
(`qmatmul_q5_K_mmvq_nw1_r2_gfx906`); per-call wall is ~12× HBM
ceiling, dominated by Q5_K dequant compute. No quick lever.

#2 = `gdn_proj_qkv_gate` (attn_qkv + attn_gate hidden-input
projections). Coder-Next stores both as **Q4_0**, but the existing
fuse-into-`mmvq_*_gate_up` check in `forward_gdn_decode` only fires
for **Q8_0** weights. The `mmvq_q4_0_gate_up` kernel exists (its own
docstring even names "attn_qkv+attn_gate in GDN" as the canonical
target) but the GDN forward never wired it. **Iter-4 lever**.

## Lever: extend GDN qkv+gate fusion to Q4_0

`forward_gdn_decode` now picks `mmvq_q4_0_gate_up` when both weights
are Q4_0, mirroring the existing Q8_0 branch. Asymmetric n_rows
(`conv_channels` for qkv, `d_inner` for gate) handled by the kernel's
`grid=max(...)` + per-output early-return, same shape contract as
the Q8_0 sibling.

## Verification

- F16 parity preserved: 35B-A3B prefill L=1 → 11 ✓ / L=2 → 271 ✓
  bit-exact vs llama.cpp (35B uses Q4_K not Q4_0 for attn_qkv, so the
  new branch is dormant on that model — confirms the dispatch
  doesn't regress existing paths).

## A/B Coder-Next-80B pp4

| metric  | iter-3   | iter-4   | Δ vs iter-3 |
|---------|---------:|---------:|------------:|
| pp128   | 431.4    | 436.5    | +1.2 %      |
| pp512   | 567.3    | 568.0    | +0.1 %      |
| pp2048  | 597.7    | 598.0    | +0.0 %      |
| tg64    |  41.1    |  41.2    | +0.2 %      |

**Null lever** in isolation — within rep variance at every L. Even
though the fusion saves 1 kernel launch + 1 redundant Q8_1 activation
read per GDN layer per token, the compute is dominated by Q4_0 weight
HBM reads (~12 MiB combined per call) and Q4_0 dequant. The 2 KiB
activation save is microscopic against that.

## Decision: keep

Even at null perf, the fix closes a documented dispatch gap — the
kernel was authored *for* this case per its own comments, and any
future Q4_0 GDN model inherits the fuse path automatically. The
intra-GDN markers added here also stay — they're no-op when the
profile timer is disabled and useful for any future iter that wants
to dig into the GDN body again.

## Cumulative iters 1–4 vs CN-80B-3 baseline

| metric  | baseline | post-iter-4 | cumulative Δ | dominant lever |
|---------|---------:|------------:|-------------:|---------------|
| pp128   | 426.4    | 436.5       | +2.4 %       | iter-3 router F16 |
| pp512   | 537.6    | 568.0       | **+5.7 %**   | iter-3 router F16 |
| pp2048  | 555.6    | 598.0       | **+7.6 %**   | iter-3 router F16 |
| tg64    |  41.3    |  41.2       | −0.2 %       | (noise)       |

Iter-3's F16 router is the only meaningful perf shift across the four
iters. Iter-1 was blocked by rocprofv3 4-rank (see iter-1 cert);
iter-2 shipped the HipEvent profiler infra; iter-3 picked the lever +
shipped the win; iter-4 closed a related dispatch gap (null perf,
positive code-quality).

## Open headroom

`gdn_ssm_out` at 30 % of GDN is the obvious next target but the
existing kernel is already on the best variant. Real movement there
needs either (a) a Q5_K MMVQ kernel rewrite (LDS scale caching,
DPP-merge across rows, etc.) or (b) a different quantisation strategy
for `ssm_out`. Both are V2 work.

## Closes

- CN-80B-7 #133 — intra-GDN profile shipped, Q4_0 fuse lever
  implemented, A/B null but kept on code-quality grounds.
