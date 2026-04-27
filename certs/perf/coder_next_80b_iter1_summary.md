# CN-80B-4 (perf iter 1) — blocked by rocprofv3 4-rank limitation

**Result: no actionable profile data captured. Filed honestly; iter-2
pivots from "post-iter-1 profile" to "instrument forward with HipEvent
section timers" to give iters 2/3/4 real measurements.**

## What was attempted

`rocprofv3 --kernel-trace` on Coder-Next-80B at pp4 / L=512 — the
canonical pp4 prefill workload from CN-80B-3. Tried multiple modes:

| mode | result |
|---|---|
| `--kernel-trace --stats` | rocprofv3 SIGABRT mid-run; empty stats CSV |
| `--kernel-trace` only | rocprofv3 SIGABRT; empty trace CSV |
| `--runtime-trace` | rocprofv3 SIGABRT before any output dir created |

Cross-checked with the same modes on Qwen3.6-35B-A3B-UD-Q4_K_S pp4
(smaller, 4-rank): same SIGABRT. So **the failure is 4-rank specific,
not Coder-Next specific**. This matches the V2.30 memory note that
flagged rocprofv3 7.1.1 as "SIGABRTs on multi-rank finalize"; we
hoped only PMC mode was affected (kernel-trace worked fine on 9B tp2
in V1-BENCH-#114), but on this rig kernel-trace also breaks at ≥ 4
ranks.

## What we have analytically

The V2.30 35B-A3B-Q4_0 prefill profile (archived in
`certs/perf/v2_30_b_profile_tour.md`) is the closest available
proxy — same arch family (qwen35moe vs qwen3next, both hybrid
GDN+full-attn+MoE+shared, both Q4_0). Top kernels there:

```
mmq_q4_0_wave64                      18.0%   (now mmq_q4_0_4warp_lds via #112)
indexed_moe_mmvq_q8_0_dp4a           14.0%
dense_gemv_f32_f16  (router)          9.9%
indexed_moe_mmvq_q4_0_q8_1            9.3%
```

Coder-Next deltas:
- 512 experts vs 128 → router gemv 4× more rows (still tiny — F32 gemv
  hidden=2048 → 512 rows is ≤ 5 µs HBM-bound, ~1 % of decode wall by
  back-of-envelope).
- 48 layers vs 40 → all kernels 1.2× more calls.
- Top-k = 10 vs 8 → topk_f32 1.25× more rounds, on 4× larger expert
  array.
- Same ffn_*_exps quants (Q4_0/Q4_1) and shexp MXFP4→Q8_0.

Without rocprofv3 the precise hot-spot remains ambiguous — could be
indexed-MoE MMVQ at the new 512-expert indexing, or topk_f32 LDS
pressure, or something else entirely.

## Iter-1 verdict

**Inconclusive.** No reliable lever to pick blindly without instrumented
data. The honest options were:
- (a) pick a heuristic lever and pray — risks shipping a null change or
  regression
- (b) do the instrumentation work first so iters 2/3/4 have data

(b) is the disciplined call. Iter-2 (`CN-80B-5`) re-scopes from
"profile post-iter-1" to "add HipEvent section timers to
`forward_one_token_pp` / `forward_prefill_pp`, run on Coder-Next pp4,
report per-section ms breakdown".

## Coder-Next pp4 baseline preserved as the iter-1 starting point

```
pp4   L=128:  426.4    L=512:  537.6    L=2048:  555.6    tg64:  41.3
pp2tp2 L=128: 164.9    L=512:  179.9    L=2048:  181.9    tg64:  46.2
```

(from `certs/perf/v1_bench_matrix/qwen3_coder_next_80b.json`)

## Bonus A/B: FLAMBEAU_VARIANT=baseline vs default

Single quick experiment that doesn't need rocprofv3: flip the global
DP4A/fusion variant gate and measure the delta. Tells us whether the
DP4A-optimised path is on the critical path for this model.

| variant                          | pp512 | tg64  |
|----------------------------------|------:|------:|
| `FLAMBEAU_VARIANT=baseline` (no DP4A) | 567.0 | 32.6 |
| default (DP4A + fusion)          | 537.6 | **41.3** |

- **Decode +27 %** with default (DP4A) vs baseline. DP4A path is the
  decisive lever for the user-visible chat throughput.
- **Prefill −5 %** with default vs baseline. Small but real. Some
  DP4A-flavoured prefill kernel choice is mildly worse for this model's
  quant mix (Coder-Next-Q4_0 = Q4_0 gate/up + Q4_1 ffn_down_exps + MXFP4
  shexp + standard attn). One-rep, within typical variance — flagged
  for iter-3 if it reproduces.

Default (DP4A) stays — net positive, and decode regression matters more
than 5 % prefill noise.

## Closes

- CN-80B-4 #130 — filed with the rocprofv3 limitation. No code change,
  no perf delta. Iter-2 absorbs the remaining "profile" work; the
  FLAMBEAU_VARIANT A/B above is the iter-1 supplementary evidence
  that informs iter-2/3 priority.
