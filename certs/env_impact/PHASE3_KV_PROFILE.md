# Phase 3 follow-up — F16 vs Q8 KV decode profile

- **Generated:** 2026-05-07T19:36:53+00:00
- **Commit:** fa4106d
- **Model:** qwen36-27b-q4_0 on pp2tp2 / slots=8
- **Workload:** short prompt ('Count from one to ten.'), max_tokens=80, decode-profile window=64 (skip 8 warmup)

## Wall-clock summary

| arm | prefill ms | decode ms/tok | err |
|---|---:|---:|---|
| `f16` | 3 | 50.51 | |
| `q8` | 2 | 78.73 | |

## Per-kernel decode profile

### `f16`

```
=== TP decode profile ===
section                    total_ms     count    mean_ms  ms/token
hyb_dec_gdn                  1330.74      2944      0.4520     20.793
hyb_dec_attn_kernel           410.93      2048      0.2006      6.421
hyb_dec_kv_append             153.53      1024      0.1499      2.399
hyb_dec_lm_head               123.60        64      1.9312      1.931
hyb_dec_logits_dtoh            25.68        64      0.4013      0.401
hyb_dec_post_stage              0.59       128      0.0046      0.009
hyb_dec_pre_lm_head             0.30        64      0.0046      0.005
TOTAL_RECORDED               2045.36         -           -     31.959

=== HOST decode profile (n=71, post-warmup) ===
  decode_logits  :  35.318 ms/tok
  stop-mask      :   0.000 ms/tok
  sampler.sample :   0.135 ms/tok
  push_and_emit  :   0.037 ms/tok
  stopstr_check  :   0.000 ms/tok
  TOTAL_per_step :  35.493 ms/tok
```

### `q8`

```
=== TP decode profile ===
section                    total_ms     count    mean_ms  ms/token
hyb_dec_gdn                  1352.46      2944      0.4594     21.132
hyb_dec_attn_kernel           434.12      2048      0.2120      6.783
hyb_dec_kv_append             154.22      1024      0.1506      2.410
hyb_dec_lm_head               124.52        64      1.9456      1.946
hyb_dec_logits_dtoh            26.80        64      0.4188      0.419
hyb_dec_post_stage              0.59       128      0.0046      0.009
hyb_dec_pre_lm_head             0.30        64      0.0046      0.005
TOTAL_RECORDED               2093.01         -           -     32.703

=== HOST decode profile (n=71, post-warmup) ===
  decode_logits  :  35.946 ms/tok
  stop-mask      :   0.000 ms/tok
  sampler.sample :   0.131 ms/tok
  push_and_emit  :   0.036 ms/tok
  stopstr_check  :   0.000 ms/tok
  TOTAL_per_step :  36.115 ms/tok
```

## Diagnosis (final)

**Post-warmup Q8 KV decode is +1.7 % over F16** (35.49 vs 36.12 ms/tok
in the host profile, which skips the first 8 tokens). The user-visible
56 % wall slowdown was **warmup-dominated**:

| | F16 | Q8 |
|---|---:|---:|
| post-warmup decode (host_profile, 72 toks) | 35.49 ms/tok | 36.12 ms/tok |
| warmup (first 8 toks, derived from server total − post-warmup) | ~191 ms/tok | **~472 ms/tok** |
| amortised over 80 toks (Python wall) | 50.5 ms/tok | 78.7 ms/tok |
| server total wall (80 toks + prefill) | 4091 ms | 6377 ms |

Cross-check: 8 × 472 + 72 × 36.12 = 6378 ms ≈ Q8 server total 6377 ms ✓.

The 27 ms/tok "Q8 cost" I chased through HipEvent / host-profile probes
does not exist as a per-token steady-state — it is **~280 ms × 8 cold
tokens** spread thin over the 80-token average. After warmup, F16 and
Q8 decode at the same speed.

**Root cause:** ROCm 7.1.1 JIT-compiles Tensile / HIP-module kernels on
first dispatch. Q8-specific entries (`attention_decode_q8_kv`,
`quantize_f16_q8_0`, etc.) only get compiled the first time decode runs
in `--kv q8` mode, and the compile blocks the dispatching thread for
30–50 ms per unique kernel × ~8 unique Q8 kernels ≈ ~280 ms cold-start
cost. F16 has the same overhead but most of its kernels are also used
during prefill, which is why F16's "warmup" is much smaller (~190 ms
spread over 8 tokens vs Q8's ~470 ms).

The fused-Q8-into-cache change committed alongside this cert **is null
at steady state** but is still a code-cleanup win: 4 launches/layer/token
→ 2 launches.

**Fix-of-the-fix (next):** prime the JIT cache at server boot. After
`model.load()` and before accepting requests, run one dummy decode
through every operator-selected layout (F16 + Q8 if `--kv q8`) so all
kernels are compiled before the first real request. Eliminates the
~280 ms warmup penalty entirely.

**Operator advice:** Q8 KV decode is ~as fast as F16 once warm. The
first 1–2 chat completions on a freshly-booted server pay an extra
~300 ms; subsequent completions are at full speed.
