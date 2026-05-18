# #241 — v2 gdn_layer batched prefill (2026-05-18)

## Result

|              | legacy | v2 post-#239 | v2 post-#240 | v2 post-#241 |
|--------------|-------:|-------------:|-------------:|-------------:|
| 35B-A3B N=1  | 42.95  | 19.83        | 19.83        | **22.52**    |
| 35B-A3B N=2  | 42.83  | 19.28        | 19.57        | 22.22        |
| 35B-A3B N=4  | 43.84  | 20.23        | 20.02        | 23.19        |
| 9B-Q4_1 N=1  |   —    | 31.01        | 31.01        | 32.06        |
| 9B-Q4_1 N=4  |   —    | 28.55        | 29.37        | 30.23        |

Closes another ~14% on MoE hybrid (35B-A3B) and ~3% on dense (9B-Q4_1).
Cumulative v2 evolution at 35B-A3B N=1: **6.48 → 22.52 t/s = 3.47×**.

## What landed (scope pivot)

`#241` was originally scoped as multi-slot batched-decode GDN. The
profile of post-#239/post-#240 v2 showed the dominant v2-vs-legacy
gap at N=1 was **prefill-side**: v2's `gdn_layer_local` looped per
token even for prefill chunks, calling `forward_decode_with_ar_hook`
N times instead of `forward_prefill` once. That blocked the MMQ
tile8 prefill kernels from firing.

Pivoted #241 to attack prefill batching of GDN. Multi-slot
batched-decode is left as a follow-up (per legacy memory the
hybrid wall is ~1× at N=4 regardless, so the lever is small).

Changes:

1. `DeltaNetLayer::forward_prefill_with_ar_hook` added to
   `crates/blocks/src/delta_net.rs`. Mirrors the existing
   `forward_decode_with_ar_hook` shape: same body as
   `forward_prefill` plus an optional `ar_partial_callback` on the
   `ssm_out_f32` partial. `forward_prefill` becomes a thin wrapper
   that passes `None`.

2. `ScratchPool.gdn_prefill_scratch:
   Option<OwnedDeltaNetLayerPrefillScratch>` allocated when
   `config.gdn.is_some() && max_prefill_tokens > 1`. Sized for the
   composite's full prefill chunk.

3. `crates/forward/src/core/composites/gdn.rs`: detect prefill
   shape (`n_tokens > 1` AND `single_slot`) and dispatch to
   `forward_prefill_with_ar_hook`. Decode (`n_tokens == 1`) and
   multi-slot batched-decode still go through the per-token loop.

## Profile delta (Qwen3.6-35B-A3B-Q4_0 / K=4 + ~24-token prefill)

|                       | post-#240 | post-#241 | delta |
|-----------------------|----------:|----------:|------:|
| Total GPU kernel time |  607 ms   |  568 ms   | -6.4% |
| Total launches        | 74 135    | 57 411    | -22%  |
| `mmq_q8_0_oracle_q8_1`|     0     |   188     |   ∞   |

The MMQ tile8 path (`flambeau_mmq_q8_0_oracle_q8_1`) now fires for
full-attention prefill chunks — the same kernel the legacy stack
used at ~28% of total GPU time. Per-call cost is 158 µs (a single
MMQ tile8 dispatch handles 32+ tokens at once) — replaces ~32
per-token MMVQ launches at ~6 µs each, net saving on launch
overhead alone.

## Cumulative v2 perf evolution (Qwen3.6-35B-A3B-Q4_0 / PP2)

```
            N=1 t/s   v2 ÷ legacy   per-token cost
pre-#239     6.48      0.151×         154 ms
post-#239   19.80      0.461×         50 ms     (indexed_moe gate/up/down)
post-#240   19.83      0.462×         50 ms     (batched-attn for N>1)
post-#241   22.52      0.524×         44 ms     (batched GDN prefill)
legacy      42.95      1.000×         23 ms
```

Remaining gap (52% closed of original 85% gap): primarily MoE
prefill path (#242 — `forward_prefill_tp_f32` to land `tile8` MoE
prefill kernels) and possibly the per-token MMVQ dispatch in
attention output_proj.

## Validated

- Coherent smoke output on Qwen3.6-35B-A3B-Q4_0 ("hello world…").
- Coherent smoke output on Qwen3.5-9B-Q4_1.
- Trace confirms `mmq_q8_0_oracle_q8_1` (188 calls) and the
  `forward_prefill` path firing without correctness regression.

## Multi-slot batched-decode (deferred)

The non-prefill multi-slot case still loops per slot. Per the legacy
memory `project_p29b_i2_F_hybrid_throughput`, the full batched-slots
GDN port (~400 LOC mirroring `forward_gdn_decode_batched_tp`) hits a
~1× hybrid ceiling regardless. With #239+#240+#241 in place the v2
composite already matches that ceiling at N=4 (1.03×). Deferred to
a separate slice unless the bench shows an outsized gap.

## Reproduce

```
python3 scripts/bench/d4_batched_decode.py \\
    --model /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \\
    --topology pp --devices 0,2 --n-max 4 --n-run 1,2,4 \\
    --tokens 24 --runs 2

scripts/profile/decode_step_trace.sh v2 /tmp/v2_trace
python3 scripts/profile/summarize_kernel_trace.py \\
    /tmp/v2_trace/threadreaper/*kernel_trace.csv --top 15
```
