# B5 TP parity drift bisect — findings

## Reproduction (2026-04-26)

`Qwen3.6-35B-A3B-UD-Q4_K_S` on TP world=2 (gpus 0,1) gives the wrong
warm-decode argmax for seed=9419:

| run                                          | warm_argmax |
|---|---|
| PP w=4 reference (`forward_one_token_pp_real`)| **11** ✓ |
| TP w=2 default                                 | 220 ✗ |
| TP w=2, `FLAMBEAU_VARIANT=baseline`            | 1510 ✗ |
| TP w=2, all Q8_0 reverts + baseline            | 1510 ✗ |

Both TP runs are **deterministic** (same inputs → same wrong argmax across
multiple invocations). Different env-flag combinations select different
kernels, which produces different F32 reduce-orders → different wrong
arguments. The structural bug isn't kernel-level.

## What we ruled out

* **AR replication** — `FLAMBEAU_TP_PROBE` shows rank-0 and rank-1
  hidden_a are bit-identical after every AR (post-attn and post-FFN).
* **Router determinism** — both ranks compute identical
  `expert_ids = [238, 112, 106, 127, 157, 66, 120, 56]` and identical
  weights at layer 0 (router weight `ffn_gate_inp` is Replicated, input
  `mid_norm_f16` is bit-identical post-AR).
* **Kernel optimisations** — `FLAMBEAU_VARIANT=baseline` +
  `FLAMBEAU_Q8_0_MMVQ_T128_VDR2=off` + `FLAMBEAU_Q8_0_GU_T128_VDR2=off`
  still wrong. Bug is structural.
* **Q8_0 fused gate+up GDN path** — `FLAMBEAU_GDN_QKV_FUSE_Q8_0=off`
  forces unfused two-call mmvq, still wrong.
* **Shared expert** — `FLAMBEAU_TP_SKIP_SHARED=1` changes the wrong
  argmax (220 → 16190), confirming both MoE and shared paths
  contribute to the corrupted layer output, but isn't the sole bug.
* **Non-determinism** — every full forward (40 layers × greedy 64-step
  decode) reproduces the same `warm_argmax + last_id` exactly.
* **GDN-attn TP path on 9B** — `Qwen3.5-9B-Q4_1` (qwen35 hybrid with the
  *identical* GDN dims as 35B-A3B: num_v_heads=16, num_k_heads=16,
  head_v_dim=256, head_k_dim=128, d_inner=4096) gives the correct
  argmax=11 on TP w=2 with both default kernels and
  `FLAMBEAU_VARIANT=baseline`. So the GDN-attn slicing + AR + post-attn
  norm pipeline is structurally fine.

## The smoking gun

`FLAMBEAU_TP_LAYER0_BISECT=1` was added to dump matched mid-layer
states in PP and TP. At layer 0, after the post-attention residual
(`x_in + GDN_delta`), the two paths already diverge:

```
PP post-attn-residual  mid_f16:        L2=1.188769  head=[0.00430, 0.00880, 0.01736, 0.03006]
TP post-AR-attn       hidden_a:        L2=1.183390  head=[0.00438, 0.01004, 0.01813, 0.03250]
```

* Element 0: ~1 ULP difference (F16 noise floor) — agrees within tolerance.
* Element 1+: 5-14 % difference — well beyond F16 noise.

L2s are within ~0.5 % so the magnitude is right, but specific
elements diverge. The pattern (element-0 closer to PP than the rest)
suggests a **stride / offset / per-row corruption** rather than a
uniform scaling bug.

The post-attn residual is `embed + AR(per-rank GDN_delta)`. embed
is replicated and bit-deterministic; AR sums two FP32 partials. If
the AR sum or the GDN-delta partials had a uniform error, *all*
elements would shift; the structured per-element divergence implies
the GDN forward at TP is producing a delta that's element-wise
different from what PP would produce — not by a uniform scale.

## What's still on the table

1. **Q8_0 attn_qkv tensor slab order at TP w=2**. The FusedQkvParallel
   slicer was bug-fixed for `[Q | K | V]` on-disk layout (verified on
   9B). 35B-A3B has Q8_0 instead of 9B's Q4_1 — but both should walk
   the same on-disk layout. If 35B-A3B's GGUF had a different sub-slab
   order, world=1 would still pass (no slicing) and only world>=2 would
   surface the bug. Need to dequantise PP's full attn_qkv and TP's
   per-rank attn_qkv slabs to F32 and byte-compare to confirm the
   on-disk order matches the assumption.
2. **F32 attn_norm.weight conversion** — both loaders use
   `half::f16::from_f32` so bit-identical, but worth printing the
   loaded F16 norm bytes from each rank vs PP to be 100 % sure.
3. **GDN state initialisation** — both PP and TP `zero_f32` the GDN
   state and conv_history. Verified by code inspection but not by
   on-device byte-dump.

## Diagnostic infrastructure landed (this session)

* `FLAMBEAU_TP_LAYER0_BISECT=1` — dumps the per-rank components in
  TP layer 0 (post-gdn partial, post-AR-attn hidden_a, mid_norm_f16,
  router output, MoE partial pre/post shared-add, shared delta) and
  the matching mid_f16 / mid_norm_f16 in PP.
* `FLAMBEAU_TP_SKIP_SHARED=1` — skip the shared-expert add in TP MoE
  layers.
* `FLAMBEAU_GDN_QKV_FUSE_Q8_0=off` — force unfused two-MMVQ
  attn_qkv + attn_gate path in GDN-TP.
* `FLAMBEAU_TP_LAYER_LIMIT=N` (already existed) — run only the first N
  layers.

These knobs let next session step into the layer-0 internals at higher
resolution (probe internal GDN scratch buffers like q_norm/k_norm/v
post-l2norm, conv_input, gdn state pre/post step) and identify the
exact step where TP and PP diverge.

## Web research — on-disk QKV layout for hybrid Qwen GDN models

Investigated whether the on-disk `attn_qkv` layout for 35B-A3B might
differ from 9B (which would explain why the same FusedQkvParallel slicer
works on 9B but not on 35B-A3B). Findings:

* **Qwen3-Next (`qwen3next` arch)** — HF `Qwen3NextGatedDeltaNet.fix_query_key_value_ordering`
  reshapes its `mixed_qkvz` tensor as
  `(num_k_heads, 2·head_k_dim + 2·(num_v_heads/num_k_heads)·head_v_dim)`,
  giving an **interleaved-by-k-group** layout (per group: Q@hkd, K@hkd,
  V@gs·hvd, Z@gs·hvd) rather than flat `[Q | K | V]` blocks. llama.cpp
  PR #16095 (merged 2025-11-28) is the first version with native
  qwen3next support; that arch ships separate `attn_q`, `attn_k`,
  `attn_v`, `attn_z` tensors *or* a `qkvz` blob with a per-arch
  unpacking step.
* **Qwen3.5 / Qwen3.6 (`qwen35moe` arch in flambeau)** — different
  arch from qwen3next. PP w=4 on Qwen3.6-35B-A3B-UD-Q4_K_S decodes to
  the correct argmax=11 with the existing `[Q | K | V]` flat-block
  interpretation. If the on-disk layout were interleaved, PP would
  also fail — the silu_out → Q/K/V offset split (`gdn_tp.rs:374-377`,
  matched in PP) is the only place Q/K/V are unstuck, and any wrong
  layout interpretation would propagate identically through PP and TP.
  PP's correctness rules out an "on-disk layout differs from slicer
  assumption" bug for qwen35moe.

So the FusedQkvParallel slicer's `[Q | K | V]` assumption is correct
for 35B-A3B. The TP bug must be in something the per-rank slicing
exposes that PP's non-sliced path doesn't — narrowing the search to:

1. A kernel edge-case at the per-rank slab dimensions (`local_conv_channels=4096`,
   `hidden=2048`) that the kernel hits at TP w=2 but not at PP w=N where
   the full tensor is loaded.
2. A scratch-buffer alias / overwrite issue specific to the TP MoE+shared
   composition (since 9B with dense FFN works at TP w=2 with otherwise
   identical GDN dims).
3. An ordering / event-sync gap that's invisible on dense FFN but
   surfaces when MoE + shared expert produce two partials added
   in-place to `partial_ffn_out` before the FFN AR.

## Status

Open. Diagnostic tooling shipped (TP_LAYER0_BISECT, TP_SKIP_SHARED,
GDN_QKV_FUSE_Q8_0=off, plus matched PP dump in layer.rs). Bug
restricted to `qwen35moe` + Q8_0 `attn_qkv` + `hidden=2048` + MoE/shared
FFN combination. Layout-misinterpretation hypothesis ruled out by
web research + PP correctness. Next session should add probes at
GDN-internal checkpoints (post-attn-norm, post-attn_qkv per-rank,
post-conv1d per-rank, post-state-step) and explicitly test whether
swapping the MoE+shared composition for a synthetic `forward_moe_only`
or `forward_shared_only` path produces a correctness signal.
