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

## Decisive bisect — MoE expert routing chaos

`FLAMBEAU_PARITY_LAYER_DUMP=1` was extended to dump `expert_ids` per
layer in both PP (`layer.rs`, the existing `[layer-dump]` infra) and
TP (`tp.rs`, rank-0 only). Side-by-side diff of all 40 layers:

| layer | PP ∩ TP | PP-only experts | TP-only experts |
|---|---|---|---|
| 0 | **8/8** | — (sets identical, minor 56↔120 order swap at positions 6-7) | — |
| 1 | 3/8 | {19, 86, 112, 167, 214} | {3, 72, 143, 228, 243} |
| 2 | 4/8 | {49, 170, 187, 217} | {33, 58, 87, 132} |
| 3 | 2/8 | … | … |
| … | rapidly drops … | | |
| 31 | **0/8** | completely disjoint | |
| 36 | 0/8 | completely disjoint | |
| 39 | 1/8 | | |

At layer 0 the routers pick the **same set** of 8 experts but with
slightly different weights (PP weights at positions 6,7 = 0.0624,
0.0572; TP = 0.0674, 0.0650). That's enough to swap positions 6↔7 in
the top-k order and to feed slightly-different per-expert weights into
`moe_combine`. Layer 0's MoE FFN output therefore differs subtly
between PP and TP, which feeds layer 1's input, which produces a
significantly different router logit vector at layer 1, which picks
**different** top-k experts (only 3/8 overlap). Once the expert paths
fork, they never re-converge — by layer 31 they're completely disjoint.

## Root cause

Not a code bug; **fundamental property of intra-expert TP combined
with MoE topology chaos**.

* TP's per-rank intra-expert sharding is mathematically correct
  (`Σ_r out_r[h] = Σ_i full_W[h, i] · full_x[i]`), but the FP32
  reduction order across the AR sum + per-rank kernel reductions
  differs from the FP32 reduction order PP would compute internally
  to a single kernel.
* That ordering difference is small (~F32 ULPs at well-conditioned
  inputs), but cancellation-heavy operations like RMSNorm denominators
  can amplify it to several percent on individual elements.
* For dense FFN (Qwen3.5-9B, Qwen3.6-27B) the drift is bounded — a
  Lipschitz function turns drift in into proportional drift out, with
  no amplification. PP and TP both produce argmax=11 at decode despite
  per-element drift of similar magnitude.
* For MoE (Qwen3.6-35B-A3B) the router's `softmax → top-k → renormalize`
  is **chaotic at the boundary** — small drift in router logits can
  swap which 8 of 256 experts are picked, and once the expert subset
  diverges between PP and TP, the FFN computations are computing
  *different functions*, not just slightly-different values of the
  same function. Divergence compounds across 40 layers.

This is a known property of MoE+TP in the literature — vLLM,
DeepSpeed-MoE, and Megatron-MoE all default to **expert parallelism**
(each rank owns a subset of experts, all-to-all activation routing)
rather than intra-expert tensor parallelism precisely because the
former is robust to F32 reduction-order changes (each rank's outputs
go to the *same* final reduction in `moe_combine`, regardless of
topology).

## Resolution paths

1. **Switch to expert-axis sharding for MoE** (large refactor).
   `ffn_*_exps` would be ColParallel on dim=0 (expert axis); each rank
   owns a subset of experts and computes the full inter dim for its
   experts. Activations route via all-to-all. Bit-exact across
   topologies because the router output is replicated and each
   expert's compute is deterministic on its rank.
2. **Accept the drift, gate per-arch.** Mark
   `qwen35moe` (and other MoE arches) as TP-incompatible at intra-expert
   sharding. CLI `--mesh-mode tp` rejects MoE arches; expert-parallel
   support comes later. Dense and dense+GDN arches stay supported.
3. **Quality-gate, not bit-exact.** Run a perplexity / chat-quality
   delta on Qwen3.6-35B-A3B PP-vs-TP. If perplexity diff is within
   acceptable bounds and chat smoke passes, ship TP for MoE without a
   bit-exact gate. Document that argmax may differ from llama.cpp
   reference for MoE+TP runs.

(3) is the cheapest and matches the V1 "consumer-grade inference"
positioning — small drift is acceptable for chat. (1) is the
right long-term move and matches the multi-rig vision.

## Status

Diagnosed. Not a fixable-this-session bug; it's a topology/algorithm
choice. Diagnostic tooling shipped. Recommendation: pick (3) for V1
ship (file the perplexity delta cert as the gate), schedule (1) for
V2 alongside the broader MoE TP refactor for Qwen3-Coder and other
MoE arches (which already need scheduling work for the K-quant
alignment wall — task #52 / TP-7-arch).
