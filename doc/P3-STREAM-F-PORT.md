# P3 — Stream F port (the ~70 fixed kernels → CUDA)

**Status:** scoped 2026-06-18, not started. The work-list tracker for roadmap
phase **P3**; see `PORTABILITY-ROADMAP.md` for the phase ledger and
`CUDA-PORT-PLAN.md` §B4/B5 for the original grouping.

## What Stream F is

The 70 kernels under `crates/kernels-hip/src/kernels/` that are **not** matmul
families (mmvq / mmq / indexed_moe — all cross-backend as of P2). These are
*fixed* kernels: not a parameterized cross-product, so they port **once** to
CUDA with **one dispatch row each** (no autotune sweep over a candidate matrix).
The backend seam is `cp.async` (CUDA) vs prefetch (HIP) inside attention; almost
everything else is pure arithmetic + a block-reduce.

Currently on CUDA: `add_f32`, `quantize_q8_1`, `quantize_q8_1_mmq` only. So **69
remain** (add_f32 already done). All 70 are still HIP-only hand `.cu`.

## Harness spine (the real P3 prerequisite — discovered 2026-06-18)

The entire Stream F sweep surface is **HIP-only**: `sweep_f32_pointwise.rs` (9
fns) + ~20 sibling modules (`sweep_rmsnorm`, `sweep_swiglu`, `sweep_rope`,
`sweep_cast`, `sweep_softmax`, `sweep_attention*`, `sweep_gdn_*`, …) all use
`flambeau_backend_hip` directly, `KernelArgs`, and hardcoded `_gfx906` /
`backend:"hip"` / `rig()`. **None can emit a CUDA cert.** So P3's first
deliverable is the generic harness, analogous to the P1 matmul spine:
`crates/bench/src/sweep_pointwise_cb.rs` — the same sweeps over `BenchBackend`,
one driver certing both rigs. Each ported kernel moves from its HIP-only
`run_*_sweep(&Path)` to a generic `run_*_sweep::<B>(&Path, arch)` here, wired
into both the HIP `match op` arm and `sweep_cuda`. CUDA kernel = hand `.cu`
mirror in `kernels-cuda/src/kernels/` (precedent: `add_f32`, the quantizers).

**Pilot — `add_f16`** (the P3 spine proof): generic sweep module + CUDA `.cu`
mirror + both cli arms. HIP gfx906 ✅ (5/5 bit-exact, `certs/hip/gfx906/add_f16.json`).
CUDA sm_86 ✅ (5/5 bit-exact, `certs/cuda/sm_86/add_f16.json`) — landed once the
rig's `/dev/nvidia-uvm` returned (the earlier `cuInit 999` was the missing uvm
node, not a code issue). **The spine is validated end-to-end on both rigs**;
every subsequent Stream F kernel rides `run_*_sweep::<B>` here.

## Port mechanics (per kernel)

Same neutralize-to-`.cuh` recipe as Stream M where it fits: move the body to
`kernels-shared/include/<k>.cuh` (zero `#ifdef __HIP`), push any intrinsic
divergence to `arch_primitives/*` (the warp-size-agnostic reduce from P2a
already exists), add a manifest cell, retire the hand `.cu`, cert on **both**
rigs (rule 2), one dispatch row. For genuinely scalar elementwise kernels the
`.cuh` is a near-verbatim copy. **Harness caveat:** several families
(attention, norms, sampler) need a `BenchBackend`-generic sweep driver + a
host reference — some exist HIP-only and must be genericized; budget that as
part of each family's first kernel (as the MoE-lds sweep was built in P2d).

## Risk tiers

- **A — trivial** (per-element, no cross-lane): elementwise / cast / scale /
  activation. Neutral `.cuh` + manifest, minutes each. Derisk the Stream F
  harness here first.
- **B — block-reduce** (warp-size-agnostic reduce, already have the primitive):
  rmsnorm / l2_norm / softmax / topk / sampler / rope / quantize-act / kv_append.
- **C — hard**: attention (online softmax, LDS tiles, the cp.async↔prefetch
  seam, SWA masking + the NaN-init fixes already landed on HIP — port them
  too), `moe_sort_by_expert` (segmented sort — CUDA likely wants `cub`, HIP is
  hand-rolled; the one kernel that may not share a `.cuh`).

## Critical path = qwen35moe-v2 Q4_K single-GPU **decode** (the P5 gate)

qwen35moe is the **non-GDN** MoE model, so the **8 GDN kernels + causal_conv1d
are OFF the P5 path** (needed later for qwen3-coder-next). The **3
p2p_allreduce_* kernels are P4/B6** (multi-GPU), not single-GPU. Paged/batched
attention + KV variants are serving-path, not the parity gate. That leaves a
focused decode spine.

---

## Wave 1 — decode spine (unblocks first single-GPU qwen35moe decode → P5)

**1a — Tier-A leaves** (port first, derisk harness):
- [x] add_f16  *(spine pilot; HIP + CUDA bit-exact)*
- [x] cast_f16_f32  *(bit-exact both rigs)*
- [x] cast_f32_f16  *(bit-exact both rigs)*
- [x] scale_f16  *(bit-exact both rigs)*
- [x] scale_f32  *(bit-exact both rigs)*
- [x] silu_f32  *(1e-5 vs F64 oracle; fast `__expf` holds cross-backend)*
- [x] swiglu_f16  *(silu(gate)*up; 5e-3 F16 envelope)*
- [x] swiglu_f32_to_q8_1  *(fused activation→Q8_1; width-32 `__shfl_xor` reduce)*
- [x] sigmoid_mul_f16  *(logistic gate; 5e-3 F16 envelope)*
- [x] apply_softcap_f32  *(1e-5 vs F64 oracle; `tanhf`)*
- [x] split_q_gate_f16  *(3D-grid strided copy; bit-exact)*

**1a COMPLETE — all 11 Tier-A leaves cross-backend.** They ride
`sweep_pointwise_cb.rs`: a shared `run_unary_sweep<B, IE, OE>` driver (1 input +
optional f32 scalar), `run_binary_f16_sweep<B>` (2 F16 inputs), and two bespoke
drivers (the Q8_1 quantizer, the 3D split). A single
`run_pointwise_sweep<B>(op, …)` dispatcher the cli's HIP and CUDA arms both call
wires each kernel in one place. Each emits a neutral-impl_id cert (registered as
a bench-reference) alongside the frozen `_gfx906` dispatch cert. CUDA `.cu`
mirrors are auto-globbed by `kernels-cuda/build.rs`. **Next: 1b Tier-B norms /
rope / activation-quant** (block-reduce + index math).

**1b — Tier-B norms / rope / activation-quant** (block-reduce + index math):
- [x] rmsnorm_f16  *(every-layer decode norm; block_reduce_sum seam; 1e-2/sqrt(k))*
- [x] rmsnorm_f32  *(neutralized to shared `.cuh`; block_reduce_sum seam)*
- [x] rmsnorm_f32_to_f16  *(3 entries: _to_f16, _to_f16_add_residual, _f16_to_f16_add_residual)*
- [x] rmsnorm_f16_add_residual  *(fused add → mid + mid_norm, 2 outputs)*
- [x] rmsnorm_q8_1_fused  *(decode RMSNorm→Q8_1 edge; phase-2 already wave-agnostic)*
- [x] l2_norm_f32  *(same block_reduce seam; bit-exact-tol 1e-5 both rigs)*

**1b enabling primitive landed:** `kernels-shared/include/block_reduce.cuh` —
`block_reduce_sum<THREADS>` reduces block-wide and broadcasts via `WARP_SIZE` +
the neutral `warp_reduce_sum`, correct on wave64 (gfx906) AND wave32 (sm_8x).
The wave64-hardcoded reduce (`tid>>6`, `__shfl_xor(...,64)`, `WARPS=THREADS/64`)
in the hand kernels was the rule-5 leak; this seams it. `rmsnorm_f32` /
`l2_norm_f32` are neutralized to shared `.cuh` (`rmsnorm.cuh` / `l2_norm.cuh`) +
manifest cells, hand `.cu` retired (generated wrappers inject the arch header).
The remaining 1b norms reuse this primitive; rope/quantize add index/quant math
on top.
- [x] rope_f16  *(MIRROR — interleaved-pair index math; 5e-3)*
- [x] rope_neox_partial_f16  *(MIRROR — split-pair partial; 5e-3)*
- [ ] rmsnorm_rope_neox_partial_f16  *(DEFER — 4 HEAD_DIM entries carry per-entry
      `__launch_bounds__` (perf-critical on gfx906) the manifest codegen can't
      emit, and phase-1 has the wave64 reduce leak. Needs manifest
      `__launch_bounds__` support, or a MIRROR preserving the launch bounds on
      the gfx906 side + a wave32-correct reduce on CUDA.)*
- [x] quantize_f16_q8_1  *(MIRROR — backend-neutral 32-thread quantizer; dequant 5e-3)*
- [ ] quantize_f16_q8_1_mmq  *(orphan — no current consumer; defer)*
- [x] quantize_f16_q8_0  *(MIRROR — Q8 KV-cache quantizer; dequant 5e-3)*

**1b status: 11/13 done.** Remaining: rmsnorm_rope_neox_partial_f16 (launch-bounds
caveat above) + quantize_f16_q8_1_mmq (orphan). The block-reduce norm family +
rope + activation quantizers are all cross-backend.

**1c — Tier-C attention decode + KV append** (the hard tier):
- [ ] attention_decode_f16
- [ ] attention_decode_f16_splitk
- [x] softmax_masked_f16  *(NEUTRALIZE — `block_reduce_softmax`; NaN-safe; 5e-3)*
- [x] kv_append_f16_batched_slots  *(MIRROR — per-slot gather/scatter copy; bit-exact)*

**1d — MoE decode glue + sampler:**
- [x] topk_f32  *(NEUTRALIZE — per-token block argmax×k + softmax over the
      WARP_SIZE seam; indices exact + weights 1e-4 both rigs)*
- [ ] moe_sort_by_expert  *(Tier C — segmented sort)*
- [x] moe_combine_no_residual_f16  *(MIRROR — pointwise fan-in)*
- [x] moe_combine_f16  *(MIRROR — fan-in + residual)*
- [x] moe_combine_two_residuals_f16  *(MIRROR — fan-in + 2 residuals)*
- [x] apply_per_expert_scale_f32  *(MIRROR — trivial; bit-exact both rigs)*
- [x] shared_expert_scale_f32  *(NEUTRALIZE — block_reduce_sum dot + sigmoid scale)*
- [x] sampler_topk_softmax_f32  *(NEUTRALIZE — Phase-1 online reduce on
      `block_reduce_softmax`; bitonic sort wave-agnostic; indices exact + probs 1e-4)*
- [x] sampler_apply_penalties_f32  *(MIRROR — pointwise; 1e-5 vs F64, device FMA)*

**1d status: 8/9.** Only moe_sort_by_expert (Tier-C segmented sort) remains.
**The `block_reduce_softmax` primitive** (`block_reduce_softmax.cuh`, NaN-safe
online (max,sum) merge over the WARP_SIZE seam) now backs softmax_masked_f16 +
sampler_topk_softmax_f32, and is the online-softmax core for Wave 1c attention.

The MoE-combine fan-in kernels + the per-expert/shared-expert scales + router
top-k are cross-backend. Remaining 1d: moe_sort_by_expert (segmented sort —
Tier C, CUDA may want `cub`) + the two sampler kernels (reductions over the
vocab). The `block_reduce_sum` (sum) + the topk argmax pattern (value+index
butterfly over `WARP_SIZE`) cover most of what the samplers need.

→ **Gate: P5 single-GPU qwen35moe-v2 Q4_K decode parity** (logit-cosine ≥ 0.9999
vs llama.cpp CUDA). Minimal parity can run decode-only (1-token prompt); real
prompts need Wave 2 prefill.

## Wave 2 — prefill + Q8 KV + dense GEMV (full prompts, `--kv q8`, other dtypes)

- [ ] attention_prefill_flash_tile_f16
- [ ] attention_prefill_f16
- [ ] attention_prefill_q8_kv
- [ ] attention_prefill_flash_tile_q8_kv
- [ ] attention_decode_q8_kv
- [ ] attention_decode_q8_kv_splitk
- [ ] kv_append_v_unit_norm_f16
- [ ] v_unit_norm_per_head_f16
- [ ] dense_gemv_f16_f16
- [ ] dense_gemv_f32_f16
- [ ] gelu_f32_to_f16
- [ ] gelu_mul_f32
- [ ] swiglu_f32
- [ ] swiglu_f32_to_f16
- [ ] moe_combine_no_residual_f32
- [x] add_f32  *(already on CUDA)*

## Wave 3 — paged + batched serving paths

- [ ] attention_decode_f16_paged
- [ ] attention_prefill_f16_paged
- [ ] attention_decode_f16_batched
- [ ] kv_append_f16_paged_prefill
- [ ] kv_append_f16_paged_slots
- [ ] dense_gemv_f16_f16_batched
- [ ] dense_gemv_f32_f16_batched

## Wave 4 — GDN (qwen3-coder-next hybrid; not on the qwen35moe P5 path)

- [ ] gdn_alpha_beta_f32
- [ ] gdn_assemble_conv_input_f32
- [ ] gdn_split_qkv_f32
- [ ] gdn_state_step_f32
- [ ] gdn_state_step_alphabeta_f32
- [ ] gdn_state_step_alphabeta_f32_batched_slots
- [ ] gdn_conv_trio_decode_f32_batched_slots
- [ ] causal_conv1d_f32

## Deferred to P4/B6 — multi-GPU collectives (NOT single-GPU; NCCL replaces these)

- [ ] p2p_allreduce_residual
- [ ] p2p_allreduce_residual_rmsnorm
- [ ] p2p_allreduce_residual_rmsnorm_q8_1

(Per CUDA-PORT-PLAN §B6: CUDA implements the `Collectives` seam with NCCL +
separate fused residual/rmsnorm kernels, not by porting the BAR1 P2P kernels.
These three are HIP-only by design.)

## Counts

| Wave | Kernels | On P5 path |
| --- | --- | --- |
| 1 (decode spine) | 32 | ✅ yes |
| 2 (prefill + Q8 KV + gemv) | 16 (1 done) | partial (prompts) |
| 3 (paged + batched) | 7 | no (serving) |
| 4 (GDN) | 8 | no (other model) |
| P4/B6 (collectives) | 3 | no (multi-GPU) |
| **total** | **70** | |

Critical path to P5 = **Wave 1 (32 kernels)**. The hard cluster inside it is
attention_decode (×2) + softmax_masked + moe_sort_by_expert; everything else is
Tier A/B.
