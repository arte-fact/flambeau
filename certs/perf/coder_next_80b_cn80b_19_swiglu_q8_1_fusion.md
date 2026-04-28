# CN-80B-19c/d — `swiglu_f32_to_q8_1` fusion

**Status:** shipped, default-on. +1.1% Coder-Next pp2tp2 decode.

## TL;DR

One new HIP kernel `flambeau_swiglu_f32_to_q8_1` collapses two `swiglu →
quantize-Q8_1` chains into one launch per call:

- **GDN tail** (`forward/gdn.rs` step 14+15, both single-rank and TP):
  was `swiglu_f32(z, out_normed) → gated_f32` then
  `quantize_q8_1(gated_f32) → gated_q8_1`. Now one kernel.
- **Shared-expert decode** (`forward/moe.rs` step 4+5, both single-rank
  and TP): was `swiglu_f32_to_f16(gate, up) → activated_f16` then
  `quantize_f16_q8_1(activated_f16) → activated_q8_1`. Now one kernel
  (also drops the F16 intermediate buffer entirely).
- **Routed-MoE decode TP path** (`forward/moe_tp.rs` step 4+5):
  same fusion at `[top_k, local_inter]`.

Default-on; `FLAMBEAU_VARIANT=baseline` opts back to the unfused chain.

CN-80B-19b (alpha+beta MMVQ fusion) was already shipping default-on
via `mmvq_q8_0_gate_up`; closed as duplicate.

## Numbers — Coder-Next pp2tp2 decode

`crates/models/qwen3-moe/tests/coder_next_pp2tp2_decode_fuse_ab.rs`
2 runs each, min, 32-token timed window, 2-step warmup, devices `[0,2,1,3]`.

| variant   | wall (32 tok) | tok/s | Δ      |
|-----------|---------------|-------|--------|
| baseline  | 711.8 / 709.6 | 45.10 | —      |
| **fused** | 707.9 / **702.0** | **45.59** | **+1.1%** |

Both fused runs faster than both baseline runs, so the +1.1% is
above noise even on a 32-token window.

Eager pp2tp2 decode (CN-80B-20 control): 45.30 tok/s — bracket sanity.

## Lever sizing — predicted vs measured

Predicted (1 launch saved per layer × 5 µs):
- 24 GDN layers × 1 = 24 launches/tok
- 48 shared-expert calls × 1 = 48 launches/tok
- Total: 72 launches/tok × 5 µs = ~360 µs/tok = ~1.6% on 22.2 ms baseline.

Measured: +1.1%. Slightly under the upper bound — the launch saving on
GDN is at the bottom of a chain that already has plenty of overlap, so
not every saved launch translates 1:1. Also the F16 intermediate buffer
elimination on shared expert is small absolute (4096 × 2 bytes × 48 =
393 KB / token; at HBM ~1 TB/s that's ~0.4 µs — negligible).

## Correctness

`crates/ops/tests/swiglu_f32_to_q8_1_parity.rs`:

> swiglu_f32_to_q8_1_matches_unfused_chain ... ok

Byte-exact equivalence vs the unfused chain across n ∈ {256, 1024,
4096, 8192} on F32 inputs sampled across mixed magnitudes. Reductions
(`__shfl_xor` over 32 lanes for both amax and sum) are bit-identical
between fused and unfused, so zero drift is expected and confirmed.

## Code

- **Kernel:** `crates/kernels-hip/src/kernels/swiglu_f32_to_q8_1.cu`
- **Op wrapper:** `crates/ops/src/hip/mlp.rs::swiglu_f32_to_q8_1`
- **Module list:** `crates/ops/src/hip/mod.rs::KERNEL_STEMS` —
  `"swiglu_f32_to_q8_1"`
- **Wire-ins:**
  - `crates/models/qwen3-moe/src/forward/gdn.rs` step 14+15
  - `crates/models/qwen3-moe/src/forward/gdn_tp.rs` step 14+15
  - `crates/models/qwen3-moe/src/forward/moe.rs` shared expert step 4+5
  - `crates/models/qwen3-moe/src/forward/moe_tp.rs` routed MoE + shared expert step 4+5
- **Parity:** `crates/ops/tests/swiglu_f32_to_q8_1_parity.rs`
- **Bench:** `crates/models/qwen3-moe/tests/coder_next_pp2tp2_decode_fuse_ab.rs`

## Why not also fold ssm_norm in (full CN-80B-19c "tail")?

The original CN-80B-19c task framing was "ssm_norm + swiglu_z +
quantize → one kernel". `ssm_norm` is a per-head RMSNorm over
`head_v_dim` (typically 128 elements per row, num_v_heads rows);
`quantize_q8_1` operates per 32-element block. The reductions happen
at different widths, so a full three-op fusion needs a custom kernel
with two-stage reductions or per-row scratch — substantially more
work than the swiglu+quantize lift. Not landed in this session.
Followup: `CN-80B-19c-extra: full GDN tail incl. ssm_norm`.
