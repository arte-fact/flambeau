//! MoE building blocks — TopK router, IndexedMoE matmul (MMVQ r2, fused
//! gate+up, MMQ prefill), weighted combine.
//! Expert bucketing (required by `indexed_moe_mmq_q4_k`) lives here as
//! [`build_expert_buckets`]. Byte-compatible with `sweep_moe::build_expert_buckets`;
//! the two will be folded into one helper when consumes this.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — every unsafe block is a `kernel.launch` or `memcpy_async` \
              over `DevicePtr`s validated by the caller; the kernel stem + entry are \
              resolved through the validated registry and the ABI matches the kernels \
              extern-C signature."
)]

use anyhow::Result;
use flambeau_backend_hip::{HipStream, KernelArgs, LaunchCfg};
use flambeau_core::{DevicePtr, MOE_SORT_MAX_EXPERTS, TOPK_MAX_EXPERTS};

use super::OpsRegistry;

/// `MMQ_X` for `indexed_moe_mmq_q4_k` — bucket width. Fixed in the kernel
/// source.
pub const INDEXED_MOE_MMQ_X: usize = 8;

/// `MMQ_Y` for `indexed_moe_mmq_q4_k` — output rows per block.
pub const INDEXED_MOE_MMQ_Y: usize = 16;

/// Shape scalars for indexed-MoE MMQ launchers. All five tile8/turbo
/// kernels take this identical tuple; grouping it gives named fields at
/// call sites and one-point change for future additions.
/// For the `down` kernels (which process per-pair activations as effective
/// tokens) `n_tokens` is set to `n_pairs = original_n_tokens * top_k` and
/// `top_k` is set to `1` — the kernel's Y indexing collapses correctly.
#[derive(Debug, Clone, Copy)]
pub struct MoeShape {
    /// Output row count (gate/up: `inter`; down: `hidden`).
    pub n_rows: usize,
    /// Y activation row axis.
    pub n_tokens: usize,
    /// Experts per token; 1 for the down kernels.
    pub top_k: usize,
    /// Super-blocks per weight row (= K / QK_K).
    pub n_sb_per_row: usize,
    /// Total experts across the MoE layer.
    pub n_experts: usize,
    /// Upper bound on `padded_total` (= `total_pairs + n_experts * 8` for
    /// the pad-to-8 sort). Kernel early-exits past the on-device
    /// actual padded_total.
    pub padded_total_upper_bound: usize,
}

/// TopK router over per-token logits. Emits `(token, slot)` → expert index
/// plus normalised softmax weights over the k selected experts per token.
/// Shapes: `logits[n_tokens, n_experts]` F32 in; `idx[n_tokens, k]` i32 out;
/// `weights[n_tokens, k]` F32 out.
pub fn topk_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    logits: DevicePtr,
    idx: DevicePtr,
    weights: DevicePtr,
    n_tokens: usize,
    n_experts: usize,
    k: usize,
) -> Result<()> {
    // Kernel's compile-time ceiling — must match `#define TOPK_MAX_EXPERTS`
    // in `kernels-hip/src/kernels/topk_softmax.cu`. Pre-this was
    // hardcoded at 128 on both sides, silently dropping experts 128..255 on
    // Qwen3.6 and triggering OOB LDS writes at the cross-warp reduce.
    assert!(
        n_experts <= TOPK_MAX_EXPERTS,
        "topk_f32: n_experts {n_experts} > {TOPK_MAX_EXPERTS} (bump TOPK_MAX_EXPERTS in core::kernel_limits + .cu #define + add cert shape)"
    );
    let module = reg.expect_module("topk_f32")?;
    let kernel = module.kernel("flambeau_topk_softmax_f32")?;

    let n_tokens_i = n_tokens as i32;
    let n_experts_i = n_experts as i32;
    let k_i = k as i32;
    let l_ptr: u64 = logits.as_usize() as u64;
    let i_ptr: u64 = idx.as_usize() as u64;
    let w_ptr: u64 = weights.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&l_ptr);
    args.push(&i_ptr);
    args.push(&w_ptr);
    args.push(&n_tokens_i);
    args.push(&n_experts_i);
    args.push(&k_i);
    // blockDim must be a multiple of wave64 and ≥64 so `n_warps = blockDim>>6`
    // is non-zero. Threads past n_experts load -INF via the kernel's guard.
    let block = (n_experts.max(64).div_ceil(64) * 64) as u32;
    let cfg = LaunchCfg::one_d(n_tokens as u32, block);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Indexed MoE MMVQ (r2 variant) — the decode-path MoE matmul. Half the
/// launches of single-row, 2 output rows per wave64.
/// Shapes:
/// - `w[n_experts, n_rows, n_sb_per_row]` Q4_K blocks
/// - `y[n_tokens, n_sb_per_row * 8]` Q8_1 blocks
/// - `expert_ids[n_tokens, top_k]` i32
/// - `dst[n_tokens, top_k, n_rows]` F32
pub fn indexed_moe_mmvq_q4_k_r2(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    // productisation: shape-aware — r4 (quarter-wave) at prefill
    // (n_tokens ≥ 32, launch-overhead-bound), r2 (half-wave) at decode
    // (n_tokens < 32, per-thread-work-bound). Measured r4 +8 % prefill but
    // -2 % decode vs r2; the split captures both.
    const R4_TOKEN_THRESHOLD: usize = 32;
    let (stem, entry, rows_per_block) = if n_tokens >= R4_TOKEN_THRESHOLD {
        ("indexed_moe_mmvq_q4_k_r4_dp4a", "flambeau_indexed_moe_mmvq_q4_k_r4_dp4a_q8_1", 4u32)
    } else {
        ("indexed_moe_mmvq_q4_k_r2_dp4a", "flambeau_indexed_moe_mmvq_q4_k_r2_dp4a_q8_1", 2u32)
    };
    let module = reg.expect_module(stem)?;
    let kernel = module.kernel(entry)?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(rows_per_block);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q6_K sibling of `indexed_moe_mmvq_q4_k_r2`. Same indexing contract —
/// `[n_tokens, top_k]` expert ids, `[n_tokens, top_k, n_rows]` F32 output,
/// `[n_tokens, n_sb_per_row * 8]` Q8_1 activations — but weights are
/// Q6_K super-blocks. Needed for UD-Q4_K_S-style mixed-quant GGUFs where
/// some `ffn_down_exps` are promoted from Q4_K to Q6_K.
/// Single-row kernel (64 threads per block, one wave64, one output row
/// per block); a multi-row r2/r4 variant is the follow-up perf lever.
pub fn indexed_moe_mmvq_q6_k(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q6_k")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q6_k_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 8.b-i2 — Q5_K indexed-MoE MMVQ. Needed for
/// Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL whose `ffn_down_exps`
/// promote to Q5_K on ~13/48 layers (UD mixed-quant). Same shape
/// contract as `indexed_moe_mmvq_q4_k` / `_q6_k`; inner arithmetic is
/// byte-identical to `mmvq_q5_k.cu` (5-bit = q4 nibble + high bit from qh).
pub fn indexed_moe_mmvq_q5_k(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q5_k")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q5_k_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 3.a — Q4_0 indexed-MoE MMVQ. Unblocks Qwen3.6-35B-A3B-Q4_0 whose
/// MoE expert weights are Q4_0 (gate+up+down in most layers). Same
/// contract as `indexed_moe_mmvq_q8_0`, 256 threads/block with VDR=2 DP4A.
pub fn indexed_moe_mmvq_q4_0(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q4_0")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q4_0_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q5_0 indexed-MoE MMVQ. Same contract as `indexed_moe_mmvq_q4_0` plus
/// the 5th-bit DP4A from `mmvq_q5_0`: per block, decode adds
/// `16 · sumi_bit · d_x · d_y` and subtracts `16 · d_x · s_y` for the
/// (q5 − 16) offset.
#[allow(clippy::too_many_arguments)]
pub fn indexed_moe_mmvq_q5_0(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q5_0")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q5_0_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q5_1 indexed-MoE MMVQ. Q5_0 indexed structure with the Q4_1-style
/// `m·s_y` correction (Q5_1 is affine: y = d·q5 + m).
#[allow(clippy::too_many_arguments)]
pub fn indexed_moe_mmvq_q5_1(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q5_1")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q5_1_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// B6 / 5.a — Q4_1 indexed-MoE MMVQ. Unblocks Qwen-published
/// Qwen3.6-35B-A3B-Q4_0 whose `ffn_down_exps` are Q4_1 (gate/up are Q4_0,
/// down is Q4_1). Same contract as `indexed_moe_mmvq_q4_0`; per-block
/// reconstruction differs (`m_x · s_y` instead of `-8 · d_x · s_y`).
pub fn indexed_moe_mmvq_q4_1(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q4_1")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q4_1_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 3.b.1 — fused gate+up Q4_0 indexed-MoE MMVQ. Reads each Q8_1
/// activation word once per block and produces both gate and up outputs,
/// halving the launch count for MoE Q4_0 decode vs calling
/// `indexed_moe_mmvq_q4_0` twice.
pub fn indexed_moe_mmvq_q4_0_gate_up(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q4_0_gate_up_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 2.a — Q8_0 indexed-MoE MMVQ. Unblocks UD-Q8_K_XL GGUFs whose MoE
/// expert weights stay Q8_0 instead of the usual Q4_K/Q4_K_S. Uses VDR=2
/// DP4A inside the inner loop (matches `mmvq_q8_0_dp4a_vdr2` pattern),
/// 256 threads/block, 1 output row per block. A multi-row r2/r4 variant
/// is the next perf lever; single-row is adequate for 2's unblock goal.
/// Contract:
/// * weights [n_experts, n_rows, n_blocks_per_row] Q8_0 blocks
/// * activations [n_tokens, n_blocks_per_row] Q8_1 blocks
/// * expert_ids [n_tokens, top_k] i32
/// * dst [n_tokens, top_k, n_rows] F32
pub fn indexed_moe_mmvq_q8_0(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q8_0")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q8_0_dp4a_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Fused gate+up MoE MMVQ (candle P30). One launch does both `gate = W_g · x`
/// and `up = W_u · x` reading `x` only once. Shapes match
/// `indexed_moe_mmvq_q4_k_r2` but with two separate weight tensors and two
/// separate F32 output tensors.
/// sorted-reorder variant of `indexed_moe_mmvq_q4_k_r2` (down
/// projection). Same ordering trick as the gate_up sorted kernel.
pub fn indexed_moe_mmvq_q4_k_r2_sorted(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q4_k_r4_sorted_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q4_k_r4_sorted_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: ((n_rows as u32).div_ceil(4), (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// fused gate+up tile8 MoE MMQ. Block = 64 rows × 8 slots = 512
/// outputs; all 8 slots guaranteed same expert via padded sort.
/// Weight tile decoded ONCE per thread per sub-block, reused across 8 cols.
/// Grid.y is an upper bound on padded_total / 8; kernel early-exits
/// blocks past the actual (on-device) padded_offsets[n_experts] → avoids
/// DtoH sync.
pub fn indexed_moe_mmq_q4_k_gate_up_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_k_gate_up_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_k_gate_up_tile8_dp4a_q8_1")?;

    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    // Upper-bound grid.y; in-kernel early-exit handles actual padded_total.
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// down-projection tile8 MoE MMQ. Same per-block layout as
/// the gate_up tile8; activation is indexed by pair_idx directly.
pub fn indexed_moe_mmq_q4_k_down_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_k_down_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_k_down_tile8_dp4a_q8_1")?;

    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 8.c — Q4_0 gate+up tile8 MoE MMQ sibling. Same contract as the Q4_K
/// `indexed_moe_mmq_q4_k_gate_up_tile8` wrapper; weight dtype is Q4_0 so
/// `n_sb_per_row` in the `MoeShape` should be set to `hidden / 32` (Q4_0
/// block size) by the caller, not `hidden / QK_K` as for Q4_K.
pub fn indexed_moe_mmq_q4_0_gate_up_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_0_gate_up_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 8.c — Q4_0 down tile8 MoE MMQ sibling.
pub fn indexed_moe_mmq_q4_0_down_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_0_down_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q5_0 gate+up tile8 MoE MMQ. Q4_0 tile8 structure + the 5th-bit
/// ladder: per-block dot is `d_x·(d_y·(sumi_nib + 16·sumi_bit) - 16·s_y)`.
#[allow(clippy::too_many_arguments)]
pub fn indexed_moe_mmq_q5_0_gate_up_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q5_0_gate_up_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q5_0_gate_up_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q5_0 down tile8 MoE MMQ.
pub fn indexed_moe_mmq_q5_0_down_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q5_0_down_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q5_0_down_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q5_1 gate+up tile8 MoE MMQ. Q5_0 tile8 structure with Q4_1-style
/// per-block reduction `d_x·d_y·(sumi_nib + 16·sumi_bit) + m_x·s_y`.
#[allow(clippy::too_many_arguments)]
pub fn indexed_moe_mmq_q5_1_gate_up_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q5_1_gate_up_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q5_1_gate_up_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q5_1 down tile8 MoE MMQ.
pub fn indexed_moe_mmq_q5_1_down_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q5_1_down_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q5_1_down_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q4_1 down tile8 MoE MMQ sibling of the Q4_0
/// down tile8. Same launch shape; the kernel's per-block dot uses the
/// Q4_1 affine `d_x·d_y·sumi + m_x·s_y` instead of Q4_0's bias-corrected
/// `d_x·(d_y·sumi - 8·s_y)`. Used by Coder-Next-Q4_0 (down=Q4_1).
pub fn indexed_moe_mmq_q4_1_down_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_1_down_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_1_down_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q4_1 gate+up tile8 MoE MMQ. Q4_0 gate+up structure with Q4_1's affine
/// reduction `d_x · d_y · sumi + m_x · s_y` per block, mirror of the
/// `indexed_moe_mmq_q4_1_down_tile8` sibling.
#[allow(clippy::too_many_arguments)]
pub fn indexed_moe_mmq_q4_1_gate_up_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_1_gate_up_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_1_gate_up_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 2.b — Q8_0 gate+up tile8 MoE MMQ sibling of the Q4_0 and Q4_K tile8
/// wrappers. Weight dtype is Q8_0 so `n_sb_per_row` in `MoeShape` is
/// `hidden / 32` (Q8_0 block size), matching the Q4_0 convention. Same
/// grid/block shape as Q4_0 tile8 — one wave64 per 64×8 output tile.
pub fn indexed_moe_mmq_q8_0_gate_up_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q8_0_gate_up_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q8_0_gate_up_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 2.b — Q8_0 down tile8 MoE MMQ sibling.
pub fn indexed_moe_mmq_q8_0_down_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q8_0_down_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q8_0_down_tile8_dp4a_q8_1")?;
    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 4.c: llamacpp-turbo 4-warp LDS-tiled indexed-MoE Q4_K gate+up MMQ.
/// MMQ_Y=128, MMQ_X=8 (aligned with padded sort), 256 threads/block.
/// Dual weight LDS tile (gate + up) + shared Y LDS tile with per-token
/// indirect gather. DS4 Q8_1 activation (per-token layout).
/// 4.d null result**: this kernel is slower than `_tile8` on Qwen3.6-35B
/// indexed-MoE workloads (−17 to −21 % end-to-end). Root cause: the turbo LDS
/// pattern amortises Y-LDS loads across many output cols (dense uses
/// MMQ_X=32-64); at MMQ_X=8 the LDS overhead dominates. Opt-in via
/// `FLAMBEAU_MOE_VARIANT=turbo`; default stays on `_tile8`.
pub fn indexed_moe_mmq_q4_k_gate_up_turbo(
    reg: &OpsRegistry,
    stream: &HipStream,
    gate_w: DevicePtr,
    up_w: DevicePtr,
    y_mmq: DevicePtr,           // DS4 Q8_1 activation — per-TOKEN layout (not per-pair)
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    shape: MoeShape,            // n_rows=inter, n_sb_per_row=hidden/QK_K
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_k_gate_up_turbo")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_k_gate_up_turbo_q8_1")?;

    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_mmq.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;

    let mut args = KernelArgs::new();
    args.push(&gw_ptr);
    args.push(&uw_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);

    // Grid: MMQ_Y=128, MMQ_X=8.
    let grid_x = (shape.n_rows as u32).div_ceil(128);
    let grid_y = (shape.padded_total_upper_bound as u32).div_ceil(8);
    // LDS: tile_y (MMQ_X=8 × 36 = 288 ints)
    // + 2 × TILE_X_TOTAL (TXS_QS=4224 + TXS_DM=128 + TXS_SC=528 = 4880 ints)
    // = 288 + 9760 = 10048 ints = 40192 B
    const SHARED_BYTES: u32 = 40448;
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (64, 4, 1),
        shared_bytes: SHARED_BYTES,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 4.c: Q4_K down sibling of `indexed_moe_mmq_q4_k_gate_up_turbo`.
/// Single weight matrix; activation indexed by per-pair sort; output also
/// indexed by per-pair.
pub fn indexed_moe_mmq_q4_k_down_turbo(
    reg: &OpsRegistry,
    stream: &HipStream,
    down_w: DevicePtr,
    y_mmq: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,            // n_tokens holds n_pairs; top_k unused (kernel signature lacks it)
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_k_down_turbo")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_k_down_turbo_q8_1")?;

    let n_rows_i = shape.n_rows as i32;
    let n_pairs_i = shape.n_tokens as i32;    // down's Y axis is per-pair
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = down_w.as_usize() as u64;
    let y_ptr: u64 = y_mmq.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;

    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_pairs_i);
    args.push(&nb_i);
    args.push(&n_experts_i);

    let grid_x = (shape.n_rows as u32).div_ceil(128);
    let grid_y = (shape.padded_total_upper_bound as u32).div_ceil(8);
    // LDS: tile_y (MMQ_X=8 × 36 = 288) + 1 × TILE_X (4880) = 5168 ints = 20672 B
    const SHARED_BYTES: u32 = 20992;
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (64, 4, 1),
        shared_bytes: SHARED_BYTES,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q6_K sibling of `indexed_moe_mmq_q4_k_down_tile8`. Same
/// tile layout (64 rows × 8 slot-cols, 1 wave64, padded-sort
/// per-block-expert invariant) with Q6_K decode (raw·y - 32·Σy bias
/// correction to avoid the byte-borrow bug). Used for UD-Q4_K_S
/// `ffn_down_exps` layers that are Q6_K-quantised.
/// 1.a — Q5_K down-projection MoE MMQ (tile8).
/// Same contract as `indexed_moe_mmq_q6_k_down_tile8` / `indexed_moe_mmq_q4_k_down_tile8`.
/// Closes the MMVQ-at-prefill hole for Qwen3-Coder-30B-A3B-UD-Q4_K_XL
/// (13/48 layers promote `ffn_down_exps` to Q5_K). Profiled as 31.56 %
/// of prefill wall in 0.b; MMQ variant mirrors Q6_K fix.
pub fn indexed_moe_mmq_q5_k_down_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q5_k_down_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q5_k_down_tile8_dp4a_q8_1")?;

    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

pub fn indexed_moe_mmq_q6_k_down_tile8(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    padded_offsets: DevicePtr,
    dst: DevicePtr,
    shape: MoeShape,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q6_k_down_tile8_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q6_k_down_tile8_dp4a_q8_1")?;

    let n_rows_i = shape.n_rows as i32;
    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let nb_i = shape.n_sb_per_row as i32;
    let n_experts_i = shape.n_experts as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&po_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    args.push(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// sorted-reorder variant of `indexed_moe_mmvq_q4_k_gate_up`
/// that takes `sorted_pair_idx` (produced by `moe_sort_by_expert`) and
/// remaps `blockIdx.y` → original (token, slot). Adjacent blocks thus
/// share the same expert → L2 cache reuse on weight tiles.
/// Must be called after `moe_sort_by_expert` has populated
/// `sorted_pair_idx[total]` with the sort permutation.
pub fn indexed_moe_mmvq_q4_k_gate_up_sorted(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    sorted_pair_idx: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q4_k_gate_up_r4_sorted_dp4a")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q4_k_gate_up_r4_sorted_dp4a_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&s_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    // r4 shape: 4 rows/block, grid.x = n_rows / 4.
    let cfg = LaunchCfg {
        grid: ((n_rows as u32).div_ceil(4), (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

pub fn indexed_moe_mmvq_q4_k_gate_up(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    // productisation: r4 variant — 4 rows per block (quarter-warp
    // per row) halves block count vs r2 and quarters vs the baseline.
    // Measured +19 % pp=512 on Qwen3.6-35B-A3B Mesh<2>, +7 % decode.
    // Parity bit-exact with llama.cpp on 8-token greedy.
    let stem = "indexed_moe_mmvq_q4_k_gate_up_r4_dp4a";
    let entry = "flambeau_indexed_moe_mmvq_q4_k_gate_up_r4_dp4a_q8_1";
    let module = reg.expect_module(stem)?;
    let kernel = module.kernel(entry)?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    // r4: 4 rows per block (quarter-wave-per-row).
    let cfg = LaunchCfg {
        grid: ((n_rows as u32).div_ceil(4), (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Indexed MoE MMQ Q4_K — prefill path. Caller pre-sorts (token, slot) pairs
/// into per-expert buckets of up to MMQ_X=8 refs. See [`build_expert_buckets`].
/// Shapes:
/// - `w[n_experts, n_rows, n_sb_per_row]` Q4_K blocks
/// - `y[n_tokens, n_sb_per_row * 8]` Q8_1 blocks
/// - `bucket_expert[n_buckets]` i32
/// - `bucket_slots[n_buckets, MMQ_X]` i32 packed `(token << 16 | slot)`, `-1` sentinel
/// - `dst[n_tokens, top_k, n_rows]` F32
pub fn indexed_moe_mmq_q4_k(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    bucket_expert: DevicePtr,
    bucket_slots: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_sb_per_row: usize,
    top_k: usize,
    n_buckets: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_k")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_k_q8_1")?;

    let n_rows_i = n_rows as i32;
    let nb_i = n_sb_per_row as i32;
    let top_k_i = top_k as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let be_ptr: u64 = bucket_expert.as_usize() as u64;
    let bs_ptr: u64 = bucket_slots.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&be_ptr);
    args.push(&bs_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&nb_i);
    args.push(&top_k_i);
    let grid_x =
        (n_rows as u32).div_ceil(INDEXED_MOE_MMQ_Y as u32);
    let cfg = LaunchCfg {
        grid: (grid_x, n_buckets as u32, 1),
        block: (128, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Shared-expert gate scaling (Qwen3.5/3.6 hybrid). Computes per-token
/// `gate[t] = sigmoid(Σ_i gate_w[i] · x[t, i])` and multiplies each row of
/// `shared_out` by its token's `gate` in-place.
/// Caller typically computes `shared_out` first via gate/up/swiglu/down
/// dense FFN using the existing `qmatmul` + `swiglu_f16` ops on the
/// `ffn_*_shexp` weights, then calls this to apply the learned scalar gate.
pub fn shared_expert_scale_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    shared_out: DevicePtr,  // in-place [n_tokens, hidden]
    x: DevicePtr,           // [n_tokens, hidden] (layer input)
    gate_w: DevicePtr,      // [hidden]
    n_tokens: usize,
    hidden: usize,
) -> Result<()> {
    let module = reg.expect_module("shared_expert_scale_f32")?;
    let kernel = module.kernel("flambeau_shared_expert_scale_f32")?;
    let n_tokens_i = n_tokens as i32;
    let hidden_i = hidden as i32;
    let so_ptr: u64 = shared_out.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let w_ptr: u64 = gate_w.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&so_ptr);
    args.push(&x_ptr);
    args.push(&w_ptr);
    args.push(&n_tokens_i);
    args.push(&hidden_i);
    let cfg = LaunchCfg::one_d(n_tokens as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Weighted combine + residual — `out = residual + Σ_k w_k · expert_out[k]`.
/// One thread per `(token, hidden)` slot. All tensors F16 except
/// `weights[n_tokens, top_k]` F32.
pub fn moe_combine_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    expert_outs: DevicePtr,
    weights: DevicePtr,
    residual: DevicePtr,
    out: DevicePtr,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
) -> Result<()> {
    let module = reg.expect_module("moe_combine_f16")?;
    let kernel = module.kernel("flambeau_moe_combine_f16")?;

    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let hidden_i = hidden as i32;
    let e_ptr: u64 = expert_outs.as_usize() as u64;
    let w_ptr: u64 = weights.as_usize() as u64;
    let r_ptr: u64 = residual.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&e_ptr);
    args.push(&w_ptr);
    args.push(&r_ptr);
    args.push(&o_ptr);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&hidden_i);
    let total = n_tokens * hidden;
    let cfg = LaunchCfg::one_d(total.div_ceil(256) as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// `moe_combine_f16` variant without an input residual.
/// `out[token, d] = Σ_k weight[token, k] * expert_outs[token, k, d]`.
/// Used by the TP-sharded MoE forward where the residual stream is
/// folded later by the AllReduce-residual kernel.
pub fn moe_combine_no_residual_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    expert_outs: DevicePtr,
    weights: DevicePtr,
    out: DevicePtr,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
) -> Result<()> {
    let module = reg.expect_module("moe_combine_no_residual_f16")?;
    let kernel = module.kernel("flambeau_moe_combine_no_residual_f16")?;

    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let hidden_i = hidden as i32;
    let e_ptr: u64 = expert_outs.as_usize() as u64;
    let w_ptr: u64 = weights.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&e_ptr);
    args.push(&w_ptr);
    args.push(&o_ptr);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&hidden_i);
    let total = n_tokens * hidden;
    let cfg = LaunchCfg::one_d(total.div_ceil(256) as u32, 256);
    // SAFETY: args reference live device pointers + CPU values; kernel
    // writes hidden-element F16 output. Caller's contract.
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 3.a.2 — `moe_combine_f16` variant that accepts two F16 residuals and
/// sums them inline. Saves one `add_f16` launch per layer per token on the
/// shared-expert path (`moe_residual = mid + shared_delta`).
pub fn moe_combine_two_residuals_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    expert_outs: DevicePtr,
    weights: DevicePtr,
    residual1: DevicePtr,
    residual2: DevicePtr,
    out: DevicePtr,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
) -> Result<()> {
    let module = reg.expect_module("moe_combine_two_residuals_f16")?;
    let kernel = module.kernel("flambeau_moe_combine_two_residuals_f16")?;

    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let hidden_i = hidden as i32;
    let e_ptr: u64 = expert_outs.as_usize() as u64;
    let w_ptr: u64 = weights.as_usize() as u64;
    let r1_ptr: u64 = residual1.as_usize() as u64;
    let r2_ptr: u64 = residual2.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&e_ptr);
    args.push(&w_ptr);
    args.push(&r1_ptr);
    args.push(&r2_ptr);
    args.push(&o_ptr);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&hidden_i);
    let total = n_tokens * hidden;
    let cfg = LaunchCfg::one_d(total.div_ceil(256) as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Group `(token, slot)` pairs by expert into buckets of up to [`INDEXED_MOE_MMQ_X`]
/// refs. Returns `(bucket_expert, bucket_slots)` where `bucket_slots[i, col]`
/// is either `token << 16 | slot` or `-1` sentinel padding. Deterministic —
/// experts are emitted in ascending-id order.
/// Implementation (C4): counting-sort over `expert_id`. Two linear passes
/// over `expert_ids`, no `HashMap`, no allocations beyond the output vecs
/// and a `Vec<u32>` of size `max_expert_id + 1`. The old `HashMap` path
/// showed up in `forward_*_prefill` profiles; for Qwen3.6 with 256 experts
/// that's a ~256-entry dense Vec — trivially cheaper than the HashMap
/// round-trip.
pub fn build_expert_buckets(
    expert_ids: &[i32],
    n_tokens: usize,
    top_k: usize,
) -> (Vec<i32>, Vec<i32>) {
    let total = n_tokens * top_k;
    if total == 0 {
        return (Vec::new(), Vec::new());
    }

    // Pass 1: find `max_expert_id` + count pairs per expert.
    let max_expert = expert_ids.iter().copied().max().unwrap_or(0);
    // Experts < 0 are invalid — caller contract — but guard cheaply so we
    // don't OOB the counts array. `as usize` casts negatives to giant
    // positives; mask first.
    debug_assert!(
        expert_ids.iter().all(|&e| e >= 0),
        "expert_ids must be non-negative"
    );
    let n_experts = (max_expert as usize) + 1;
    let mut counts = vec![0u32; n_experts];
    for &e in expert_ids {
        counts[e as usize] += 1;
    }

    // Pass 2: compute per-expert write offsets by prefix-summing counts
    // after inflating each to its bucket-padded count
    // (`ceil(count / MMQ_X) * MMQ_X`). This sizes the output exactly.
    let mut starts = vec![0usize; n_experts];
    let mut padded_total = 0usize;
    for e in 0..n_experts {
        starts[e] = padded_total;
        let c = counts[e] as usize;
        if c > 0 {
            let padded = c.div_ceil(INDEXED_MOE_MMQ_X) * INDEXED_MOE_MMQ_X;
            padded_total += padded;
        }
    }
    let n_buckets = padded_total / INDEXED_MOE_MMQ_X;

    // Allocate output in one shot and initialise to -1 (the pad sentinel).
    let mut bucket_slots = vec![-1i32; padded_total];
    let mut bucket_expert = Vec::with_capacity(n_buckets);
    // One bucket_expert entry per MMQ_X chunk. We can fill this up-front
    // because we already know each expert contributes
    // `count.div_ceil(MMQ_X)` chunks in ascending-id order.
    for e in 0..n_experts {
        let c = counts[e] as usize;
        if c == 0 {
            continue;
        }
        let chunks = c.div_ceil(INDEXED_MOE_MMQ_X);
        for _ in 0..chunks {
            bucket_expert.push(e as i32);
        }
    }

    // Pass 3 (over pairs): scatter each packed pair into its expert's
    // reserved region, bumping a per-expert cursor. `starts` acts as the
    // cursor — we dense-overwrite the -1 pads for real refs.
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let e = expert_ids[t * top_k + slot] as usize;
            let packed = ((t as i32) << 16) | (slot as i32);
            let dst = starts[e];
            bucket_slots[dst] = packed;
            starts[e] = dst + 1;
        }
    }

    (bucket_expert, bucket_slots)
}

// ---------------------------------------------------------------------------
// sort (token, slot) pairs by expert_id so same-expert groups can
// be processed by a real MMQ kernel (instead of the current per-pair
// MMVQ at prefill).
// Given `expert_ids[total]` (total = n_tokens * top_k) with values in
// [0, n_experts), produces:
// counts[n_experts] — #pairs per expert (atomic histogram)
// offsets[n_experts + 1] — exclusive prefix-sum, offsets[n_experts]=total
// sorted_pair_idx[total] — input pair indices grouped by expert
// Caller must zero `counts` before invocation. `cursors` is a scratch
// buffer of n_experts ints used internally by the scatter kernel.
// Three sequential kernel launches; no host round-trip.
// ---------------------------------------------------------------------------
pub fn moe_sort_by_expert(
    reg: &OpsRegistry,
    stream: &HipStream,
    expert_ids: DevicePtr,      // [total] i32
    counts: DevicePtr,          // [n_experts] i32, pre-zeroed
    offsets: DevicePtr,         // [n_experts + 1] i32 (written)
    cursors: DevicePtr,         // [n_experts] i32 scratch (written)
    sorted_pair_idx: DevicePtr, // [total] i32 (written)
    total: usize,
    n_experts: usize,
) -> Result<()> {
    assert!(
        n_experts <= MOE_SORT_MAX_EXPERTS,
        "moe_sort_by_expert: n_experts {n_experts} > {MOE_SORT_MAX_EXPERTS} (bump MOE_SORT_MAX_EXPERTS in core::kernel_limits + matching `#define` in kernels-hip/src/kernels/moe_sort.cu)"
    );
    let module = reg.expect_module("moe_sort_by_expert")?;
    let k_zero = module.kernel("flambeau_moe_sort_zero_counts")?;
    let k_count = module.kernel("flambeau_moe_sort_count")?;
    let k_scan = module.kernel("flambeau_moe_sort_scan_offsets")?;
    // deterministic single-thread scatter so sorted_pair_idx
    // is bit-reproducible across runs and ranks. The racing-atomic
    // variant was opt-in via FLAMBEAU_MOE_SCATTER=race (deleted in S3 —
    // it corrupted TP determinism and the perf gain was marginal).
    let k_scatter = module.kernel("flambeau_moe_sort_scatter_det")?;

    let total_i = total as i32;
    let n_experts_i = n_experts as i32;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let c_ptr: u64 = counts.as_usize() as u64;
    let o_ptr: u64 = offsets.as_usize() as u64;
    let k_ptr: u64 = cursors.as_usize() as u64;
    let s_ptr: u64 = sorted_pair_idx.as_usize() as u64;

    // Kernel 0: zero counts
    {
        let mut args = KernelArgs::new();
        args.push(&c_ptr);
        args.push(&n_experts_i);
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (512, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_zero.launch(stream, cfg, args)? };
    }
    // Kernel 1: histogram
    {
        let mut args = KernelArgs::new();
        args.push(&e_ptr);
        args.push(&c_ptr);
        args.push(&total_i);
        const BLOCK: u32 = 256;
        let grid = (total as u32).div_ceil(BLOCK);
        let cfg = LaunchCfg::one_d(grid, BLOCK);
        unsafe { k_count.launch(stream, cfg, args)? };
    }
    // Kernel 2: scan to offsets + init cursors (single block, 512 threads)
    {
        let mut args = KernelArgs::new();
        args.push(&c_ptr);
        args.push(&o_ptr);
        args.push(&k_ptr);
        args.push(&n_experts_i);
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (512, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_scan.launch(stream, cfg, args)? };
    }
    // Kernel 3: scatter (deterministic by default, racing on opt-in)
    {
        let mut args = KernelArgs::new();
        args.push(&e_ptr);
        args.push(&k_ptr);
        args.push(&s_ptr);
        args.push(&total_i);
        // Deterministic single-thread scatter — see kernel comment.
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_scatter.launch(stream, cfg, args)? };
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// padded variant of `moe_sort_by_expert` that rounds each expert's
// range to a multiple of 8. Produces BOTH the standard (unpadded) outputs
// AND a `sorted_pair_idx_padded[total_padded]` / `padded_offsets[n_experts+1]`
// pair where padded slots repeat the last real pair_idx (so an 8-slot
// per-block MMQ kernel can assume all 8 slots in its block share an expert).
// ---------------------------------------------------------------------------
/// 1.b — pad-to-16 sibling of `moe_sort_by_expert_padded`. Currently
/// unreachable — the tile16 MMQ kernel it was designed to feed was NULL
/// on 35B-UD-Q4_K_S (1.b moved to `_unverified/`). Kept in case a
/// future tile16-class attempt with different kernel internals wants the
/// pad-to-16 invariant; the scan_padded_offsets_16 kernel is already
/// compiled into `moe_sort_by_expert`.
#[allow(dead_code)]
pub fn moe_sort_by_expert_padded_16(
    reg: &OpsRegistry,
    stream: &HipStream,
    expert_ids: DevicePtr,
    counts: DevicePtr,
    offsets: DevicePtr,
    cursors: DevicePtr,
    sorted_pair_idx: DevicePtr,
    padded_offsets: DevicePtr,
    sorted_pair_idx_padded: DevicePtr,
    total: usize,
    n_experts: usize,
    max_tokens: usize,
    top_k: usize,
) -> Result<()> {
    moe_sort_by_expert(
        reg,
        stream,
        expert_ids,
        counts,
        offsets,
        cursors,
        sorted_pair_idx,
        total,
        n_experts,
    )?;

    let module = reg.expect_module("moe_sort_by_expert")?;
    let k_scan_padded = module.kernel("flambeau_moe_sort_scan_padded_offsets_16")?;
    let k_pad_copy = module.kernel("flambeau_moe_sort_pad_copy")?;

    let n_experts_i = n_experts as i32;
    let c_ptr: u64 = counts.as_usize() as u64;
    let o_ptr: u64 = offsets.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let spi_ptr: u64 = sorted_pair_idx.as_usize() as u64;
    let spip_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;

    {
        let mut args = KernelArgs::new();
        args.push(&c_ptr);
        args.push(&po_ptr);
        args.push(&n_experts_i);
        let cfg = LaunchCfg { grid: (1, 1, 1), block: (512, 1, 1), shared_bytes: 0 };
        unsafe { k_scan_padded.launch(stream, cfg, args)? };
    }
    {
        let mut args = KernelArgs::new();
        args.push(&spi_ptr);
        args.push(&o_ptr);
        args.push(&c_ptr);
        args.push(&po_ptr);
        args.push(&spip_ptr);
        args.push(&n_experts_i);
        // pad-to-16 worst case: each of `total` pairs + up to 15 padding slots per expert.
        let max_per_expert = (max_tokens * top_k + 15) & !15;
        let grid_x = (max_per_expert as u32).div_ceil(256);
        let cfg = LaunchCfg {
            grid: (grid_x.max(1), n_experts as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_pad_copy.launch(stream, cfg, args)? };
    }
    Ok(())
}

pub fn moe_sort_by_expert_padded(
    reg: &OpsRegistry,
    stream: &HipStream,
    expert_ids: DevicePtr,                 // [total] i32
    counts: DevicePtr,                     // [n_experts] i32, overwritten
    offsets: DevicePtr,                    // [n_experts + 1] i32 (written)
    cursors: DevicePtr,                    // [n_experts] i32 scratch
    sorted_pair_idx: DevicePtr,            // [total] i32 (written, unpadded)
    padded_offsets: DevicePtr,             // [n_experts + 1] i32 (written)
    sorted_pair_idx_padded: DevicePtr,     // [total_padded_cap] i32 (written)
    total: usize,
    n_experts: usize,
    max_tokens: usize,
    top_k: usize,
) -> Result<()> {
    // 1. Run the standard (unpadded) sort — fills counts, offsets, cursors,
    // sorted_pair_idx.
    moe_sort_by_expert(
        reg,
        stream,
        expert_ids,
        counts,
        offsets,
        cursors,
        sorted_pair_idx,
        total,
        n_experts,
    )?;

    let module = reg.expect_module("moe_sort_by_expert")?;
    let k_scan_padded = module.kernel("flambeau_moe_sort_scan_padded_offsets")?;
    let k_pad_copy = module.kernel("flambeau_moe_sort_pad_copy")?;

    let n_experts_i = n_experts as i32;
    let c_ptr: u64 = counts.as_usize() as u64;
    let o_ptr: u64 = offsets.as_usize() as u64;
    let po_ptr: u64 = padded_offsets.as_usize() as u64;
    let spi_ptr: u64 = sorted_pair_idx.as_usize() as u64;
    let spip_ptr: u64 = sorted_pair_idx_padded.as_usize() as u64;

    // 2. Scan counts → padded_offsets (single-block Blelloch).
    {
        let mut args = KernelArgs::new();
        args.push(&c_ptr);
        args.push(&po_ptr);
        args.push(&n_experts_i);
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (512, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_scan_padded.launch(stream, cfg, args)? };
    }
    // 3. Copy + pad-fill. grid.x covers the worst case (all pairs to one
    // expert, rounded up to mult of 8); blocks that fall outside an
    // expert's padded range early-exit.
    {
        let mut args = KernelArgs::new();
        args.push(&spi_ptr);
        args.push(&o_ptr);
        args.push(&c_ptr);
        args.push(&po_ptr);
        args.push(&spip_ptr);
        args.push(&n_experts_i);
        let max_per_expert = (max_tokens * top_k + 7) & !7;
        let grid_x = (max_per_expert as u32).div_ceil(256);
        let cfg = LaunchCfg {
            grid: (grid_x.max(1), n_experts as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_pad_copy.launch(stream, cfg, args)? };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucketing_groups_by_expert_and_pads() {
        // 3 tokens × top_k=2. Experts chosen to produce one bucket with a
        // tail-pad and one full bucket.
        let ids = vec![0i32, 1, 0, 1, 0, 2];
        let (be, bs) = build_expert_buckets(&ids, 3, 2);
        // Expert 0 has 3 refs → one bucket with 5× -1 tail pad.
        // Expert 1 has 2 refs → one bucket with 6× -1 tail pad.
        // Expert 2 has 1 ref → one bucket with 7× -1 tail pad.
        assert_eq!(be, vec![0, 1, 2]);
        assert_eq!(bs.len(), 3 * INDEXED_MOE_MMQ_X);
        // Expert 0 refs: (t=0,slot=0), (t=1,slot=0), (t=2,slot=0).
        assert_eq!(bs[0], 0);
        assert_eq!(bs[1], 1 << 16);
        assert_eq!(bs[2], 2 << 16);
        assert_eq!(bs[3], -1);
        // Expert 1 refs: (t=0,slot=1), (t=1,slot=1).
        let b1 = INDEXED_MOE_MMQ_X;
        assert_eq!(bs[b1], 1);
        assert_eq!(bs[b1 + 1], (1 << 16) | 1);
        assert_eq!(bs[b1 + 2], -1);
    }

    #[test]
    fn bucketing_splits_chunks_over_mmq_x() {
        // One expert hit 10 times → two buckets: full 8 + tail 2.
        let ids: Vec<i32> = (0..10).map(|_| 7i32).collect();
        let (be, bs) = build_expert_buckets(&ids, 5, 2);
        assert_eq!(be, vec![7, 7]);
        assert_eq!(bs.len(), 2 * INDEXED_MOE_MMQ_X);
        // Second bucket has 2 real refs (entries 8 and 9 = (t=4,slot=0)
        // and (t=4,slot=1)) then 6 sentinels.
        let b2 = INDEXED_MOE_MMQ_X;
        assert_eq!(bs[b2], 4 << 16);
        assert_eq!(bs[b2 + 1], (4 << 16) | 1);
        for i in 2..INDEXED_MOE_MMQ_X {
            assert_eq!(bs[b2 + i], -1);
        }
    }
}
