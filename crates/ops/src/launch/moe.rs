//! MoE building blocks — TopK router, IndexedMoE matmul (MMVQ, fused
//! gate+up, tile8 MMQ prefill), weighted combine.

use anyhow::Result;
use flambeau_core::{LaunchArgs, LaunchCfg};
use flambeau_core::{DevicePtr, MOE_SORT_MAX_EXPERTS, TOPK_MAX_EXPERTS};

pub use crate::sig::MoeShape;

/// TopK router over per-token logits. Emits `(token, slot)` → expert index
/// plus normalised softmax weights over the k selected experts per token.
/// Shapes: `logits[n_tokens, n_experts]` F32 in; `idx[n_tokens, k]` i32 out;
/// `weights[n_tokens, k]` F32 out.
pub fn topk_f32<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    bufs: crate::TopkBuffers,
    shape: crate::TopkShape,
) -> Result<()> {
    let crate::TopkBuffers { logits, idx, weights } = bufs;
    let crate::TopkShape { n_tokens, n_experts, k } = shape;
    // Kernel's compile-time ceiling — must match `#define TOPK_MAX_EXPERTS`
    // in `kernels-hip/src/kernels/topk_softmax.cu`. Pre-this was
    // hardcoded at 128 on both sides, silently dropping experts 128..255 on
    // Qwen3.6 and triggering OOB LDS writes at the cross-warp reduce.
    assert!(
        n_experts <= TOPK_MAX_EXPERTS,
        "topk_f32: n_experts {n_experts} > {TOPK_MAX_EXPERTS} (bump TOPK_MAX_EXPERTS in core::kernel_limits + .cu #define + add cert shape)"
    );
    let module = reg.expect_module("topk_f32")?;
    let kernel = B::kernel(module, "flambeau_topk_softmax_f32")?;

    let n_tokens_i = n_tokens as i32;
    let n_experts_i = n_experts as i32;
    let k_i = k as i32;
    let l_ptr: u64 = logits.as_usize() as u64;
    let i_ptr: u64 = idx.as_usize() as u64;
    let w_ptr: u64 = weights.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&l_ptr);
    args.push_arg(&i_ptr);
    args.push_arg(&w_ptr);
    args.push_arg(&n_tokens_i);
    args.push_arg(&n_experts_i);
    args.push_arg(&k_i);
    // blockDim must be a multiple of wave64 and ≥64 so `n_warps = blockDim>>6`
    // is non-zero. Threads past n_experts load -INF via the kernel's guard.
    let block = (n_experts.max(64).div_ceil(64) * 64) as u32;
    let cfg = LaunchCfg::one_d(n_tokens as u32, block);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Apply per-expert scalar to routing weights:
/// `expert_weights[k] *= expert_scales[expert_ids[k]]` for k in 0..top_k.
///
/// Folds gemma4's `ffn_down_exps.scale` (F32 [n_experts]) into the
/// routing weights so the existing `moe_combine_*` kernels apply the
/// per-expert post-down scaling for free. Mathematically equivalent
/// to candle's `quantized_gemma4.rs:2521` (`moe_out.broadcast_mul(
/// expert_scales)` before the weighted-sum).
pub fn apply_per_expert_scale_f32<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    expert_weights: DevicePtr,
    expert_ids: DevicePtr,
    expert_scales: DevicePtr,
    n_tokens: usize,
    top_k: usize,
) -> Result<()> {
    assert!(
        top_k > 0 && top_k <= 64,
        "apply_per_expert_scale_f32: top_k {top_k} not in 1..=64"
    );
    assert!(
        n_tokens > 0,
        "apply_per_expert_scale_f32: n_tokens must be > 0"
    );
    let module = reg.expect_module("apply_per_expert_scale_f32")?;
    let kernel = B::kernel(module, "flambeau_apply_per_expert_scale_f32")?;
    let w_ptr: u64 = expert_weights.as_usize() as u64;
    let i_ptr: u64 = expert_ids.as_usize() as u64;
    let s_ptr: u64 = expert_scales.as_usize() as u64;
    let k_i = top_k as i32;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&i_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&k_i);
    // gridDim.x = n_tokens, blockDim.x = max(64, top_k) (kernel guards
    // `k < top_k` so unused threads idle). top_k=8 on gemma4 26B-A4B.
    let cfg = LaunchCfg::one_d(n_tokens as u32, 64);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Token count from which the Q4_K MoE MMVQ switches from r2 to r4: the
/// boundary of the `indexed_moe_mmvq_q4_k_r2` / `_r4` rows of
/// `dispatch/hip/gfx906.toml` (checked by `tests::q4k_moe_mmvq_split_matches_dispatch`).
pub const Q4K_MOE_MMVQ_R4_MIN_TOKENS: usize = 32;

/// Q4_K indexed MoE MMVQ: r2 (2 output rows per wave64) below
/// [`Q4K_MOE_MMVQ_R4_MIN_TOKENS`] tokens, r4 (4 rows) from there on.
/// Shapes:
/// - `w[n_experts, n_rows, n_sb_per_row]` Q4_K blocks
/// - `y[n_tokens, n_sb_per_row * 8]` Q8_1 blocks
/// - `expert_ids[n_tokens, top_k]` i32
/// - `dst[n_tokens, top_k, n_rows]` F32
pub fn indexed_moe_mmvq_q4_k_r2<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let (stem, entry, rows_per_block) = if n_tokens >= Q4K_MOE_MMVQ_R4_MIN_TOKENS {
        (
            "indexed_moe_mmvq_q4_k_r4_dp4a",
            "flambeau_indexed_moe_mmvq_q4_k_r4_dp4a_q8_1",
            4u32,
        )
    } else {
        (
            "indexed_moe_mmvq_q4_k_r2_dp4a",
            "flambeau_indexed_moe_mmvq_q4_k_r2_dp4a_q8_1",
            2u32,
        )
    };
    let module = reg.expect_module(stem)?;
    let kernel = B::kernel(module, entry)?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(rows_per_block);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q3_K MoE MMVQ. Routes through `indexed_moe_mmvq_q3_k_r2_dp4a`:
/// wave64, 2 rows per block, half-warp DPP reduce.
pub fn indexed_moe_mmvq_q3_k<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_q3_k_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_q3_k_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ4_XS MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ4_XS blocks.
pub fn indexed_moe_mmvq_iq4_xs<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq4_xs_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq4_xs_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ4_NL MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ4_NL blocks.
pub fn indexed_moe_mmvq_iq4_nl<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row: n_blocks_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq4_nl_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq4_nl_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ3_XXS MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ3_XXS blocks.
pub fn indexed_moe_mmvq_iq3_xxs<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq3_xxs_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq3_xxs_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ3_S MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ3_S blocks.
pub fn indexed_moe_mmvq_iq3_s<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq3_s_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq3_s_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_XXS MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ2_XXS blocks.
pub fn indexed_moe_mmvq_iq2_xxs<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq2_xxs_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq2_xxs_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_XS MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ2_XS blocks.
pub fn indexed_moe_mmvq_iq2_xs<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq2_xs_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq2_xs_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_S MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ2_S blocks.
pub fn indexed_moe_mmvq_iq2_s<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq2_s_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq2_s_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ1_S MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ1_S blocks.
pub fn indexed_moe_mmvq_iq1_s<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq1_s_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq1_s_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ1_M MoE MMVQ. Same indexing contract as `indexed_moe_mmvq_q4_k`;
/// weights are IQ1_M blocks.
pub fn indexed_moe_mmvq_iq1_m<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_iq1_m_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_iq1_m_r2_dp4a_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q6_K MoE MMVQ. Routes through `indexed_moe_mmvq_q6_k_r2_dp4a`:
/// wave64, 2 rows per block, half-warp DPP reduce. Same indexing
/// contract as the Q4_K sibling: `[n_tokens, top_k]` expert ids,
/// `[n_tokens, top_k, n_rows]` F32 output, `[n_tokens, n_sb_per_row
/// * 8]` Q8_1 activations; weights are Q6_K super-blocks. Needed
/// for UD-Q4_K_S-style mixed-quant GGUFs where some
/// `ffn_down_exps` are promoted from Q4_K to Q6_K.
pub fn indexed_moe_mmvq_q6_k<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_q6_k_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_q6_k_r2_dp4a_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q5_K MoE MMVQ. Routes through `indexed_moe_mmvq_q5_k_r2_dp4a`:
/// wave64, 2 rows per block, half-warp DPP reduce.
pub fn indexed_moe_mmvq_q5_k<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_q5_k_r2_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_q5_k_r2_dp4a_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let grid_x = (n_rows as u32).div_ceil(2);
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_0 indexed-MoE MMVQ. Unblocks Qwen3.6-35B-A3B-Q4_0 whose
/// MoE expert weights are Q4_0 (gate+up+down in most layers). Same
/// contract as `indexed_moe_mmvq_q8_0`, 256 threads/block with VDR=2 DP4A.
pub fn indexed_moe_mmvq_q4_0<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row: n_blocks_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_q4_0")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_q4_0_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_1 indexed-MoE MMVQ. Unblocks Qwen-published
/// Qwen3.6-35B-A3B-Q4_0 whose `ffn_down_exps` are Q4_1 (gate/up are Q4_0,
/// down is Q4_1). Same contract as `indexed_moe_mmvq_q4_0`; per-block
/// reconstruction differs (`m_x · s_y` instead of `-8 · d_x · s_y`).
pub fn indexed_moe_mmvq_q4_1<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row: n_blocks_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_q4_1")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_q4_1_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Fused gate+up Q4_0 indexed-MoE MMVQ. Reads each Q8_1
/// activation word once per block and produces both gate and up outputs,
/// halving the launch count for MoE Q4_0 decode vs calling
/// `indexed_moe_mmvq_q4_0` twice.
pub fn indexed_moe_mmvq_q4_0_gate_up<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqGateUpBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqGateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, gate_out, up_out } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row: n_blocks_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_q4_0_gate_up_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q8_0 indexed-MoE MMVQ. Unblocks UD-Q8_K_XL GGUFs whose MoE
/// expert weights stay Q8_0 instead of the usual Q4_K/Q4_K_S. Uses VDR=2
/// DP4A inside the inner loop (matches `mmvq_q8_0_dp4a_vdr2` pattern),
/// 256 threads/block, 1 output row per block. A multi-row r2/r4 variant
/// is the next perf lever; single-row is adequate for 2's unblock goal.
/// Contract:
/// * weights [n_experts, n_rows, n_blocks_per_row] Q8_0 blocks
/// * activations [n_tokens, n_blocks_per_row] Q8_1 blocks
/// * expert_ids [n_tokens, top_k] i32
/// * dst [n_tokens, top_k, n_rows] F32
pub fn indexed_moe_mmvq_q8_0<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row: n_blocks_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_q8_0")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_q8_0_dp4a_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_blocks_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Fused gate+up Q8_0 indexed-MoE MMVQ. Sibling of `indexed_moe_mmvq_q8_0`
/// (single weight) and `indexed_moe_mmvq_q4_0_gate_up` (Q4_0). Reads each
/// Q8_1 activation int32 once per block and produces both gate and up
/// outputs, halving the MoE decode launch count for Q8_0 expert weights.
pub fn indexed_moe_mmvq_q8_0_gate_up<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqGateUpBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqGateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, gate_out, up_out } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row: n_blocks_per_row } = shape;
    let module = reg.expect_module("indexed_moe_mmvq_q8_0_gate_up_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmvq_q8_0_gate_up_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ4_XS MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ4_XS).
///
pub fn indexed_moe_mmq_iq4_xs_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq4_xs_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq4_xs_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ4_NL MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ4_NL).
///
pub fn indexed_moe_mmq_iq4_nl_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq4_nl_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq4_nl_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ3_XXS MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ3_XXS).
///
pub fn indexed_moe_mmq_iq3_xxs_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq3_xxs_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq3_xxs_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ3_S MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ3_S).
///
pub fn indexed_moe_mmq_iq3_s_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq3_s_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq3_s_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_XXS MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ2_XXS).
///
pub fn indexed_moe_mmq_iq2_xxs_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq2_xxs_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq2_xxs_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_XS MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ2_XS).
///
pub fn indexed_moe_mmq_iq2_xs_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq2_xs_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq2_xs_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_S MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ2_S).
///
pub fn indexed_moe_mmq_iq2_s_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq2_s_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq2_s_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ1_S MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ1_S).
///
pub fn indexed_moe_mmq_iq1_s_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq1_s_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq1_s_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ1_M MoE MMQ tile8 gate+up — fused gate+up MoE prefill kernel. Same
/// padded-sort + tile8 contract as `indexed_moe_mmq_q4_k_gate_up_tile8`;
/// kernel buildable and registered, not yet wired into TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ1_M).
///
pub fn indexed_moe_mmq_iq1_m_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq1_m_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq1_m_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// fused gate+up tile8 MoE MMQ. Block = 64 rows × 8 slots = 512
/// outputs; all 8 slots guaranteed same expert via padded sort.
/// Weight tile decoded ONCE per thread per sub-block, reused across 8 cols.
/// Grid.y is an upper bound on padded_total / 8; kernel early-exits
/// blocks past the actual (on-device) padded_offsets[n_experts] → avoids
/// DtoH sync.
pub fn indexed_moe_mmq_q4_k_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q4_k_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q4_k_gate_up_tile8_dp4a_q8_1")?;

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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    // Upper-bound grid.y; in-kernel early-exit handles actual padded_total.
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q3_K MoE MMQ tile8 gate+up. Same contract + launch shape as the
/// Q4_K sibling; caller sets `n_sb_per_row = hidden / QK_K`.
pub fn indexed_moe_mmq_q3_k_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q3_k_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q3_k_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ4_XS MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ4_XS).
///
pub fn indexed_moe_mmq_iq4_xs_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq4_xs_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq4_xs_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ4_NL MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ4_NL).
///
pub fn indexed_moe_mmq_iq4_nl_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq4_nl_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq4_nl_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ3_XXS MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ3_XXS).
///
pub fn indexed_moe_mmq_iq3_xxs_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq3_xxs_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq3_xxs_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ3_S MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ3_S).
///
pub fn indexed_moe_mmq_iq3_s_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq3_s_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq3_s_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_XXS MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ2_XXS).
///
pub fn indexed_moe_mmq_iq2_xxs_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq2_xxs_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq2_xxs_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_XS MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ2_XS).
///
pub fn indexed_moe_mmq_iq2_xs_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq2_xs_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq2_xs_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ2_S MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ2_S).
///
pub fn indexed_moe_mmq_iq2_s_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq2_s_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq2_s_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ1_S MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ1_S).
///
pub fn indexed_moe_mmq_iq1_s_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq1_s_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq1_s_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// IQ1_M MoE MMQ tile8 down — single-output companion to the gate+up
/// kernel. Kernel buildable + registered; dormant in TP MoE prefill
/// dispatch (`moe_tp.rs::tile8_dt_ok` does not include IQ1_M).
///
pub fn indexed_moe_mmq_iq1_m_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_iq1_m_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_iq1_m_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// down-projection tile8 MoE MMQ. Same per-block layout as
/// the gate_up tile8; activation is indexed by pair_idx directly.
pub fn indexed_moe_mmq_q4_k_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q4_k_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q4_k_down_tile8_dp4a_q8_1")?;

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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q3_K MoE MMQ tile8 down — single-output sibling of the Q3_K
/// gate+up; same contract as the Q4_K down wrapper.
pub fn indexed_moe_mmq_q3_k_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q3_k_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q3_k_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_0 gate+up tile8 MoE MMQ sibling. Same contract as the Q4_K
/// `indexed_moe_mmq_q4_k_gate_up_tile8` wrapper; weight dtype is Q4_0 so
/// `n_sb_per_row` in the `MoeShape` should be set to `hidden / 32` (Q4_0
/// block size) by the caller, not `hidden / QK_K` as for Q4_K.
pub fn indexed_moe_mmq_q4_0_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q4_0_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_0 down tile8 MoE MMQ sibling.
pub fn indexed_moe_mmq_q4_0_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q4_0_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_1 down tile8 MoE MMQ sibling of the Q4_0
/// down tile8. Same launch shape; the kernel's per-block dot uses the
/// Q4_1 affine `d_x·d_y·sumi + m_x·s_y` instead of Q4_0's bias-corrected
/// `d_x·(d_y·sumi - 8·s_y)`. Used by Coder-Next-Q4_0 (down=Q4_1).
pub fn indexed_moe_mmq_q4_1_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q4_1_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q4_1_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q8_0 gate+up tile8 MoE MMQ sibling of the Q4_0 and Q4_K tile8
/// wrappers. Weight dtype is Q8_0 so `n_sb_per_row` in `MoeShape` is
/// `hidden / 32` (Q8_0 block size), matching the Q4_0 convention. Same
/// grid/block shape as Q4_0 tile8 — one wave64 per 64×8 output tile.
pub fn indexed_moe_mmq_q8_0_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q8_0_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q8_0_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q8_0 down tile8 MoE MMQ sibling.
pub fn indexed_moe_mmq_q8_0_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q8_0_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q8_0_down_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q6_K sibling of `indexed_moe_mmq_q4_k_down_tile8`. Same
/// tile layout (64 rows × 8 slot-cols, 1 wave64, padded-sort
/// per-block-expert invariant) with Q6_K decode (raw·y - 32·Σy bias
/// correction to avoid the byte-borrow bug). Used for UD-Q4_K_S
/// `ffn_down_exps` layers that are Q6_K-quantised.
/// Q5_K down-projection MoE MMQ (tile8).
/// Same contract as `indexed_moe_mmq_q6_k_down_tile8` / `indexed_moe_mmq_q4_k_down_tile8`.
/// Closes the MMVQ-at-prefill hole for Qwen3-Coder-30B-A3B-UD-Q4_K_XL
/// (13/48 layers promote `ffn_down_exps` to Q5_K). Uses the Q6_K
/// decode fix.
pub fn indexed_moe_mmq_q5_k_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q5_k_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q5_k_down_tile8_dp4a_q8_1")?;

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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q5_K gate+up tile8 MoE MMQ. Q5_K down tile8 decode (per-element ql+qh
/// 5-bit, per-sub-block (sc, m) → d·sumi_d·sc − dmin·sumi_y·m) fused for
/// two output projections.
pub fn indexed_moe_mmq_q5_k_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q5_k_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q5_k_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q6_K gate+up tile8 MoE MMQ. Q6_K down tile8 decode (ql+qh 6-bit lookup,
/// per-16-element scale, `(raw − 32)·y = raw·y − 32·Σy` bias-correction)
/// fused for two output projections.
pub fn indexed_moe_mmq_q6_k_gate_up_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8GateUpBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8GateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, gate_out, up_out } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q6_k_gate_up_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q6_k_gate_up_tile8_dp4a_q8_1")?;
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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

pub fn indexed_moe_mmq_q6_k_down_tile8<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmqTile8DownBuffers,
    shape: MoeShape,
) -> Result<()> {
    let crate::MoeMmqTile8DownBuffers { weights: w, act: y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst } = buffers;
    let module = reg.expect_module("indexed_moe_mmq_q6_k_down_tile8_dp4a")?;
    let kernel = B::kernel(module, "flambeau_indexed_moe_mmq_q6_k_down_tile8_dp4a_q8_1")?;

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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&s_ptr);
    args.push_arg(&po_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    args.push_arg(&n_experts_i);
    let grid_y = shape.padded_total_upper_bound.div_ceil(8) as u32;
    let cfg = LaunchCfg {
        grid: ((shape.n_rows as u32).div_ceil(64), grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

pub fn indexed_moe_mmvq_q4_k_gate_up<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MoeMmvqGateUpBuffers,
    shape: crate::MoeMmvqShape,
) -> Result<()> {
    let crate::MoeMmvqGateUpBuffers { gate_w: w_gate, up_w: w_up, act: y, expert_ids, gate_out, up_out } = buffers;
    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;
    let stem = "indexed_moe_mmvq_q4_k_gate_up_r4_dp4a";
    let entry = "flambeau_indexed_moe_mmvq_q4_k_gate_up_r4_dp4a_q8_1";
    let module = reg.expect_module(stem)?;
    let kernel = B::kernel(module, entry)?;

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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&e_ptr);
    args.push_arg(&go_ptr);
    args.push_arg(&uo_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&nb_i);
    // r4: 4 rows per block (quarter-wave-per-row).
    let cfg = LaunchCfg {
        grid: ((n_rows as u32).div_ceil(4), (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Shared-expert gate scaling (Qwen3.5/3.6 hybrid). Computes per-token
/// `gate[t] = sigmoid(Σ_i gate_w[i] · x[t, i])` and multiplies each row of
/// `shared_out` by its token's `gate` in-place.
/// Caller typically computes `shared_out` first via gate/up/swiglu/down
/// dense FFN using the existing `qmatmul` + `swiglu_f16` ops on the
/// `ffn_*_shexp` weights, then calls this to apply the learned scalar gate.
pub fn shared_expert_scale_f32<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    shared_out: DevicePtr, // in-place [n_tokens, hidden]
    x: DevicePtr,          // [n_tokens, hidden] (layer input)
    gate_w: DevicePtr,     // [hidden]
    n_tokens: usize,
    hidden: usize,
) -> Result<()> {
    let module = reg.expect_module("shared_expert_scale_f32")?;
    let kernel = B::kernel(module, "flambeau_shared_expert_scale_f32")?;
    let n_tokens_i = n_tokens as i32;
    let hidden_i = hidden as i32;
    let so_ptr: u64 = shared_out.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let w_ptr: u64 = gate_w.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&so_ptr);
    args.push_arg(&x_ptr);
    args.push_arg(&w_ptr);
    args.push_arg(&n_tokens_i);
    args.push_arg(&hidden_i);
    let cfg = LaunchCfg::one_d(n_tokens as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Weighted combine + residual — `out = residual + Σ_k w_k · expert_out[k]`.
/// One thread per `(token, hidden)` slot. All tensors F16 except
/// `weights[n_tokens, top_k]` F32.
pub fn moe_combine_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MoeCombineBuffers,
    shape: crate::MoeCombineShape,
) -> Result<()> {
    let module = reg.expect_module("moe_combine_f16")?;
    let kernel = B::kernel(module, "flambeau_moe_combine_f16")?;

    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let hidden_i = shape.hidden as i32;
    let e_ptr: u64 = buf.expert_outs.as_usize() as u64;
    let w_ptr: u64 = buf.weights.as_usize() as u64;
    let r_ptr: u64 = buf.residual.as_usize() as u64;
    let o_ptr: u64 = buf.out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&e_ptr);
    args.push_arg(&w_ptr);
    args.push_arg(&r_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&hidden_i);
    let total = shape.n_tokens * shape.hidden;
    let cfg = LaunchCfg::one_d(total.div_ceil(256) as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// `moe_combine_f16` variant without an input residual.
/// `out[token, d] = Σ_k weight[token, k] * expert_outs[token, k, d]`.
/// Used by the TP-sharded MoE forward where the residual stream is
/// folded later by the AllReduce-residual kernel.
pub fn moe_combine_no_residual_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MoeCombineNoResidualBuffers,
    shape: crate::MoeCombineShape,
) -> Result<()> {
    let module = reg.expect_module("moe_combine_no_residual_f16")?;
    let kernel = B::kernel(module, "flambeau_moe_combine_no_residual_f16")?;

    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let hidden_i = shape.hidden as i32;
    let e_ptr: u64 = buf.expert_outs.as_usize() as u64;
    let w_ptr: u64 = buf.weights.as_usize() as u64;
    let o_ptr: u64 = buf.out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&e_ptr);
    args.push_arg(&w_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&hidden_i);
    let total = shape.n_tokens * shape.hidden;
    let cfg = LaunchCfg::one_d(total.div_ceil(256) as u32, 256);
    // SAFETY: args reference live device pointers + CPU values; kernel
    // writes hidden-element F16 output. Caller's contract.
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// F32-throughout sibling of `moe_combine_no_residual_f16`. Reads F32
/// expert outputs and writes F32 partial. Required by the gemma4
/// head_dim=512 + Q8_0 path where V-norm spikes propagate into down
/// outputs and F16 cast saturates.
pub fn moe_combine_no_residual_f32<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MoeCombineNoResidualBuffers,
    shape: crate::MoeCombineShape,
) -> Result<()> {
    let module = reg.expect_module("moe_combine_no_residual_f32")?;
    let kernel = B::kernel(module, "flambeau_moe_combine_no_residual_f32")?;

    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let hidden_i = shape.hidden as i32;
    let e_ptr: u64 = buf.expert_outs.as_usize() as u64;
    let w_ptr: u64 = buf.weights.as_usize() as u64;
    let o_ptr: u64 = buf.out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&e_ptr);
    args.push_arg(&w_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&hidden_i);
    let total = shape.n_tokens * shape.hidden;
    let cfg = LaunchCfg::one_d(total.div_ceil(256) as u32, 256);
    // SAFETY: args reference live device pointers + CPU values; kernel
    // writes hidden-element F32 output. Caller's contract.
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// `moe_combine_f16` variant that accepts two F16 residuals and
/// sums them inline. Saves one `add_f16` launch per layer per token on the
/// shared-expert path (`moe_residual = mid + shared_delta`).
pub fn moe_combine_two_residuals_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MoeCombineTwoResidualsBuffers,
    shape: crate::MoeCombineShape,
) -> Result<()> {
    let module = reg.expect_module("moe_combine_two_residuals_f16")?;
    let kernel = B::kernel(module, "flambeau_moe_combine_two_residuals_f16")?;

    let n_tokens_i = shape.n_tokens as i32;
    let top_k_i = shape.top_k as i32;
    let hidden_i = shape.hidden as i32;
    let e_ptr: u64 = buf.expert_outs.as_usize() as u64;
    let w_ptr: u64 = buf.weights.as_usize() as u64;
    let r1_ptr: u64 = buf.residual1.as_usize() as u64;
    let r2_ptr: u64 = buf.residual2.as_usize() as u64;
    let o_ptr: u64 = buf.out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&e_ptr);
    args.push_arg(&w_ptr);
    args.push_arg(&r1_ptr);
    args.push_arg(&r2_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_tokens_i);
    args.push_arg(&top_k_i);
    args.push_arg(&hidden_i);
    let total = shape.n_tokens * shape.hidden;
    let cfg = LaunchCfg::one_d(total.div_ceil(256) as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
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
pub fn moe_sort_by_expert<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MoeSortBuffers,
    shape: crate::MoeSortShape,
) -> Result<()> {
    assert!(
        shape.n_experts <= MOE_SORT_MAX_EXPERTS,
        "moe_sort_by_expert: n_experts {} > {MOE_SORT_MAX_EXPERTS} (bump MOE_SORT_MAX_EXPERTS in core::kernel_limits + matching `#define` in kernels-hip/src/kernels/moe_sort.cu)",
        shape.n_experts
    );
    let module = reg.expect_module("moe_sort_by_expert")?;
    let k_zero = B::kernel(module, "flambeau_moe_sort_zero_counts")?;
    let k_count = B::kernel(module, "flambeau_moe_sort_count")?;
    let k_scan = B::kernel(module, "flambeau_moe_sort_scan_offsets")?;
    // deterministic single-thread scatter so sorted_pair_idx
    // is bit-reproducible across runs and ranks. The racing-atomic
    // variant was opt-in via FLAMBEAU_MOE_SCATTER=race (deleted in S3 —
    // it corrupted TP determinism and the perf gain was marginal).
    let k_scatter = B::kernel(module, "flambeau_moe_sort_scatter_det")?;

    let total_i = shape.total as i32;
    let n_experts_i = shape.n_experts as i32;
    let e_ptr: u64 = buf.expert_ids.as_usize() as u64;
    let c_ptr: u64 = buf.counts.as_usize() as u64;
    let o_ptr: u64 = buf.offsets.as_usize() as u64;
    let k_ptr: u64 = buf.cursors.as_usize() as u64;
    let s_ptr: u64 = buf.sorted_pair_idx.as_usize() as u64;

    // Kernel 0: zero counts
    {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&c_ptr);
        args.push_arg(&n_experts_i);
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (512, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_zero, stream, cfg, args)?;
    }
    // Kernel 1: histogram
    {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&e_ptr);
        args.push_arg(&c_ptr);
        args.push_arg(&total_i);
        const BLOCK: u32 = 256;
        let grid = (shape.total as u32).div_ceil(BLOCK);
        let cfg = LaunchCfg::one_d(grid, BLOCK);
        B::launch(&k_count, stream, cfg, args)?;
    }
    // Kernel 2: scan to offsets + init cursors (single block, 512 threads)
    {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&c_ptr);
        args.push_arg(&o_ptr);
        args.push_arg(&k_ptr);
        args.push_arg(&n_experts_i);
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (512, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_scan, stream, cfg, args)?;
    }
    // Kernel 3: scatter (deterministic by default, racing on opt-in)
    {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&e_ptr);
        args.push_arg(&k_ptr);
        args.push_arg(&s_ptr);
        args.push_arg(&total_i);
        // Deterministic single-thread scatter — see kernel comment.
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_scatter, stream, cfg, args)?;
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
/// Pad-to-16 sibling of `moe_sort_by_expert_padded`. Currently
/// unreachable — the tile16 MMQ kernel it was designed to feed was NULL
/// on 35B-UD-Q4_K_S (moved to `_unverified/`). Kept in case a
/// future tile16-class attempt with different kernel internals wants the
/// pad-to-16 invariant; the scan_padded_offsets_16 kernel is already
/// compiled into `moe_sort_by_expert`.
pub fn moe_sort_by_expert_padded_16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MoeSortPaddedBuffers,
    shape: crate::MoeSortPaddedShape,
) -> Result<()> {
    moe_sort_by_expert(
        reg,
        stream,
        crate::MoeSortBuffers {
            expert_ids: buf.expert_ids,
            counts: buf.counts,
            offsets: buf.offsets,
            cursors: buf.cursors,
            sorted_pair_idx: buf.sorted_pair_idx,
        },
        crate::MoeSortShape {
            total: shape.total,
            n_experts: shape.n_experts,
        },
    )?;

    let module = reg.expect_module("moe_sort_by_expert")?;
    let k_scan_padded = B::kernel(module, "flambeau_moe_sort_scan_padded_offsets_16")?;
    let k_pad_copy = B::kernel(module, "flambeau_moe_sort_pad_copy")?;

    let n_experts_i = shape.n_experts as i32;
    let c_ptr: u64 = buf.counts.as_usize() as u64;
    let o_ptr: u64 = buf.offsets.as_usize() as u64;
    let po_ptr: u64 = buf.padded_offsets.as_usize() as u64;
    let spi_ptr: u64 = buf.sorted_pair_idx.as_usize() as u64;
    let spip_ptr: u64 = buf.sorted_pair_idx_padded.as_usize() as u64;

    {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&c_ptr);
        args.push_arg(&po_ptr);
        args.push_arg(&n_experts_i);
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (512, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_scan_padded, stream, cfg, args)?;
    }
    {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&spi_ptr);
        args.push_arg(&o_ptr);
        args.push_arg(&c_ptr);
        args.push_arg(&po_ptr);
        args.push_arg(&spip_ptr);
        args.push_arg(&n_experts_i);
        // pad-to-16 worst case: each of `total` pairs + up to 15 padding slots per expert.
        let max_per_expert = (shape.max_tokens * shape.top_k + 15) & !15;
        let grid_x = (max_per_expert as u32).div_ceil(256);
        let cfg = LaunchCfg {
            grid: (grid_x.max(1), shape.n_experts as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_pad_copy, stream, cfg, args)?;
    }
    Ok(())
}

pub fn moe_sort_by_expert_padded<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MoeSortPaddedBuffers,
    shape: crate::MoeSortPaddedShape,
) -> Result<()> {
    // 1. Run the standard (unpadded) sort — fills counts, offsets, cursors,
    // sorted_pair_idx.
    moe_sort_by_expert(
        reg,
        stream,
        crate::MoeSortBuffers {
            expert_ids: buf.expert_ids,
            counts: buf.counts,
            offsets: buf.offsets,
            cursors: buf.cursors,
            sorted_pair_idx: buf.sorted_pair_idx,
        },
        crate::MoeSortShape {
            total: shape.total,
            n_experts: shape.n_experts,
        },
    )?;

    let module = reg.expect_module("moe_sort_by_expert")?;
    let k_scan_padded = B::kernel(module, "flambeau_moe_sort_scan_padded_offsets")?;
    let k_pad_copy = B::kernel(module, "flambeau_moe_sort_pad_copy")?;

    let n_experts_i = shape.n_experts as i32;
    let c_ptr: u64 = buf.counts.as_usize() as u64;
    let o_ptr: u64 = buf.offsets.as_usize() as u64;
    let po_ptr: u64 = buf.padded_offsets.as_usize() as u64;
    let spi_ptr: u64 = buf.sorted_pair_idx.as_usize() as u64;
    let spip_ptr: u64 = buf.sorted_pair_idx_padded.as_usize() as u64;

    // 2. Scan counts → padded_offsets (single-block Blelloch).
    {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&c_ptr);
        args.push_arg(&po_ptr);
        args.push_arg(&n_experts_i);
        let cfg = LaunchCfg {
            grid: (1, 1, 1),
            block: (512, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_scan_padded, stream, cfg, args)?;
    }
    // 3. Copy + pad-fill. grid.x covers the worst case (all pairs to one
    // expert, rounded up to mult of 8); blocks that fall outside an
    // expert's padded range early-exit.
    {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&spi_ptr);
        args.push_arg(&o_ptr);
        args.push_arg(&c_ptr);
        args.push_arg(&po_ptr);
        args.push_arg(&spip_ptr);
        args.push_arg(&n_experts_i);
        let max_per_expert = (shape.max_tokens * shape.top_k + 7) & !7;
        let grid_x = (max_per_expert as u32).div_ceil(256);
        let cfg = LaunchCfg {
            grid: (grid_x.max(1), shape.n_experts as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_pad_copy, stream, cfg, args)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Q4K_MOE_MMVQ_R4_MIN_TOKENS;

    /// `m` predicate of the gfx906 dispatch row whose impl is `impl_id`.
    fn row_m(toml: &str, impl_id: &str) -> String {
        let row = toml
            .split("[[")
            .find(|row| row.contains(&format!("impl    = \"{impl_id}\"")))
            .unwrap_or_else(|| panic!("no dispatch row for {impl_id}"));
        let m = row.split("m = \"").nth(1).expect("row has an m predicate");
        m[..m.find('"').unwrap()].to_string()
    }

    /// The launcher picks r2 / r4 by token count; the dispatch rows that
    /// certify them must split at the same boundary.
    #[test]
    fn q4k_moe_mmvq_split_matches_dispatch() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../dispatch/hip/gfx906.toml");
        let toml = std::fs::read_to_string(path).unwrap();
        let t = Q4K_MOE_MMVQ_R4_MIN_TOKENS;
        assert_eq!(row_m(&toml, "indexed_moe_mmvq_q4_k_r2"), format!("1..{}", t - 1));
        assert_eq!(row_m(&toml, "indexed_moe_mmvq_q4_k_r4"), format!(">={t}"));
    }
}
