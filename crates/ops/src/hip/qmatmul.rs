//! Quantised matrix-multiply — MMVQ (decode) + MMQ (prefill) dispatch.
//! This module is the only place where `impl_id` strings get turned into
//! kernel stems + entry names + launch configs. Model code calls
//! [`qmatmul`] (auto-dispatched) or the lower-level [`mmvq`] / [`mmq`]
//! directly when M is statically known.
//! All launches use the committed `dispatch_qmatmul` table in
//! `flambeau-backend-hip`, which mirrors the non-overlapping predicates in
//! `dispatch/hip/gfx906.toml`.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — every unsafe block is a `kernel.launch` or `memcpy_async` \
              over `DevicePtr`s validated by the caller; the kernel stem + entry are \
              resolved through the validated registry and the ABI matches the kernels \
              extern-C signature."
)]

use anyhow::{anyhow, bail, Result};
use flambeau_backend_hip::{dispatch_qmatmul, HipStream, KernelArgs, LaunchCfg};
use flambeau_core::{DevicePtr, QDtype, QMatMulCfg};

use super::OpsRegistry;

/// Auto-dispatched QMatMul. Given M, selects MMVQ (M=1..3 for Q8_0, M=1..127
/// for K-quants) or MMQ (M≥128 for all supported dtypes). For mid-M on
/// K-quants, loops MMVQ across rows — the dispatcher's job is to tell us
/// which impl wins, not to paper over that MMVQ is per-row.
/// Takes both the standard [`flambeau_quant::BlockQ8_1`] layout and the DS4
/// [`flambeau_quant::BlockQ8_1Mmq`] layout. The dispatcher picks:
/// - `act_q8_1` → MMVQ and Mmq4Warp kernels (36 B per-row blocks)
/// - `act_q8_1_mmq` → MmqLdsX64 kernel (144 B DS4 blocks)
///   Decode-path callers that never hit the MmqLdsX64 recipe can pass a null
///   [`DevicePtr`] for `act_q8_1_mmq`; use [`qmatmul_decode`] for ergonomics.
pub fn qmatmul(
    ctx: crate::OpCtx<'_>,
    buf: crate::QmatmulBuffers,
    shape: crate::MatmulShape,
    dtype_weight: QDtype,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::QmatmulBuffers {
        weights,
        act_q8_1,
        act_q8_1_mmq,
        dst,
    } = buf;
    let crate::MatmulShape { m, k, n } = shape;
    // F16 short-circuits the dispatch table: there is no F16 MMQ, so all
    // M (including L>1 prefill) routes through per-row MMVQ.
    if dtype_weight == QDtype::F16 {
        let _ = act_q8_1_mmq;
        // 9.a — tile-M kernel at m >= 8: each block handles 64 output
        // rows × 8 activation rows (512 outputs/block) with weight HBM
        // read once per K-sub-block per thread instead of per activation
        // row. 5 multi-row kept for m < 8 where tile partial-fill
        // hurts grid occupancy.
        if m >= 8 {
            return mmq_f16_tile_launch(reg, stream, weights, act_q8_1, dst, n, m, k);
        }
        return mmq_f16_launch(reg, stream, weights, act_q8_1, dst, n, m, k);
    }
    // Q4_1 at m ∈ {2, 3, 4}: per-N compile-time batched MMVQ.
    // Sibling of K1's Q4_0 batched path. The first cut of Q4_1 batched
    // (runtime-N=8 loop) hit ~59 VGPRs and lost on every anchor — the
    // per-N template compiles down to ~20-30 VGPRs which preserves
    // gfx906 wave occupancy. Outside this window Q4_1 falls through to
    // the dispatch_qmatmul() table (row-by-row m-loop).
    if dtype_weight == QDtype::Q4_1 && (2..=4).contains(&m) {
        mmvq_q4_1_batched(

            crate::OpCtx { reg, stream },

            crate::MmvqBuffers { weights, act_q8_1, dst },

            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },

        )?;
        let _ = act_q8_1_mmq;
        return Ok(());
    }
    if dtype_weight == QDtype::Q8_0 && (2..=4).contains(&m) {
        mmvq_q8_0_row_tile_batched(

            crate::OpCtx { reg, stream },

            crate::MmvqBuffers { weights, act_q8_1, dst },

            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },

        )?;
        let _ = act_q8_1_mmq;
        return Ok(());
    }
    if dtype_weight == QDtype::Q4_K && (2..=4).contains(&m) {
        mmvq_q4_k_batched(

            crate::OpCtx { reg, stream },

            crate::MmvqBuffers { weights, act_q8_1, dst },

            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },

        )?;
        let _ = act_q8_1_mmq;
        return Ok(());
    }
    if dtype_weight == QDtype::Q6_K && (2..=4).contains(&m) {
        mmvq_q6_k_batched(

            crate::OpCtx { reg, stream },

            crate::MmvqBuffers { weights, act_q8_1, dst },

            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },

        )?;
        let _ = act_q8_1_mmq;
        return Ok(());
    }
    if (dtype_weight == QDtype::Q5_1 && m < 32)
        || (dtype_weight == QDtype::Q4_0 && m < 32)
        || (dtype_weight == QDtype::Q5_0 && m < 32)
    {
        // K1 — Q4_0 at m ∈ {2, 3, 4}: single-launch batched MMVQ with
        // compile-time N specialization. Each N has its own kernel
        // entry so VGPR usage stays close to the single-row baseline
        // (~17 VGPRs) and gfx906 occupancy is preserved. Outside this
        // window we keep the row-by-row MMVQ short-circuit:
        //   m == 1     → no amortization to win; single-row hot path
        //   m ∈ {5..7} → no batched specialization yet; falls back
        //   m >= 8     → ≥ 32 routes to MMQ via dispatch table above
        // Output layout matches the row-by-row loop's `dst[i, n]` slot-
        // major convention since the batched kernel writes
        // `dst[s * n_rows + row]` for the same s = row-batch index.
        if dtype_weight == QDtype::Q4_0 && (2..=4).contains(&m) {
            mmvq_q4_0_row_tile_batched(

                crate::OpCtx { reg, stream },

                crate::MmvqBuffers { weights, act_q8_1, dst },

                crate::MmvqBatchShape { n_rows: n, k, n_slots: m },

            )?;
            let _ = act_q8_1_mmq;
            return Ok(());
        }

        let (stem, entry, threads) = match dtype_weight {
            QDtype::Q4_0 => ("mmvq_q4_0", "flambeau_mmvq_q4_0_q8_1", 256u32),
            QDtype::Q5_0 => ("mmvq_q5_0", "flambeau_mmvq_q5_0_q8_1", 256),
            QDtype::Q5_1 => ("mmvq_q5_1", "flambeau_mmvq_q5_1_q8_1", 256),
            _ => unreachable!(),
        };
        let act_row_bytes = (k / 32) * std::mem::size_of::<flambeau_quant::BlockQ8_1>();
        let dst_row_bytes = n * 4;
        assert_eq!(k % 32, 0, "{stem} requires k % 32 == 0");
        let n_blocks = k / 32;
        for i in 0..m {
            mmvq_simple_launch(
                reg,
                stream,
                stem,
                entry,
                weights,
                act_q8_1.offset_bytes(i * act_row_bytes),
                dst.offset_bytes(i * dst_row_bytes),
                n,
                n_blocks,
                threads,
                1,
            )?;
        }
        let _ = act_q8_1_mmq;
        return Ok(());
    }

    // Q5_K at m ∈ {2, 3, 4}: row-tile batched MMVQ (R=8 rows per block,
    // LDS-resident activation strip per super-block). Replaced the
    // earlier r2 batched at this dispatch row; cert.md shows 1.50×–2.68×
    // vs the r2 sibling on decode-class shapes.
    if dtype_weight == QDtype::Q5_K && (2..=4).contains(&m) {
        mmvq_q5_k_row_tile_batched(

            crate::OpCtx { reg, stream },

            crate::MmvqBuffers { weights, act_q8_1, dst },

            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },

        )?;
        let _ = act_q8_1_mmq;
        return Ok(());
    }

    let cfg = QMatMulCfg {
        dtype_weight,
        dtype_activation: QDtype::Q8_1,
        m,
        k,
        n,
    };
    let desc = dispatch_qmatmul(&cfg).ok_or_else(|| {
        anyhow!(
            "no QMatMul impl for dtype={} m={m} k={k} n={n}",
            dtype_weight.name()
        )
    })?;
    let recipe = Recipe::from_impl_id(desc.impl_id)?;

    match recipe.kind {
        RecipeKind::Mmvq => {
            // 4-fix-C: `nb_per_row` (kernel arg) counts WEIGHT
            // superblocks (size = `block_elems(dtype_weight)`); the
            // activation row stride uses Q8_1's QK8_1=32 element blocks.
            // For Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 these coincide (block_elems=32).
            // For Q4_K/Q5_K/Q6_K (block_elems=256=QK_K) they DIVERGE — the
            // pre-fix `act_row_bytes = nb_per_row * sizeof(BlockQ8_1)` was
            // 8× too small, so row i ≥ 1 read partway into row 0's
            // activation. Caused the 4 forward_prefill_pp L>1 bug on
            // every model with K-quant weights routed through this path
            // (e.g. ssm_out=Q5_K on Qwen3.5/3.6 hybrid).
            let nb_per_row = k / block_elems(dtype_weight);
            let act_blocks_per_row = k / 32; // QK8_1 = 32 elements per Q8_1 block.
            let w_row_bytes = 0; // weight doesn't stride per batch
            let act_row_bytes =
                act_blocks_per_row * std::mem::size_of::<flambeau_quant::BlockQ8_1>();
            let dst_row_bytes = n * 4;
            for i in 0..m {
                mmvq_launch(
                    reg,
                    stream,
                    recipe,
                    weights.offset_bytes(i * w_row_bytes),
                    act_q8_1.offset_bytes(i * act_row_bytes),
                    dst.offset_bytes(i * dst_row_bytes),
                    n,
                    nb_per_row,
                )?;
            }
            Ok(())
        }
        RecipeKind::MmqOracle | RecipeKind::Mmq4Warp => {
            let nb_per_row = k / block_elems(dtype_weight);
            mmq_launch(
                reg, stream, recipe, weights, act_q8_1, dst, n, m, nb_per_row,
            )
        }
        RecipeKind::MmqLdsX64 => {
            if act_q8_1_mmq.as_usize() == 0 {
                bail!(
                    "qmatmul dispatched to MmqLdsX64 (impl {}) but caller \
                     did not populate act_q8_1_mmq — run \
                     ops::norm::quantize_f16_q8_1_mmq upstream or use \
                     qmatmul_decode() if this is a decode path",
                    desc.impl_id
                );
            }
            let nb_per_row = k / block_elems(dtype_weight);
            mmq_lds_x64_launch(
                reg,
                stream,
                recipe,
                weights,
                act_q8_1_mmq,
                dst,
                n,
                m,
                nb_per_row,
            )
        }
        RecipeKind::MmqWave64 => {
            mmq_wave64_launch(reg, stream, recipe, weights, act_q8_1, dst, n, m, k)
        }
    }
}

/// Decode-path MMVQ for a single activation row. Caller is responsible for
/// ensuring `m == 1` in the dispatch sense (one Q8_1-quantised vector of K
/// elements). `n_rows` is the output dim (weight rows).
/// Fused gate+up Q8_0 MMVQ for dense shared-expert FFN. Reads Q8_1
/// activation once, computes both matmuls (same as our MoE Q4_K gate_up
/// pattern but simpler — no min subtraction, no expert indexing).
/// Only wired in the DP4A-VDR2 variant; otherwise the caller does two
/// independent `mmvq(...)` calls.
/// **TP-perf-c5** — Q4_0 single-row MMVQ with 128 threads/block (the
/// thin-block schedule that wins for Q4_1 on gfx906 per ).
/// Captures the latency-hiding 2-blocks-per-CU shape that the 256t
/// `mmvq_q4_0` baseline can't reach. Used through env opt-in.
pub fn mmvq_q4_0_t128(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    y_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("mmvq_q4_0_t128")?;
    let kernel = module.kernel("flambeau_mmvq_q4_0_t128_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 128);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// **K1** — Q4_0 batched MMVQ: one launch covers `n_slots` activation
/// rows, each block reads its weight row once and applies it across all
/// slots in the inner kbx loop. Amortizes weight HBM reads across
/// activations; the lever for the `qmatmul(m=N)` row-by-row null
/// observed at L=2 spec verify (see
/// `feedback_qmatmul_small_m_no_amortize`). Sibling of the existing
/// `mmvq_q4_1_q8_1_batched` kernel (this is the same pattern with Q4_0
/// math: `-8·d_x·s_y` bias correction in place of Q4_1's `m_x·s_y`).
///
/// Activation layout: `y_q8_1` is `[n_slots, n_blocks_per_row]`
/// row-major (slot stride = `n_blocks_per_row * sizeof(BlockQ8_1)`).
/// Output layout: `dst` is `[n_slots, n_rows]` slot-major F32 — matches
/// the `qmatmul` ABI's `[m, n] = [batch, output]` convention.
///
/// `n_slots` ∈ [2, 4]. Per-N compile-time specialization keeps VGPR
/// usage close to the single-row baseline (~17 VGPRs) so gfx906 wave
/// occupancy stays at 8+ waves/SIMD. A prior version with a runtime
/// `n_slots` loop bounded by MAX_N=8 measured at 59 VGPRs / 4 waves —
/// occupancy loss drowned the weight-amortization win. Callers outside
/// {2, 3, 4} should fall back to row-by-row MMVQ.
pub fn mmvq_q4_0_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q4_0_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_0_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_0_q8_1_batched_n4",
        _ => bail!(
            "mmvq_q4_0_batched: n_slots={n_slots} outside [2, 4]; \
             single-row callers should use mmvq_q4_0 row-by-row"
        ),
    };
    let module = ctx.reg.expect_module("mmvq_q4_0_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Q4_K sibling of [`mmvq_q4_0_batched`]. Single-row Q4_K MMVQ structure
/// (64-thread wave64, on-the-fly per-element decode) with an inner N-slot
/// loop. Kernel arg `n_blocks_per_row` here is *super-blocks* per row
/// (k / 256), not Q8_1 32-elem blocks — matches the single-row Q4_K MMVQ.
pub fn mmvq_q4_k_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q4_k_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_k_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_k_q8_1_batched_n4",
        _ => bail!("mmvq_q4_k_batched: n_slots={n_slots} outside [2, 4]"),
    };
    let module = ctx.reg.expect_module("mmvq_q4_k_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_superblocks_i = (k / flambeau_quant::QK_K) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_superblocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 64);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Q6_K sibling of [`mmvq_q4_k_batched`]. Single-row Q6_K decode
/// (4 elements per lane per super-block via the 6-bit ql+qh layout)
/// with an inner N-slot loop. Takes `k` (counted in elements, not blocks);
/// kernel arg is `k / QK_K` super-blocks.
pub fn mmvq_q6_k_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q6_k_q8_1_batched_n2",
        3 => "flambeau_mmvq_q6_k_q8_1_batched_n3",
        4 => "flambeau_mmvq_q6_k_q8_1_batched_n4",
        _ => bail!("mmvq_q6_k_batched: n_slots={n_slots} outside [2, 4]"),
    };
    let module = ctx.reg.expect_module("mmvq_q6_k_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_superblocks_i = (k / flambeau_quant::QK_K) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_superblocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 64);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Q8_0 sibling of [`mmvq_q4_0_batched`]. Same per-N compile-time
/// template; symmetric signed-8-bit drops the nibble unpack and the
/// `-8·s_y` correction.
pub fn mmvq_q8_0_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q8_0_q8_1_batched_n2",
        3 => "flambeau_mmvq_q8_0_q8_1_batched_n3",
        4 => "flambeau_mmvq_q8_0_q8_1_batched_n4",
        _ => bail!(
            "mmvq_q8_0_batched: n_slots={n_slots} outside [2, 4]; \
             single-row callers should use mmvq_q8_0 row-by-row"
        ),
    };
    let module = ctx.reg.expect_module("mmvq_q8_0_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Q4_1 sibling of [`mmvq_q4_0_batched`]. Same per-N compile-time
/// template pattern (n2/n3/n4 entries) — Q4_1 carries an explicit
/// `min` per block, so the per-block bias is `+ m_x · s_y` rather than
/// Q4_0's `- 8 · d_x · s_y`. Targets the same dispatch window
/// (`n_slots` ∈ [2, 4]) and the same weight-amortization regime.
pub fn mmvq_q4_1_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q4_1_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_1_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_1_q8_1_batched_n4",
        _ => bail!(
            "mmvq_q4_1_batched: n_slots={n_slots} outside [2, 4]; \
             single-row callers should use mmvq_q4_1 row-by-row"
        ),
    };
    let module = ctx.reg.expect_module("mmvq_q4_1_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// **K3** — Q5_K batched MMVQ with r2 multi-row + per-N activation
/// columns. Sibling of [`mmvq_q4_0_batched`] for Q5_K weights (the
/// `ssm_*` projections in GDN layers — second-largest decode bucket
/// per S2 rocprof at 760 ms / spec phase). Same lever: one launch
/// covers N activation rows; each block reads its Q5_K super-block
/// once and applies it across the N columns. Block grid:
/// `ceil(n_rows / 2)` (r2 retained from the single-col Q5_K kernel).
///
/// `n_slots` ∈ [2, 4]. Outside that the caller falls back to the
/// row-by-row m-loop in [`qmatmul`].
pub fn mmvq_q5_k_r2_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q5_k_r2_q8_1_batched_n2",
        3 => "flambeau_mmvq_q5_k_r2_q8_1_batched_n3",
        4 => "flambeau_mmvq_q5_k_r2_q8_1_batched_n4",
        _ => bail!("mmvq_q5_k_r2_batched: n_slots={n_slots} outside [2, 4]"),
    };
    let module = ctx.reg.expect_module("mmvq_q5_k_r2_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_superblocks_i = (k / 256) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_superblocks_i);
    // 64 threads/block = 1 wave64 (r2 multi-row pattern); grid = ceil(n_rows / 2).
    let n_row_pairs = n_rows.div_ceil(2) as u32;
    let cfg = LaunchCfg::one_d(n_row_pairs, 64);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Row-tiled sibling of [`mmvq_q5_k_r2_batched`]. Each block owns
/// `R = 8` consecutive rows (4 wave64 × half-warp split) and shares one
/// LDS-resident Q8_1 activation strip per super-block across all N
/// decode slots; cuts activation HBM traffic ~8× vs the r2 kernel at
/// the same output count.
///
/// `n_slots` ∈ [2, 4]. Output ABI identical to `mmvq_q5_k_r2_batched`
/// (`dst[N, n_rows]` slot-major F32).
pub fn mmvq_q5_k_row_tile_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n2",
        3 => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n3",
        4 => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n4",
        _ => bail!("mmvq_q5_k_row_tile_batched: n_slots={n_slots} outside [2, 4]"),
    };
    let module = ctx.reg.expect_module("mmvq_q5_k_row_tile_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_superblocks_i = (k / 256) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_superblocks_i);
    let grid = (n_rows as u32).div_ceil(8);
    let cfg = LaunchCfg::one_d(grid, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// **TP-perf-c5** — fused gate+up Q4_0 t128. Same gate+up activation
/// sharing as `mmvq_q4_0_gate_up`, but with the t128 schedule for the
/// gfx906 latency-bound regime.
pub fn mmvq_q4_0_gate_up_t128(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqGateUpBuffers,
    shape: crate::MmvqGateUpShape,
) -> Result<()> {
    let crate::MmvqGateUpBuffers { gate_w, up_w, act_q8_1: y_q8_1, gate_out, up_out } = buffers;
    let crate::MmvqGateUpShape { n_rows_gate, n_rows_up, k } = shape;
    let module = ctx.reg.expect_module("mmvq_q4_0_gate_up_t128_dp4a")?;
    let kernel = module.kernel("flambeau_mmvq_q4_0_gate_up_t128_dp4a_q8_1")?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&gw_ptr);
    args.push(&uw_ptr);
    args.push(&y_ptr);
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&n_rows_g);
    args.push(&n_rows_u);
    args.push(&n_blocks_i);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, 128);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// **C6-i1** — Q4_0 single-warp (64 t/block) MMVQ. Sibling of `mmvq_q4_0_t128`
/// targeting an even smaller block to pack more in-flight blocks per CU at
/// gfx906 decode (where occupancy is the lever, not VALU). One wave64/block
/// = no LDS reduce, just an in-place gfx906 DPP butterfly.
pub fn mmvq_q4_0_warpcoop64(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    y_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("mmvq_q4_0_warpcoop64")?;
    let kernel = module.kernel("flambeau_mmvq_q4_0_warpcoop64_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 64);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// **TP-perf-c3** — fused K+V Q4_0 dense MMVQ with F16 destination.
/// K and V always share shape (`[local_n_kv_heads · head_dim, hidden]`)
/// in TP full-attn layers and read the same Q8_1 activation. This
/// kernel collapses both MMVQs and both downstream `cast_f32_to_f16`
/// launches into a single launch — net 3 launches saved per full-attn
/// layer per rank.
pub fn mmvq_q4_0_kv_f16dst(
    reg: &OpsRegistry,
    stream: &HipStream,
    k_w: DevicePtr,
    v_w: DevicePtr,
    y_q8_1: DevicePtr,
    k_out_f16: DevicePtr,
    v_out_f16: DevicePtr,
    n_rows_kv: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("mmvq_q4_0_kv_f16dst_dp4a")?;
    let kernel = module.kernel("flambeau_mmvq_q4_0_kv_f16dst_dp4a_q8_1")?;
    let n_rows_i = n_rows_kv as i32;
    let n_blocks_i = (k / 32) as i32;
    let kw_ptr: u64 = k_w.as_usize() as u64;
    let vw_ptr: u64 = v_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let kout_ptr: u64 = k_out_f16.as_usize() as u64;
    let vout_ptr: u64 = v_out_f16.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&kw_ptr);
    args.push(&vw_ptr);
    args.push(&y_ptr);
    args.push(&kout_ptr);
    args.push(&vout_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows_kv as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// **TP-perf-c1** — fused gate+up Q4_0 dense MMVQ. Same shape contract
/// and asymmetric-row support as `mmvq_q8_0_gate_up`, just for Q4_0
/// weights. Halves the call count (and the Q8_1 activation HBM read)
/// when both weights are Q4_0 — the common case on Qwen3.6-x-Q4_0
/// (attn_qkv+attn_gate in GDN, ffn_gate+ffn_up in dense FFN).
pub fn mmvq_q4_0_gate_up(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqGateUpBuffers,
    shape: crate::MmvqGateUpShape,
) -> Result<()> {
    let crate::MmvqGateUpBuffers { gate_w, up_w, act_q8_1: y_q8_1, gate_out, up_out } = buffers;
    let crate::MmvqGateUpShape { n_rows_gate, n_rows_up, k } = shape;
    let module = ctx.reg.expect_module("mmvq_q4_0_gate_up_dp4a")?;
    let kernel = module.kernel("flambeau_mmvq_q4_0_gate_up_dp4a_q8_1")?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&gw_ptr);
    args.push(&uw_ptr);
    args.push(&y_ptr);
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&n_rows_g);
    args.push(&n_rows_u);
    args.push(&n_blocks_i);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// **K5** — Q4_0 fused gate+up MMVQ batched across N activation cols.
/// Sibling of [`mmvq_q4_0_gate_up`] (single-col fused) and
/// [`mmvq_q4_0_batched`] (multi-col single-weight). Combines both
/// levers: each block reads ONE gate weight row + ONE up weight row +
/// ONE shared activation strip per col, computing 2 × N output values
/// per (col, row) pair.
///
/// Output layout: `gate_out[N, n_rows_gate]`, `up_out[N, n_rows_up]`,
/// slot-major F32. Asymmetric-row support preserved (n_rows_gate vs
/// n_rows_up may differ; per-row do_gate/do_up short-circuit).
///
/// `n_slots` ∈ [2, 4]. Used by the K6 batched-GDN paired-L=2 forward.
pub fn mmvq_q4_0_gate_up_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqGateUpBuffers,
    shape: crate::MmvqGateUpBatchShape,
) -> Result<()> {
    let crate::MmvqGateUpBuffers { gate_w, up_w, act_q8_1: y_q8_1, gate_out, up_out } = buffers;
    let crate::MmvqGateUpBatchShape { n_rows_gate, n_rows_up, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n4",
        _ => bail!("mmvq_q4_0_gate_up_batched: n_slots={n_slots} outside [2, 4]"),
    };
    let module = ctx.reg.expect_module("mmvq_q4_0_gate_up_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&gw_ptr);
    args.push(&uw_ptr);
    args.push(&y_ptr);
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&n_rows_g);
    args.push(&n_rows_u);
    args.push(&n_blocks_i);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Row-tiled sibling of [`mmvq_q4_0_gate_up_batched`]. Each block owns
/// `R = 4` consecutive rows of the gate+up matmul and shares one
/// LDS-resident Q8_1 activation strip across the N decode slots; cuts
/// activation HBM traffic ~4× vs the per-row K5 kernel at the same
/// output count.
///
/// `n_slots` ∈ [2, 4]. Output ABI identical to `mmvq_q4_0_gate_up_batched`
/// (`gate_out[N, n_rows_gate]`, `up_out[N, n_rows_up]`, slot-major F32).
pub fn mmvq_q4_0_gate_up_row_tile_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqGateUpBuffers,
    shape: crate::MmvqGateUpBatchShape,
) -> Result<()> {
    let crate::MmvqGateUpBuffers { gate_w, up_w, act_q8_1: y_q8_1, gate_out, up_out } = buffers;
    let crate::MmvqGateUpBatchShape { n_rows_gate, n_rows_up, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n4",
        _ => bail!("mmvq_q4_0_gate_up_row_tile_batched: n_slots={n_slots} outside [2, 4]"),
    };
    let module = ctx.reg.expect_module("mmvq_q4_0_gate_up_row_tile_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&gw_ptr);
    args.push(&uw_ptr);
    args.push(&y_ptr);
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&n_rows_g);
    args.push(&n_rows_u);
    args.push(&n_blocks_i);
    let max_rows = n_rows_gate.max(n_rows_up);
    let grid = (max_rows as u32).div_ceil(4);
    let cfg = LaunchCfg::one_d(grid, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Row-tiled sibling of [`mmvq_q4_0_batched`]. Each block owns `R = 4`
/// consecutive output rows and shares one LDS-resident Q8_1 activation
/// strip across the N decode slots. Single-weight-matrix variant of
/// [`mmvq_q4_0_gate_up_row_tile_batched`] for projections that aren't
/// gate+up fused (`ssm_out`, attention output_proj, etc.).
///
/// `n_slots` ∈ [2, 4]. Output ABI identical to `mmvq_q4_0_batched`
/// (`dst[N, n_rows]`, slot-major F32).
pub fn mmvq_q4_0_row_tile_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n4",
        _ => bail!("mmvq_q4_0_row_tile_batched: n_slots={n_slots} outside [2, 4]"),
    };
    let module = ctx.reg.expect_module("mmvq_q4_0_row_tile_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let grid = (n_rows as u32).div_ceil(4);
    let cfg = LaunchCfg::one_d(grid, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Row-tiled sibling of `mmvq_q8_0_batched`. R=4 output rows per block
/// share one LDS-resident Q8_1 activation strip across the N decode
/// slots; cuts activation HBM traffic ~4× vs the per-row Q8_0 batched
/// kernel at the same output count. Used for non-fused Q8_0
/// projections (GDN α/β on Qwen3.6 hybrids, etc.).
///
/// `n_slots` ∈ [2, 4]. Output ABI identical to `mmvq_q8_0_batched`
/// (`dst[N, n_rows]`, slot-major F32).
pub fn mmvq_q8_0_row_tile_batched(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let entry = match n_slots {
        2 => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n4",
        _ => bail!("mmvq_q8_0_row_tile_batched: n_slots={n_slots} outside [2, 4]"),
    };
    let module = ctx.reg.expect_module("mmvq_q8_0_row_tile_batched")?;
    let kernel = module.kernel(entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let grid = (n_rows as u32).div_ceil(4);
    let cfg = LaunchCfg::one_d(grid, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// **C8-i1** — fused gate+up Q4_1 dense MMVQ. Sibling of `mmvq_q4_0_gate_up`
/// for Q4_1 weights (Qwen3.5-9B-Q4_1 / 27B-Q4_1 dense FFN). Reads the Q8_1
/// activation once per block, produces both gate and up outputs.
pub fn mmvq_q4_1_gate_up(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqGateUpBuffers,
    shape: crate::MmvqGateUpShape,
) -> Result<()> {
    let crate::MmvqGateUpBuffers { gate_w, up_w, act_q8_1: y_q8_1, gate_out, up_out } = buffers;
    let crate::MmvqGateUpShape { n_rows_gate, n_rows_up, k } = shape;
    let module = ctx.reg.expect_module("mmvq_q4_1_gate_up_dp4a")?;
    let kernel = module.kernel("flambeau_mmvq_q4_1_gate_up_dp4a_q8_1")?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&gw_ptr);
    args.push(&uw_ptr);
    args.push(&y_ptr);
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&n_rows_g);
    args.push(&n_rows_u);
    args.push(&n_blocks_i);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

pub fn mmvq_q8_0_gate_up(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqGateUpBuffers,
    shape: crate::MmvqGateUpShape,
) -> Result<()> {
    let crate::MmvqGateUpBuffers { gate_w, up_w, act_q8_1: y_q8_1, gate_out, up_out } = buffers;
    let crate::MmvqGateUpShape { n_rows_gate, n_rows_up, k } = shape;
    let (stem, entry, threads) = (
        "mmvq_q8_0_gate_up_t128_vdr2",
        "flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1",
        128u32,
    );
    let module = ctx.reg.expect_module(stem)?;
    let kernel = module.kernel(entry)?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&gw_ptr);
    args.push(&uw_ptr);
    args.push(&y_ptr);
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&n_rows_g);
    args.push(&n_rows_u);
    args.push(&n_blocks_i);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, threads);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Fused gate+up Q5_K dense MMVQ. Mirror of [`mmvq_q8_0_gate_up`]: reads
/// the Q8_1 activation once per inner-loop block and computes both dot
/// products. Used by the qwen3.5/3.6 shared-expert FFN path when
/// `ffn_gate_shexp` and `ffn_up_shexp` are both Q5_K.
pub fn mmvq_q5_k_gate_up(
    ctx: crate::OpCtx<'_>,
    buffers: crate::MmvqGateUpBuffers,
    shape: crate::MmvqGateUpShape,
) -> Result<()> {
    let crate::MmvqGateUpBuffers { gate_w, up_w, act_q8_1: y_q8_1, gate_out, up_out } = buffers;
    let crate::MmvqGateUpShape { n_rows_gate, n_rows_up, k } = shape;
    if k % flambeau_quant::QK_K != 0 {
        bail!(
            "mmvq_q5_k_gate_up: k={k} must be a multiple of QK_K={}",
            flambeau_quant::QK_K
        );
    }
    let module = ctx.reg.expect_module("mmvq_q5_k_gate_up_dp4a")?;
    let kernel = module.kernel("flambeau_mmvq_q5_k_gate_up_dp4a_q8_1")?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_sb_per_row = (k / flambeau_quant::QK_K) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&gw_ptr);
    args.push(&uw_ptr);
    args.push(&y_ptr);
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&n_rows_g);
    args.push(&n_rows_u);
    args.push(&n_sb_per_row);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, 256);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

pub fn mmvq(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    k: usize,
    dtype_weight: QDtype,
) -> Result<()> {
    if dtype_weight == QDtype::F16 {
        return mmvq_f16_launch(reg, stream, weights, act_q8_1, dst, n_rows, k);
    }
    // 3.a — Q4_0 / Q5_0 bypass the dispatch table. Unblock Qwen3.6-35B
    // -A3B-Q4_0 which uses Q4_0 for attn/FFN and Q5_0 for shared-expert FFN.
    // Single productionised kernel per dtype, no shape-dependent selection.
    // 8.d NULL: tried the r2 half-warp-per-row variant (mmvq_q4_0_r2.cu,
    // moved to _unverified/) — it drops the DP4A advantage of the single-row
    // kernel (F32 FMA per lane vs DP4A 4×INT8 per lane). Measured −21 %
    // decode regression AND a logit re-accumulation-order argmax shift on
    // seed 9419. The candle P29 r2 pattern wins on K-quants (sub-block
    // scales block DP4A) but strictly loses on Q4_0's flat-block DP4A path.
    assert_eq!(k % 32, 0, "MMVQ requires k % 32 == 0");
    let n_blocks_q32 = k / 32;
    if dtype_weight == QDtype::Q4_0 {
        return mmvq_simple_launch(
            reg,
            stream,
            "mmvq_q4_0",
            "flambeau_mmvq_q4_0_q8_1",
            weights,
            act_q8_1,
            dst,
            n_rows,
            n_blocks_q32,
            256,
            1,
        );
    }
    if dtype_weight == QDtype::Q5_0 {
        return mmvq_simple_launch(
            reg,
            stream,
            "mmvq_q5_0",
            "flambeau_mmvq_q5_0_q8_1",
            weights,
            act_q8_1,
            dst,
            n_rows,
            n_blocks_q32,
            256,
            1,
        );
    }
    if dtype_weight == QDtype::Q5_1 {
        return mmvq_simple_launch(
            reg,
            stream,
            "mmvq_q5_1",
            "flambeau_mmvq_q5_1_q8_1",
            weights,
            act_q8_1,
            dst,
            n_rows,
            n_blocks_q32,
            256,
            1,
        );
    }
    let cfg = QMatMulCfg {
        dtype_weight,
        dtype_activation: QDtype::Q8_1,
        m: 1,
        k,
        n: n_rows,
    };
    let desc = dispatch_qmatmul(&cfg)
        .ok_or_else(|| anyhow!("no MMVQ impl for dtype={} k={k}", dtype_weight.name()))?;
    let recipe = Recipe::from_impl_id(desc.impl_id)?;
    if recipe.kind != RecipeKind::Mmvq {
        bail!("dispatch for m=1 returned non-MMVQ impl {}", desc.impl_id);
    }
    let nb_per_row = k / block_elems(dtype_weight);
    mmvq_launch(
        reg, stream, recipe, weights, act_q8_1, dst, n_rows, nb_per_row,
    )
}

/// Weight × Q8_1 MMVQ writing directly into an F16 destination, saturating
/// at ±F16_MAX. Equivalent shape to [`mmvq`] but skips the `mmvq_f32`
/// scratch + the separate `cast_f32_to_f16` launch. Kernel bodies are
/// shared with the F32 variants via templated `__device__` thunks in
/// `mmvq_q{4_0, 4_1, 8_0}*.cu`; this entry point is opt-in per consumer
/// (#120). Caller must ensure `dst_f16` is sized `n_rows × sizeof(fp16)`.
///
/// Supported dtypes: `Q4_0`, `Q4_1`, `Q8_0`. Other dtypes bail —
/// extending the set is mechanical (add the templated thunk in the
/// kernel + a `match` arm here).
pub fn mmvq_f16_direct(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst_f16: DevicePtr,
    n_rows: usize,
    k: usize,
    dtype_weight: QDtype,
) -> Result<()> {
    // Per-dtype (stem, entry, threads, rows_per_block, units_per_row).
    // units_per_row matches the kernel's second-to-last int param:
    //   - Q4_0..Q8_0  → n_blocks_per_row     = k / 32
    //   - Q2_K..Q8_K  → n_superblocks_per_row = k / QK_K (256)
    assert_eq!(k % 32, 0, "mmvq_f16_direct requires k % 32 == 0");
    let n_blocks_q32 = k / 32;
    let n_superblocks = k / 256;
    let (stem, entry, threads, rows_per_block, units) = match dtype_weight {
        QDtype::Q4_0 => (
            "mmvq_q4_0",
            "flambeau_mmvq_q4_0_q8_1_f16",
            256u32,
            1u32,
            n_blocks_q32,
        ),
        QDtype::Q4_1 => (
            "mmvq_q4_1_t128",
            "flambeau_mmvq_q4_1_t128_q8_1_f16",
            128,
            1,
            n_blocks_q32,
        ),
        QDtype::Q5_0 => (
            "mmvq_q5_0",
            "flambeau_mmvq_q5_0_q8_1_f16",
            256,
            1,
            n_blocks_q32,
        ),
        QDtype::Q5_1 => (
            "mmvq_q5_1",
            "flambeau_mmvq_q5_1_q8_1_f16",
            256,
            1,
            n_blocks_q32,
        ),
        QDtype::Q8_0 => (
            "mmvq_q8_0_t128_vdr2",
            "flambeau_mmvq_q8_0_t128_vdr2_q8_1_f16",
            128,
            1,
            n_blocks_q32,
        ),
        QDtype::Q2_K => (
            "mmvq_q2_k_r2_dp4a",
            "flambeau_mmvq_q2_k_r2_dp4a_q8_1_f16",
            64,
            2,
            n_superblocks,
        ),
        QDtype::Q3_K => (
            "mmvq_q3_k_r2_dp4a",
            "flambeau_mmvq_q3_k_r2_dp4a_q8_1_f16",
            64,
            2,
            n_superblocks,
        ),
        QDtype::Q4_K => (
            "mmvq_q4_k_r2_dp4a",
            "flambeau_mmvq_q4_k_r2_dp4a_q8_1_f16",
            64,
            2,
            n_superblocks,
        ),
        QDtype::Q5_K => (
            "mmvq_q5_k_r2",
            "flambeau_mmvq_q5_k_r2_q8_1_f16",
            64,
            2,
            n_superblocks,
        ),
        QDtype::Q6_K => (
            "mmvq_q6_k_dp4a",
            "flambeau_mmvq_q6_k_dp4a_q8_1_f16",
            64,
            1,
            n_superblocks,
        ),
        QDtype::Q8_K => (
            "mmvq_q8_k",
            "flambeau_mmvq_q8_K_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        QDtype::IQ1_S => (
            "mmvq_iq1_s_dp4a",
            "flambeau_mmvq_iq1_s_dp4a_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        QDtype::IQ1_M => (
            "mmvq_iq1_m_dp4a",
            "flambeau_mmvq_iq1_m_dp4a_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        QDtype::IQ2_XXS => (
            "mmvq_iq2_xxs_dp4a",
            "flambeau_mmvq_iq2_xxs_dp4a_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        QDtype::IQ2_XS => (
            "mmvq_iq2_xs_dp4a",
            "flambeau_mmvq_iq2_xs_dp4a_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        QDtype::IQ2_S => (
            "mmvq_iq2_s_dp4a",
            "flambeau_mmvq_iq2_s_dp4a_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        QDtype::IQ3_XXS => (
            "mmvq_iq3_xxs_dp4a",
            "flambeau_mmvq_iq3_xxs_dp4a_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        QDtype::IQ3_S => (
            "mmvq_iq3_s_dp4a",
            "flambeau_mmvq_iq3_s_dp4a_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        QDtype::IQ4_NL => (
            "mmvq_iq4_nl_dp4a",
            "flambeau_mmvq_iq4_nl_dp4a_q8_1_f16",
            256,
            1,
            n_blocks_q32,
        ),
        QDtype::IQ4_XS => (
            "mmvq_iq4_xs_dp4a",
            "flambeau_mmvq_iq4_xs_dp4a_q8_1_f16",
            256,
            1,
            n_superblocks,
        ),
        other => bail!("mmvq_f16_direct: no F16-direct kernel for {}", other.name()),
    };
    mmvq_simple_launch(
        reg,
        stream,
        stem,
        entry,
        weights,
        act_q8_1,
        dst_f16,
        n_rows,
        units,
        threads,
        rows_per_block,
    )
}

/// Common launch path for Q-weight MMVQ kernels taking
/// (w, y_q8_1, dst, n_rows, n_units_per_row).
///
/// Caller must pass `(threads, rows_per_block)` explicitly. `threads`
/// is the kernel's `blockDim.x`; over-launching against a kernel with
/// `__launch_bounds__` triggers a HIP launch failure, under-launching
/// silently produces wrong results (warp-count math reads stale lanes).
/// `rows_per_block` is how many output rows a single block writes —
/// 1 for traditional single-row kernels, 2 for r2 K-quant multi-row.
/// Grid = `ceil(n_rows / rows_per_block)`. `units_per_row` is the
/// caller-defined inner-loop count (n_blocks for 32-element-block
/// dtypes, n_superblocks for K-quants). #120 / #120-followup.
pub fn mmvq_simple_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    module_stem: &'static str,
    kernel_entry: &'static str,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    units_per_row: usize,
    threads: u32,
    rows_per_block: u32,
) -> Result<()> {
    let module = reg.expect_module(module_stem)?;
    let kernel = module.kernel(kernel_entry)?;
    let n_rows_i = n_rows as i32;
    let n_units_i = units_per_row as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_units_i);
    let grid = (n_rows as u32).div_ceil(rows_per_block);
    let cfg = LaunchCfg::one_d(grid, threads);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 9.a — tile-M F16 MMQ. 64 threads/block = wave64 × 1 row each, MMQ_Y
/// = 64 output rows per block, MMQ_X = 8 activation rows per block. Each
/// thread owns one weight row's dot against all 8 activation rows; weight
/// is loaded into registers once per K-sub-block and reused 8×. The
/// activation tile (8 Q8_1 blocks) is read from L1 per sub-block across
/// the 64 threads.
fn mmq_f16_tile_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    k: usize,
) -> Result<()> {
    assert_eq!(k % 32, 0, "mmq_f16_tile requires k % 32 == 0");
    if n_tokens == 0 {
        return Ok(());
    }
    let module = reg.expect_module("mmq_f16_tile")?;
    let kernel = module.kernel("flambeau_mmq_f16_tile_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&n_blocks_i);
    // Block = 64 threads (wave64), grid = (⌈n_rows / 64⌉, ⌈n_tokens / 8⌉).
    let cfg = LaunchCfg {
        grid: (
            (n_rows as u32).div_ceil(64),
            (n_tokens as u32).div_ceil(8),
            1,
        ),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 5.a — multi-row F16 MMQ. Same kernel math as mmvq_f16_launch, but
/// picks up n_tokens via grid.y. Stem `mmq_f16_q8_1`, block = 256, grid =
/// (n_rows, n_tokens).
fn mmq_f16_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    k: usize,
) -> Result<()> {
    assert_eq!(k % 32, 0, "mmq_f16_q8_1 requires k % 32 == 0");
    if n_tokens == 0 {
        return Ok(());
    }
    let module = reg.expect_module("mmq_f16_q8_1")?;
    let kernel = module.kernel("flambeau_mmq_f16_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&n_blocks_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, n_tokens as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Direct launch for the F16-weight × Q8_1-activation MMVQ.
/// Kernel stem `mmvq_f16_q8_1`, block = 256 threads, grid = n_rows.
fn mmvq_f16_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    assert_eq!(k % 32, 0, "mmvq_f16_q8_1 requires k % 32 == 0");
    let module = reg.expect_module("mmvq_f16_q8_1")?;
    let kernel = module.kernel("flambeau_mmvq_f16_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Prefill-path MMQ for M batch rows. `m` must be ≥ the dispatch floor for
/// the dtype (e.g. 128 for Q4_K/Q6_K/Q8_0 4-warp).
pub fn mmq(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    m: usize,
    k: usize,
    n: usize,
    dtype_weight: QDtype,
) -> Result<()> {
    let cfg = QMatMulCfg {
        dtype_weight,
        dtype_activation: QDtype::Q8_1,
        m,
        k,
        n,
    };
    let desc = dispatch_qmatmul(&cfg).ok_or_else(|| {
        anyhow!(
            "no MMQ impl for dtype={} m={m} k={k} n={n}",
            dtype_weight.name()
        )
    })?;
    let recipe = Recipe::from_impl_id(desc.impl_id)?;
    let nb_per_row = k / block_elems(dtype_weight);
    mmq_launch(
        reg, stream, recipe, weights, act_q8_1, dst, n, m, nb_per_row,
    )
}

// ---------------------------------------------------------------------------
// internal: impl_id → launch recipe
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecipeKind {
    Mmvq,
    MmqOracle,
    Mmq4Warp,
    /// 5 port of candle's 4-warp LDS-tiled MMQ with DS4 Q8_1 activation
    /// layout. Differs from `Mmq4Warp` in: 2D block dims (64, 4, 1), 9 scalar
    /// args (ncols_x, nrows_x, ncols_y, stride_col_y, stride_row_x, nrows_dst),
    /// dynamic shared-memory bytes, and the Y buffer pre-formatted as
    /// `BlockQ8_1Mmq` via `flambeau_quantize_q8_1_mmq`.
    MmqLdsX64,
    /// wave64 MMQ for Q5_K (candle port `mul_mat_q5_K_gfx906_v2`).
    /// Consumes the standard `flambeau_block_q8_1` layout (NOT DS4), so no
    /// special activation quantise is needed at call sites. Grid =
    /// (ceil(N/64), ceil(M/8)), block = 64 threads (1 warp), 8 output cols
    /// per tile. Args: `(vx, vy, dst, ncols_x=K, nrows_x=N, ncols_y=M,
    /// nrows_y=K, nrows_dst=N)`.
    MmqWave64,
}

#[derive(Debug, Clone, Copy)]
struct Recipe {
    kind: RecipeKind,
    stem: &'static str,
    entry: &'static str,
    threads: u32,
    /// MMVQ only — output rows per block.
    rows_per_block: u32,
    /// Mmq4Warp only — (rows_per_tile, batches_per_tile).
    mmq_tile: (u32, u32),
}

/// MmqLdsX64-only per-recipe constants: weight block elements + dynamic LDS
/// bytes. Indexed by recipe `stem`. Centralised here (not on Recipe) so the
/// 30+ MMVQ/MmqWave64 recipes don't carry irrelevant fields.
fn mmq_lds_x64_params(stem: &str) -> Result<(u32, u32)> {
    Ok(match stem {
        // Q4_0 / Q4_1 4warp_lds: MMQ_Y=128, MMQ_X=64, block_elems=32. LDS
        // budget per 5 derivation in mmq_lds_x64_launch.
        "mmq_q4_0_4warp_lds" | "mmq_q4_1_4warp_lds" => (32, 7584 * 4),
        // Q4_K turbo: MMQ_Y=128, MMQ_X=16, block_elems=256 (QK_K). LDS:
        // tile_y(MMQ_X*36=576) + x_qs(4224) + x_dm(128 half2=128 ints) +
        // x_sc(528) = 5456 ints = 21824 B; round up to 22528 B.
        "mmq_q4_K_turbo" => (256, 22528),
        other => bail!("mmq_lds_x64_params: no entry for stem {other}"),
    })
}

impl Recipe {
    fn from_impl_id(impl_id: &str) -> Result<Self> {
        // C9-followup — Q8_0 t128_vdr2 (combine occupancy + ILP).
        // Single-row Q8_0 dispatch rows are aliased to the t128_vdr2 kernel
        // (≥+3 % on Qwen3.6-27B-Q8_0 and Qwen3.5-27B-Q8_0).
        let impl_id = match impl_id {
            "qmatmul_q8_0_mmvq_single_row_gfx906" => "qmatmul_q8_0_mmvq_t128_vdr2_gfx906",
            "qmatmul_q8_0_mmvq_dp4a_vdr2_gfx906" => "qmatmul_q8_0_mmvq_t128_vdr2_gfx906",
            other => other,
        };
        Ok(match impl_id {
            "qmatmul_q8_0_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q8_0",
                entry: "flambeau_mmvq_q8_0_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q8_0_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q8_0_dp4a",
                entry: "flambeau_mmvq_q8_0_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q8_0_mmvq_dp4a_vdr2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q8_0_dp4a_vdr2",
                entry: "flambeau_mmvq_q8_0_dp4a_vdr2_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q8_0_mmvq_llamacpp_style_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q8_0_llamacpp_style",
                entry: "flambeau_mmvq_q8_0_llamacpp_style_q8_1",
                threads: 128,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q8_0_mmvq_t128_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q8_0_t128",
                entry: "flambeau_mmvq_q8_0_t128_q8_1",
                threads: 128,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q8_0_mmvq_t128_vdr2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q8_0_t128_vdr2",
                entry: "flambeau_mmvq_q8_0_t128_vdr2_q8_1",
                threads: 128,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q2_K_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q2_k",
                entry: "flambeau_mmvq_q2_K_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q2_K_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q2_k_r2",
                entry: "flambeau_mmvq_q2_K_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q2_K_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q2_k_dp4a",
                entry: "flambeau_mmvq_q2_k_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q2_K_mmvq_r2_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q2_k_r2_dp4a",
                entry: "flambeau_mmvq_q2_k_r2_dp4a_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q3_K_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q3_k",
                entry: "flambeau_mmvq_q3_k_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q3_K_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q3_k_r2",
                entry: "flambeau_mmvq_q3_k_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q3_K_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q3_k_dp4a",
                entry: "flambeau_mmvq_q3_k_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q3_K_mmvq_r2_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q3_k_r2_dp4a",
                entry: "flambeau_mmvq_q3_k_r2_dp4a_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q8_K_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q8_k",
                entry: "flambeau_mmvq_q8_K_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q4_K_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_k_r2",
                entry: "flambeau_mmvq_q4_k_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q4_K_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_k_r2_dp4a",
                entry: "flambeau_mmvq_q4_k_r2_dp4a_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q4_K_mmvq_nw1_r4_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_k_r4",
                entry: "flambeau_mmvq_q4_k_r4_q8_1",
                threads: 64,
                rows_per_block: 4,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq4_nl_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq4_nl",
                entry: "flambeau_mmvq_iq4_nl_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq4_nl_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq4_nl_r2",
                entry: "flambeau_mmvq_iq4_nl_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq4_nl_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq4_nl_dp4a",
                entry: "flambeau_mmvq_iq4_nl_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq4_xs_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq4_xs",
                entry: "flambeau_mmvq_iq4_xs_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq4_xs_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq4_xs_r2",
                entry: "flambeau_mmvq_iq4_xs_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq4_xs_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq4_xs_dp4a",
                entry: "flambeau_mmvq_iq4_xs_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq3_xxs_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq3_xxs",
                entry: "flambeau_mmvq_iq3_xxs_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq3_xxs_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq3_xxs_r2",
                entry: "flambeau_mmvq_iq3_xxs_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq3_xxs_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq3_xxs_dp4a",
                entry: "flambeau_mmvq_iq3_xxs_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq3_s_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq3_s",
                entry: "flambeau_mmvq_iq3_s_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq3_s_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq3_s_r2",
                entry: "flambeau_mmvq_iq3_s_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq3_s_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq3_s_dp4a",
                entry: "flambeau_mmvq_iq3_s_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_xxs_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_xxs",
                entry: "flambeau_mmvq_iq2_xxs_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_xxs_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_xxs_r2",
                entry: "flambeau_mmvq_iq2_xxs_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_xxs_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_xxs_dp4a",
                entry: "flambeau_mmvq_iq2_xxs_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_xs_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_xs",
                entry: "flambeau_mmvq_iq2_xs_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_xs_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_xs_r2",
                entry: "flambeau_mmvq_iq2_xs_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_xs_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_xs_dp4a",
                entry: "flambeau_mmvq_iq2_xs_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_s_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_s",
                entry: "flambeau_mmvq_iq2_s_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_s_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_s_r2",
                entry: "flambeau_mmvq_iq2_s_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq2_s_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq2_s_dp4a",
                entry: "flambeau_mmvq_iq2_s_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq1_s_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq1_s",
                entry: "flambeau_mmvq_iq1_s_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq1_s_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq1_s_r2",
                entry: "flambeau_mmvq_iq1_s_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq1_s_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq1_s_dp4a",
                entry: "flambeau_mmvq_iq1_s_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq1_m_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq1_m",
                entry: "flambeau_mmvq_iq1_m_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq1_m_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq1_m_r2",
                entry: "flambeau_mmvq_iq1_m_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_iq1_m_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_iq1_m_dp4a",
                entry: "flambeau_mmvq_iq1_m_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            //— dense MMQ wave64 for IQ family.
            // MMQ_Y=64, TILE_N=8 (same tile shape as the K-quant wave64 MMQ).
            "qmatmul_iq4_xs_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq4_xs_wave64",
                entry: "flambeau_mmq_iq4_xs_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_iq3_s_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq3_s_wave64",
                entry: "flambeau_mmq_iq3_s_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_iq4_nl_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq4_nl_wave64",
                entry: "flambeau_mmq_iq4_nl_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_iq3_xxs_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq3_xxs_wave64",
                entry: "flambeau_mmq_iq3_xxs_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_iq2_xxs_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq2_xxs_wave64",
                entry: "flambeau_mmq_iq2_xxs_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_iq2_xs_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq2_xs_wave64",
                entry: "flambeau_mmq_iq2_xs_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_iq2_s_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq2_s_wave64",
                entry: "flambeau_mmq_iq2_s_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_iq1_s_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq1_s_wave64",
                entry: "flambeau_mmq_iq1_s_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_iq1_m_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_iq1_m_wave64",
                entry: "flambeau_mmq_iq1_m_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_q4_1_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_1",
                entry: "flambeau_mmvq_q4_1_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q4_1_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_1_r2",
                entry: "flambeau_mmvq_q4_1_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q4_1_mmvq_r2_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_1_r2_dp4a",
                entry: "flambeau_mmvq_q4_1_r2_dp4a_q8_1",
                threads: 256,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q4_1_mmvq_t128_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_1_t128",
                entry: "flambeau_mmvq_q4_1_t128_q8_1",
                threads: 128,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q5_K_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q5_k_r2",
                entry: "flambeau_mmvq_q5_k_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q5_K_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q5_k_dp4a",
                entry: "flambeau_mmvq_q5_k_dp4a_q8_1",
                threads: 256,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q5_K_mmvq_r2_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q5_k_r2_dp4a",
                entry: "flambeau_mmvq_q5_k_r2_dp4a_q8_1",
                threads: 64,
                rows_per_block: 2,
                mmq_tile: (0, 0),
            },
            "qmatmul_q6_K_mmvq_nw1_r4_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q6_k_r4",
                entry: "flambeau_mmvq_q6_k_r4_q8_1",
                threads: 64,
                rows_per_block: 4,
                mmq_tile: (0, 0),
            },
            "qmatmul_q6_K_mmvq_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q6_k_dp4a",
                entry: "flambeau_mmvq_q6_k_dp4a_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q6_K_mmvq_single_row_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q6_k",
                entry: "flambeau_mmvq_q6_k_q8_1",
                threads: 64,
                rows_per_block: 1,
                mmq_tile: (0, 0),
            },
            "qmatmul_q8_0_mmq_oracle_gfx906" => Self {
                kind: RecipeKind::MmqOracle,
                stem: "mmq_q8_0_oracle",
                entry: "flambeau_mmq_q8_0_oracle_q8_1",
                // i fix: the oracle kernel requires 256 threads / 4
                // warps — its cross-warp reduce reads `s_warp[0..4]`. Launching
                // with 64 threads (1 warp) leaves `s_warp[1..4]` uninitialised
                // and the __shfl_xor reduce inside `if (warp == 0)` then sums
                // garbage lanes. The sweep harness already launches at 256
                // (see `sweep_mmq::Dtype::launch_threads`) so the cert didn't
                // catch it.
                threads: 256,
                rows_per_block: 0,
                mmq_tile: (1, 1),
            },
            "qmatmul_q8_0_mmq_4warp_lds_gfx906" => Self {
                kind: RecipeKind::Mmq4Warp,
                stem: "mmq_q8_0_4warp",
                entry: "flambeau_mmq_q8_0_4warp_q8_1",
                threads: 256,
                rows_per_block: 0,
                mmq_tile: (32, 8),
            },
            "qmatmul_q8_0_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q8_0_wave64",
                entry: "flambeau_mmq_q8_0_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                // MMQ_Y=64 rows × TILE_N=8 cols per tile (matches K-quant wave64)
                mmq_tile: (64, 8),
            },
            "qmatmul_q8_0_mmq_wave64_tile32_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q8_0_wave64_tile32",
                entry: "flambeau_mmq_q8_0_wave64_tile32_q8_1",
                threads: 64,
                rows_per_block: 0,
                // 1.d — TILE_N=32 experimental.
                mmq_tile: (64, 32),
            },
            "qmatmul_q8_0_mmq_wave64_tile16_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q8_0_wave64_tile16",
                entry: "flambeau_mmq_q8_0_wave64_tile16_q8_1",
                threads: 64,
                rows_per_block: 0,
                // MMQ_Y=64 rows × TILE_N=16 — halves weight HBM bandwidth vs
                // the TILE_N=8 variant at the cost of doubled per-thread
                // accumulator VGPR (16 floats).
                mmq_tile: (64, 16),
            },
            "qmatmul_q4_1_mmq_4warp_lds_gfx906" => Self {
                kind: RecipeKind::MmqLdsX64,
                stem: "mmq_q4_1_4warp_lds",
                entry: "flambeau_mmq_q4_1_4warp_lds_q8_1",
                threads: 0, // unused for MmqLdsX64 (2D block dims hard-coded in launcher)
                rows_per_block: 0,
                // MMQ_Y=128, MMQ_X=64 — used by the launcher to compute the grid
                // and dynamic LDS bytes. These MUST match mmq_q4_1_4warp_lds.cu.
                mmq_tile: (128, 64),
            },
            // 3.a: wave64 MMQ for Q4_1. Same shape family as Q8_0/K-quant
            // wave64 kernels — MMQ_Y=64, TILE_N=8, 64 threads, DP4A inner.
            "qmatmul_q4_1_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q4_1_wave64",
                entry: "flambeau_mmq_q4_1_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            // 9.e: TILE_N=16 port from mmq_q8_0_wave64_tile16. Each
            // decoded Q4_1 weight tile (8 v[] entries) reused across 16
            // output cols instead of 8 → halves weight HBM bandwidth.
            // Targets the 40.8 % of 9B prefill wall-time the 4warp_lds
            // variant was consuming per 9.a audit.
            "qmatmul_q4_1_mmq_wave64_tile16_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q4_1_wave64_tile16",
                entry: "flambeau_mmq_q4_1_wave64_tile16_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 16),
            },
            // 8.a: wave64 MMQ for Q4_0. Closes the 8.5× prefill gap to
            // llama.cpp on Qwen3.6-35B-A3B-Q4_0 at dense + MoE shapes.
            "qmatmul_q4_0_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q4_0_wave64",
                entry: "flambeau_mmq_q4_0_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            // 4-warp LDS-tiled Q4_0 MMQ — port of the Q4_1
            // 4warp_lds structure with bias-correction in the dot. Closes
            // the 2.5× dense-prefill gap (27B-Q4_0 73 → ~190 tok/s pp4).
            // MUST match mmq_q4_0_4warp_lds.cu's MMQ_Y=128, MMQ_X=64.
            "qmatmul_q4_0_mmq_4warp_lds_gfx906" => Self {
                kind: RecipeKind::MmqLdsX64,
                stem: "mmq_q4_0_4warp_lds",
                entry: "flambeau_mmq_q4_0_4warp_lds_q8_1",
                threads: 0,
                rows_per_block: 0,
                mmq_tile: (128, 64),
            },
            // 0.a: wave64 MMQ for Q5_0. Same tile shape + launch as Q4_0;
            // inner loop adds the 5th-bit `16·bit·y` DP4A term (3's
            // mmvq_q5_0 pattern).
            "qmatmul_q5_0_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q5_0_wave64",
                entry: "flambeau_mmq_q5_0_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_q5_1_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q5_1_wave64",
                entry: "flambeau_mmq_q5_1_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_q4_K_mmq_4warp_lds_gfx906" => Self {
                kind: RecipeKind::Mmq4Warp,
                stem: "mmq_q4_K_4warp",
                entry: "flambeau_mmq_q4_K_4warp_q8_1",
                threads: 128,
                rows_per_block: 0,
                mmq_tile: (16, 8),
            },
            // 4.b kernel, wired (2026-04-27): llamacpp-turbo
            // 4-warp LDS-tiled Q4_K MMQ with DS4 Q8_1 activation. MMQ_Y=128,
            // MMQ_X=16, NWARPS=4 → 256 threads per (64, 4, 1) block. Promoted
            // to default at m≥128 (Q4_K wave64 owns m=32..127 below). Dynamic
            // LDS = 22528 B; weight block = QK_K = 256 elems.
            "qmatmul_q4_K_mmq_turbo_gfx906" => Self {
                kind: RecipeKind::MmqLdsX64,
                stem: "mmq_q4_K_turbo",
                entry: "flambeau_mmq_q4_K_turbo_q8_1",
                threads: 0, // unused for MmqLdsX64
                rows_per_block: 0,
                // MMQ_Y=128, MMQ_X=16 (must match mmq_q4_K_turbo.cu).
                mmq_tile: (128, 16),
            },
            "qmatmul_q4_K_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q4_K_wave64",
                entry: "flambeau_mmq_q4_K_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                // MMQ_Y = 64 rows, TILE_N = 8 cols per tile
                mmq_tile: (64, 8),
            },
            "qmatmul_q5_K_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q5_K_wave64",
                entry: "flambeau_mmq_q5_K_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                // MMQ_Y = 64 rows, TILE_N = 8 cols per tile
                mmq_tile: (64, 8),
            },
            "qmatmul_q6_K_mmq_4warp_lds_gfx906" => Self {
                kind: RecipeKind::Mmq4Warp,
                stem: "mmq_q6_K_4warp",
                entry: "flambeau_mmq_q6_K_4warp_q8_1",
                threads: 128,
                rows_per_block: 0,
                mmq_tile: (16, 8),
            },
            "qmatmul_q6_K_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q6_K_wave64",
                entry: "flambeau_mmq_q6_K_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                // MMQ_Y = 64 rows, TILE_N = 8 cols per tile
                mmq_tile: (64, 8),
            },
            "qmatmul_q8_K_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q8_K_wave64",
                entry: "flambeau_mmq_q8_K_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_q2_K_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q2_K_wave64",
                entry: "flambeau_mmq_q2_K_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            "qmatmul_q3_K_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q3_K_wave64",
                entry: "flambeau_mmq_q3_K_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            other => bail!("no launch recipe registered for impl_id {other}"),
        })
    }
}

fn mmvq_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    recipe: Recipe,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_units: usize,
) -> Result<()> {
    let module = reg.expect_module(recipe.stem)?;
    let kernel = module.kernel(recipe.entry)?;
    let n_rows_i = n_rows as i32;
    let n_units_i = n_units as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let a_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&a_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_units_i);
    let grid = (n_rows as u32).div_ceil(recipe.rows_per_block);
    let cfg = LaunchCfg::one_d(grid, recipe.threads);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

fn mmq_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    recipe: Recipe,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_batches: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module(recipe.stem)?;
    let kernel = module.kernel(recipe.entry)?;
    let n_rows_i = n_rows as i32;
    let n_batches_i = n_batches as i32;
    let n_bpr_i = n_blocks_per_row as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let a_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&a_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_batches_i);
    args.push(&n_bpr_i);
    let (rows_per_tile, batches_per_tile) = recipe.mmq_tile;
    let grid_x = (n_rows as u32).div_ceil(rows_per_tile);
    let grid_y = (n_batches as u32).div_ceil(batches_per_tile);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (recipe.threads, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Launch the 4-warp LDS-tiled MMQ with DS4 Q8_1 activation layout.
/// Expected inputs:
/// weights : flambeau_block_q4_1 * [n_rows, n_blocks_per_row]
/// act_q8_1 : flambeau_block_q8_1_mmq * [n_big_blocks_k, n_batches] (NOTE: MMQ layout)
/// dst : f32 * [n_batches, n_rows] (col-major in our naming)
/// where `n_big_blocks_k = n_blocks_per_row * QK4_1 / (4 * QK8_1) = n_blocks_per_row / 4`.
fn mmq_lds_x64_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    recipe: Recipe,
    weights: DevicePtr,
    act_q8_1_mmq: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_batches: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module(recipe.stem)?;
    let kernel = module.kernel(recipe.entry)?;

    // Kernel expects (ncols_x, nrows_x, ncols_y, stride_col_y, stride_row_x, nrows_dst).
    // ncols_x = K (elements)
    // nrows_x = N (weight rows)
    // ncols_y = M (batch rows)
    // stride_col_y = ncols_y (Y is (big_k, col) row-major in blocks)
    // stride_row_x = n_blocks_per_row (X row stride in weight blocks)
    // nrows_dst = n_rows
    let (block_elems_w, shared_bytes) = mmq_lds_x64_params(recipe.stem)?;
    let ncols_x = (n_blocks_per_row * block_elems_w as usize) as i32;
    let nrows_x = n_rows as i32;
    let ncols_y = n_batches as i32;
    let stride_col_y = n_batches as i32;
    let stride_row_x = n_blocks_per_row as i32;
    let nrows_dst = n_rows as i32;

    let w_ptr: u64 = weights.as_usize() as u64;
    let a_ptr: u64 = act_q8_1_mmq.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&a_ptr);
    args.push(&d_ptr);
    args.push(&ncols_x);
    args.push(&nrows_x);
    args.push(&ncols_y);
    args.push(&stride_col_y);
    args.push(&stride_row_x);
    args.push(&nrows_dst);

    // Dynamic LDS byte budget — sourced from `mmq_lds_x64_params`. Must match
    // each kernel's exact `extern __shared__` budget; see that function's
    // per-stem comment for the derivation.
    let (rows_per_tile, batches_per_tile) = recipe.mmq_tile;
    let grid_x = (n_rows as u32).div_ceil(rows_per_tile);
    let grid_y = (n_batches as u32).div_ceil(batches_per_tile);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        // 2D block: (WARP_SIZE, MMQ_NWARPS, 1) = (64, 4, 1).
        block: (64, 4, 1),
        shared_bytes,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Launch the wave64 Q5_K MMQ (). Expected inputs:
/// weights : flambeau_block_q5_K * [n_rows, K/QK_K]
/// act_q8_1 : flambeau_block_q8_1 * [n_batches, K/QK8_1] (standard layout)
/// dst : f32 * [n_batches, n_rows] (col-major; `dst[col*nrows_dst+row]`)
fn mmq_wave64_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    recipe: Recipe,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_batches: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module(recipe.stem)?;
    let kernel = module.kernel(recipe.entry)?;
    let ncols_x = k as i32;
    let nrows_x = n_rows as i32;
    let ncols_y = n_batches as i32;
    let nrows_y = k as i32; // Y's K dim in elements (blocks = K / QK8_1)
    let nrows_dst = n_rows as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let a_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&a_ptr);
    args.push(&d_ptr);
    args.push(&ncols_x);
    args.push(&nrows_x);
    args.push(&ncols_y);
    args.push(&nrows_y);
    args.push(&nrows_dst);
    let (rows_per_tile, batches_per_tile) = recipe.mmq_tile;
    let grid_x = (n_rows as u32).div_ceil(rows_per_tile);
    let grid_y = (n_batches as u32).div_ceil(batches_per_tile);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (recipe.threads, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

fn block_elems(dtype: QDtype) -> usize {
    use flambeau_quant::{QK8_0, QK_K};
    match dtype {
        QDtype::Q8_0 | QDtype::Q8_1 | QDtype::Q4_0 | QDtype::Q4_1 | QDtype::Q5_0 | QDtype::Q5_1 => {
            QK8_0
        }
        QDtype::Q2_K | QDtype::Q3_K | QDtype::Q4_K | QDtype::Q5_K | QDtype::Q6_K | QDtype::Q8_K => {
            QK_K
        }
        // IQ4_NL is a 32-elem block (like Q4_0); IQ4_XS is a 256-elem
        // super-block (like Q4_K).
        QDtype::IQ4_NL => QK8_0,
        QDtype::IQ4_XS => QK_K,
        // IQ3_XXS and IQ3_S are both 256-elem super-blocks (same family
        // shape as Q3_K / Q4_K).
        QDtype::IQ3_XXS | QDtype::IQ3_S => QK_K,
        // IQ2 + IQ1 family all share the 256-elem super-block shape.
        QDtype::IQ2_XXS | QDtype::IQ2_XS | QDtype::IQ2_S | QDtype::IQ1_S | QDtype::IQ1_M => QK_K,
        // F16 has no native block; return the Q8_1 stride (QK8_1) so the
        // caller's `n_blocks_per_row = k / block_elems` matches the
        // activation side, which is what the F16 MMVQ kernel iterates over.
        QDtype::F16 => 32,
        other => panic!("qmatmul weight dtype {other:?} not supported"),
    }
}
