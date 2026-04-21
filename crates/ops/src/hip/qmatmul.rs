//! Quantised matrix-multiply — MMVQ (decode) + MMQ (prefill) dispatch.
//!
//! This module is the only place where `impl_id` strings get turned into
//! kernel stems + entry names + launch configs. Model code calls
//! [`qmatmul`] (auto-dispatched) or the lower-level [`mmvq`] / [`mmq`]
//! directly when M is statically known.
//!
//! All launches use the committed `dispatch_qmatmul` table in
//! `flambeau-backend-hip`, which mirrors the non-overlapping predicates in
//! `dispatch/hip/gfx906.toml`.

use anyhow::{anyhow, bail, Result};
use flambeau_backend_hip::{dispatch_qmatmul, HipStream, KernelArgs, LaunchCfg};
use flambeau_core::{DevicePtr, QDtype, QMatMulCfg};

use super::OpsRegistry;

/// Auto-dispatched QMatMul. Given M, selects MMVQ (M=1..3 for Q8_0, M=1..127
/// for K-quants) or MMQ (M≥128 for all supported dtypes). For mid-M on
/// K-quants, loops MMVQ across rows — the dispatcher's job is to tell us
/// which impl wins, not to paper over that MMVQ is per-row.
///
/// `act_q8_1` must already be Q8_1-quantised; use [`super::norm::quantize_q8_1`]
/// or the fused `rmsnorm_quant_q8_1` upstream.
pub fn qmatmul(
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
            "no QMatMul impl for dtype={} m={m} k={k} n={n}",
            dtype_weight.name()
        )
    })?;
    let recipe = Recipe::from_impl_id(desc.impl_id)?;

    match recipe.kind {
        RecipeKind::Mmvq => {
            let nb_per_row = k / block_elems(dtype_weight);
            // MMVQ is per-activation-row. Loop M times for M > 1.
            let w_row_bytes = 0; // weight doesn't stride per batch
            let act_row_bytes = nb_per_row * std::mem::size_of::<flambeau_quant::BlockQ8_1>();
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
            mmq_launch(reg, stream, recipe, weights, act_q8_1, dst, n, m, nb_per_row)
        }
    }
}

/// Decode-path MMVQ for a single activation row. Caller is responsible for
/// ensuring `m == 1` in the dispatch sense (one Q8_1-quantised vector of K
/// elements). `n_rows` is the output dim (weight rows).
/// Fused gate+up Q8_0 MMVQ for dense shared-expert FFN. Reads Q8_1
/// activation once, computes both matmuls (same as our MoE Q4_K gate_up
/// pattern but simpler — no min subtraction, no expert indexing).
///
/// Only wired in the DP4A-VDR2 variant; otherwise the caller does two
/// independent `mmvq(...)` calls.
pub fn mmvq_q8_0_gate_up(
    reg: &OpsRegistry,
    stream: &HipStream,
    gate_w: DevicePtr,
    up_w: DevicePtr,
    y_q8_1: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    n_rows_gate: usize,
    n_rows_up: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("mmvq_q8_0_gate_up_dp4a")?;
    let kernel = module.kernel("flambeau_mmvq_q8_0_gate_up_dp4a_q8_1")?;
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
    unsafe { kernel.launch(stream, cfg, args)? };
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
    let cfg = QMatMulCfg {
        dtype_weight,
        dtype_activation: QDtype::Q8_1,
        m: 1,
        k,
        n: n_rows,
    };
    let desc = dispatch_qmatmul(&cfg).ok_or_else(|| {
        anyhow!("no MMVQ impl for dtype={} k={k}", dtype_weight.name())
    })?;
    let recipe = Recipe::from_impl_id(desc.impl_id)?;
    if recipe.kind != RecipeKind::Mmvq {
        bail!("dispatch for m=1 returned non-MMVQ impl {}", desc.impl_id);
    }
    let nb_per_row = k / block_elems(dtype_weight);
    mmvq_launch(reg, stream, recipe, weights, act_q8_1, dst, n_rows, nb_per_row)
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
    mmq_launch(reg, stream, recipe, weights, act_q8_1, dst, n, m, nb_per_row)
}

// ---------------------------------------------------------------------------
// internal: impl_id → launch recipe
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecipeKind {
    Mmvq,
    MmqOracle,
    Mmq4Warp,
}

#[derive(Debug, Clone, Copy)]
struct Recipe {
    kind: RecipeKind,
    stem: &'static str,
    entry: &'static str,
    threads: u32,
    /// MMVQ only — output rows per block.
    rows_per_block: u32,
    /// MMQ only — (rows_per_tile, batches_per_tile).
    mmq_tile: (u32, u32),
}

impl Recipe {
    fn from_impl_id(impl_id: &str) -> Result<Self> {
        // A/B variant override: FLAMBEAU_VARIANT=dp4a swaps MMVQ kernels to
        // their DP4A variants without touching the dispatch table. Lets
        // `ab_decode` compare variants by setting env per-side.
        // V1.7.6.f productisation: DP4A variants are the default. Env var
        // kept as an opt-out / A/B escape hatch — `FLAMBEAU_VARIANT=baseline`
        // reverts to the pre-DP4A scalar kernels, `=dp4a` or `=llamacpp_style`
        // pins older experiments. Default (env unset) picks the fastest
        // measured path: VDR=2 for Q8_0, DP4A for Q6_K, Q4_K MoE DP4A via moe.rs.
        let variant = std::env::var("FLAMBEAU_VARIANT").ok();
        let force_baseline = variant.as_deref() == Some("baseline");
        let force_dp4a_only = variant.as_deref() == Some("dp4a");
        let force_llamacpp = variant.as_deref() == Some("llamacpp_style");
        let impl_id = match (impl_id, force_baseline, force_dp4a_only, force_llamacpp) {
            // opt-out to pre-DP4A scalar
            (_, true, _, _) => impl_id,
            // llamacpp-style (A/B research only)
            ("qmatmul_q8_0_mmvq_single_row_gfx906", _, _, true) => {
                "qmatmul_q8_0_mmvq_llamacpp_style_gfx906"
            }
            // Old dp4a-only variant (VDR=1) for regression bench
            ("qmatmul_q8_0_mmvq_single_row_gfx906", _, true, _) => {
                "qmatmul_q8_0_mmvq_dp4a_gfx906"
            }
            // Default: fastest productised paths.
            ("qmatmul_q8_0_mmvq_single_row_gfx906", _, _, _) => {
                "qmatmul_q8_0_mmvq_dp4a_vdr2_gfx906"
            }
            ("qmatmul_q6_K_mmvq_nw1_r4_gfx906", _, _, _) => {
                "qmatmul_q6_K_mmvq_dp4a_gfx906"
            }
            _ => impl_id,
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
            "qmatmul_q4_K_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_k_r2",
                entry: "flambeau_mmvq_q4_k_r2_q8_1",
                threads: 64,
                rows_per_block: 2,
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
                // V1.7.3-i fix: the oracle kernel requires 256 threads / 4
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
            "qmatmul_q4_K_mmq_4warp_lds_gfx906" => Self {
                kind: RecipeKind::Mmq4Warp,
                stem: "mmq_q4_K_4warp",
                entry: "flambeau_mmq_q4_K_4warp_q8_1",
                threads: 128,
                rows_per_block: 0,
                mmq_tile: (16, 8),
            },
            "qmatmul_q6_K_mmq_4warp_lds_gfx906" => Self {
                kind: RecipeKind::Mmq4Warp,
                stem: "mmq_q6_K_4warp",
                entry: "flambeau_mmq_q6_K_4warp_q8_1",
                threads: 128,
                rows_per_block: 0,
                mmq_tile: (16, 8),
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

fn block_elems(dtype: QDtype) -> usize {
    use flambeau_quant::{QK8_0, QK_K};
    match dtype {
        QDtype::Q8_0 | QDtype::Q8_1 => QK8_0,
        QDtype::Q4_K | QDtype::Q5_K | QDtype::Q6_K => QK_K,
        other => panic!("qmatmul weight dtype {other:?} not supported"),
    }
}
