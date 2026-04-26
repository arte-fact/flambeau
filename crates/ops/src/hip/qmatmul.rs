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
///
/// Takes both the standard [`flambeau_quant::BlockQ8_1`] layout and the DS4
/// [`flambeau_quant::BlockQ8_1Mmq`] layout. The dispatcher picks:
///   - `act_q8_1`      → MMVQ and Mmq4Warp kernels (36 B per-row blocks)
///   - `act_q8_1_mmq`  → MmqLdsX64 kernel (144 B DS4 blocks)
///
/// Decode-path callers that never hit the MmqLdsX64 recipe can pass a null
/// [`DevicePtr`] for `act_q8_1_mmq`; use [`qmatmul_decode`] for ergonomics.
pub fn qmatmul(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    act_q8_1_mmq: DevicePtr,
    dst: DevicePtr,
    m: usize,
    k: usize,
    n: usize,
    dtype_weight: QDtype,
) -> Result<()> {
    // V2.21.b — F16 weight short-circuits the dispatch table. Row-by-row
    // MMVQ for M rows covers both the L=1 warmup prefill (which enters via
    // qmatmul rather than mmvq) and the L>1 prefill path until V2.21.d
    // lands a proper F16 MMQ. Per-row cost is ~30 µs roofline; at L=1024 ×
    // 117 F16 matmuls that's 3.6 s of F16 MMVQ — prefill-slow but
    // correct, matching V2's "make it load first" pattern.
    if dtype_weight == QDtype::F16 {
        let _ = act_q8_1_mmq;
        // V2.29.a — tile-M kernel at m >= 8: each block handles 64 output
        // rows × 8 activation rows (512 outputs/block) with weight HBM
        // read once per K-sub-block per thread instead of per activation
        // row. V2.25 multi-row kept for m < 8 where tile partial-fill
        // hurts grid occupancy.
        if m >= 8 {
            return mmq_f16_tile_launch(reg, stream, weights, act_q8_1, dst, n, m, k);
        }
        return mmq_f16_launch(reg, stream, weights, act_q8_1, dst, n, m, k);
    }
    // V2.28.b — Q4_0 prefill at m >= 32 routes through the wave64 MMQ tile
    // via the dispatch table; decode (m < 32) stays on the V2.23 single-row
    // MMVQ short-circuit.
    //
    // V2.23.a — Q5_0 / Q5_1 still MMVQ row-by-row (no tile kernel yet).
    // V2.30.a — Q5_0 at m >= 32 routes through the wave64 tile via
    // dispatch_qmatmul; m < 32 stays on the MMVQ short-circuit. Q5_1 has no
    // tile kernel yet, still MMVQ row-by-row.
    if dtype_weight == QDtype::Q5_1
        || (dtype_weight == QDtype::Q4_0 && m < 32)
        || (dtype_weight == QDtype::Q5_0 && m < 32)
    {
        let (stem, entry) = match dtype_weight {
            QDtype::Q4_0 => ("mmvq_q4_0", "flambeau_mmvq_q4_0_q8_1"),
            QDtype::Q5_0 => ("mmvq_q5_0", "flambeau_mmvq_q5_0_q8_1"),
            QDtype::Q5_1 => ("mmvq_q5_1", "flambeau_mmvq_q5_1_q8_1"),
            _ => unreachable!(),
        };
        let act_row_bytes = (k / 32) * std::mem::size_of::<flambeau_quant::BlockQ8_1>();
        let dst_row_bytes = n * 4;
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
                k,
            )?;
        }
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
            // V2.34-fix-C: `nb_per_row` (kernel arg) counts WEIGHT
            // superblocks (size = `block_elems(dtype_weight)`); the
            // activation row stride uses Q8_1's QK8_1=32 element blocks.
            // For Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 these coincide (block_elems=32).
            // For Q4_K/Q5_K/Q6_K (block_elems=256=QK_K) they DIVERGE — the
            // pre-fix `act_row_bytes = nb_per_row * sizeof(BlockQ8_1)` was
            // 8× too small, so row i ≥ 1 read partway into row 0's
            // activation. Caused the V2.34 forward_prefill_pp L>1 bug on
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
            mmq_launch(reg, stream, recipe, weights, act_q8_1, dst, n, m, nb_per_row)
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
                reg, stream, recipe, weights, act_q8_1_mmq, dst, n, m, nb_per_row,
            )
        }
        RecipeKind::MmqWave64 => {
            mmq_wave64_launch(
                reg, stream, recipe, weights, act_q8_1, dst, n, m, k,
            )
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
/// **TP-perf-c5** — Q4_0 single-row MMVQ with 128 threads/block (the
/// thin-block schedule that wins for Q4_1 on gfx906 per V2.2.b).
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

/// **TP-perf-c5** — fused gate+up Q4_0 t128. Same gate+up activation
/// sharing as `mmvq_q4_0_gate_up`, but with the t128 schedule for the
/// gfx906 latency-bound regime.
pub fn mmvq_q4_0_gate_up_t128(
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
    let module = reg.expect_module("mmvq_q4_0_gate_up_t128_dp4a")?;
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
    unsafe { kernel.launch(stream, cfg, args)? };
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

/// **C6-i1** — fused gate+up Q4_0 single-warp. Sibling of
/// `mmvq_q4_0_gate_up_t128`, same activation-sharing pattern, single-warp
/// reduce.
pub fn mmvq_q4_0_gate_up_warpcoop64(
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
    let module = reg.expect_module("mmvq_q4_0_gate_up_warpcoop64")?;
    let kernel = module.kernel("flambeau_mmvq_q4_0_gate_up_warpcoop64_q8_1")?;
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
    let cfg = LaunchCfg::one_d(grid, 64);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// **TP-perf-c4** — Q5_K r2 MMVQ writing directly to F16 destination.
/// Mirrors `mmvq_q5_k_r2_q8_1` exactly, except the per-row epilogue
/// casts FP32 → F16 inside the kernel. Used for ssm_out which is the
/// only Q5_K MMVQ in the TP GDN path that immediately casts F32→F16.
/// Block grid is `(n_rows + 1) / 2` (2 rows per block, like the F32 version).
pub fn mmvq_q5_k_r2_f16dst(
    reg: &OpsRegistry,
    stream: &HipStream,
    weights: DevicePtr,
    y_q8_1: DevicePtr,
    dst_f16: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("mmvq_q5_k_r2_f16dst")?;
    let kernel = module.kernel("flambeau_mmvq_q5_k_r2_f16dst_q8_1")?;
    let n_rows_i = n_rows as i32;
    // Q5_K superblock = 256 elems → n_superblocks_per_row = k / 256.
    let n_superblocks_i = (k / 256) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst_f16.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_superblocks_i);
    let grid = ((n_rows + 1) / 2) as u32;
    let cfg = LaunchCfg::one_d(grid, 64);
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
    let module = reg.expect_module("mmvq_q4_0_gate_up_dp4a")?;
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
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// **C8-i1** — fused gate+up Q4_1 dense MMVQ. Sibling of `mmvq_q4_0_gate_up`
/// for Q4_1 weights (Qwen3.5-9B-Q4_1 / 27B-Q4_1 dense FFN). Reads the Q8_1
/// activation once per block, produces both gate and up outputs.
pub fn mmvq_q4_1_gate_up(
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
    let module = reg.expect_module("mmvq_q4_1_gate_up_dp4a")?;
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
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

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
    // C9-followup-2: default to t128_vdr2 schedule (same combined lever
    // that wins +3% on Q8_0 single-row). FLAMBEAU_Q8_0_GU_T128_VDR2=off
    // reverts to the 256t baseline.
    let opt_out = std::env::var("FLAMBEAU_Q8_0_GU_T128_VDR2").as_deref() == Ok("off");
    let (stem, entry, threads) = if opt_out {
        ("mmvq_q8_0_gate_up_dp4a", "flambeau_mmvq_q8_0_gate_up_dp4a_q8_1", 256u32)
    } else {
        (
            "mmvq_q8_0_gate_up_t128_vdr2",
            "flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1",
            128u32,
        )
    };
    let module = reg.expect_module(stem)?;
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
    // V2.21.b — F16 weight × Q8_1 activation bypasses the dispatch table.
    if dtype_weight == QDtype::F16 {
        return mmvq_f16_launch(reg, stream, weights, act_q8_1, dst, n_rows, k);
    }
    // V2.23.a — Q4_0 / Q5_0 bypass the dispatch table. Unblock Qwen3.6-35B
    // -A3B-Q4_0 which uses Q4_0 for attn/FFN and Q5_0 for shared-expert FFN.
    // Single productionised kernel per dtype, no shape-dependent selection.
    // V2.28.d NULL: tried the r2 half-warp-per-row variant (mmvq_q4_0_r2.cu,
    // moved to _unverified/) — it drops the DP4A advantage of the single-row
    // kernel (F32 FMA per lane vs DP4A 4×INT8 per lane). Measured −21 %
    // decode regression AND a logit re-accumulation-order argmax shift on
    // seed 9419. The candle P29 r2 pattern wins on K-quants (sub-block
    // scales block DP4A) but strictly loses on Q4_0's flat-block DP4A path.
    if dtype_weight == QDtype::Q4_0 {
        return mmvq_simple_launch(
            reg, stream, "mmvq_q4_0", "flambeau_mmvq_q4_0_q8_1",
            weights, act_q8_1, dst, n_rows, k,
        );
    }
    if dtype_weight == QDtype::Q5_0 {
        return mmvq_simple_launch(
            reg, stream, "mmvq_q5_0", "flambeau_mmvq_q5_0_q8_1",
            weights, act_q8_1, dst, n_rows, k,
        );
    }
    if dtype_weight == QDtype::Q5_1 {
        return mmvq_simple_launch(
            reg, stream, "mmvq_q5_1", "flambeau_mmvq_q5_1_q8_1",
            weights, act_q8_1, dst, n_rows, k,
        );
    }
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

/// V2.23.a — common launch path for single-block-per-row Q-weight MMVQ
/// kernels that take (w, y_q8_1, dst, n_rows, n_blocks_per_row) and expect
/// 256 threads. Used for Q4_0, Q5_0 and (via `mmvq()` short-circuit) any
/// future single-kernel MMVQ variants.
fn mmvq_simple_launch(
    reg: &OpsRegistry,
    stream: &HipStream,
    module_stem: &'static str,
    kernel_entry: &'static str,
    weights: DevicePtr,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    assert_eq!(k % 32, 0, "{module_stem} requires k % 32 == 0");
    let module = reg.expect_module(module_stem)?;
    let kernel = module.kernel(kernel_entry)?;
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

/// V2.29.a — tile-M F16 MMQ. 64 threads/block = wave64 × 1 row each, MMQ_Y
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
        grid: ((n_rows as u32).div_ceil(64), (n_tokens as u32).div_ceil(8), 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// V2.25.a — multi-row F16 MMQ. Same kernel math as mmvq_f16_launch, but
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

/// V2.21.b — direct launch for the F16-weight × Q8_1-activation MMVQ.
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
    /// V2.2.d.P5 port of candle's 4-warp LDS-tiled MMQ with DS4 Q8_1 activation
    /// layout. Differs from `Mmq4Warp` in: 2D block dims (64, 4, 1), 9 scalar
    /// args (ncols_x, nrows_x, ncols_y, stride_col_y, stride_row_x, nrows_dst),
    /// dynamic shared-memory bytes, and the Y buffer pre-formatted as
    /// `BlockQ8_1Mmq` via `flambeau_quantize_q8_1_mmq`.
    MmqLdsX64,
    /// V2.2.d fix 1 — wave64 MMQ for Q5_K (candle port `mul_mat_q5_K_gfx906_v2`).
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
        let force_q8_r4 = variant.as_deref() == Some("q8_r4");
        let force_q4_1_wave64 = variant.as_deref() == Some("q4_1_wave64");
        let force_q4_1_tile16 = variant.as_deref() == Some("q4_1_tile16");
        let force_q4_k_r4 = variant.as_deref() == Some("q4_k_r4");
        let force_q8_tile32 = variant.as_deref() == Some("q8_tile32");
        // C9-i1 — Q8_0 single-row MMVQ t128 schedule (sibling of mmvq_q4_0_t128).
        // Q8_0 MMVQ eats ~80 % of Qwen3.6-27B-Q8_0 / Coder-30B-Q8_0 decode wall;
        // t128 trades 1 block / CU at occupancy ceiling for 2 blocks / CU
        // (latency-hiding lever). Opt-in via `FLAMBEAU_Q8_0_MMVQ_T128=on`.
        let force_q8_t128 =
            std::env::var("FLAMBEAU_Q8_0_MMVQ_T128").as_deref() == Ok("on");
        // C9-followup — combine t128 occupancy lever with VDR=2 inner loop.
        // After multi-run bench (≥+3 % on both Qwen3.6-27B-Q8_0 and
        // Qwen3.5-27B-Q8_0), this is the new Q8_0 single-row default. Set
        // `FLAMBEAU_Q8_0_MMVQ_T128_VDR2=off` to opt out back to the
        // pre-c9-followup vdr2 schedule.
        let q8_t128_vdr2_setting =
            std::env::var("FLAMBEAU_Q8_0_MMVQ_T128_VDR2").ok();
        let q8_t128_vdr2_opt_out = q8_t128_vdr2_setting.as_deref() == Some("off");
        let q8_t128_vdr2_default_on = !q8_t128_vdr2_opt_out;
        let impl_id = match (impl_id, force_baseline, force_dp4a_only, force_llamacpp) {
            // V2.7: baseline-opt-out reverts tile16 → tile8 for regression A/B.
            // Must precede the general `force_baseline` catch-all below so the
            // MMQ rename happens even when baseline is requested.
            ("qmatmul_q8_0_mmq_wave64_tile16_gfx906", true, _, _) => {
                "qmatmul_q8_0_mmq_wave64_gfx906"
            }
            // V2.13 attempted FLAMBEAU_Q4_1_WAVE64 intercept. NULL RESULT:
            // wave64 single-warp Q4_1 was 3.5× SLOWER than 4warp_lds
            // (1319 → 4647 ms / 528 calls). Q4_1's 32-element block is 8×
            // smaller than K-quants' 256-element super-block, so wave64
            // emits 16× more tile-blocks for the same output area and loses
            // on launch overhead. 4warp_lds's 128×64 tile + activation LDS
            // tiling is structurally correct for finer-grained quants;
            // a proper Q4_1 improvement needs 4warp-level work, not V2.3.b
            // single-warp wave64. Kernel kept in-tree for future reference.
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
            // V2.31.b: A/B toggle for Q8_0 r4 MMVQ.
            ("qmatmul_q8_0_mmvq_single_row_gfx906", _, _, _) if force_q8_r4 => {
                "qmatmul_q8_0_mmvq_r4_dp4a_gfx906"
            }
            // V2.31.f: A/B Q4_1 MMQ variants at 100 W (re-test V2.29.e's
            // null result which was at 200 W envelope).
            ("qmatmul_q4_1_mmq_4warp_lds_gfx906", _, _, _) if force_q4_1_wave64 => {
                "qmatmul_q4_1_mmq_wave64_gfx906"
            }
            ("qmatmul_q4_1_mmq_4warp_lds_gfx906", _, _, _) if force_q4_1_tile16 => {
                "qmatmul_q4_1_mmq_wave64_tile16_gfx906"
            }
            // V2.31.e: A/B Q4_K MMVQ r4 (quarter-wave per row). Targets the
            // Qwen3-Coder-30B dense attention Q/K/V/O projections (non-MoE
            // path) where mmvq_q4_k_r2 is 30 % of decode wall.
            ("qmatmul_q4_K_mmvq_nw1_r2_gfx906", _, _, _) if force_q4_k_r4 => {
                "qmatmul_q4_K_mmvq_nw1_r4_gfx906"
            }
            // V2.31.d: A/B Q8_0 MMQ TILE_N=32 variant. Targets 27B prefill
            // where mmq_q8_0_wave64_tile16 is 87 % of wall.
            ("qmatmul_q8_0_mmq_wave64_tile16_gfx906", _, _, _) if force_q8_tile32 => {
                "qmatmul_q8_0_mmq_wave64_tile32_gfx906"
            }
            // C9 — Q8_0 MMVQ t128 schedule (opt-in, FLAMBEAU_Q8_0_MMVQ_T128=on).
            ("qmatmul_q8_0_mmvq_single_row_gfx906", _, _, _) if force_q8_t128 => {
                "qmatmul_q8_0_mmvq_t128_gfx906"
            }
            ("qmatmul_q8_0_mmvq_dp4a_vdr2_gfx906", _, _, _) if force_q8_t128 => {
                "qmatmul_q8_0_mmvq_t128_gfx906"
            }
            // C9-followup — Q8_0 t128_vdr2 (combine occupancy + ILP).
            // Default-on after multi-run bench. Opt out via
            // FLAMBEAU_Q8_0_MMVQ_T128_VDR2=off to fall through to vdr2.
            ("qmatmul_q8_0_mmvq_single_row_gfx906", _, _, _) if q8_t128_vdr2_default_on => {
                "qmatmul_q8_0_mmvq_t128_vdr2_gfx906"
            }
            ("qmatmul_q8_0_mmvq_dp4a_vdr2_gfx906", _, _, _) if q8_t128_vdr2_default_on => {
                "qmatmul_q8_0_mmvq_t128_vdr2_gfx906"
            }
            // Pre-c9-followup default (vdr2 256t).
            ("qmatmul_q8_0_mmvq_single_row_gfx906", _, _, _) => {
                "qmatmul_q8_0_mmvq_dp4a_vdr2_gfx906"
            }
            // V2.3.d.1: Q6_K DP4A runtime intercept removed — dispatch row
            // `qmatmul_q6_K_mmvq_dp4a_gfx906` now points at the real DP4A
            // kernel directly with its own sweep cert.
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
            "qmatmul_q8_0_mmvq_r4_dp4a_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q8_0_r4_dp4a",
                entry: "flambeau_mmvq_q8_0_r4_dp4a_q8_1",
                threads: 64,
                rows_per_block: 4,
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
            "qmatmul_q4_K_mmvq_nw1_r2_gfx906" => Self {
                kind: RecipeKind::Mmvq,
                stem: "mmvq_q4_k_r2",
                entry: "flambeau_mmvq_q4_k_r2_q8_1",
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
                // V2.31.d — TILE_N=32 experimental.
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
                threads: 0,      // unused for MmqLdsX64 (2D block dims hard-coded in launcher)
                rows_per_block: 0,
                // MMQ_Y=128, MMQ_X=64 — used by the launcher to compute the grid
                // and dynamic LDS bytes. These MUST match mmq_q4_1_4warp_lds.cu.
                mmq_tile: (128, 64),
            },
            // V2.13.a: wave64 MMQ for Q4_1. Same shape family as Q8_0/K-quant
            // wave64 kernels — MMQ_Y=64, TILE_N=8, 64 threads, DP4A inner.
            "qmatmul_q4_1_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q4_1_wave64",
                entry: "flambeau_mmq_q4_1_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            // V2.29.e: TILE_N=16 port from mmq_q8_0_wave64_tile16. Each
            // decoded Q4_1 weight tile (8 v[] entries) reused across 16
            // output cols instead of 8 → halves weight HBM bandwidth.
            // Targets the 40.8 % of 9B prefill wall-time the 4warp_lds
            // variant was consuming per V2.29.a audit.
            "qmatmul_q4_1_mmq_wave64_tile16_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q4_1_wave64_tile16",
                entry: "flambeau_mmq_q4_1_wave64_tile16_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 16),
            },
            // V2.28.a: wave64 MMQ for Q4_0. Closes the 8.5× prefill gap to
            // llama.cpp on Qwen3.6-35B-A3B-Q4_0 at dense + MoE shapes.
            "qmatmul_q4_0_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q4_0_wave64",
                entry: "flambeau_mmq_q4_0_wave64_q8_1",
                threads: 64,
                rows_per_block: 0,
                mmq_tile: (64, 8),
            },
            // V2.30.a: wave64 MMQ for Q5_0. Same tile shape + launch as Q4_0;
            // inner loop adds the 5th-bit `16·bit·y` DP4A term (V2.23's
            // mmvq_q5_0 pattern).
            "qmatmul_q5_0_mmq_wave64_gfx906" => Self {
                kind: RecipeKind::MmqWave64,
                stem: "mmq_q5_0_wave64",
                entry: "flambeau_mmq_q5_0_wave64_q8_1",
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
///   weights  : flambeau_block_q4_1 * [n_rows, n_blocks_per_row]
///   act_q8_1 : flambeau_block_q8_1_mmq * [n_big_blocks_k, n_batches]  (NOTE: MMQ layout)
///   dst      : f32 * [n_batches, n_rows]  (col-major in our naming)
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
    //   ncols_x = K (elements)
    //   nrows_x = N (weight rows)
    //   ncols_y = M (batch rows)
    //   stride_col_y = ncols_y (Y is (big_k, col) row-major in blocks)
    //   stride_row_x = n_blocks_per_row (X row stride in Q4_1 blocks)
    //   nrows_dst    = n_rows
    const QK4_1: usize = 32;
    let ncols_x = (n_blocks_per_row * QK4_1) as i32;
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

    // Dynamic LDS byte budget. Must match mmq_q4_1_4warp_lds.cu exactly.
    // Layout (int-addressed):
    //   tile_y: pad_up(mmq_x * MMQ_TILE_Y_K, MMQ_NWARPS * WARP_SIZE) ints
    //   x_qs:   MMQ_Y * (MMQ_TILE_NE_K + 1) ints
    //   x_dm:   (MMQ_Y * (MMQ_TILE_NE_K / QI4_1) + MMQ_Y / QI4_1) half2
    // With MMQ_Y=128, MMQ_X=64, MMQ_TILE_NE_K=32, QI4_1=4, QI8_1=8,
    //      MMQ_TILE_Y_K = 32 + 32/8 = 36, MMQ_NWARPS=4, WARP_SIZE=64:
    //   tile_y = pad_up(64*36, 256) = 2304 ints
    //   x_qs   = 128 * 33           = 4224 ints
    //   x_dm   = 128*8 + 32         = 1056 half2 = 1056 ints (4 B each)
    // Total ints = 7584  →  30_336 B
    const SHARED_BYTES: u32 = 7584 * 4;

    let (rows_per_tile, batches_per_tile) = recipe.mmq_tile;
    let grid_x = (n_rows as u32).div_ceil(rows_per_tile);
    let grid_y = (n_batches as u32).div_ceil(batches_per_tile);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        // 2D block: (WARP_SIZE, MMQ_NWARPS, 1) = (64, 4, 1).
        block: (64, 4, 1),
        shared_bytes: SHARED_BYTES,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Launch the wave64 Q5_K MMQ (V2.2.d fix 1). Expected inputs:
///   weights  : flambeau_block_q5_K * [n_rows, K/QK_K]
///   act_q8_1 : flambeau_block_q8_1 * [n_batches, K/QK8_1]  (standard layout)
///   dst      : f32 * [n_batches, n_rows]  (col-major; `dst[col*nrows_dst+row]`)
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
    let nrows_y = k as i32;  // Y's K dim in elements (blocks = K / QK8_1)
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
        QDtype::Q8_0 | QDtype::Q8_1 | QDtype::Q4_0 | QDtype::Q4_1 | QDtype::Q5_0 | QDtype::Q5_1 => QK8_0,
        QDtype::Q4_K | QDtype::Q5_K | QDtype::Q6_K => QK_K,
        // V2.21.b — F16 is "1 element per block" in terms of the quant-block
        // unit used for `n_blocks_per_row = k / block_elems`. The F16 MMVQ
        // kernel multiplies F16 weight by Q8_1 activation (QK8_1=32), so the
        // inner loop iterates over Q8_1 blocks — the weight side has no blocks,
        // but `n_blocks_per_row` reflects the Q8_1 activation stride. Return 32
        // to match the caller's `k / 32` computation used for the Q8_1 side.
        QDtype::F16 => 32,
        other => panic!("qmatmul weight dtype {other:?} not supported"),
    }
}
