//! Quantised matrix-multiply — MMVQ (decode) + MMQ (prefill) dispatch.
//! This module is the only place where `impl_id` strings get turned into
//! kernel stems + entry names + launch configs. Model code calls
//! [`qmatmul`] (auto-dispatched) or the lower-level [`mmvq`] / [`mmq`]
//! directly when M is statically known.
//! All launches use the committed `dispatch_qmatmul` table in
//! `flambeau-backend-hip`, which mirrors the non-overlapping predicates in
//! `dispatch/hip/gfx906.toml`.

use anyhow::{anyhow, bail, Result};
use flambeau_core::{LaunchArgs, LaunchCfg};
use flambeau_core::{DevicePtr, QDtype, QMatMulCfg};
use flambeau_quant::q8_1_few::FewScale;


/// Auto-dispatched QMatMul. Given M, selects MMVQ (M=1..3 for Q8_0, M=1..127
/// for K-quants) or MMQ (M≥128 for all supported dtypes). For mid-M on
/// K-quants, loops MMVQ across rows — the dispatcher's job is to tell us
/// which impl wins, not to paper over that MMVQ is per-row.
/// Takes both the standard [`flambeau_quant::BlockQ8_1`] layout and the DS4
/// [`flambeau_quant::BlockQ8_1Mmq`] layout. The dispatcher picks:
/// - `act_q8_1` → MMVQ and Q8_1-layout MMQ kernels (36 B per-row blocks)
/// - `act_q8_1_mmq` → MmqLdsX64 kernel (144 B DS4 blocks)
///   Decode-path callers that never hit the MmqLdsX64 recipe can pass a null
///   [`DevicePtr`] for `act_q8_1_mmq`; use [`qmatmul_decode`] for ergonomics.
pub fn qmatmul<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::QmatmulBuffers,
    shape: crate::MatmulShape,
    dtype_weight: QDtype,
) -> Result<()> {
    let crate::QmatmulBuffers {
        weights,
        act_q8_1,
        act_q8_1_mmq,
        act_q8_1_few,
        dst,
    } = buf;
    let crate::MatmulShape { m, k, n } = shape;
    // F16 short-circuits the dispatch table: there is no F16 MMQ, so all
    // M (including L>1 prefill) routes through per-row MMVQ.
    if dtype_weight == QDtype::F16 {
        let _ = act_q8_1_mmq;
        // Tile-M kernel at m >= 8: each block handles 64 output
        // rows × 8 activation rows (512 outputs/block) with weight HBM
        // read once per K-sub-block per thread instead of per activation
        // row. 5 multi-row kept for m < 8 where tile partial-fill
        // hurts grid occupancy.
        let bufs = crate::MmvqBuffers { weights, act_q8_1, dst };
        if m >= 8 {
            return mmq_f16_tile_launch(reg, stream, bufs, n, m, k);
        }
        return mmq_f16_launch(reg, stream, bufs, n, m, k);
    }
    // Q4_1 at m ∈ {2, 3, 4}: per-N compile-time batched MMVQ.
    // Sibling of the Q4_0 batched path. The first cut of Q4_1 batched
    // (runtime-N=8 loop) hit ~59 VGPRs and lost on every anchor — the
    // per-N template compiles down to ~20-30 VGPRs which preserves
    // gfx906 wave occupancy. Outside this window Q4_1 falls through to
    // the B::dispatch_qmatmul() table (row-by-row m-loop).
    if m >= 2 {
        if let Some(recipe) = mmvq_batched::<B>(reg, dtype_weight, m) {
            let _ = act_q8_1_mmq;
            return mmvq_batched_launch(
                reg,
                stream,
                recipe,
                crate::MmvqBuffers { weights, act_q8_1, dst },
                crate::MmvqBatchShape { n_rows: n, k, n_slots: m },
                block_elems(dtype_weight),
            );
        }
    }
    if dtype_weight == QDtype::Q4_1 && (2..=4).contains(&m) {
        mmvq_q4_1_batched(

            reg, stream,

            crate::MmvqBuffers { weights, act_q8_1, dst },

            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },

        )?;
        let _ = act_q8_1_mmq;
        return Ok(());
    }
    if dtype_weight == QDtype::Q8_0 && (2..=4).contains(&m) {
        mmvq_q8_0_row_tile_batched(

            reg, stream,

            crate::MmvqBuffers { weights, act_q8_1, dst },

            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },

        )?;
        let _ = act_q8_1_mmq;
        return Ok(());
    }
    // Super-block dtypes at m = 2..8: row-tiled batched dp4a MMVQ. The
    // LDS-tiled MMQ pays a fixed ~1.6 decode passes to stage the weight tile;
    // here the weights stay in registers and only the activation strip is
    // staged, so a 5-row verify forward costs ~1.5 decode steps, not ~2.9.
    if rowtile_batched_k_owns(dtype_weight, m, k) {
        mmvq_dp4a_row_tile_batched_k(
            reg,
            stream,
            dtype_weight,
            crate::MmvqBuffers { weights, act_q8_1, dst },
            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },
        )?;
        let _ = act_q8_1_mmq;
        return Ok(());
    }
    // Route to per-row / batched MMVQ for m < 32 (no MMQ amortization) OR
    // whenever k or n is not 128-aligned. The LDS-tiled MMQ this dtype would
    // otherwise dispatch to at m >= 32 produces wrong/NaN output unless both
    // the contraction (k) and the output width (n) are 128-aligned. The
    // gemma4 shared expert hits both edges under TP: its down matmul has
    // k = inter_per_rank (2112/2 = 1056, k % 128 == 32) and its gate/up have
    // n = inter_per_rank = 1056 (n % 128 == 32). MMVQ consumes the standard
    // Q8_1 activation and only needs k % 32.
    if (m < 32 || k % 128 != 0 || n % 128 != 0)
        && matches!(dtype_weight, QDtype::Q4_0 | QDtype::Q5_0 | QDtype::Q5_1)
    {
        // Q4_0 at m ∈ {2, 3, 4}: single-launch batched MMVQ with
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

                reg, stream,

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
                reg, stream,
                KernelEntry { stem, entry },
                crate::MmvqBuffers {
                    weights,
                    act_q8_1: act_q8_1.offset_bytes(i * act_row_bytes),
                    dst: dst.offset_bytes(i * dst_row_bytes),
                },
                n,
                n_blocks,
                MmvqLaunchTune { threads, rows_per_block: 1 },
            )?;
        }
        let _ = act_q8_1_mmq;
        return Ok(());
    }

    let mut recipe = resolve_dispatch_recipe::<B>(dtype_weight, m, k, n)?;
    if recipe.kind == RecipeKind::MmqFew && act_q8_1_few.as_usize() == 0 {
        recipe = few_fallback(recipe)?;
    }
    launch_recipe(
        reg,
        stream,
        recipe,
        crate::QmatmulBuffers {
            weights,
            act_q8_1,
            act_q8_1_mmq,
            act_q8_1_few,
            dst,
        },
        shape,
        dtype_weight,
    )
}

/// Launch `recipe` over the whole `shape`.
fn launch_recipe<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    recipe: Recipe,
    buf: crate::QmatmulBuffers,
    shape: crate::MatmulShape,
    dtype_weight: QDtype,
) -> Result<()> {
    let crate::QmatmulBuffers {
        weights,
        act_q8_1,
        act_q8_1_mmq,
        act_q8_1_few,
        dst,
    } = buf;
    let crate::MatmulShape { m, k, n } = shape;
    match recipe.kind {
        RecipeKind::Mmvq => {
            // `nb_per_row` (kernel arg) counts WEIGHT superblocks (size =
            // `block_elems(dtype_weight)`); the activation row stride uses
            // Q8_1's QK8_1=32 element blocks. For Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
            // these coincide (block_elems=32); for the K-quants
            // (block_elems=256=QK_K) they DIVERGE: a stride of
            // `nb_per_row * sizeof(BlockQ8_1)` is 8× too small and row i ≥ 1
            // reads into row 0's activation.
            let nb_per_row = k / block_elems(dtype_weight);
            let act_blocks_per_row = k / 32; // QK8_1 = 32 elements per Q8_1 block.
            let w_row_bytes = 0; // weight doesn't stride per batch
            let act_row_bytes =
                act_blocks_per_row * std::mem::size_of::<flambeau_quant::BlockQ8_1>();
            let dst_row_bytes = n * 4;
            for i in 0..m {
                mmvq_launch(reg, stream,
                    recipe,
                    crate::MmvqBuffers {
                        weights: weights.offset_bytes(i * w_row_bytes),
                        act_q8_1: act_q8_1.offset_bytes(i * act_row_bytes),
                        dst: dst.offset_bytes(i * dst_row_bytes),
                    },
                    n,
                    nb_per_row,
                )?;
            }
            Ok(())
        }
        RecipeKind::MmqOracle => {
            let nb_per_row = k / block_elems(dtype_weight);
            mmq_launch(
                reg, stream,
                recipe,
                crate::MmvqBuffers { weights, act_q8_1, dst },
                n,
                m,
                nb_per_row,
            )
        }
        RecipeKind::MmqLdsX64 => {
            if act_q8_1_mmq.as_usize() == 0 {
                bail!(
                    "qmatmul dispatched to MmqLdsX64 (stem {}) but caller \
                     did not populate act_q8_1_mmq — run \
                     ops::norm::quantize_f16_q8_1_mmq upstream or use \
                     qmatmul_decode() if this is a decode path",
                    recipe.stem
                );
            }
            let nb_per_row = k / block_elems(dtype_weight);
            mmq_lds_x64_launch(
                reg, stream,
                recipe,
                crate::MmvqBuffers { weights, act_q8_1: act_q8_1_mmq, dst },
                n,
                m,
                nb_per_row,
            )
        }
        RecipeKind::MmqMma => {
            if act_q8_1_mmq.as_usize() == 0 {
                bail!(
                    "qmatmul dispatched to MmqMma (stem {}) but caller did not \
                     populate act_q8_1_mmq — run ops::norm::quantize_f16_q8_1_mmq \
                     upstream",
                    recipe.stem
                );
            }
            let nb_per_row = k / block_elems(dtype_weight);
            mmq_mma_launch(
                reg, stream,
                recipe,
                crate::MmvqBuffers { weights, act_q8_1: act_q8_1_mmq, dst },
                n,
                m,
                nb_per_row,
            )
        }
        RecipeKind::MmvqBatched => mmvq_batched_launch(
            reg,
            stream,
            recipe,
            crate::MmvqBuffers { weights, act_q8_1, dst },
            crate::MmvqBatchShape { n_rows: n, k, n_slots: m },
            block_elems(dtype_weight),
        ),
        RecipeKind::MmqFew => mmq_few_launch(
            reg,
            stream,
            recipe,
            crate::MmvqBuffers { weights, act_q8_1: act_q8_1_few, dst },
            shape,
            dtype_weight,
        ),
        RecipeKind::MmqWave64 => {
            mmq_wave64_launch(
                reg, stream,
                recipe,
                crate::MmvqBuffers { weights, act_q8_1, dst },
                n,
                m,
                k,
            )
        }
    }
}

/// Resolve the dispatch-table recipe for `(dtype, m, k, n)`, applying the
/// `k % 128 != 0` re-dispatch that swaps a DS4-layout MMQ recipe for the
/// dtype's MMVQ (which consumes the standard Q8_1 activation and only needs
/// `k % 32`). Shared by [`qmatmul`] (to launch) and
/// [`qmatmul_act_layout`] (to classify) so the two can never
/// disagree on which recipe a shape resolves to.
fn resolve_dispatch_recipe<B: flambeau_core::KernelBackend>(
    dtype_weight: QDtype,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Recipe> {
    let cfg = QMatMulCfg {
        dtype_weight,
        dtype_activation: QDtype::Q8_1,
        m,
        k,
        n,
    };
    let desc = B::dispatch_qmatmul(&cfg).ok_or_else(|| {
        anyhow!(
            "no QMatMul impl for dtype={} m={m} k={k} n={n}",
            dtype_weight.name()
        )
    })?;
    let mut recipe = Recipe::from_impl_id(desc.impl_id)?;
    // MmqLdsX64 / MmqMma (DS4 144-B/block activation) require k % 128 == 0.
    // The m-range dispatch table doesn't encode that, so a non-128-aligned
    // contraction dim (e.g. a shared-expert intermediate) can resolve here.
    // Re-dispatch as if m=1 to obtain the dtype's MMVQ impl, which consumes
    // the standard Q8_1 activation; the Mmvq arm then loops it over the real
    // m rows.
    if matches!(recipe.kind, RecipeKind::MmqLdsX64 | RecipeKind::MmqMma) && k % 128 != 0 {
        let mmvq_cfg = QMatMulCfg {
            dtype_weight,
            dtype_activation: QDtype::Q8_1,
            m: 1,
            k,
            n,
        };
        let mmvq_desc = B::dispatch_qmatmul(&mmvq_cfg).ok_or_else(|| {
            anyhow!(
                "no MMVQ fallback impl for dtype={} k={k} n={n} (k % 128 != 0)",
                dtype_weight.name()
            )
        })?;
        recipe = Recipe::from_impl_id(mmvq_desc.impl_id)?;
    }
    if recipe.kind == RecipeKind::MmqFew && !mmq_few_serves(m, k) {
        recipe = few_fallback(recipe)?;
    }
    Ok(recipe)
}

/// Activation layout a QMatMul impl reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActLayout {
    /// Standard 36-B [`flambeau_quant::BlockQ8_1`] rows (`act_q8_1`).
    Q8_1,
    /// DS4 144-B [`flambeau_quant::BlockQ8_1Mmq`] blocks (`act_q8_1_mmq`).
    Mmq,
    /// Int8 MMA B-fragment layout of the few-column MMQ
    /// ([`flambeau_quant::q8_1_few`]), with its scale record.
    Few(flambeau_quant::q8_1_few::FewScale),
}

/// The layouts read by the projections that share one activation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ActLayouts {
    q8_1: bool,
    mmq: bool,
    few_d4: bool,
    few_ds4: bool,
}

impl ActLayouts {
    /// Whether some projection of the set reads `layout`.
    pub fn reads(self, layout: ActLayout) -> bool {
        match layout {
            ActLayout::Q8_1 => self.q8_1,
            ActLayout::Mmq => self.mmq,
            ActLayout::Few(FewScale::D4) => self.few_d4,
            ActLayout::Few(FewScale::Ds4) => self.few_ds4,
        }
    }

    /// Whether some projection of the set reads the few-column layout.
    pub fn reads_few(self) -> bool {
        self.few_d4 || self.few_ds4
    }

    /// The scale record of the few-column activation the set reads, if any.
    /// One activation buffer holds one record: a set reading both is refused.
    pub fn few_scale(self) -> Result<Option<FewScale>> {
        match (self.few_d4, self.few_ds4) {
            (false, false) => Ok(None),
            (true, false) => Ok(Some(FewScale::D4)),
            (false, true) => Ok(Some(FewScale::Ds4)),
            (true, true) => bail!("one activation read in both few-column scale records"),
        }
    }
}

impl From<ActLayout> for ActLayouts {
    fn from(layout: ActLayout) -> Self {
        std::iter::once(layout).collect()
    }
}

impl FromIterator<ActLayout> for ActLayouts {
    fn from_iter<I: IntoIterator<Item = ActLayout>>(iter: I) -> Self {
        iter.into_iter().fold(Self::default(), |mut set, layout| {
            match layout {
                ActLayout::Q8_1 => set.q8_1 = true,
                ActLayout::Mmq => set.mmq = true,
                ActLayout::Few(FewScale::D4) => set.few_d4 = true,
                ActLayout::Few(FewScale::Ds4) => set.few_ds4 = true,
            }
            set
        })
    }
}

/// The activation layout the auto-dispatched [`qmatmul`] reads for
/// `(dtype_weight, shape)`. Callers that pre-quantise activations produce
/// only this layout.
///
/// MUST track [`qmatmul`]'s routing: every early-return short-circuit there
/// consumes the standard layout — only the terminal `MmqLdsX64` / `MmqMma`
/// recipe arms read DS4. The dispatch-table resolution is shared verbatim via
/// [`resolve_dispatch_recipe`], so the only mirrored logic is the pre-dispatch
/// short-circuit guards below; keep them in lockstep with `qmatmul`.
pub fn qmatmul_act_layout<B: flambeau_core::KernelBackend>(
    shape: crate::MatmulShape,
    dtype_weight: QDtype,
) -> ActLayout {
    let crate::MatmulShape { m, k, n } = shape;
    // F16 has no MMQ — all M route through per-row MMVQ (standard layout).
    if dtype_weight == QDtype::F16 {
        return ActLayout::Q8_1;
    }
    // Compile-time batched MMVQ short-circuits (standard layout).
    if rowtile_batched_k_owns(dtype_weight, m, k) {
        return ActLayout::Q8_1;
    }
    if matches!(dtype_weight, QDtype::Q4_1 | QDtype::Q8_0) && (2..=4).contains(&m) {
        return ActLayout::Q8_1;
    }
    // Q4_0/Q5_0/Q5_1 route to per-row / batched MMVQ with no MMQ amortization
    // (m < 32) or non-128-aligned dims (standard layout).
    if (m < 32 || k % 128 != 0 || n % 128 != 0)
        && matches!(dtype_weight, QDtype::Q4_0 | QDtype::Q5_0 | QDtype::Q5_1)
    {
        return ActLayout::Q8_1;
    }
    match resolve_dispatch_recipe::<B>(dtype_weight, m, k, n) {
        Ok(recipe) => recipe.kind.act_layout(dtype_weight),
        Err(_) => ActLayout::Q8_1,
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
pub fn mmvq_q4_0_t128<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    weights: DevicePtr,
    y_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("mmvq_q4_0_t128")?;
    let kernel = B::kernel(module, "flambeau_mmvq_q4_0_t128_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 128);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_0 batched MMVQ: one launch covers `n_slots` activation
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
pub fn mmvq_q4_0_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q4_0_batched")?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Shapes the row-tiled batched dp4a MMVQ serves ahead of the dispatch table.
fn rowtile_batched_k_owns(dtype_weight: QDtype, m: usize, k: usize) -> bool {
    matches!(
        dtype_weight,
        QDtype::Q6_K
            | QDtype::IQ4_XS
            | QDtype::IQ3_S
            | QDtype::IQ3_XXS
            | QDtype::IQ2_S
    ) && (2..=8).contains(&m)
        && k % flambeau_quant::QK_K == 0
}

/// Row-tiled batched dp4a MMVQ for the 256-element super-block dtypes still
/// served ahead of the dispatch table (Q6_K, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S),
/// `n_slots` ∈ [2, 8]: 8 weight rows per block share one
/// LDS-staged activation strip and each weight int is decoded once for every
/// column. Standard Q8_1 activation, slot-major `dst`.
pub fn mmvq_dp4a_row_tile_batched_k<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    dtype: QDtype,
    buffers: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: y_q8_1, dst } = buffers;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    if !(2..=8).contains(&n_slots) {
        bail!("mmvq_dp4a_row_tile_batched_k: n_slots={n_slots} outside [2, 8]");
    }
    let (stem, entries): (_, [&'static str; 7]) = match dtype {
        QDtype::IQ4_XS => (
            "mmvq_iq4_xs_dp4a_batched",
            [
                "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n2",
                "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n3",
                "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n4",
                "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n5",
                "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n6",
                "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n7",
                "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n8",
            ],
        ),
        QDtype::IQ3_S => (
            "mmvq_iq3_s_dp4a_batched",
            [
                "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n2",
                "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n3",
                "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n4",
                "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n5",
                "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n6",
                "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n7",
                "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n8",
            ],
        ),
        QDtype::IQ3_XXS => (
            "mmvq_iq3_xxs_dp4a_batched",
            [
                "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n2",
                "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n3",
                "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n4",
                "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n5",
                "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n6",
                "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n7",
                "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n8",
            ],
        ),
        QDtype::IQ2_S => (
            "mmvq_iq2_s_dp4a_batched",
            [
                "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n2",
                "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n3",
                "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n4",
                "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n5",
                "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n6",
                "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n7",
                "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n8",
            ],
        ),
        QDtype::Q6_K => (
            "mmvq_q6_k_dp4a_batched",
            [
                "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n2",
                "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n3",
                "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n4",
                "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n5",
                "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n6",
                "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n7",
                "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n8",
            ],
        ),
        other => bail!("mmvq_dp4a_row_tile_batched_k: no kernel for {other:?}"),
    };
    let module = reg.expect_module(stem)?;
    let kernel = B::kernel(module, entries[n_slots - 2])?;
    let n_rows_i = n_rows as i32;
    let n_superblocks_i = (k / flambeau_quant::QK_K) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_superblocks_i);
    let cfg = LaunchCfg::one_d(n_rows.div_ceil(8) as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q8_0 sibling of [`mmvq_q4_0_batched`]. Same per-N compile-time
/// template; symmetric signed-8-bit drops the nibble unpack and the
/// `-8·s_y` correction.
pub fn mmvq_q8_0_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q8_0_batched")?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_1 sibling of [`mmvq_q4_0_batched`]. Same per-N compile-time
/// template pattern (n2/n3/n4 entries) — Q4_1 carries an explicit
/// `min` per block, so the per-block bias is `+ m_x · s_y` rather than
/// Q4_0's `- 8 · d_x · s_y`. Targets the same dispatch window
/// (`n_slots` ∈ [2, 4]) and the same weight-amortization regime.
pub fn mmvq_q4_1_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q4_1_batched")?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q5_K batched MMVQ with r2 multi-row + per-N activation
/// columns. Sibling of [`mmvq_q4_0_batched`] for Q5_K weights (the
/// `ssm_*` projections in GDN layers — second-largest decode bucket
/// per S2 rocprof at 760 ms / spec phase). Same lever: one launch
/// covers N activation rows; each block reads its Q5_K super-block
/// once and applies it across the N columns. Block grid:
/// `ceil(n_rows / 2)` (r2 retained from the single-col Q5_K kernel).
///
/// `n_slots` ∈ [2, 4]. Outside that the caller falls back to the
/// row-by-row m-loop in [`qmatmul`].
pub fn mmvq_q5_k_r2_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q5_k_r2_batched")?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_i = n_rows as i32;
    let n_superblocks_i = (k / 256) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_superblocks_i);
    // 64 threads/block = 1 wave64 (r2 multi-row pattern); grid = ceil(n_rows / 2).
    let n_row_pairs = n_rows.div_ceil(2) as u32;
    let cfg = LaunchCfg::one_d(n_row_pairs, 64);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Fused gate+up Q4_0 t128: gate and up share one activation read, with
/// the t128 schedule for the gfx906 latency-bound regime.
pub fn mmvq_q4_0_gate_up_t128<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::MmvqGateUpBuffers,
    shape: crate::MmvqGateUpShape,
) -> Result<()> {
    let crate::MmvqGateUpBuffers { gate_w, up_w, act_q8_1: y_q8_1, gate_out, up_out } = buffers;
    let crate::MmvqGateUpShape { n_rows_gate, n_rows_up, k } = shape;
    let module = reg.expect_module("mmvq_q4_0_gate_up_t128_dp4a")?;
    let kernel = B::kernel(module, "flambeau_mmvq_q4_0_gate_up_t128_dp4a_q8_1")?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&gw_ptr);
    args.push_arg(&uw_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&n_rows_g);
    args.push_arg(&n_rows_u);
    args.push_arg(&n_blocks_i);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, 128);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_0 single-warp (64 t/block) MMVQ. Sibling of `mmvq_q4_0_t128`
/// targeting an even smaller block to pack more in-flight blocks per CU at
/// gfx906 decode (where occupancy is the lever, not VALU). One wave64/block
/// = no LDS reduce, just an in-place gfx906 DPP butterfly.
pub fn mmvq_q4_0_warpcoop64<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    weights: DevicePtr,
    y_q8_1: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("mmvq_q4_0_warpcoop64")?;
    let kernel = B::kernel(module, "flambeau_mmvq_q4_0_warpcoop64_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 64);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// **TP-perf-c3** — fused K+V Q4_0 dense MMVQ with F16 destination.
/// K and V always share shape (`[local_n_kv_heads · head_dim, hidden]`)
/// in TP full-attn layers and read the same Q8_1 activation. This
/// kernel collapses both MMVQs and both downstream `cast_f32_to_f16`
/// launches into a single launch — net 3 launches saved per full-attn
/// layer per rank.
pub fn mmvq_q4_0_kv_f16dst<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MmvqKvF16Buffers,
    n_rows_kv: usize,
    k: usize,
) -> Result<()> {
    let crate::MmvqKvF16Buffers {
        k_w,
        v_w,
        y_q8_1,
        k_out_f16,
        v_out_f16,
    } = buf;
    let module = reg.expect_module("mmvq_q4_0_kv_f16dst_dp4a")?;
    let kernel = B::kernel(module, "flambeau_mmvq_q4_0_kv_f16dst_dp4a_q8_1")?;
    let n_rows_i = n_rows_kv as i32;
    let n_blocks_i = (k / 32) as i32;
    let kw_ptr: u64 = k_w.as_usize() as u64;
    let vw_ptr: u64 = v_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let kout_ptr: u64 = k_out_f16.as_usize() as u64;
    let vout_ptr: u64 = v_out_f16.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&kw_ptr);
    args.push_arg(&vw_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&kout_ptr);
    args.push_arg(&vout_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows_kv as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_0 fused gate+up MMVQ batched across N activation cols.
/// Sibling of [`mmvq_q4_0_gate_up_t128`] (single-col fused) and
/// [`mmvq_q4_0_batched`] (multi-col single-weight). Combines both
/// levers: each block reads ONE gate weight row + ONE up weight row +
/// ONE shared activation strip per col, computing 2 × N output values
/// per (col, row) pair.
///
/// Output layout: `gate_out[N, n_rows_gate]`, `up_out[N, n_rows_up]`,
/// slot-major F32. Asymmetric-row support preserved (n_rows_gate vs
/// n_rows_up may differ; per-row do_gate/do_up short-circuit).
///
/// `n_slots` ∈ [2, 4]. Used by the batched-GDN paired-L=2 forward.
pub fn mmvq_q4_0_gate_up_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q4_0_gate_up_batched")?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&gw_ptr);
    args.push_arg(&uw_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&n_rows_g);
    args.push_arg(&n_rows_u);
    args.push_arg(&n_blocks_i);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Row-tiled sibling of [`mmvq_q4_0_gate_up_batched`]. Each block owns
/// `R = 4` consecutive rows of the gate+up matmul and shares one
/// LDS-resident Q8_1 activation strip across the N decode slots; cuts
/// activation HBM traffic ~4× vs the per-row kernel at the same
/// output count.
///
/// `n_slots` ∈ [2, 4]. Output ABI identical to `mmvq_q4_0_gate_up_batched`
/// (`gate_out[N, n_rows_gate]`, `up_out[N, n_rows_up]`, slot-major F32).
pub fn mmvq_q4_0_gate_up_row_tile_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q4_0_gate_up_row_tile_batched")?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&gw_ptr);
    args.push_arg(&uw_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&n_rows_g);
    args.push_arg(&n_rows_u);
    args.push_arg(&n_blocks_i);
    let max_rows = n_rows_gate.max(n_rows_up);
    let grid = (max_rows as u32).div_ceil(4);
    let cfg = LaunchCfg::one_d(grid, 256);
    B::launch(&kernel, stream, cfg, args)?;
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
pub fn mmvq_q4_0_row_tile_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q4_0_row_tile_batched")?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let grid = (n_rows as u32).div_ceil(4);
    let cfg = LaunchCfg::one_d(grid, 256);
    B::launch(&kernel, stream, cfg, args)?;
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
pub fn mmvq_q8_0_row_tile_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q8_0_row_tile_batched")?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let grid = (n_rows as u32).div_ceil(4);
    let cfg = LaunchCfg::one_d(grid, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

pub fn mmvq_q8_0_gate_up<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module(stem)?;
    let kernel = B::kernel(module, entry)?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_blocks_i = (k / 32) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&gw_ptr);
    args.push_arg(&uw_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&n_rows_g);
    args.push_arg(&n_rows_u);
    args.push_arg(&n_blocks_i);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, threads);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Fused gate+up Q5_K dense MMVQ. Like [`mmvq_q8_0_gate_up`], reads
/// the Q8_1 activation once per inner-loop block and computes both dot
/// products. Used by the qwen3.5/3.6 shared-expert FFN path when
/// `ffn_gate_shexp` and `ffn_up_shexp` are both Q5_K.
pub fn mmvq_q5_k_gate_up<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
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
    let module = reg.expect_module("mmvq_q5_k_gate_up_dp4a")?;
    let kernel = B::kernel(module, "flambeau_mmvq_q5_k_gate_up_dp4a_q8_1")?;
    let n_rows_g = n_rows_gate as i32;
    let n_rows_u = n_rows_up as i32;
    let n_sb_per_row = (k / flambeau_quant::QK_K) as i32;
    let gw_ptr: u64 = gate_w.as_usize() as u64;
    let uw_ptr: u64 = up_w.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let u_ptr: u64 = up_out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&gw_ptr);
    args.push_arg(&uw_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&g_ptr);
    args.push_arg(&u_ptr);
    args.push_arg(&n_rows_g);
    args.push_arg(&n_rows_u);
    args.push_arg(&n_sb_per_row);
    let grid = n_rows_gate.max(n_rows_up) as u32;
    let cfg = LaunchCfg::one_d(grid, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

pub fn mmvq<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MmvqBuffers,
    shape: crate::MmvqShape,
    dtype_weight: QDtype,
) -> Result<()> {
    let crate::MmvqBuffers {
        weights,
        act_q8_1,
        dst,
    } = buf;
    let crate::MmvqShape { n_rows, k } = shape;
    if dtype_weight == QDtype::F16 {
        return mmvq_f16_launch(
            reg, stream,
            crate::MmvqBuffers { weights, act_q8_1, dst },
            n_rows,
            k,
        );
    }
    assert_eq!(k % 32, 0, "MMVQ requires k % 32 == 0");
    let recipe = mmvq_m1::<B>(reg, dtype_weight)?.recipe;
    mmvq_launch(
        reg, stream,
        recipe,
        crate::MmvqBuffers { weights, act_q8_1, dst },
        n_rows,
        k / block_elems(dtype_weight),
    )
}

/// The m = 1 MMVQ kernel of one weight dtype, as cached in
/// [`crate::Registry`]: [`mmvq`] launches `recipe.entry`,
/// [`mmvq_f16_direct`] `recipe.entry_f16`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MmvqM1 {
    recipe: Recipe,
    /// `recipe.entry_f16` is set and the loaded module contains it.
    f16: bool,
}

/// The m = 1 kernel of `dtype_weight`, resolved on the registry's first use
/// of that dtype and read from its cache afterwards. A dtype that does not
/// resolve is cached as `None`; asking for it again re-runs the resolver to
/// report the error (a launch that fails anyway).
fn mmvq_m1<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    dtype_weight: QDtype,
) -> Result<MmvqM1> {
    let cached = *reg
        .mmvq_m1
        .get_or_init(dtype_weight, || resolve_mmvq_m1::<B>(reg, dtype_weight).ok());
    match cached {
        Some(m1) => Ok(m1),
        None => resolve_mmvq_m1::<B>(reg, dtype_weight),
    }
}

fn resolve_mmvq_m1<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    dtype_weight: QDtype,
) -> Result<MmvqM1> {
    let recipe = mmvq_m1_recipe::<B>(dtype_weight)?;
    let f16 = recipe.entry_f16.is_some_and(|entry_f16| {
        reg.module(recipe.stem)
            .is_some_and(|module| B::kernel(module, entry_f16).is_ok())
    });
    Ok(MmvqM1 { recipe, f16 })
}

/// The single-row (m = 1) MMVQ recipe for `dtype_weight`: the m = 1 dispatch
/// row's recipe, or the fixed kernel of the dtypes that bypass the dispatch
/// table (Q4_0 / Q5_0 / Q5_1: one productionised kernel each, no
/// shape-dependent choice). The m = 1 dispatch depends on the dtype only,
/// which is what makes [`mmvq_m1`]'s per-dtype cache exact.
fn mmvq_m1_recipe<B: flambeau_core::KernelBackend>(dtype_weight: QDtype) -> Result<Recipe> {
    let fixed = |stem, entry, entry_f16| Recipe {
        kind: RecipeKind::Mmvq,
        stem,
        entry,
        entry_f16: Some(entry_f16),
        threads: 256,
        rows_per_block: 1,
        mmq_tile: (0, 0),
    };
    match dtype_weight {
        QDtype::Q4_0 => {
            return Ok(fixed("mmvq_q4_0", "flambeau_mmvq_q4_0_q8_1", "flambeau_mmvq_q4_0_q8_1_f16"))
        }
        QDtype::Q5_0 => {
            return Ok(fixed("mmvq_q5_0", "flambeau_mmvq_q5_0_q8_1", "flambeau_mmvq_q5_0_q8_1_f16"))
        }
        QDtype::Q5_1 => {
            return Ok(fixed("mmvq_q5_1", "flambeau_mmvq_q5_1_q8_1", "flambeau_mmvq_q5_1_q8_1_f16"))
        }
        _ => {}
    }
    let cfg = QMatMulCfg {
        dtype_weight,
        dtype_activation: QDtype::Q8_1,
        m: 1,
        k: 0,
        n: 0,
    };
    let desc = B::dispatch_qmatmul(&cfg)
        .ok_or_else(|| anyhow!("no MMVQ impl for dtype={}", dtype_weight.name()))?;
    let recipe = Recipe::from_impl_id(desc.impl_id)?;
    if recipe.kind != RecipeKind::Mmvq {
        bail!("dispatch for m=1 returned non-MMVQ impl {}", desc.impl_id);
    }
    Ok(recipe)
}

/// True when the m = 1 kernel of `dtype_weight` (see [`mmvq_m1_recipe`]) has
/// an F16-output entry and the registry's module contains it. Read from the
/// registry's per-dtype cache after the first call.
/// Most activation columns one batched MMVQ launch covers (its `_n8` entry).
const MMVQ_BATCH_MAX: usize = 8;

/// (threads, rows per block) of a batched-MMVQ impl's entries — one warp, the
/// rows its cell was instantiated with — or `None` if `impl_id` is not one.
pub fn mmvq_batched_shape(impl_id: &str) -> Option<(u32, u32)> {
    Recipe::from_impl_id(impl_id)
        .ok()
        .filter(|recipe| recipe.kind == RecipeKind::MmvqBatched)
        .map(|recipe| (recipe.threads, recipe.rows_per_block))
}

/// The `<base>_n1` .. `<base>_n8` entry names of a batched MMVQ cell; with
/// `from 2`, a cell instantiated for 2..8 columns only.
macro_rules! batched_entries {
    ($base:literal) => {
        [
            Some(concat!($base, "_n1")),
            Some(concat!($base, "_n2")),
            Some(concat!($base, "_n3")),
            Some(concat!($base, "_n4")),
            Some(concat!($base, "_n5")),
            Some(concat!($base, "_n6")),
            Some(concat!($base, "_n7")),
            Some(concat!($base, "_n8")),
        ]
    };
    ($base:literal, from 2) => {
        [
            None,
            Some(concat!($base, "_n2")),
            Some(concat!($base, "_n3")),
            Some(concat!($base, "_n4")),
            Some(concat!($base, "_n5")),
            Some(concat!($base, "_n6")),
            Some(concat!($base, "_n7")),
            Some(concat!($base, "_n8")),
        ]
    };
}

/// Per batched-MMVQ stem, its entry for 1..=`MMVQ_BATCH_MAX` columns (`None`
/// where the cell has none). Each name is `<recipe.entry>_n<cols>` (checked by
/// `recipe_table_tests`).
const MMVQ_BATCHED_ENTRIES: &[(&str, [Option<&str>; MMVQ_BATCH_MAX])] = &[
    ("mmvq_q8_0_dp4a_vdr2_batched", batched_entries!("flambeau_mmvq_q8_0_dp4a_vdr2_batched_q8_1")),
    ("mmvq_iq4_nl_dp4a_t128_vdr2_batched", batched_entries!("flambeau_mmvq_iq4_nl_dp4a_t128_vdr2_batched_q8_1")),
    ("mmvq_iq3_s_dp4a_t128_vdr2_batched", batched_entries!("flambeau_mmvq_iq3_s_dp4a_t128_vdr2_batched_q8_1")),
    ("mmvq_q3_k_dp4a_t128_vdr2_batched", batched_entries!("flambeau_mmvq_q3_k_dp4a_t128_vdr2_batched_q8_1")),
    ("mmvq_q6_k_dp4a_vdr2_batched", batched_entries!("flambeau_mmvq_q6_k_dp4a_vdr2_batched_q8_1")),
    ("mmvq_q4_k_dp4a_t128_vdr2_batched", batched_entries!("flambeau_mmvq_q4_k_dp4a_t128_vdr2_batched_q8_1")),
    ("mmvq_q5_k_dp4a_t128_vdr2_batched", batched_entries!("flambeau_mmvq_q5_k_dp4a_t128_vdr2_batched_q8_1")),
    ("mmvq_iq4_xs_dp4a_t128_vdr4_batched", batched_entries!("flambeau_mmvq_iq4_xs_dp4a_t128_vdr4_batched_q8_1")),
    ("mmvq_q4_k_dp4a_batched", batched_entries!("flambeau_mmvq_q4_k_dp4a_q8_1_batched", from 2)),
    ("mmvq_q5_k_dp4a_batched", batched_entries!("flambeau_mmvq_q5_k_dp4a_q8_1_batched", from 2)),
];

/// The dispatched batched-MMVQ recipe of `dtype_weight` at `m` columns, or
/// `None` when the arch's dispatch row at that `m` is not batched (then the
/// caller keeps its other paths). Resolved once per (dtype, m) up to
/// `MMVQ_BATCH_MAX`, beyond which the answer at `MMVQ_BATCH_MAX` is reused.
fn mmvq_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    dtype_weight: QDtype,
    m: usize,
) -> Option<Recipe> {
    let m = m.min(MMVQ_BATCH_MAX);
    reg.mmvq_batched[m - 2]
        .get_or_init(dtype_weight, || {
            let cfg = QMatMulCfg { dtype_weight, dtype_activation: QDtype::Q8_1, m, k: 0, n: 0 };
            B::dispatch_qmatmul(&cfg)
                .and_then(|desc| Recipe::from_impl_id(desc.impl_id).ok())
                .filter(|recipe| recipe.kind == RecipeKind::MmvqBatched)
                .map(|recipe| MmvqBatchedRecipe { recipe })
        })
        .map(|cached| cached.recipe)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MmvqBatchedRecipe {
    recipe: Recipe,
}

/// Launch a batched MMVQ over `n_slots` slot-major activation columns, in
/// chunks of at most `MMVQ_BATCH_MAX` (entry `<recipe.entry>_n<cols>`).
fn mmvq_batched_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    recipe: Recipe,
    bufs: crate::MmvqBuffers,
    shape: crate::MmvqBatchShape,
    block_elems: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1, dst } = bufs;
    let crate::MmvqBatchShape { n_rows, k, n_slots } = shape;
    let module = reg.expect_module(recipe.stem)?;
    let entries = MMVQ_BATCHED_ENTRIES
        .iter()
        .find(|(stem, _)| *stem == recipe.stem)
        .map(|(_, entries)| entries)
        .ok_or_else(|| anyhow!("batched MMVQ stem {} has no entry table", recipe.stem))?;
    let act_col_bytes = (k / 32) * std::mem::size_of::<flambeau_quant::BlockQ8_1>();
    let dst_col_bytes = n_rows * 4;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / block_elems) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let mut c0 = 0;
    while c0 < n_slots {
        let cols = (n_slots - c0).min(MMVQ_BATCH_MAX);
        let entry = entries[cols - 1].ok_or_else(|| {
            anyhow!("batched MMVQ stem {} has no {cols}-column entry", recipe.stem)
        })?;
        let kernel = B::kernel(module, entry)?;
        let a_ptr: u64 = act_q8_1.offset_bytes(c0 * act_col_bytes).as_usize() as u64;
        let d_ptr: u64 = dst.offset_bytes(c0 * dst_col_bytes).as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&w_ptr);
        args.push_arg(&a_ptr);
        args.push_arg(&d_ptr);
        args.push_arg(&n_rows_i);
        args.push_arg(&n_blocks_i);
        let grid = (n_rows as u32).div_ceil(recipe.rows_per_block);
        B::launch(&kernel, stream, LaunchCfg::one_d(grid, recipe.threads), args)?;
        c0 += cols;
    }
    Ok(())
}

pub fn mmvq_f16_supported<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    dtype_weight: QDtype,
) -> bool {
    if dtype_weight == QDtype::F16 || dtype_weight == QDtype::BF16 {
        return false;
    }
    reg.mmvq_m1
        .get_or_init(dtype_weight, || resolve_mmvq_m1::<B>(reg, dtype_weight).ok())
        .is_some_and(|m1| m1.f16)
}

/// Weight × Q8_1 MMVQ writing directly into an F16 destination, saturating
/// at ±F16_MAX. Same kernel as [`mmvq`] (the m = 1 dispatch row), launched
/// through its F16-output entry: skips the F32 scratch and the separate
/// `cast_f32_to_f16` launch. Callers check [`mmvq_f16_supported`] first.
/// Caller must ensure `dst_f16` is sized `n_rows × sizeof(fp16)`.
pub fn mmvq_f16_direct<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MmvqBuffers,
    shape: crate::MmvqShape,
    dtype_weight: QDtype,
) -> Result<()> {
    let crate::MmvqShape { n_rows, k } = shape;
    assert_eq!(k % 32, 0, "MMVQ requires k % 32 == 0");
    let MmvqM1 { recipe, f16 } = mmvq_m1::<B>(reg, dtype_weight)?;
    let entry_f16 = recipe.entry_f16.filter(|_| f16).ok_or_else(|| {
        anyhow!(
            "mmvq_f16_direct: {} kernel {} has no F16-output entry in this build",
            dtype_weight.name(),
            recipe.stem
        )
    })?;
    mmvq_launch(
        reg, stream,
        Recipe { entry: entry_f16, ..recipe },
        buf,
        n_rows,
        k / block_elems(dtype_weight),
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
/// dtypes, n_superblocks for K-quants).
/// Module + kernel-entry pair for direct kernel launches that don't
/// flow through `Recipe`. File-private; if a third launcher needs this
/// shape outside `qmatmul.rs`, lift to `sig.rs`.
#[derive(Copy, Clone, Debug)]
pub struct KernelEntry {
    pub stem: &'static str,
    pub entry: &'static str,
}

/// Launch-tuning knobs for single-row `mmvq_simple_launch`-style
/// kernels (vs MMQ tile launchers that use `Recipe`).
#[derive(Copy, Clone, Debug)]
pub struct MmvqLaunchTune {
    pub threads: u32,
    pub rows_per_block: u32,
}

pub fn mmvq_simple_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    entry: KernelEntry,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    units_per_row: usize,
    tune: MmvqLaunchTune,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1, dst } = bufs;
    let module = reg.expect_module(entry.stem)?;
    let kernel = B::kernel(module, entry.entry)?;
    let n_rows_i = n_rows as i32;
    let n_units_i = units_per_row as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_units_i);
    let grid = (n_rows as u32).div_ceil(tune.rows_per_block);
    let cfg = LaunchCfg::one_d(grid, tune.threads);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Tile-M F16 MMQ. 64 threads/block = wave64 × 1 row each, MMQ_Y
/// = 64 output rows per block, MMQ_X = 8 activation rows per block. Each
/// thread owns one weight row's dot against all 8 activation rows; weight
/// is loaded into registers once per K-sub-block and reused 8×. The
/// activation tile (8 Q8_1 blocks) is read from L1 per sub-block across
/// the 64 threads.
fn mmq_f16_tile_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    n_tokens: usize,
    k: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1, dst } = bufs;
    assert_eq!(k % 32, 0, "mmq_f16_tile requires k % 32 == 0");
    if n_tokens == 0 {
        return Ok(());
    }
    let module = reg.expect_module("mmq_f16_tile")?;
    let kernel = B::kernel(module, "flambeau_mmq_f16_tile_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&n_blocks_i);
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
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Multi-row F16 MMQ. Same kernel math as mmvq_f16_launch, but
/// picks up n_tokens via grid.y. Stem `mmq_f16_q8_1`, block = 256, grid =
/// (n_rows, n_tokens).
fn mmq_f16_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    n_tokens: usize,
    k: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1, dst } = bufs;
    assert_eq!(k % 32, 0, "mmq_f16_q8_1 requires k % 32 == 0");
    if n_tokens == 0 {
        return Ok(());
    }
    let module = reg.expect_module("mmq_f16_q8_1")?;
    let kernel = B::kernel(module, "flambeau_mmq_f16_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_tokens_i);
    args.push_arg(&n_blocks_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, n_tokens as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Direct launch for the F16-weight × Q8_1-activation MMVQ.
/// Kernel stem `mmvq_f16_q8_1`, block = 256 threads, grid = n_rows.
fn mmvq_f16_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1, dst } = bufs;
    assert_eq!(k % 32, 0, "mmvq_f16_q8_1 requires k % 32 == 0");
    let module = reg.expect_module("mmvq_f16_q8_1")?;
    let kernel = B::kernel(module, "flambeau_mmvq_f16_q8_1")?;
    let n_rows_i = n_rows as i32;
    let n_blocks_i = (k / 32) as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_blocks_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Prefill-path MMQ for M batch rows. `m` must be ≥ the dispatch floor for
/// the dtype (e.g. 128 for Q4_K/Q6_K/Q8_0 4-warp).
pub fn mmq<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::MmvqBuffers,
    shape: crate::MatmulShape,
    dtype_weight: QDtype,
) -> Result<()> {
    let crate::MmvqBuffers {
        weights,
        act_q8_1,
        dst,
    } = buf;
    let crate::MatmulShape { m, k, n } = shape;
    let cfg = QMatMulCfg {
        dtype_weight,
        dtype_activation: QDtype::Q8_1,
        m,
        k,
        n,
    };
    let desc = B::dispatch_qmatmul(&cfg).ok_or_else(|| {
        anyhow!(
            "no MMQ impl for dtype={} m={m} k={k} n={n}",
            dtype_weight.name()
        )
    })?;
    let recipe = Recipe::from_impl_id(desc.impl_id)?;
    let nb_per_row = k / block_elems(dtype_weight);
    mmq_launch(
        reg, stream,
        recipe,
        crate::MmvqBuffers { weights, act_q8_1, dst },
        n,
        m,
        nb_per_row,
    )
}

// ---------------------------------------------------------------------------
// internal: impl_id → launch recipe
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecipeKind {
    Mmvq,
    MmqOracle,
    /// 5 port of candle's 4-warp LDS-tiled MMQ with DS4 Q8_1 activation
    /// layout. Differs from `MmqOracle` in: 2D block dims (64, 4, 1), 9 scalar
    /// args (ncols_x, nrows_x, ncols_y, stride_col_y, stride_row_x, nrows_dst),
    /// dynamic shared-memory bytes, and the Y buffer pre-formatted as
    /// `BlockQ8_1Mmq` via `flambeau_quantize_q8_1_mmq`.
    MmqLdsX64,
    /// Int8 tensor-core MMQ (mma.sync m16n8k32) for Q4_0 on sm_80+. Same DS4
    /// 144-B activation and 9 scalar args as MmqLdsX64, same (128, 64) tile,
    /// but block (32, NWARPS)=(32, 8) and static shared (dynamic bytes = 0).
    /// CUDA-only; gfx906 has no matrix core and keeps the dp4a MmqLdsX64 path.
    MmqMma,
    /// wave64 MMQ for Q5_K.
    /// Consumes the standard `flambeau_block_q8_1` layout (NOT DS4), so no
    /// special activation quantise is needed at call sites. Grid =
    /// (ceil(N/64), ceil(M/8)), block = 64 threads (1 warp), 8 output cols
    /// per tile. Args: `(vx, vy, dst, ncols_x=K, nrows_x=N, ncols_y=M,
    /// nrows_y=K, nrows_dst=N)`.
    MmqWave64,
    /// One output row per block over up to `MMVQ_BATCH_MAX` activation
    /// columns (entries `<entry>_n1` .. `_n8`); more columns run in chunks.
    MmvqBatched,
    /// Few-column int8 MMA (mma.sync m16n8k32, cp.async ring) for 1..24
    /// activation columns on sm_80+. Reads the B-fragment activation layout
    /// ([`ActLayout::Few`]) through `act_q8_1_mmq`; one entry per column-group
    /// count (`<entry>_c<ncg>_t1`), dynamic shared memory. CUDA-only.
    MmqFew,
}

impl RecipeKind {
    /// The activation layout a recipe of this kind reads for weights of
    /// `dtype_weight`.
    fn act_layout(self, dtype_weight: QDtype) -> ActLayout {
        match self {
            RecipeKind::MmqLdsX64 | RecipeKind::MmqMma => ActLayout::Mmq,
            RecipeKind::MmqFew => ActLayout::Few(few_scale(dtype_weight)),
            RecipeKind::Mmvq
            | RecipeKind::MmvqBatched
            | RecipeKind::MmqOracle
            | RecipeKind::MmqWave64 => ActLayout::Q8_1,
        }
    }
}

/// Scale record of the few-column activation the few-column MMQ of
/// `dtype_weight` reads: Q4_K / Q5_K need `d · Σq` for their mins.
fn few_scale(dtype_weight: QDtype) -> FewScale {
    match dtype_weight {
        QDtype::Q4_K | QDtype::Q5_K => FewScale::Ds4,
        _ => FewScale::D4,
    }
}

#[derive(Debug, Clone, Copy)]
struct Recipe {
    kind: RecipeKind,
    stem: &'static str,
    entry: &'static str,
    /// MMVQ only — the entry of the same module whose body is `entry`'s,
    /// writing F16 (saturated at ±65504) instead of F32.
    entry_f16: Option<&'static str>,
    threads: u32,
    /// MMVQ only — output rows per block.
    rows_per_block: u32,
    /// MMQ only — (rows_per_tile, batches_per_tile) of the launch grid.
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
        "mmq_q4_K_lds_tiled" => (256, 22528),
        // Generic LDS-tiled (mmq_lds_tiled.cuh), MMQ_Y=64, MMQ_X=64:
        // 4 * (MMQ_X * 36 + Tr::x_ints(64)). Q8_0/IQ4_NL weight blocks are 32 elems.
        "mmq_q5_K_lds_tiled" => (256, 4 * (64 * 36 + 4488)),
        "mmq_q6_K_lds_tiled" => (256, 4 * (64 * 36 + 4490)),
        "mmq_iq4_xs_lds_tiled" => (256, 4 * (64 * 36 + 4688)),
        "mmq_q4_K_lds_tiled_y64" => (256, 4 * (64 * 36 + 2440)),
        "mmq_q3_K_lds_tiled" => (256, 4 * (64 * 36 + 4488)),
        "mmq_iq4_nl_lds_tiled" => (32, 4 * (64 * 36 + 4688)),
        "mmq_q8_0_lds_tiled" => (32, 4 * (64 * 36 + 4688)),
        // 8- and 16-column small-batch tiles.
        "mmq_q5_K_lds_tiled_x8" => (256, 4 * (8 * 36 + 4488)),
        "mmq_q6_K_lds_tiled_x8" => (256, 4 * (8 * 36 + 4490)),
        "mmq_iq4_xs_lds_tiled_x8" => (256, 4 * (8 * 36 + 4688)),
        "mmq_q4_K_lds_tiled_y64_x8" => (256, 4 * (8 * 36 + 2440)),
        "mmq_q3_K_lds_tiled_x8" => (256, 4 * (8 * 36 + 4488)),
        "mmq_iq4_nl_lds_tiled_x8" => (32, 4 * (8 * 36 + 4688)),
        #[cfg(feature = "bench")]
        "mmq_q5_K_lds_tiled_x16" | "mmq_q3_K_lds_tiled_x16" => (256, 4 * (16 * 36 + 4488)),
        "mmq_q4_K_lds_tiled_y64_x16" => (256, 4 * (16 * 36 + 2440)),
        other => bail!("mmq_lds_x64_params: no entry for stem {other}"),
    })
}

impl Recipe {
    fn from_impl_id(impl_id: &str) -> Result<Self> {
        QMATMUL_RECIPES
            .iter()
            .find(|row| row.ids.contains(&impl_id))
            .map(|row| row.recipe)
            .ok_or_else(|| anyhow!("no launch recipe registered for impl_id {impl_id}"))
    }
}

/// `impl_id` that [`qmatmul_bench`] launches through the production path
/// ([`qmatmul`], dispatch table and its short-circuits included).
#[cfg(feature = "bench")]
pub const QMATMUL_BENCH_PRODUCTION: &str = "dispatch";

/// Bench-only candidates that reuse a registered recipe: bench id -> recipe
/// impl_id. No dispatch descriptor names a bench id, so [`qmatmul`] never
/// resolves one.
#[cfg(feature = "bench")]
const BENCH_QMATMUL_ALIASES: &[(&str, &str)] = &[
    ("bench:q3_k_lds_x8", "qmatmul_q3_K_mmq_lds_tiled_x8_gfx906"),
    ("bench:q4_k_lds_x8", "qmatmul_q4_K_mmq_lds_tiled_y64_x8_gfx906"),
    ("bench:q5_k_lds_x8", "qmatmul_q5_K_mmq_lds_tiled_x8_gfx906"),
    ("bench:q3_k_lds_x64", "qmatmul_q3_K_mmq_lds_tiled_gfx906"),
    ("bench:q4_k_lds_x64", "qmatmul_q4_K_mmq_lds_tiled_y64_gfx906"),
    ("bench:q4_k_lds_x16", "qmatmul_q4_K_mmq_lds_tiled_y64_x16_gfx906"),
    ("bench:q4_k_few_mma", "qmatmul_q4_k_mmq_few_mma"),
    ("bench:q5_k_lds_x64", "qmatmul_q5_K_mmq_lds_tiled_gfx906"),
];

/// Bench-only candidates without a dispatch row: the Q5_K and Q3_K
/// few-column MMA (swept, band not yet measured); uncertified, the Q3_K and
/// Q5_K 16-column LDS-tiled cells and the wave64 K-quant MMQ; and, under
/// `cfg(unverified)`, the HIP-only Q4_K r4 MMVQ.
#[cfg(feature = "bench")]
const BENCH_QMATMUL_RECIPES: &[RecipeRow] = &[
    RecipeRow {
        ids: &["qmatmul_q5_k_mmq_few_mma", "bench:q5_k_few_mma"],
        recipe: Recipe {
            kind: RecipeKind::MmqFew,
            stem: "mmq_few_q5_k_mma",
            entry: "flambeau_mmq_few_q5_k_mma",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q3_k_mmq_few_mma", "bench:q3_k_few_mma"],
        recipe: Recipe {
            kind: RecipeKind::MmqFew,
            stem: "mmq_few_q3_k_mma",
            entry: "flambeau_mmq_few_q3_k_mma",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (0, 0),
        },
    },
    #[cfg(unverified)]
    RecipeRow {
        ids: &["bench:q4_k_mmvq_r4"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q4_k_r4",
            entry: "flambeau_mmvq_q4_k_r4_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 4,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["bench:q3_k_lds_x16"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q3_K_lds_tiled_x16",
            entry: "flambeau_mmq_q3_K_lds_tiled_x16_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 16),
        },
    },
    RecipeRow {
        ids: &["bench:q3_k_wave64"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q3_K_wave64",
            entry: "flambeau_mmq_q3_K_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["bench:q4_k_wave64"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q4_K_wave64",
            entry: "flambeau_mmq_q4_K_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["bench:q5_k_lds_x16"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q5_K_lds_tiled_x16",
            entry: "flambeau_mmq_q5_K_lds_tiled_x16_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 16),
        },
    },
    RecipeRow {
        ids: &["bench:q5_k_wave64"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q5_K_wave64",
            entry: "flambeau_mmq_q5_K_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
];

#[cfg(feature = "bench")]
fn bench_recipe(impl_id: &str) -> Result<Recipe> {
    if let Some(row) = BENCH_QMATMUL_RECIPES.iter().find(|row| row.ids.contains(&impl_id)) {
        return Ok(row.recipe);
    }
    match BENCH_QMATMUL_ALIASES.iter().find(|(bench, _)| *bench == impl_id) {
        Some((_, id)) => Recipe::from_impl_id(id),
        None => Recipe::from_impl_id(impl_id),
    }
}

/// QMatMul through one named implementation, for timing: `impl_id` is
/// [`QMATMUL_BENCH_PRODUCTION`], a bench id, or any registered impl_id. The
/// caller provides the activation layout [`qmatmul_bench_act_layout`] names.
#[cfg(feature = "bench")]
pub fn qmatmul_bench<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    impl_id: &str,
    buf: crate::QmatmulBuffers,
    shape: crate::MatmulShape,
    dtype_weight: QDtype,
) -> Result<()> {
    if impl_id == QMATMUL_BENCH_PRODUCTION {
        return qmatmul::<B>(reg, stream, buf, shape, dtype_weight);
    }
    let recipe = bench_recipe(impl_id)?;
    if recipe.kind == RecipeKind::MmvqBatched {
        return mmvq_batched_launch(
            reg,
            stream,
            recipe,
            crate::MmvqBuffers {
                weights: buf.weights,
                act_q8_1: buf.act_q8_1,
                dst: buf.dst,
            },
            crate::MmvqBatchShape {
                n_rows: shape.n,
                k: shape.k,
                n_slots: shape.m,
            },
            block_elems(dtype_weight),
        );
    }
    launch_recipe(reg, stream, recipe, buf, shape, dtype_weight)
}

/// The activation layout [`qmatmul_bench`] with `impl_id` reads.
#[cfg(feature = "bench")]
pub fn qmatmul_bench_act_layout<B: flambeau_core::KernelBackend>(
    impl_id: &str,
    shape: crate::MatmulShape,
    dtype_weight: QDtype,
) -> Result<ActLayout> {
    if impl_id == QMATMUL_BENCH_PRODUCTION {
        return Ok(qmatmul_act_layout::<B>(shape, dtype_weight));
    }
    Ok(bench_recipe(impl_id)?.kind.act_layout(dtype_weight))
}

/// Launch recipe of every QMatMul impl a dispatch descriptor (or an alias of
/// one) can name, keyed by the impl_ids that resolve to it. Arch-neutral: HIP
/// and CUDA descriptors share rows. `recipe_table_tests` keeps it in step with
/// `QMATMUL_GFX906` / `QMATMUL_SM86` in both directions.
struct RecipeRow {
    ids: &'static [&'static str],
    recipe: Recipe,
}

const QMATMUL_RECIPES: &[RecipeRow] = &[
    RecipeRow {
        ids: &["qmatmul_q8_0_mmvq_dp4a_gfx906", "qmatmul_q8_0_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q8_0_dp4a",
            entry: "flambeau_mmvq_q8_0_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_q8_0_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q8_0_mmvq_dp4a_vdr2_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_q8_0_dp4a_vdr2_batched",
            entry: "flambeau_mmvq_q8_0_dp4a_vdr2_batched_q8_1",
            entry_f16: None,
            threads: 32,
            rows_per_block: 4,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q8_0_mmvq_t128_vdr2", "qmatmul_q8_0_mmvq_single_row_gfx906", "qmatmul_q8_0_mmvq_dp4a_vdr2"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q8_0_t128_vdr2",
            entry: "flambeau_mmvq_q8_0_t128_vdr2_q8_1",
            entry_f16: Some("flambeau_mmvq_q8_0_t128_vdr2_q8_1_f16"),
            threads: 128,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q2_k_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q2_k_dp4a",
            entry: "flambeau_mmvq_q2_k_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_q2_k_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q2_K_mmvq_r2_dp4a_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q2_k_r2_dp4a",
            entry: "flambeau_mmvq_q2_k_r2_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_q2_k_r2_dp4a_q8_1_f16"),
            threads: 64,
            rows_per_block: 2,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q3_k_mmvq_dp4a_t128_vdr2"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q3_k_dp4a_t128_vdr2",
            entry: "flambeau_mmvq_q3_k_dp4a_t128_vdr2_q8_1",
            entry_f16: Some("flambeau_mmvq_q3_k_dp4a_t128_vdr2_q8_1_f16"),
            threads: 128,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q3_k_mmvq_dp4a_t128_vdr2_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_q3_k_dp4a_t128_vdr2_batched",
            entry: "flambeau_mmvq_q3_k_dp4a_t128_vdr2_batched_q8_1",
            entry_f16: None,
            threads: 32,
            rows_per_block: 4,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q3_K_mmvq_r2_dp4a_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q3_k_r2_dp4a",
            entry: "flambeau_mmvq_q3_k_r2_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_q3_k_r2_dp4a_q8_1_f16"),
            threads: 64,
            rows_per_block: 2,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q8_k_mmvq_single_row"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q8_k",
            entry: "flambeau_mmvq_q8_K_q8_1",
            entry_f16: Some("flambeau_mmvq_q8_K_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_k_mmvq_dp4a_t128_vdr2"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q4_k_dp4a_t128_vdr2",
            entry: "flambeau_mmvq_q4_k_dp4a_t128_vdr2_q8_1",
            entry_f16: Some("flambeau_mmvq_q4_k_dp4a_t128_vdr2_q8_1_f16"),
            threads: 128,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_k_mmvq_dp4a_t128_vdr2_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_q4_k_dp4a_t128_vdr2_batched",
            entry: "flambeau_mmvq_q4_k_dp4a_t128_vdr2_batched_q8_1",
            entry_f16: None,
            threads: 32,
            rows_per_block: 2,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_k_mmvq_dp4a_t128_vdr2"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q5_k_dp4a_t128_vdr2",
            entry: "flambeau_mmvq_q5_k_dp4a_t128_vdr2_q8_1",
            entry_f16: Some("flambeau_mmvq_q5_k_dp4a_t128_vdr2_q8_1_f16"),
            threads: 128,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_k_mmvq_dp4a_t128_vdr2_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_q5_k_dp4a_t128_vdr2_batched",
            entry: "flambeau_mmvq_q5_k_dp4a_t128_vdr2_batched_q8_1",
            entry_f16: None,
            threads: 32,
            rows_per_block: 2,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq4_xs_dp4a_t128_vdr4",
            entry: "flambeau_mmvq_iq4_xs_dp4a_t128_vdr4_q8_1",
            entry_f16: Some("flambeau_mmvq_iq4_xs_dp4a_t128_vdr4_q8_1_f16"),
            threads: 128,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_iq4_xs_dp4a_t128_vdr4_batched",
            entry: "flambeau_mmvq_iq4_xs_dp4a_t128_vdr4_batched_q8_1",
            entry_f16: None,
            threads: 32,
            rows_per_block: 2,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_nl_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq4_nl_dp4a",
            entry: "flambeau_mmvq_iq4_nl_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_iq4_nl_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_nl_mmvq_dp4a_t128_vdr2"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq4_nl_dp4a_t128_vdr2",
            entry: "flambeau_mmvq_iq4_nl_dp4a_t128_vdr2_q8_1",
            entry_f16: Some("flambeau_mmvq_iq4_nl_dp4a_t128_vdr2_q8_1_f16"),
            threads: 128,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq3_s_mmvq_dp4a_t128_vdr2_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_iq3_s_dp4a_t128_vdr2_batched",
            entry: "flambeau_mmvq_iq3_s_dp4a_t128_vdr2_batched_q8_1",
            entry_f16: None,
            threads: 32,
            rows_per_block: 4,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_nl_mmvq_dp4a_t128_vdr2_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_iq4_nl_dp4a_t128_vdr2_batched",
            entry: "flambeau_mmvq_iq4_nl_dp4a_t128_vdr2_batched_q8_1",
            entry_f16: None,
            threads: 32,
            rows_per_block: 4,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq3_xxs_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq3_xxs_dp4a",
            entry: "flambeau_mmvq_iq3_xxs_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_iq3_xxs_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq3_s_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq3_s_dp4a",
            entry: "flambeau_mmvq_iq3_s_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_iq3_s_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq3_s_mmvq_dp4a_t128_vdr2"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq3_s_dp4a_t128_vdr2",
            entry: "flambeau_mmvq_iq3_s_dp4a_t128_vdr2_q8_1",
            entry_f16: Some("flambeau_mmvq_iq3_s_dp4a_t128_vdr2_q8_1_f16"),
            threads: 128,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq2_xxs_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq2_xxs_dp4a",
            entry: "flambeau_mmvq_iq2_xxs_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_iq2_xxs_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq2_xs_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq2_xs_dp4a",
            entry: "flambeau_mmvq_iq2_xs_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_iq2_xs_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq2_s_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq2_s_dp4a",
            entry: "flambeau_mmvq_iq2_s_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_iq2_s_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq1_s_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq1_s_dp4a",
            entry: "flambeau_mmvq_iq1_s_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_iq1_s_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq1_m_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_iq1_m_dp4a",
            entry: "flambeau_mmvq_iq1_m_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_iq1_m_dp4a_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq3_s_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_iq3_s_wave64",
            entry: "flambeau_mmq_iq3_s_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq3_xxs_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_iq3_xxs_wave64",
            entry: "flambeau_mmq_iq3_xxs_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq2_xxs_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_iq2_xxs_wave64",
            entry: "flambeau_mmq_iq2_xxs_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq2_xs_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_iq2_xs_wave64",
            entry: "flambeau_mmq_iq2_xs_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq2_s_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_iq2_s_wave64",
            entry: "flambeau_mmq_iq2_s_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq1_s_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_iq1_s_wave64",
            entry: "flambeau_mmq_iq1_s_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq1_m_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_iq1_m_wave64",
            entry: "flambeau_mmq_iq1_m_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_1_mmvq_dp4a_gfx906", "qmatmul_q4_1_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q4_1",
            entry: "flambeau_mmvq_q4_1_q8_1",
            entry_f16: Some("flambeau_mmvq_q4_1_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    // CUDA sm_86 dispatches Q4_0/Q5_0/Q5_1 decode to the single-row base
    // mmvq (the autotuned `_mmvq_dp4a` row); HIP routes these through MMQ,
    // so there was no shared mmvq arm. The base kernel is the sweep
    // oracle (256t, 1 row/block) and is present in the sm_86 catalogue.
    RecipeRow {
        ids: &["qmatmul_q4_0_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q4_0",
            entry: "flambeau_mmvq_q4_0_q8_1",
            entry_f16: Some("flambeau_mmvq_q4_0_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_0_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q5_0",
            entry: "flambeau_mmvq_q5_0_q8_1",
            entry_f16: Some("flambeau_mmvq_q5_0_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_1_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q5_1",
            entry: "flambeau_mmvq_q5_1_q8_1",
            entry_f16: Some("flambeau_mmvq_q5_1_q8_1_f16"),
            threads: 256,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_1_mmvq_t128_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q4_1_t128",
            entry: "flambeau_mmvq_q4_1_t128_q8_1",
            entry_f16: Some("flambeau_mmvq_q4_1_t128_q8_1_f16"),
            threads: 128,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_K_mmvq_nw1_r2_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q5_k_r2",
            entry: "flambeau_mmvq_q5_k_r2_q8_1",
            entry_f16: Some("flambeau_mmvq_q5_k_r2_q8_1_f16"),
            threads: 64,
            rows_per_block: 2,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q6_k_mmvq_dp4a"],
        recipe: Recipe {
            kind: RecipeKind::Mmvq,
            stem: "mmvq_q6_k_dp4a",
            entry: "flambeau_mmvq_q6_k_dp4a_q8_1",
            entry_f16: Some("flambeau_mmvq_q6_k_dp4a_q8_1_f16"),
            threads: 64,
            rows_per_block: 1,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q6_k_mmvq_dp4a_vdr2_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_q6_k_dp4a_vdr2_batched",
            entry: "flambeau_mmvq_q6_k_dp4a_vdr2_batched_q8_1",
            entry_f16: None,
            threads: 32,
            rows_per_block: 2,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q8_0_mmq_oracle_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqOracle,
            stem: "mmq_q8_0_oracle",
            entry: "flambeau_mmq_q8_0_oracle_q8_1",
            entry_f16: None,
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
    },
    RecipeRow {
        ids: &["qmatmul_q4_1_mmq_4warp_lds_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q4_1_4warp_lds",
            entry: "flambeau_mmq_q4_1_4warp_lds_q8_1",
            entry_f16: None,
            threads: 0, // unused for MmqLdsX64 (2D block dims hard-coded in launcher)
            rows_per_block: 0,
    // MMQ_Y=128, MMQ_X=64 — used by the launcher to compute the grid
    // and dynamic LDS bytes. These MUST match mmq_q4_1_4warp_lds.cu.
            mmq_tile: (128, 64),
        },
    },
    // wave64 MMQ for Q4_1. Same shape family as Q8_0/K-quant
    // wave64 kernels — MMQ_Y=64, TILE_N=8, 64 threads, DP4A inner.
    RecipeRow {
        ids: &["qmatmul_q4_1_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q4_1_wave64",
            entry: "flambeau_mmq_q4_1_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    // wave64 MMQ for Q4_0. Closes the prefill gap to
    // llama.cpp on Qwen3.6-35B-A3B-Q4_0 at dense + MoE shapes.
    RecipeRow {
        ids: &["qmatmul_q4_0_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q4_0_wave64",
            entry: "flambeau_mmq_q4_0_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    // 4-warp LDS-tiled Q4_0 MMQ — port of the Q4_1
    // 4warp_lds structure with bias-correction in the dot. Closes
    // the 2.5× dense-prefill gap (27B-Q4_0 73 → ~190 tok/s pp4).
    // MUST match mmq_q4_0_4warp_lds.cu's MMQ_Y=128, MMQ_X=64.
    RecipeRow {
        ids: &["qmatmul_q4_0_mmq_4warp_lds_gfx906", "qmatmul_q4_0_mmq_4warp_lds"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q4_0_4warp_lds",
            entry: "flambeau_mmq_q4_0_4warp_lds_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (128, 64),
        },
    },
    // Int8 tensor-core MMQ for Q4_0 (sm_80+). MUST match
    // mmq_q4_0_mma.cu's mmq_q4_0_mma_row<128, 64, 4, 2>.
    RecipeRow {
        ids: &["qmatmul_q4_0_mma"],
        recipe: Recipe {
            kind: RecipeKind::MmqMma,
            stem: "mmq_q4_0_mma",
            entry: "flambeau_mmq_q4_0_mma_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (256, 128),
        },
    },
    // wave64 MMQ for Q5_0. Same tile shape + launch as Q4_0;
    // inner loop adds the 5th-bit `16·bit·y` DP4A term
    // (mmvq_q5_0 pattern).
    RecipeRow {
        ids: &["qmatmul_q5_0_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q5_0_wave64",
            entry: "flambeau_mmq_q5_0_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_1_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q5_1_wave64",
            entry: "flambeau_mmq_q5_1_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_K_mmq_lds_tiled_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q5_K_lds_tiled",
            entry: "flambeau_mmq_q5_K_lds_tiled_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 64),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q6_K_mmq_lds_tiled_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q6_K_lds_tiled",
            entry: "flambeau_mmq_q6_K_lds_tiled_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 64),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_xs_mmq_lds_tiled_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_iq4_xs_lds_tiled",
            entry: "flambeau_mmq_iq4_xs_lds_tiled_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 64),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_K_mmq_lds_tiled_y64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q4_K_lds_tiled_y64",
            entry: "flambeau_mmq_q4_K_lds_tiled_y64_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 64),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q3_K_mmq_lds_tiled_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q3_K_lds_tiled",
            entry: "flambeau_mmq_q3_K_lds_tiled_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 64),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_nl_mmq_lds_tiled_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_iq4_nl_lds_tiled",
            entry: "flambeau_mmq_iq4_nl_lds_tiled_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 64),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q8_0_mmq_lds_tiled_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q8_0_lds_tiled",
            entry: "flambeau_mmq_q8_0_lds_tiled_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 64),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_K_mmq_lds_tiled_x8_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q5_K_lds_tiled_x8",
            entry: "flambeau_mmq_q5_K_lds_tiled_x8_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q6_K_mmq_lds_tiled_x8_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q6_K_lds_tiled_x8",
            entry: "flambeau_mmq_q6_K_lds_tiled_x8_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_xs_mmq_lds_tiled_x8_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_iq4_xs_lds_tiled_x8",
            entry: "flambeau_mmq_iq4_xs_lds_tiled_x8_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_K_mmq_lds_tiled_y64_x8_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q4_K_lds_tiled_y64_x8",
            entry: "flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_k_mmq_few_mma"],
        recipe: Recipe {
            kind: RecipeKind::MmqFew,
            stem: "mmq_few_q4_k_mma",
            entry: "flambeau_mmq_few_q4_k_mma",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_K_mmq_lds_tiled_y64_x16_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q4_K_lds_tiled_y64_x16",
            entry: "flambeau_mmq_q4_K_lds_tiled_y64_x16_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 16),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q4_k_mmvq_dp4a_row_tile_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_q4_k_dp4a_batched",
            entry: "flambeau_mmvq_q4_k_dp4a_q8_1_batched",
            entry_f16: None,
            threads: 256,
            rows_per_block: 8,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q5_k_mmvq_dp4a_row_tile_batched"],
        recipe: Recipe {
            kind: RecipeKind::MmvqBatched,
            stem: "mmvq_q5_k_dp4a_batched",
            entry: "flambeau_mmvq_q5_k_dp4a_q8_1_batched",
            entry_f16: None,
            threads: 256,
            rows_per_block: 8,
            mmq_tile: (0, 0),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q3_K_mmq_lds_tiled_x8_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_q3_K_lds_tiled_x8",
            entry: "flambeau_mmq_q3_K_lds_tiled_x8_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_iq4_nl_mmq_lds_tiled_x8_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqLdsX64,
            stem: "mmq_iq4_nl_lds_tiled_x8",
            entry: "flambeau_mmq_iq4_nl_lds_tiled_x8_q8_1",
            entry_f16: None,
            threads: 0,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q8_K_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q8_K_wave64",
            entry: "flambeau_mmq_q8_K_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
    RecipeRow {
        ids: &["qmatmul_q2_K_mmq_wave64_gfx906"],
        recipe: Recipe {
            kind: RecipeKind::MmqWave64,
            stem: "mmq_q2_K_wave64",
            entry: "flambeau_mmq_q2_K_wave64_q8_1",
            entry_f16: None,
            threads: 64,
            rows_per_block: 0,
            mmq_tile: (64, 8),
        },
    },
];

/// Resolve a dispatch `impl_id` to its kernel stem (the `.hsaco` / `.cubin`
/// catalogue name). Test/diagnostic hook: lets a cross-backend check assert that
/// every dispatched impl_id maps to a stem present in the registry.
/// # Errors
/// Propagates `Recipe::from_impl_id` when no recipe is registered.
#[doc(hidden)]
pub fn recipe_stem_for(impl_id: &str) -> Result<&'static str> {
    Ok(Recipe::from_impl_id(impl_id)?.stem)
}

fn mmvq_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    recipe: Recipe,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    n_units: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1, dst } = bufs;
    let module = reg.expect_module(recipe.stem)?;
    let kernel = B::kernel(module, recipe.entry)?;
    let n_rows_i = n_rows as i32;
    let n_units_i = n_units as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let a_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&a_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_units_i);
    let grid = (n_rows as u32).div_ceil(recipe.rows_per_block);
    let cfg = LaunchCfg::one_d(grid, recipe.threads);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

fn mmq_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    recipe: Recipe,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    n_batches: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1, dst } = bufs;
    let module = reg.expect_module(recipe.stem)?;
    let kernel = B::kernel(module, recipe.entry)?;
    let n_rows_i = n_rows as i32;
    let n_batches_i = n_batches as i32;
    let n_bpr_i = n_blocks_per_row as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let a_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&a_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&n_batches_i);
    args.push_arg(&n_bpr_i);
    let (rows_per_tile, batches_per_tile) = recipe.mmq_tile;
    let grid_x = (n_rows as u32).div_ceil(rows_per_tile);
    let grid_y = (n_batches as u32).div_ceil(batches_per_tile);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (recipe.threads, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Launch the 4-warp LDS-tiled MMQ with DS4 Q8_1 activation layout.
/// Expected inputs:
/// weights : flambeau_block_q4_1 * [n_rows, n_blocks_per_row]
/// act_q8_1 : flambeau_block_q8_1_mmq * [n_big_blocks_k, n_batches] (NOTE: MMQ layout)
/// dst : f32 * [n_batches, n_rows] (col-major in our naming)
/// where `n_big_blocks_k = n_blocks_per_row * QK4_1 / (4 * QK8_1) = n_blocks_per_row / 4`.
fn mmq_lds_x64_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    recipe: Recipe,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    n_batches: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: act_q8_1_mmq, dst } = bufs;
    let module = reg.expect_module(recipe.stem)?;
    let kernel = B::kernel(module, recipe.entry)?;

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
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&a_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&ncols_x);
    args.push_arg(&nrows_x);
    args.push_arg(&ncols_y);
    args.push_arg(&stride_col_y);
    args.push_arg(&stride_row_x);
    args.push_arg(&nrows_dst);

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
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Launch the int8 tensor-core Q4_0 MMQ (mma.sync). Same 9 args, activation,
/// and (128, 64) tile as `mmq_lds_x64_launch`; block is (32, NWARPS)=(32, 8)
/// and shared memory is static (dynamic bytes = 0).
/// Per few-column MMQ stem, the batched MMVQ impl that serves its shapes
/// when the few-column kernel cannot: a k it does not tile, more than 24
/// columns, or a caller that did not produce the few-column activation.
const MMQ_FEW_FALLBACK: &[(&str, &str)] = &[
    ("mmq_few_q4_k_mma", "qmatmul_q4_k_mmvq_dp4a_t128_vdr2_batched"),
    ("mmq_few_q5_k_mma", "qmatmul_q5_k_mmvq_dp4a_t128_vdr2_batched"),
    ("mmq_few_q3_k_mma", "qmatmul_q3_k_mmvq_dp4a_t128_vdr2_batched"),
];

fn few_fallback(recipe: Recipe) -> Result<Recipe> {
    let id = MMQ_FEW_FALLBACK
        .iter()
        .find(|(stem, _)| *stem == recipe.stem)
        .map(|(_, id)| *id)
        .ok_or_else(|| anyhow!("few-column MMQ stem {} has no fallback", recipe.stem))?;
    Recipe::from_impl_id(id)
}

/// Super-block units per cp.async ring stage of the few-column MMQ at
/// `ncg` column groups.
fn mmq_few_units_per_stage(ncg: usize) -> usize {
    if ncg > 2 { 1 } else { 2 }
}

/// Whether the few-column MMQ serves `m` columns at contraction `k`: 1..3
/// groups of 8 columns, whole ring stages of super-blocks, and at least 8
/// super-blocks so the 4 warps splitting K all work (llama.cpp's gate).
fn mmq_few_serves(m: usize, k: usize) -> bool {
    let ncg = m.div_ceil(MMQ_FEW_NCOLS);
    let nunits = k / flambeau_quant::QK_K;
    (1..=MMQ_FEW_MAX_COLS).contains(&m)
        && k % flambeau_quant::QK_K == 0
        && nunits >= 8
        && nunits % mmq_few_units_per_stage(ncg) == 0
}

/// Per few-column MMQ stem, its entries for 1, 2 and 3 column groups.
const MMQ_FEW_ENTRIES: &[(&str, [&str; 3])] = &[
    (
        "mmq_few_q4_k_mma",
        [
            "flambeau_mmq_few_q4_k_mma_c1_t1",
            "flambeau_mmq_few_q4_k_mma_c2_t1",
            "flambeau_mmq_few_q4_k_mma_c3_t1",
        ],
    ),
    (
        "mmq_few_q5_k_mma",
        [
            "flambeau_mmq_few_q5_k_mma_c1_t1",
            "flambeau_mmq_few_q5_k_mma_c2_t1",
            "flambeau_mmq_few_q5_k_mma_c3_t1",
        ],
    ),
    (
        "mmq_few_q3_k_mma",
        [
            "flambeau_mmq_few_q3_k_mma_c1_t1",
            "flambeau_mmq_few_q3_k_mma_c2_t1",
            "flambeau_mmq_few_q3_k_mma_c3_t1",
        ],
    ),
];

/// Ring row stride (bytes) of the few-column MMQ for `units` super-blocks of
/// `unit_bytes` per stage: whole 16-byte chunks, or for units that do not fill
/// whole chunks the worst case over the even start offsets, padded to an odd
/// chunk count (`mmq_few_mma.cuh` `sram_stride`).
fn mmq_few_sram_stride(unit_bytes: usize, units: usize) -> usize {
    let bytes = units * unit_bytes;
    if bytes % 16 == 0 {
        return bytes;
    }
    let chunks = (14 + bytes + 15) / 16;
    16 * if chunks % 2 == 0 { chunks + 1 } else { chunks }
}

/// Activation columns per group of the few-column MMQ.
const MMQ_FEW_NCOLS: usize = flambeau_quant::q8_1_few::FEW_NCOLS;

/// Most activation columns the few-column MMQ serves: 3 groups of 8.
pub const MMQ_FEW_MAX_COLS: usize = 3 * MMQ_FEW_NCOLS;

/// Launch a few-column MMQ: `ceil(m / 8)` column groups (at most 3), blocks of
/// 4 warps over 16 weight rows splitting K by super-blocks. `act` holds the
/// few-column activation of [`few_scale`]`(dtype_weight)`.
fn mmq_few_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    recipe: Recipe,
    bufs: crate::MmvqBuffers,
    shape: crate::MatmulShape,
    dtype_weight: QDtype,
) -> Result<()> {
    const NWARPS: u32 = 4;
    const NROWS: usize = 16;
    const NSTAGES: usize = 2;
    let crate::MmvqBuffers { weights, act_q8_1: act, dst } = bufs;
    let crate::MatmulShape { m, k, n } = shape;
    let ncg = m.div_ceil(MMQ_FEW_NCOLS);
    let unit_bytes = match dtype_weight {
        QDtype::Q4_K => std::mem::size_of::<flambeau_quant::BlockQ4K>(),
        QDtype::Q5_K => std::mem::size_of::<flambeau_quant::BlockQ5K>(),
        QDtype::Q3_K => std::mem::size_of::<flambeau_quant::BlockQ3K>(),
        other => bail!("mmq_few: no few-column decoder for {other:?}"),
    };
    let nunits = k / flambeau_quant::QK_K;
    let units_per_stage = mmq_few_units_per_stage(ncg);
    if !mmq_few_serves(m, k) {
        bail!("mmq_few: m = {m}, k = {k} outside the few-column tiling");
    }
    if act.as_usize() == 0 {
        bail!("mmq_few: no few-column activation");
    }
    // Whole-chunk super-blocks are copied as is (16-byte cp.async); the
    // others from the boundary before them, decoded with 2-byte loads.
    let align = if unit_bytes % 16 == 0 { 16 } else { 2 };
    if weights.as_usize() % align != 0 {
        bail!("mmq_few: weight rows must be {align}-byte aligned");
    }
    let entries = MMQ_FEW_ENTRIES
        .iter()
        .find(|(stem, _)| *stem == recipe.stem)
        .map(|(_, entries)| entries)
        .ok_or_else(|| anyhow!("few-column MMQ stem {} has no entry table", recipe.stem))?;
    let module = reg.expect_module(recipe.stem)?;
    let kernel = B::kernel(module, entries[ncg - 1])?;

    let w_ptr: u64 = weights.as_usize() as u64;
    let y_ptr: u64 = act.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let nunits_i = nunits as i32;
    let nrows_i = n as i32;
    let ncols_i = m as i32;
    let stride_row_x = (nunits * unit_bytes) as i32;
    let stride_col_dst = n as i32;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&nunits_i);
    args.push_arg(&nrows_i);
    args.push_arg(&ncols_i);
    args.push_arg(&stride_row_x);
    args.push_arg(&stride_col_dst);
    let shared_bytes =
        (NWARPS as usize * NSTAGES * NROWS * mmq_few_sram_stride(unit_bytes, units_per_stage)) as u32;
    let cfg = LaunchCfg {
        grid: (n.div_ceil(NROWS) as u32, 1, 1),
        block: (32, NWARPS, 1),
        shared_bytes,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

fn mmq_mma_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    recipe: Recipe,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    n_batches: usize,
    n_blocks_per_row: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1: act_q8_1_mmq, dst } = bufs;
    let module = reg.expect_module(recipe.stem)?;
    let kernel = B::kernel(module, recipe.entry)?;

    let ncols_x = (n_blocks_per_row * 32) as i32; // Q4_0 block = 32 elements
    let nrows_x = n_rows as i32;
    let ncols_y = n_batches as i32;
    let stride_col_y = n_batches as i32;
    let stride_row_x = n_blocks_per_row as i32;
    let nrows_dst = n_rows as i32;

    let w_ptr: u64 = weights.as_usize() as u64;
    let a_ptr: u64 = act_q8_1_mmq.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&a_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&ncols_x);
    args.push_arg(&nrows_x);
    args.push_arg(&ncols_y);
    args.push_arg(&stride_col_y);
    args.push_arg(&stride_row_x);
    args.push_arg(&nrows_dst);

    let (rows_per_tile, batches_per_tile) = recipe.mmq_tile;
    let grid_x = (n_rows as u32).div_ceil(rows_per_tile);
    let grid_y = (n_batches as u32).div_ceil(batches_per_tile);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (32, 16, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Launch the wave64 Q5_K MMQ (). Expected inputs:
/// weights : flambeau_block_q5_K * [n_rows, K/QK_K]
/// act_q8_1 : flambeau_block_q8_1 * [n_batches, K/QK8_1] (standard layout)
/// dst : f32 * [n_batches, n_rows] (col-major; `dst[col*nrows_dst+row]`)
fn mmq_wave64_launch<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    recipe: Recipe,
    bufs: crate::MmvqBuffers,
    n_rows: usize,
    n_batches: usize,
    k: usize,
) -> Result<()> {
    let crate::MmvqBuffers { weights, act_q8_1, dst } = bufs;
    let module = reg.expect_module(recipe.stem)?;
    let kernel = B::kernel(module, recipe.entry)?;
    let ncols_x = k as i32;
    let nrows_x = n_rows as i32;
    let ncols_y = n_batches as i32;
    let nrows_y = k as i32; // Y's K dim in elements (blocks = K / QK8_1)
    let nrows_dst = n_rows as i32;
    let w_ptr: u64 = weights.as_usize() as u64;
    let a_ptr: u64 = act_q8_1.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&a_ptr);
    args.push_arg(&d_ptr);
    args.push_arg(&ncols_x);
    args.push_arg(&nrows_x);
    args.push_arg(&ncols_y);
    args.push_arg(&nrows_y);
    args.push_arg(&nrows_dst);
    let (rows_per_tile, batches_per_tile) = recipe.mmq_tile;
    let grid_x = (n_rows as u32).div_ceil(rows_per_tile);
    let grid_y = (n_batches as u32).div_ceil(batches_per_tile);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (recipe.threads, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
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

#[cfg(test)]
mod recipe_table_tests {
    use super::{RecipeKind, MMVQ_BATCHED_ENTRIES, QMATMUL_RECIPES};

    #[test]
    fn every_impl_id_names_one_recipe() {
        let mut seen = std::collections::HashSet::new();
        for row in QMATMUL_RECIPES {
            for id in row.ids {
                assert!(seen.insert(*id), "impl_id {id} listed in two recipe rows");
            }
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn every_gfx906_descriptor_has_a_recipe() {
        for d in flambeau_backend_hip::QMATMUL_GFX906 {
            assert!(super::Recipe::from_impl_id(d.impl_id).is_ok(), "no recipe for {}", d.impl_id);
        }
    }

    /// The sm_86 Q4_K few-column band and its fallbacks, as callers see them.
    #[cfg(feature = "cuda")]
    #[test]
    fn sm86_q4_k_few_column_layouts() {
        use super::{qmatmul_act_layout, ActLayout, FewScale};
        use flambeau_core::QDtype;
        let at = |m, k| {
            qmatmul_act_layout::<flambeau_backend_cuda::CudaKernels>(
                crate::MatmulShape { m, k, n: 5120 },
                QDtype::Q4_K,
            )
        };
        assert_eq!(at(5, 5120), ActLayout::Few(FewScale::Ds4));
        assert_eq!(at(24, 17408), ActLayout::Few(FewScale::Ds4));
        assert_eq!(at(17, 5120 + 256), ActLayout::Few(FewScale::Ds4), "3 groups: 1 unit per stage");
        assert_eq!(at(5, 5120 + 256), ActLayout::Q8_1, "odd super-blocks at 2 units per stage");
        assert_eq!(at(5, 1792), ActLayout::Q8_1, "fewer than 8 super-blocks");
        assert_eq!(at(2, 5120), ActLayout::Q8_1);
        assert_eq!(at(25, 5120), ActLayout::Q8_1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn every_sm86_descriptor_has_a_recipe() {
        for d in flambeau_backend_cuda::impls::QMATMUL_SM86 {
            assert!(super::Recipe::from_impl_id(d.impl_id).is_ok(), "no recipe for {}", d.impl_id);
        }
    }

    /// Every batched recipe has an entry table, named after its base entry.
    #[test]
    fn every_batched_recipe_has_its_entries() {
        for row in QMATMUL_RECIPES.iter().filter(|r| r.recipe.kind == RecipeKind::MmvqBatched) {
            let (_, entries) = MMVQ_BATCHED_ENTRIES
                .iter()
                .find(|(stem, _)| *stem == row.recipe.stem)
                .unwrap_or_else(|| panic!("no entry table for {}", row.recipe.stem));
            for (i, e) in entries.iter().enumerate() {
                if let Some(e) = e {
                    assert_eq!(*e, format!("{}_n{}", row.recipe.entry, i + 1));
                }
            }
        }
    }

    /// A recipe no descriptor of either backend can name is dead: the launch
    /// path only resolves recipes from dispatched descriptors.
    #[cfg(all(feature = "hip", feature = "cuda"))]
    #[test]
    fn every_recipe_serves_a_descriptor() {
        let descriptors: Vec<&str> = flambeau_backend_hip::QMATMUL_GFX906
            .iter()
            .chain(flambeau_backend_cuda::impls::QMATMUL_SM86)
            .map(|d| d.impl_id)
            .collect();
        for row in QMATMUL_RECIPES {
            assert!(
                row.ids.iter().any(|id| descriptors.contains(id)),
                "recipe {:?} is named by no QMatMul descriptor",
                row.ids
            );
        }
    }
}
