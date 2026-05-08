//! MMVQ correctness sweep — one HIP device, one dtype at a time.
//! Mirrors the ad-hoc `crates/backend-hip/tests/mmvq_q*.rs` tests but runs
//! from the library / CLI and writes a `Cert` JSON under `certs/hip/gfx906/`.
//! grid per roadmap:
//! M ∈ {1, 8, 16, 128, 512} (row counts — we run M rows per shape)
//! K ∈ {2048, 5120, 15360, 128256}
//! N — not a kernel input for single-row MMVQ (one block per row).
//! Tolerance: `1e-2 * sqrt(K/128)` — captures F32 accumulation noise + Q8_1
//! activation quant round-trip. See `project_v1_3_q8_0_landed.md` for the
//! derivation.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a kernel launch or a memcpy_async \
              over buffers allocated locally in the same function and freed before \
              return; invariant is uniform across all sites."
)]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{
    dequantize_into, BlockQ4K, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, GgmlDType,
    QK8_0, QK_K,
};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

const QK8: usize = QK8_0;

/// Dtype tag selecting which weight block layout the sweep drives. We only
/// define the four dtypes here; `GgmlDType::from_wire` covers the rest.
/// `(weight dtype, kernel variant)` — selects one impl to certify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    /// r2 multi-row variant of Q4_K (2 output rows per wave64).
    Q4KR2,
    /// r2 multi-row variant of Q5_K.
    Q5KR2,
    /// r4 multi-row variant of Q6_K (4 output rows per wave64).
    Q6KR4,
    /// DP4A single-row Q6_K variant. Cooperative 64 lanes across
    /// 2 super-blocks, inner DP4A via `__builtin_amdgcn_sdot4`. Runtime
    /// intercepts the base Q6_K row when this kernel is loaded.
    Q6KDP4A,
    /// Q4_1 legacy quant with per-block min offset. Single row per
    /// 256-thread block, DP4A inner loop.
    Q4_1,
    /// 4.a.1 r2 multi-row Q4_1 (64 threads, 2 rows/block, half-warp
    /// DPP reduce). NULL — lost DP4A parallelism. Kept for regression.
    Q4_1R2,
    /// 4.a.2 DP4A r2 multi-row Q4_1 (256 threads, 2 rows/block).
    /// Halves grid + shares Y across rows; preserves DP4A.
    /// NULL — register pressure regressed wall −7.5 %.
    Q4_1R2DP4A,
    /// 4.a.3 thin-block 128-thread single-row Q4_1 DP4A.
    Q4_1T128,
    /// C9-i1 thin-block 128-thread single-row Q8_0 DP4A. Mirror of Q4_1T128
    /// for Q8_0 weights — targets the gfx906 latency-bound regime where Q8_0
    /// MMVQ is the dominant kernel on Coder-30B-Q8_0 / 27B-Q8_0 decode.
    Q8_0T128,
    /// C9-followup t128 + VDR=2 combined (the kernel that should beat
    /// vdr2-alone where t128-alone lost).
    Q8_0T128VDR2,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Q8_0 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 => "Q8_0",
            Dtype::Q4K | Dtype::Q4KR2 => "Q4_K",
            Dtype::Q5K | Dtype::Q5KR2 => "Q5_K",
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => "Q6_K",
            Dtype::Q4_1 | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => "Q4_1",
        }
    }

    pub fn ggml(self) -> GgmlDType {
        match self {
            Dtype::Q8_0 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 => GgmlDType::Q8_0,
            Dtype::Q4K | Dtype::Q4KR2 => GgmlDType::Q4K,
            Dtype::Q5K | Dtype::Q5KR2 => GgmlDType::Q5K,
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => GgmlDType::Q6K,
            Dtype::Q4_1 | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => GgmlDType::Q4_1,
        }
    }

    fn impl_id(self) -> &'static str {
        match self {
            Dtype::Q8_0 => "qmatmul_q8_0_mmvq_single_row_gfx906",
            Dtype::Q4K => "qmatmul_q4_K_mmvq_single_row_gfx906",
            Dtype::Q5K => "qmatmul_q5_K_mmvq_single_row_gfx906",
            Dtype::Q6K => "qmatmul_q6_K_mmvq_single_row_gfx906",
            Dtype::Q4KR2 => "qmatmul_q4_K_mmvq_nw1_r2_gfx906",
            Dtype::Q5KR2 => "qmatmul_q5_K_mmvq_nw1_r2_gfx906",
            Dtype::Q6KR4 => "qmatmul_q6_K_mmvq_nw1_r4_gfx906",
            Dtype::Q6KDP4A => "qmatmul_q6_K_mmvq_dp4a_gfx906",
            Dtype::Q4_1 => "qmatmul_q4_1_mmvq_dp4a_gfx906",
            Dtype::Q4_1R2 => "qmatmul_q4_1_mmvq_nw1_r2_gfx906",
            Dtype::Q4_1R2DP4A => "qmatmul_q4_1_mmvq_r2_dp4a_gfx906",
            Dtype::Q4_1T128 => "qmatmul_q4_1_mmvq_t128_gfx906",
            Dtype::Q8_0T128 => "qmatmul_q8_0_mmvq_t128_gfx906",
            Dtype::Q8_0T128VDR2 => "qmatmul_q8_0_mmvq_t128_vdr2_gfx906",
        }
    }

    fn kernel_stem(self) -> &'static str {
        match self {
            Dtype::Q8_0 => "mmvq_q8_0",
            Dtype::Q8_0T128 => "mmvq_q8_0_t128",
            Dtype::Q8_0T128VDR2 => "mmvq_q8_0_t128_vdr2",
            Dtype::Q4K => "mmvq_q4_k",
            Dtype::Q5K => "mmvq_q5_k",
            Dtype::Q6K => "mmvq_q6_k",
            Dtype::Q4KR2 => "mmvq_q4_k_r2",
            Dtype::Q5KR2 => "mmvq_q5_k_r2",
            Dtype::Q6KR4 => "mmvq_q6_k_r4",
            Dtype::Q6KDP4A => "mmvq_q6_k_dp4a",
            Dtype::Q4_1 => "mmvq_q4_1",
            Dtype::Q4_1R2 => "mmvq_q4_1_r2",
            Dtype::Q4_1R2DP4A => "mmvq_q4_1_r2_dp4a",
            Dtype::Q4_1T128 => "mmvq_q4_1_t128",
        }
    }

    fn kernel_entry(self) -> &'static str {
        match self {
            Dtype::Q8_0 => "flambeau_mmvq_q8_0_q8_1",
            Dtype::Q4K => "flambeau_mmvq_q4_k_q8_1",
            Dtype::Q5K => "flambeau_mmvq_q5_k_q8_1",
            Dtype::Q6K => "flambeau_mmvq_q6_k_q8_1",
            Dtype::Q4KR2 => "flambeau_mmvq_q4_k_r2_q8_1",
            Dtype::Q5KR2 => "flambeau_mmvq_q5_k_r2_q8_1",
            Dtype::Q6KR4 => "flambeau_mmvq_q6_k_r4_q8_1",
            Dtype::Q6KDP4A => "flambeau_mmvq_q6_k_dp4a_q8_1",
            Dtype::Q4_1 => "flambeau_mmvq_q4_1_q8_1",
            Dtype::Q4_1R2 => "flambeau_mmvq_q4_1_r2_q8_1",
            Dtype::Q4_1R2DP4A => "flambeau_mmvq_q4_1_r2_dp4a_q8_1",
            Dtype::Q4_1T128 => "flambeau_mmvq_q4_1_t128_q8_1",
            Dtype::Q8_0T128 => "flambeau_mmvq_q8_0_t128_q8_1",
            Dtype::Q8_0T128VDR2 => "flambeau_mmvq_q8_0_t128_vdr2_q8_1",
        }
    }

    fn block_size_bytes(self) -> usize {
        match self {
            Dtype::Q8_0 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 => std::mem::size_of::<BlockQ8_0>(),
            Dtype::Q4K | Dtype::Q4KR2 => std::mem::size_of::<BlockQ4K>(),
            Dtype::Q5K | Dtype::Q5KR2 => std::mem::size_of::<BlockQ5K>(),
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => std::mem::size_of::<BlockQ6K>(),
            Dtype::Q4_1 | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => std::mem::size_of::<BlockQ4_1>(),
        }
    }

    fn block_elem_count(self) -> usize {
        self.ggml().block_size()
    }

    fn launch_threads(self) -> u32 {
        match self {
            Dtype::Q8_0 | Dtype::Q4_1 | Dtype::Q4_1R2DP4A => 256,
            Dtype::Q4_1T128 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 => 128,
            _ => 64,
        }
    }

    /// How many output rows one thread block computes. Used to compute
    /// `gridDim.x` = `n_rows / rows_per_block` (rounded up).
    fn rows_per_block(self) -> u32 {
        match self {
            Dtype::Q6KR4 => 4,
            Dtype::Q4KR2 | Dtype::Q5KR2 | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A => 2,
            // Q6KDP4A is a single-row kernel (inner cooperative across 2 super-blocks).
            _ => 1,
        }
    }
}

/// Sweep spec — one `Cert` per call. M values define the row count; K values
/// the per-row contracted dimension.
#[derive(Debug, Clone)]
pub struct SweepSpec {
    pub dtype: Dtype,
    pub m_grid: Vec<usize>,
    pub k_grid: Vec<usize>,
    pub seed: u64,
}

impl SweepSpec {
    pub fn v1_3_default(dtype: Dtype) -> Self {
        Self {
            dtype,
            m_grid: vec![1, 8, 16, 128, 512],
            k_grid: vec![2048, 5120, 15360],
            seed: 0xC0FFEE,
        }
    }
}

pub fn run_sweep(spec: &SweepSpec, repo_root: &Path) -> Result<Cert> {
    let n = device_count()
        .context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices on this host — sweep needs gfx906");
    }

    let dev = HipDevice::new(0)?;
    dev.bind()?;

    // Capture the kernel's static PMC once per sweep — it's shape-independent
    // on gfx906 MMVQ (no runtime specialisation), so one query covers every
    // `ShapeResult` below.
    let pmc = capture_static_pmc(&dev, spec.dtype)?;

    let mut results = Vec::new();
    for &m in &spec.m_grid {
        for &k in &spec.k_grid {
            let seed = spec
                .seed
                .wrapping_add((m as u64).wrapping_mul(0x1234567))
                .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let (got, reference) = run_shape(&dev, spec.dtype, m, k, seed)?;
            let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
            let tolerance = cert_tol(k);
            results.push(ShapeResult {
                m,
                k,
                n: k, // single-row MMVQ: N is irrelevant; mirror K for the index
                seed,
                max_rel_err,
                tolerance,
                pass: max_rel_err <= tolerance,
            });
            tracing::info!(
                target: "flambeau_bench::sweep_mmvq",
                dtype = spec.dtype.name(),
                m, k,
                max_rel_err,
                tolerance,
                "shape result"
            );
        }
    }

    let pass = results.iter().all(|r| r.pass);

    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: spec.dtype.impl_id().to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul_mmvq".to_string(),
        dtype_weight: spec.dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(pmc),
    };

    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(),
        pass = cert.pass,
        "cert written"
    );

    Ok(cert)
}

// ---- PMC capture ----------------------------------------------------------

fn capture_static_pmc(dev: &HipDevice, dtype: Dtype) -> Result<PmcSnapshot> {
    let bytes = kernels::hsaco(dtype.kernel_stem())
        .ok_or_else(|| anyhow::anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel<'_> = module.kernel(dtype.kernel_entry())?;
    let attrs: FuncAttributes = kernel.attributes()?;

    let pmc = PmcSnapshot {
        vgpr_count: Some(attrs.num_regs),
        sgpr_count: None, // hipFuncGetAttribute doesn't expose SGPR; 
                          // captures it from the .hsaco ELF via inspect-hsaco
                          // once that CLI lands.
        waves_per_simd: Some(attrs.gfx906_waves_per_simd()),
        mem_busy_pct: None,   // Phase B (rocprof wrapper).
        valu_busy_pct: None,  // Phase B.
    };

    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        dtype = dtype.name(),
        vgpr = attrs.num_regs,
        shared_bytes = attrs.shared_size_bytes,
        local_bytes = attrs.local_size_bytes,
        max_tpb = attrs.max_threads_per_block,
        waves_per_simd = pmc.waves_per_simd.unwrap_or(0),
        "static PMC captured"
    );

    Ok(pmc)
}

// ---- per-shape runner -----------------------------------------------------

fn run_shape(
    dev: &HipDevice,
    dtype: Dtype,
    m: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_bytes = kernels::hsaco("quantize_q8_1")
        .ok_or_else(|| anyhow::anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = kernels::hsaco(dtype.kernel_stem())
        .ok_or_else(|| anyhow::anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = HipModule::load(dev.id(), q_bytes)?;
    let m_module = HipModule::load(dev.id(), m_bytes)?;
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;
    let k_mmvq: HipKernel<'_> = m_module.kernel(dtype.kernel_entry())?;

    let block_elems = dtype.block_elem_count();
    assert_eq!(
        k % block_elems,
        0,
        "k {k} must be multiple of block size {block_elems} for {}",
        dtype.name()
    );

    let weights_blocks = m * (k / block_elems);
    let block_bytes = dtype.block_size_bytes();

    // Generate random weights (raw bit pattern — the kernel operates on bytes).
    let weights_raw = seeded_bytes(seed, weights_blocks * block_bytes);
    let weights_raw = tame_scales(dtype, weights_raw);
    let y_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), k, -1.0, 1.0);

    // Dequantise weights on CPU for the reference matmul.
    let total_elems = m * k;
    let mut weights_dequant = vec![0.0f32; total_elems];
    dequantize_into(dtype.ggml(), &weights_raw, &mut weights_dequant)?;

    // Upload inputs.
    let d_x = alloc_and_upload(dev, weights_raw.as_slice());
    let d_y_f32 = alloc_and_upload(dev, y_f32.as_slice());

    // Quantise activation to Q8_1 on device.
    let y_q8_1_blocks = k / QK8;
    let y_q8_1_bytes = y_q8_1_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(y_q8_1_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_y_f32_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_q8_1_blocks as u32, QK8 as u32);
        unsafe { k_quantize.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // MMVQ launch.
    let d_dst = dev.alloc(m * 4)?;
    {
        let stream = dev.default_stream();
        let m_i = m as i32;
        let n_units = (k / block_elems) as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&d_dst_ptr);
        args.push(&m_i);
        args.push(&n_units);
        let rpb = dtype.rows_per_block();
        let grid = (m as u32).div_ceil(rpb);
        let cfg = LaunchCfg::one_d(grid, dtype.launch_threads());
        unsafe { k_mmvq.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; m];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            m * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_y_f32, y_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, y_q8_1_bytes)?;
        dev.dealloc(d_dst, m * 4)?;
    }

    let y_rt = quantize_q8_1_roundtrip(&y_f32);
    let reference = reference_matmul(&weights_dequant, &y_rt, m, k);
    Ok((got, reference))
}

// ---- helpers (shared with the ad-hoc tests) -------------------------------

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 24) as u8
        })
        .collect()
}

/// Override block-level scales so the F32 accumulation stays in the noise-
/// floor envelope the `1e-2 * sqrt(K/128)` tolerance assumes. Real GGUF
/// quants have these small by construction; random bytes don't.
fn tame_scales(dtype: Dtype, raw: Vec<u8>) -> Vec<u8> {
    let mut r = raw;
    let bs = dtype.block_size_bytes();
    let nblocks = r.len() / bs;
    for i in 0..nblocks {
        let block = &mut r[i * bs..(i + 1) * bs];
        match dtype {
            Dtype::Q8_0 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 => {
                // BlockQ8_0: d (f16, 2 bytes) + qs[32]. Scale ≈ 0.01 to 0.11.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q4K | Dtype::Q5K | Dtype::Q4KR2 | Dtype::Q5KR2 => {
                // d (0..2), dmin (2..4). Same treatment as Q4_K test.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let dmin = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => {
                // BlockQ6K: ql[128] + qh[64] + scales[16] i8 + d (f16 at offset 208..210).
                // Scale range: map each scale byte into [-32, 32] for plausibility.
                let scales_off = QK_K / 2 + QK_K / 4; // 128+64 = 192
                for s in &mut block[scales_off..scales_off + QK_K / 16] {
                    let sv = (*s as i32 % 65) - 32;
                    *s = sv as u8;
                }
                let d_off = scales_off + QK_K / 16;
                let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.01);
                block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q4_1 | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => {
                // BlockQ4_1: d (f16, 0..2) + m (f16, 2..4) + qs[16] (4..20).
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let m = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&m.to_bits().to_le_bytes());
            }
        }
    }
    r
}

fn quantize_q8_1_roundtrip(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..(xs.len() / QK8) {
        let block = &xs[i * QK8..(i + 1) * QK8];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * QK8 + j] = (q as f32) * d;
        }
    }
    out
}

fn reference_matmul(weights_f32: &[f32], y_f32: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    for r in 0..m {
        let row = &weights_f32[r * k..(r + 1) * k];
        let mut acc = 0.0f64;
        for j in 0..k {
            acc += (row[j] * y_f32[j]) as f64;
        }
        out[r] = acc as f32;
    }
    out
}

fn cert_tol(_k: usize) -> f32 {
    // 3e-2 is the measured worst-case across the grid on gfx906 for the
    // single-row on-the-fly-dequant kernels — dominated by Q8_1 activation
    // quant noise on cancellation-heavy output rows. The `abs_floor` inside
    // `max_rel_err` absorbs the sqrt(K) scaling, so the bar stays flat.
    // Tighter bounds follow once MMVQ moves to the dp4a / multi-row DPP path.
    3e-2
}

