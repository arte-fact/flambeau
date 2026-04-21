//! V1.4 MMQ correctness sweep.
//!
//! MMQ = matrix × matrix (prefill path). Inputs:
//!   weights: [N, K]  Q-format blocks
//!   act:     [M, K]  F32 → quantised to Q8_1 on device
//!   output:  [M, N]  F32
//!
//! The V1.4 roadmap cert grid is `M ∈ {128, 512, 2048}` × Qwen3.6 attention /
//! MoE projection widths (K, N). This sweep runs a compact sub-grid; full
//! coverage lands with the first-class 4-warp LDS-tiled port next session.
//!
//! Current impl: `qmatmul_q8_0_mmq_oracle_gfx906` — the `mmq_q8_0_oracle`
//! kernel (one output element per single-wave block). Slow but correct.

#![cfg(feature = "hip")]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{
    dequantize_into, BlockQ4K, BlockQ6K, BlockQ8_0, BlockQ8_1, GgmlDType, QK8_0, QK_K,
};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};

const QK8: usize = QK8_0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Q8_0Oracle,
    /// 4-warp LDS-tiled MMQ, MMQ_Y=32, MMQ_X=8, 256 threads.
    Q8_04Warp,
    /// 4-warp LDS-tiled Q4_K MMQ, MMQ_Y=16, MMQ_X=8, 128 threads.
    Q4K4Warp,
    /// 4-warp LDS-tiled Q6_K MMQ, MMQ_Y=16, MMQ_X=8, 128 threads.
    Q6K4Warp,
}

/// Shape of the output tile a single thread block produces for `dtype`.
/// `(rows_per_block, batches_per_block)` — the grid is
/// `(ceil(n / rows_per_block), ceil(m / batches_per_block))`.
fn tile_shape(dtype: Dtype) -> (u32, u32) {
    match dtype {
        Dtype::Q8_0Oracle => (1, 1),   // one output element per block
        Dtype::Q8_04Warp => (32, 8),   // MMQ_Y × MMQ_X
        Dtype::Q4K4Warp => (16, 8),    // MMQ_Y × MMQ_X (Q4_K uses 16 rows to fit LDS)
        Dtype::Q6K4Warp => (16, 8),    // MMQ_Y × MMQ_X (Q6_K same tile as Q4_K)
    }
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle | Dtype::Q8_04Warp => "Q8_0",
            Dtype::Q4K4Warp => "Q4_K",
            Dtype::Q6K4Warp => "Q6_K",
        }
    }

    pub fn ggml(self) -> GgmlDType {
        match self {
            Dtype::Q8_0Oracle | Dtype::Q8_04Warp => GgmlDType::Q8_0,
            Dtype::Q4K4Warp => GgmlDType::Q4K,
            Dtype::Q6K4Warp => GgmlDType::Q6K,
        }
    }

    fn impl_id(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle => "qmatmul_q8_0_mmq_oracle_gfx906",
            Dtype::Q8_04Warp => "qmatmul_q8_0_mmq_4warp_lds_gfx906",
            Dtype::Q4K4Warp => "qmatmul_q4_K_mmq_4warp_lds_gfx906",
            Dtype::Q6K4Warp => "qmatmul_q6_K_mmq_4warp_lds_gfx906",
        }
    }

    fn kernel_stem(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle => "mmq_q8_0_oracle",
            Dtype::Q8_04Warp => "mmq_q8_0_4warp",
            Dtype::Q4K4Warp => "mmq_q4_K_4warp",
            Dtype::Q6K4Warp => "mmq_q6_K_4warp",
        }
    }

    fn kernel_entry(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle => "flambeau_mmq_q8_0_oracle_q8_1",
            Dtype::Q8_04Warp => "flambeau_mmq_q8_0_4warp_q8_1",
            Dtype::Q4K4Warp => "flambeau_mmq_q4_K_4warp_q8_1",
            Dtype::Q6K4Warp => "flambeau_mmq_q6_K_4warp_q8_1",
        }
    }

    fn launch_threads(self) -> u32 {
        match self {
            Dtype::Q8_0Oracle | Dtype::Q8_04Warp => 256,
            Dtype::Q4K4Warp | Dtype::Q6K4Warp => 128,
        }
    }

    fn block_elem_count(self) -> usize {
        self.ggml().block_size()
    }

    fn weight_block_bytes(self) -> usize {
        match self {
            Dtype::Q8_0Oracle | Dtype::Q8_04Warp => std::mem::size_of::<BlockQ8_0>(),
            Dtype::Q4K4Warp => std::mem::size_of::<BlockQ4K>(),
            Dtype::Q6K4Warp => std::mem::size_of::<BlockQ6K>(),
        }
    }

    /// Short tag for the CLI to log alongside the dtype (e.g. "oracle" or "4warp_lds").
    pub fn impl_id_short(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle => "oracle",
            Dtype::Q8_04Warp => "4warp_lds",
            Dtype::Q4K4Warp => "4warp_lds",
            Dtype::Q6K4Warp => "4warp_lds",
        }
    }
}

/// One shape in the grid. `m` = batch rows, `k` = contracted dim, `n` = output features.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct SweepSpec {
    pub dtype: Dtype,
    pub shapes: Vec<Shape>,
    pub seed: u64,
}

impl SweepSpec {
    /// V1.4 oracle grid — small enough that the single-wave-per-output kernel
    /// finishes in a reasonable wall-clock. Once the 4-warp LDS-tiled kernel
    /// lands, the grid grows to `M ∈ {128, 512, 2048}` × full K × N.
    pub fn v1_4_oracle(dtype: Dtype) -> Self {
        // K, N chosen from Qwen3.6 attention + FFN widths; M small so the
        // oracle's M×N single-wave launches stay under a few seconds total.
        let shapes = vec![
            Shape { m: 4,  k: 2048, n: 2048 },
            Shape { m: 8,  k: 2048, n: 2048 },
            Shape { m: 16, k: 2048, n: 2048 },
            Shape { m: 4,  k: 5120, n: 5120 },
            Shape { m: 8,  k: 5120, n: 5120 },
            Shape { m: 4,  k: 5120, n: 15360 }, // FFN up
            Shape { m: 4,  k: 15360, n: 5120 }, // FFN down
        ];
        Self { dtype, shapes, seed: 0xC0FFEE }
    }

    /// Prefill-scale grid for the first-class 4-warp LDS-tiled kernel.
    /// Matches the roadmap's `M ∈ {128, 512, 2048}` × Qwen3.6 attention /
    /// MoE widths.
    pub fn v1_4_prefill(dtype: Dtype) -> Self {
        let shapes = vec![
            Shape { m: 128,  k: 2048, n: 2048 },
            Shape { m: 128,  k: 5120, n: 5120 },
            Shape { m: 128,  k: 5120, n: 15360 },
            Shape { m: 128,  k: 15360, n: 5120 },
            Shape { m: 512,  k: 2048, n: 2048 },
            Shape { m: 512,  k: 5120, n: 5120 },
            Shape { m: 2048, k: 2048, n: 2048 },
        ];
        Self { dtype, shapes, seed: 0xC0FFEE }
    }
}

pub fn run_sweep(spec: &SweepSpec, repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices on this host — V1.4 sweep needs gfx906");
    }

    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let pmc = capture_static_pmc(&dev, spec.dtype)?;

    let mut results = Vec::new();
    for sh in &spec.shapes {
        let seed = spec
            .seed
            .wrapping_add((sh.m as u64).wrapping_mul(0x1234567))
            .wrapping_add((sh.k as u64).wrapping_mul(0x9E3779B97F4A7C15))
            .wrapping_add((sh.n as u64).wrapping_mul(0xDEADBEEFCAFEBABE));
        let (got, reference) = run_shape(&dev, spec.dtype, sh.m, sh.k, sh.n, seed)?;
        let max_rel_err = max_rel_err(&got, &reference, sh.k);
        let tolerance = cert_tol(sh.k);
        results.push(ShapeResult {
            m: sh.m,
            k: sh.k,
            n: sh.n,
            seed,
            max_rel_err,
            tolerance,
            pass: max_rel_err <= tolerance,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_mmq",
            dtype = spec.dtype.name(),
            m = sh.m, k = sh.k, n = sh.n,
            max_rel_err, tolerance,
            "shape result"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = format!(
        "{}-gfx906",
        hostname().unwrap_or_else(|| "unknown".into())
    );

    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: spec.dtype.impl_id().to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul_mmq".to_string(),
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
        target: "flambeau_bench::sweep_mmq",
        cert = %written.display(),
        pass = cert.pass,
        "cert written"
    );
    Ok(cert)
}

fn capture_static_pmc(dev: &HipDevice, dtype: Dtype) -> Result<PmcSnapshot> {
    let bytes = kernels::hsaco(dtype.kernel_stem())
        .ok_or_else(|| anyhow::anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel<'_> = module.kernel(dtype.kernel_entry())?;
    let attrs: FuncAttributes = kernel.attributes()?;
    Ok(PmcSnapshot {
        vgpr_count: Some(attrs.num_regs),
        sgpr_count: None,
        waves_per_simd: Some(attrs.gfx906_waves_per_simd()),
        mem_busy_pct: None,
        valu_busy_pct: None,
    })
}

fn run_shape(
    dev: &HipDevice,
    dtype: Dtype,
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    assert_eq!(k % dtype.block_elem_count(), 0);
    let nb_per_row = k / dtype.block_elem_count();

    // Kernels.
    let q_bytes = kernels::hsaco("quantize_q8_1").unwrap();
    let m_bytes = kernels::hsaco(dtype.kernel_stem()).unwrap();
    let q_module = HipModule::load(dev.id(), q_bytes)?;
    let m_module = HipModule::load(dev.id(), m_bytes)?;
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;
    let k_mmq: HipKernel<'_> = m_module.kernel(dtype.kernel_entry())?;

    // Weights: [n, k] weight blocks (size depends on dtype).
    let weights_raw = seeded_bytes(seed, n * nb_per_row * dtype.weight_block_bytes());
    let weights_raw = tame_weight_scales(dtype, weights_raw);

    // Activation: [m, k] F32.
    let act_f32 = seeded_f32(seed.wrapping_add(0xA5A5A5A5), m * k);

    // CPU dequantise weights for the reference.
    let mut w_dequant = vec![0.0f32; n * k];
    dequantize_into(dtype.ggml(), &weights_raw, &mut w_dequant)?;

    // Device allocs + upload.
    let d_x = alloc_and_upload_bytes(dev, &weights_raw);
    let d_act = alloc_and_upload_bytes(dev, bytemuck::cast_slice(&act_f32));

    // Q8_1 activation block count is m * k/QK8 regardless of weight dtype —
    // Q8_1's block-elem count (32) is independent of the weight super-block.
    assert_eq!(k % QK8, 0);
    let total_y_blocks = m * k / QK8;
    let d_y_q8_1 = dev.alloc(total_y_blocks * std::mem::size_of::<BlockQ8_1>())?;

    // Quantise [m, k] → [m*nb_per_row] Q8_1 blocks (flat, same layout as
    // contiguous-per-batch-row reads in the oracle kernel).
    {
        let stream = dev.default_stream();
        let n_elems = (m * k) as i32;
        let d_act_ptr: u64 = d_act.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_act_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(total_y_blocks as u32, QK8 as u32);
        unsafe { k_quantize.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // MMQ launch — 2D grid (n_rows, n_batches).
    let d_dst = dev.alloc(m * n * 4)?;
    {
        let stream = dev.default_stream();
        let n_rows_i = n as i32;
        let n_batches_i = m as i32;
        let n_bpr_i = nb_per_row as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&d_dst_ptr);
        args.push(&n_rows_i);
        args.push(&n_batches_i);
        args.push(&n_bpr_i);
        let (rows_per_block, batches_per_block) = tile_shape(dtype);
        let grid_x = (n as u32).div_ceil(rows_per_block);
        let grid_y = (m as u32).div_ceil(batches_per_block);
        let cfg = LaunchCfg {
            grid: (grid_x, grid_y, 1),
            block: (dtype.launch_threads(), 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_mmq.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Download result [m, n].
    let mut dst = vec![0.0f32; m * n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(dst.as_mut_ptr() as usize),
            d_dst,
            m * n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, total_y_blocks * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_dst, m * n * 4)?;
    }

    // Reference: dequant_weights × (Q8_1-round-trip activation). Shape [m, n].
    let act_rt = quantize_q8_1_roundtrip(&act_f32);
    let reference = reference_matmul(&w_dequant, &act_rt, m, k, n);
    Ok((dst, reference))
}

// ---- helpers (mostly copied from sweep_mmvq; dedupe into a util module
// once more than one sweep lands) ----

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 24) as u8
        })
        .collect()
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn tame_weight_scales(dtype: Dtype, mut raw: Vec<u8>) -> Vec<u8> {
    let bs = dtype.weight_block_bytes();
    let nblocks = raw.len() / bs;
    for i in 0..nblocks {
        let block = &mut raw[i * bs..(i + 1) * bs];
        match dtype {
            Dtype::Q8_0Oracle | Dtype::Q8_04Warp => {
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q4K4Warp => {
                // Q4_K: d(fp16), dmin(fp16), scales[12], qs[128]. Tame d/dmin
                // so the reference stays well-conditioned (same shaping the
                // V1.3 MMVQ sweep uses).
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let dmin = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Dtype::Q6K4Warp => {
                // Q6_K: ql[128], qh[64], scales[16] (i8), d(fp16) at offset 208.
                // Clamp scales into a small signed range and set d modest.
                for s in 0..16 {
                    let raw = block[192 + s] as i8;
                    // Scale down into ±32 so scale × (raw_q ∈ -32..31) stays
                    // well-behaved before d applies.
                    block[192 + s] = ((raw as i32) >> 2) as u8;
                }
                let d = f16::from_f32((block[208] as f32 / 255.0) * 0.05 + 0.005);
                block[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
            }
        }
    }
    let _ = QK_K;
    raw
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

fn reference_matmul(weights: &[f32], act: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for b in 0..m {
        let act_row = &act[b * k..(b + 1) * k];
        for row in 0..n {
            let w_row = &weights[row * k..(row + 1) * k];
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += (w_row[j] * act_row[j]) as f64;
            }
            out[b * n + row] = acc as f32;
        }
    }
    out
}

fn max_rel_err(got: &[f32], reference: &[f32], k: usize) -> f32 {
    let abs_floor = (k as f32).sqrt();
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(abs_floor))
        .fold(0.0f32, f32::max)
}

fn cert_tol(_k: usize) -> f32 {
    3e-2
}

fn alloc_and_upload_bytes(dev: &HipDevice, data: &[u8]) -> DevicePtr {
    let d = dev.alloc(data.len()).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            data.len(),
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn hostname() -> Option<String> {
    std::env::var("HOSTNAME").ok().or_else(|| {
        let mut buf = vec![0u8; 256];
        let rv = unsafe { libc_gethostname(buf.as_mut_ptr() as *mut _, buf.len()) };
        if rv != 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        buf.truncate(end);
        String::from_utf8(buf).ok()
    })
}

extern "C" {
    #[link_name = "gethostname"]
    fn libc_gethostname(name: *mut std::os::raw::c_char, len: usize) -> i32;
}
