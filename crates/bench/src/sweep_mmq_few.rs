//! Few-column int8 MMA (`mmq_few_{q4_k,q5_k,q3_k}_mma`) correctness sweep. The
//! activation is quantised on the host into the few-column layout
//! ([`flambeau_quant::q8_1_few`], with the scale record the type reads), so
//! the cert isolates the matmul; the reference is an f64 dot product of the
//! dequantised weights with the dequantised activation. Columns 1..24 (every
//! column-group count, partial groups included), K at 8..68 super-blocks, a
//! row count off the 16-row tile; Q3_K rows are not 16-byte aligned.
//!
//! Backend-generic; emits the CUDA (sm_86) certs. CUDA-only kernels.

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use flambeau_quant::q8_1_few::{
    few_ds_offset, few_qs_offset, quantize_f16_q8_1_few, FewScale, FEW_NCOLS, FEW_UNIT_GROUP_BYTES,
};
use flambeau_quant::{dequantize_to_vec, BlockQ3K, BlockQ4K, BlockQ5K, GgmlDType, QK8_1, QK_K};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};

const TOL: f32 = 3e-2;
const TOL_FORMULA: &str =
    "|err| <= 3e-2 * max(|ref|, sqrt(k))  (ref: f64 over dequantised operands)";
/// (weight rows, k)
const SHAPES: &[(usize, usize)] = &[(1000, 5120), (512, 17408), (2048, 2048), (256, 6144)];
const MS: &[usize] = &[1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 24];

/// A weight type of the few-column MMA.
#[derive(Debug, Clone, Copy)]
struct FewType {
    name: &'static str,
    stem: &'static str,
    entries: [&'static str; 3],
    impl_id: &'static str,
    ggml: GgmlDType,
    block_bytes: usize,
    scale: FewScale,
}

const Q4_K: FewType = FewType {
    name: "Q4_K",
    stem: "mmq_few_q4_k_mma",
    entries: [
        "flambeau_mmq_few_q4_k_mma_c1_t1",
        "flambeau_mmq_few_q4_k_mma_c2_t1",
        "flambeau_mmq_few_q4_k_mma_c3_t1",
    ],
    impl_id: "qmatmul_q4_k_mmq_few_mma",
    ggml: GgmlDType::Q4K,
    block_bytes: std::mem::size_of::<BlockQ4K>(),
    scale: FewScale::Ds4,
};

const Q5_K: FewType = FewType {
    name: "Q5_K",
    stem: "mmq_few_q5_k_mma",
    entries: [
        "flambeau_mmq_few_q5_k_mma_c1_t1",
        "flambeau_mmq_few_q5_k_mma_c2_t1",
        "flambeau_mmq_few_q5_k_mma_c3_t1",
    ],
    impl_id: "qmatmul_q5_k_mmq_few_mma",
    ggml: GgmlDType::Q5K,
    block_bytes: std::mem::size_of::<BlockQ5K>(),
    scale: FewScale::Ds4,
};

const Q3_K: FewType = FewType {
    name: "Q3_K",
    stem: "mmq_few_q3_k_mma",
    entries: [
        "flambeau_mmq_few_q3_k_mma_c1_t1",
        "flambeau_mmq_few_q3_k_mma_c2_t1",
        "flambeau_mmq_few_q3_k_mma_c3_t1",
    ],
    impl_id: "qmatmul_q3_k_mmq_few_mma",
    ggml: GgmlDType::Q3K,
    block_bytes: std::mem::size_of::<BlockQ3K>(),
    scale: FewScale::D4,
};

/// Q4_K sweep.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep_q4_k<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run::<B>(Q4_K, repo_root, arch)
}

/// Q5_K sweep.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep_q5_k<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run::<B>(Q5_K, repo_root, arch)
}

/// Q3_K sweep.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep_q3_k<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run::<B>(Q3_K, repo_root, arch)
}

fn run<B: BenchBackend>(ty: FewType, repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image(ty.stem).ok_or_else(|| anyhow!("{} not compiled (CUDA-only)", ty.stem))?;
    let module = B::load(&dev, bytes)?;
    let kernels = ty.entries.map(|e| B::kernel(&module, e));
    let pmc = B::attributes(kernels[1].as_ref().map_err(|e| anyhow!("{e:?}"))?)?;

    let mut results = Vec::new();
    for &(n, k) in SHAPES {
        let w_seed = 0x4B_u64 ^ (n as u64 * 7919) ^ (k as u64 * 104_729);
        let weights = weights(ty, w_seed, n, k);
        let w_f32 = dequantize_to_vec(ty.ggml, &weights, n * k)?;
        for &m in MS {
            let seed = w_seed ^ (m as u64 * 31);
            let kernel =
                kernels[m.div_ceil(FEW_NCOLS) - 1].as_ref().map_err(|e| anyhow!("{e:?}"))?;
            let (got, reference) =
                run_case::<B>(&dev, kernel, ty, &weights, &w_f32, n, k, m, seed)?;
            let max_rel = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
            results.push(ShapeResult {
                m,
                k,
                n,
                seed,
                max_rel_err: max_rel,
                tolerance: TOL,
                pass: max_rel <= TOL,
            });
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: ty.impl_id.to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_mmq_few".to_string(),
        dtype_weight: ty.name.to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: TOL_FORMULA.to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// `n` rows of seeded super-blocks of `ty` with tamed scales (the shaping the
/// MMQ sweep uses).
fn weights(ty: FewType, seed: u64, n: usize, k: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut raw: Vec<u8> = (0..n * (k / QK_K) * ty.block_bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect();
    for b in raw.chunks_exact_mut(ty.block_bytes) {
        match ty.ggml {
            GgmlDType::Q3K => {
                // hmask[32], qs[64], scales[12], d: 6-bit scales kept small.
                for s in &mut b[96..108] {
                    *s %= 32;
                }
                let d = f16::from_f32((f32::from(b[108]) / 255.0) * 0.05 + 0.005);
                b[108..110].copy_from_slice(&d.to_le_bytes());
            }
            _ => {
                // Q4_K / Q5_K: d, dmin first.
                let d = f16::from_f32((f32::from(b[0]) / 255.0) * 0.1 + 0.01);
                let dmin = f16::from_f32((f32::from(b[1]) / 255.0) * 0.05);
                b[0..2].copy_from_slice(&d.to_le_bytes());
                b[2..4].copy_from_slice(&dmin.to_le_bytes());
            }
        }
    }
    raw
}

/// The activation values the few layout `y` encodes (`q · d`), row-major
/// `[m][k]`.
fn dequantize_few(y: &[u8], k: usize, m: usize, scale: FewScale) -> Vec<f32> {
    let n_groups = m.div_ceil(FEW_NCOLS);
    let mut out = vec![0.0f32; m * k];
    for col in 0..m {
        let (gi, g) = (col / FEW_NCOLS, col % FEW_NCOLS);
        for i in 0..k {
            let (unit, s, kk) = (i / QK_K, (i % QK_K) / QK8_1, i % QK8_1);
            let rec = (unit * n_groups + gi) * FEW_UNIT_GROUP_BYTES;
            let q = y[rec + few_qs_offset(s, kk - kk % 4, g) + kk % 4] as i8;
            let at = rec + few_ds_offset(s, g);
            let d = match scale {
                FewScale::D4 => f32::from_le_bytes([y[at], y[at + 1], y[at + 2], y[at + 3]]),
                FewScale::Ds4 => f16::from_le_bytes([y[at], y[at + 1]]).to_f32(),
            };
            out[col * k + i] = f32::from(q) * d;
        }
    }
    out
}

/// Ring row stride of the kernel (`mmq_few_mma.cuh` `sram_stride`).
fn sram_stride(unit_bytes: usize, units: usize) -> usize {
    let bytes = units * unit_bytes;
    if bytes % 16 == 0 {
        return bytes;
    }
    let chunks = (14 + bytes + 15) / 16;
    16 * if chunks % 2 == 0 { chunks + 1 } else { chunks }
}

#[allow(clippy::too_many_arguments, clippy::undocumented_unsafe_blocks)]
fn run_case<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    ty: FewType,
    weights: &[u8],
    w_f32: &[f32],
    n: usize,
    k: usize,
    m: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let x: Vec<f16> =
        seeded_f32_range(seed, m * k, -1.0, 1.0).into_iter().map(f16::from_f32).collect();
    let y = quantize_f16_q8_1_few(&x, k, m, ty.scale);
    let y_f32 = dequantize_few(&y, k, m, ty.scale);

    let mut reference = vec![0.0f32; m * n];
    for col in 0..m {
        let a = &y_f32[col * k..(col + 1) * k];
        for row in 0..n {
            let w = &w_f32[row * k..(row + 1) * k];
            let dot: f64 = w.iter().zip(a).map(|(&w, &a)| f64::from(w) * f64::from(a)).sum();
            reference[col * n + row] = dot as f32;
        }
    }

    let d_w = alloc_and_upload(dev, weights);
    let d_y = alloc_and_upload(dev, &y);
    let d_dst = alloc_and_upload(dev, &vec![f32::NAN; m * n]);
    let ncg = m.div_ceil(FEW_NCOLS);
    let units_per_stage = if ncg > 2 { 1 } else { 2 };
    {
        let stream = dev.default_stream();
        let nunits = (k / QK_K) as i32;
        let (n_i, m_i) = (n as i32, m as i32);
        let stride_row_x = ((k / QK_K) * ty.block_bytes) as i32;
        let w = d_w.as_usize() as u64;
        let yp = d_y.as_usize() as u64;
        let dp = d_dst.as_usize() as u64;
        let mut a = <B::Args<'_>>::default();
        a.push_arg(&w);
        a.push_arg(&yp);
        a.push_arg(&dp);
        a.push_arg(&nunits);
        a.push_arg(&n_i);
        a.push_arg(&m_i);
        a.push_arg(&stride_row_x);
        a.push_arg(&n_i);
        let cfg = LaunchCfg {
            grid: (n.div_ceil(16) as u32, 1, 1),
            block: (32, 4, 1),
            shared_bytes: (4 * 2 * 16 * sram_stride(ty.block_bytes, units_per_stage)) as u32,
        };
        B::launch(kernel, stream, cfg, a)?;
        stream.synchronize()?;
    }
    let mut got = vec![0.0f32; m * n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            m * n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, weights.len())?;
        dev.dealloc(d_y, y.len())?;
        dev.dealloc(d_dst, m * n * 4)?;
    }
    Ok((got, reference))
}
