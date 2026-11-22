//! F16 weight × Q8_1 activation MMVQ / MMQ correctness sweep (backend-generic).
//! Unsloth's UD-Q8_K_XL GGUFs reserve F16 for i-matrix-flagged layers. The sweep
//! validates the templated `mmvq_f16` / `mmq_f16` / `mmq_f16_tile` kernels against
//! an F32 reference that dequantises the Q8_1 activation back to F32, multiplies
//! in F32 with the F16 weight upcast, and sums. Shapes cover 27B-UD-Q8_K_XL:
//! * GDN attn_gate: n=6144, k=5120
//! * GDN ssm_out: n=5120, k=6144
//! * synthetic narrow shape for block-count coverage.
//!
//! Generic over `BenchBackend` so it emits HIP (gfx906) + CUDA (sm_86) certs.

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use flambeau_quant::{BlockQ8_1, QK8_1};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};

const SHAPES: &[(usize, usize)] = &[
    (256, 256),     // smoke
    (6144, 5120),   // Qwen3.6-27B-UD-Q8_K_XL attn_gate
    (5120, 6144),   // Qwen3.6-27B-UD-Q8_K_XL ssm_out
    (248320, 5120), // Qwen3.6-27B LM head / embeddings (F16 in UD)
];

const TOL: f32 = 3e-2;
const TOL_FORMULA: &str = "|err| <= 3e-2 * max(|ref|, sqrt(k) * q8_step)";

/// Which F16-weight kernel a sweep targets.
#[derive(Clone, Copy)]
pub enum F16SweepKind {
    Mmvq,
    Mmq,
    MmqTile,
}

impl F16SweepKind {
    pub fn name(self) -> &'static str {
        match self {
            F16SweepKind::Mmvq => "mmvq",
            F16SweepKind::Mmq => "mmq",
            F16SweepKind::MmqTile => "mmq_tile",
        }
    }
}

/// Backend-generic entry point for the three F16-weight kernel sweeps.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_f16_sweep<B: BenchBackend>(
    kind: F16SweepKind,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    match kind {
        F16SweepKind::Mmvq => run_f16_mmvq_sweep::<B>(repo_root, arch),
        F16SweepKind::Mmq => run_f16_mmq_sweep::<B>(repo_root, arch),
        F16SweepKind::MmqTile => run_f16_mmq_tile_sweep::<B>(repo_root, arch),
    }
}

fn pmc_from<B: BenchBackend>(kernel: &B::Kernel<'_>) -> Result<PmcSnapshot> {
    Ok(B::attributes(kernel)?)
}

/// Single-row MMVQ sweep — `flambeau_mmvq_f16_q8_1` at one activation row.
pub fn run_f16_mmvq_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("mmvq_f16_q8_1").ok_or_else(|| anyhow!("mmvq_f16_q8_1 not compiled"))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, "flambeau_mmvq_f16_q8_1")?;
    let pmc = pmc_from::<B>(&kernel)?;

    let mut results = Vec::new();
    for &(n, k) in SHAPES {
        let seed = 0xF16FACE ^ (n as u64 * 7919) ^ (k as u64 * 101);
        let (got, reference) = run_mmvq_shape::<B>(&dev, &kernel, n, k, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        results.push(ShapeResult {
            m: 1,
            k,
            n,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: max_rel <= TOL,
        });
    }
    finish::<B>(&dev, "qmatmul_f16_mmvq", "qmatmul_mmvq", arch, results, pmc, repo_root)
}

/// Multi-row MMQ sweep — `flambeau_mmq_f16_q8_1` (grid.y = n_tokens, launch-overhead amortisation).
pub fn run_f16_mmq_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("mmq_f16_q8_1").ok_or_else(|| anyhow!("mmq_f16_q8_1 not compiled"))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, "flambeau_mmq_f16_q8_1")?;
    let pmc = pmc_from::<B>(&kernel)?;

    let cases = [
        (256usize, 256usize, 1usize),
        (256, 256, 8),
        (6144, 5120, 16),
        (5120, 6144, 8),
    ];
    let mut results = Vec::new();
    for (n, k, n_tokens) in cases {
        let seed = 0x25A0u64 ^ (n as u64 * 7919) ^ (k as u64 * 101) ^ (n_tokens as u64 * 37);
        let (got, reference) = run_mmq_shape::<B>(&dev, &kernel, n, k, n_tokens, seed, false)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        results.push(ShapeResult {
            m: n_tokens,
            k,
            n,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: max_rel <= TOL,
        });
    }
    finish::<B>(&dev, "qmatmul_f16_mmq", "qmatmul", arch, results, pmc, repo_root)
}

/// Tile-M MMQ sweep — `flambeau_mmq_f16_tile_q8_1` (MMQ_Y=64 × TILE_N=8, weight HBM amortisation).
pub fn run_f16_mmq_tile_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("mmq_f16_tile").ok_or_else(|| anyhow!("mmq_f16_tile not compiled"))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, "flambeau_mmq_f16_tile_q8_1")?;
    let pmc = pmc_from::<B>(&kernel)?;

    let cases = [
        (256usize, 256usize, 8usize),
        (256, 256, 5), // under-fill TILE_N=8 → slot-token -1 early-exit
        (6144, 5120, 16),
        (5120, 6144, 8),
    ];
    let mut results = Vec::new();
    for (n, k, n_tokens) in cases {
        let seed = 0x29A0u64 ^ (n as u64 * 7919) ^ (k as u64 * 101) ^ (n_tokens as u64 * 37);
        let (got, reference) = run_mmq_shape::<B>(&dev, &kernel, n, k, n_tokens, seed, true)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        results.push(ShapeResult {
            m: n_tokens,
            k,
            n,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: max_rel <= TOL,
        });
    }
    finish::<B>(&dev, "qmatmul_f16_mmq_tile", "qmatmul", arch, results, pmc, repo_root)
}

#[allow(clippy::too_many_arguments)]
fn finish<B: BenchBackend>(
    dev: &B::Device,
    impl_id: &str,
    op: &str,
    arch: &str,
    results: Vec<ShapeResult>,
    pmc: PmcSnapshot,
    repo_root: &Path,
) -> Result<Cert> {
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: impl_id.to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: op.to_string(),
        dtype_weight: "F16".to_string(),
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

fn quantize_row_q8_1(xs: &[f32]) -> Vec<BlockQ8_1> {
    assert_eq!(xs.len() % QK8_1, 0);
    let nb = xs.len() / QK8_1;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let b = &xs[i * QK8_1..(i + 1) * QK8_1];
        let amax = b.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK8_1];
        let mut sum_i: i32 = 0;
        for (j, &v) in b.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            qs[j] = q;
            sum_i += q as i32;
        }
        out.push(BlockQ8_1 {
            d: f16::from_f32(d),
            s: f16::from_f32(d * sum_i as f32),
            qs,
        });
    }
    out
}

#[allow(clippy::undocumented_unsafe_blocks)]
fn run_mmvq_shape<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    n: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let w_f32 = seeded_f32_range(seed, n * k, -0.5, 0.5);
    let x_f32 = seeded_f32_range(seed.wrapping_add(0xA1), k, -0.5, 0.5);
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_q = quantize_row_q8_1(&x_f32);

    let d_w = alloc_and_upload(dev, &w_f16);
    let d_x = alloc_and_upload(dev, &x_q);
    let d_out = dev.alloc(n * 4)?;

    {
        let stream = dev.default_stream();
        let n_rows_i = n as i32;
        let n_blocks_i = (k / 32) as i32;
        let w_ptr: u64 = d_w.as_usize() as u64;
        let y_ptr: u64 = d_x.as_usize() as u64;
        let o_ptr: u64 = d_out.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&w_ptr);
        args.push_arg(&y_ptr);
        args.push_arg(&o_ptr);
        args.push_arg(&n_rows_i);
        args.push_arg(&n_blocks_i);
        B::launch(kernel, stream, LaunchCfg::one_d(n as u32, 256), args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_out,
            n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_f16.len() * 2)?;
        dev.dealloc(d_x, x_q.len() * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_out, n * 4)?;
    }

    let reference = f16_q8_1_reference(&w_f16, &x_q, n, k, 1);
    Ok((got, reference))
}

#[allow(clippy::undocumented_unsafe_blocks)]
fn run_mmq_shape<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    n: usize,
    k: usize,
    n_tokens: usize,
    seed: u64,
    tile: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let w_f32 = seeded_f32_range(seed, n * k, -0.5, 0.5);
    let x_f32 = seeded_f32_range(seed.wrapping_add(0xA1), n_tokens * k, -0.5, 0.5);
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let mut x_q: Vec<BlockQ8_1> = Vec::with_capacity(n_tokens * (k / QK8_1));
    for row in 0..n_tokens {
        x_q.extend(quantize_row_q8_1(&x_f32[row * k..(row + 1) * k]));
    }

    let d_w = alloc_and_upload(dev, &w_f16);
    let d_x = alloc_and_upload(dev, &x_q);
    let d_out = dev.alloc(n_tokens * n * 4)?;

    {
        let stream = dev.default_stream();
        let n_rows_i = n as i32;
        let n_tokens_i = n_tokens as i32;
        let n_blocks_i = (k / 32) as i32;
        let w_ptr: u64 = d_w.as_usize() as u64;
        let y_ptr: u64 = d_x.as_usize() as u64;
        let o_ptr: u64 = d_out.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&w_ptr);
        args.push_arg(&y_ptr);
        args.push_arg(&o_ptr);
        args.push_arg(&n_rows_i);
        args.push_arg(&n_tokens_i);
        args.push_arg(&n_blocks_i);
        let cfg = if tile {
            LaunchCfg {
                grid: ((n as u32).div_ceil(64), (n_tokens as u32).div_ceil(8), 1),
                block: (64, 1, 1),
                shared_bytes: 0,
            }
        } else {
            LaunchCfg {
                grid: (n as u32, n_tokens as u32, 1),
                block: (256, 1, 1),
                shared_bytes: 0,
            }
        };
        B::launch(kernel, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_out,
            n_tokens * n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_f16.len() * 2)?;
        dev.dealloc(d_x, x_q.len() * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_out, n_tokens * n * 4)?;
    }

    let reference = f16_q8_1_reference(&w_f16, &x_q, n, k, n_tokens);
    Ok((got, reference))
}

/// Reference: F16 weight × Q8_1-dequant activation in F32, `[n_tokens][n]` row-major.
fn f16_q8_1_reference(w_f16: &[f16], x_q: &[BlockQ8_1], n: usize, k: usize, n_tokens: usize) -> Vec<f32> {
    let mut reference = vec![0.0f32; n_tokens * n];
    let w_back: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    for tok in 0..n_tokens {
        let blocks = &x_q[tok * (k / QK8_1)..(tok + 1) * (k / QK8_1)];
        let x_dequant: Vec<f32> = blocks
            .iter()
            .flat_map(|b| {
                let d = b.d.to_f32();
                (0..QK8_1).map(move |i| d * b.qs[i] as f32)
            })
            .collect();
        for row in 0..n {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += (w_back[row * k + j] * x_dequant[j]) as f64;
            }
            reference[tok * n + row] = acc as f32;
        }
    }
    reference
}
