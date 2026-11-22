//! F16 → Q8_1 few-column MMQ B-fragment layout correctness sweep. The device
//! output of both scale formats (D4, DS4) is compared byte for byte with the
//! host reference [`flambeau_quant::q8_1_few::quantize_f16_q8_1_few`], which
//! runs the kernel's IEEE f32 arithmetic: any differing byte fails. Shapes:
//! the Qwen3.8-27B verify contractions at 1..24 columns, partial column
//! groups (9, 17) included.
//!
//! Backend-generic; emits the CUDA (sm_86) cert. CUDA-only kernel.

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use flambeau_quant::q8_1_few::{few_bytes, quantize_f16_q8_1_few, FewScale, FEW_NCOLS};
use flambeau_quant::QK_K;
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, rig_for, seeded_f32_range};

const TOL_FORMULA: &str =
    "bytes differing from the host reference (q8_1_few::quantize_f16_q8_1_few) / total bytes == 0";
const KS: &[usize] = &[2048, 5120, 6144, 17408];
const MS: &[usize] = &[1, 2, 5, 8, 9, 16, 17, 24];

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("quantize_f16_q8_1_few")
        .ok_or_else(|| anyhow!("quantize_f16_q8_1_few not compiled (CUDA-only)"))?;
    let module = B::load(&dev, bytes)?;
    let d4 = B::kernel(&module, "flambeau_quantize_f16_q8_1_few_d4")?;
    let ds4 = B::kernel(&module, "flambeau_quantize_f16_q8_1_few_ds4")?;
    let pmc = B::attributes(&ds4)?;

    let mut results = Vec::new();
    for (scale, kernel, n) in [(FewScale::D4, &d4, 4usize), (FewScale::Ds4, &ds4, 2)] {
        for &k in KS {
            for &m in MS {
                let seed = 0xF3_u64 ^ (k as u64 * 7919) ^ (m as u64 * 104_729) ^ n as u64;
                let differing = run_case::<B>(&dev, kernel, scale, k, m, seed)?;
                let frac = differing as f32 / few_bytes(k, m) as f32;
                results.push(ShapeResult {
                    m,
                    k,
                    n,
                    seed,
                    max_rel_err: frac,
                    tolerance: 0.0,
                    pass: differing == 0,
                });
            }
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: format!("quantize_f16_q8_1_few_{arch}"),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "quantize_f16_q8_1_few".to_string(),
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

/// Number of output bytes that differ from the host reference.
#[allow(clippy::undocumented_unsafe_blocks)]
fn run_case<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    scale: FewScale,
    k: usize,
    m: usize,
    seed: u64,
) -> Result<usize> {
    let src_f32 = seeded_f32_range(seed, k * m, -2.0, 2.0);
    let mut src: Vec<f16> = src_f32.iter().map(|v| f16::from_f32(*v)).collect();
    // One all-zero sub-block exercises the amax == 0 record.
    src[..32].fill(f16::ZERO);
    let reference = quantize_f16_q8_1_few(&src, k, m, scale);

    let out_bytes = few_bytes(k, m);
    let d_src = alloc_and_upload(dev, &src);
    let d_out = alloc_and_upload(dev, &vec![0xA5u8; out_bytes]);
    {
        let stream = dev.default_stream();
        let (k_i, m_i) = (k as i32, m as i32);
        let x = d_src.as_usize() as u64;
        let y = d_out.as_usize() as u64;
        let mut a = <B::Args<'_>>::default();
        a.push_arg(&x);
        a.push_arg(&y);
        a.push_arg(&k_i);
        a.push_arg(&m_i);
        let cfg = LaunchCfg {
            grid: (m.next_multiple_of(FEW_NCOLS) as u32, (k / QK_K) as u32, 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        B::launch(kernel, stream, cfg, a)?;
        stream.synchronize()?;
    }
    let mut got = vec![0u8; out_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_out,
            out_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_src, src.len() * 2)?;
        dev.dealloc(d_out, out_bytes)?;
    }
    Ok(got.iter().zip(&reference).filter(|(a, b)| a != b).count())
}
