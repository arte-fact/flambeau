//! Device-position F16 KV append correctness sweep. The `_dpos` kernel reads
//! the write row from a device buffer (`pos_ptr[0]`) instead of a host scalar,
//! so a captured decode graph advances the KV write cursor with no rebind.
//! Covers the absolute (ring_depth == 0) and ring-wrapped write positions.
//! The append is a copy: the written K and V rows must equal the sources bit
//! for bit, and every other cache row must stay untouched.
//!
//! Backend-generic; emits the CUDA (sm_86) cert. CUDA-only kernel.

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, rig_for, seeded_f32_range};

const TOL: f32 = 0.0;
const TOL_FORMULA: &str = "bit-exact copy of the K/V rows; other rows untouched";
const SENTINEL: u16 = 0x7BFF;

struct Case {
    kv_width: usize,
    position: usize,
    /// Ring slab depth in rows (0 = absolute write position).
    ring_depth: usize,
}

const CASES: &[Case] = &[
    Case { kv_width: 1024, position: 32, ring_depth: 0 },
    Case { kv_width: 1024, position: 300, ring_depth: 0 },
    Case { kv_width: 1024, position: 1641, ring_depth: 0 },
    Case { kv_width: 2048, position: 511, ring_depth: 0 },
    Case { kv_width: 512, position: 77, ring_depth: 0 },
    // Ring-wrapped: write lands at position % ring_depth.
    Case { kv_width: 1024, position: 4000, ring_depth: 1152 },
    Case { kv_width: 2048, position: 1200, ring_depth: 1152 },
];

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("kv_append_f16_dpos")
        .ok_or_else(|| anyhow!("kv_append_f16_dpos not compiled (CUDA-only)"))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, "flambeau_kv_append_f16_dpos")?;
    let pmc = Some(B::attributes(&kernel)?);

    let mut results = Vec::new();
    for case in CASES {
        let seed = 0x5D3A ^ (case.kv_width as u64 * 7919) ^ (case.position as u64 * 101)
            ^ (case.ring_depth as u64 * 13);
        let mismatches = run_case::<B>(&dev, &kernel, case, seed)?;
        let err = mismatches as f32;
        results.push(ShapeResult {
            m: case.ring_depth,
            k: case.position,
            n: case.kv_width,
            seed,
            max_rel_err: err,
            tolerance: TOL,
            pass: mismatches == 0,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: format!("kv_append_f16_dpos_{arch}"),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "kv_append".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: TOL_FORMULA.to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// Number of cache elements (K and V slabs) that differ from the expected
/// state: the source row at the write row, the sentinel everywhere else.
#[allow(clippy::undocumented_unsafe_blocks)]
fn run_case<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    case: &Case,
    seed: u64,
) -> Result<usize> {
    let Case { kv_width, position, ring_depth } = *case;
    let write_row = if ring_depth > 0 { position % ring_depth } else { position };
    let cache_rows = if ring_depth > 0 { ring_depth } else { position + 1 };
    let cache_n = cache_rows * kv_width;

    let k_src: Vec<f16> = seeded_f32_range(seed ^ 0x11, kv_width, -4.0, 4.0)
        .iter().map(|v| f16::from_f32(*v)).collect();
    let v_src: Vec<f16> = seeded_f32_range(seed ^ 0x22, kv_width, -4.0, 4.0)
        .iter().map(|v| f16::from_f32(*v)).collect();
    let pos_i: Vec<i32> = vec![position as i32];
    let fill: Vec<f16> = vec![f16::from_bits(SENTINEL); cache_n];

    let d_k_src = alloc_and_upload(dev, &k_src);
    let d_v_src = alloc_and_upload(dev, &v_src);
    let d_pos = alloc_and_upload(dev, &pos_i);
    let d_k_cache = alloc_and_upload(dev, &fill);
    let d_v_cache = alloc_and_upload(dev, &fill);

    {
        let stream = dev.default_stream();
        let kv_width_i = kv_width as i32;
        let ring_depth_i = ring_depth as i32;
        let ks = d_k_src.as_usize() as u64;
        let vs = d_v_src.as_usize() as u64;
        let kc = d_k_cache.as_usize() as u64;
        let vc = d_v_cache.as_usize() as u64;
        let pp = d_pos.as_usize() as u64;
        let mut a = <B::Args<'_>>::default();
        a.push_arg(&ks);
        a.push_arg(&vs);
        a.push_arg(&kc);
        a.push_arg(&vc);
        a.push_arg(&kv_width_i);
        a.push_arg(&ring_depth_i);
        a.push_arg(&pp);
        let cfg = LaunchCfg {
            grid: (kv_width.div_ceil(256) as u32, 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        B::launch(kernel, stream, cfg, a)?;
        stream.synchronize()?;
    }

    let mut k_cache: Vec<f16> = vec![f16::from_f32(0.0); cache_n];
    let mut v_cache: Vec<f16> = vec![f16::from_f32(0.0); cache_n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(k_cache.as_mut_ptr() as usize),
            d_k_cache,
            cache_n * 2,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(v_cache.as_mut_ptr() as usize),
            d_v_cache,
            cache_n * 2,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_k_src, kv_width * 2)?;
        dev.dealloc(d_v_src, kv_width * 2)?;
        dev.dealloc(d_pos, 4)?;
        dev.dealloc(d_k_cache, cache_n * 2)?;
        dev.dealloc(d_v_cache, cache_n * 2)?;
    }

    let mut mismatches = 0;
    for (cache, src) in [(&k_cache, &k_src), (&v_cache, &v_src)] {
        for (i, got) in cache.iter().enumerate() {
            let (row, col) = (i / kv_width, i % kv_width);
            let want = if row == write_row { src[col].to_bits() } else { SENTINEL };
            if got.to_bits() != want {
                mismatches += 1;
            }
        }
    }
    Ok(mismatches)
}
