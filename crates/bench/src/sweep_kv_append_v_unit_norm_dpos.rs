//! Device-position fused V-unit-norm + KV append correctness sweep.
//! The `_dpos` kernels read the write row from a device buffer (`pos_ptr[0]`)
//! instead of a host scalar, so a captured decode graph advances the KV write
//! cursor with no rebind. Covers all four head_dim instantiations and both the
//! absolute (ring_depth == 0) and ring-wrapped write-position cases. Checked
//! against an F32 reference: K is a direct copy, V is per-head unit-RMSNorm.
//!
//! Backend-generic; emits the CUDA (sm_86) cert. CUDA-only kernel.

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};

const TOL: f32 = 2e-2;
const TOL_FORMULA: &str = "|err| <= 2e-2 * max(|ref|, 1)";
const EPS: f32 = 1e-6;

struct Case {
    head_dim: usize,
    n_kv_heads: usize,
    position: usize,
    /// Ring slab depth in rows (0 = absolute write position).
    ring_depth: usize,
}

const CASES: &[Case] = &[
    Case { head_dim: 64, n_kv_heads: 4, position: 300, ring_depth: 0 },
    Case { head_dim: 128, n_kv_heads: 8, position: 511, ring_depth: 0 },
    Case { head_dim: 256, n_kv_heads: 8, position: 1641, ring_depth: 0 },
    Case { head_dim: 512, n_kv_heads: 4, position: 2126, ring_depth: 0 },
    // Ring-wrapped: write lands at position % ring_depth.
    Case { head_dim: 256, n_kv_heads: 8, position: 4000, ring_depth: 1152 },
    Case { head_dim: 256, n_kv_heads: 8, position: 1200, ring_depth: 1152 },
];

fn entry(head_dim: usize) -> Option<&'static str> {
    match head_dim {
        64 => Some("flambeau_kv_append_v_unit_norm_f16_dpos_d64"),
        128 => Some("flambeau_kv_append_v_unit_norm_f16_dpos_d128"),
        256 => Some("flambeau_kv_append_v_unit_norm_f16_dpos_d256"),
        512 => Some("flambeau_kv_append_v_unit_norm_f16_dpos_d512"),
        _ => None,
    }
}

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("kv_append_v_unit_norm_f16_dpos")
        .ok_or_else(|| anyhow!("kv_append_v_unit_norm_f16_dpos not compiled (CUDA-only)"))?;
    let module = B::load(&dev, bytes)?;
    let mut pmc = None;

    let mut results = Vec::new();
    for case in CASES {
        let name = entry(case.head_dim).ok_or_else(|| anyhow!("bad head_dim"))?;
        let kernel = B::kernel(&module, name)?;
        if pmc.is_none() {
            pmc = Some(B::attributes(&kernel)?);
        }
        let seed = 0x7A9E ^ (case.head_dim as u64 * 7919) ^ (case.position as u64 * 101)
            ^ (case.ring_depth as u64 * 13);
        let (got, reference) = run_case::<B>(&dev, &kernel, case, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult {
            m: case.n_kv_heads,
            k: case.position,
            n: case.head_dim,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: max_rel <= TOL,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: format!("kv_append_v_unit_norm_f16_dpos_{arch}"),
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

#[allow(clippy::undocumented_unsafe_blocks)]
fn run_case<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    case: &Case,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let Case { head_dim, n_kv_heads, position, ring_depth } = *case;
    let kv_width = n_kv_heads * head_dim;
    let write_row = if ring_depth > 0 { position % ring_depth } else { position };
    let cache_rows = if ring_depth > 0 { ring_depth } else { position + 1 };

    let k_src: Vec<f16> = seeded_f32_range(seed ^ 0x11, kv_width, -0.5, 0.5)
        .iter().map(|v| f16::from_f32(*v)).collect();
    let v_src: Vec<f16> = seeded_f32_range(seed ^ 0x22, kv_width, -0.8, 0.8)
        .iter().map(|v| f16::from_f32(*v)).collect();
    let pos_i: Vec<i32> = vec![position as i32];

    let d_k_src = alloc_and_upload(dev, &k_src);
    let d_v_src = alloc_and_upload(dev, &v_src);
    let d_pos = alloc_and_upload(dev, &pos_i);
    let cache_n = cache_rows * kv_width;
    let d_k_cache = dev.alloc(cache_n * 2)?;
    let d_v_cache = dev.alloc(cache_n * 2)?;

    {
        let stream = dev.default_stream();
        let n_kv_heads_i = n_kv_heads as i32;
        let ring_depth_i = ring_depth as i32;
        let ks = d_k_src.as_usize() as u64;
        let vs = d_v_src.as_usize() as u64;
        let kc = d_k_cache.as_usize() as u64;
        let vc = d_v_cache.as_usize() as u64;
        let pp = d_pos.as_usize() as u64;
        let eps = EPS;
        let mut a = <B::Args<'_>>::default();
        a.push_arg(&ks);
        a.push_arg(&vs);
        a.push_arg(&kc);
        a.push_arg(&vc);
        a.push_arg(&n_kv_heads_i);
        a.push_arg(&eps);
        a.push_arg(&ring_depth_i);
        a.push_arg(&pp);
        let cfg = LaunchCfg {
            grid: (1, n_kv_heads as u32, 1),
            block: (head_dim as u32, 1, 1),
            shared_bytes: 0,
        };
        B::launch(kernel, stream, cfg, a)?;
        stream.synchronize()?;
    }

    // Read back only the written row of K and V.
    let mut k_row: Vec<f16> = vec![f16::from_f32(0.0); kv_width];
    let mut v_row: Vec<f16> = vec![f16::from_f32(0.0); kv_width];
    let row_bytes = kv_width * 2;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(k_row.as_mut_ptr() as usize),
            d_k_cache.offset_bytes(write_row * row_bytes),
            row_bytes,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(v_row.as_mut_ptr() as usize),
            d_v_cache.offset_bytes(write_row * row_bytes),
            row_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_k_src, k_src.len() * 2)?;
        dev.dealloc(d_v_src, v_src.len() * 2)?;
        dev.dealloc(d_pos, pos_i.len() * 4)?;
        dev.dealloc(d_k_cache, cache_n * 2)?;
        dev.dealloc(d_v_cache, cache_n * 2)?;
    }

    // got = [K row | V row]; reference = [copy(K) | unit-norm(V)].
    let mut got = Vec::with_capacity(2 * kv_width);
    got.extend(k_row.iter().map(|v| v.to_f32()));
    got.extend(v_row.iter().map(|v| v.to_f32()));

    let mut reference = Vec::with_capacity(2 * kv_width);
    reference.extend(k_src.iter().map(|v| f16::from_f32(v.to_f32()).to_f32()));
    for h in 0..n_kv_heads {
        let base = h * head_dim;
        let mut sq = 0.0f64;
        for d in 0..head_dim {
            let x = v_src[base + d].to_f32();
            sq += (x * x) as f64;
        }
        let inv_rms = 1.0f32 / ((sq as f32) / head_dim as f32 + EPS).sqrt();
        for d in 0..head_dim {
            let x = v_src[base + d].to_f32();
            reference.push(f16::from_f32(x * inv_rms).to_f32());
        }
    }
    Ok((got, reference))
}
