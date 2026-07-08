//! Device-position embedding-row gather correctness sweep. The
//! `embed_gather_f16_dpos` kernel copies one hidden-width row
//! `embd[token_id * hidden + j]` → `dst[j]`, reading `token_id` on-device from
//! `ctx_ptr[1]` (the decode-graph `[position, token_id]` context), so a captured
//! decode graph re-sources the embedding row per token with no memcpy rebind.
//! Checked against the exact embedding row (bit-identical F16 copy).
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

const TOL: f32 = 0.0;
const TOL_FORMULA: &str = "bit-identical F16 row copy (|err| == 0)";

struct Case {
    hidden: usize,
    vocab: usize,
    token_id: usize,
}

const CASES: &[Case] = &[
    Case { hidden: 3840, vocab: 4096, token_id: 0 },
    Case { hidden: 3840, vocab: 4096, token_id: 2731 },
    Case { hidden: 3840, vocab: 4096, token_id: 4095 },
    Case { hidden: 2048, vocab: 1000, token_id: 511 },
    Case { hidden: 5376, vocab: 512, token_id: 137 },
];

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("embed_gather_f16_dpos")
        .ok_or_else(|| anyhow!("embed_gather_f16_dpos not compiled (CUDA-only)"))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, "flambeau_embed_gather_f16_dpos")?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    for case in CASES {
        let seed = 0xE3B ^ (case.hidden as u64 * 7919) ^ (case.token_id as u64 * 101);
        let (got, reference) = run_case::<B>(&dev, &kernel, case, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult {
            m: 1,
            k: case.token_id,
            n: case.hidden,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: got == reference,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: format!("embed_gather_f16_dpos_{arch}"),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "embed".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
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

#[allow(clippy::undocumented_unsafe_blocks)]
fn run_case<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    case: &Case,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let Case { hidden, vocab, token_id } = *case;
    let embd: Vec<f16> = seeded_f32_range(seed, vocab * hidden, -1.0, 1.0)
        .iter().map(|v| f16::from_f32(*v)).collect();
    // Decode context: [position, token_id]; the kernel reads index 1.
    let ctx_i: Vec<i32> = vec![0, token_id as i32];

    let d_embd = alloc_and_upload(dev, &embd);
    let d_ctx = alloc_and_upload(dev, &ctx_i);
    let d_dst = dev.alloc(hidden * 2)?;

    {
        let stream = dev.default_stream();
        let hidden_i = hidden as i32;
        let e = d_embd.as_usize() as u64;
        let d = d_dst.as_usize() as u64;
        let c = d_ctx.as_usize() as u64;
        let mut a = <B::Args<'_>>::default();
        a.push_arg(&e);
        a.push_arg(&d);
        a.push_arg(&hidden_i);
        a.push_arg(&c);
        let threads = 256u32;
        let cfg = LaunchCfg {
            grid: ((hidden as u32).div_ceil(threads), 1, 1),
            block: (threads, 1, 1),
            shared_bytes: 0,
        };
        B::launch(kernel, stream, cfg, a)?;
        stream.synchronize()?;
    }

    let mut dst: Vec<f16> = vec![f16::from_f32(0.0); hidden];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(dst.as_mut_ptr() as usize),
            d_dst,
            hidden * 2,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_embd, embd.len() * 2)?;
        dev.dealloc(d_ctx, ctx_i.len() * 4)?;
        dev.dealloc(d_dst, hidden * 2)?;
    }

    let got: Vec<f32> = dst.iter().map(|v| v.to_f32()).collect();
    let reference: Vec<f32> =
        embd[token_id * hidden..(token_id + 1) * hidden].iter().map(|v| v.to_f32()).collect();
    Ok((got, reference))
}
