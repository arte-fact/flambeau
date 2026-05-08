//! shared-expert gate-scale correctness sweep.

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

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("shared_expert_scale_f32")
        .ok_or_else(|| anyhow::anyhow!("shared_expert_scale_f32 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_shared_expert_scale_f32")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // (n_tokens, hidden). Qwen3.6: hidden=2048. Decode n_tokens=1,
    // prefill varies.
    let shapes = [(1usize, 2048usize), (8, 2048), (128, 2048)];
    let mut results = Vec::new();
    for (n_tokens, hidden) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1039 + (hidden as u64) * 47);
        let max_rel_err = run_shape(&dev, &kernel, n_tokens, hidden, seed)?;
        let tol = 1e-5;
        results.push(ShapeResult {
            m: n_tokens,
            k: hidden,
            n: 1,
            seed,
            max_rel_err,
            tolerance: tol,
            pass: max_rel_err <= tol,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "shared_expert_scale_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "shared_expert_scale".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-5 * max(|ref|, 1)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(PmcSnapshot {
            vgpr_count: Some(attrs.num_regs),
            sgpr_count: None,
            waves_per_simd: Some(attrs.gfx906_waves_per_simd()),
            mem_busy_pct: None,
            valu_busy_pct: None,
        }),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    n_tokens: usize,
    hidden: usize,
    seed: u64,
) -> Result<f32> {
    let x = seeded_f32_range(seed, n_tokens * hidden, -0.5, 0.5);
    let gate_w = seeded_f32_range(seed.wrapping_add(0xA1), hidden, -0.5, 0.5);
    let shared_out_in = seeded_f32_range(seed.wrapping_add(0xA2), n_tokens * hidden, -0.5, 0.5);

    // CPU reference.
    let mut reference = vec![0.0f32; n_tokens * hidden];
    for t in 0..n_tokens {
        let mut dot = 0.0f64;
        for i in 0..hidden {
            dot += (gate_w[i] as f64) * (x[t * hidden + i] as f64);
        }
        let g = (1.0 / (1.0 + (-dot).exp())) as f32;
        for i in 0..hidden {
            reference[t * hidden + i] = shared_out_in[t * hidden + i] * g;
        }
    }

    let d_so = alloc_and_upload(dev, &shared_out_in);
    let d_x = alloc_and_upload(dev, &x);
    let d_w = alloc_and_upload(dev, &gate_w);
    {
        let stream = dev.default_stream();
        let n_tokens_i = n_tokens as i32;
        let hidden_i = hidden as i32;
        let so_ptr: u64 = d_so.as_usize() as u64;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let w_ptr: u64 = d_w.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&so_ptr);
        args.push(&x_ptr);
        args.push(&w_ptr);
        args.push(&n_tokens_i);
        args.push(&hidden_i);
        let cfg = LaunchCfg::one_d(n_tokens as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }
    let mut got = vec![0.0f32; n_tokens * hidden];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_so,
            n_tokens * hidden * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_so, n_tokens * hidden * 4)?;
        dev.dealloc(d_x, x.len() * 4)?;
        dev.dealloc(d_w, gate_w.len() * 4)?;
    }
    Ok(max_rel_err_with_floor(&got, &reference, 1.0))
}

