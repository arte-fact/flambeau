//! V1.6.2 masked softmax correctness sweep.

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
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("softmax_masked_f16").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_softmax_masked_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Attention-score shapes: (n_rows = n_heads * n_q_tokens, k = n_kv_tokens).
    // Qwen3.6 GQA: 32 heads, head_dim=128.
    // Decode: n_q = 1, so n_rows = 32.
    // Prefill: n_q = 512, so n_rows = 32*512 = 16k rows, k up to seq_len.
    let shapes = [
        (32usize, 128usize),    // decode, short context
        (32, 1024),             // decode, medium context
        (32, 4096),             // decode, long context
        (32 * 128, 128),        // 128-token prefill, short context
        (32 * 128, 4096),       // 128-token prefill, long-ish context
    ];
    let scale = 1.0 / (128.0f32).sqrt(); // head_dim=128 attention scale
    let seed = 0xDECADEu64;

    let mut results = Vec::new();
    for (m, k) in shapes {
        let (got_causal, ref_causal) = run_shape(&dev, &kernel, m, k, scale, seed, true)?;
        let err_causal = max_rel_err_with_floor(&got_causal, &ref_causal, 1e-6);
        let (got_plain, ref_plain) = run_shape(&dev, &kernel, m, k, scale, seed, false)?;
        let err_plain = max_rel_err_with_floor(&got_plain, &ref_plain, 1e-6);
        let max_err = err_causal.max(err_plain);
        let tol = 5e-3;
        results.push(ShapeResult {
            m,
            k,
            n: k,
            seed,
            max_rel_err: max_err,
            tolerance: tol,
            pass: max_err <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_softmax",
            m, k, err_causal, err_plain, tol,
            "softmax shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "softmax_masked_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "softmax_masked".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "|err| <= 5e-3 * max(|ref|, 1e-6)".to_string(),
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
    m: usize,
    k: usize,
    scale: f32,
    seed: u64,
    causal: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let scores_f32 = seeded_f32_range(seed, m * k, -2.0, 2.0);
    let scores_f16: Vec<f16> = scores_f32.iter().map(|v| f16::from_f32(*v)).collect();

    // Causal mask: entry at (row, col) = -INF when col > (row mod n_q_per_head)
    // ... but we don't know n_heads/n_q_tokens partition at this layer. The
    // generic softmax kernel takes the mask buffer as input and applies it
    // verbatim. For cert purposes we use a random mask-or-null:
    //   causal=true  → additive mask of -INF at a deterministic set of
    //                  columns per row (tests the masked path);
    //   causal=false → mask pointer = null (tests the unmasked path).
    let mask_f16: Vec<f16> = if causal {
        (0..m * k)
            .map(|idx| {
                let col = idx % k;
                let boundary = (idx / k) % k;
                if col > boundary {
                    f16::from_f32(f32::NEG_INFINITY)
                } else {
                    f16::from_f32(0.0)
                }
            })
            .collect()
    } else {
        vec![]
    };

    let d_scores = alloc_and_upload(dev, &scores_f16);
    let d_mask = if causal {
        alloc_and_upload(dev, &mask_f16)
    } else {
        DevicePtr::NULL
    };
    let d_out = dev.alloc(m * k * 2)?;

    {
        let stream = dev.default_stream();
        let m_i = m as i32;
        let k_i = k as i32;
        let scale_f = scale;
        let d_s_ptr: u64 = d_scores.as_usize() as u64;
        let d_m_ptr: u64 = d_mask.as_usize() as u64;
        let d_o_ptr: u64 = d_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_s_ptr);
        args.push(&d_m_ptr);
        args.push(&d_o_ptr);
        args.push(&m_i);
        args.push(&k_i);
        args.push(&scale_f);
        let cfg = LaunchCfg::one_d(m as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut out_f16: Vec<f16> = vec![f16::from_f32(0.0); m * k];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out_f16.as_mut_ptr() as usize),
            d_out,
            m * k * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_scores, m * k * 2)?;
        if causal {
            dev.dealloc(d_mask, m * k * 2)?;
        }
        dev.dealloc(d_out, m * k * 2)?;
    }
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();

    // Reference: F32 softmax in "textbook" three-pass form.
    let mut reference = vec![0.0f32; m * k];
    for row in 0..m {
        let sr = &scores_f32[row * k..(row + 1) * k];
        let sr_f16_cast: Vec<f32> = sr.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
        // Row max after scaling + masking.
        let mut mx = f32::NEG_INFINITY;
        for (j, &v) in sr_f16_cast.iter().enumerate() {
            let mut u = scale * v;
            if causal {
                u += mask_f16[row * k + j].to_f32();
            }
            if u > mx {
                mx = u;
            }
        }
        let mut sum = 0.0f64;
        for (j, &v) in sr_f16_cast.iter().enumerate() {
            let mut u = scale * v;
            if causal {
                u += mask_f16[row * k + j].to_f32();
            }
            sum += (u - mx).exp() as f64;
        }
        for (j, &v) in sr_f16_cast.iter().enumerate() {
            let mut u = scale * v;
            if causal {
                u += mask_f16[row * k + j].to_f32();
            }
            let e = ((u - mx).exp() as f64) / sum;
            reference[row * k + j] = f16::from_f32(e as f32).to_f32();
        }
    }
    Ok((got, reference))
}

