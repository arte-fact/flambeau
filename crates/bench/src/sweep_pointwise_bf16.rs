//! MTP-4-C-5 — BF16 pointwise correctness sweeps.
//!
//! Four kernels, four sweep entrypoints:
//!   * `split_q_gate_bf16`        — bit-exact (pure copy)
//!   * `sigmoid_mul_bf16`         — BF16 round on output
//!   * `swiglu_f32_to_bf16`       — F32 silu*b → BF16 round
//!   * `rope_neox_partial_bf16`   — trig+mul, in-place BF16 round

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
use half::bf16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

// ── shared helper ────────────────────────────────────────────────────

fn pmc(attrs: &FuncAttributes) -> PmcSnapshot {
    PmcSnapshot {
        vgpr_count: Some(attrs.num_regs),
        sgpr_count: None,
        waves_per_simd: Some(attrs.gfx906_waves_per_simd()),
        mem_busy_pct: None,
        valu_busy_pct: None,
    }
}

// ── split_q_gate_bf16 ────────────────────────────────────────────────

pub fn run_split_q_gate_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("split_q_gate_bf16").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_split_q_gate_bf16")?;
    let attrs = kernel.attributes()?;

    // (n_tokens, n_heads, head_dim)
    let cases = [
        (1usize, 24usize, 256usize),  // MTP Q-proj split (Qwen3.6)
        (1, 16, 128),                  // smaller head_dim sanity
        (8, 16, 256),                  // multi-token
    ];
    let mut results = Vec::new();
    for (nt, nh, hd) in cases {
        let seed = 0xBF16_5910u64 ^ (nt * 7919) as u64 ^ (nh * 31) as u64 ^ (hd * 13) as u64;
        let n = nt * nh * 2 * hd;
        let fused_f32 = seeded_f32_range(seed, n, -0.5, 0.5);
        let fused: Vec<bf16> = fused_f32.iter().map(|v| bf16::from_f32(*v)).collect();
        let d_f = alloc_and_upload(&dev, &fused);
        let d_q = dev.alloc(nt * nh * hd * 2)?;
        let d_g = dev.alloc(nt * nh * hd * 2)?;

        let stream = dev.default_stream();
        let nt_i = nt as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let f_ptr: u64 = d_f.as_usize() as u64;
        let q_ptr: u64 = d_q.as_usize() as u64;
        let g_ptr: u64 = d_g.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&f_ptr); args.push(&q_ptr); args.push(&g_ptr);
        args.push(&nt_i); args.push(&nh_i); args.push(&hd_i);
        let threads = 128u32;
        let grid_z = (hd as u32).div_ceil(threads);
        let cfg = LaunchCfg {
            grid: (nt as u32, nh as u32, grid_z),
            block: (threads, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;

        let mut got_q = vec![bf16::from_f32(0.0); nt * nh * hd];
        let mut got_g = vec![bf16::from_f32(0.0); nt * nh * hd];
        unsafe {
            dev.memcpy_async(stream, CopyDirection::DeviceToHost,
                DevicePtr(got_q.as_mut_ptr() as usize), d_q, nt * nh * hd * 2)?;
            dev.memcpy_async(stream, CopyDirection::DeviceToHost,
                DevicePtr(got_g.as_mut_ptr() as usize), d_g, nt * nh * hd * 2)?;
        }
        stream.synchronize()?;
        unsafe {
            dev.dealloc(d_f, n * 2)?;
            dev.dealloc(d_q, nt * nh * hd * 2)?;
            dev.dealloc(d_g, nt * nh * hd * 2)?;
        }
        // Bit-exact check.
        let mut diff = 0u32;
        for t in 0..nt {
            for h in 0..nh {
                for d in 0..hd {
                    let fb_q = fused[(t * nh + h) * 2 * hd + d];
                    let fb_g = fused[(t * nh + h) * 2 * hd + hd + d];
                    if got_q[(t * nh + h) * hd + d].to_bits() != fb_q.to_bits() { diff += 1; }
                    if got_g[(t * nh + h) * hd + d].to_bits() != fb_g.to_bits() { diff += 1; }
                }
            }
        }
        results.push(ShapeResult {
            m: nt, k: nh, n: hd, seed,
            max_rel_err: diff as f32, tolerance: 0.0, pass: diff == 0,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "split_q_gate_bf16_gfx906".into(),
        backend: "hip".into(), arch: "gfx906".into(),
        op: "split_q_gate".into(),
        dtype_weight: "BF16".into(), dtype_activation: "BF16".into(),
        tolerance_formula: "exact bit match (pure strided copy)".into(),
        results, pass,
        emitted_at: now_utc_iso8601(), rig: rig(),
        pmc: Some(pmc(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// ── sigmoid_mul_bf16 ─────────────────────────────────────────────────

pub fn run_sigmoid_mul_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("sigmoid_mul_bf16").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_sigmoid_mul_bf16")?;
    let attrs = kernel.attributes()?;

    let shapes = [256usize, 6144, 8192];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xBF16_516Du64 ^ (n as u64 * 13);
        let g_f32 = seeded_f32_range(seed, n, -3.0, 3.0);
        let x_f32 = seeded_f32_range(seed.wrapping_add(0xA1), n, -1.0, 1.0);
        let g: Vec<bf16> = g_f32.iter().map(|v| bf16::from_f32(*v)).collect();
        let x: Vec<bf16> = x_f32.iter().map(|v| bf16::from_f32(*v)).collect();
        let d_g = alloc_and_upload(&dev, &g);
        let d_x = alloc_and_upload(&dev, &x);
        let d_y = dev.alloc(n * 2)?;

        let stream = dev.default_stream();
        let n_i = n as i32;
        let g_ptr: u64 = d_g.as_usize() as u64;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&g_ptr); args.push(&x_ptr); args.push(&y_ptr); args.push(&n_i);
        let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;

        let mut got = vec![bf16::from_f32(0.0); n];
        unsafe {
            dev.memcpy_async(stream, CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_y, n * 2)?;
        }
        stream.synchronize()?;
        unsafe {
            dev.dealloc(d_g, n * 2)?; dev.dealloc(d_x, n * 2)?; dev.dealloc(d_y, n * 2)?;
        }
        let got_f32: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();
        // Reference: sigmoid(g_lifted) * x_lifted, then BF16 round.
        let g_lift: Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
        let x_lift: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
        let reference: Vec<f32> = g_lift.iter().zip(x_lift.iter())
            .map(|(&gv, &xv)| {
                let sig = if gv >= 0.0 { 1.0 / (1.0 + (-gv).exp()) }
                          else { let z = gv.exp(); z / (1.0 + z) };
                bf16::from_f32(sig * xv).to_f32()
            }).collect();
        let max_rel = max_rel_err_with_floor(&got_f32, &reference, 1e-3);
        let tol = 1e-2;
        results.push(ShapeResult {
            m: 1, k: n, n, seed,
            max_rel_err: max_rel, tolerance: tol, pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "sigmoid_mul_bf16_gfx906".into(),
        backend: "hip".into(), arch: "gfx906".into(),
        op: "sigmoid_mul".into(),
        dtype_weight: "BF16".into(), dtype_activation: "BF16".into(),
        tolerance_formula: "|err| <= 1e-2 (BF16 output round)".into(),
        results, pass,
        emitted_at: now_utc_iso8601(), rig: rig(),
        pmc: Some(pmc(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// ── swiglu_f32_to_bf16 ───────────────────────────────────────────────

pub fn run_swiglu_f32_to_bf16_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("swiglu_f32_to_bf16").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_swiglu_f32_to_bf16")?;
    let attrs = kernel.attributes()?;

    let shapes = [256usize, 5120, 25600];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xBF16_5610u64 ^ (n as u64 * 13);
        let a_f32 = seeded_f32_range(seed, n, -2.0, 2.0);
        let b_f32 = seeded_f32_range(seed.wrapping_add(0xA1), n, -1.0, 1.0);
        let d_a = alloc_and_upload(&dev, &a_f32);
        let d_b = alloc_and_upload(&dev, &b_f32);
        let d_y = dev.alloc(n * 2)?;

        let stream = dev.default_stream();
        let n_i = n as i32;
        let a_ptr: u64 = d_a.as_usize() as u64;
        let b_ptr: u64 = d_b.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&a_ptr); args.push(&b_ptr); args.push(&y_ptr); args.push(&n_i);
        let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;

        let mut got = vec![bf16::from_f32(0.0); n];
        unsafe {
            dev.memcpy_async(stream, CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_y, n * 2)?;
        }
        stream.synchronize()?;
        unsafe {
            dev.dealloc(d_a, n * 4)?; dev.dealloc(d_b, n * 4)?; dev.dealloc(d_y, n * 2)?;
        }
        let got_f32: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();
        let reference: Vec<f32> = a_f32.iter().zip(b_f32.iter())
            .map(|(&av, &bv)| {
                let silu = av / (1.0 + (-av).exp());
                bf16::from_f32(silu * bv).to_f32()
            }).collect();
        let max_rel = max_rel_err_with_floor(&got_f32, &reference, 1e-3);
        let tol = 1e-2;
        results.push(ShapeResult {
            m: 1, k: n, n, seed,
            max_rel_err: max_rel, tolerance: tol, pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "swiglu_f32_to_bf16_gfx906".into(),
        backend: "hip".into(), arch: "gfx906".into(),
        op: "swiglu_f32_to_bf16".into(),
        dtype_weight: "F32".into(), dtype_activation: "BF16".into(),
        tolerance_formula: "|err| <= 1e-2 (BF16 output round)".into(),
        results, pass,
        emitted_at: now_utc_iso8601(), rig: rig(),
        pmc: Some(pmc(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// ── rope_neox_partial_bf16 ───────────────────────────────────────────

pub fn run_rope_neox_partial_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("rope_neox_partial_bf16").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_rope_neox_partial_bf16")?;
    let attrs = kernel.attributes()?;

    // (n_tokens, n_heads, head_dim, rotated_dims, position)
    let cases = [
        (1usize, 24usize, 256usize, 64usize, 17i32),  // MTP-style (Qwen3.6)
        (1, 16, 128, 64, 1024),
        (4, 24, 256, 128, 0),
    ];
    let theta_base = 1.0e7_f32;
    let mut results = Vec::new();
    for (nt, nh, hd, rd, pos) in cases {
        let seed = 0xBF16_C0DEu64
            ^ (nt as u64 * 7919) ^ (nh as u64 * 31)
            ^ (hd as u64 * 13)   ^ (rd as u64 * 5);
        let n = nt * nh * hd;
        let x_f32 = seeded_f32_range(seed, n, -0.5, 0.5);
        let x_bf16: Vec<bf16> = x_f32.iter().map(|v| bf16::from_f32(*v)).collect();
        let positions: Vec<i32> = (0..nt).map(|_| pos).collect();
        let d_x = alloc_and_upload(&dev, &x_bf16);
        let d_p = alloc_and_upload(&dev, &positions);

        let stream = dev.default_stream();
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let rd_i = rd as i32;
        let theta = theta_base;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let p_ptr: u64 = d_p.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&x_ptr); args.push(&p_ptr); args.push(&theta);
        args.push(&nh_i); args.push(&hd_i); args.push(&rd_i);
        let cfg = LaunchCfg {
            grid: (nt as u32, nh as u32, 1),
            block: ((rd / 2) as u32, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;

        let mut got = vec![bf16::from_f32(0.0); n];
        unsafe {
            dev.memcpy_async(stream, CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_x, n * 2)?;
        }
        stream.synchronize()?;
        unsafe { dev.dealloc(d_x, n * 2)?; dev.dealloc(d_p, nt * 4)?; }

        // Reference: same partial-NeoX RoPE on BF16-lifted inputs.
        let x_lift: Vec<f32> = x_bf16.iter().map(|v| v.to_f32()).collect();
        let mut reference = x_lift.clone();
        let half = rd / 2;
        for ti in 0..nt {
            for hi in 0..nh {
                let base = (ti * nh + hi) * hd;
                for pi in 0..half {
                    let exponent = 2.0 * pi as f32 / rd as f32;
                    let inv_freq = 1.0 / theta_base.powf(exponent);
                    let angle = positions[ti] as f32 * inv_freq;
                    let c = angle.cos();
                    let s = angle.sin();
                    let x0 = x_lift[base + pi];
                    let x1 = x_lift[base + pi + half];
                    reference[base + pi]        = bf16::from_f32(x0 * c - x1 * s).to_f32();
                    reference[base + pi + half] = bf16::from_f32(x0 * s + x1 * c).to_f32();
                }
            }
        }
        let got_f32: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();
        let max_rel = max_rel_err_with_floor(&got_f32, &reference, 1e-3);
        let tol = 2e-2;
        results.push(ShapeResult {
            m: nt, k: nh, n: hd, seed,
            max_rel_err: max_rel, tolerance: tol, pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "rope_neox_partial_bf16_gfx906".into(),
        backend: "hip".into(), arch: "gfx906".into(),
        op: "rope_neox_partial".into(),
        dtype_weight: "BF16".into(), dtype_activation: "BF16".into(),
        tolerance_formula: "|err| <= 2e-2 (BF16 in-place + trig round)".into(),
        results, pass,
        emitted_at: now_utc_iso8601(), rig: rig(),
        pmc: Some(pmc(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}
