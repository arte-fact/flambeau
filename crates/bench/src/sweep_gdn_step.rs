//! V1.7.2.F — Gated-Delta-Net fused step correctness sweep.
//!
//! Reference: candle's `delta_net_single_step` tensor-op chain, unrolled
//! to a flat per-(b, h) scalar loop so the host and device follow the
//! same op order and compare bit-wise in F32:
//!
//!   state[col, i] *= exp(gate[t])
//!   sk[col]       = Σ_i state[col, i] * k[t, i]
//!   delta[col]    = (v[t, col] - sk[col]) * beta[t]
//!   state[col, i] += k[t, i] * delta[col]
//!   attn[t, col]  = Σ_i state[col, i] * q[t, i]
//!
//! Shape budget covers Qwen3.6-35B (`num_v_heads = 32`, `head_v_dim = 128`,
//! GQA `n_rep = 32 / 16 = 2`), both decode (L = 1) and a prefill chunk
//! (L = 8). A small-batch shape with n_rep = 1 sanity-checks the no-GQA
//! path.

#![cfg(feature = "hip")]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};

const S_V: usize = 128;

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("gdn_state_step_f32")
        .ok_or_else(|| anyhow::anyhow!("gdn_state_step_f32 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_gdn_state_step_f32_s128")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // (B, H_v, L, n_rep). Qwen3.6 GDN layer: H_v=32, n_rep=2. Decode L=1,
    // prefill chunk L=8. Small batch + n_rep=1 covers the no-GQA path.
    let shapes = [
        (1usize, 32usize, 1usize, 2usize),
        (1, 32, 8, 2),
        (1, 4, 4, 1),
    ];
    let mut results = Vec::new();
    for (b, h_v, l, n_rep) in shapes {
        let seed = 0xC0FFEE
            ^ ((b as u64) * 1013 + (h_v as u64) * 47 + (l as u64) * 17 + (n_rep as u64) * 3);
        let max_rel_err = run_shape(&dev, &kernel, b, h_v, l, n_rep, seed)?;
        // F32 end-to-end, ~S_v*L mul-adds per column per token. Slight
        // reassociation between CPU (column-major scalar) and GPU (warp
        // reduction) leaves a small bound.
        let tol = 1e-4;
        results.push(ShapeResult {
            m: b * h_v,
            k: l,
            n: n_rep,
            seed,
            max_rel_err,
            tolerance: tol,
            pass: max_rel_err <= tol,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()));
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "gdn_state_step_f32_s128_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "gdn_state_step".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-4 * max(|ref|, 1)".to_string(),
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
    b: usize,
    h_v: usize,
    l: usize,
    n_rep: usize,
    seed: u64,
) -> Result<f32> {
    assert!(h_v % n_rep == 0, "h_v must be a multiple of n_rep");
    let h_kv = h_v / n_rep;

    let qk_elems = b * h_kv * l * S_V;
    let v_elems = b * h_v * l * S_V;
    let gb_elems = b * h_v * l;
    let state_elems = b * h_v * S_V * S_V;
    let attn_elems = b * h_v * l * S_V;

    // Gate scaled so exp(gate) stays well below 1 — keeps the recurrence
    // numerically tame over L steps and avoids state blow-up that would
    // dominate the tolerance.
    let mut gate = seeded_f32(seed ^ 0xA1, gb_elems);
    for g in gate.iter_mut() {
        *g = -(g.abs() + 0.1);
    }
    let mut beta = seeded_f32(seed ^ 0xB2, gb_elems);
    for b_val in beta.iter_mut() {
        *b_val = 0.1 + 0.4 * (b_val.abs());
    }
    let q = seeded_f32(seed ^ 0xC3, qk_elems);
    let k = seeded_f32(seed ^ 0xD4, qk_elems);
    let v = seeded_f32(seed ^ 0xE5, v_elems);
    let state_init = seeded_f32(seed ^ 0xF6, state_elems)
        .into_iter()
        .map(|x| x * 0.01)
        .collect::<Vec<_>>();

    // CPU reference. Col-outer layout matches the kernel so we can compare
    // the post-run state element-wise without a transpose.
    let mut state_ref = state_init.clone();
    let mut attn_ref = vec![0.0f32; attn_elems];
    for bi in 0..b {
        for hv in 0..h_v {
            let hkv = hv % h_kv;
            let bh = bi * h_v + hv;
            let bh_kv = bi * h_kv + hkv;
            for t in 0..l {
                let g = (gate[bh * l + t] as f64).exp() as f32;
                let bt = beta[bh * l + t];
                for col in 0..S_V {
                    // state[col, *] *= g
                    let state_base = (bh * S_V + col) * S_V;
                    for i in 0..S_V {
                        state_ref[state_base + i] *= g;
                    }
                    // sk = Σ state[col, i] * k[t, i]
                    let mut sk = 0.0f32;
                    for i in 0..S_V {
                        sk += state_ref[state_base + i]
                            * k[((bh_kv * l) + t) * S_V + i];
                    }
                    let v_col = v[((bh * l) + t) * S_V + col];
                    let delta = (v_col - sk) * bt;
                    // state[col, i] += k[i] * delta
                    for i in 0..S_V {
                        state_ref[state_base + i] +=
                            k[((bh_kv * l) + t) * S_V + i] * delta;
                    }
                    // attn[t, col] = Σ state[col, i] * q[t, i]
                    let mut a = 0.0f32;
                    for i in 0..S_V {
                        a += state_ref[state_base + i]
                            * q[((bh_kv * l) + t) * S_V + i];
                    }
                    attn_ref[((bh * l) + t) * S_V + col] = a;
                }
            }
        }
    }

    let d_q = alloc_and_upload(dev, &q);
    let d_k = alloc_and_upload(dev, &k);
    let d_v = alloc_and_upload(dev, &v);
    let d_gate = alloc_and_upload(dev, &gate);
    let d_beta = alloc_and_upload(dev, &beta);
    let d_state_in = alloc_and_upload(dev, &state_init);
    let d_state_out = dev.alloc(state_elems * 4)?;
    let d_attn = dev.alloc(attn_elems * 4)?;
    {
        let stream = dev.default_stream();
        let b_i = b as i32;
        let h_i = h_v as i32;
        let l_i = l as i32;
        let n_rep_i = n_rep as i32;
        let q_ptr: u64 = d_q.as_usize() as u64;
        let k_ptr: u64 = d_k.as_usize() as u64;
        let v_ptr: u64 = d_v.as_usize() as u64;
        let gate_ptr: u64 = d_gate.as_usize() as u64;
        let beta_ptr: u64 = d_beta.as_usize() as u64;
        let sin_ptr: u64 = d_state_in.as_usize() as u64;
        let sout_ptr: u64 = d_state_out.as_usize() as u64;
        let ao_ptr: u64 = d_attn.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&q_ptr);
        args.push(&k_ptr);
        args.push(&v_ptr);
        args.push(&gate_ptr);
        args.push(&beta_ptr);
        args.push(&sin_ptr);
        args.push(&sout_ptr);
        args.push(&ao_ptr);
        args.push(&b_i);
        args.push(&h_i);
        args.push(&l_i);
        args.push(&n_rep_i);
        let cfg = LaunchCfg {
            grid: (h_v as u32, b as u32, (S_V as u32) / 4),
            block: (64, 4, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got_state = vec![0.0f32; state_elems];
    let mut got_attn = vec![0.0f32; attn_elems];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_state.as_mut_ptr() as usize),
            d_state_out,
            state_elems * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_attn.as_mut_ptr() as usize),
            d_attn,
            attn_elems * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_q, qk_elems * 4)?;
        dev.dealloc(d_k, qk_elems * 4)?;
        dev.dealloc(d_v, v_elems * 4)?;
        dev.dealloc(d_gate, gb_elems * 4)?;
        dev.dealloc(d_beta, gb_elems * 4)?;
        dev.dealloc(d_state_in, state_elems * 4)?;
        dev.dealloc(d_state_out, state_elems * 4)?;
        dev.dealloc(d_attn, attn_elems * 4)?;
    }

    let state_err = max_rel_err(&got_state, &state_ref);
    let attn_err = max_rel_err(&got_attn, &attn_ref);
    Ok(state_err.max(attn_err))
}

fn alloc_and_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            (u as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(1.0))
        .fold(0.0f32, f32::max)
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
