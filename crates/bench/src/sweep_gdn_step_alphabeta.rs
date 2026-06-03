//! C10 — differential cert for `gdn_state_step_alphabeta_f32_s128`
//! versus the unfused chain `gdn_alpha_beta_f32 ∘ gdn_state_step_f32_s128`.
//! The fused kernel inlines the softplus / sigmoid / exp ops the
//! alpha-beta kernel performs upstream, then runs the same recurrent
//! step. Same op order, same warp-reduce signatures — bit-identical at
//! FP32 in principle. This sweep verifies that on the live silicon by
//! running both paths on identical inputs and comparing post-step
//! `state_out` and `attn_out` element-wise.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — kernel launches + memcpy_async over local \
              allocations; same invariant as sweep_gdn_step.rs"
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

const S_V: usize = 128;

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    // Modules — the fused kernel under test, plus the two-kernel
    // baseline it replaces.
    let kb_fused = kernels::hsaco("gdn_state_step_alphabeta_f32")
        .ok_or_else(|| anyhow::anyhow!("gdn_state_step_alphabeta_f32 not compiled"))?;
    let mod_fused = HipModule::load(dev.id(), kb_fused)?;
    let kfused: HipKernel<'_> = mod_fused.kernel("flambeau_gdn_state_step_alphabeta_f32_s128")?;
    let attrs: FuncAttributes = kfused.attributes()?;

    let kb_ab = kernels::hsaco("gdn_alpha_beta_f32")
        .ok_or_else(|| anyhow::anyhow!("gdn_alpha_beta_f32 not compiled"))?;
    let mod_ab = HipModule::load(dev.id(), kb_ab)?;
    let kab: HipKernel<'_> = mod_ab.kernel("flambeau_gdn_alpha_beta_f32")?;

    let kb_step = kernels::hsaco("gdn_state_step_f32")
        .ok_or_else(|| anyhow::anyhow!("gdn_state_step_f32 not compiled"))?;
    let mod_step = HipModule::load(dev.id(), kb_step)?;
    let kstep: HipKernel<'_> = mod_step.kernel("flambeau_gdn_state_step_f32_s128")?;

    let shapes = [
        // Qwen3.6-35B-A3B GDN block: H_v=32, n_rep=2 (16 H_kv heads).
        (1usize, 32usize, 1usize, 2usize),
        (1, 32, 8, 2),
        // No-GQA path.
        (1, 4, 4, 1),
    ];
    let mut results = Vec::new();
    for (b, h_v, l, n_rep) in shapes {
        let seed = 0xC0FF1010
            ^ ((b as u64) * 1013 + (h_v as u64) * 47 + (l as u64) * 17 + (n_rep as u64) * 3);
        let max_rel_err = run_shape(
            &dev,
            &kfused,
            &kab,
            &kstep,
            GdnAlphaBetaShape { b, h_v, l, n_rep },
            seed,
        )?;
        // Fused vs unfused are arithmetically identical at FP32 — same
        // softplus/sigmoid/exp ops, same warp reductions. Tight tol.
        let tol = 1e-5;
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
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "gdn_state_step_alphabeta_f32_s128_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "gdn_state_step_alphabeta".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-5 vs unfused chain (alpha_beta + state_step)".to_string(),
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

/// Shared shape for `sweep_gdn_step_alphabeta`'s `run_shape` harness.
#[derive(Copy, Clone, Debug)]
struct GdnAlphaBetaShape {
    b: usize,
    h_v: usize,
    l: usize,
    n_rep: usize,
}

fn run_shape(
    dev: &HipDevice,
    kfused: &HipKernel<'_>,
    kab: &HipKernel<'_>,
    kstep: &HipKernel<'_>,
    shape: GdnAlphaBetaShape,
    seed: u64,
) -> Result<f32> {
    let GdnAlphaBetaShape { b, h_v, l, n_rep } = shape;
    assert!(h_v % n_rep == 0, "h_v must be a multiple of n_rep");
    let h_kv = h_v / n_rep;

    let qk_elems = b * h_kv * l * S_V;
    let v_elems = b * h_v * l * S_V;
    let gb_elems = b * h_v * l;
    let head_elems = h_v;
    let state_elems = b * h_v * S_V * S_V;
    let attn_elems = b * h_v * l * S_V;

    // Inputs — same generation pattern as sweep_gdn_step.rs.
    let alpha_in = seeded_f32_range(seed ^ 0xA1, gb_elems, -0.5, 0.5);
    let beta_in = seeded_f32_range(seed ^ 0xB2, gb_elems, -0.5, 0.5);
    let ssm_dt_bias = seeded_f32_range(seed ^ 0xD7, head_elems, -0.2, 0.2);
    let ssm_a = seeded_f32_range(seed ^ 0xA8, head_elems, -1.0, -0.1); // negative → exp damps
    let q = seeded_f32_range(seed ^ 0xC3, qk_elems, -0.5, 0.5);
    let k = seeded_f32_range(seed ^ 0xD4, qk_elems, -0.5, 0.5);
    let v = seeded_f32_range(seed ^ 0xE5, v_elems, -0.5, 0.5);
    let state_init = seeded_f32_range(seed ^ 0xF6, state_elems, -0.5, 0.5)
        .into_iter()
        .map(|x| x * 0.01)
        .collect::<Vec<_>>();

    // Device buffers shared by both runs.
    let d_q = alloc_and_upload(dev, &q);
    let d_k = alloc_and_upload(dev, &k);
    let d_v = alloc_and_upload(dev, &v);
    let d_alpha = alloc_and_upload(dev, &alpha_in);
    let d_beta_in = alloc_and_upload(dev, &beta_in);
    let d_dt = alloc_and_upload(dev, &ssm_dt_bias);
    let d_a = alloc_and_upload(dev, &ssm_a);
    let d_state_init = alloc_and_upload(dev, &state_init);

    // Run 1 — unfused: alpha_beta → gate/beta scratch, then state_step.
    let d_state_unfused = dev.alloc(state_elems * 4)?;
    let d_attn_unfused = dev.alloc(attn_elems * 4)?;
    let d_gate_scratch = dev.alloc(gb_elems * 4)?;
    let d_beta_scratch = dev.alloc(gb_elems * 4)?;
    {
        // Reset state_unfused = state_init.
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToDevice,
                d_state_unfused,
                d_state_init,
                state_elems * 4,
            )?;
        }
        // alpha_beta launch.
        {
            let stream = dev.default_stream();
            let num_v_i = h_v as i32;
            let n_tokens_i = l as i32;
            let a_ptr: u64 = d_alpha.as_usize() as u64;
            let b_ptr: u64 = d_beta_in.as_usize() as u64;
            let dt_ptr: u64 = d_dt.as_usize() as u64;
            let sa_ptr: u64 = d_a.as_usize() as u64;
            let g_ptr: u64 = d_gate_scratch.as_usize() as u64;
            let bo_ptr: u64 = d_beta_scratch.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&a_ptr);
            args.push(&b_ptr);
            args.push(&dt_ptr);
            args.push(&sa_ptr);
            args.push(&g_ptr);
            args.push(&bo_ptr);
            args.push(&num_v_i);
            args.push(&n_tokens_i);
            let cfg = LaunchCfg {
                grid: (l as u32, 1, 1),
                block: (h_v as u32, 1, 1),
                shared_bytes: 0,
            };
            unsafe { kab.launch(stream, cfg, args)? };
        }
        // state_step launch.
        {
            let stream = dev.default_stream();
            let b_i = b as i32;
            let h_i = h_v as i32;
            let l_i = l as i32;
            let n_rep_i = n_rep as i32;
            let rep_inner_i: i32 = 0;
            let q_ptr: u64 = d_q.as_usize() as u64;
            let k_ptr: u64 = d_k.as_usize() as u64;
            let v_ptr: u64 = d_v.as_usize() as u64;
            let g_ptr: u64 = d_gate_scratch.as_usize() as u64;
            let bo_ptr: u64 = d_beta_scratch.as_usize() as u64;
            let sin_ptr: u64 = d_state_unfused.as_usize() as u64;
            let sout_ptr: u64 = d_state_unfused.as_usize() as u64;
            let ao_ptr: u64 = d_attn_unfused.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&q_ptr);
            args.push(&k_ptr);
            args.push(&v_ptr);
            args.push(&g_ptr);
            args.push(&bo_ptr);
            args.push(&sin_ptr);
            args.push(&sout_ptr);
            args.push(&ao_ptr);
            args.push(&b_i);
            args.push(&h_i);
            args.push(&l_i);
            args.push(&n_rep_i);
            args.push(&rep_inner_i);
            let cfg = LaunchCfg {
                grid: (h_v as u32, b as u32, (S_V as u32) / 4),
                block: (64, 4, 1),
                shared_bytes: 0,
            };
            unsafe { kstep.launch(stream, cfg, args)? };
        }
        dev.default_stream().synchronize()?;
    }

    // Run 2 — fused state_step_alphabeta. Fresh state buffer = state_init.
    let d_state_fused = dev.alloc(state_elems * 4)?;
    let d_attn_fused = dev.alloc(attn_elems * 4)?;
    {
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToDevice,
                d_state_fused,
                d_state_init,
                state_elems * 4,
            )?;
        }
        let stream = dev.default_stream();
        let b_i = b as i32;
        let h_i = h_v as i32;
        let l_i = l as i32;
        let n_rep_i = n_rep as i32;
        let rep_inner_i: i32 = 0;
        let q_ptr: u64 = d_q.as_usize() as u64;
        let k_ptr: u64 = d_k.as_usize() as u64;
        let v_ptr: u64 = d_v.as_usize() as u64;
        let alpha_ptr: u64 = d_alpha.as_usize() as u64;
        let beta_ptr: u64 = d_beta_in.as_usize() as u64;
        let dt_ptr: u64 = d_dt.as_usize() as u64;
        let sa_ptr: u64 = d_a.as_usize() as u64;
        let sin_ptr: u64 = d_state_fused.as_usize() as u64;
        let sout_ptr: u64 = d_state_fused.as_usize() as u64;
        let ao_ptr: u64 = d_attn_fused.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&q_ptr);
        args.push(&k_ptr);
        args.push(&v_ptr);
        args.push(&alpha_ptr);
        args.push(&beta_ptr);
        args.push(&dt_ptr);
        args.push(&sa_ptr);
        args.push(&sin_ptr);
        args.push(&sout_ptr);
        args.push(&ao_ptr);
        args.push(&b_i);
        args.push(&h_i);
        args.push(&l_i);
        args.push(&n_rep_i);
        args.push(&rep_inner_i);
        let cfg = LaunchCfg {
            grid: (h_v as u32, b as u32, (S_V as u32) / 4),
            block: (64, 4, 1),
            shared_bytes: 0,
        };
        unsafe { kfused.launch(stream, cfg, args)? };
        dev.default_stream().synchronize()?;
    }

    let mut got_state_unfused = vec![0.0f32; state_elems];
    let mut got_attn_unfused = vec![0.0f32; attn_elems];
    let mut got_state_fused = vec![0.0f32; state_elems];
    let mut got_attn_fused = vec![0.0f32; attn_elems];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_state_unfused.as_mut_ptr() as usize),
            d_state_unfused,
            state_elems * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_attn_unfused.as_mut_ptr() as usize),
            d_attn_unfused,
            attn_elems * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_state_fused.as_mut_ptr() as usize),
            d_state_fused,
            state_elems * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_attn_fused.as_mut_ptr() as usize),
            d_attn_fused,
            attn_elems * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_q, qk_elems * 4)?;
        dev.dealloc(d_k, qk_elems * 4)?;
        dev.dealloc(d_v, v_elems * 4)?;
        dev.dealloc(d_alpha, gb_elems * 4)?;
        dev.dealloc(d_beta_in, gb_elems * 4)?;
        dev.dealloc(d_dt, head_elems * 4)?;
        dev.dealloc(d_a, head_elems * 4)?;
        dev.dealloc(d_state_init, state_elems * 4)?;
        dev.dealloc(d_state_unfused, state_elems * 4)?;
        dev.dealloc(d_attn_unfused, attn_elems * 4)?;
        dev.dealloc(d_gate_scratch, gb_elems * 4)?;
        dev.dealloc(d_beta_scratch, gb_elems * 4)?;
        dev.dealloc(d_state_fused, state_elems * 4)?;
        dev.dealloc(d_attn_fused, attn_elems * 4)?;
    }

    let state_err = max_rel_err_with_floor(&got_state_fused, &got_state_unfused, 1.0);
    let attn_err = max_rel_err_with_floor(&got_attn_fused, &got_attn_unfused, 1.0);
    Ok(state_err.max(attn_err))
}
