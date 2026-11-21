//! `gdn_state_step_alphabeta_f32_s128_kstates` cert: differential against the
//! certified `gdn_state_step_alphabeta_f32_s128` on the same device, which is
//! exactly the snapshot + replay path the MTP cycle used before:
//! - `kstates` slot t == the base kernel's state after `L = t + 1` tokens from
//!   the same input state (the replay of a prefix), bit for bit;
//! - attention output == the base kernel's at the full `L` (the verify pass);
//! - input state read from the live buffer (`rs_idx = 0`) or from a `kstates`
//!   slot (`rs_idx = j`), including the slot the pass overwrites (j <= L).
//!
//! Backend-generic; emits `certs/<backend>/<arch>/gdn_state_step_alphabeta_f32_kstates.json`.

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, rig_for, seeded_f32_range};

const STEM: &str = "gdn_state_step_alphabeta_f32_kstates";
const ENTRY: &str = "flambeau_gdn_state_step_alphabeta_f32_s128_kstates";
const S_V: usize = 128;
const WARPS_PER_BLOCK: u32 = 4;
const TOL_FORMULA: &str = "bit-identical to gdn_state_step_alphabeta_f32_s128: kstates slot t == base state after \
                           L = t + 1 from the same input, attn_out == base attn_out at L; max_rel_err = number of \
                           differing f32 elements";

struct Case {
    h: usize,
    n_rep: usize,
    rep_inner: bool,
    l: usize,
    /// 0: input = live state; j > 0: input = kstates slot j - 1.
    rs_idx: usize,
}

const CASES: &[Case] = &[
    Case {
        h: 2,
        n_rep: 1,
        rep_inner: false,
        l: 1,
        rs_idx: 0,
    },
    Case {
        h: 2,
        n_rep: 1,
        rep_inner: false,
        l: 5,
        rs_idx: 0,
    },
    Case {
        h: 48,
        n_rep: 3,
        rep_inner: false,
        l: 5,
        rs_idx: 0,
    },
    Case {
        h: 48,
        n_rep: 3,
        rep_inner: true,
        l: 5,
        rs_idx: 0,
    },
    Case {
        h: 48,
        n_rep: 3,
        rep_inner: false,
        l: 5,
        rs_idx: 1,
    },
    Case {
        h: 48,
        n_rep: 3,
        rep_inner: false,
        l: 5,
        rs_idx: 3,
    },
    Case {
        h: 48,
        n_rep: 3,
        rep_inner: false,
        l: 5,
        rs_idx: 5,
    },
    Case {
        h: 24,
        n_rep: 3,
        rep_inner: false,
        l: 2,
        rs_idx: 2,
    },
    Case {
        h: 24,
        n_rep: 3,
        rep_inner: false,
        l: 9,
        rs_idx: 7,
    },
];

fn ptr(p: DevicePtr) -> u64 {
    p.as_usize() as u64
}

fn download<D: Device>(dev: &D, src: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    // SAFETY: `src` holds at least `n` f32; synced below.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            src,
            n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(out)
}

fn bit_mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
#[allow(clippy::too_many_lines, clippy::undocumented_unsafe_blocks)]
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image(STEM).ok_or_else(|| anyhow!("{STEM} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, ENTRY)?;
    let pmc = B::attributes(&kernel)?;
    let base_image = B::image("gdn_state_step_alphabeta_f32")
        .ok_or_else(|| anyhow!("base GDN step not compiled"))?;
    let base_module = B::load(&dev, base_image)?;
    let base = B::kernel(&base_module, "flambeau_gdn_state_step_alphabeta_f32_s128")?;
    let warp_size: u32 = if arch.starts_with("sm_") { 32 } else { 64 };

    let mut results = Vec::new();
    for case in CASES {
        let Case {
            h,
            n_rep,
            rep_inner,
            l,
            rs_idx,
        } = *case;
        let h_kv = h / n_rep;
        let slots = l.max(rs_idx);
        let slot_elems = h * S_V * S_V;
        let seed = 0x6D75_0000
            ^ ((h as u64) << 12)
            ^ ((l as u64) << 4)
            ^ rs_idx as u64
            ^ u64::from(rep_inner) << 20;
        let q = seeded_f32_range(seed, l * h_kv * S_V, -0.3, 0.3);
        let k = seeded_f32_range(seed ^ 0x11, l * h_kv * S_V, -0.3, 0.3);
        let v = seeded_f32_range(seed ^ 0x22, l * h * S_V, -0.3, 0.3);
        let alpha = seeded_f32_range(seed ^ 0x33, l * h, -1.0, 1.0);
        let beta = seeded_f32_range(seed ^ 0x44, l * h, -1.0, 1.0);
        let dt_bias = seeded_f32_range(seed ^ 0x55, h, -0.3, 0.3);
        let ssm_a: Vec<f32> = (0..h).map(|i| -(0.05 + 0.01 * i as f32)).collect();
        let state0 = seeded_f32_range(seed ^ 0x66, slot_elems, -0.1, 0.1);
        let filler = seeded_f32_range(seed ^ 0x77, slots * slot_elems, -5.0, 5.0);

        let d_q = alloc_and_upload(&dev, &q);
        let d_k = alloc_and_upload(&dev, &k);
        let d_v = alloc_and_upload(&dev, &v);
        let d_al = alloc_and_upload(&dev, &alpha);
        let d_be = alloc_and_upload(&dev, &beta);
        let d_dt = alloc_and_upload(&dev, &dt_bias);
        let d_sa = alloc_and_upload(&dev, &ssm_a);
        let d_state0 = alloc_and_upload(&dev, &state0);
        let mut kstates_init = filler.clone();
        let live_init = if rs_idx == 0 {
            state0.clone()
        } else {
            kstates_init[(rs_idx - 1) * slot_elems..rs_idx * slot_elems].copy_from_slice(&state0);
            seeded_f32_range(seed ^ 0x88, slot_elems, -5.0, 5.0)
        };
        let d_live = alloc_and_upload(&dev, &live_init);
        let d_ks = alloc_and_upload(&dev, &kstates_init);
        let d_idx = alloc_and_upload(&dev, &[rs_idx as i32]);
        let d_ao = dev.alloc(l * h * S_V * 4)?;
        let d_ref_state = dev.alloc(slot_elems * 4)?;
        let d_ref_ao = dev.alloc(l * h * S_V * 4)?;
        let d_tmp_ao = dev.alloc(l * h * S_V * 4)?;

        let cfg = LaunchCfg {
            grid: (h as u32, 1, S_V as u32 / WARPS_PER_BLOCK),
            block: (warp_size, WARPS_PER_BLOCK, 1),
            shared_bytes: 0,
        };
        let (hh, nr, ril) = (h as i32, n_rep as i32, i32::from(rep_inner));
        let stream = dev.default_stream();
        {
            let ll = l as i32;
            let ptrs = [
                d_q, d_k, d_v, d_al, d_be, d_dt, d_sa, d_live, d_ks, d_idx, d_ao,
            ]
            .map(ptr);
            let mut a = <B::Args<'_>>::default();
            for p in &ptrs {
                a.push_arg(p);
            }
            a.push_arg(&hh);
            a.push_arg(&ll);
            a.push_arg(&nr);
            a.push_arg(&ril);
            B::launch(&kernel, stream, cfg, a)?;
            stream.synchronize()?;
        }
        let got_ks = download(&dev, d_ks, slots * slot_elems)?;
        let got_ao = download(&dev, d_ao, l * h * S_V)?;
        let got_live = download(&dev, d_live, slot_elems)?;

        let base_run = |l_run: usize, ao: DevicePtr| -> Result<()> {
            let (bb, ll) = (1i32, l_run as i32);
            let ptrs = [
                d_q,
                d_k,
                d_v,
                d_al,
                d_be,
                d_dt,
                d_sa,
                d_state0,
                d_ref_state,
                ao,
            ]
            .map(ptr);
            let mut a = <B::Args<'_>>::default();
            for p in &ptrs {
                a.push_arg(p);
            }
            a.push_arg(&bb);
            a.push_arg(&hh);
            a.push_arg(&ll);
            a.push_arg(&nr);
            a.push_arg(&ril);
            B::launch(&base, stream, cfg, a)?;
            stream.synchronize()?;
            Ok(())
        };
        let mut bad = bit_mismatches(&got_live, &live_init);
        base_run(l, d_ref_ao)?;
        bad += bit_mismatches(&got_ao, &download(&dev, d_ref_ao, l * h * S_V)?);
        for t in 1..=l {
            base_run(t, d_tmp_ao)?;
            let want = download(&dev, d_ref_state, slot_elems)?;
            bad += bit_mismatches(&got_ks[(t - 1) * slot_elems..t * slot_elems], &want);
        }
        for slot in l..slots {
            bad += bit_mismatches(
                &got_ks[slot * slot_elems..(slot + 1) * slot_elems],
                &kstates_init[slot * slot_elems..(slot + 1) * slot_elems],
            );
        }

        unsafe {
            for (p, n) in [
                (d_q, q.len()),
                (d_k, k.len()),
                (d_v, v.len()),
                (d_al, alpha.len()),
                (d_be, beta.len()),
                (d_dt, dt_bias.len()),
                (d_sa, ssm_a.len()),
                (d_state0, slot_elems),
                (d_live, slot_elems),
                (d_ks, slots * slot_elems),
                (d_idx, 1),
                (d_ao, l * h * S_V),
                (d_ref_state, slot_elems),
                (d_ref_ao, l * h * S_V),
                (d_tmp_ao, l * h * S_V),
            ] {
                dev.dealloc(p, n * 4)?;
            }
        }
        results.push(ShapeResult {
            m: l,
            k: rs_idx,
            n: h,
            seed,
            max_rel_err: bad as f32,
            tolerance: 0.0,
            pass: bad == 0,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: STEM.to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "gdn_state_step_alphabeta".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
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
