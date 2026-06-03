//! Parity for `flambeau_gdn_state_step_alphabeta_f32_s128_batched_slots`
//! vs the per-slot reference (`flambeau_gdn_state_step_alphabeta_f32_s128`
//! run B times with B_kernel=1, once per independent slot state buffer).
//!
//! Same compute, same memory layout for q/k/v/alpha/beta/attn_out (all
//! slot-major `[B, L, H, S_v]`); only the per-slot state base pointer
//! is fetched indirectly through a `[B]` u64 array instead of computed
//! by stride. Outputs must be **bit-equal** since the FP32
//! accumulation order is identical.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture; same shape rationale as siblings"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "kernel-shape math bounded by GGUF dims"
)]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;

const S_V: usize = 128;
const WARP_SIZE: u32 = 64;
const WARPS_PER_BLOCK: u32 = 4;

fn maybe_skip() -> bool {
    matches!(device_count(), Ok(n) if n >= 1)
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
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

fn copy_back_f32(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            src,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

struct Outs {
    ref_state: Vec<Vec<f32>>, // per-slot state after per-slot call
    ref_attn: Vec<f32>,       // concatenated [B, L, H, S_v]
    bs_state: Vec<Vec<f32>>,  // per-slot state after batched-slots call
    bs_attn: Vec<f32>,
}

fn run_both(
    b: usize,
    h_v: usize,
    h_kv: usize,
    l: usize,
    n_rep: usize,
    rep_inner: bool,
    seed: u64,
) -> Outs {
    assert_eq!(h_v, h_kv * n_rep);
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let ref_module =
        HipModule::load(0, kernels::hsaco("gdn_state_step_alphabeta_f32").unwrap()).unwrap();
    let bs_module = HipModule::load(
        0,
        kernels::hsaco("gdn_state_step_alphabeta_f32_batched_slots").unwrap(),
    )
    .unwrap();
    let k_ref: HipKernel<'_> = ref_module
        .kernel("flambeau_gdn_state_step_alphabeta_f32_s128")
        .unwrap();
    let k_bs: HipKernel<'_> = bs_module
        .kernel("flambeau_gdn_state_step_alphabeta_f32_s128_batched_slots")
        .unwrap();

    // Activations slot-major [B, L, H, S_v] / [B, L, H_kv, S_v].
    let q_len = b * l * h_kv * S_V;
    let v_len = b * l * h_v * S_V;
    let alpha_len = b * l * h_v;
    let q_f32 = seeded_f32(seed, q_len);
    let k_f32 = seeded_f32(seed.wrapping_add(11), q_len);
    let v_f32 = seeded_f32(seed.wrapping_add(23), v_len);
    let alpha_f32 = seeded_f32(seed.wrapping_add(37), alpha_len);
    let beta_f32 = seeded_f32(seed.wrapping_add(53), alpha_len);
    let dt_bias_f32 = seeded_f32(seed.wrapping_add(71), h_v)
        .iter()
        .map(|x| x * 0.1)
        .collect::<Vec<_>>();
    let ssm_a_f32 = seeded_f32(seed.wrapping_add(83), h_v)
        .iter()
        .map(|x| -0.5 - 0.5 * x.abs())
        .collect::<Vec<_>>();

    let state_len_per_slot = h_v * S_V * S_V;
    let state_init: Vec<Vec<f32>> = (0..b)
        .map(|i| {
            seeded_f32(
                seed.wrapping_add(101).wrapping_add(i as u64),
                state_len_per_slot,
            )
        })
        .collect();

    let d_q = alloc_and_upload(&dev, &q_f32);
    let d_k = alloc_and_upload(&dev, &k_f32);
    let d_v = alloc_and_upload(&dev, &v_f32);
    let d_alpha = alloc_and_upload(&dev, &alpha_f32);
    let d_beta = alloc_and_upload(&dev, &beta_f32);
    let d_dt = alloc_and_upload(&dev, &dt_bias_f32);
    let d_sa = alloc_and_upload(&dev, &ssm_a_f32);

    let attn_len = b * l * h_v * S_V;
    let d_ref_attn = dev.alloc(attn_len * 4).unwrap();
    let d_bs_attn = dev.alloc(attn_len * 4).unwrap();

    // Per-slot state buffers (independent allocations) for REF run.
    let d_ref_state: Vec<DevicePtr> = state_init
        .iter()
        .map(|s| alloc_and_upload(&dev, s))
        .collect();
    // Per-slot state buffers (independent allocations) for BS run — fresh
    // copies of the same init data so both kernels start from the same
    // state.
    let d_bs_state: Vec<DevicePtr> = state_init
        .iter()
        .map(|s| alloc_and_upload(&dev, s))
        .collect();

    // Slot-pointer array for BS run (in-place: same array for in and out).
    let bs_ptr_u64: Vec<u64> = d_bs_state.iter().map(|p| p.as_usize() as u64).collect();
    let d_bs_ptrs = alloc_and_upload(&dev, &bs_ptr_u64);

    let h_kernel_i = h_v as i32;
    let l_i = l as i32;
    let n_rep_i = n_rep as i32;
    let rep_i: i32 = if rep_inner { 1 } else { 0 };

    let grid_z = (S_V as u32) / WARPS_PER_BLOCK;

    // REF: run per-slot with B=1.
    for (slot, &state_ptr) in d_ref_state.iter().enumerate().take(b) {
        let stream = dev.default_stream();
        let b_kernel_i = 1i32;
        let q_off: u64 = (d_q.as_usize() + slot * l * h_kv * S_V * 4) as u64;
        let k_off: u64 = (d_k.as_usize() + slot * l * h_kv * S_V * 4) as u64;
        let v_off: u64 = (d_v.as_usize() + slot * l * h_v * S_V * 4) as u64;
        let alpha_off: u64 = (d_alpha.as_usize() + slot * l * h_v * 4) as u64;
        let beta_off: u64 = (d_beta.as_usize() + slot * l * h_v * 4) as u64;
        let dt_ptr: u64 = d_dt.as_usize() as u64;
        let sa_ptr: u64 = d_sa.as_usize() as u64;
        let sin_ptr: u64 = state_ptr.as_usize() as u64;
        let sout_ptr: u64 = state_ptr.as_usize() as u64;
        let attn_off: u64 = (d_ref_attn.as_usize() + slot * l * h_v * S_V * 4) as u64;
        let mut args = KernelArgs::new();
        args.push(&q_off);
        args.push(&k_off);
        args.push(&v_off);
        args.push(&alpha_off);
        args.push(&beta_off);
        args.push(&dt_ptr);
        args.push(&sa_ptr);
        args.push(&sin_ptr);
        args.push(&sout_ptr);
        args.push(&attn_off);
        args.push(&b_kernel_i);
        args.push(&h_kernel_i);
        args.push(&l_i);
        args.push(&n_rep_i);
        args.push(&rep_i);
        let cfg = LaunchCfg {
            grid: (h_v as u32, 1, grid_z),
            block: (WARP_SIZE, WARPS_PER_BLOCK, 1),
            shared_bytes: 0,
        };
        unsafe { k_ref.launch(stream, cfg, args).unwrap() };
    }
    dev.default_stream().synchronize().unwrap();

    // BS: one launch with B = b.
    {
        let stream = dev.default_stream();
        let b_kernel_i = b as i32;
        let q_ptr: u64 = d_q.as_usize() as u64;
        let k_ptr: u64 = d_k.as_usize() as u64;
        let v_ptr: u64 = d_v.as_usize() as u64;
        let alpha_ptr: u64 = d_alpha.as_usize() as u64;
        let beta_ptr: u64 = d_beta.as_usize() as u64;
        let dt_ptr: u64 = d_dt.as_usize() as u64;
        let sa_ptr: u64 = d_sa.as_usize() as u64;
        let sin_arr: u64 = d_bs_ptrs.as_usize() as u64;
        let sout_arr: u64 = d_bs_ptrs.as_usize() as u64;
        let attn_ptr: u64 = d_bs_attn.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&q_ptr);
        args.push(&k_ptr);
        args.push(&v_ptr);
        args.push(&alpha_ptr);
        args.push(&beta_ptr);
        args.push(&dt_ptr);
        args.push(&sa_ptr);
        args.push(&sin_arr);
        args.push(&sout_arr);
        args.push(&attn_ptr);
        args.push(&b_kernel_i);
        args.push(&h_kernel_i);
        args.push(&l_i);
        args.push(&n_rep_i);
        args.push(&rep_i);
        let cfg = LaunchCfg {
            grid: (h_v as u32, b as u32, grid_z),
            block: (WARP_SIZE, WARPS_PER_BLOCK, 1),
            shared_bytes: 0,
        };
        unsafe { k_bs.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let ref_state: Vec<Vec<f32>> = d_ref_state
        .iter()
        .map(|p| copy_back_f32(&dev, *p, state_len_per_slot))
        .collect();
    let ref_attn = copy_back_f32(&dev, d_ref_attn, attn_len);
    let bs_state: Vec<Vec<f32>> = d_bs_state
        .iter()
        .map(|p| copy_back_f32(&dev, *p, state_len_per_slot))
        .collect();
    let bs_attn = copy_back_f32(&dev, d_bs_attn, attn_len);

    unsafe {
        for p in &d_ref_state {
            dev.dealloc(*p, state_len_per_slot * 4).unwrap();
        }
        for p in &d_bs_state {
            dev.dealloc(*p, state_len_per_slot * 4).unwrap();
        }
        dev.dealloc(d_bs_ptrs, bs_ptr_u64.len() * 8).unwrap();
        dev.dealloc(d_q, q_f32.len() * 4).unwrap();
        dev.dealloc(d_k, k_f32.len() * 4).unwrap();
        dev.dealloc(d_v, v_f32.len() * 4).unwrap();
        dev.dealloc(d_alpha, alpha_f32.len() * 4).unwrap();
        dev.dealloc(d_beta, beta_f32.len() * 4).unwrap();
        dev.dealloc(d_dt, dt_bias_f32.len() * 4).unwrap();
        dev.dealloc(d_sa, ssm_a_f32.len() * 4).unwrap();
        dev.dealloc(d_ref_attn, attn_len * 4).unwrap();
        dev.dealloc(d_bs_attn, attn_len * 4).unwrap();
    }

    Outs {
        ref_state,
        ref_attn,
        bs_state,
        bs_attn,
    }
}

fn assert_bit_equal_vec(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut diffs = 0;
    let mut worst = (0usize, 0.0f32);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() {
            diffs += 1;
            let e = (x - y).abs();
            if e > worst.1 {
                worst = (i, e);
            }
        }
    }
    if diffs > 0 {
        eprintln!(
            "[{label}] {diffs}/{} bits differ; worst idx={}, ref={}, bs={}, |Δ|={:.3e}",
            a.len(),
            worst.0,
            a[worst.0],
            b[worst.0],
            worst.1
        );
    }
    assert_eq!(
        diffs, 0,
        "{label}: outputs not bit-equal ({diffs} mismatches)"
    );
}

#[test]
fn parity_b2_l1_h32_rep1() {
    if !maybe_skip() {
        eprintln!("[skip] no HIP device");
        return;
    }
    // L=1 is the decode case. H=32, n_rep=1 → H_kv=32.
    let outs = run_both(2, 32, 32, 1, 1, false, 0xC0FFEE);
    for s in 0..2 {
        assert_bit_equal_vec(
            &format!("state slot {s}"),
            &outs.ref_state[s],
            &outs.bs_state[s],
        );
    }
    assert_bit_equal_vec("attn", &outs.ref_attn, &outs.bs_attn);
}

#[test]
fn parity_b4_l1_h32_rep4_outer() {
    if !maybe_skip() {
        return;
    }
    // n_rep=4 / rep_outer (qwen35moe). H=32, H_kv=8.
    let outs = run_both(4, 32, 8, 1, 4, false, 0xFEED_FACE);
    for s in 0..4 {
        assert_bit_equal_vec(
            &format!("state slot {s}"),
            &outs.ref_state[s],
            &outs.bs_state[s],
        );
    }
    assert_bit_equal_vec("attn", &outs.ref_attn, &outs.bs_attn);
}

#[test]
fn parity_b4_l1_h32_rep4_inner() {
    if !maybe_skip() {
        return;
    }
    // n_rep=4 / rep_inner (qwen3next).
    let outs = run_both(4, 32, 8, 1, 4, true, 0x1234_5678);
    for s in 0..4 {
        assert_bit_equal_vec(
            &format!("state slot {s}"),
            &outs.ref_state[s],
            &outs.bs_state[s],
        );
    }
    assert_bit_equal_vec("attn", &outs.ref_attn, &outs.bs_attn);
}

#[test]
fn parity_b3_l1_h16_rep1() {
    if !maybe_skip() {
        return;
    }
    // Odd B count.
    let outs = run_both(3, 16, 16, 1, 1, false, 0xABCDEF01);
    for s in 0..3 {
        assert_bit_equal_vec(
            &format!("state slot {s}"),
            &outs.ref_state[s],
            &outs.bs_state[s],
        );
    }
    assert_bit_equal_vec("attn", &outs.ref_attn, &outs.bs_attn);
}
