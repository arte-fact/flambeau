//! Parity for `gdn_conv_trio_decode_f32_batched_slots` vs the per-slot
//! reference (`assemble_conv_input` memcpy + `causal_conv1d_f32` + shift
//! memcpy, run B times against B independent history buffers). Bit-equal
//! at FP32 — same op order, just one launch instead of 3N.

#![expect(clippy::undocumented_unsafe_blocks, reason = "test fixture; same shape rationale as siblings")]
#![expect(clippy::cast_possible_wrap, reason = "kernel-shape math bounded by GGUF dims")]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;

fn maybe_skip() -> bool {
    matches!(device_count(), Ok(n) if n >= 1)
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    ref_conv_out: Vec<f32>,
    ref_history: Vec<Vec<f32>>,
    bs_conv_out: Vec<f32>,
    bs_history: Vec<Vec<f32>>,
}

fn run_both(n_slots: usize, conv_channels: usize, conv_kernel: usize, seed: u64) -> Outs {
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let conv1d_module = HipModule::load(0, kernels::hsaco("causal_conv1d_f32").unwrap()).unwrap();
    let trio_module = HipModule::load(0, kernels::hsaco("gdn_conv_trio_decode_f32_batched_slots").unwrap()).unwrap();
    let k_conv1d: HipKernel<'_> = conv1d_module.kernel("flambeau_causal_conv1d_f32").unwrap();
    let k_trio: HipKernel<'_> = trio_module.kernel("flambeau_gdn_conv_trio_decode_f32_batched_slots").unwrap();

    let qkv = seeded_f32(seed, n_slots * conv_channels);
    let weight = seeded_f32(seed.wrapping_add(7), conv_channels * conv_kernel);
    let hist_init: Vec<Vec<f32>> = (0..n_slots)
        .map(|i| {
            seeded_f32(
                seed.wrapping_add(101).wrapping_add(i as u64),
                (conv_kernel - 1) * conv_channels,
            )
        })
        .collect();

    let d_qkv = alloc_and_upload(&dev, &qkv);
    let d_weight = alloc_and_upload(&dev, &weight);

    // REF buffers
    let d_ref_history: Vec<DevicePtr> =
        hist_init.iter().map(|h| alloc_and_upload(&dev, h)).collect();
    let d_ref_conv_out = dev.alloc(n_slots * conv_channels * 4).unwrap();
    let d_ref_conv_input = dev.alloc(conv_kernel * conv_channels * 4).unwrap();

    // REF: per-slot trio
    let row_bytes = conv_channels * 4;
    {
        let stream = dev.default_stream();
        for s in 0..n_slots {
            // assemble: K-1 history rows then 1 qkv row
            unsafe {
                dev.memcpy_async(
                    stream,
                    CopyDirection::DeviceToDevice,
                    d_ref_conv_input,
                    d_ref_history[s],
                    (conv_kernel - 1) * row_bytes,
                )
                .unwrap();
                dev.memcpy_async(
                    stream,
                    CopyDirection::DeviceToDevice,
                    DevicePtr(d_ref_conv_input.as_usize() + (conv_kernel - 1) * row_bytes),
                    DevicePtr(d_qkv.as_usize() + s * row_bytes),
                    row_bytes,
                )
                .unwrap();
            }
            // causal_conv1d_f32: y[t=0, c] = Σ_k w[c,k] · conv_input[k, c]
            let n_new_i: i32 = 1;
            let cc_i = conv_channels as i32;
            let ck_i = conv_kernel as i32;
            let ci_ptr: u64 = d_ref_conv_input.as_usize() as u64;
            let w_ptr: u64 = d_weight.as_usize() as u64;
            let y_ptr: u64 = (d_ref_conv_out.as_usize() + s * row_bytes) as u64;
            let mut args = KernelArgs::new();
            args.push(&ci_ptr);
            args.push(&w_ptr);
            args.push(&y_ptr);
            args.push(&n_new_i);
            args.push(&cc_i);
            args.push(&ck_i);
            let threads: u32 = 256;
            let cfg = LaunchCfg {
                grid: ((conv_channels as u32).div_ceil(threads), 1, 1),
                block: (threads, 1, 1),
                shared_bytes: 0,
            };
            unsafe { k_conv1d.launch(stream, cfg, args).unwrap() };
            // shift: history = last K-1 rows of conv_input
            unsafe {
                dev.memcpy_async(
                    stream,
                    CopyDirection::DeviceToDevice,
                    d_ref_history[s],
                    DevicePtr(d_ref_conv_input.as_usize() + row_bytes),
                    (conv_kernel - 1) * row_bytes,
                )
                .unwrap();
            }
        }
        stream.synchronize().unwrap();
    }

    // BS buffers (fresh history copies)
    let d_bs_history: Vec<DevicePtr> =
        hist_init.iter().map(|h| alloc_and_upload(&dev, h)).collect();
    let d_bs_conv_out = dev.alloc(n_slots * conv_channels * 4).unwrap();
    let bs_ptr_u64: Vec<u64> = d_bs_history.iter().map(|p| p.as_usize() as u64).collect();
    let d_bs_ptrs = alloc_and_upload(&dev, &bs_ptr_u64);
    {
        let stream = dev.default_stream();
        let n_slots_i = n_slots as i32;
        let cc_i = conv_channels as i32;
        let ck_i = conv_kernel as i32;
        let ptrs_arr: u64 = d_bs_ptrs.as_usize() as u64;
        let q_ptr: u64 = d_qkv.as_usize() as u64;
        let w_ptr: u64 = d_weight.as_usize() as u64;
        let o_ptr: u64 = d_bs_conv_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&ptrs_arr);
        args.push(&q_ptr);
        args.push(&w_ptr);
        args.push(&o_ptr);
        args.push(&n_slots_i);
        args.push(&cc_i);
        args.push(&ck_i);
        let threads: u32 = 256;
        let cfg = LaunchCfg {
            grid: (
                (conv_channels as u32).div_ceil(threads),
                n_slots as u32,
                1,
            ),
            block: (threads, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_trio.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let ref_conv_out = copy_back_f32(&dev, d_ref_conv_out, n_slots * conv_channels);
    let ref_history: Vec<Vec<f32>> = d_ref_history
        .iter()
        .map(|p| copy_back_f32(&dev, *p, (conv_kernel - 1) * conv_channels))
        .collect();
    let bs_conv_out = copy_back_f32(&dev, d_bs_conv_out, n_slots * conv_channels);
    let bs_history: Vec<Vec<f32>> = d_bs_history
        .iter()
        .map(|p| copy_back_f32(&dev, *p, (conv_kernel - 1) * conv_channels))
        .collect();

    unsafe {
        for p in &d_ref_history { dev.dealloc(*p, hist_init[0].len() * 4).unwrap(); }
        for p in &d_bs_history { dev.dealloc(*p, hist_init[0].len() * 4).unwrap(); }
        dev.dealloc(d_qkv, qkv.len() * 4).unwrap();
        dev.dealloc(d_weight, weight.len() * 4).unwrap();
        dev.dealloc(d_ref_conv_out, n_slots * conv_channels * 4).unwrap();
        dev.dealloc(d_ref_conv_input, conv_kernel * conv_channels * 4).unwrap();
        dev.dealloc(d_bs_conv_out, n_slots * conv_channels * 4).unwrap();
        dev.dealloc(d_bs_ptrs, bs_ptr_u64.len() * 8).unwrap();
    }

    Outs { ref_conv_out, ref_history, bs_conv_out, bs_history }
}

fn assert_bit_equal(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut diffs = 0usize;
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
            "[{label}] {diffs}/{} bits differ; worst idx={} (ref={}, bs={}, |Δ|={:.3e})",
            a.len(),
            worst.0,
            a[worst.0],
            b[worst.0],
            worst.1
        );
    }
    assert_eq!(diffs, 0, "{label}: outputs not bit-equal");
}

#[test]
fn parity_b2_cc12288_k4() {
    if !maybe_skip() {
        return;
    }
    // Qwen3.6-35B-A3B GDN shape: conv_channels=12288, conv_kernel=4.
    let outs = run_both(2, 12288, 4, 0xC0FFEE);
    assert_bit_equal("conv_out", &outs.ref_conv_out, &outs.bs_conv_out);
    for s in 0..2 {
        assert_bit_equal(
            &format!("history slot {s}"),
            &outs.ref_history[s],
            &outs.bs_history[s],
        );
    }
}

#[test]
fn parity_b4_cc12288_k4_qwen36_35b_shape() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(4, 12288, 4, 0xFEED_FACE);
    assert_bit_equal("conv_out", &outs.ref_conv_out, &outs.bs_conv_out);
    for s in 0..4 {
        assert_bit_equal(
            &format!("history slot {s}"),
            &outs.ref_history[s],
            &outs.bs_history[s],
        );
    }
}

#[test]
fn parity_b3_cc4096_k4() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(3, 4096, 4, 0x1234_5678);
    assert_bit_equal("conv_out", &outs.ref_conv_out, &outs.bs_conv_out);
    for s in 0..3 {
        assert_bit_equal(
            &format!("history slot {s}"),
            &outs.ref_history[s],
            &outs.bs_history[s],
        );
    }
}

#[test]
fn parity_b1_cc12288_k4() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(1, 12288, 4, 0xABCDEF01);
    assert_bit_equal("conv_out", &outs.ref_conv_out, &outs.bs_conv_out);
    assert_bit_equal("history slot 0", &outs.ref_history[0], &outs.bs_history[0]);
}
