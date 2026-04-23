//! Parity: `flambeau_mmvq_q8_0_dp4a_q8_1` vs the scalar `flambeau_mmvq_q8_0_q8_1`.
//!
//! Runs both kernels on the same quantised inputs and checks bit-level
//! equivalence (abs diff ≤ 5e-3 × max(|ref|, 1)). If they disagree the DP4A
//! variant has a bug.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every `unsafe {}` below is a kernel launch or `memcpy_async`               whose invariant is uniform: host/device buffers live for the bounded               `synchronize()` that follows, pointers are freshly allocated above, kernel               ABIs match kernels-hip. Per-site SAFETY comments would just repeat this."
)]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_0, BlockQ8_1, QK8_0};
use half::f16;

const QK: usize = QK8_0;

fn deterministic_rand_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            state ^= state >> 30;
            state = state.wrapping_mul(0xBF58476D1CE4E5B9);
            state ^= state >> 27;
            state = state.wrapping_mul(0x94D049BB133111EB);
            state ^= state >> 31;
            ((state as i32) as f32) / (i32::MAX as f32)
        })
        .collect()
}

fn quantize_q8_0(xs: &[f32]) -> Vec<BlockQ8_0> {
    assert_eq!(xs.len() % QK, 0);
    let nb = xs.len() / QK;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let chunk = &xs[i * QK..(i + 1) * QK];
        let amax = chunk.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let qs: [i8; 32] = {
            let mut arr = [0i8; 32];
            for (j, &x) in chunk.iter().enumerate() {
                arr[j] = (x * id).round().clamp(-128.0, 127.0) as i8;
            }
            arr
        };
        out.push(BlockQ8_0 { d: f16::from_f32(d), qs });
    }
    out
}

fn upload<T: Copy + 'static>(dev: &HipDevice, data: &[T]) -> DevicePtr {
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

fn run_kernel(stem: &str, entry: &'static str, n_rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let blocks_per_row = k / QK;
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let q_bytes = kernels::hsaco("quantize_q8_1").unwrap();
    let m_bytes = kernels::hsaco(stem).unwrap_or_else(|| panic!("kernel {stem} not compiled"));
    let q_module = HipModule::load(0, q_bytes).unwrap();
    let m_module = HipModule::load(0, m_bytes).unwrap();
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1").unwrap();
    let k_mmvq: HipKernel<'_> = m_module.kernel(entry).unwrap();

    let weights_f32 = deterministic_rand_f32(seed, n_rows * k);
    let x_blocks = quantize_q8_0(&weights_f32);
    let y_f32 = deterministic_rand_f32(seed.wrapping_add(7), k);

    let d_x = upload(&dev, &x_blocks);
    let d_y_f32 = upload(&dev, &y_f32);
    let y_blocks = k / QK;
    let d_y_q8_1 = dev.alloc(y_blocks * std::mem::size_of::<BlockQ8_1>()).unwrap();
    let d_dst = dev.alloc(n_rows * 4).unwrap();

    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_y_f32_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks as u32, QK as u32);
        unsafe { k_quantize.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_blocks_i = blocks_per_row as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&d_dst_ptr);
        args.push(&n_rows_i);
        args.push(&n_blocks_i);
        let cfg = LaunchCfg::one_d(n_rows as u32, 256);
        unsafe { k_mmvq.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let mut dst = vec![0.0f32; n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(dst.as_mut_ptr() as usize),
            d_dst,
            n_rows * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    unsafe {
        dev.dealloc(d_x, x_blocks.len() * std::mem::size_of::<BlockQ8_0>()).unwrap();
        dev.dealloc(d_y_f32, y_f32.len() * 4).unwrap();
        dev.dealloc(d_y_q8_1, y_blocks * std::mem::size_of::<BlockQ8_1>()).unwrap();
        dev.dealloc(d_dst, n_rows * 4).unwrap();
    }

    dst
}

#[test]
fn mmvq_q8_0_dp4a_matches_scalar_small() {
    if device_count().unwrap_or(0) < 1 {
        eprintln!("[skip] no HIP");
        return;
    }
    for &(n_rows, k, seed) in &[
        (4usize, 128usize, 0x111u64),
        (8, 2048, 0x222),
        (16, 4096, 0x333),
        // Shapes used by Qwen3.6-35B-A3B Q8_0 decode dispatch.
        (8192, 2048, 0xA11),   // attn_qkv
        (4096, 2048, 0xB22),   // attn_gate
        (2048, 4096, 0xC33),   // ssm_out
        (512, 2048, 0xD44),    // attn_k / ffn_shexp
    ] {
        let scalar = run_kernel("mmvq_q8_0", "flambeau_mmvq_q8_0_q8_1", n_rows, k, seed);
        let dp4a   = run_kernel("mmvq_q8_0_dp4a", "flambeau_mmvq_q8_0_dp4a_q8_1", n_rows, k, seed);
        eprintln!("[n_rows={n_rows} k={k}] scalar={:?} dp4a={:?}", scalar, dp4a);
        let max_abs = scalar.iter().zip(&dp4a).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = scalar.iter().fold(0.0f32, |a, &b| a.max(b.abs())).max(1.0);
        let rel = max_abs / max_ref;
        eprintln!("  max_abs_diff={max_abs:.4e}  rel={rel:.4e}");
        assert!(rel < 5e-3, "dp4a diverges from scalar: rel={rel:.3e}");
    }
}
