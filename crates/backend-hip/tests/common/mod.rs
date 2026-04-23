//! Shared helpers for the MMVQ correctness tests.
//! Not an external API — just DRY for the test-crate.

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_1, QK8_0, QK_K};

pub const QK8: usize = QK8_0;
pub const QK: usize = QK_K;

pub fn maybe_skip() -> bool {
    match device_count() {
        Ok(n) if n >= 1 => true,
        _ => {
            eprintln!("[skip] no HIP device");
            false
        }
    }
}

pub fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 24) as u8
        })
        .collect()
}

pub fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Q8_1 quantise-dequantise round-trip, matching the on-device kernel.
pub fn quantize_q8_1_roundtrip(xs: &[f32]) -> Vec<f32> {
    assert_eq!(xs.len() % QK8, 0);
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..(xs.len() / QK8) {
        let block = &xs[i * QK8..(i + 1) * QK8];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * QK8 + j] = (q as f32) * d;
        }
    }
    out
}

pub fn reference_matmul(weights_f32: &[f32], y_f32: &[f32], n_rows: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let row = &weights_f32[r * k..(r + 1) * k];
        let mut acc = 0.0f64;
        for j in 0..k {
            acc += (row[j] * y_f32[j]) as f64;
        }
        out[r] = acc as f32;
    }
    out
}

pub fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(1.0))
        .fold(0.0f32, f32::max)
}

pub fn alloc_and_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    // SAFETY: `d` is a fresh `bytes`-sized device allocation. `data` is a live
    // host slice sized exactly `bytes`. We `synchronize` immediately after, so
    // the host slice definitely outlives the copy.
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

pub fn download_f32(dev: &HipDevice, src: DevicePtr, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    // SAFETY: `out` is a fresh `len*4`-byte host allocation. `src` is the
    // caller's device pointer, required to be live and sized `≥ len*4`.
    // We `synchronize` immediately after, so `out` outlives the copy.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            src,
            len * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

/// Launch `flambeau_quantize_row_q8_1` over a freshly-allocated `d_y_q8_1`
/// buffer. Returns the allocated pointer; caller must dealloc.
pub fn quantize_q8_1_on_device(
    dev: &HipDevice,
    d_y_f32: DevicePtr,
    k: usize,
) -> (DevicePtr, usize) {
    let q_bytes = kernels::hsaco("quantize_q8_1").expect("quantize_q8_1 not compiled");
    let module = HipModule::load(dev.id(), q_bytes).unwrap();
    let kernel: HipKernel<'_> = module.kernel("flambeau_quantize_row_q8_1").unwrap();

    let y_blocks = k / QK8;
    let bytes = y_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(bytes).unwrap();

    let stream = dev.default_stream();
    let n_elems = k as i32;
    let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
    let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_y_f32_ptr);
    args.push(&d_y_q8_1_ptr);
    args.push(&n_elems);
    let cfg = LaunchCfg::one_d(y_blocks as u32, QK8 as u32);
    // SAFETY: kernel ABI is `(const f32*, BlockQ8_1*, int)`; `args` holds
    // `&d_y_f32_ptr`, `&d_y_q8_1_ptr`, `&n_elems` which live until the
    // synchronize below. Device pointers are both live and correctly sized
    // (`d_y_f32` by caller, `d_y_q8_1` by the alloc above).
    unsafe { kernel.launch(stream, cfg, args).unwrap() };
    stream.synchronize().unwrap();

    // `module` goes out of scope at fn end and unloads; the launch has
    // already completed, so no outstanding work references it.
    let _ = module;
    (d_y_q8_1, bytes)
}

pub fn cert_tol(k: usize) -> f32 {
    1e-2 * (k as f32 / 128.0).sqrt()
}
