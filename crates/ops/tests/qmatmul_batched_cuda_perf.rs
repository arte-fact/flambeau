//! Dispatch-path microbench of the sm_86 decode / MTP-verify projections:
//! `qmatmul()` at m = 1 and m = 5 over the FFN shapes (17408 x 5120,
//! 5120 x 17408) for each weight dtype, so a profiler sees exactly the kernels
//! the runtime picks. Timings are wall-clock per call (launch + sync); run
//! under nsys for kernel time:
//!
//! ```sh
//! cargo test --release -p flambeau-ops --features cuda \
//!     --test qmatmul_batched_cuda_perf -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "perf bench — every unsafe block is a memcpy over locally-allocated bounded buffers."
)]

use std::time::Instant;

use flambeau_core::{CopyDirection, Device, DevicePtr, QDtype, Stream};
use flambeau_ops::cuda::{qmatmul::qmatmul, CudaDevice};
use flambeau_ops::{CudaOpsRegistry, MatmulShape, QmatmulBuffers};

/// (dtype, elements per weight block, bytes per weight block).
const DTYPES: &[(QDtype, usize, usize)] = &[
    (QDtype::IQ4_XS, 256, 136),
    (QDtype::Q5_K, 256, 176),
    (QDtype::Q4_K, 256, 144),
    (QDtype::Q6_K, 256, 210),
    (QDtype::Q3_K, 256, 110),
    (QDtype::IQ4_NL, 32, 18),
    (QDtype::Q8_0, 32, 34),
];
const SHAPES: &[(usize, usize)] = &[(17408, 5120), (5120, 17408)];
const MAX_COLS: usize = 8;
const ITERS: usize = 50;

fn upload(dev: &CudaDevice, bytes: &[u8]) -> DevicePtr {
    let d = dev.alloc(bytes.len()).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(bytes.as_ptr() as usize),
            bytes.len(),
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

/// Bytes with every fp16 field small and finite: the timing is data-independent,
/// but NaN / Inf scales would make the kernels' float paths unrepresentative.
fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as u8) & 0x3F
        })
        .collect()
}

/// `cols` columns of valid Q8_1 activation blocks (d, s = d * sum(qs), qs)
/// quantised on the host from values in [-1, 1).
fn q8_1_columns(seed: u64, cols: usize, k: usize) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(cols * (k / 32) * 36);
    for _ in 0..cols * (k / 32) {
        let x: Vec<f32> = (0..32)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let q: Vec<i8> = x.iter().map(|v| if d > 0.0 { (v / d).round() as i8 } else { 0 }).collect();
        let sum: i32 = q.iter().map(|&v| v as i32).sum();
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(d * sum as f32).to_le_bytes());
        out.extend(q.iter().map(|&v| v as u8));
    }
    out
}

#[test]
#[ignore = "perf bench: run explicitly on the RTX"]
fn qmatmul_m1_vs_m5_per_dtype() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let reg = CudaOpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();
    for &(dtype, blk_elems, blk_bytes) in DTYPES {
        for &(n, k) in SHAPES {
            let w = upload(&dev, &seeded_bytes(0xA1, n * (k / blk_elems) * blk_bytes));
            let y = upload(&dev, &q8_1_columns(0xB2, MAX_COLS, k));
            let dst = dev.alloc(MAX_COLS * n * 4).unwrap();
            let mut us = [0.0f64; 2];
            for (slot, m) in [1usize, 5].into_iter().enumerate() {
                let run = || {
                    qmatmul(
                        &reg,
                        stream,
                        QmatmulBuffers { weights: w, act_q8_1: y, act_q8_1_mmq: DevicePtr::NULL, dst },
                        MatmulShape { m, k, n },
                        dtype,
                    )
                    .unwrap()
                };
                for _ in 0..3 {
                    run();
                }
                stream.synchronize().unwrap();
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    run();
                }
                stream.synchronize().unwrap();
                us[slot] = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            }
            println!(
                "{:7} {n:>5}x{k:<5} m=1 {:8.1} us   m=5 {:8.1} us   ratio {:.2}",
                dtype.name(),
                us[0],
                us[1],
                us[1] / us[0]
            );
            unsafe {
                dev.dealloc(w, n * (k / blk_elems) * blk_bytes).unwrap();
                dev.dealloc(y, MAX_COLS * (k / 32) * 36).unwrap();
                dev.dealloc(dst, MAX_COLS * n * 4).unwrap();
            }
        }
    }
}

fn download(dev: &CudaDevice, src: DevicePtr, n: usize) -> Vec<f32> {
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

/// `qmatmul` at m columns (batched dispatch, chunked past 8) agrees with m
/// single-column calls on every column, for each dtype.
#[test]
#[ignore = "needs the RTX"]
fn batched_columns_match_single_column_calls() {
    const M_MAX: usize = 12;
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let reg = CudaOpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();
    let (n, k) = (1000usize, 2048usize);
    for &(dtype, blk_elems, blk_bytes) in DTYPES {
        let w = upload(&dev, &seeded_bytes(0xC3, n * (k / blk_elems) * blk_bytes));
        let y = upload(&dev, &q8_1_columns(0xD4, M_MAX, k));
        let dst = dev.alloc(M_MAX * n * 4).unwrap();
        let col_bytes = (k / 32) * 36;
        let mut reference = Vec::with_capacity(M_MAX * n);
        for c in 0..M_MAX {
            let bufs = QmatmulBuffers {
                weights: w,
                act_q8_1: y.offset_bytes(c * col_bytes),
                act_q8_1_mmq: DevicePtr::NULL,
                dst,
            };
            qmatmul(&reg, stream, bufs, MatmulShape { m: 1, k, n }, dtype).unwrap();
            reference.extend(download(&dev, dst, n));
        }
        let scale = reference.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        for m in [2usize, 5, 8, M_MAX] {
            let bufs = QmatmulBuffers { weights: w, act_q8_1: y, act_q8_1_mmq: DevicePtr::NULL, dst };
            qmatmul(&reg, stream, bufs, MatmulShape { m, k, n }, dtype).unwrap();
            let got = download(&dev, dst, m * n);
            let err = got
                .iter()
                .zip(&reference[..m * n])
                .fold(0.0f32, |a, (g, r)| a.max((g - r).abs()));
            assert!(err / scale < 1e-4, "{} m={m}: max err {err} (scale {scale})", dtype.name());
        }
        unsafe {
            dev.dealloc(w, n * (k / blk_elems) * blk_bytes).unwrap();
            dev.dealloc(y, M_MAX * col_bytes).unwrap();
            dev.dealloc(dst, M_MAX * n * 4).unwrap();
        }
    }
}
