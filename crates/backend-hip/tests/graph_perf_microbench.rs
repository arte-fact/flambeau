//! Time `N` small kernel launches via direct dispatch vs
//! `HipGraphExec::launch` replay. Reports per-launch CPU overhead delta.
//! Run with:
//!   cargo test -p flambeau-backend-hip --release \
//!     --test graph_perf_microbench -- --nocapture --ignored

use flambeau_backend_hip::{
    bind, HipDevice, HipGraphExec, HipModule, HipStream, KernelArgs, LaunchCfg,
};
use flambeau_core::{Device, Stream};

fn maybe_skip() -> bool {
    if std::env::var("HIP_SKIP_BUILD").is_ok() {
        eprintln!("HIP_SKIP_BUILD set — graph perf microbench skipped");
        return false;
    }
    bind(0).is_ok()
}

#[test]
#[ignore] // run explicitly: cargo test ... -- --ignored
fn graph_replay_vs_direct_dispatch_perf() {
    if !maybe_skip() {
        return;
    }
    let dev = HipDevice::new(0).expect("HipDevice::new(0)");

    let hsaco = flambeau_kernels_hip::hsaco("scale_f32").expect("scale_f32 hsaco");
    let module = HipModule::load(0, hsaco).expect("load scale_f32 module");
    let kernel = module.kernel("flambeau_scale_f32").expect("kernel fn");

    let n: usize = 1024;
    let bytes = n * 4;
    let x_dev = dev.alloc(bytes).unwrap();
    let y_dev = dev.alloc(bytes).unwrap();

    let stream = HipStream::new_non_blocking(0).unwrap();
    let x_ptr_u64: u64 = x_dev.as_usize() as u64;
    let y_ptr_u64: u64 = y_dev.as_usize() as u64;
    let n_i: i32 = n as i32;
    let scale: f32 = 1.0;

    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);

    // Number of kernel launches per replay/iteration.
    const N_LAUNCHES: usize = 40;
    // Number of iterations to time over.
    const ITERS: usize = 200;
    // Warmup iterations to skip.
    const WARMUP: usize = 20;

    // === Path A: direct dispatch ===
    let mut direct_us: Vec<u128> = Vec::with_capacity(ITERS);
    for it in 0..(WARMUP + ITERS) {
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..N_LAUNCHES {
            let mut args = KernelArgs::new();
            args.push(&x_ptr_u64);
            args.push(&y_ptr_u64);
            args.push(&n_i);
            args.push(&scale);
            // SAFETY: all arg storage references live for the duration of
            // launch; x/y/dev are live; cfg is well-formed.
            unsafe { kernel.launch(&stream, cfg, args).unwrap() };
        }
        stream.synchronize().unwrap();
        let dt = t0.elapsed().as_micros();
        if it >= WARMUP {
            direct_us.push(dt);
        }
    }

    // === Path B: graph capture + replay ===
    let cap_stream = HipStream::new_non_blocking(0).unwrap();
    let exec = HipGraphExec::capture(&cap_stream, |s| {
        for _ in 0..N_LAUNCHES {
            let mut args = KernelArgs::new();
            args.push(&x_ptr_u64);
            args.push(&y_ptr_u64);
            args.push(&n_i);
            args.push(&scale);
            unsafe { kernel.launch(s, cfg, args)? };
        }
        Ok(())
    })
    .expect("HipGraphExec::capture");
    assert_eq!(exec.num_kernel_nodes(), N_LAUNCHES);

    let mut graph_us: Vec<u128> = Vec::with_capacity(ITERS);
    for it in 0..(WARMUP + ITERS) {
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        exec.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let dt = t0.elapsed().as_micros();
        if it >= WARMUP {
            graph_us.push(dt);
        }
    }

    fn stats(v: &[u128]) -> (u128, u128, u128) {
        let mut s: Vec<u128> = v.to_vec();
        s.sort_unstable();
        let mean = s.iter().sum::<u128>() / s.len() as u128;
        let p50 = s[s.len() / 2];
        let p99 = s[s.len() * 99 / 100];
        (mean, p50, p99)
    }
    let (dm, dp50, dp99) = stats(&direct_us);
    let (gm, gp50, gp99) = stats(&graph_us);

    eprintln!("\n=== HIP graph perf microbench ({N_LAUNCHES} launches/iter, {ITERS} iters) ===");
    eprintln!(
        "direct dispatch: mean={dm}us  p50={dp50}us  p99={dp99}us  ({:.2} us/launch)",
        dm as f64 / N_LAUNCHES as f64
    );
    eprintln!(
        "graph replay   : mean={gm}us  p50={gp50}us  p99={gp99}us  ({:.2} us/launch)",
        gm as f64 / N_LAUNCHES as f64
    );
    let speedup = dm as f64 / gm as f64;
    let saved_us = (dm as i128) - (gm as i128);
    eprintln!("speedup: {:.2}x  (saved {} us per replay)", speedup, saved_us);

    // SAFETY: no outstanding work on the streams.
    unsafe {
        dev.dealloc(x_dev, bytes).unwrap();
        dev.dealloc(y_dev, bytes).unwrap();
    }
}
