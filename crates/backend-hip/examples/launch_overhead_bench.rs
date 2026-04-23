//! V2.1 step 1 — measure the full Rust-side launch path cost per kernel
//! invocation, compared against `rocprof`'s host-side `hipModuleLaunchKernel`
//! timing. The delta is our Rust-FFI overhead — the hypothesised residual
//! ~1.5-3 ms/token gap to llama.cpp per the V1.7.6 cert.
//!
//! Usage:
//!   cargo run --release -p flambeau-backend-hip --example launch_overhead_bench --features hip
//!
//! Then run it under rocprof:
//!   rocprof --hip-trace -o /tmp/lob/out.csv target/release/examples/launch_overhead_bench
//!   # inspect /tmp/lob/out.hip_stats.csv → hipModuleLaunchKernel avg_ns
//!   # compare that to "avg per-call" printed by this binary's wall-clock
//!
//! The kernel is the cheapest we have (`scale_f32`, 1-element buffer, no
//! real work) so kernel-side time is ~kernel-launch-latency and the
//! measurement is dominated by the launch path.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "launch-overhead micro-bench: every unsafe block is a `kernel.launch`, \
              `kernel.launch_raw`, or `dev.dealloc`; args + buffers are freshly \
              allocated in `main` and live until program exit. Per-site SAFETY \
              comments would just repeat the same invariant."
)]

use std::time::Instant;

use flambeau_backend_hip::{device_count, HipDevice, HipModule, KernelArgs, LaunchCfg};
#[expect(unused_imports, reason = "traits imported for type inference on `HipDevice` methods; not named directly")]
use flambeau_core::{Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;

fn main() -> anyhow::Result<()> {
    let n_iters: usize = std::env::var("FLAMBEAU_LOB_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let sync_every: usize = std::env::var("FLAMBEAU_LOB_SYNC_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 = sync only at end (best ILP)

    if device_count().unwrap_or(0) < 1 {
        eprintln!("no HIP device — skipping");
        return Ok(());
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let module_bytes = kernels::hsaco("scale_f32").expect("scale_f32 not compiled");
    let module = HipModule::load(0, module_bytes)?;
    let kernel = module.kernel("flambeau_scale_f32")?;

    // 1-element scratch buffer; scale by 1.0 = no observable work.
    let d_ptr = dev.alloc(4)?;
    let stream = dev.default_stream();
    let n_elems: i32 = 1;
    let scale: f32 = 1.0f32;
    let dst_u64: u64 = d_ptr.as_usize() as u64;

    // Warm-up: 1k launches so first-launch JIT/cache effects don't skew.
    for _ in 0..1000 {
        let mut args = KernelArgs::new();
        args.push(&dst_u64);      // x
        args.push(&dst_u64);      // y (same buffer, 1.0 scale = no-op)
        args.push(&n_elems);      // n
        args.push(&scale);        // scale
        let cfg = LaunchCfg::one_d(1, 32);
        unsafe { kernel.launch(stream, cfg, args)? };
    }
    stream.synchronize()?;

    let cfg = LaunchCfg::one_d(1, 32);

    // --- Path A: fresh KernelArgs per iter (V1 hot-path shape) ---
    let t0 = Instant::now();
    for i in 0..n_iters {
        let mut args = KernelArgs::new();
        args.push(&dst_u64);      // x
        args.push(&dst_u64);      // y
        args.push(&n_elems);      // n
        args.push(&scale);        // scale
        unsafe { kernel.launch(stream, cfg, args)? };
        if sync_every > 0 && (i + 1) % sync_every == 0 {
            stream.synchronize()?;
        }
    }
    stream.synchronize()?;
    let dt_a = t0.elapsed();

    // --- Path B: V2.1 pre-allocated pool, launch_raw on hot path ---
    //
    // Build the arg pointer array ONCE, then launch_raw in the loop.
    // Tests whether the Vec allocation + push calls are the dominant
    // Rust-side cost, or if it's the FFI crossing itself.
    let mut args_pool = KernelArgs::new();
    args_pool.push(&dst_u64);
    args_pool.push(&dst_u64);
    args_pool.push(&n_elems);
    args_pool.push(&scale);
    let args_raw = args_pool.raw_ptrs();

    let t0 = Instant::now();
    for i in 0..n_iters {
        unsafe { kernel.launch_raw(stream, cfg, args_raw)? };
        if sync_every > 0 && (i + 1) % sync_every == 0 {
            stream.synchronize()?;
        }
    }
    stream.synchronize()?;
    let dt_b = t0.elapsed();

    let ns_per_call_a = dt_a.as_nanos() as f64 / n_iters as f64;
    let ns_per_call_b = dt_b.as_nanos() as f64 / n_iters as f64;
    eprintln!(
        "[lob] {n_iters} iters, sync_every={sync_every}"
    );
    eprintln!(
        "[lob]  A (fresh KernelArgs):  {:>7.3} ms total → {:>6.3} µs/launch",
        dt_a.as_secs_f64() * 1000.0,
        ns_per_call_a / 1000.0
    );
    eprintln!(
        "[lob]  B (pool + launch_raw): {:>7.3} ms total → {:>6.3} µs/launch  (saves {:.3} µs/call)",
        dt_b.as_secs_f64() * 1000.0,
        ns_per_call_b / 1000.0,
        (ns_per_call_a - ns_per_call_b) / 1000.0
    );
    eprintln!(
        "[lob]  gfx906 rocprof hipModuleLaunchKernel avg (same binary earlier): ~2.25 µs"
    );

    unsafe { dev.dealloc(d_ptr, 4)? };
    Ok(())
}
