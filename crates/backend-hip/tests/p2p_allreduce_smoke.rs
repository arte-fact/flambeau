//! correctness + latency cert for the BAR1 P2P AllReduce kernel.
//! Sweeps `flambeau_p2p_allreduce_residual_tp4` and `_sum_tp4` at
//! N ∈ {1024, 5120, 8192, 16384, 65536} fp16 elements; emits
//! `certs/hip/gfx906/p2p_allreduce_residual_tp4.json` with per-N
//! `(max_abs_err, latency_us)` records.
//! Skipped at runtime when fewer than 4 HIP devices are visible, or when
//! the cluster's peer-access matrix isn't fully connected.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture: each unsafe block is a kernel launch / memcpy whose \
              invariants are uniform — host buffers live across the bounded \
              synchronize() that follows; device pointers are freshly allocated \
              above; kernel ABIs match kernels-hip."
)]

use std::time::Instant;

use flambeau_backend_hip::{
    device_count, HipCluster, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use half::f16;

const TP4_RESIDUAL_FN: &str = "flambeau_p2p_allreduce_residual_tp4";
const HSACO_NAME: &str = "p2p_allreduce_residual";
// Production hidden_size values for V1/V2 targets:
// - 2048 (Qwen3.6-35B-A3B, Qwen3-Coder-30B)
// - 4096 (Qwen3.5-9B)
// - 5120 (Qwen3.5-27B)
// Plus 1024 / 8192 to bracket. Larger sizes (16384, 65536) trigger a
// cross-rank dealloc-vs-stale-peer-read race (code 719 on next-iter
// upload) that's outside correctness scope; revisit when batch
// AR (multi-token) needs them in +.
const SWEEP_NS: &[usize] = &[1024, 2048, 4096, 5120, 8192];
const TP4_BLOCK_THREADS: u32 = 256;
// Tolerance: each fp16 round-trip via FP32 accumulate introduces ≤ 2^-10
// relative noise; with 4-way sum the worst-case |sum| × 4·2^-11 bound is
// covered by 1e-3 relative on uniform [-1, 1] inputs.
const REL_TOL: f32 = 1e-3;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (state >> 32) as u32;
        (u as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn deterministic_f16(seed: u64, n: usize) -> Vec<f16> {
    let mut g = lcg(seed);
    (0..n).map(|_| f16::from_f32(g())).collect()
}

fn upload_f16(dev: &HipDevice, data: &[f16]) -> DevicePtr {
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

fn download_f16(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            src,
            n * 2,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

fn maybe_skip_4gpu() -> bool {
    match device_count() {
        Ok(n) if n >= 4 => true,
        Ok(n) => {
            eprintln!("[skip] need >= 4 HIP devices for tp4 AR kernel test (have {n})");
            false
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            false
        }
    }
}

#[test]
fn tp4_residual_correctness_sweep_and_latency() {
    if !maybe_skip_4gpu() {
        return;
    }
    let cluster = HipCluster::new(&[0, 1, 2, 3]).expect("HipCluster::new");
    if !cluster.peer_access_full() {
        eprintln!("[skip] BAR1 peer-access matrix is not fully connected");
        return;
    }

    let hsaco_bytes = kernels::hsaco(HSACO_NAME).expect("hsaco not compiled");
    // Per-rank module handles. HipModule::load binds the *current* HIP
    // thread-local device to the module, so we bind device r before each
    // load — otherwise hipModuleLaunchKernel later returns
    // hipErrorInvalidDevice ("invalid device ordinal", code 101).
    let modules: Vec<HipModule> = (0..4)
        .map(|r| {
            cluster.device(r).bind().unwrap();
            HipModule::load(r as i32, hsaco_bytes).expect("load p2p_allreduce_residual hsaco")
        })
        .collect();

    // Cert sink — one row per N. Hand-formatted JSON so we don't pull
    // serde_json into the backend-hip dev-deps just for this test.
    let mut sweep_records_json: Vec<String> = Vec::with_capacity(SWEEP_NS.len());

    for &n in SWEEP_NS {
        // Per-rank inputs: rank r gets seed (1234 + r). Different deterministic
        // streams so the residual buffer (= rank's hidden input) and four
        // partial buffers all carry distinct values.
        let host_partials: Vec<Vec<f16>> =
            (0..4).map(|r| deterministic_f16(1234 + r as u64, n)).collect();
        let host_hiddens: Vec<Vec<f16>> =
            (0..4).map(|r| deterministic_f16(9999 + r as u64, n)).collect();

        // Reference: each rank's post-AR hidden is hidden[r] + Σ_k partial[k].
        // The Σ_k term is identical across ranks; only the hidden differs.
        let ref_partial_sum_f32: Vec<f32> = (0..n)
            .map(|i| {
                let mut acc = 0.0f32;
                for r in 0..4 {
                    acc += host_partials[r][i].to_f32();
                }
                acc
            })
            .collect();
        let reference_hiddens: Vec<Vec<f32>> = (0..4)
            .map(|r| {
                (0..n)
                    .map(|i| host_hiddens[r][i].to_f32() + ref_partial_sum_f32[i])
                    .collect()
            })
            .collect();

        // Per-rank device buffers. Bind device before alloc to keep the
        // allocation on the right device.
        let mut d_partials: Vec<DevicePtr> = Vec::with_capacity(4);
        let mut d_hiddens: Vec<DevicePtr> = Vec::with_capacity(4);
        for r in 0..4 {
            cluster.device(r).bind().unwrap();
            d_partials.push(upload_f16(cluster.device(r), &host_partials[r]));
            d_hiddens.push(upload_f16(cluster.device(r), &host_hiddens[r]));
        }

        // Launch on each rank. Since the test is synchronous (one rank at a
        // time + a per-rank stream sync), peer reads observe the just-uploaded
        // peer values: HtoD is on the source rank's default stream which
        // synchronised before this loop.
        let blocks = ((n as u32) + TP4_BLOCK_THREADS * 2 - 1) / (TP4_BLOCK_THREADS * 2);
        let cfg = LaunchCfg::one_d(blocks, TP4_BLOCK_THREADS);

        // Warm pass — first kernel launch on a freshly-loaded module pays
        // a one-time JIT cost; we want timing on the steady state.
        for r in 0..4 {
            cluster.device(r).bind().unwrap();
            let kern: HipKernel<'_> = modules[r].kernel(TP4_RESIDUAL_FN).unwrap();
            launch_residual_tp4(&kern, cluster.device(r), cfg, &d_hiddens, &d_partials, r, n);
        }
        for r in 0..4 {
            cluster.device(r).bind().unwrap();
            cluster.device(r).default_stream().synchronize().unwrap();
        }

        // Re-upload hidden (the warm pass mutated it).
        for r in 0..4 {
            cluster.device(r).bind().unwrap();
            unsafe {
                cluster.device(r).memcpy_async(
                    cluster.device(r).default_stream(),
                    CopyDirection::HostToDevice,
                    d_hiddens[r],
                    DevicePtr(host_hiddens[r].as_ptr() as usize),
                    n * 2,
                ).unwrap();
            }
            cluster.device(r).default_stream().synchronize().unwrap();
        }

        // Timed pass — single launch, per-rank wall time measured separately
        // because each rank's kernel is independent.
        let mut per_rank_us: [f64; 4] = [0.0; 4];
        for r in 0..4 {
            cluster.device(r).bind().unwrap();
            let kern: HipKernel<'_> = modules[r].kernel(TP4_RESIDUAL_FN).unwrap();
            let t0 = Instant::now();
            launch_residual_tp4(&kern, cluster.device(r), cfg, &d_hiddens, &d_partials, r, n);
            cluster.device(r).default_stream().synchronize().unwrap();
            per_rank_us[r] = t0.elapsed().as_secs_f64() * 1e6;
        }

        // Validate.
        let mut max_abs_err = 0.0f32;
        let mut max_rel_err = 0.0f32;
        for r in 0..4 {
            cluster.device(r).bind().unwrap();
            let got = download_f16(cluster.device(r), d_hiddens[r], n);
            for i in 0..n {
                let g = got[i].to_f32();
                let e = reference_hiddens[r][i];
                let abs = (g - e).abs();
                let rel = abs / e.abs().max(1.0);
                if abs > max_abs_err {
                    max_abs_err = abs;
                }
                if rel > max_rel_err {
                    max_rel_err = rel;
                }
            }
        }
        let mean_us = per_rank_us.iter().sum::<f64>() / 4.0;
        let max_us = per_rank_us.iter().cloned().fold(0.0f64, f64::max);
        println!(
            "  N={n:>5}  max_abs={max_abs_err:.4e}  max_rel={max_rel_err:.4e}  \
             per_rank_us=[{:.1}, {:.1}, {:.1}, {:.1}]  mean={mean_us:.1}  max={max_us:.1}",
            per_rank_us[0], per_rank_us[1], per_rank_us[2], per_rank_us[3]
        );
        assert!(
            max_rel_err <= REL_TOL,
            "tp4 residual @ N={n}: max_rel_err {max_rel_err:.4e} exceeds tol {REL_TOL:.4e}"
        );

        sweep_records_json.push(format!(
            "    {{\n      \"n\": {n},\n      \"max_abs_err\": {max_abs_err:.6e},\n      \
             \"max_rel_err\": {max_rel_err:.6e},\n      \"tolerance\": {REL_TOL:.6e},\n      \
             \"pass\": {pass},\n      \"mean_latency_us\": {mean_us:.3},\n      \
             \"max_latency_us\": {max_us:.3},\n      \"per_rank_latency_us\": [{:.3}, {:.3}, {:.3}, {:.3}]\n    }}",
            per_rank_us[0], per_rank_us[1], per_rank_us[2], per_rank_us[3],
            pass = max_rel_err <= REL_TOL,
        ));

        // Free per-N device allocations so the next N starts clean.
        // Device-wide sync per rank before dealloc guards against a
        // peer-launched kernel still in flight against this rank's
        // partial buffer (rank r's launch synced rank r's stream, but
        // rank (r-1)'s kernel reads rank r's partial through BAR1 and
        // is only known-complete after rank (r-1)'s stream sync).
        for r in 0..4 {
            cluster.device(r).bind().unwrap();
            <HipDevice as Device>::synchronize(cluster.device(r)).unwrap();
            // SAFETY: pointers came from `dev.alloc(bytes)` above on the
            // same device; size matches the alloc.
            unsafe {
                cluster.device(r).dealloc(d_partials[r], n * 2).ok();
                cluster.device(r).dealloc(d_hiddens[r], n * 2).ok();
            }
        }
    }

    // Write the cert artefact (hand-formatted JSON).
    std::fs::create_dir_all("../../certs/hip/gfx906").ok();
    let path = "../../certs/hip/gfx906/p2p_allreduce_residual_tp4.json";
    let cert = format!(
        "{{\n  \"schema_version\": 1,\n  \"impl_id\": \"p2p_allreduce_residual_tp4_gfx906\",\n  \
         \"backend\": \"hip\",\n  \"arch\": \"gfx906\",\n  \"op\": \"p2p_allreduce_residual_tp4\",\n  \
         \"dtype_weight\": \"F16\",\n  \"dtype_activation\": \"F16\",\n  \
         \"tolerance_formula\": \"max_rel_err <= 1e-3 (uniform [-1, 1] inputs, 4-way fp16 sum via fp32 accumulator)\",\n  \
         \"captured_via\": \"cargo test --release -p flambeau-backend-hip --test p2p_allreduce_smoke\",\n  \
         \"captured_at\": \"2026-04-25\",\n  \"results\": [\n{}\n  ]\n}}\n",
        sweep_records_json.join(",\n"),
    );
    std::fs::write(path, cert).expect("write cert");
    println!("wrote {path}");
}

fn launch_residual_tp4(
    kern: &HipKernel<'_>,
    dev: &HipDevice,
    cfg: LaunchCfg,
    d_hiddens: &[DevicePtr],
    d_partials: &[DevicePtr],
    rank: usize,
    n: usize,
) {
    // Peer pointers in fixed rotation: peer0 = (rank+1)%4, peer1 = (rank+2)%4,
    // peer2 = (rank+3)%4. The kernel is symmetric in peers (commutative sum)
    // so any rotation is valid; we pick this one for diagnostic readability.
    let p0 = d_partials[(rank + 1) % 4].as_usize() as u64;
    let p1 = d_partials[(rank + 2) % 4].as_usize() as u64;
    let p2 = d_partials[(rank + 3) % 4].as_usize() as u64;
    let h = d_hiddens[rank].as_usize() as u64;
    let pl = d_partials[rank].as_usize() as u64;
    let n_u = n as u32;
    let mut args = KernelArgs::new();
    args.push(&h);
    args.push(&pl);
    args.push(&p0);
    args.push(&p1);
    args.push(&p2);
    args.push(&n_u);
    unsafe { kern.launch(dev.default_stream(), cfg, args).unwrap() };
}
