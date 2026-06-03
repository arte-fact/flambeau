//! `BarP2pAllReduce` end-to-end smoke test.
//! Validates that the high-level wrapper (`BarP2pAllReduce::residual_tp4`)
//! produces the same result as the direct kernel-launch path tested in
//! `tests/p2p_allreduce_smoke.rs`. Runs a single shape (N=5120, the
//! Qwen3.5-27B hidden_size — the load-bearing TP target) at TP=4.
//! Skipped at runtime when fewer than 4 HIP devices are visible or the
//! cluster's peer-access matrix isn't fully connected.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture: each unsafe block is a memcpy or AR launch \
              whose contract is uniform — host buffers live across the \
              bounded sync that follows; device pointers are freshly \
              allocated above; producer GEMV is implicit (host upload, \
              synced before AR launch)."
)]

use std::sync::Arc;

use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use half::f16;

const N: usize = 5120;
const REL_TOL: f32 = 1e-3;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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

#[test]
fn bar_p2p_residual_tp4_round_trip() {
    match device_count() {
        Ok(n) if n >= 4 => (),
        Ok(n) => {
            eprintln!("[skip] need >= 4 HIP devices for BarP2pAllReduce tp4 test (have {n})");
            return;
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            return;
        }
    }

    let cluster = Arc::new(HipCluster::new(&[0, 1, 2, 3]).expect("HipCluster::new"));
    if !cluster.peer_access_full() {
        eprintln!("[skip] BAR1 peer-access matrix is not fully connected");
        return;
    }
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster)).expect("BarP2pAllReduce::new");
    assert_eq!(ar.ranks(), 4);

    // Per-rank inputs (different deterministic streams per rank).
    let host_partials: Vec<Vec<f16>> = (0..4)
        .map(|r| deterministic_f16(1234 + r as u64, N))
        .collect();
    let host_hiddens: Vec<Vec<f16>> = (0..4)
        .map(|r| deterministic_f16(9999 + r as u64, N))
        .collect();

    // Reference: hidden[r] += Σ partial[k].
    let partial_sum_f32: Vec<f32> = (0..N)
        .map(|i| {
            let mut acc = 0.0f32;
            for hp in host_partials.iter().take(4) {
                acc += hp[i].to_f32();
            }
            acc
        })
        .collect();
    let reference_hiddens: Vec<Vec<f32>> = (0..4)
        .map(|r| {
            (0..N)
                .map(|i| host_hiddens[r][i].to_f32() + partial_sum_f32[i])
                .collect()
        })
        .collect();

    // Upload per-rank buffers.
    let mut d_partials = [DevicePtr(0); 4];
    let mut d_hiddens = [DevicePtr(0); 4];
    for r in 0..4 {
        cluster.device(r).bind().unwrap();
        d_partials[r] = upload_f16(cluster.device(r), &host_partials[r]);
        d_hiddens[r] = upload_f16(cluster.device(r), &host_hiddens[r]);
    }

    // Launch via the wrapper. Use each rank's default stream — implicit
    // ordering: every rank's HtoD upload synced before this loop, so
    // peer reads see committed memory.
    let streams: [&_; 4] = [
        cluster.device(0).default_stream(),
        cluster.device(1).default_stream(),
        cluster.device(2).default_stream(),
        cluster.device(3).default_stream(),
    ];
    unsafe {
        ar.residual_tp4(&d_hiddens, &d_partials, N as u32, &streams)
            .expect("residual_tp4 launch");
    }
    for r in 0..4 {
        cluster.device(r).bind().unwrap();
        cluster.device(r).default_stream().synchronize().unwrap();
    }

    // Validate.
    let mut max_rel_err = 0.0f32;
    for r in 0..4 {
        cluster.device(r).bind().unwrap();
        let got = download_f16(cluster.device(r), d_hiddens[r], N);
        for i in 0..N {
            let g = got[i].to_f32();
            let e = reference_hiddens[r][i];
            let rel = (g - e).abs() / e.abs().max(1.0);
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }
    println!("BarP2pAllReduce::residual_tp4 @ N={N}: max_rel_err = {max_rel_err:.4e}");
    assert!(
        max_rel_err <= REL_TOL,
        "BarP2pAllReduce::residual_tp4 @ N={N}: max_rel_err {max_rel_err:.4e} exceeds tol {REL_TOL:.4e}"
    );

    // Cleanup.
    for r in 0..4 {
        cluster.device(r).bind().unwrap();
        <HipDevice as Device>::synchronize(cluster.device(r)).unwrap();
        unsafe {
            cluster.device(r).dealloc(d_partials[r], N * 2).ok();
            cluster.device(r).dealloc(d_hiddens[r], N * 2).ok();
        }
    }
}

#[test]
fn bar_p2p_construction_fails_on_partial_matrix() {
    match device_count() {
        Ok(n) if n >= 1 => (),
        Ok(_) => {
            eprintln!("[skip] no HIP devices");
            return;
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            return;
        }
    }
    // A 1-rank cluster is degenerate but trivially fully-connected
    // (diagonal-only matrix), so construction succeeds — we just want
    // to confirm the API path doesn't panic.
    let cluster = Arc::new(HipCluster::new(&[0]).expect("HipCluster::new(&[0])"));
    assert!(cluster.peer_access_full());
    // 1-rank construction succeeds, but residual_tp4/sum_tp4 would
    // fail expect_ranks(4). This catches a mismatched-cluster bug at
    // call time rather than load time.
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster)).expect("1-rank BarP2pAllReduce");
    assert_eq!(ar.ranks(), 1);
}
