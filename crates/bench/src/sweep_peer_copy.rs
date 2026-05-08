//! peer_copy_via_host correctness + bandwidth sweep.
//! Runs a round-trip between every (src, dst) pair in the cluster at
//! several payload sizes, asserts bit-exact payload preservation, and
//! reports effective single-direction bandwidth. With `device_count < 2`
//! the sweep degenerates to a same-rank dtod sanity check.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a kernel launch or a memcpy_async \
              over buffers allocated locally in the same function and freed before \
              return; invariant is uniform across all sites."
)]

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::rig;

/// Payload sizes we cert against — cover the PP hot path (≤ 512 KB) and
/// one bulk point to confirm the single-chunk path degrades gracefully
/// past the 4 MiB chunk threshold (future pipelined fast-path).
const SHAPES_BYTES: &[usize] = &[
    4 * 1024,        // hidden-state F16 at hidden=2048 (1 token)
    64 * 1024,       // ~16 tokens of hidden state
    256 * 1024,      // ~64 tokens, top end of decode prefill chunks
    4 * 1024 * 1024, // 4 MiB boundary
    16 * 1024 * 1024,
];

/// Number of warmup + measured iterations per (src, dst, shape).
const WARMUP: usize = 2;
const ITERS: usize = 8;

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        anyhow::bail!("no HIP devices");
    }
    let device_ids: Vec<i32> = (0..n).collect();
    let cluster = HipCluster::new(&device_ids)?;

    let mut results = Vec::new();
    for &bytes in SHAPES_BYTES {
        let (max_err, best_gbps) = run_shape(&cluster, bytes)?;
        let tol = 0.0; // bit-exact byte-pattern round-trip
        results.push(ShapeResult {
            m: bytes,
            k: 1,
            n: 1,
            seed: (bytes as u64) ^ 0xABCDEF,
            max_rel_err: max_err,
            tolerance: tol,
            pass: max_err == 0.0 && best_gbps > 0.0,
        });
        eprintln!(
            "peer_copy_via_host: {:>10} B → best {:>6.2} GB/s (across {} pair(s))",
            bytes,
            best_gbps,
            device_ids.len().max(1)
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "peer_copy_via_host_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "peer_copy_via_host".to_string(),
        dtype_weight: "U8".to_string(),
        dtype_activation: "U8".to_string(),
        tolerance_formula: "exact byte-pattern round-trip across all (src, dst) pairs".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(PmcSnapshot {
            vgpr_count: None,
            sgpr_count: None,
            waves_per_simd: None,
            mem_busy_pct: None,
            valu_busy_pct: None,
        }),
    };
    cluster.dispose()?;
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// Run the correctness + bandwidth sweep for one payload size. Returns
/// `(max byte-mismatch ratio, best single-direction bandwidth in GB/s)`.
fn run_shape(cluster: &HipCluster, bytes: usize) -> Result<(f32, f32)> {
    let ranks = cluster.ranks();

    // Host-side seeded pattern — every byte determined by offset so a
    // bit-wise round-trip check catches any byte-swapping bug.
    let pattern: Vec<u8> = (0..bytes).map(|i| (i.wrapping_mul(0x9E) as u8) ^ 0xA5).collect();

    // Per-rank allocations: `src_buf[r]` holds the outbound payload on
    // rank r; `dst_buf[r]` catches inbound data on rank r. We seed only
    // rank 0's src with `pattern`; `src_buf[r]` for r > 0 is whatever
    // lands there after a peer-copy from rank 0.
    let src_bufs: Vec<DevicePtr> = (0..ranks)
        .map(|r| cluster.device(r).alloc(bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let dst_bufs: Vec<DevicePtr> = (0..ranks)
        .map(|r| cluster.device(r).alloc(bytes))
        .collect::<Result<Vec<_>, _>>()?;

    // Prime rank 0's src with the pattern.
    {
        let dev0 = cluster.device(0);
        dev0.bind()?;
        // SAFETY: dev0 src_bufs[0] has `bytes` valid bytes; `pattern` has the same length.
        unsafe {
            dev0.memcpy_async(
                dev0.default_stream(),
                CopyDirection::HostToDevice,
                src_bufs[0],
                DevicePtr(pattern.as_ptr() as usize),
                bytes,
            )?;
        }
        dev0.default_stream().synchronize()?;
    }

    // One (src, dst) pair for each ordered rank pair. For ranks == 1 we
    // still exercise the same-rank DtoD short-circuit.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    if ranks == 1 {
        pairs.push((0, 0));
    } else {
        for src in 0..ranks {
            for dst in 0..ranks {
                if src != dst {
                    pairs.push((src, dst));
                }
            }
        }
    }

    let mut max_err = 0.0f32;
    let mut best_gbps = 0.0f32;

    for (src_rank, dst_rank) in pairs {
        // Ensure src rank holds the pattern: relay through rank 0 → src_rank
        // unless src_rank is already 0. Use the same peer-copy we're
        // certifying, which is fine because the correctness assert fires
        // on the final dst_buf.
        if src_rank != 0 {
            // SAFETY: both buffers are `bytes` long on their respective devices.
            unsafe {
                cluster.peer_copy_via_host(
                    src_bufs[src_rank],
                    src_rank,
                    src_bufs[0],
                    0,
                    bytes,
                )?;
            }
        }

        // Warmup.
        for _ in 0..WARMUP {
            unsafe {
                cluster.peer_copy_via_host(
                    dst_bufs[dst_rank],
                    dst_rank,
                    src_bufs[src_rank],
                    src_rank,
                    bytes,
                )?;
            }
        }
        // Measured iterations.
        let t0 = Instant::now();
        for _ in 0..ITERS {
            unsafe {
                cluster.peer_copy_via_host(
                    dst_bufs[dst_rank],
                    dst_rank,
                    src_bufs[src_rank],
                    src_rank,
                    bytes,
                )?;
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let gbps = (bytes as f32 * ITERS as f32) / dt / 1e9;
        if gbps > best_gbps {
            best_gbps = gbps;
        }

        // Read back dst_buf and compare against pattern.
        let mut got = vec![0u8; bytes];
        let dst_dev = cluster.device(dst_rank);
        dst_dev.bind()?;
        // SAFETY: dst_bufs[dst_rank] has `bytes` valid device bytes; `got` is the same length.
        unsafe {
            dst_dev.memcpy_async(
                dst_dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                dst_bufs[dst_rank],
                bytes,
            )?;
        }
        dst_dev.default_stream().synchronize()?;

        let mismatches = got.iter().zip(&pattern).filter(|(a, b)| a != b).count();
        if mismatches > 0 {
            max_err = max_err.max(mismatches as f32 / bytes as f32);
            eprintln!(
                "  peer_copy_via_host {src_rank} → {dst_rank} ({bytes} B): {mismatches} byte mismatches"
            );
        }
    }

    // Teardown.
    for (r, ptr) in src_bufs.into_iter().enumerate() {
        // SAFETY: came from `cluster.device(r).alloc(bytes)`.
        unsafe {
            cluster.device(r).dealloc(ptr, bytes)?;
        }
    }
    for (r, ptr) in dst_bufs.into_iter().enumerate() {
        unsafe {
            cluster.device(r).dealloc(ptr, bytes)?;
        }
    }

    Ok((max_err, best_gbps))
}

