//! End-to-end RCCL vs CPU-host-bounce correctness — the cert gate.
//! Skips cleanly if fewer than 2 HIP devices are present. On the 4× MI50 rig
//! this exercises Mesh<2> and Mesh<4> for AllReduce / AllGather / AllToAll /
//! Broadcast against the reference implementation in `flambeau-runtime`.
//! Currently gated `#[ignore]`: `ncclAllReduce` segfaults on ROCm 7.2.1 +
//! 4× MI50 + PCIe (no xGMI). The pure-C equivalent crashes the same way, so
//! this is an environmental / RCCL-build issue, not an FFI bug. Re-enable
//! with `cargo test --features rccl -- --ignored` on a rig with a
//! known-working RCCL build.

#![cfg(feature = "rccl")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or `memcpy_async` \
              over host/device buffers that live for the bounded `synchronize()` that \
              follows; per-site SAFETY comments would just repeat this."
)]

use std::sync::Arc;
use std::thread;

use flambeau_backend_hip::{device_count, HipDevice, HipMesh};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_runtime::{
    collective::{AllGather, AllReduce, AllToAll, Broadcast, RefMesh},
    CollectiveCfg, CollectiveDType, RankId, ReduceOp,
};

fn mesh_size_or_skip(min: i32) -> Option<i32> {
    match device_count() {
        Ok(n) if n >= min => Some(n.min(4)),
        Ok(n) => {
            eprintln!("[skip] need >= {min} HIP devices, have {n}");
            None
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            None
        }
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice::<u8, f32>(b).to_vec()
}

// Run the same workload N ways on the RCCL mesh and the CPU ref mesh and
// compare element-wise.
fn run_both_meshes<F>(n: u32, init: impl Fn(u32) -> Vec<u8> + Send + Sync + 'static, run: F)
where
    F: Fn(RccCtx, RefCtx, Vec<u8>) + Send + Sync + 'static + Clone,
{
    let devs: Vec<i32> = (0..n as i32).collect();
    let hip_mesh = HipMesh::new(&devs).expect("HipMesh::new");
    let ref_mesh = RefMesh::new(n);

    let init = Arc::new(init);
    let mut handles = Vec::with_capacity(n as usize);
    for r in 0..n {
        let hip_m = Arc::clone(&hip_mesh);
        let ref_m = Arc::clone(&ref_mesh);
        let init = Arc::clone(&init);
        let run = run.clone();
        handles.push(thread::spawn(move || {
            let hip_rank = hip_m.rank_handle(RankId(r));
            let ref_rank = ref_m.rank_handle(RankId(r));
            let dev = HipDevice::new(hip_rank.device_id()).expect("HipDevice::new");
            dev.bind().unwrap();
            // `ncclCommInitRank` is called here, not in `HipMesh::new`, and
            // blocks until every rank's thread has called it. All N driver
            // threads must reach this point together.
            hip_rank.connect().expect("ncclCommInitRank");
            let input = (init)(r);
            run(
                RccCtx {
                    rank: hip_rank,
                    dev,
                },
                RefCtx { rank: ref_rank },
                input,
            );
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

struct RccCtx {
    rank: flambeau_backend_hip::HipRankHandle,
    dev: HipDevice,
}
struct RefCtx {
    rank: flambeau_runtime::RefRankHandle,
}

#[test]
fn all_reduce_sum_f32_mesh_vs_ref() {
    let Some(n) = mesh_size_or_skip(2) else { return };
    let n = n as u32;
    let elem_count = 1024usize;
    run_both_meshes(
        n,
        move |r| f32_bytes(&vec![r as f32 + 1.0; elem_count]),
        move |mut rcc, refc, input| {
            let cfg = CollectiveCfg::new(elem_count, CollectiveDType::F32, ReduceOp::Sum);

            // Reference
            let mut ref_buf = input.clone();
            refc.rank.all_reduce(&mut ref_buf, &cfg).unwrap();
            let ref_out = as_f32(&ref_buf);

            // RCCL via host-bounce
            let mut rcc_buf = input.clone();
            rcc.rank.all_reduce_host(&rcc.dev, &mut rcc_buf, &cfg).unwrap();
            let rcc_out = as_f32(&rcc_buf);

            assert_eq!(ref_out.len(), rcc_out.len());
            for (a, b) in ref_out.iter().zip(&rcc_out) {
                assert!((a - b).abs() < 1e-4, "mismatch: ref={a} rccl={b}");
            }
            // Let the unused `mut` for rcc go unused-warn-free.
            let _ = &mut rcc;
        },
    );
}

#[test]
fn all_gather_f32_mesh_vs_ref() {
    let Some(n) = mesh_size_or_skip(2) else { return };
    let n = n as u32;
    let elem_count = 8usize;
    run_both_meshes(
        n,
        move |r| f32_bytes(&(0..elem_count).map(|i| (r * 100 + i as u32) as f32).collect::<Vec<_>>()),
        move |rcc, refc, input| {
            let cfg = CollectiveCfg::new(elem_count, CollectiveDType::F32, ReduceOp::Sum);
            let recv_len = cfg.buffer_bytes() * rcc.rank.rank_count() as usize;

            let mut ref_recv = vec![0u8; recv_len];
            refc.rank.all_gather(&input, &mut ref_recv, &cfg).unwrap();
            let ref_out = as_f32(&ref_recv);

            // RCCL path — alloc send + recv on device, copy, collective, copy back.
            let dev = &rcc.dev;
            let d_send = dev.alloc(input.len()).unwrap();
            let d_recv = dev.alloc(recv_len).unwrap();
            unsafe {
                dev.memcpy_async(
                    dev.default_stream(),
                    CopyDirection::HostToDevice,
                    d_send,
                    DevicePtr(input.as_ptr() as usize),
                    input.len(),
                )
                .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            unsafe {
                rcc.rank
                    .all_gather_device(d_send, d_recv, &cfg, dev.default_stream())
                    .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            let mut host_recv = vec![0u8; recv_len];
            unsafe {
                dev.memcpy_async(
                    dev.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(host_recv.as_mut_ptr() as usize),
                    d_recv,
                    recv_len,
                )
                .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            unsafe {
                dev.dealloc(d_send, input.len()).unwrap();
                dev.dealloc(d_recv, recv_len).unwrap();
            }
            let rcc_out = as_f32(&host_recv);
            assert_eq!(ref_out, rcc_out);
        },
    );
}

#[test]
fn broadcast_f32_mesh_vs_ref() {
    let Some(n) = mesh_size_or_skip(2) else { return };
    let n = n as u32;
    let root = RankId(n - 1);
    let elem_count = 16usize;
    run_both_meshes(
        n,
        move |r| {
            if r == n - 1 {
                f32_bytes(&(0..elem_count).map(|i| i as f32 + 1.0).collect::<Vec<_>>())
            } else {
                vec![0u8; elem_count * 4]
            }
        },
        move |rcc, refc, input| {
            let cfg = CollectiveCfg::new(elem_count, CollectiveDType::F32, ReduceOp::Sum);

            let mut ref_buf = input.clone();
            refc.rank.broadcast(&mut ref_buf, root, &cfg).unwrap();

            let dev = &rcc.dev;
            let d_buf = dev.alloc(input.len()).unwrap();
            unsafe {
                dev.memcpy_async(
                    dev.default_stream(),
                    CopyDirection::HostToDevice,
                    d_buf,
                    DevicePtr(input.as_ptr() as usize),
                    input.len(),
                )
                .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            unsafe {
                rcc.rank
                    .broadcast_device(d_buf, root, &cfg, dev.default_stream())
                    .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            let mut back = vec![0u8; input.len()];
            unsafe {
                dev.memcpy_async(
                    dev.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(back.as_mut_ptr() as usize),
                    d_buf,
                    input.len(),
                )
                .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            unsafe {
                dev.dealloc(d_buf, input.len()).unwrap();
            }
            assert_eq!(as_f32(&ref_buf), as_f32(&back));
        },
    );
}

#[test]
fn all_to_all_f32_mesh_vs_ref() {
    let Some(n) = mesh_size_or_skip(2) else { return };
    let n = n as u32;
    let shard_elems = 4usize;
    run_both_meshes(
        n,
        move |r| {
            // Each rank sends shard r' the value (r * 100 + r').
            let mut v = Vec::with_capacity(shard_elems * n as usize);
            for peer in 0..n {
                for i in 0..shard_elems {
                    v.push((r * 100 + peer * 10 + i as u32) as f32);
                }
            }
            f32_bytes(&v)
        },
        move |rcc, refc, input| {
            let cfg = CollectiveCfg::new(shard_elems, CollectiveDType::F32, ReduceOp::Sum);
            let total = cfg.buffer_bytes() * rcc.rank.rank_count() as usize;

            let mut ref_recv = vec![0u8; total];
            refc.rank.all_to_all(&input, &mut ref_recv, &cfg).unwrap();

            let dev = &rcc.dev;
            let d_send = dev.alloc(total).unwrap();
            let d_recv = dev.alloc(total).unwrap();
            unsafe {
                dev.memcpy_async(
                    dev.default_stream(),
                    CopyDirection::HostToDevice,
                    d_send,
                    DevicePtr(input.as_ptr() as usize),
                    total,
                )
                .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            unsafe {
                rcc.rank
                    .all_to_all_device(d_send, d_recv, &cfg, dev.default_stream())
                    .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            let mut host_recv = vec![0u8; total];
            unsafe {
                dev.memcpy_async(
                    dev.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(host_recv.as_mut_ptr() as usize),
                    d_recv,
                    total,
                )
                .unwrap();
            }
            dev.default_stream().synchronize().unwrap();
            unsafe {
                dev.dealloc(d_send, total).unwrap();
                dev.dealloc(d_recv, total).unwrap();
            }
            assert_eq!(as_f32(&ref_recv), as_f32(&host_recv));
        },
    );
}
