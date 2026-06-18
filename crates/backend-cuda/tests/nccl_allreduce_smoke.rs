//! NCCL `DeviceAllReduce` smoke tests that fit a single GPU.
//!
//! - 1-rank world: AllReduce is identity (proves the FFI links + a real
//!   `ncclAllReduce` executes on the device).
//! - 2-ranks-on-1-GPU (`ncclCommInitAll([0,0])` + group API): proves the
//!   reduction math (1 + 2 = 3 on every rank). Skipped if NCCL declines two
//!   communicators on one device.
//!
//! The real 2-distinct-GPU PCIe-P2P cert (B6 gate) needs a 2nd NVIDIA card.
#![cfg(feature = "nccl")]

use flambeau_backend_cuda::CudaCluster;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_runtime::collective::DeviceAllReduce;
use flambeau_runtime::{Cluster, CollectiveDType};

fn upload(dev: &flambeau_backend_cuda::CudaDevice, data: &[f32]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    // SAFETY: host slice outlives the synchronous HtoD copy; d has `bytes`.
    unsafe {
        dev.memcpy_async(dev.default_stream(), CopyDirection::HostToDevice, d, DevicePtr(data.as_ptr() as usize), bytes)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn download(dev: &flambeau_backend_cuda::CudaDevice, ptr: DevicePtr, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    // SAFETY: out has n elems; ptr holds n f32.
    unsafe {
        dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost, DevicePtr(out.as_mut_ptr() as usize), ptr, n * 4)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

#[test]
fn nccl_single_rank_allreduce_is_identity() {
    let cluster = CudaCluster::new(&[0]).expect("CudaCluster init");
    assert_eq!(cluster.ranks(), 1);
    assert!(cluster.peer_access_full(), "1-rank cluster is vacuously peer-full");
    let dev = cluster.device(0);
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = upload(dev, &data);

    cluster
        .rank_handle(0)
        .all_reduce_sum(buf, data.len(), CollectiveDType::F32, dev, dev.default_stream())
        .expect("ncclAllReduce");
    dev.default_stream().synchronize().unwrap();

    assert_eq!(download(dev, buf, data.len()), data);
    // SAFETY: buf was alloc'd with 16 bytes.
    unsafe { dev.dealloc(buf, 16).unwrap() };
}

#[test]
fn nccl_two_rank_one_gpu_allreduce_sums() {
    let cluster = match CudaCluster::new(&[0, 0]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: NCCL declined two comms on one GPU: {e}");
            return;
        }
    };
    assert_eq!(cluster.ranks(), 2);
    let dev = cluster.device(0);
    let n = 4usize;
    let bufs: Vec<DevicePtr> = (0..2).map(|r| upload(dev, &vec![(r + 1) as f32; n])).collect();
    let streams: Vec<_> = (0..2).map(|_| dev.new_stream().unwrap()).collect();

    cluster
        .grouped(|| {
            for r in 0..2 {
                cluster
                    .rank_handle(r)
                    .all_reduce_sum(bufs[r], n, CollectiveDType::F32, dev, &streams[r])?;
            }
            Ok(())
        })
        .expect("grouped ncclAllReduce");
    for s in &streams {
        s.synchronize().unwrap();
    }

    for &b in &bufs {
        assert_eq!(download(dev, b, n), vec![3.0f32; n], "rank-0 (1) + rank-1 (2) = 3");
    }
    for b in bufs {
        // SAFETY: each buf was alloc'd with n*4 bytes.
        unsafe { dev.dealloc(b, n * 4).unwrap() };
    }
}
