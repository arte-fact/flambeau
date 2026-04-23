//! Standalone RCCL AllReduce — no CPU-ref interleaving, single shot.
//!
//! Gated `#[ignore]` because `ncclAllReduce` segfaults on the ROCm 7.2.1 +
//! 4× MI50 rig we currently ship against. The pure-C equivalent crashes the
//! same way, so the issue is environmental (likely P2P / PCIe topology), not
//! our FFI. Re-enable with `cargo test --features rccl -- --ignored` on a
//! rig with a known-working RCCL build — hipdarc's integration tests are the
//! closest known-good reference.

#![cfg(feature = "rccl")]

use std::sync::Arc;
use std::thread;

use flambeau_backend_hip::{device_count, HipDevice, HipMesh};
use flambeau_runtime::{CollectiveCfg, CollectiveDType, RankId, ReduceOp};

#[test]
fn rccl_all_reduce_sum_f32_mesh_4_standalone() {
    let n = match device_count() {
        Ok(n) if n >= 2 => n.min(4) as u32,
        Ok(n) => {
            eprintln!("[skip] need >= 2 HIP devices, have {n}");
            return;
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            return;
        }
    };
    let devs: Vec<i32> = (0..n as i32).collect();
    let mesh = HipMesh::new(&devs).expect("HipMesh::new");
    let elem_count = 256usize;
    let cfg = CollectiveCfg::new(elem_count, CollectiveDType::F32, ReduceOp::Sum);

    let mut handles = Vec::with_capacity(n as usize);
    for r in 0..n {
        let mesh = Arc::clone(&mesh);
        handles.push(thread::spawn(move || {
            let rank = mesh.rank_handle(RankId(r));
            let dev = HipDevice::new(rank.device_id()).unwrap();
            dev.bind().unwrap();
            rank.connect().unwrap();
            let mut buf: Vec<u8> = bytemuck::cast_slice(&vec![r as f32 + 1.0; elem_count]).to_vec();
            rank.all_reduce_host(&dev, &mut buf, &cfg).unwrap();
            let out: &[f32] = bytemuck::cast_slice(&buf);
            let expected: f32 = (1..=n).map(|i| i as f32).sum();
            for v in out {
                assert!((*v - expected).abs() < 1e-4, "rank {r}: got {v} want {expected}");
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
