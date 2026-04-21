//! Smallest possible RCCL test — just init a 2-GPU mesh and destroy it.
//! Useful for isolating init-vs-collective-vs-destroy regressions.

#![cfg(feature = "rccl")]

use flambeau_backend_hip::{device_count, HipMesh};

#[test]
fn rccl_init_destroy_mesh_2() {
    match device_count() {
        Ok(n) if n >= 2 => {}
        Ok(n) => {
            eprintln!("[skip] need >= 2 HIP devices, have {n}");
            return;
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            return;
        }
    }
    let mesh = HipMesh::new(&[0, 1]).expect("HipMesh::new");
    assert_eq!(mesh.rank_count(), 2);
    drop(mesh);
}
