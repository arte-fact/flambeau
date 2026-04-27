//! Diagnostic — bracket every 2-device sub-cluster and print whether
//! `peer_access_full` is true. Useful when the 4-card matrix shows full
//! access but a 2-device sub-cluster mysteriously fails (which is what
//! AUTO-4d hit — `[0,2]` and `[1,3]` both report off-diagonal 0 on this
//! rig despite `[0,1,2,3]` showing full access).

use flambeau_backend_hip::{device_count, HipCluster};

fn maybe_skip() -> Option<i32> {
    match device_count() {
        Ok(n) if n > 0 => Some(n),
        _ => {
            eprintln!("[skip] no HIP devices");
            None
        }
    }
}

#[test]
fn pair_matrix_each_two_device_subset() {
    let Some(n) = maybe_skip() else { return };
    let n = n as i32;
    if n < 2 {
        eprintln!("[skip] need >= 2 devices (have {n})");
        return;
    }
    println!("Bracketing every ordered 2-device pair on a {n}-device rig:");
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            let cluster = match HipCluster::new(&[a, b]) {
                Ok(c) => c,
                Err(e) => {
                    println!("  [{a},{b}] HipCluster::new FAILED: {e}");
                    continue;
                }
            };
            let mat = cluster.peer_access_matrix();
            let off_diag = mat[0][1] && mat[1][0];
            println!(
                "  [{a},{b}] peer_access_full={} off_diag(0->1)={} (1->0)={}",
                cluster.peer_access_full(),
                mat[0][1],
                mat[1][0],
            );
            let _ = off_diag; // explicit print captures both directions
        }
    }
}
