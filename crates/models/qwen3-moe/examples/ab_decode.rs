//! A/B decode-perf harness on disjoint GPU sets.
//!
//! Spawns two `decode_profile` children sequentially (A then B), each on its
//! own 2-GPU cluster (`FLAMBEAU_A_DEVICES` vs `FLAMBEAU_B_DEVICES`). Each
//! child receives `FLAMBEAU_VARIANT` set to the per-side variant label —
//! `decode_profile` reads that and switches its forward path.
//!
//! Concurrent execution was prototyped and fails: HIP's HSA runtime init
//! races when two processes touch disjoint GPU sets simultaneously, and
//! two HipCluster instances in one process have shared rocBLAS / stream
//! state that crashes under concurrent forwards. Sequential is honest and
//! gives equivalent information (warm-up is per-side so thermal/cache state
//! starts fresh for each).
//!
//! Reports per-side tok/s, the A/B ratio, and parity (last_id must match
//! since both sides use greedy decode from the same seed — any mismatch
//! means the variant broke correctness).
//!
//! Usage:
//!   FLAMBEAU_QWEN3_GGUF=... \
//!     FLAMBEAU_A_DEVICES=0,1 FLAMBEAU_A_VARIANT=baseline \
//!     FLAMBEAU_B_DEVICES=2,3 FLAMBEAU_B_VARIANT=fused_gdn_norm \
//!     FLAMBEAU_AB_STEPS=64 \
//!     cargo run --release -p flambeau-qwen3-moe --features hip --example ab_decode

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Debug, Deserialize)]
struct ChildResult {
    devices: Vec<i32>,
    #[expect(dead_code, reason = "part of the child-process JSON contract; reserved for future variance reporting")]
    steps: u32,
    wall_secs: f64,
    tok_per_sec: f64,
    last_id: u32,
}

fn run_side(label: &str, devices: &str, variant: &str, steps: &str) -> Result<ChildResult> {
    let exe = std::env::current_exe()?;
    let bin_dir = exe
        .parent()
        .ok_or_else(|| anyhow!("no parent dir for current_exe"))?;
    let child_bin = bin_dir.join("decode_profile");
    if !child_bin.exists() {
        return Err(anyhow!(
            "decode_profile binary not found at {}",
            child_bin.display()
        ));
    }

    let json_out = PathBuf::from(format!("/tmp/flambeau_ab_{label}.json"));
    let _ = std::fs::remove_file(&json_out);
    let gguf = std::env::var("FLAMBEAU_QWEN3_GGUF")
        .map_err(|_unset| anyhow!("FLAMBEAU_QWEN3_GGUF unset"))?;

    eprintln!("[{label}] spawn decode_profile devices={devices} variant={variant} steps={steps}");
    let output = Command::new(&child_bin)
        .env("FLAMBEAU_QWEN3_GGUF", &gguf)
        .env("FLAMBEAU_DEVICES", devices)
        .env("FLAMBEAU_MESH_RANKS", "")
        .env("FLAMBEAU_PROFILE_STEPS", steps)
        .env("FLAMBEAU_VARIANT", variant)
        .env("FLAMBEAU_RESULT_JSON", &json_out)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "side {label} (devices={devices}, variant={variant}) failed:\n{stderr}"
        ));
    }

    let json = std::fs::read_to_string(&json_out)
        .map_err(|e| anyhow!("reading {}: {e}", json_out.display()))?;
    let r: ChildResult = serde_json::from_str(&json)?;
    eprintln!(
        "[{label}] devices={:?} variant={variant} → {:.2} tok/s ({:.3} s, last_id={})",
        r.devices, r.tok_per_sec, r.wall_secs, r.last_id,
    );
    Ok(r)
}

fn main() -> Result<()> {
    let a_dev = std::env::var("FLAMBEAU_A_DEVICES").unwrap_or_else(|_| "0,1".into());
    let b_dev = std::env::var("FLAMBEAU_B_DEVICES").unwrap_or_else(|_| "2,3".into());
    let a_var = std::env::var("FLAMBEAU_A_VARIANT").unwrap_or_else(|_| "baseline".into());
    let b_var = std::env::var("FLAMBEAU_B_VARIANT").unwrap_or_else(|_| "baseline".into());
    let steps = std::env::var("FLAMBEAU_AB_STEPS").unwrap_or_else(|_| "64".into());

    eprintln!(
        "ab_decode: A=devices[{a_dev}]/variant[{a_var}]  vs  B=devices[{b_dev}]/variant[{b_var}]  steps={steps}"
    );

    let a = run_side("A", &a_dev, &a_var, &steps)?;
    let b = run_side("B", &b_dev, &b_var, &steps)?;

    let parity_ok = a.last_id == b.last_id;
    let ratio = b.tok_per_sec / a.tok_per_sec;

    eprintln!("— result —");
    eprintln!("A ({:<20}) devs={:?}: {:>6.2} tok/s  last_id={}",
        a_var, a.devices, a.tok_per_sec, a.last_id);
    eprintln!("B ({:<20}) devs={:?}: {:>6.2} tok/s  last_id={}",
        b_var, b.devices, b.tok_per_sec, b.last_id);
    eprintln!("B/A: {:.3}×  ({:+.1}%)", ratio, (ratio - 1.0) * 100.0);
    if !parity_ok {
        eprintln!(
            "!! parity mismatch: A.last_id={} vs B.last_id={} — variant broke correctness",
            a.last_id, b.last_id
        );
        std::process::exit(1);
    }
    Ok(())
}
