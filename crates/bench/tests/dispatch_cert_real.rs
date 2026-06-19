//! GPU-free CI ratchet: every row in the *real* committed dispatch tables must
//! be backed by a green cert. This is the cheap half of autotune-as-CI — it
//! needs no device (only reads the `.toml` + the cert `.json`), so it runs in
//! any CI, and it catches the most common dispatch staleness: a row pointing at
//! an impl whose cert is missing, red, or renamed. (The perf half — a certified
//! variant now winning — is `bench autotune --check`, which needs the GPU.)

use std::path::{Path, PathBuf};

use flambeau_bench::dispatch::cert_check;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("canonicalize repo root")
}

#[test]
fn committed_dispatch_tables_are_cert_backed() {
    let root = repo_root();
    let mut checked_any = false;
    for rel in ["dispatch/hip/gfx906.toml", "dispatch/cuda/sm_86.toml"] {
        let path = root.join(rel);
        if !path.exists() {
            continue;
        }
        checked_any = true;
        let report = cert_check(&root, &path).expect("cert_check ran");
        assert!(
            report.ok(),
            "{rel}: {} dispatch row(s) lack a green cert:\n{}",
            report.failures.len(),
            report.failures.iter().map(|(i, e)| format!("  {i}: {e}")).collect::<Vec<_>>().join("\n"),
        );
        eprintln!("{rel}: {} rows cert-backed", report.rows_checked);
    }
    assert!(checked_any, "no committed dispatch tables found under {}", root.display());
}
