//! Unit tests for the cert JSON writer + `cert-check` gate.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use flambeau_bench::cert::{Cert, ShapeResult, SCHEMA_VERSION};
use flambeau_bench::dispatch::cert_check;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tempdir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let p = std::env::temp_dir().join(format!(
        "flambeau-cert-check-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write_dispatch(root: &Path, impl_id: &str, cert_rel: &str) -> PathBuf {
    let dispatch_dir = root.join("dispatch/hip");
    std::fs::create_dir_all(&dispatch_dir).unwrap();
    let toml = format!(
        r#"
[[qmatmul]]
dtype = "Q4_K"
dtype_q = "Q8_1"
shape = {{ m = "any", k = "any", n = "any" }}
impl = "{impl_id}"
cert = "{cert_rel}"
"#
    );
    let path = dispatch_dir.join("gfx906.toml");
    std::fs::write(&path, toml).unwrap();
    path
}

fn write_green_cert(root: &Path, impl_id: &str, rel_path: &str) {
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: impl_id.to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul_mmvq".to_string(),
        dtype_weight: "Q4_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "test".to_string(),
        results: vec![ShapeResult {
            m: 1,
            k: 128,
            n: 128,
            seed: 0,
            max_rel_err: 1e-5,
            tolerance: 1e-2,
            pass: true,
        }],
        pass: true,
        emitted_at: "1970-01-01T00:00:00Z".to_string(),
        rig: "test".to_string(),
        pmc: None,
    };
    let full = root.join(rel_path);
    std::fs::create_dir_all(full.parent().unwrap()).unwrap();
    std::fs::write(&full, serde_json::to_string_pretty(&cert).unwrap()).unwrap();
}

#[test]
fn cert_check_green_path() {
    let root = tempdir();
    let dispatch = write_dispatch(
        &root,
        "qmatmul_q4_K_mmvq_single_row_gfx906",
        "certs/hip/gfx906/qmatmul_q4_K_mmvq_single_row_gfx906.json",
    );
    write_green_cert(
        &root,
        "qmatmul_q4_K_mmvq_single_row_gfx906",
        "certs/hip/gfx906/qmatmul_q4_K_mmvq_single_row_gfx906.json",
    );
    let report = cert_check(&root, &dispatch).unwrap();
    assert!(report.ok(), "expected green, got {:?}", report.failures);
    assert_eq!(report.rows_checked, 1);
}

#[test]
fn cert_check_missing_cert_fails() {
    let root = tempdir();
    let dispatch = write_dispatch(
        &root,
        "qmatmul_q4_K_mmvq_single_row_gfx906",
        "certs/hip/gfx906/qmatmul_q4_K_mmvq_single_row_gfx906.json",
    );
    // Skip the cert-write step.
    let report = cert_check(&root, &dispatch).unwrap();
    assert!(!report.ok());
    assert_eq!(report.failures.len(), 1);
    assert!(format!("{}", report.failures[0].1).contains("missing"));
}

#[test]
fn cert_check_impl_id_mismatch_fails() {
    let root = tempdir();
    let dispatch = write_dispatch(
        &root,
        "qmatmul_q4_K_mmvq_single_row_gfx906",
        "certs/hip/gfx906/qmatmul_q4_K_mmvq_single_row_gfx906.json",
    );
    write_green_cert(
        &root,
        "some_other_impl", // deliberately wrong
        "certs/hip/gfx906/qmatmul_q4_K_mmvq_single_row_gfx906.json",
    );
    let report = cert_check(&root, &dispatch).unwrap();
    assert!(!report.ok());
    assert!(format!("{}", report.failures[0].1).contains("impl_id"));
}

#[test]
fn cert_check_pass_false_fails() {
    let root = tempdir();
    let dispatch = write_dispatch(
        &root,
        "q",
        "certs/hip/gfx906/q.json",
    );
    // Write a cert with pass=false.
    let mut cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "q".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul_mmvq".to_string(),
        dtype_weight: "Q4_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "t".to_string(),
        results: vec![],
        pass: false,
        emitted_at: "t".to_string(),
        rig: "t".to_string(),
        pmc: None,
    };
    cert.pass = false;
    let cert_path = root.join("certs/hip/gfx906/q.json");
    std::fs::create_dir_all(cert_path.parent().unwrap()).unwrap();
    std::fs::write(&cert_path, serde_json::to_string_pretty(&cert).unwrap()).unwrap();
    let report = cert_check(&root, &dispatch).unwrap();
    assert!(!report.ok());
    assert!(format!("{}", report.failures[0].1).contains("pass=false"));
}
