//! Pins which quantised-matmul call sites hand the few-column activation to
//! `qmatmul` and which pass none. A site without it falls back to the batched
//! MMVQ inside the few-column band (`MMQ_FEW_FALLBACK`), so a new or changed
//! site must be listed here on purpose, never fall back silently.
//!
//! Reads the sources: every `QuantWeight::qmatmul` call (third argument: the
//! few-column tensor) and every `act_q8_1_few:` field of a `QmatmulBuffers`
//! literal (or parameter).

use std::path::{Path, PathBuf};

const NULL_TENSOR: &str = "&Tensor::null()";
const NULL_PTR: &str = "flambeau_core::DevicePtr::NULL";

/// (file under `crates/`, few-column argument of each site in source order).
const SITES: &[(&str, &[&str])] = &[
    (
        "forward/src/core/composites/dense_ffn.rs",
        &["act_norm_few", "act_norm_few", "act_gated_few"],
    ),
    ("forward/src/core/composites/moe_ffn.rs", &[NULL_TENSOR, NULL_TENSOR]),
    ("forward/src/core/composites/mtp.rs", &["&act_few", NULL_TENSOR]),
    ("forward/src/core/composites/output_head.rs", &[NULL_TENSOR]),
    (
        "forward/src/core/composites/standard_attn.rs",
        &["act_norm_few", "act_norm_few", "act_norm_few", "act_norm_few", "act_attn_out_few"],
    ),
    (
        "forward/src/core/composites/standard_attn_mixed.rs",
        &[NULL_TENSOR, NULL_TENSOR, NULL_TENSOR, NULL_TENSOR, NULL_TENSOR],
    ),
    // QuantWeight::qmatmul: its parameter, then the field it fills.
    ("forward/src/ctx.rs", &["&Tensor<flambeau_model_ops::Q8_1>", "act_q8_1_few.ptr"]),
    (
        "model-ops/src/delta_net.rs",
        &[NULL_PTR, NULL_PTR, NULL_PTR, "x_few", "x_few", "x_few", "x_few", "gated_few"],
    ),
    ("model-ops/src/shared_expert.rs", &[NULL_PTR, NULL_PTR, NULL_PTR]),
    ("model-ops/src/ops/qmatmul.rs", &[NULL_PTR]),
];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// Top-level comma-separated arguments of the call whose `(` ends at `open`.
fn call_args(src: &str, open: usize) -> Vec<String> {
    let mut depth = 1;
    let mut args = Vec::new();
    let mut cur = String::new();
    for c in src[open..].chars() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if depth == 0 {
            break;
        }
        if c == ',' && depth == 1 {
            args.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() {
        args.push(cur.trim().to_string());
    }
    args
}

/// The few-column argument of every site in `src`, in source order.
fn few_args(src: &str) -> Vec<String> {
    let mut sites: Vec<(usize, String)> = Vec::new();
    for (at, _) in src.match_indices(".qmatmul(") {
        // `QuantWeight::qmatmul(act, mmq, few, out, shape, ops)`; the
        // three-argument `Ops::qmatmul(QmatmulBuffers, ..)` is counted by
        // its `act_q8_1_few` field below.
        let args = call_args(src, at + ".qmatmul(".len());
        if args.len() == 6 {
            sites.push((at, args[2].clone()));
        }
    }
    for (at, _) in src.match_indices("act_q8_1_few:") {
        let rest = &src[at + "act_q8_1_few:".len()..];
        let end = rest.find([',', '\n', '}']).unwrap_or(rest.len());
        sites.push((at, rest[..end].trim().to_string()));
    }
    sites.sort_by_key(|(at, _)| *at);
    sites.into_iter().map(|(_, arg)| arg).collect()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn few_column_sites_are_the_listed_ones() {
    let root = crates_dir();
    for (file, expected) in SITES {
        let src = std::fs::read_to_string(root.join(file)).unwrap();
        assert_eq!(few_args(&src), *expected, "{file}: few-column arguments changed");
    }
}

/// No quantised-matmul site exists outside the listed files.
#[test]
fn no_unlisted_site() {
    let root = crates_dir();
    let mut files = Vec::new();
    for dir in ["forward/src", "model-ops/src", "models", "server/src", "server-core/src"] {
        let dir = root.join(dir);
        if dir.is_dir() {
            rust_files(&dir, &mut files);
        }
    }
    let unlisted: Vec<_> = files
        .iter()
        .filter(|f| {
            let src = std::fs::read_to_string(f).unwrap();
            !few_args(&src).is_empty()
        })
        .map(|f| f.strip_prefix(&root).unwrap().to_string_lossy().into_owned())
        .filter(|f| !f.contains("/tests/") && !SITES.iter().any(|(listed, _)| listed == f))
        .collect();
    assert!(unlisted.is_empty(), "quantised-matmul sites in unlisted files: {unlisted:?}");
}
