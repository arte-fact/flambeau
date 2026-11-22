//! Pins which quantised-matmul call sites hand the few-column activation to
//! `qmatmul` and which pass none. A site without it falls back to the batched
//! MMVQ inside the few-column band (`MMQ_FEW_FALLBACK`), so a new or changed
//! site must be listed here on purpose, never fall back silently.
//!
//! Reads the sources: every `QuantWeight::qmatmul` call (third argument: the
//! few-column activation) and every `act_q8_1_few:` field of a
//! `QmatmulBuffers` literal (or parameter). A second table pins where each
//! wired activation's scale record comes from: a site shared by several
//! projections takes `ActLayouts::few_scale` (Ds4 when its readers need both
//! records; the D4 readers then fall back), never a hand-picked record.

use std::path::{Path, PathBuf};

const NONE: &str = "flambeau_ops::FewAct::NONE";

/// (file under `crates/`, few-column argument of each site in source order).
const SITES: &[(&str, &[&str])] = &[
    (
        "forward/src/core/composites/dense_ffn.rs",
        &["act_norm_few", "act_norm_few", "act_gated_few"],
    ),
    ("forward/src/core/composites/moe_ffn.rs", &[NONE, NONE]),
    ("forward/src/core/composites/mtp.rs", &["few", NONE]),
    ("forward/src/core/composites/output_head.rs", &[NONE]),
    (
        "forward/src/core/composites/standard_attn.rs",
        &["act_norm_few", "act_norm_few", "act_norm_few", "act_norm_few", "act_attn_out_few"],
    ),
    (
        "forward/src/core/composites/standard_attn_mixed.rs",
        &[NONE, NONE, NONE, NONE, NONE],
    ),
    // QuantWeight::qmatmul: its parameter (the field is filled by shorthand).
    ("forward/src/ctx.rs", &["flambeau_ops::FewAct"]),
    (
        "model-ops/src/delta_net.rs",
        &[NONE, NONE, NONE, "x_few", "x_few", "x_few", "x_few", "gated_few"],
    ),
    ("model-ops/src/shared_expert.rs", &[NONE, NONE, NONE]),
    ("model-ops/src/ops/qmatmul.rs", &[NONE]),
];

/// (file under `crates/`, the scale argument of each `FewAct::new` in source
/// order). Shared sites take their `ActLayouts`' record; single-reader sites
/// the record of their one layout.
const SCALE_SOURCES: &[(&str, &[&str])] = &[
    ("model-ops/src/ops/quantize.rs", &["scale"]),
    ("model-ops/src/delta_net.rs", &["x_layouts.few_scale()", "Some(scale)"]),
];

/// Files that produce a few-column activation through
/// `quantize_f16_for_layouts`, with the layouts argument of each call.
const LAYOUT_SOURCES: &[(&str, &[&str])] = &[
    ("forward/src/core/composites/dense_ffn.rs", &["norm_layouts", "down_layout.into()"]),
    ("forward/src/core/composites/mtp.rs", &["layout.into()"]),
    ("forward/src/core/composites/standard_attn.rs", &["layouts", "out_layout.into()"]),
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

/// The `index`-th top-level argument of every call of `name(` in `src`.
fn nth_args(src: &str, name: &str, index: usize) -> Vec<String> {
    src.match_indices(name)
        .map(|(at, _)| call_args(src, at + name.len()))
        .filter(|args| args.len() > index)
        .map(|args| args[index].clone())
        .collect()
}

#[test]
fn few_scale_comes_from_the_site_layouts() {
    let root = crates_dir();
    for (file, expected) in SCALE_SOURCES {
        let src = std::fs::read_to_string(root.join(file)).unwrap();
        assert_eq!(nth_args(&src, "FewAct::new(", 1), *expected, "{file}: few-column scale source changed");
    }
    for (file, expected) in LAYOUT_SOURCES {
        let src = std::fs::read_to_string(root.join(file)).unwrap();
        assert_eq!(
            nth_args(&src, "quantize_f16_for_layouts(", 1),
            *expected,
            "{file}: few-column layouts source changed"
        );
    }
}
