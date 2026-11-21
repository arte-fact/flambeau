//! `KernelDescriptor` dispatch tables for CUDA archs — the Rust-typed counterpart of
//! `dispatch/cuda/<arch>.toml` — and the catalogs of certified kernels outside
//! them. `dispatch_qmatmul` returns the matching row.

use flambeau_core::{first_match, KernelDescriptor, QDtype, QMatMulCfg};

mod direct_call;
mod qmatmul;
pub use direct_call::{BENCH_REFERENCE_KERNELS_SM86, DIRECT_CALL_KERNELS_SM86};
pub use qmatmul::QMATMUL_SM86;

/// Pick the `QMatMul` `KernelDescriptor` for `cfg` from the sm_86 table.
pub fn dispatch_qmatmul(cfg: &QMatMulCfg) -> Option<&'static KernelDescriptor> {
    first_match(QMATMUL_SM86, cfg.dtype_weight, cfg.dtype_activation, cfg.m)
}

/// `[[attention_decode]]` rows of `dispatch/cuda/sm_86.toml`, keyed on the KV
/// dtype (`dtype_weight`) with F16 Q / output. Each row names the decode path
/// the composite runs for that KV layout: F16 single-pass up to the split-K
/// threshold and split-K above it (direct calls); Q8_0 the 128-thread
/// vec_dpos kernel on global layers (SWA layers keep the split-K path).
pub const ATTENTION_DECODE_SM86: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "AttentionDecode",
        impl_id: "attention_decode_f16",
        backend: "cuda",
        arch: "sm_86",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, 4),
        cert_rel_path: "certs/cuda/sm_86/attention_decode_f16.json",
    },
    KernelDescriptor {
        op_name: "AttentionDecode",
        impl_id: "attention_decode_q8_kv_vec_dpos_sm_86",
        backend: "cuda",
        arch: "sm_86",
        dtype_weight: QDtype::Q8_0,
        dtype_activation: QDtype::F16,
        m_range: (1, 4),
        cert_rel_path: "certs/cuda/sm_86/attention_decode_q8_kv_vec_dpos_sm_86.json",
    },
];

/// `[[attention_prefill]]` rows of `dispatch/cuda/sm_86.toml` (F16 KV and
/// Q / output). From 16 query rows the f16 tensor-core flash attention
/// (head_dim 128 / 256 / 512; other head_dims stay on flash_tile); below,
/// the LDS-tiled flash_tile path (4+ rows) or the scalar kernel.
pub const ATTENTION_PREFILL_SM86: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "AttentionPrefill",
        impl_id: "attention_prefill_flash_tile_f16",
        backend: "cuda",
        arch: "sm_86",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, 15),
        cert_rel_path: "certs/cuda/sm_86/attention_prefill_flash_tile_f16.json",
    },
    KernelDescriptor {
        op_name: "AttentionPrefill",
        impl_id: "attention_prefill_mma",
        backend: "cuda",
        arch: "sm_86",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (16, usize::MAX),
        cert_rel_path: "certs/cuda/sm_86/attention_prefill_mma.json",
    },
];

/// Pick the prefill-attention row for a `kv_dtype` KV cache from the sm_86 table.
pub fn dispatch_attention_prefill(kv_dtype: QDtype, m: usize) -> Option<&'static KernelDescriptor> {
    first_match(ATTENTION_PREFILL_SM86, kv_dtype, QDtype::F16, m)
}

/// Pick the decode-attention row for a `kv_dtype` KV cache from the sm_86 table.
pub fn dispatch_attention_decode(kv_dtype: QDtype, m: usize) -> Option<&'static KernelDescriptor> {
    first_match(ATTENTION_DECODE_SM86, kv_dtype, QDtype::F16, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
    }

    /// `(dtype, dtype_q, m predicate, impl)` of every `[[section]]` row of
    /// `dispatch/cuda/sm_86.toml`.
    fn toml_rows(section: &str) -> std::collections::BTreeSet<(String, String, String, String)> {
        let src = std::fs::read_to_string(repo_root().join("dispatch/cuda/sm_86.toml")).expect("read sm_86.toml");
        let header = format!("[[{section}]]");
        let field = |block: &str, key: &str| -> String {
            block
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once('=')?;
                    (k.trim() == key).then(|| v.split('#').next().unwrap_or("").trim().to_owned())
                })
                .unwrap_or_default()
        };
        let unquote = |v: String| v.trim_matches('"').to_owned();
        // Blocks start at a line that is a table header; a `[[x]]` inside a
        // comment is not one.
        let mut blocks: Vec<(String, String)> = Vec::new();
        for line in src.lines() {
            let t = line.trim();
            if t.starts_with("[[") {
                blocks.push((t.to_owned(), String::new()));
            } else if let Some((_, body)) = blocks.last_mut() {
                body.push_str(line);
                body.push('\n');
            }
        }
        blocks
            .iter()
            .filter(|(h, _)| *h == header)
            .map(|(_, block)| {
                let shape = field(block, "shape");
                let m = shape
                    .split_once("m =")
                    .and_then(|(_, rest)| rest.split('"').nth(1))
                    .unwrap_or("")
                    .to_owned();
                (unquote(field(block, "dtype")), unquote(field(block, "dtype_q")), m, unquote(field(block, "impl")))
            })
            .collect()
    }

    /// The same rows from a Rust dispatch table, `m_range` written the TOML way.
    fn rust_rows(table: &[KernelDescriptor]) -> std::collections::BTreeSet<(String, String, String, String)> {
        table
            .iter()
            .map(|d| {
                let m = match d.m_range {
                    (1, usize::MAX) => "any".to_owned(),
                    (lo, usize::MAX) => format!(">={lo}"),
                    (lo, hi) if lo == hi => lo.to_string(),
                    (lo, hi) => format!("{lo}..{hi}"),
                };
                (d.dtype_weight.name().to_owned(), d.dtype_activation.name().to_owned(), m, d.impl_id.to_owned())
            })
            .collect()
    }

    /// The runtime dispatches from the Rust tables, cert-check reads the TOML:
    /// they must name the same rows.
    #[test]
    fn rust_tables_match_the_toml() {
        for (section, table) in [
            ("qmatmul", QMATMUL_SM86),
            ("attention_decode", ATTENTION_DECODE_SM86),
            ("attention_prefill", ATTENTION_PREFILL_SM86),
        ] {
            let toml = toml_rows(section);
            let rust = rust_rows(table);
            let only_toml: Vec<_> = toml.difference(&rust).collect();
            let only_rust: Vec<_> = rust.difference(&toml).collect();
            assert!(
                only_toml.is_empty() && only_rust.is_empty(),
                "[[{section}]] disagrees — TOML only: {only_toml:?}; Rust only: {only_rust:?}"
            );
        }
    }

    fn registered() -> impl Iterator<Item = (&'static str, &'static str)> {
        QMATMUL_SM86
            .iter()
            .chain(ATTENTION_DECODE_SM86)
            .chain(ATTENTION_PREFILL_SM86)
            .map(|d| (d.impl_id, d.cert_rel_path))
            .chain(DIRECT_CALL_KERNELS_SM86.iter().map(|d| (d.impl_id, d.cert_rel_path)))
            .chain(BENCH_REFERENCE_KERNELS_SM86.iter().map(|d| (d.impl_id, d.cert_rel_path)))
    }

    #[test]
    fn every_registered_cert_path_exists_on_disk() {
        let root = repo_root();
        let missing: Vec<_> =
            registered().filter(|(_, path)| !root.join(path).exists()).map(|(id, _)| id).collect();
        assert!(missing.is_empty(), "registered sm_86 impls without a cert on disk: {missing:?}");
    }

    #[test]
    fn every_impl_id_is_registered_once() {
        let mut seen = std::collections::HashSet::new();
        let dup: Vec<_> = registered().map(|(id, _)| id).filter(|id| !seen.insert(*id)).collect();
        assert!(dup.is_empty(), "impl_ids registered twice: {dup:?}");
    }

    #[test]
    fn uncertified_certs_are_not_registered() {
        let both: Vec<_> = registered().map(|(id, _)| id).filter(|id| UNCERTIFIED_CERTS.contains(id)).collect();
        assert!(both.is_empty(), "listed as uncertified but registered: {both:?}");
    }

    #[test]
    fn every_cert_path_names_its_impl() {
        let bad: Vec<_> = registered()
            .filter(|(id, path)| *path != format!("certs/cuda/sm_86/{id}.json"))
            .map(|(id, _)| id)
            .collect();
        assert!(bad.is_empty(), "cert_rel_path does not match impl_id: {bad:?}");
    }

    /// Known rule-2 holes: certs on disk that no sweep writes, so they cannot
    /// be registered as certified. Shrinks only.
    const UNCERTIFIED_CERTS: &[&str] = &[];

    /// Every cert JSON under `certs/cuda/sm_86/` is registered in
    /// `QMATMUL_SM86`, `DIRECT_CALL_KERNELS_SM86` or
    /// `BENCH_REFERENCE_KERNELS_SM86`, or listed in `UNCERTIFIED_CERTS`.
    #[test]
    fn every_cert_on_disk_is_registered() {
        let dir = repo_root().join("certs/cuda/sm_86");
        let known: std::collections::HashSet<_> =
            registered().map(|(id, _)| id).chain(UNCERTIFIED_CERTS.iter().copied()).collect();
        let mut orphans: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .filter_map(|e| e.unwrap().file_name().to_string_lossy().strip_suffix(".json").map(str::to_string))
            .filter(|stem| !known.contains(stem.as_str()))
            .collect();
        orphans.sort();
        assert!(
            orphans.is_empty(),
            "cert files under certs/cuda/sm_86/ without a registration: {orphans:?}. Either delete \
             the cert, or add it to QMATMUL_SM86, DIRECT_CALL_KERNELS_SM86 or \
             BENCH_REFERENCE_KERNELS_SM86."
        );
    }
}
