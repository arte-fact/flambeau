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

    fn registered() -> impl Iterator<Item = (&'static str, &'static str)> {
        QMATMUL_SM86
            .iter()
            .chain(ATTENTION_DECODE_SM86)
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
    const UNCERTIFIED_CERTS: &[&str] = &[
        // why: no sweep for the CUDA tensor-core prefill attention yet (step 5).
        "attention_prefill_mma",
    ];

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
