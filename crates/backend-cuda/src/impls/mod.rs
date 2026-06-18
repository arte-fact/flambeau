//! `KernelDescriptor` dispatch tables for CUDA archs — the Rust-typed counterpart of
//! `dispatch/cuda/<arch>.toml`. `dispatch_qmatmul` returns the matching row.

use flambeau_core::{first_match, KernelDescriptor, QMatMulCfg};

mod qmatmul;
pub use qmatmul::QMATMUL_SM86;

/// Pick the `QMatMul` `KernelDescriptor` for `cfg` from the sm_86 table.
pub fn dispatch_qmatmul(cfg: &QMatMulCfg) -> Option<&'static KernelDescriptor> {
    first_match(QMATMUL_SM86, cfg.dtype_weight, cfg.dtype_activation, cfg.m)
}
