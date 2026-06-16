//! flambeau-kernels-shared — algorithmic-core `.cuh` headers (under `include/`,
//! `#include`d by `kernels-hip` / `kernels-cuda` via `build.rs` include paths)
//! plus the [`codegen`] helper that turns `instantiations.toml` into per-backend
//! wrapper `.cu` files (used by those crates' build scripts).
//!
//! ZERO backend-specific intrinsics in this crate's `include/*.cuh`.
//! HIP-only or CUDA-only intrinsics belong in their respective backend crates.

pub mod codegen;
