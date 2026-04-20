//! flambeau-kernels-shared — crate is Rust-side metadata only. Algorithmic-core `.cuh`
//! headers live under `include/` and are `#include`d by `kernels-hip` and `kernels-cuda`
//! through `build.rs` include paths.
//!
//! ZERO backend-specific intrinsics in this crate's `include/*.cuh`.
//! HIP-only or CUDA-only intrinsics belong in their respective backend crates.
