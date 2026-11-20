//! flambeau-kernels-cuda — CUDA kernel `.cu` sources compiled to `.cubin` at
//! build time.
//!
//! `build.rs` enumerates every `src/kernels/*.cu`, runs `nvcc` with
//! `-arch=sm_86` (default) or `CUDA_ARCH=…`, and emits a `cubin.rs` with a
//! `pub const <STEM>_CUBIN: &[u8]` per kernel plus a `CATALOGUE` slice of
//! `(name, cubin_bytes)`.
//!
//! Rust code in `backend-cuda` loads these slices via `cuModuleLoadData` at
//! the first launch. No kernel is launched without a matching cert (see
//! `certs/cuda/sm_86/`); the bench sweep produces them.
//!
//! `CUDA_SKIP_BUILD=1` skips nvcc and an empty `src/kernels/` needs no
//! toolchain; both yield an empty `CATALOGUE`, so `cubin()` returns `None`.

#![forbid(unsafe_op_in_unsafe_fn)]

include!(concat!(env!("OUT_DIR"), "/cubin.rs"));

/// Look up a kernel module's compiled cubin bytes by file-stem
/// (e.g. `"add_f32"`, `"mmvq_q8_0"`). Returns `None` if this build didn't
/// compile that kernel — either because the `.cu` file doesn't exist or
/// `CUDA_SKIP_BUILD=1` skipped compilation.
pub fn cubin(name: &str) -> Option<&'static [u8]> {
    CATALOGUE.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

/// Every kernel compiled into this crate, for enumeration / logging.
pub fn catalogue() -> &'static [(&'static str, &'static [u8])] {
    CATALOGUE
}

/// Parameter count of the `extern "C" __global__` entry point `entry`, read
/// from its source at build time. `None` for an entry this build did not
/// compile or whose signature the source scan could not read.
pub fn param_count(entry: &str) -> Option<usize> {
    PARAM_COUNTS
        .binary_search_by(|(n, _)| (*n).cmp(entry))
        .ok()
        .map(|i| usize::from(PARAM_COUNTS[i].1))
}
