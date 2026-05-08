//! flambeau-kernels-hip — HIP kernel `.cu` sources compiled to `.hsaco` at
//! build time.
//!
//! `build.rs` enumerates every `src/kernels/*.cu`, runs `hipcc` with
//! `--offload-arch=gfx906` (default) or `HIP_OFFLOAD_ARCH=…`, and emits an
//! `hsaco.rs` with a `pub const <STEM>_HSACO: &[u8]` per kernel plus a
//! `CATALOGUE` slice of `(name, hsaco_bytes)`.
//!
//! Rust code in `backend-hip` loads these slices via `hipModuleLoadData` at
//! the first launch. No kernel is launched without a matching cert (see
//! `certs/hip/gfx906/`); the bench sweep produces them.
//!
//! `HIP_SKIP_BUILD=1` skips hipcc; the generated `hsaco.rs` is still valid
//! but `CATALOGUE` is empty, so `hipModuleLoadData` failures land cleanly.

#![forbid(unsafe_op_in_unsafe_fn)]

include!(concat!(env!("OUT_DIR"), "/hsaco.rs"));

/// Look up a kernel module's compiled ELF bytes by file-stem
/// (e.g. `"mmvq_q8_0"`, `"quantize_q8_1"`). Returns `None` if this build
/// didn't compile that kernel — either because the `.cu` file doesn't exist
/// or `HIP_SKIP_BUILD=1` skipped compilation.
pub fn hsaco(name: &str) -> Option<&'static [u8]> {
    CATALOGUE.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

/// Every kernel compiled into this crate, for enumeration / logging.
pub fn catalogue() -> &'static [(&'static str, &'static [u8])] {
    CATALOGUE
}
