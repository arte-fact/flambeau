//! Shared bench sweep helpers.
//!
//! Every `sweep_*.rs` used to open-code the same five boilerplate pieces:
//! hostname lookup, rig-string assembly, an `alloc_and_upload<T>` upload
//! helper, a splitmix-style `seeded_f32` generator, and a max-relative-
//! error comparator. Extracted here so each sweep becomes ~40 LOC lighter
//! and bugfixes in these primitives propagate everywhere automatically.
//!
//! The two variable parts are exposed as parameters:
//! - `seeded_f32_range(seed, n, lo, hi)` lets callers pick the input
//!   dynamic range (swiglu wants `[-2, 2]`, attention wants `[-0.5, 0.5]`,
//!   etc.) instead of hard-coding a range.
//! - `max_rel_err_with_floor(got, ref, abs_floor)` takes the tolerance
//!   floor directly — callers that scale with `sqrt(k)` or `sqrt(head_dim)`
//!   compute their floor at the call site.

#![cfg(feature = "hip")]

use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

// splitmix64 constants — matches the sequence every sweep previously
// open-coded, so cert diffs stay bit-identical across the migration.
const SPLITMIX_MUL: u64 = 6_364_136_223_846_793_005;
const SPLITMIX_INC: u64 = 1_442_695_040_888_963_407;

/// POSIX hostname lookup. Returns `None` if `gethostname` failed or the
/// result is not UTF-8. Checks `HOSTNAME` env var first as a fast path.
#[must_use]
pub fn hostname() -> Option<String> {
    std::env::var("HOSTNAME").ok().or_else(|| {
        let mut buf = vec![0u8; 256];
        // SAFETY: `buf` owns 256 bytes; `libc_gethostname` writes at most
        // that many and NUL-terminates on success.
        let rv = unsafe { libc_gethostname(buf.as_mut_ptr().cast(), buf.len()) };
        if rv != 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        buf.truncate(end);
        String::from_utf8(buf).ok()
    })
}

/// The rig identifier that every sweep writes into its cert: the hostname
/// suffixed with `-gfx906`, or `unknown-gfx906` when hostname lookup fails.
#[must_use]
pub fn rig() -> String {
    format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()))
}

/// Allocate `data.len() * size_of::<T>()` bytes on `dev`, copy `data` in,
/// sync the default stream, return the device pointer.
///
/// # Panics
/// Panics on allocation or copy failure. Used only by the sweep harness,
/// which runs in single-threaded test context — a panic here is a broken
/// fixture, not a runtime bug to bubble up to callers.
#[must_use]
pub fn alloc_and_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).expect("alloc_and_upload: alloc");
    // SAFETY: `d` owns `bytes`; `data` is `bytes` of valid host memory.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .expect("alloc_and_upload: memcpy");
    }
    dev.default_stream()
        .synchronize()
        .expect("alloc_and_upload: sync");
    d
}

/// Deterministic `f32` generator in `[lo, hi]` seeded by `seed`. Uses the
/// splitmix64 constants the individual sweeps already agreed on, so
/// identical `(seed, n, lo, hi)` reproduces bit-identical results — cert
/// diffs stay clean when a sweep migrates from local `seeded_f32` to this.
#[must_use]
pub fn seeded_f32_range(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let span = hi - lo;
    let mut s = seed.wrapping_mul(SPLITMIX_MUL).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(SPLITMIX_MUL).wrapping_add(SPLITMIX_INC);
        let u = (s >> 32) as u32;
        out.push((u as f32 / u32::MAX as f32) * span + lo);
    }
    out
}

/// Relative-error reducer with an explicit absolute floor. For each pair
/// `(g, r)`, computes `|g - r| / max(|r|, abs_floor)`; returns the max.
/// The floor prevents small-denominator blow-up near zero.
#[must_use]
pub fn max_rel_err_with_floor(got: &[f32], reference: &[f32], abs_floor: f32) -> f32 {
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(abs_floor))
        .fold(0.0f32, f32::max)
}

extern "C" {
    #[link_name = "gethostname"]
    fn libc_gethostname(name: *mut std::os::raw::c_char, len: usize) -> i32;
}
// Rust guideline compliant 2026-02-21
