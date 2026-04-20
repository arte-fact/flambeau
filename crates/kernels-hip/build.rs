//! build.rs — compiles HIP `.cu` sources under `src/` to `.hsaco` via `hipcc`.
//!
//! V1.0 stub: intentionally does nothing until V1.2 lands the first kernel. Real
//! implementation mirrors candle's `candle-hip-kernels/build.rs`:
//!   - enumerate `src/*.cu`
//!   - for each (arch in [gfx906, gfx908, gfx90a, gfx942, gfx1031, gfx1100]):
//!     `hipcc --genco --offload-arch=<arch> <file>.cu -o $OUT_DIR/<file>.<arch>.hsaco`
//!   - emit `cargo:rerun-if-changed=` for every .cu / .cuh under src/ and the
//!     sibling `kernels-shared/include/` dir.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
