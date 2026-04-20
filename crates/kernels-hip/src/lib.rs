//! flambeau-kernels-hip — `.cu` sources + `arch_primitives/<gfxNNN>.cuh` compiled
//! through `build.rs` into `.hsaco` archives per target arch.
//!
//! V1.2+ populates this with candle ports (P29 multi-row MMVQ, llamacpp-turbo 4-warp MMQ,
//! P30 fused gate+up, D1 rmsnorm+Q8_1). See `doc/candle-prior-art.md`.
