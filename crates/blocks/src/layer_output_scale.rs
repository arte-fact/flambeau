//! Per-layer output scalar helper.
//!
//! Gemma 4 ships a per-layer F32 scalar (`layer_output_scale`) read
//! from the GGUF; when present and `!= 1.0`, the layer's final
//! residual stream is multiplied by it before being handed off to the
//! next layer. Other models may grow analogous knobs; centralising
//! the `Option<f32>`-aware dispatch + the `== 1.0` short-circuit
//! keeps the call sites a single line.

#![cfg(feature = "hip")]

use anyhow::{Context, Result};
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;

/// Apply `x = x * scale` over `hidden` F16 entries when `scale`
/// is `Some(v)` and `v != 1.0`. No-op otherwise. Source and dest may
/// alias (the canonical use is in-place on the layer's hidden buffer).
pub fn apply_layer_output_scale_f16<O: Ops>(
    ops: &O,
    x: DevicePtr,
    hidden: usize,
    scale: Option<f32>,
) -> Result<()> {
    let Some(v) = scale else { return Ok(()) };
    if v == 1.0 {
        return Ok(());
    }
    ops.scale_f16(x, x, hidden, v)
        .context("apply_layer_output_scale_f16")
}
