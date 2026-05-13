//! Final-logit softcap wrapper.
//!
//! Gemma 4's `f_final_logit_softcapping = 30.0` (across all 5 audited
//! GGUFs) applies `y = tanh(x / cap) * cap` to the LM-head logits
//! before sampling. The wrapper here is a thin call into the
//! `apply_softcap_f32` kernel; it short-circuits when `cap == 0.0`
//! (softcap disabled) for forward-compatibility with non-gemma4
//! variants that may toggle it off.

#![cfg(feature = "hip")]

use anyhow::{Context, Result};
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;

/// Apply the gemma4 logit softcap in-place (or with `x == y`).
/// `cap = 0.0` is a no-op.
pub fn apply_logit_softcap<O: Ops>(
    ops: &O,
    x: DevicePtr,
    y: DevicePtr,
    n_logits: usize,
    cap: f32,
) -> Result<()> {
    if cap == 0.0 {
        return Ok(());
    }
    ops.apply_softcap_f32(x, y, n_logits, cap)
        .context("apply_logit_softcap")
}
