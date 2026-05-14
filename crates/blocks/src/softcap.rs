//! Logit / attention softcap helpers.
//!
//! Gemma 4 specifies `y = tanh(x / cap) * cap` over the final
//! LM-head logits (`f_final_logit_softcapping`); the same `tanh`-
//! scaled cap also shows up in some attention-logit configurations
//! (Gemma 2's `f_attn_logit_softcapping`, Qwen 3.5's optional cap).
//! Centralising the wrapper here lets every model that needs it
//! reuse the same shape without copying the per-call dispatch + the
//! `cap == 0` short-circuit.

#![cfg(feature = "hip")]

use anyhow::{Context, Result};
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;

/// Apply `y[i] = tanh(x[i] / cap) * cap` over `n_logits` F32 entries.
/// `cap == 0.0` is a no-op (softcap disabled). Source and dest may
/// alias (`x == y` for in-place softcap).
pub fn apply_logit_softcap_f32<O: Ops>(
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
        .context("apply_logit_softcap_f32")
}
