//! `SharedExpert` — the per-token-sigmoid-gated dense FFN that
//! Qwen3-MoE arches add alongside the routed experts. Output is the
//! scaled FFN result; the caller folds it into the residual stream
//! via `moe_combine_two_residuals_f16` (or a separate `add_f16`).
//!
//! Pipeline:
//!
//! 1. `quantize_f16_q8_1(x_norm)`
//! 2-3. gate + up (fused `mmvq_q8_0_gate_up` when both Q8_0, else two `mmvq`)
//! 4-5. fused `swiglu_f32_to_q8_1` when `intermediate % 32 == 0`,
//!      else `swiglu_f32_to_f16` + `quantize_f16_q8_1`
//! 6. down `mmvq` → F32
//! 7. `cast_f16_to_f32(x_norm)` + `shared_expert_scale_f32` (in place)
//! 8. `cast_f32_to_f16` → `shared_out`

use anyhow::{bail, Context, Result};
use flambeau_core::device::DevicePtr;
use flambeau_core::op::QDtype;
use flambeau_ops::Ops;

use crate::WeightHandle;

#[derive(Copy, Clone)]
pub struct SharedExpertDecodeScratch {
    pub x_q8_1: DevicePtr,        // Q8_1 [hidden / 32]
    pub gate_f32: DevicePtr,      // F32 [intermediate]
    pub up_f32: DevicePtr,        // F32 [intermediate]
    pub activated_f16: DevicePtr, // F16 [intermediate] (only used when fused-swiglu path is off)
    pub activated_q8_1: DevicePtr, // Q8_1 [intermediate / 32]
    pub down_f32: DevicePtr,      // F32 [hidden] — scaled in place
    pub x_norm_f32: DevicePtr,    // F32 [hidden] — gate-scale uses this
}

pub struct SharedExpert {
    pub ffn_gate_inp_shexp: DevicePtr, // F16 [hidden] — per-token gate weights
    pub ffn_gate_shexp: WeightHandle,  // [intermediate, hidden]
    pub ffn_up_shexp: WeightHandle,    // [intermediate, hidden]
    pub ffn_down_shexp: WeightHandle,  // [hidden, intermediate]
    pub hidden: usize,
    pub intermediate: usize,
}

impl SharedExpert {
    pub fn new(
        ffn_gate_inp_shexp: DevicePtr,
        ffn_gate_shexp: WeightHandle,
        ffn_up_shexp: WeightHandle,
        ffn_down_shexp: WeightHandle,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        if ffn_gate_shexp.dims != [intermediate, hidden] {
            bail!(
                "ffn_gate_shexp dims {:?} != [{}, {}]",
                ffn_gate_shexp.dims,
                intermediate,
                hidden
            );
        }
        if ffn_up_shexp.dims != [intermediate, hidden] {
            bail!(
                "ffn_up_shexp dims {:?} != [{}, {}]",
                ffn_up_shexp.dims,
                intermediate,
                hidden
            );
        }
        if ffn_down_shexp.dims != [hidden, intermediate] {
            bail!(
                "ffn_down_shexp dims {:?} != [{}, {}]",
                ffn_down_shexp.dims,
                hidden,
                intermediate
            );
        }
        Ok(Self {
            ffn_gate_inp_shexp,
            ffn_gate_shexp,
            ffn_up_shexp,
            ffn_down_shexp,
            hidden,
            intermediate,
        })
    }

    /// One decode step. Reads `x_norm` (F16 `[hidden]`); writes the
    /// per-token-scaled FFN output to `shared_out` (F16 `[hidden]`).
    /// Caller composes `shared_out` with the residual + routed-expert
    /// output via `moe_combine_two_residuals_f16`.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        shared_out: DevicePtr,
        scratch: SharedExpertDecodeScratch,
    ) -> Result<()> {
        let hidden = self.hidden;
        let inter = self.intermediate;

        // 1. Quantise x_norm → Q8_1.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, hidden)
            .context("shexp x_norm → Q8_1")?;

        // 2+3. gate + up. Fuse when both Q8_0.
        let fuse_gate_up = self.ffn_gate_shexp.dtype == QDtype::Q8_0
            && self.ffn_up_shexp.dtype == QDtype::Q8_0;
        if fuse_gate_up {
            ops.mmvq_q8_0_gate_up(
                self.ffn_gate_shexp.ptr,
                self.ffn_up_shexp.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                scratch.up_f32,
                inter,
                inter,
                hidden,
            )
            .context("shexp gate+up fused mmvq_q8_0")?;
        } else {
            ops.mmvq(
                self.ffn_gate_shexp.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                inter,
                hidden,
                self.ffn_gate_shexp.dtype,
            )
            .context("shexp gate mmvq")?;
            ops.mmvq(
                self.ffn_up_shexp.ptr,
                scratch.x_q8_1,
                scratch.up_f32,
                inter,
                hidden,
                self.ffn_up_shexp.dtype,
            )
            .context("shexp up mmvq")?;
        }

        // 4+5. Fused swiglu→Q8_1 when intermediate aligns with QK8_1=32;
        // unfused fallback for off-multiples.
        if inter % 32 == 0 {
            ops.swiglu_f32_to_q8_1(
                scratch.gate_f32,
                scratch.up_f32,
                scratch.activated_q8_1,
                inter,
            )
            .context("shexp swiglu_f32_to_q8_1")?;
        } else {
            ops.swiglu_f32_to_f16(
                scratch.gate_f32,
                scratch.up_f32,
                scratch.activated_f16,
                inter,
            )
            .context("shexp swiglu_f32_to_f16")?;
            ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, inter)
                .context("shexp quantise activated → Q8_1")?;
        }

        // 6. down matmul → F32.
        ops.mmvq(
            self.ffn_down_shexp.ptr,
            scratch.activated_q8_1,
            scratch.down_f32,
            hidden,
            inter,
            self.ffn_down_shexp.dtype,
        )
        .context("shexp down mmvq")?;

        // 7. Cast x_norm to F32 + apply per-token sigmoid gate scaling
        // in place on `down_f32`.
        ops.cast_f16_to_f32(x_norm, scratch.x_norm_f32, hidden)
            .context("shexp cast x_norm → f32")?;
        ops.shared_expert_scale_f32(
            scratch.down_f32,
            scratch.x_norm_f32,
            self.ffn_gate_inp_shexp,
            1,
            hidden,
        )
        .context("shexp shared_expert_scale_f32")?;

        // 8. Cast scaled output back to F16.
        ops.cast_f32_to_f16(scratch.down_f32, shared_out, hidden)
            .context("shexp cast → f16")
    }
}
