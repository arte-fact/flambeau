//! Per-layer side-channel embedding block.
//!
//! Gemma 4 E2B / E4B applies a token-derived F32 side-channel table
//! to each layer's output as a residual delta:
//!
//! ```text
//! gate_f32      = inp_gate @ pe_in                 (dense_gemv_f32_f16)
//! activated_f32 = gelu(gate_f32) * table_slice     (gelu_mul_f32)
//! activated_f16 = cast(activated_f32)              (cast_f32_to_f16)
//! proj_f32      = proj @ activated_f16             (dense_gemv_f32_f16)
//! proj_f16      = cast(proj_f32)                   (cast_f32_to_f16)
//! normed_f16    = rmsnorm_f16(proj_f16, post_norm) (rmsnorm_f16)
//! x_out         = pe_in + normed_f16               (add_f16)
//! ```
//!
//! Only the per-layer apply lives here. The (host-side) build of
//! `inp_per_layer_table[]` is GGUF-coupled and stays in the gemma4
//! crate alongside the model-specific tensor names.

use anyhow::{Context, Result};
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;

/// Per-layer weights for the side-channel apply. All F32 on disk;
/// `post_norm_f16` gets cast to F16 at upload time to match
/// `rmsnorm_f16`'s contract.
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbedLayerWeights {
    /// F32 `[pe, hidden]`.
    pub inp_gate: DevicePtr,
    /// F32 `[hidden, pe]`.
    pub proj: DevicePtr,
    /// F16 `[hidden]` (cast from on-disk F32).
    pub post_norm_f16: DevicePtr,
}

/// Per-call scratch buffers — caller-allocated, reused per layer.
/// Sized so `gate_out_f32`, `activated_f32`, `activated_f16` cover
/// `pe` elements; `proj_out_f32`, `proj_out_f16`, `normed_f16` cover
/// `hidden`. The `_f32` / `_f16` sibling slots may overlap with the
/// layer's FFN scratch when the FFN intermediate buffers happen to be
/// large enough (the gemma4 PP path reuses `scratch.gate_f32` etc.).
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbedDecodeScratch {
    pub gate_out_f32: DevicePtr,
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub proj_out_f32: DevicePtr,
    pub proj_out_f16: DevicePtr,
    pub normed_f16: DevicePtr,
}

/// Side-channel embedding block. Holds the per-layer weights plus the
/// shape config; constructed once per layer at upload time and reused
/// across forward calls.
pub struct PerLayerEmbedBlock {
    pub weights: PerLayerEmbedLayerWeights,
    pub pe: usize,
    pub hidden: usize,
    pub rms_norm_eps: f32,
}

impl PerLayerEmbedBlock {
    pub fn new(
        weights: PerLayerEmbedLayerWeights,
        pe: usize,
        hidden: usize,
        rms_norm_eps: f32,
    ) -> Self {
        Self {
            weights,
            pe,
            hidden,
            rms_norm_eps,
        }
    }

    /// Apply the side-channel post-block. Reads `pe_in` (F16 `[hidden]`,
    /// the layer's output residual) plus `table_slice` (F32 `[pe]`, this
    /// layer's slice of the prebuilt per-layer table) and writes the
    /// updated residual to `x_out` (F16 `[hidden]`). `pe_in` and `x_out`
    /// may alias for in-place update.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        pe_in: DevicePtr,
        table_slice: DevicePtr,
        scratch: PerLayerEmbedDecodeScratch,
        x_out: DevicePtr,
    ) -> Result<()> {
        let PerLayerEmbedDecodeScratch {
            gate_out_f32,
            activated_f32,
            activated_f16,
            proj_out_f32,
            proj_out_f16,
            normed_f16,
        } = scratch;
        ops.dense_gemv_f32_f16(self.weights.inp_gate, pe_in, gate_out_f32, self.pe, self.hidden)
            .context("per_layer_embd inp_gate")?;
        ops.gelu_mul_f32(gate_out_f32, table_slice, activated_f32, self.pe)
            .context("per_layer_embd gelu_mul")?;
        ops.cast_f32_to_f16(activated_f32, activated_f16, self.pe)
            .context("per_layer_embd cast activated → f16")?;
        ops.dense_gemv_f32_f16(self.weights.proj, activated_f16, proj_out_f32, self.hidden, self.pe)
            .context("per_layer_embd proj")?;
        ops.cast_f32_to_f16(proj_out_f32, proj_out_f16, self.hidden)
            .context("per_layer_embd cast proj → f16")?;
        ops.rmsnorm_f16(
            proj_out_f16,
            self.weights.post_norm_f16,
            normed_f16,
            1,
            self.hidden,
            self.rms_norm_eps,
        )
        .context("per_layer_embd post_norm")?;
        ops.add_f16(pe_in, normed_f16, x_out, self.hidden)
            .context("per_layer_embd residual add")?;
        Ok(())
    }
}

/// Compute the device pointer for `inp_per_layer_table[il]` — `il * pe`
/// F32 elements into `table_base`. Helper kept on the block module for
/// callers that compute per-layer slices outside the forward call.
pub fn table_slice_ptr(table_base: DevicePtr, il: usize, pe: usize) -> DevicePtr {
    table_base.offset_bytes(il * pe * 4)
}
