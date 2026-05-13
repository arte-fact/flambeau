//! tensor-parallel `forward_dense_ffn_decode`.
//! Thin shim that constructs a [`flambeau_blocks::DenseMlpTp`] from the
//! per-rank sliced [`DeviceTensor`]s and forwards the call. The block
//! does the actual kernel-launch sequence; this wrapper preserves the
//! qwen3-moe-specific entry point (cfg-driven shape derivation +
//! `DenseFfnScratch` adaptation) so existing callers don't need to
//! change.
//!
//! See `flambeau_blocks::dense_mlp::DenseMlpTp` for the kernel
//! pipeline (1 quantise → 2+3 gate/up fused-or-split → 4 activation
//! → 5 quantise → 6 down → 7 cast → partial_ffn_out) and the
//! activation/fast-path policy (Q8_0 / Q4_0_t128 / Q4_1 gate+up
//! fused, SwiGLU for qwen3.x).

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "shim — no unsafe blocks in this file; flag inherited from prior body."
)]

use anyhow::{bail, Context, Result};
use flambeau_blocks::{
    Activation, DenseMlpDecodeScratch, DenseMlpPrefillScratch, DenseMlpTp,
};
use flambeau_core::DevicePtr;
use flambeau_ops::hip::{HipOps, HipStream, OpsRegistry};

use super::common::qdtype_of;
use super::dense_ffn::DenseFfnScratch;
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

fn build_block(
    cfg: &Qwen3MoEConfig,
    ffn_gate: &DeviceTensor,
    ffn_up: &DeviceTensor,
    ffn_down: &DeviceTensor,
    tp_world: u32,
) -> Result<DenseMlpTp> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    if inter % world != 0 {
        bail!("moe_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;

    let gate_h = flambeau_blocks::WeightHandle {
        ptr: ffn_gate.ptr,
        dtype: qdtype_of(ffn_gate.dtype)?,
        dims: [ffn_gate.dims[0] as usize, ffn_gate.dims[1] as usize],
    };
    let up_h = flambeau_blocks::WeightHandle {
        ptr: ffn_up.ptr,
        dtype: qdtype_of(ffn_up.dtype)?,
        dims: [ffn_up.dims[0] as usize, ffn_up.dims[1] as usize],
    };
    let down_h = flambeau_blocks::WeightHandle {
        ptr: ffn_down.ptr,
        dtype: qdtype_of(ffn_down.dtype)?,
        dims: [ffn_down.dims[0] as usize, ffn_down.dims[1] as usize],
    };
    DenseMlpTp::new(gate_h, up_h, down_h, hidden, local_inter, Activation::SwiGLU)
}

/// Per-rank decode for one dense FFN layer. Delegates to
/// [`DenseMlpTp::forward_decode`].
#[expect(
    clippy::too_many_arguments,
    reason = "preserves the historic flat parameter list for callers."
)]
pub fn forward_dense_ffn_decode_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate: &DeviceTensor,
    ffn_up: &DeviceTensor,
    ffn_down: &DeviceTensor,
    scratch: &mut DenseFfnScratch,
    x_norm: DevicePtr,
    partial_ffn_out: DevicePtr,
    tp_world: u32,
    pre_quantized: bool,
) -> Result<()> {
    let block = build_block(cfg, ffn_gate, ffn_up, ffn_down, tp_world)
        .context("DenseMlpTp::new (decode)")?;
    let blocks_scratch = DenseMlpDecodeScratch {
        x_q8_1: scratch.x_q8_1,
        gate_f32: scratch.gate_f32,
        up_f32: scratch.up_f32,
        activated_f16: scratch.activated_f16,
        activated_q8_1: scratch.activated_q8_1,
        down_f32: scratch.down_f32,
        down_f16: scratch.down_f16,
    };
    let hip_ops = HipOps::new(ops, stream);
    block.forward_decode(&hip_ops, x_norm, partial_ffn_out, blocks_scratch, pre_quantized)
}

/// Per-rank L-token prefill for one dense FFN layer. Delegates to
/// [`DenseMlpTp::forward_prefill`].
#[expect(
    clippy::too_many_arguments,
    reason = "preserves the historic flat parameter list for callers."
)]
pub fn forward_dense_ffn_prefill_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate: &DeviceTensor,
    ffn_up: &DeviceTensor,
    ffn_down: &DeviceTensor,
    scratch: &mut super::dense_ffn::DenseFfnPrefillScratch,
    x_norm: DevicePtr,
    partial_ffn_out: DevicePtr,
    n_tokens: usize,
    tp_world: u32,
) -> Result<()> {
    let block = build_block(cfg, ffn_gate, ffn_up, ffn_down, tp_world)
        .context("DenseMlpTp::new (prefill)")?;
    let blocks_scratch = DenseMlpPrefillScratch {
        max_tokens: scratch.max_tokens,
        x_q8_1: scratch.x_q8_1,
        x_q8_1_mmq: scratch.x_q8_1_mmq,
        gate_f32: scratch.gate_f32,
        up_f32: scratch.up_f32,
        activated_f16: scratch.activated_f16,
        activated_q8_1: scratch.activated_q8_1,
        activated_q8_1_mmq: scratch.activated_q8_1_mmq,
        down_f32: scratch.down_f32,
        down_f16: scratch.down_f16,
    };
    let hip_ops = HipOps::new(ops, stream);
    block.forward_prefill(&hip_ops, x_norm, partial_ffn_out, blocks_scratch, n_tokens)
}

#[cfg(test)]
mod tests {
    // Substantive tests need GPU; covered by parity smoke
    // (compares TP at world=1 against PP forward_dense_ffn_decode).
}
