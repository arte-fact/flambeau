//! Dense-FFN forward (decode + prefill) — the single gate/up/down triple
//! used by arch=qwen35 (Qwen3.5-9B). MoE models route through
//! `forward::moe` instead; dense models bypass the router and call the
//! `forward_dense_ffn_*` entry points directly from `forward_layer_*`.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition — every unsafe block is a kernel.launch or \
              memcpy_async over DevicePtrs owned by the session's scratch / weights / \
              KV cache. Buffers live for the whole session; sync is driven by the top- \
              level forward_*_decode/prefill caller."
)]

use anyhow::{Context, Result};
use flambeau_core::{Device, DevicePtr};
use flambeau_ops::hip::{
    cast::cast_f32_to_f16,
    mlp::{add_f16, swiglu_f32},
    norm::{quantize_f16_q8_1, quantize_f16_q8_1_mmq},
    qmatmul::{mmvq_q8_0_gate_up, qmatmul},
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::BlockQ8_1;

use super::common::qdtype_of;
use crate::config::Qwen3MoEConfig;
use crate::weights::DenseFfnWeights;

// ---------------------------------------------------------------------------
// V2.2.c — dense FFN decode (arch=qwen35).
//
// One gate+up+down triple per layer (no router, no experts, no shared expert).
// Structurally identical to forward_shared_expert_decode minus the
// sigmoid-gate post-scale. `forward_dense_ffn_decode` writes the residual sum
// `x_out = residual + FFN(x_norm)` in one shot so callers don't need to add
// later.
// ---------------------------------------------------------------------------

pub struct DenseFfnScratch {
    // Q8_1 of `x_norm`, shared across gate/up matmuls.
    pub x_q8_1: DevicePtr,
    // Dense gate/up matmul outputs, F32 [inter].
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    // SwiGLU output + F16 round-trip for the down matmul input.
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    // Down matmul outputs (F32 → F16) for the residual add.
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
    // Bookkeeping.
    x_q8_1_bytes: usize,
    inter_f32_bytes: usize,
    inter_f16_bytes: usize,
    inter_q8_1_bytes: usize,
    hidden_f32_bytes: usize,
    hidden_f16_bytes: usize,
    disposed: bool,
}

impl DenseFfnScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(inter % 32 == 0, "inter (moe_intermediate_size) must be a multiple of QK8_1=32");

        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let inter_f32_bytes = inter * 4;
        let inter_f16_bytes = inter * 2;
        let inter_q8_1_bytes = (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let hidden_f32_bytes = hidden * 4;
        let hidden_f16_bytes = hidden * 2;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let gate_f32 = device.alloc(inter_f32_bytes)?;
        let up_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f16 = device.alloc(inter_f16_bytes)?;
        let activated_q8_1 = device.alloc(inter_q8_1_bytes)?;
        let down_f32 = device.alloc(hidden_f32_bytes)?;
        let down_f16 = device.alloc(hidden_f16_bytes)?;

        Ok(Self {
            x_q8_1,
            gate_f32,
            up_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            down_f16,
            x_q8_1_bytes,
            inter_f32_bytes,
            inter_f16_bytes,
            inter_q8_1_bytes,
            hidden_f32_bytes,
            hidden_f16_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.gate_f32, self.inter_f32_bytes)?;
            device.dealloc(self.up_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f16, self.inter_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.inter_q8_1_bytes)?;
            device.dealloc(self.down_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.down_f16, self.hidden_f16_bytes)?;
        }
        Ok(())
    }
}

impl Drop for DenseFfnScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "DenseFfnScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One decode step of a dense FFN block (arch=qwen35). Writes
/// `x_out = residual + ffn_down(swiglu(ffn_gate(x_norm), ffn_up(x_norm)))`
/// into `x_out`. No router, no experts, no sigmoid-gate scaling.
pub fn forward_dense_ffn_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    dense: &DenseFfnWeights,
    scratch: &mut DenseFfnScratch,
    x_norm: DevicePtr,
    residual: DevicePtr,
    x_out: DevicePtr,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;

    // 1. Quantise x_norm → Q8_1 once, reused for gate/up.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
        .context("dense ffn x_norm → Q8_1")?;

    // 2+3. gate + up matmuls share x_q8_1. V2.20.b — when both are Q8_0
    //      (Qwen3.6-27B dense path, 66.3% of decode wall pre-fusion), fuse
    //      into one mmvq_q8_0_gate_up launch. Kernel is the same one the
    //      full-attn layer uses for K+V fusion; parity cert in
    //      `crates/bench/tests/mmvq_q8_0_gate_up_parity.rs` proves bit-exact
    //      equivalence to two independent single-row calls. Disabled only
    //      by the coarse `FLAMBEAU_VARIANT=baseline` or the specific
    //      `FLAMBEAU_DENSE_GATE_UP=unfused`.
    let global_baseline = std::env::var("FLAMBEAU_VARIANT").as_deref() == Ok("baseline");
    let specific_off = std::env::var("FLAMBEAU_DENSE_GATE_UP").as_deref() == Ok("unfused");
    let fuse_gate_up = !global_baseline && !specific_off
        && dense.ffn_gate.dtype == flambeau_quant::GgmlDType::Q8_0
        && dense.ffn_up.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_gate_up {
        mmvq_q8_0_gate_up(
            ops,
            stream,
            dense.ffn_gate.ptr,
            dense.ffn_up.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            scratch.up_f32,
            inter,
            inter,
            hidden,
        )
        .context("dense ffn gate+up fused mmvq_q8_0")?;
    } else {
        qmatmul(
            ops,
            stream,
            dense.ffn_gate.ptr,
            scratch.x_q8_1,
            DevicePtr(0),
            scratch.gate_f32,
            1,
            hidden,
            inter,
            qdtype_of(dense.ffn_gate.dtype)?,
        )
        .context("dense ffn gate qmatmul")?;
        qmatmul(
            ops,
            stream,
            dense.ffn_up.ptr,
            scratch.x_q8_1,
            DevicePtr(0),
            scratch.up_f32,
            1,
            hidden,
            inter,
            qdtype_of(dense.ffn_up.dtype)?,
        )
        .context("dense ffn up qmatmul")?;
    }

    // 4+5. V2.23.d.2 fused SwiGLU → F16 + quantise; skips the standalone cast.
    flambeau_ops::hip::mlp::swiglu_f32_to_f16(
        ops,
        stream,
        scratch.gate_f32,
        scratch.up_f32,
        scratch.activated_f16,
        inter,
    )
    .context("dense ffn swiglu_f32_to_f16")?;
    quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        inter,
    )
    .context("dense ffn quantise activated → Q8_1")?;

    // 6. down matmul: weight[hidden, inter] × activated[inter] → down_f32[hidden].
    //    Decode path: m=1 never hits MmqLdsX64.
    qmatmul(
        ops,
        stream,
        dense.ffn_down.ptr,
        scratch.activated_q8_1,
        DevicePtr(0),
        scratch.down_f32,
        1,
        inter,
        hidden,
        qdtype_of(dense.ffn_down.dtype)?,
    )
    .context("dense ffn down qmatmul")?;

    // 7. Cast down F32→F16, residual add into x_out.
    cast_f32_to_f16(ops, stream, scratch.down_f32, scratch.down_f16, hidden)
        .context("dense ffn cast down → f16")?;
    add_f16(ops, stream, residual, scratch.down_f16, x_out, hidden)
        .context("dense ffn residual: residual + down")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// V2.2.c — dense FFN prefill (arch=qwen35, L tokens).
// ---------------------------------------------------------------------------

pub struct DenseFfnPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    /// V2.2.d.P8 — DS4 Q8_1 MMQ layout sibling of `x_q8_1`, consumed by the
    /// 4-warp LDS-tiled Q4_1 MMQ (and future Q4_K / Q6_K MMQ turbo kernels)
    /// at m ≥ 128. Populated from `x_q8_1` F16 source via the
    /// `flambeau_quantize_f16_q8_1_mmq` kernel alongside the standard quant.
    pub x_q8_1_mmq: DevicePtr,
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    /// V2.2.d.P8 — DS4 sibling of `activated_q8_1` for the down-projection.
    pub activated_q8_1_mmq: DevicePtr,
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
    x_q8_1_bytes: usize,
    x_q8_1_mmq_bytes: usize,
    inter_f32_bytes: usize,
    inter_f16_bytes: usize,
    inter_q8_1_bytes: usize,
    inter_q8_1_mmq_bytes: usize,
    hidden_f32_bytes: usize,
    hidden_f16_bytes: usize,
    disposed: bool,
}

impl DenseFfnPrefillScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice, max_tokens: usize) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        assert!(max_tokens >= 1);
        assert!(
            hidden % 128 == 0,
            "hidden must be a multiple of QK8_1_MMQ=128"
        );
        assert!(
            inter % 128 == 0,
            "moe_intermediate_size must be a multiple of QK8_1_MMQ=128"
        );
        let mmq_block = std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let x_q8_1_bytes = max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let x_q8_1_mmq_bytes = max_tokens * (hidden / 128) * mmq_block;
        let inter_f32_bytes = max_tokens * inter * 4;
        let inter_f16_bytes = max_tokens * inter * 2;
        let inter_q8_1_bytes = max_tokens * (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let inter_q8_1_mmq_bytes = max_tokens * (inter / 128) * mmq_block;
        let hidden_f32_bytes = max_tokens * hidden * 4;
        let hidden_f16_bytes = max_tokens * hidden * 2;
        Ok(Self {
            max_tokens,
            x_q8_1: device.alloc(x_q8_1_bytes)?,
            x_q8_1_mmq: device.alloc(x_q8_1_mmq_bytes)?,
            gate_f32: device.alloc(inter_f32_bytes)?,
            up_f32: device.alloc(inter_f32_bytes)?,
            activated_f32: device.alloc(inter_f32_bytes)?,
            activated_f16: device.alloc(inter_f16_bytes)?,
            activated_q8_1: device.alloc(inter_q8_1_bytes)?,
            activated_q8_1_mmq: device.alloc(inter_q8_1_mmq_bytes)?,
            down_f32: device.alloc(hidden_f32_bytes)?,
            down_f16: device.alloc(hidden_f16_bytes)?,
            x_q8_1_bytes,
            x_q8_1_mmq_bytes,
            inter_f32_bytes,
            inter_f16_bytes,
            inter_q8_1_bytes,
            inter_q8_1_mmq_bytes,
            hidden_f32_bytes,
            hidden_f16_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.x_q8_1_mmq, self.x_q8_1_mmq_bytes)?;
            device.dealloc(self.gate_f32, self.inter_f32_bytes)?;
            device.dealloc(self.up_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f16, self.inter_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.inter_q8_1_bytes)?;
            device.dealloc(self.activated_q8_1_mmq, self.inter_q8_1_mmq_bytes)?;
            device.dealloc(self.down_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.down_f16, self.hidden_f16_bytes)?;
        }
        Ok(())
    }
}

impl Drop for DenseFfnPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "DenseFfnPrefillScratch dropped without dispose(device); buffers leaked"
            );
        }
    }
}

pub fn forward_dense_ffn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    dense: &DenseFfnWeights,
    scratch: &mut DenseFfnPrefillScratch,
    x_norm: DevicePtr,
    residual: DevicePtr,
    x_out: DevicePtr,
    n_tokens: usize,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;

    // Quantise x_norm to BOTH Q8_1 layouts: the standard per-row layout
    // consumed by MMVQ / Mmq4Warp kernels, and the DS4 MMQ layout consumed
    // by the 4-warp LDS-tiled turbo kernel (Q4_1 at m ≥ 128). qmatmul()
    // dispatches to whichever matches the weight dtype + M.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("dense ffn prefill x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(ops, stream, x_norm, scratch.x_q8_1_mmq, hidden, n_tokens)
        .context("dense ffn prefill x_norm → Q8_1 (MMQ DS4)")?;

    qmatmul(
        ops, stream,
        dense.ffn_gate.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.gate_f32,
        n_tokens, hidden, inter,
        qdtype_of(dense.ffn_gate.dtype)?,
    ).context("dense ffn prefill gate qmatmul")?;
    qmatmul(
        ops, stream,
        dense.ffn_up.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.up_f32,
        n_tokens, hidden, inter,
        qdtype_of(dense.ffn_up.dtype)?,
    ).context("dense ffn prefill up qmatmul")?;

    // V2.23.d.2 fused SwiGLU → F16 (replaces swiglu_f32 + cast_f32_to_f16).
    flambeau_ops::hip::mlp::swiglu_f32_to_f16(ops, stream, scratch.gate_f32, scratch.up_f32, scratch.activated_f16, n_tokens * inter)
        .context("dense ffn prefill swiglu_f32_to_f16")?;
    quantize_f16_q8_1(ops, stream, scratch.activated_f16, scratch.activated_q8_1, n_tokens * inter)
        .context("dense ffn prefill quantise activated → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(ops, stream, scratch.activated_f16, scratch.activated_q8_1_mmq, inter, n_tokens)
        .context("dense ffn prefill quantise activated → Q8_1 (MMQ DS4)")?;

    qmatmul(
        ops, stream,
        dense.ffn_down.ptr,
        scratch.activated_q8_1, scratch.activated_q8_1_mmq,
        scratch.down_f32,
        n_tokens, inter, hidden,
        qdtype_of(dense.ffn_down.dtype)?,
    ).context("dense ffn prefill down qmatmul")?;

    cast_f32_to_f16(ops, stream, scratch.down_f32, scratch.down_f16, n_tokens * hidden)
        .context("dense ffn prefill cast down → f16")?;
    add_f16(ops, stream, residual, scratch.down_f16, x_out, n_tokens * hidden)
        .context("dense ffn prefill residual: residual + down")?;
    Ok(())
}

