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

use anyhow::Result;
use flambeau_core::{Device, DevicePtr};
use flambeau_ops::hip::{HipDevice, HipStream, OpsRegistry};
use flambeau_quant::BlockQ8_1;

use super::common::qdtype_of;
use crate::config::Qwen3MoEConfig;
use crate::weights::DenseFfnWeights;

// ---------------------------------------------------------------------------
// dense FFN decode (arch=qwen35).
// One gate+up+down triple per layer (no router, no experts, no shared expert).
// Structurally identical to forward_shared_expert_decode minus the
// sigmoid-gate post-scale. `forward_dense_ffn_decode` writes the residual sum
// `x_out = residual + FFN(x_norm)` in one shot so callers don't need to add
// later.
// ---------------------------------------------------------------------------

/// Wrapper that owns an [`flambeau_blocks::OwnedDenseMlpDecodeScratch`]
/// + the per-stage [`flambeau_blocks::RawAllocTracker`] holding its
/// allocations. Field access (`scratch.x_q8_1`, `scratch.gate_f32`, …)
/// flows through `Deref` to the inner block scratch.
pub struct DenseFfnScratch {
    inner: flambeau_blocks::OwnedDenseMlpDecodeScratch,
    tracker: flambeau_blocks::RawAllocTracker,
    disposed: bool,
}

impl DenseFfnScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let mut tracker = flambeau_blocks::RawAllocTracker::new();
        let dims = flambeau_blocks::DenseMlpScratchDims {
            hidden: cfg.hidden_size,
            intermediate: cfg.moe_intermediate_size,
        };
        let inner =
            flambeau_blocks::DenseMlp::alloc_decode_scratch(device, &mut tracker, dims)?;
        Ok(Self { inner, tracker, disposed: false })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        self.tracker.dispose(device)
    }

    /// Build a by-value view shaped for
    /// `flambeau_blocks::DenseMlp::forward_decode`.
    pub fn view(&self) -> flambeau_blocks::DenseMlpDecodeScratch {
        self.inner.view()
    }
}

impl std::ops::Deref for DenseFfnScratch {
    type Target = flambeau_blocks::OwnedDenseMlpDecodeScratch;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for DenseFfnScratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Build a `flambeau_blocks::DenseMlp` from already-unpacked dense FFN
/// weights + the model config. The block holds only `WeightHandle`s;
/// `ModelWeights` keeps owning the underlying allocations.
pub fn build_dense_mlp_block(
    dense: &DenseFfnWeights,
    cfg: &Qwen3MoEConfig,
) -> Result<flambeau_blocks::DenseMlp> {
    use super::common::mat_shape;
    let g_dtype = qdtype_of(dense.ffn_gate.dtype)?;
    let u_dtype = qdtype_of(dense.ffn_up.dtype)?;
    let d_dtype = qdtype_of(dense.ffn_down.dtype)?;
    let (g_rows, g_k) = mat_shape(&dense.ffn_gate)?;
    let (u_rows, u_k) = mat_shape(&dense.ffn_up)?;
    let (d_rows, d_k) = mat_shape(&dense.ffn_down)?;
    flambeau_blocks::DenseMlp::new(
        flambeau_blocks::WeightHandle { ptr: dense.ffn_gate.ptr, dtype: g_dtype, dims: [g_rows, g_k] },
        flambeau_blocks::WeightHandle { ptr: dense.ffn_up.ptr, dtype: u_dtype, dims: [u_rows, u_k] },
        flambeau_blocks::WeightHandle { ptr: dense.ffn_down.ptr, dtype: d_dtype, dims: [d_rows, d_k] },
        cfg.hidden_size,
        cfg.moe_intermediate_size,
    )
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
    let block = build_dense_mlp_block(dense, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_decode(&hipops, x_norm, residual, x_out, scratch.view())
}

// ---------------------------------------------------------------------------
// dense FFN prefill (arch=qwen35, L tokens).
// ---------------------------------------------------------------------------

pub struct DenseFfnPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    /// 8 — DS4 Q8_1 MMQ layout sibling of `x_q8_1`, consumed by the
    /// 4-warp LDS-tiled Q4_1 MMQ (and future Q4_K / Q6_K MMQ turbo kernels)
    /// at m ≥ 128. Populated from `x_q8_1` F16 source via the
    /// `flambeau_quantize_f16_q8_1_mmq` kernel alongside the standard quant.
    pub x_q8_1_mmq: DevicePtr,
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    /// 8 — DS4 sibling of `activated_q8_1` for the down-projection.
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

impl DenseFfnPrefillScratch {
    /// View shaped for `flambeau_blocks::DenseMlp::forward_prefill`.
    pub fn view(&self) -> flambeau_blocks::DenseMlpPrefillScratch {
        flambeau_blocks::DenseMlpPrefillScratch {
            max_tokens: self.max_tokens,
            x_q8_1: self.x_q8_1,
            x_q8_1_mmq: self.x_q8_1_mmq,
            gate_f32: self.gate_f32,
            up_f32: self.up_f32,
            activated_f16: self.activated_f16,
            activated_q8_1: self.activated_q8_1,
            activated_q8_1_mmq: self.activated_q8_1_mmq,
            down_f32: self.down_f32,
            down_f16: self.down_f16,
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
    let block = build_dense_mlp_block(dense, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_prefill(&hipops, x_norm, residual, x_out, scratch.view(), n_tokens)
}

