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
use flambeau_core::DevicePtr;
use flambeau_ops::hip::{HipDevice, HipStream, OpsRegistry};

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

/// Prefill workspace for the dense-FFN block. Wraps
/// [`flambeau_blocks::OwnedDenseMlpPrefillScratch`]; field access flows
/// through `Deref` to the inner block scratch.
pub struct DenseFfnPrefillScratch {
    inner: flambeau_blocks::OwnedDenseMlpPrefillScratch,
    tracker: flambeau_blocks::RawAllocTracker,
    disposed: bool,
}

impl DenseFfnPrefillScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice, max_tokens: usize) -> Result<Self> {
        let mut tracker = flambeau_blocks::RawAllocTracker::new();
        let dims = flambeau_blocks::DenseMlpScratchDims {
            hidden: cfg.hidden_size,
            intermediate: cfg.moe_intermediate_size,
        };
        let inner = flambeau_blocks::DenseMlp::alloc_prefill_scratch(
            device, &mut tracker, dims, max_tokens,
        )?;
        Ok(Self { inner, tracker, disposed: false })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        self.tracker.dispose(device)
    }

    pub fn view(&self) -> flambeau_blocks::DenseMlpPrefillScratch {
        self.inner.view()
    }
}

impl std::ops::Deref for DenseFfnPrefillScratch {
    type Target = flambeau_blocks::OwnedDenseMlpPrefillScratch;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for DenseFfnPrefillScratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
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

