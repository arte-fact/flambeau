//! Gated-Delta-Net (GDN) forward: decode (one token) + prefill (L tokens).
//! GDN is Qwen3.5-Next / Qwen3.6's hybrid-recurrent layer — it replaces the
//! attention block on even-numbered layer indices with an SSM-style state
//! update. F32 precision end-to-end between MMVQ cast-up and the gated F16
//! output, per-rank state lives in `session::GdnLayerState`.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::hip::{HipDevice, HipStream, OpsRegistry};

use crate::config::Qwen3MoEConfig;
use crate::session::{GdnLayerState, LayerCache};
use crate::weights::{DeviceTensor, GdnWeights};

// ---------------------------------------------------------------------------
// c2 — Gated-Delta-Net decode-step forward.
// ---------------------------------------------------------------------------

/// Workspace buffers for one decode step of a GDN layer. Sized against
/// `Qwen3MoEConfig::gdn` (the hybrid arch's SSM dims).
/// The GDN path keeps F32 precision end-to-end from the post-MMVQ cast
/// through the state update, the ssm_norm, and the gated output. Only
/// the input activation coming in and the delta output going out are
/// F16 — everything in between is F32.
/// Workspace for one decode step of a GDN (DeltaNet) layer. Wraps
/// [`flambeau_blocks::OwnedDeltaNetLayerDecodeScratch`]; field access
/// flows through `Deref` to the inner block scratch.
pub struct GdnScratch {
    inner: flambeau_blocks::OwnedDeltaNetLayerDecodeScratch,
    tracker: flambeau_blocks::RawAllocTracker,
    disposed: bool,
}

impl GdnScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let gdn = cfg.gdn.as_ref().context("GdnScratch requires cfg.gdn")?;
        let mut tracker = flambeau_blocks::RawAllocTracker::new();
        let dims = flambeau_blocks::DeltaNetScratchDims {
            hidden: cfg.hidden_size,
            d_inner: gdn.d_inner,
            num_v_heads: gdn.num_v_heads,
            num_k_heads: gdn.num_k_heads,
            head_k_dim: gdn.head_k_dim,
            head_v_dim: gdn.head_v_dim(),
            conv_channels: gdn.conv_channels(),
            conv_kernel: gdn.conv_kernel,
        };
        let inner =
            flambeau_blocks::DeltaNetLayer::alloc_decode_scratch(device, &mut tracker, dims)?;
        Ok(Self { inner, tracker, disposed: false })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        self.tracker.dispose(device)
    }

    pub fn view(&self) -> flambeau_blocks::DeltaNetLayerDecodeScratch {
        self.inner.view()
    }
}

impl std::ops::Deref for GdnScratch {
    type Target = flambeau_blocks::OwnedDeltaNetLayerDecodeScratch;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for GdnScratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Build a `flambeau_blocks::DeltaNetLayer` from the layer's GDN
/// weights + the model config. Asserts the V1 GDN invariants
/// (split alpha/beta, no fused ssm_ba).
pub fn build_delta_net_block(
    attn_norm: &DeviceTensor,
    weights: &GdnWeights,
    cfg: &Qwen3MoEConfig,
) -> Result<flambeau_blocks::DeltaNetLayer> {
    use super::common::qdtype_of;
    let gdn = cfg.gdn.as_ref().context("build_delta_net_block requires cfg.gdn")?;
    if weights.ssm_ba.is_some() {
        bail!("V1 GDN forward expects split ssm_alpha/ssm_beta; fused ssm_ba is unsupported");
    }
    let ssm_alpha = weights
        .ssm_alpha
        .as_ref()
        .context("V1 GDN forward requires ssm_alpha")?;
    let ssm_beta = weights
        .ssm_beta
        .as_ref()
        .context("V1 GDN forward requires ssm_beta")?;
    let qkv_dt = qdtype_of(weights.attn_qkv.dtype)?;
    let gate_dt = qdtype_of(weights.attn_gate.dtype)?;
    let alpha_dt = qdtype_of(ssm_alpha.dtype)?;
    let beta_dt = qdtype_of(ssm_beta.dtype)?;
    let out_dt = qdtype_of(weights.ssm_out.dtype)?;
    let head_v_dim = gdn.head_v_dim();
    let conv_channels = gdn.conv_channels();
    flambeau_blocks::DeltaNetLayer::new(
        flambeau_blocks::WeightHandle {
            ptr: weights.attn_qkv.ptr,
            dtype: qkv_dt,
            dims: [conv_channels, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: weights.attn_gate.ptr,
            dtype: gate_dt,
            dims: [gdn.d_inner, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: ssm_alpha.ptr,
            dtype: alpha_dt,
            dims: [gdn.num_v_heads, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: ssm_beta.ptr,
            dtype: beta_dt,
            dims: [gdn.num_v_heads, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: weights.ssm_out.ptr,
            dtype: out_dt,
            dims: [cfg.hidden_size, gdn.d_inner],
        },
        weights.ssm_dt_bias.ptr,
        weights.ssm_a.ptr,
        weights.ssm_conv1d.ptr,
        weights.ssm_norm.ptr,
        attn_norm.ptr,
        cfg.hidden_size,
        gdn.d_inner,
        gdn.num_v_heads,
        gdn.num_k_heads,
        gdn.head_k_dim,
        head_v_dim,
        conv_channels,
        gdn.conv_kernel,
        cfg.rms_norm_eps,
        cfg.arch == "qwen3next",
    )
}

/// Decode step for one GDN layer. Consumes `x_in` (F16 `[hidden]`) and
/// writes the pre-residual output to `delta_out` (F16 `[hidden]`). Updates
/// the layer's recurrent state + conv1d history in-place.
/// c2 scope:
/// - Split `ssm_alpha` + `ssm_beta` projections (Qwen3.6 / Qwen3.5 convention).
/// Fused `ssm_ba` (Qwen3-Next) is rejected with a bail for now.
/// - F32 recurrent arithmetic end-to-end from post-MMVQ cast to output
/// projection input (matches candle's `delta_net.rs` precision).
/// - Host alpha/beta/gate compute on `num_v_heads` floats per layer per
/// token. Flagged as follow-up for fusion into a single device kernel
/// (~2 memcpy roundtrips per GDN layer per decode step = 60 roundtrips
/// per decode at 30 GDN layers).
pub fn forward_gdn_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &GdnWeights,
    layer_state: &mut GdnLayerState,
    scratch: &mut GdnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
) -> Result<()> {
    let block = build_delta_net_block(attn_norm, weights, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_decode(
        &hipops,
        device,
        stream,
        x_in,
        delta_out,
        layer_state.state,
        layer_state.conv_history,
        scratch.view(),
    )
}


// `run_mmvq_from_tensor` moved to `forward::common`.

/// Route a `LayerCache` entry through the GDN forward, pulling the
/// correct `GdnLayerState` out of the enum and the matching `GdnWeights`
/// out of the layer's `AttnWeights` variant.
pub fn forward_gdn_layer_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut GdnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
) -> Result<()> {
    let LayerCache::Gdn(state) = layer_cache else {
        bail!(
            "layer {} is not a GDN layer (cache variant mismatch)",
            layer_weights.layer_idx
        );
    };
    let crate::weights::AttnWeights::Gdn(g) = &layer_weights.attn else {
        bail!("layer {} weights are not Gdn variant", layer_weights.layer_idx);
    };
    forward_gdn_decode(
        ops,
        stream,
        device,
        cfg,
        &layer_weights.attn_norm,
        g,
        state,
        scratch,
        x_in,
        delta_out,
    )
}


// ---------------------------------------------------------------------------
// f2 — Gated-Delta-Net prefill (L > 1).
// ---------------------------------------------------------------------------

/// Workspace for one prefill chunk of a GDN layer. Sized once against
/// `(cfg, max_tokens)`. Most buffers scale linearly with L; the state
/// tensor is per-layer (doesn't grow with L) and lives in the session.
/// Prefill workspace for a GDN layer. Wraps
/// [`flambeau_blocks::OwnedDeltaNetLayerPrefillScratch`] (the
/// block-shaped buffers) and adds qwen3-moe-specific batched-decode
/// extras (`slot_state_ptrs`, `slot_conv_history_ptrs`). All
/// allocations share one [`flambeau_blocks::RawAllocTracker`].
pub struct GdnPrefillScratch {
    pub inner: flambeau_blocks::OwnedDeltaNetLayerPrefillScratch,
    /// `[max_tokens] u64` device pointer array used by
    /// `gdn_state_step_alphabeta_f32_s128_batched_slots` to address
    /// each batched slot's `GdnLayerState::state` base pointer
    /// indirectly. Caller populates via memcpy_async per call.
    pub slot_state_ptrs: DevicePtr,
    /// `[max_tokens] u64` device pointer array used by
    /// `gdn_conv_trio_decode_f32_batched_slots` to address each
    /// slot's `GdnLayerState::conv_history` base pointer indirectly
    /// in one fused conv-trio launch per layer.
    pub slot_conv_history_ptrs: DevicePtr,
    tracker: flambeau_blocks::RawAllocTracker,
    disposed: bool,
}

impl GdnPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        let gdn = cfg.gdn.as_ref().context("GdnPrefillScratch requires cfg.gdn")?;
        let mut tracker = flambeau_blocks::RawAllocTracker::new();
        let dims = flambeau_blocks::DeltaNetScratchDims {
            hidden: cfg.hidden_size,
            d_inner: gdn.d_inner,
            num_v_heads: gdn.num_v_heads,
            num_k_heads: gdn.num_k_heads,
            head_k_dim: gdn.head_k_dim,
            head_v_dim: gdn.head_v_dim(),
            conv_channels: gdn.conv_channels(),
            conv_kernel: gdn.conv_kernel,
        };
        let inner = flambeau_blocks::DeltaNetLayer::alloc_prefill_scratch(
            device, &mut tracker, dims, max_tokens,
        )?;
        let (slot_state_ptrs, _) = tracker.alloc_u64(device, max_tokens)?;
        let (slot_conv_history_ptrs, _) = tracker.alloc_u64(device, max_tokens)?;
        Ok(Self {
            inner,
            slot_state_ptrs,
            slot_conv_history_ptrs,
            tracker,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        self.tracker.dispose(device)
    }

    pub fn view(&self) -> flambeau_blocks::DeltaNetLayerPrefillScratch {
        self.inner.view()
    }
}

impl std::ops::Deref for GdnPrefillScratch {
    type Target = flambeau_blocks::OwnedDeltaNetLayerPrefillScratch;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for GdnPrefillScratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Assemble `conv_input[(K-1) + L, conv_channels]` from the layer's
/// `conv_history[(K-1), conv_channels]` + the fresh `qkv_mixed[L, conv_channels]`.
pub(super) fn assemble_conv_input_prefill(
    device: &HipDevice,
    stream: &HipStream,
    history: DevicePtr,
    qkv_mixed: DevicePtr,
    conv_input: DevicePtr,
    n_tokens: usize,
    conv_channels: usize,
    conv_kernel: usize,
) -> Result<()> {
    let row_bytes = conv_channels * 4;
    let hist_rows = conv_kernel - 1;
    // SAFETY: all three buffers have at least the bytes we touch.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            conv_input,
            history,
            hist_rows * row_bytes,
        )?;
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            conv_input.offset_bytes(hist_rows * row_bytes),
            qkv_mixed,
            n_tokens * row_bytes,
        )?;
    }
    Ok(())
}

/// After the conv has read `conv_input[(K-1) + L]`, update the layer's
/// history slot to the last `K-1` rows — `conv_input[L..L+K-1]`. One
/// memcpy (may alias if L == 0, but prefill has L ≥ 1).
pub(super) fn shift_conv_history_prefill(
    device: &HipDevice,
    stream: &HipStream,
    conv_input: DevicePtr,
    history: DevicePtr,
    n_tokens: usize,
    conv_channels: usize,
    conv_kernel: usize,
) -> Result<()> {
    let row_bytes = conv_channels * 4;
    let hist_rows = conv_kernel - 1;
    // SAFETY: conv_input has (K-1 + L) valid rows; history has K-1.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            history,
            conv_input.offset_bytes(n_tokens * row_bytes),
            hist_rows * row_bytes,
        )?;
    }
    Ok(())
}

/// Prefill step for one GDN layer. Consumes `x_in` (F16 `[L, hidden]`),
/// updates `layer_state.state` + `layer_state.conv_history` across all L
/// tokens, and writes the pre-residual `delta_out` (F16 `[L, hidden]`).
pub fn forward_gdn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &GdnWeights,
    layer_state: &mut GdnLayerState,
    scratch: &mut GdnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    n_tokens: usize,
    state_event: Option<&flambeau_backend_hip::HipEvent>,
) -> Result<()> {
    let block = build_delta_net_block(attn_norm, weights, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_prefill(
        &hipops,
        device,
        stream,
        x_in,
        delta_out,
        layer_state.state,
        layer_state.conv_history,
        scratch.view(),
        n_tokens,
        state_event,
    )
}

// `run_qmatmul_from_tensor` moved to `forward::common`.

