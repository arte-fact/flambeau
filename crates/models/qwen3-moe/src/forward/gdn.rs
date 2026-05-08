//! Gated-Delta-Net (GDN) forward: decode (one token) + prefill (L tokens).
//! GDN is Qwen3.5-Next / Qwen3.6's hybrid-recurrent layer — it replaces the
//! attention block on even-numbered layer indices with an SSM-style state
//! update. F32 precision end-to-end between MMVQ cast-up and the gated F16
//! output, per-rank state lives in `session::GdnLayerState`.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::hip::{
    cast::cast_f32_to_f16,
    conv::causal_conv1d_f32,
    mlp::{scale_f32, silu_f32, swiglu_f32},
    norm::{
        l2_norm_f32, quantize_f16_q8_1, quantize_f16_q8_1_mmq, quantize_q8_1,
        quantize_q8_1_mmq, rmsnorm_f16, rmsnorm_f32,
    },
    recurrent::{gdn_split_qkv_f32, gdn_state_step_alphabeta_f32_s128},
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::BlockQ8_1;

use super::common::run_qmatmul_from_tensor;
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
pub struct GdnScratch {
    // Fused norm-and-quant output for attn_qkv/attn_gate/ssm_alpha/ssm_beta.
    pub x_q8_1: DevicePtr,
    // F32 mmvq outputs (re-used across projections where sizes permit).
    pub qkv_mixed_f32: DevicePtr,   // [conv_channels]
    pub z_f32: DevicePtr,           // [d_inner]
    pub alpha_f32: DevicePtr,       // [num_v_heads]
    pub beta_f32: DevicePtr,        // [num_v_heads]
    // Conv1d staging: [conv_kernel, conv_channels] packed F32. First
    // (conv_kernel-1) rows come from the layer's conv_history; last row
    // is the current qkv_mixed.
    pub conv_input: DevicePtr,
    pub conv_out: DevicePtr,        // [conv_channels] — silu_in
    pub silu_out: DevicePtr,        // [conv_channels] — post-silu conv-out (Q|K|V packed)
    // L2-normalised Q and K per head (F32).
    pub q_norm_f32: DevicePtr,      // [num_k_heads, head_k_dim]
    pub k_norm_f32: DevicePtr,      // [num_k_heads, head_k_dim]
    // State-step output (F32): [num_v_heads, head_v_dim].
    pub state_out: DevicePtr,
    // ssm_norm output and gated output (F32).
    pub out_normed: DevicePtr,      // [num_v_heads, head_v_dim]
    pub gated_f32: DevicePtr,       // [d_inner]
    pub gated_q8_1: DevicePtr,      // Q8_1 blocks for ssm_out mmvq input
    pub ssm_out_f32: DevicePtr,     // [hidden]
    // Device gate + beta, consumed by `gdn_state_step_f32_s128`.
    pub gate_device: DevicePtr,     // [num_v_heads] F32
    pub beta_device: DevicePtr,     // [num_v_heads] F32
    // Bookkeeping for teardown.
    x_q8_1_bytes: usize,
    conv_channels_f32_bytes: usize,
    d_inner_f32_bytes: usize,
    num_v_heads_f32_bytes: usize,
    conv_input_bytes: usize,
    qk_f32_bytes: usize,
    v_f32_bytes: usize,
    gated_q8_1_bytes: usize,
    hidden_f32_bytes: usize,
    disposed: bool,
}

impl GdnScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let gdn = cfg.gdn.as_ref().context("GdnScratch requires cfg.gdn")?;
        let hidden = cfg.hidden_size;
        let d_inner = gdn.d_inner;
        let num_v_heads = gdn.num_v_heads;
        let num_k_heads = gdn.num_k_heads;
        let head_k_dim = gdn.head_k_dim;
        let head_v_dim = gdn.head_v_dim();
        let conv_channels = gdn.conv_channels();
        let conv_kernel = gdn.conv_kernel;

        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(
            head_k_dim == 128 && head_v_dim == 128,
            "gdn_state_step kernel only instantiated at S_v=128"
        );

        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let conv_channels_f32_bytes = conv_channels * 4;
        let d_inner_f32_bytes = d_inner * 4;
        let num_v_heads_f32_bytes = num_v_heads * 4;
        let conv_input_bytes = conv_kernel * conv_channels * 4;
        let qk_f32_bytes = num_k_heads * head_k_dim * 4;
        let v_f32_bytes = num_v_heads * head_v_dim * 4;
        let gated_q8_1_bytes = (d_inner / 32) * std::mem::size_of::<BlockQ8_1>();
        let hidden_f32_bytes = hidden * 4;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let qkv_mixed_f32 = device.alloc(conv_channels_f32_bytes)?;
        let z_f32 = device.alloc(d_inner_f32_bytes)?;
        let alpha_f32 = device.alloc(num_v_heads_f32_bytes)?;
        let beta_f32 = device.alloc(num_v_heads_f32_bytes)?;
        let conv_input = device.alloc(conv_input_bytes)?;
        let conv_out = device.alloc(conv_channels_f32_bytes)?;
        let silu_out = device.alloc(conv_channels_f32_bytes)?;
        let q_norm_f32 = device.alloc(qk_f32_bytes)?;
        let k_norm_f32 = device.alloc(qk_f32_bytes)?;
        let state_out = device.alloc(v_f32_bytes)?;
        let out_normed = device.alloc(v_f32_bytes)?;
        let gated_f32 = device.alloc(d_inner_f32_bytes)?;
        let gated_q8_1 = device.alloc(gated_q8_1_bytes)?;
        let ssm_out_f32 = device.alloc(hidden_f32_bytes)?;
        let gate_device = device.alloc(num_v_heads_f32_bytes)?;
        let beta_device = device.alloc(num_v_heads_f32_bytes)?;

        Ok(Self {
            x_q8_1,
            qkv_mixed_f32,
            z_f32,
            alpha_f32,
            beta_f32,
            conv_input,
            conv_out,
            silu_out,
            q_norm_f32,
            k_norm_f32,
            state_out,
            out_normed,
            gated_f32,
            gated_q8_1,
            ssm_out_f32,
            gate_device,
            beta_device,
            x_q8_1_bytes,
            conv_channels_f32_bytes,
            d_inner_f32_bytes,
            num_v_heads_f32_bytes,
            conv_input_bytes,
            qk_f32_bytes,
            v_f32_bytes,
            gated_q8_1_bytes,
            hidden_f32_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        // SAFETY: every pointer came from `device.alloc(bytes)` above.
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.qkv_mixed_f32, self.conv_channels_f32_bytes)?;
            device.dealloc(self.z_f32, self.d_inner_f32_bytes)?;
            device.dealloc(self.alpha_f32, self.num_v_heads_f32_bytes)?;
            device.dealloc(self.beta_f32, self.num_v_heads_f32_bytes)?;
            device.dealloc(self.conv_input, self.conv_input_bytes)?;
            device.dealloc(self.conv_out, self.conv_channels_f32_bytes)?;
            device.dealloc(self.silu_out, self.conv_channels_f32_bytes)?;
            device.dealloc(self.q_norm_f32, self.qk_f32_bytes)?;
            device.dealloc(self.k_norm_f32, self.qk_f32_bytes)?;
            device.dealloc(self.state_out, self.v_f32_bytes)?;
            device.dealloc(self.out_normed, self.v_f32_bytes)?;
            device.dealloc(self.gated_f32, self.d_inner_f32_bytes)?;
            device.dealloc(self.gated_q8_1, self.gated_q8_1_bytes)?;
            device.dealloc(self.ssm_out_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.gate_device, self.num_v_heads_f32_bytes)?;
            device.dealloc(self.beta_device, self.num_v_heads_f32_bytes)?;
        }
        Ok(())
    }
}

impl Drop for GdnScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "GdnScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

impl GdnScratch {
    /// Build a by-value view shaped for
    /// `flambeau_blocks::DeltaNetLayer::forward_decode`. All fields
    /// are `Copy`, so the view passes by value.
    pub fn view(&self) -> flambeau_blocks::DeltaNetLayerDecodeScratch {
        flambeau_blocks::DeltaNetLayerDecodeScratch {
            x_q8_1: self.x_q8_1,
            qkv_mixed_f32: self.qkv_mixed_f32,
            z_f32: self.z_f32,
            alpha_f32: self.alpha_f32,
            beta_f32: self.beta_f32,
            conv_input: self.conv_input,
            conv_out: self.conv_out,
            silu_out: self.silu_out,
            q_norm_f32: self.q_norm_f32,
            k_norm_f32: self.k_norm_f32,
            state_out: self.state_out,
            out_normed: self.out_normed,
            gated_f32: self.gated_f32,
            gated_q8_1: self.gated_q8_1,
            ssm_out_f32: self.ssm_out_f32,
        }
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
pub struct GdnPrefillScratch {
    pub max_tokens: usize,
    pub x_norm_f16: DevicePtr,      // F16 [L, hidden] — 8 unfused rmsnorm sink
    pub x_q8_1: DevicePtr,
    pub x_q8_1_mmq: DevicePtr,      // DS4 layout sibling of x_q8_1 for MmqLdsX64
    pub qkv_mixed_f32: DevicePtr,   // [L, conv_channels]
    pub z_f32: DevicePtr,           // [L, d_inner]
    pub alpha_f32: DevicePtr,       // [L, num_v_heads]
    pub beta_f32: DevicePtr,        // [L, num_v_heads]
    pub conv_input: DevicePtr,      // [(conv_kernel-1) + L, conv_channels]
    pub conv_out: DevicePtr,        // [L, conv_channels]
    pub silu_out: DevicePtr,        // [L, conv_channels]
    pub q_norm_f32: DevicePtr,      // [L, num_k_heads, head_k_dim]
    pub k_norm_f32: DevicePtr,      // [L, num_k_heads, head_k_dim]
    pub v_f32: DevicePtr,           // [L, num_v_heads, head_v_dim]
    pub state_out: DevicePtr,       // [L, num_v_heads, head_v_dim]
    pub out_normed: DevicePtr,
    pub gated_f32: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub gated_q8_1_mmq: DevicePtr,  // DS4 layout sibling of gated_q8_1
    pub ssm_out_f32: DevicePtr,     // [L, hidden]
    pub gate_device: DevicePtr,     // [L, num_v_heads]
    pub beta_device: DevicePtr,     // [L, num_v_heads]
    // Bookkeeping.
    x_norm_f16_bytes: usize,
    x_q8_1_bytes: usize,
    x_q8_1_mmq_bytes: usize,
    qkv_mixed_bytes: usize,
    z_bytes: usize,
    alpha_beta_bytes: usize,
    conv_input_bytes: usize,
    conv_out_bytes: usize,
    silu_out_bytes: usize,
    qk_bytes: usize,
    v_bytes: usize,
    state_out_bytes: usize,
    out_normed_bytes: usize,
    gated_f32_bytes: usize,
    gated_q8_1_bytes: usize,
    gated_q8_1_mmq_bytes: usize,
    ssm_out_bytes: usize,
    gate_device_bytes: usize,
    disposed: bool,
}

impl GdnPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let gdn = cfg.gdn.as_ref().context("GdnPrefillScratch requires cfg.gdn")?;
        let hidden = cfg.hidden_size;
        let d_inner = gdn.d_inner;
        let num_v_heads = gdn.num_v_heads;
        let num_k_heads = gdn.num_k_heads;
        let head_k_dim = gdn.head_k_dim;
        let head_v_dim = gdn.head_v_dim();
        let conv_channels = gdn.conv_channels();
        let conv_kernel = gdn.conv_kernel;

        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(
            hidden % 128 == 0,
            "hidden must be a multiple of QK8_1_MMQ=128 for the DS4 layout"
        );
        assert!(
            d_inner % 128 == 0,
            "d_inner must be a multiple of QK8_1_MMQ=128 for the DS4 layout"
        );
        assert!(
            head_k_dim == 128 && head_v_dim == 128,
            "gdn_state_step kernel only instantiated at S_v=128"
        );

        let mmq_block = std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let x_norm_f16_bytes = max_tokens * hidden * 2;
        let x_q8_1_bytes =
            max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let x_q8_1_mmq_bytes = max_tokens * (hidden / 128) * mmq_block;
        let qkv_mixed_bytes = max_tokens * conv_channels * 4;
        let z_bytes = max_tokens * d_inner * 4;
        let alpha_beta_bytes = max_tokens * num_v_heads * 4;
        let conv_input_bytes = ((conv_kernel - 1) + max_tokens) * conv_channels * 4;
        let conv_out_bytes = max_tokens * conv_channels * 4;
        let silu_out_bytes = max_tokens * conv_channels * 4;
        let qk_bytes = max_tokens * num_k_heads * head_k_dim * 4;
        let v_bytes = max_tokens * num_v_heads * head_v_dim * 4;
        let state_out_bytes = max_tokens * num_v_heads * head_v_dim * 4;
        let out_normed_bytes = state_out_bytes;
        let gated_f32_bytes = max_tokens * d_inner * 4;
        let gated_q8_1_bytes =
            max_tokens * (d_inner / 32) * std::mem::size_of::<BlockQ8_1>();
        let gated_q8_1_mmq_bytes = max_tokens * (d_inner / 128) * mmq_block;
        let ssm_out_bytes = max_tokens * hidden * 4;
        let gate_device_bytes = max_tokens * num_v_heads * 4;

        let x_norm_f16 = device.alloc(x_norm_f16_bytes)?;
        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let x_q8_1_mmq = device.alloc(x_q8_1_mmq_bytes)?;
        let qkv_mixed_f32 = device.alloc(qkv_mixed_bytes)?;
        let z_f32 = device.alloc(z_bytes)?;
        let alpha_f32 = device.alloc(alpha_beta_bytes)?;
        let beta_f32 = device.alloc(alpha_beta_bytes)?;
        let conv_input = device.alloc(conv_input_bytes)?;
        let conv_out = device.alloc(conv_out_bytes)?;
        let silu_out = device.alloc(silu_out_bytes)?;
        let q_norm_f32 = device.alloc(qk_bytes)?;
        let k_norm_f32 = device.alloc(qk_bytes)?;
        let v_f32 = device.alloc(v_bytes)?;
        let state_out = device.alloc(state_out_bytes)?;
        let out_normed = device.alloc(out_normed_bytes)?;
        let gated_f32 = device.alloc(gated_f32_bytes)?;
        let gated_q8_1 = device.alloc(gated_q8_1_bytes)?;
        let gated_q8_1_mmq = device.alloc(gated_q8_1_mmq_bytes)?;
        let ssm_out_f32 = device.alloc(ssm_out_bytes)?;
        let gate_device = device.alloc(gate_device_bytes)?;
        let beta_device = device.alloc(gate_device_bytes)?;

        Ok(Self {
            max_tokens,
            x_norm_f16,
            x_q8_1,
            x_q8_1_mmq,
            qkv_mixed_f32,
            z_f32,
            alpha_f32,
            beta_f32,
            conv_input,
            conv_out,
            silu_out,
            q_norm_f32,
            k_norm_f32,
            v_f32,
            state_out,
            out_normed,
            gated_f32,
            gated_q8_1,
            gated_q8_1_mmq,
            ssm_out_f32,
            gate_device,
            beta_device,
            x_norm_f16_bytes,
            x_q8_1_bytes,
            x_q8_1_mmq_bytes,
            qkv_mixed_bytes,
            z_bytes,
            alpha_beta_bytes,
            conv_input_bytes,
            conv_out_bytes,
            silu_out_bytes,
            qk_bytes,
            v_bytes,
            state_out_bytes,
            out_normed_bytes,
            gated_f32_bytes,
            gated_q8_1_bytes,
            gated_q8_1_mmq_bytes,
            ssm_out_bytes,
            gate_device_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        // SAFETY: every pointer came from `device.alloc(bytes)` above.
        unsafe {
            device.dealloc(self.x_norm_f16, self.x_norm_f16_bytes)?;
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.x_q8_1_mmq, self.x_q8_1_mmq_bytes)?;
            device.dealloc(self.qkv_mixed_f32, self.qkv_mixed_bytes)?;
            device.dealloc(self.z_f32, self.z_bytes)?;
            device.dealloc(self.alpha_f32, self.alpha_beta_bytes)?;
            device.dealloc(self.beta_f32, self.alpha_beta_bytes)?;
            device.dealloc(self.conv_input, self.conv_input_bytes)?;
            device.dealloc(self.conv_out, self.conv_out_bytes)?;
            device.dealloc(self.silu_out, self.silu_out_bytes)?;
            device.dealloc(self.q_norm_f32, self.qk_bytes)?;
            device.dealloc(self.k_norm_f32, self.qk_bytes)?;
            device.dealloc(self.v_f32, self.v_bytes)?;
            device.dealloc(self.state_out, self.state_out_bytes)?;
            device.dealloc(self.out_normed, self.out_normed_bytes)?;
            device.dealloc(self.gated_f32, self.gated_f32_bytes)?;
            device.dealloc(self.gated_q8_1, self.gated_q8_1_bytes)?;
            device.dealloc(self.gated_q8_1_mmq, self.gated_q8_1_mmq_bytes)?;
            device.dealloc(self.ssm_out_f32, self.ssm_out_bytes)?;
            device.dealloc(self.gate_device, self.gate_device_bytes)?;
            device.dealloc(self.beta_device, self.gate_device_bytes)?;
        }
        Ok(())
    }
}

impl Drop for GdnPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "GdnPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
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
/// The GDN state-step kernel is already L-aware — it keeps the per-head
/// `[S_v, S_v]` state register-resident across the entire L recurrence
/// loop in a single launch. The α / β / gate compute is currently a L-wide
/// loop over the single-token kernel; future fusion into a proper batched
/// variant saves O(L) launches but is not required for correctness.
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
    if n_tokens == 0 {
        bail!("forward_gdn_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_gdn_prefill: n_tokens={n_tokens} > scratch.max_tokens={}; caller must chunk",
            scratch.max_tokens
        );
    }

    let gdn = cfg.gdn.as_ref().context("forward_gdn_prefill requires cfg.gdn")?;
    let hidden = cfg.hidden_size;
    let d_inner = gdn.d_inner;
    let num_v_heads = gdn.num_v_heads;
    let num_k_heads = gdn.num_k_heads;
    let head_k_dim = gdn.head_k_dim;
    let head_v_dim = gdn.head_v_dim();
    let conv_channels = gdn.conv_channels();
    let conv_kernel = gdn.conv_kernel;
    let qk_size = num_k_heads * head_k_dim;
    let v_size = num_v_heads * head_v_dim;

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

    // intra-prefill section markers, mirror of the L=1
    // decode marks defined in forward_gdn_decode (gdn_start, gdn_norm_quant,
    // gdn_proj_qkv_gate, gdn_proj_alpha_beta, gdn_conv1d, gdn_silu,
    // gdn_l2norm_qk, gdn_state_step, gdn_ssm_norm, gdn_swiglu_quant,
    // gdn_ssm_out, gdn_cast_f16). All names suffixed `_p` to keep
    // L=2-prefill marks separable from L=1 decode marks in the same
    // profile run (e.g. when spec-decode hits a reject and the L=1
    // redo path also fires gdn_*).
    flambeau_backend_hip::profile::mark("gdn_start_p", device, stream)?;

    // 1. rmsnorm(x_in) → F16 scratch, then quantise to BOTH Q8_1 layouts.
    // 8 de-fuses the old rmsnorm_quant_q8_1 so we can emit the
    // DS4 (BlockQ8_1Mmq) layout consumed by the new Q4_1 MMQ kernel at
    // m ≥ 128 alongside the standard per-row layout for MMVQ.
    rmsnorm_f16(
        ops,
        stream,
        x_in,
        attn_norm.ptr,
        scratch.x_norm_f16,
        n_tokens,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("gdn prefill attn_norm")?;
    quantize_f16_q8_1(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden,
    )
    .context("gdn prefill x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens,
    )
    .context("gdn prefill x_norm → Q8_1 (MMQ DS4)")?;
    flambeau_backend_hip::profile::mark("gdn_norm_quant_p", device, stream)?;

    // 2..5. Hidden-input projections at M = L. attn_qkv / attn_gate are Q4_1
    // on Qwen3.5-9B → route to MmqLdsX64 at m ≥ 128. ssm_alpha / ssm_beta are
    // Q5_K / other → never route to MmqLdsX64 (dispatch has no row). The mmq
    // buffer is passed to all four; dispatch picks per-weight-dtype.
    run_qmatmul_from_tensor(
        ops,
        stream,
        &weights.attn_qkv,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.qkv_mixed_f32,
        n_tokens,
        hidden,
        conv_channels,
        "attn_qkv",
    )?;
    run_qmatmul_from_tensor(
        ops,
        stream,
        &weights.attn_gate,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.z_f32,
        n_tokens,
        hidden,
        d_inner,
        "attn_gate",
    )?;
    flambeau_backend_hip::profile::mark("gdn_proj_qkv_gate_p", device, stream)?;
    run_qmatmul_from_tensor(
        ops,
        stream,
        ssm_alpha,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.alpha_f32,
        n_tokens,
        hidden,
        num_v_heads,
        "ssm_alpha",
    )?;
    run_qmatmul_from_tensor(
        ops,
        stream,
        ssm_beta,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.beta_f32,
        n_tokens,
        hidden,
        num_v_heads,
        "ssm_beta",
    )?;
    flambeau_backend_hip::profile::mark("gdn_proj_alpha_beta_p", device, stream)?;

    // 6. Conv1d across L tokens: assemble [history, qkv_mixed] → conv_input,
    // run conv, then shift history to the last (K-1) rows.
    assemble_conv_input_prefill(
        device,
        stream,
        layer_state.conv_history,
        scratch.qkv_mixed_f32,
        scratch.conv_input,
        n_tokens,
        conv_channels,
        conv_kernel,
    )?;
    causal_conv1d_f32(
        ops,
        stream,
        scratch.conv_input,
        weights.ssm_conv1d.ptr,
        scratch.conv_out,
        n_tokens,
        conv_channels,
        conv_kernel,
    )
    .context("prefill causal_conv1d_f32")?;
    shift_conv_history_prefill(
        device,
        stream,
        scratch.conv_input,
        layer_state.conv_history,
        n_tokens,
        conv_channels,
        conv_kernel,
    )?;
    flambeau_backend_hip::profile::mark("gdn_conv1d_p", device, stream)?;

    // 7. silu(conv_out) → silu_out (F32 [L, conv_channels]).
    silu_f32(
        ops,
        stream,
        scratch.conv_out,
        scratch.silu_out,
        n_tokens * conv_channels,
    )
    .context("prefill silu_f32(conv_out)")?;
    flambeau_backend_hip::profile::mark("gdn_silu_p", device, stream)?;

    // 8. Split silu_out into Q / K / V contiguous buffers. fused
    // `gdn_split_qkv_f32` kernel replaces the 3×L memcpy loop (~1500
    // driver calls per layer at L=512). The unfused fallback was kept
    // as `FLAMBEAU_QKV_FUSED=0` regression A/B; deleted in S6 — null
    // on every anchor cell.
    gdn_split_qkv_f32(
        ops,
        stream,
        scratch.silu_out,
        scratch.q_norm_f32,
        scratch.k_norm_f32,
        scratch.v_f32,
        n_tokens,
        qk_size,
        v_size,
    )
    .context("prefill gdn_split_qkv_f32")?;

    // 9. L2-normalise Q and K per head (row = head, k = head_k_dim).
    l2_norm_f32(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.q_norm_f32,
        n_tokens * num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("prefill l2_norm Q")?;
    l2_norm_f32(
        ops,
        stream,
        scratch.k_norm_f32,
        scratch.k_norm_f32,
        n_tokens * num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("prefill l2_norm K")?;

    // 10. Scale Q by 1/sqrt(head_k_dim) — in place across all L.
    let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
    scale_f32(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.q_norm_f32,
        n_tokens * qk_size,
        q_scale,
    )
    .context("prefill scale_f32 Q")?;
    flambeau_backend_hip::profile::mark("gdn_l2norm_qk_p", device, stream)?;

    // 11–12. C10 fused state-step (default) absorbs α/β/gate; baseline
    // chain available via FLAMBEAU_VARIANT=baseline. State-step
    // event-ordering preserved across both branches (0.a).
    // /14 — q/k repeat layout differs by arch; see decode-path
    // comment in `forward_gdn_decode` for the explanation.
    let n_rep = num_v_heads / num_k_heads;
    let rep_inner_layout = cfg.arch == "qwen3next";
    if let Some(ev) = state_event {
        ev.stream_wait(stream)
            .context("gdn state_step stream_wait")?;
    }
    gdn_state_step_alphabeta_f32_s128(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.k_norm_f32,
        scratch.v_f32,
        scratch.alpha_f32,
        scratch.beta_f32,
        weights.ssm_dt_bias.ptr,
        weights.ssm_a.ptr,
        layer_state.state,
        layer_state.state,
        scratch.state_out,
        1,
        num_v_heads,
        n_tokens,
        n_rep,
        rep_inner_layout,
    )
    .context("prefill gdn_state_step_alphabeta_f32_s128 (C10 fused)")?;
    if let Some(ev) = state_event {
        ev.record(stream).context("gdn state_step record")?;
    }
    flambeau_backend_hip::profile::mark("gdn_state_step_p", device, stream)?;

    // 13. ssm_norm per-head over L × num_v_heads rows.
    let ssm_norm_k = weights
        .ssm_norm
        .dims
        .first()
        .copied()
        .context("ssm_norm missing dim")? as usize;
    if ssm_norm_k != head_v_dim {
        bail!("ssm_norm dim {ssm_norm_k} != head_v_dim {head_v_dim}");
    }
    rmsnorm_f32(
        ops,
        stream,
        scratch.state_out,
        weights.ssm_norm.ptr,
        scratch.out_normed,
        n_tokens * num_v_heads,
        head_v_dim,
        cfg.rms_norm_eps,
    )
    .context("prefill ssm_norm (rmsnorm_f32)")?;
    flambeau_backend_hip::profile::mark("gdn_ssm_norm_p", device, stream)?;

    // 14. Gated: `gated = silu(z) * out_normed` across [L, d_inner].
    if v_size != d_inner {
        bail!(
            "GDN layout: num_v_heads * head_v_dim ({v_size}) != d_inner ({d_inner})"
        );
    }
    swiglu_f32(
        ops,
        stream,
        scratch.z_f32,
        scratch.out_normed,
        scratch.gated_f32,
        n_tokens * d_inner,
    )
    .context("prefill swiglu_f32(z, out_normed)")?;

    // 15. Quantise gated → both Q8_1 layouts for the ssm_out matmul.
    quantize_q8_1(
        ops,
        stream,
        scratch.gated_f32,
        scratch.gated_q8_1,
        n_tokens * d_inner,
    )
    .context("prefill quantise gated → Q8_1 (std)")?;
    quantize_q8_1_mmq(
        ops,
        stream,
        scratch.gated_f32,
        scratch.gated_q8_1_mmq,
        d_inner,
        n_tokens,
    )
    .context("prefill quantise gated → Q8_1 (MMQ DS4)")?;
    flambeau_backend_hip::profile::mark("gdn_swiglu_quant_p", device, stream)?;

    // 16. ssm_out projection at M = L. ssm_out is typically Q5_K / Q8_0 on
    // V1 models and does not route to MmqLdsX64, but we pass the mmq buffer
    // so dispatch has it available if a Q4_1 ssm_out lands in some future
    // GGUF dtype mix.
    run_qmatmul_from_tensor(
        ops,
        stream,
        &weights.ssm_out,
        scratch.gated_q8_1,
        scratch.gated_q8_1_mmq,
        scratch.ssm_out_f32,
        n_tokens,
        d_inner,
        hidden,
        "ssm_out",
    )?;
    flambeau_backend_hip::profile::mark("gdn_ssm_out_p", device, stream)?;

    // 17. Cast back to F16 for the outer residual path.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.ssm_out_f32,
        delta_out,
        n_tokens * hidden,
    )
    .context("prefill cast ssm_out → f16")?;
    flambeau_backend_hip::profile::mark("gdn_cast_f16_p", device, stream)?;

    Ok(())
}

// `run_qmatmul_from_tensor` moved to `forward::common`.

