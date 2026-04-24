//! Per-layer forward composition — the dispatcher that chains attn/gdn +
//! (moe | dense_ffn) into one layer's output, for both decode and prefill.
//!
//! Everything this module produces is already-residual-summed `x_out = x_in
//! + attn_delta + ffn_out(residual = mid + shared_delta)` matching
//! `moe_combine_f16`'s semantics (no double-residual).

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition — every unsafe block is a kernel.launch or \
              memcpy_async over DevicePtrs owned by the session's scratch / weights / \
              KV cache. Buffers live for the whole session; sync is driven by the top- \
              level forward_*_decode/prefill caller."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::{Device, DevicePtr};
use flambeau_ops::hip::{
    mlp::add_f16,
    norm::{rmsnorm_f16, rmsnorm_f16_add_residual},
    HipDevice, HipStream, OpsRegistry,
};

use super::{
    DenseFfnPrefillScratch, DenseFfnScratch, FullAttnPrefillScratch, FullAttnScratch,
    GdnPrefillScratch, GdnScratch, MoePrefillScratch, MoeScratch, SharedExpertPrefillScratch,
    SharedExpertScratch,
};
use super::dense_ffn::{forward_dense_ffn_decode, forward_dense_ffn_prefill};
use super::attn::{forward_full_attn_layer_decode, forward_full_attn_prefill};
use super::gdn::{forward_gdn_layer_decode, forward_gdn_prefill};
use super::moe::{
    forward_moe_ffn_decode, forward_moe_ffn_prefill, forward_router_decode,
    forward_router_prefill, forward_shared_expert_decode, forward_shared_expert_prefill,
};
use crate::config::Qwen3MoEConfig;
use crate::session::LayerCache;

// ---------------------------------------------------------------------------
// V1.7.3-e4 — per-layer composition (residual sums + ffn/post-attn norm +
// routed/shared fan-in).
// ---------------------------------------------------------------------------

/// Every scratch the per-layer composition function touches. Owning them
/// in one struct keeps the `forward_layer_decode` signature readable and
/// lets the session allocate exactly once.
///
/// We always carry a `FullAttnScratch` and a `GdnScratch` even though
/// each layer uses only one; the unused one sits idle. The `SharedExpertScratch`
/// is `Option` because dense qwen3moe arches have no shared expert.
pub struct LayerForwardScratch {
    pub full_attn: Option<FullAttnScratch>,
    pub gdn: Option<GdnScratch>,
    pub moe: Option<MoeScratch>,
    pub shared: Option<SharedExpertScratch>,
    /// Present iff `cfg.is_dense_ffn()` (arch=qwen35); replaces the moe/shared
    /// scratches on that path.
    pub dense_ffn: Option<DenseFfnScratch>,
    /// F16 `[hidden]` — holds the post-attention residual (`x_in + attn_delta`).
    pub mid_f16: DevicePtr,
    /// F16 `[hidden]` — holds rmsnorm(mid, post_attn_norm_or_ffn_norm).
    pub mid_norm_f16: DevicePtr,
    /// F16 `[hidden]` — holds the shared-expert delta (only populated on
    /// hybrid layers with a shared expert).
    pub shared_delta_f16: DevicePtr,
    /// F16 `[hidden]` — holds `mid + shared_delta` fed into moe_combine
    /// as the residual.
    pub moe_residual_f16: DevicePtr,
    hidden_bytes: usize,
    disposed: bool,
}

impl LayerForwardScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let hidden_bytes = hidden * 2;

        let full_attn = Some(FullAttnScratch::new(cfg, device)?);
        let gdn = Some(GdnScratch::new(cfg, device)?);
        // Dense-FFN arches (qwen35) skip the MoE router + shared expert
        // scratch entirely. Allocate dense scratch in its place.
        let (moe, shared, dense_ffn) = if cfg.is_dense_ffn() {
            (None, None, Some(DenseFfnScratch::new(cfg, device)?))
        } else {
            let moe = Some(MoeScratch::new(cfg, device)?);
            let shared = if cfg.shared_expert_intermediate_size.is_some() {
                Some(SharedExpertScratch::new(cfg, device)?)
            } else {
                None
            };
            (moe, shared, None)
        };

        let mid_f16 = device.alloc(hidden_bytes)?;
        let mid_norm_f16 = device.alloc(hidden_bytes)?;
        let shared_delta_f16 = device.alloc(hidden_bytes)?;
        let moe_residual_f16 = device.alloc(hidden_bytes)?;

        Ok(Self {
            full_attn,
            gdn,
            moe,
            shared,
            dense_ffn,
            mid_f16,
            mid_norm_f16,
            shared_delta_f16,
            moe_residual_f16,
            hidden_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.mid_f16, self.hidden_bytes)?;
            device.dealloc(self.mid_norm_f16, self.hidden_bytes)?;
            device.dealloc(self.shared_delta_f16, self.hidden_bytes)?;
            device.dealloc(self.moe_residual_f16, self.hidden_bytes)?;
        }
        if let Some(s) = self.full_attn.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.gdn.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.moe.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.shared.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.dense_ffn.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for LayerForwardScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "LayerForwardScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One decode step of a full layer block — dispatches on `cfg.is_recurrent(il)`
/// to the GDN or full-attention path, then runs the MoE FFN (with optional
/// shared expert) and composes all the residual sums.
///
/// Math, per layer:
///   attn_delta = attn(attn_norm(x_in))                 (forward_{full_attn,gdn}_decode)
///   mid        = x_in + attn_delta                     (add_f16)
///   mid_norm   = post_attn_norm(mid)                   (rmsnorm_f16)
///   shared     = shared_expert(mid_norm)               (optional; forward_shared_expert_decode)
///   moe_res    = mid + shared                          (add_f16, if shared)
///   x_out      = moe_res + Σ w_k · expert_k(mid_norm)  (forward_moe_ffn_decode)
///
/// On arches without a shared expert (dense qwen3moe), the `shared` and
/// `moe_res` steps are skipped and `moe_res = mid` is passed directly.
/// V2.27.a-i3 — optional slot bundle for graph-captureable decode.
/// Only full-attn layers contribute slots; GDN layers advance state
/// in-place across replays and need no slot updates.
#[derive(Clone, Copy, Debug)]
pub struct LayerDecodeSlots {
    pub full_attn: super::attn::AttnDecodeSlots,
}

pub fn forward_layer_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut LayerForwardScratch,
    x_in: DevicePtr,
    x_out: DevicePtr,
    position: usize,
    slots: Option<LayerDecodeSlots>,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let il = layer_weights.layer_idx;

    // 1. Attention (full-attn or GDN) → attn_delta in `mid_f16`.
    // We reuse mid_f16 as the delta slot first, then overwrite it with the
    // post-attention residual sum on the next line.
    if cfg.is_recurrent(il) {
        let gdn = scratch
            .gdn
            .as_mut()
            .context("LayerForwardScratch.gdn missing")?;
        forward_gdn_layer_decode(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            layer_cache,
            gdn,
            x_in,
            scratch.mid_f16,
        )?;
    } else {
        let full_attn = scratch
            .full_attn
            .as_mut()
            .context("LayerForwardScratch.full_attn missing")?;
        forward_full_attn_layer_decode(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            layer_cache,
            full_attn,
            x_in,
            scratch.mid_f16,
            position,
            slots.map(|s| s.full_attn),
        )?;
    }

    // 2+3. V2.23.a.1 fused: `mid = x_in + attn_delta; mid_norm = rmsnorm(mid)*w`.
    // Saves one kernel launch per layer per token vs the old add_f16 + rmsnorm
    // pair. Both outputs consumed downstream.
    let post_norm = layer_weights
        .post_attention_norm
        .as_ref()
        .or(layer_weights.ffn_norm.as_ref())
        .context("layer missing both post_attention_norm and ffn_norm")?;
    rmsnorm_f16_add_residual(
        ops,
        stream,
        x_in,
        scratch.mid_f16,
        post_norm.ptr,
        scratch.mid_f16,
        scratch.mid_norm_f16,
        1,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("fused post-attn add+rmsnorm")?;

    // 4. FFN. Two flavours:
    //    - arch=qwen35 (dense): single gate/up/down triple, no router. Writes
    //      `x_out = mid + FFN(mid_norm)` directly.
    //    - MoE arches: optional shared expert delta + router + routed MoE
    //      (residual folded into moe_combine).
    if cfg.is_dense_ffn() {
        let dense_w = layer_weights
            .ffn
            .dense
            .as_ref()
            .context("dense FFN forward: layer.ffn.dense missing")?;
        let dense_scratch = scratch
            .dense_ffn
            .as_mut()
            .context("LayerForwardScratch.dense_ffn missing")?;
        forward_dense_ffn_decode(
            ops,
            stream,
            cfg,
            dense_w,
            dense_scratch,
            scratch.mid_norm_f16,
            scratch.mid_f16,
            x_out,
        )?;
        return Ok(());
    }

    // MoE path — optional shared expert delta. V2.23.a.2 skips the explicit
    // `add_f16(mid, shared_delta)` by passing both residuals to
    // `moe_combine_two_residuals_f16`, saving one launch per layer per token.
    let (moe_residual, shared_extra) = if let (Some(shared_w), Some(shared_scratch)) =
        (layer_weights.ffn.shared.as_ref(), scratch.shared.as_mut())
    {
        forward_shared_expert_decode(
            ops,
            stream,
            cfg,
            shared_w,
            shared_scratch,
            scratch.mid_norm_f16,
            scratch.shared_delta_f16,
        )?;
        (scratch.mid_f16, Some(scratch.shared_delta_f16))
    } else {
        // Dense arch with no shared expert: combine's residual is just mid.
        (scratch.mid_f16, None)
    };

    // 5. Router (dense F32 GEMV + topk).
    let moe = scratch
        .moe
        .as_mut()
        .context("LayerForwardScratch.moe missing")?;
    let ffn_gate_inp = layer_weights
        .ffn
        .ffn_gate_inp
        .as_ref()
        .context("forward_layer_decode MoE branch: ffn.ffn_gate_inp missing")?;
    forward_router_decode(
        ops,
        stream,
        cfg,
        ffn_gate_inp,
        moe,
        scratch.mid_norm_f16,
    )?;

    // 6. Routed MoE FFN — fuses the residual add in moe_combine (and optionally
    // the shared-expert delta residual via V2.23.a.2 two-residuals variant).
    forward_moe_ffn_decode(
        ops,
        stream,
        cfg,
        &layer_weights.ffn,
        moe,
        scratch.mid_norm_f16,
        moe_residual,
        shared_extra,
        x_out,
    )?;

    Ok(())
}


// ---------------------------------------------------------------------------
// V1.7.3-f4 — per-layer prefill + forward_prefill end-to-end.
// ---------------------------------------------------------------------------

/// Scratches a per-layer prefill step touches: sibling of
/// `LayerForwardScratch` sized against `(cfg, max_tokens)`.
pub struct LayerPrefillScratch {
    pub max_tokens: usize,
    pub full_attn: Option<FullAttnPrefillScratch>,
    pub gdn: Option<GdnPrefillScratch>,
    pub moe: Option<MoePrefillScratch>,
    pub shared: Option<SharedExpertPrefillScratch>,
    /// Present iff `cfg.is_dense_ffn()`.
    pub dense_ffn: Option<DenseFfnPrefillScratch>,
    pub mid_f16: DevicePtr,           // F16 [L, hidden] — post-attn residual
    pub mid_norm_f16: DevicePtr,      // F16 [L, hidden] — rmsnorm(mid)
    pub shared_delta_f16: DevicePtr,  // F16 [L, hidden]
    pub moe_residual_f16: DevicePtr,  // F16 [L, hidden] — mid + shared
    hidden_bytes: usize,
    disposed: bool,
}

impl LayerPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1);
        let hidden_bytes = max_tokens * cfg.hidden_size * 2;

        let full_attn = Some(FullAttnPrefillScratch::new(cfg, device, max_tokens)?);
        let gdn = Some(GdnPrefillScratch::new(cfg, device, max_tokens)?);
        let (moe, shared, dense_ffn) = if cfg.is_dense_ffn() {
            (None, None, Some(DenseFfnPrefillScratch::new(cfg, device, max_tokens)?))
        } else {
            let moe = Some(MoePrefillScratch::new(cfg, device, max_tokens)?);
            let shared = if cfg.shared_expert_intermediate_size.is_some() {
                Some(SharedExpertPrefillScratch::new(cfg, device, max_tokens)?)
            } else {
                None
            };
            (moe, shared, None)
        };

        let mid_f16 = device.alloc(hidden_bytes)?;
        let mid_norm_f16 = device.alloc(hidden_bytes)?;
        let shared_delta_f16 = device.alloc(hidden_bytes)?;
        let moe_residual_f16 = device.alloc(hidden_bytes)?;

        Ok(Self {
            max_tokens,
            full_attn,
            gdn,
            moe,
            shared,
            dense_ffn,
            mid_f16,
            mid_norm_f16,
            shared_delta_f16,
            moe_residual_f16,
            hidden_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.mid_f16, self.hidden_bytes)?;
            device.dealloc(self.mid_norm_f16, self.hidden_bytes)?;
            device.dealloc(self.shared_delta_f16, self.hidden_bytes)?;
            device.dealloc(self.moe_residual_f16, self.hidden_bytes)?;
        }
        if let Some(s) = self.full_attn.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.gdn.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.moe.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.shared.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.dense_ffn.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for LayerPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "LayerPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One prefill chunk through one full layer. Mirrors
/// V2.26.a-i5b — optional slot bundle for graph-capture. Holds the
/// per-layer slots that need updating per ubatch (pos-varying scalars
/// and KV-append dsts). Only the full-attn layer contributes slots in
/// the V1 target (qwen35 dense, qwen35moe); GDN + dense FFN are
/// pos-independent from a kernel-arg perspective.
#[derive(Clone, Copy, Debug)]
pub struct LayerPrefillSlots {
    pub full_attn: super::attn::AttnPrefillSlots,
}

/// `forward_layer_decode`'s math but over `[L, hidden]` tensors.
pub fn forward_layer_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut LayerPrefillScratch,
    x_in: DevicePtr,
    x_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
    slots: Option<LayerPrefillSlots>,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let il = layer_weights.layer_idx;

    // 1. Attention (full-attn or GDN) → mid_f16 (attn delta).
    if cfg.is_recurrent(il) {
        let LayerCache::Gdn(state) = layer_cache else {
            bail!("layer {il} expected GDN cache");
        };
        let gdn = scratch.gdn.as_mut().context("LayerPrefillScratch.gdn missing")?;
        let crate::weights::AttnWeights::Gdn(g) = &layer_weights.attn else {
            bail!("layer {il} expected GDN weights");
        };
        forward_gdn_prefill(
            ops,
            stream,
            device,
            cfg,
            &layer_weights.attn_norm,
            g,
            state,
            gdn,
            x_in,
            scratch.mid_f16,
            n_tokens,
        )?;
    } else {
        let LayerCache::FullAttn(kv) = layer_cache else {
            bail!("layer {il} expected FullAttn cache");
        };
        let full_attn = scratch
            .full_attn
            .as_mut()
            .context("LayerPrefillScratch.full_attn missing")?;
        let crate::weights::AttnWeights::FullAttn(fa) = &layer_weights.attn else {
            bail!("layer {il} expected FullAttn weights");
        };
        forward_full_attn_prefill(
            ops,
            stream,
            device,
            cfg,
            &layer_weights.attn_norm,
            fa,
            kv,
            full_attn,
            x_in,
            scratch.mid_f16,
            n_tokens,
            start_position,
            slots.map(|s| s.full_attn),
        )?;
    }

    // 2. Residual: mid = x_in + attn_delta (in-place on mid_f16).
    add_f16(
        ops,
        stream,
        x_in,
        scratch.mid_f16,
        scratch.mid_f16,
        n_tokens * hidden,
    )
    .context("prefill layer residual: x_in + attn_delta")?;

    // 3. post-attention / ffn norm.
    let post_norm = layer_weights
        .post_attention_norm
        .as_ref()
        .or(layer_weights.ffn_norm.as_ref())
        .context("layer missing both post_attention_norm and ffn_norm")?;
    rmsnorm_f16(
        ops,
        stream,
        scratch.mid_f16,
        post_norm.ptr,
        scratch.mid_norm_f16,
        n_tokens,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("prefill post-attn rmsnorm")?;

    // 4. FFN. Dense (qwen35) or MoE + optional shared expert.
    if cfg.is_dense_ffn() {
        let dense_w = layer_weights
            .ffn
            .dense
            .as_ref()
            .context("dense FFN prefill: layer.ffn.dense missing")?;
        let dense_scratch = scratch
            .dense_ffn
            .as_mut()
            .context("LayerPrefillScratch.dense_ffn missing")?;
        forward_dense_ffn_prefill(
            ops,
            stream,
            cfg,
            dense_w,
            dense_scratch,
            scratch.mid_norm_f16,
            scratch.mid_f16,
            x_out,
            n_tokens,
        )?;
        return Ok(());
    }

    // MoE path.
    let moe_residual = if let (Some(shared_w), Some(shared_scratch)) =
        (layer_weights.ffn.shared.as_ref(), scratch.shared.as_mut())
    {
        forward_shared_expert_prefill(
            ops,
            stream,
            cfg,
            shared_w,
            shared_scratch,
            scratch.mid_norm_f16,
            scratch.shared_delta_f16,
            n_tokens,
        )?;
        add_f16(
            ops,
            stream,
            scratch.mid_f16,
            scratch.shared_delta_f16,
            scratch.moe_residual_f16,
            n_tokens * hidden,
        )
        .context("prefill moe residual: mid + shared_delta")?;
        scratch.moe_residual_f16
    } else {
        scratch.mid_f16
    };

    // 5. Router (dense F32 GEMV × L + topk).
    let moe = scratch
        .moe
        .as_mut()
        .context("LayerPrefillScratch.moe missing")?;
    let ffn_gate_inp = layer_weights
        .ffn
        .ffn_gate_inp
        .as_ref()
        .context("forward_layer_prefill MoE branch: ffn.ffn_gate_inp missing")?;
    forward_router_prefill(
        ops,
        stream,
        cfg,
        ffn_gate_inp,
        moe,
        scratch.mid_norm_f16,
        n_tokens,
    )?;

    // 6. Routed MoE FFN — fuses the residual add in moe_combine.
    forward_moe_ffn_prefill(
        ops,
        stream,
        cfg,
        &layer_weights.ffn,
        moe,
        scratch.mid_norm_f16,
        moe_residual,
        x_out,
        n_tokens,
    )?;

    Ok(())
}
