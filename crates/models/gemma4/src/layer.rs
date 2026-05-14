//! Single-layer forward composition for Gemma 4 (decode path).
//!
//! Pipeline mirrors `/artefact/llama.cpp/src/models/gemma4-iswa.cpp`
//! lines 40–200, minus the per-layer side-channel embedding (S5-B)
//! and the MoE branch (S6):
//!
//! 1. RMSNorm `attn_norm`, Q8_1 quantise.
//! 2. Q projection (`wq`) → F16, per-head learned RMSNorm with
//!    `attn_q_norm`, partial NeoX RoPE.
//! 3. K projection (`wk`) → F16, per-head learned RMSNorm with
//!    `attn_k_norm`, partial NeoX RoPE.
//! 4. V projection: if `wv` present, `wv` × x; else V = K **pre-norm**
//!    (the "alternative attention" path in
//!    `gemma4-iswa.cpp:83-86`). V then gets an **unlearned** RMSNorm
//!    (no weight tensor — pass a unit-weight buffer). V does **not**
//!    get RoPE.
//! 5. KV-cache append.
//! 6. Attention decode (with `window_size = spec.window`,
//!    `softmax_scale = 1.0` per gemma4's `f_attention_scale`).
//! 7. Output projection (`wo`).
//! 8. RMSNorm `post_attention_norm`.
//! 9. Residual add `attn_residual = x_in + post_attn_norm(attn_out)`.
//! 10. RMSNorm `ffn_norm`.
//! 11. Dense FFN: `down(GELU(gate) * up)` using `gelu_f32_to_f16`.
//! 12. RMSNorm `post_ffw_norm`.
//! 13. Residual add `x_out = attn_residual + post_ffw_norm(...)`.
//! 14. (Optional) per-layer scalar `layer_output_scale`.
//!
//! Limitations of S5-A (now lifted in S6-A — shared-KV tail supported):
//! - `ffn_kind == Moe` bails with a clear error (S6-B).
//! - Per-layer side-channel embedding bails with a clear error (S5-B-2).
//!
//! Shared-KV tail (S6-A): when `spec.has_kv == false`, the K/V
//! projections, per-head K/V norms, K-side RoPE, and KV-cache append
//! are all skipped. The caller routes `kv_cache` to the source
//! layer's cache (`spec.kv_share_src`); the per-layer Q projection
//! still runs against this layer's `attn_q` weight, then attention
//! reads from the routed cache. The tail layer must NOT mutate the
//! source cache (no append) — the `&mut` borrow is structural only.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_blocks::WeightHandle;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::Ops;
use flambeau_runtime::{CacheLayout, F16Contig, KvCache, Q8Contig};

use crate::layout::{FfnKind, LayerSpec};
use crate::scratch::{LayerDecodeScratch, LayerPrefillScratch};

/// Device-resident weights for one Gemma 4 layer (attention + dense
/// FFN). Optional fields (`attn_k`, `attn_v`, `attn_k_norm`,
/// `layer_output_scale`) honour the per-layer presence rules from
/// `weights.rs::AttnTensors`.
pub struct Gemma4LayerWeights {
    pub attn_norm: DevicePtr,
    pub attn_q: WeightHandle,
    pub attn_k: Option<WeightHandle>,
    pub attn_v: Option<WeightHandle>,
    pub attn_output: WeightHandle,
    pub attn_q_norm: DevicePtr,
    pub attn_k_norm: Option<DevicePtr>,
    pub post_attention_norm: DevicePtr,
    /// Optional per-layer scalar (gemma4 `layer_output_scale`, F32 [1]).
    /// Downloaded at upload time so we can apply it via `scale_f16`
    /// without an extra device-side broadcast op.
    pub layer_output_scale: Option<f32>,

    pub ffn_norm: DevicePtr,
    pub ffn_gate: WeightHandle,
    pub ffn_up: WeightHandle,
    pub ffn_down: WeightHandle,
    pub post_ffw_norm: DevicePtr,

    /// Per-layer side-channel embed weights (E2B / E4B only).
    pub per_layer_embed: Option<crate::per_layer_embd::PerLayerEmbedLayerWeights>,

    /// MoE-specific weights (26B-A4B only). `Some` when
    /// `spec.ffn_kind == Moe`; the dense `ffn_gate` / `ffn_up` /
    /// `ffn_down` fields above are then **also** present (they serve
    /// as the parallel shared-MLP branch of the MoE composer — see
    /// `gemma4/src/moe.rs::forward_ffn_moe`).
    pub moe: Option<crate::moe::Gemma4MoeFfnWeights>,
}

/// One decode step through a single Gemma 4 layer.
///
/// Writes one new token's attention output + FFN to `x_out`, modifying
/// `kv_cache` in place (one new row appended). The caller manages the
/// outer residual stream and the per-layer scratch buffers.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_decode<L: CacheLayout, O: Ops>(
    ops: &O,
    device: &HipDevice,
    stream: &HipStream,
    weights: &Gemma4LayerWeights,
    spec: &LayerSpec,
    rms_norm_eps: f32,
    ff_len: usize,
    hidden: usize,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut LayerDecodeScratch<'_>,
    x_in: DevicePtr,
    x_out: DevicePtr,
    position: usize,
    // Per-layer side-channel embed slice (F32 [pe]) for this layer.
    // `Some(ptr, pe)` iff the model has per-layer-embd AND
    // `weights.per_layer_embed` is `Some`.
    per_layer_slice: Option<(DevicePtr, usize)>,
    // MoE composer scratch — required when `spec.ffn_kind == Moe`
    // (caller pre-allocates one scratch sized for the widest MoE
    // layer's `n_embd_per_layer`/`n_experts`/`top_k`); `None` on
    // dense layers.
    moe_scratch: Option<&crate::moe::Gemma4MoeScratch>,
) -> Result<()> {
    if spec.ffn_kind != FfnKind::Dense && spec.ffn_kind != FfnKind::Moe {
        bail!(
            "forward_layer_decode: ffn_kind {:?} not supported",
            spec.ffn_kind
        );
    }
    if spec.ffn_kind == FfnKind::Moe {
        // Pre-validate so we fail fast rather than midway through attn.
        if weights.moe.is_none() {
            bail!(
                "forward_layer_decode: layer {} is MoE but weights.moe is None — \
                 upload path must populate Gemma4MoeFfnWeights",
                spec.index
            );
        }
        if moe_scratch.is_none() {
            bail!(
                "forward_layer_decode: layer {} is MoE but moe_scratch is None — \
                 caller must pass a Gemma4MoeScratch (session-allocated)",
                spec.index
            );
        }
    }
    // Shared-KV tail layers (`has_kv == false`) skip K/V projection +
    // K/V norm + K-side RoPE + KV append; they query the routed cache
    // directly. Validate that the layer-load skipped attn_k/attn_k_norm
    // accordingly.
    let (attn_k, attn_k_norm_w) = if spec.has_kv {
        let k = weights.attn_k.as_ref().ok_or_else(|| {
            anyhow::anyhow!("layer {}: attn_k missing on a has_kv layer", spec.index)
        })?;
        let kn = weights
            .attn_k_norm
            .ok_or_else(|| anyhow::anyhow!("layer {}: attn_k_norm missing", spec.index))?;
        (Some(k), Some(kn))
    } else {
        (None, None)
    };

    let head_dim = spec.head_dim;
    let n_heads = spec.n_heads;
    let n_kv_heads = spec.n_kv_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let window: i32 = spec.window as i32;
    // Gemma 4 sets `f_attention_scale = 1.0f` (no pre-attention
    // scaling). See llama.cpp `model.cpp:1638`.
    let softmax_scale: f32 = 1.0;

    // 1. RMSNorm(x_in, attn_norm) + Q8_1 quant in one launch.
    ops.rmsnorm_quant_q8_1(
        x_in,
        weights.attn_norm,
        scratch.x_q8_1,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("attn_norm + quant")?;

    // 2. Q projection. Plain (non-gated) — output is q_width rows.
    ops.mmvq(
        weights.attn_q.ptr,
        scratch.x_q8_1,
        scratch.mmvq_f32,
        q_width,
        hidden,
        weights.attn_q.dtype,
    )
    .context("mmvq attn_q")?;
    ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.q_f16, q_width)
        .context("cast Q → f16")?;

    // 3+4. K, V projections + norms + K-side RoPE + KV append.
    // Skipped on shared-KV tail layers (kv_cache is routed to the
    // source layer's cache by the caller; that layer's earlier
    // iteration wrote K/V).
    let kv_layout = L::NAME;
    if kv_layout == Q8Contig::NAME {
        bail!("forward_layer_decode: Q8 KV layout not supported in S5-A");
    }
    if kv_layout != F16Contig::NAME {
        bail!("forward_layer_decode: unsupported KV layout {kv_layout}");
    }
    if spec.has_kv {
        let attn_k = attn_k.expect("has_kv invariant");
        let attn_k_norm_w = attn_k_norm_w.expect("has_kv invariant");

        ops.mmvq(
            attn_k.ptr,
            scratch.x_q8_1,
            scratch.mmvq_f32,
            kv_width,
            hidden,
            attn_k.dtype,
        )
        .context("mmvq attn_k")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.k_f16, kv_width)
            .context("cast K → f16")?;

        if let Some(attn_v) = weights.attn_v.as_ref() {
            ops.mmvq(
                attn_v.ptr,
                scratch.x_q8_1,
                scratch.mmvq_f32,
                kv_width,
                hidden,
                attn_v.dtype,
            )
            .context("mmvq attn_v")?;
            ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.v_f16, kv_width)
                .context("cast V → f16")?;
        } else {
            // SAFETY: K/V buffers hold kv_width F16 values on device.
            unsafe {
                device.memcpy_async(
                    stream,
                    CopyDirection::DeviceToDevice,
                    scratch.v_f16,
                    scratch.k_f16,
                    kv_width * 2,
                )?;
            }
        }

        // Per-head Q / K learned RMSNorm; V unlearned (unit weight).
        ops.rmsnorm_f16(
            scratch.q_f16,
            weights.attn_q_norm,
            scratch.q_f16,
            n_heads,
            head_dim,
            rms_norm_eps,
        )
        .context("attn_q_norm")?;
        ops.rmsnorm_f16(
            scratch.k_f16,
            attn_k_norm_w,
            scratch.k_f16,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
        )
        .context("attn_k_norm")?;
        ops.rmsnorm_f16(
            scratch.v_f16,
            scratch.v_ones_f16,
            scratch.v_f16,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
        )
        .context("attn_v unlearned rmsnorm")?;

        // Position upload + RoPE on Q + K (NOT V).
        scratch.positions_host[0] = position as i32;
        // SAFETY: scratch.positions is i32 [1]; positions_host outlives the bounded sync.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                scratch.positions,
                DevicePtr(scratch.positions_host.as_ptr() as usize),
                4,
            )?;
        }
        ops.rope_neox_partial_f16(
            scratch.q_f16,
            scratch.positions,
            spec.rope_freq_base,
            1,
            n_heads,
            head_dim,
            spec.rope_dim,
        )
        .context("rope Q")?;
        ops.rope_neox_partial_f16(
            scratch.k_f16,
            scratch.positions,
            spec.rope_freq_base,
            1,
            n_kv_heads,
            head_dim,
            spec.rope_dim,
        )
        .context("rope K")?;

        // Append fresh K/V row to this layer's own cache.
        // SAFETY: K/V buffers hold kv_width F16 values on `device`.
        unsafe {
            kv_cache
                .append(device, stream, scratch.k_f16, scratch.v_f16, 1)
                .map_err(|e| anyhow::anyhow!("kv_cache.append: {e}"))?;
        }
    } else {
        // Shared-KV tail: Q gets per-head norm + RoPE (the layer's
        // own attn_q_norm + rope_freq_base apply), but K/V/RoPE-K +
        // append are skipped. The cache passed in is the source
        // layer's cache; n_tokens reflects the source's tail.
        ops.rmsnorm_f16(
            scratch.q_f16,
            weights.attn_q_norm,
            scratch.q_f16,
            n_heads,
            head_dim,
            rms_norm_eps,
        )
        .context("attn_q_norm (shared-KV tail)")?;
        scratch.positions_host[0] = position as i32;
        // SAFETY: see above.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                scratch.positions,
                DevicePtr(scratch.positions_host.as_ptr() as usize),
                4,
            )?;
        }
        ops.rope_neox_partial_f16(
            scratch.q_f16,
            scratch.positions,
            spec.rope_freq_base,
            1,
            n_heads,
            head_dim,
            spec.rope_dim,
        )
        .context("rope Q (shared-KV tail)")?;
    }

    let n_tokens_kv = kv_cache.current_tokens();
    ops.attention_decode_f16(
        scratch.q_f16,
        kv_cache.k_buffer(),
        kv_cache.v_buffer(),
        scratch.attn_out_f16,
        n_heads,
        n_kv_heads,
        head_dim,
        n_tokens_kv,
        softmax_scale,
        window,
    )
    .context("attention_decode_f16")?;

    // 10. Output projection.
    ops.quantize_f16_q8_1(scratch.attn_out_f16, scratch.x_q8_1, q_width)
        .context("quantize attn_out → Q8_1")?;
    ops.mmvq(
        weights.attn_output.ptr,
        scratch.x_q8_1,
        scratch.mmvq_f32,
        hidden,
        q_width,
        weights.attn_output.dtype,
    )
    .context("mmvq attn_output")?;
    ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.attn_out_f16, hidden)
        .context("cast attn_output → f16")?;

    // 11. post_attention_norm RMSNorm on attn_out, then add residual.
    ops.rmsnorm_f16(
        scratch.attn_out_f16,
        weights.post_attention_norm,
        scratch.post_attn_norm_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("post_attention_norm")?;
    ops.add_f16(
        x_in,
        scratch.post_attn_norm_f16,
        scratch.attn_residual_f16,
        hidden,
    )
    .context("residual_add post-attn")?;

    // 12-16. FFN section. Dense and MoE branches both produce `x_out
    // = post_ffw_norm(FFN(attn_residual)) + attn_residual`. The dense
    // branch is the inline gate/up/GELU/down sequence; the MoE branch
    // delegates to `forward_ffn_moe` (shared MLP || routed experts →
    // sum → post_ffw_norm → residual).
    if let (FfnKind::Moe, Some(moe_w), Some(moe_s)) =
        (spec.ffn_kind, weights.moe.as_ref(), moe_scratch)
    {
        crate::moe::forward_ffn_moe(
            ops,
            weights,
            moe_w,
            scratch.x_q8_1,
            scratch.mmvq_f32,
            scratch.gate_f32,
            scratch.up_f32,
            scratch.activated_f16,
            scratch.activated_q8_1,
            scratch.down_f32,
            moe_s,
            scratch.attn_residual_f16,
            x_out,
            ff_len,
            hidden,
            rms_norm_eps,
        )?;
    } else {
        // Dense FFN (existing path).
        // 12. RMSNorm(ffn_norm) on the post-attn residual.
        ops.rmsnorm_quant_q8_1(
            scratch.attn_residual_f16,
            weights.ffn_norm,
            scratch.x_q8_1,
            1,
            hidden,
            rms_norm_eps,
        )
        .context("ffn_norm + quant")?;

        // 13. Gate + up projections.
        ops.mmvq(
            weights.ffn_gate.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            ff_len,
            hidden,
            weights.ffn_gate.dtype,
        )
        .context("mmvq ffn_gate")?;
        ops.mmvq(
            weights.ffn_up.ptr,
            scratch.x_q8_1,
            scratch.up_f32,
            ff_len,
            hidden,
            weights.ffn_up.dtype,
        )
        .context("mmvq ffn_up")?;

        // 14. GELU(gate) * up, fused F32 → F16.
        ops.gelu_f32_to_f16(scratch.gate_f32, scratch.up_f32, scratch.activated_f16, ff_len)
            .context("gelu_f32_to_f16")?;

        // 15. Quantise + ffn_down projection.
        ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, ff_len)
            .context("quantize activated → Q8_1")?;
        ops.mmvq(
            weights.ffn_down.ptr,
            scratch.activated_q8_1,
            scratch.down_f32,
            hidden,
            ff_len,
            weights.ffn_down.dtype,
        )
        .context("mmvq ffn_down")?;
        ops.cast_f32_to_f16(scratch.down_f32, scratch.post_ffw_norm_f16, hidden)
            .context("cast ffn_down → f16")?;

        // 16. post_ffw_norm + residual add.
        ops.rmsnorm_f16(
            scratch.post_ffw_norm_f16,
            weights.post_ffw_norm,
            scratch.post_ffw_norm_f16,
            1,
            hidden,
            rms_norm_eps,
        )
        .context("post_ffw_norm")?;
        ops.add_f16(
            scratch.attn_residual_f16,
            scratch.post_ffw_norm_f16,
            x_out,
            hidden,
        )
        .context("residual_add post-ffn")?;
    }

    // 17. Optional per-layer side-channel embedding (E2B / E4B).
    if let (Some(pe_w), Some((slice, pe))) =
        (weights.per_layer_embed, per_layer_slice)
    {
        crate::per_layer_embd::forward_per_layer_post_block(
            ops,
            pe_w,
            x_out,                              // pe_in = current layer output
            slice,                              // table_slice [pe] F32
            scratch.gate_f32,                   // gate_out_f32 (sized ff_len ≥ pe)
            scratch.up_f32,                     // activated_f32 (ff_len ≥ pe)
            scratch.activated_f16,              // activated_f16 (ff_len ≥ pe)
            scratch.down_f32,                   // proj_out_f32 (hidden)
            scratch.post_attn_norm_f16,         // proj_out_f16 (hidden) — reused
            scratch.ffn_norm_f16,               // normed_f16 (hidden) — reused
            x_out,                              // result (in-place residual add)
            pe,
            hidden,
            rms_norm_eps,
        )?;
    }

    // 18. Optional per-layer output scalar.
    if let Some(scale_v) = weights.layer_output_scale {
        if scale_v != 1.0 {
            ops.scale_f16(x_out, x_out, hidden, scale_v)
                .context("layer_output_scale")?;
        }
    }

    Ok(())
}

/// Multi-token prefill through a single Gemma 4 layer. Mirrors
/// [`forward_layer_decode`] but operates on `n_tokens` rows. The
/// `start_position` is the cache tail length **before** this chunk
/// is appended.
///
/// Limitations of S8-B-A:
/// - Dense FFN only (MoE prefill = #23 follow-up).
/// - Per-layer side-channel embedding (E2B/E4B) bails (#22 follow-up).
/// - layer_output_scale applied uniformly across rows when present.
/// - Q8 KV layout bails (F16 only).
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_prefill<L: CacheLayout, O: Ops>(
    ops: &O,
    device: &HipDevice,
    stream: &HipStream,
    weights: &Gemma4LayerWeights,
    spec: &LayerSpec,
    rms_norm_eps: f32,
    ff_len: usize,
    hidden: usize,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut LayerPrefillScratch<'_>,
    x_in: DevicePtr,
    x_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_layer_prefill: n_tokens=0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_layer_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    if spec.ffn_kind != FfnKind::Dense {
        bail!("forward_layer_prefill: MoE prefill not supported in S8-B-A; see #23");
    }
    if weights.per_layer_embed.is_some() {
        bail!("forward_layer_prefill: per-layer-embd in prefill not supported in S8-B-A; see #22");
    }
    let kv_layout = L::NAME;
    if kv_layout == Q8Contig::NAME {
        bail!("forward_layer_prefill: Q8 KV not supported");
    }
    if kv_layout != F16Contig::NAME {
        bail!("forward_layer_prefill: unsupported KV layout {kv_layout}");
    }

    let head_dim = spec.head_dim;
    let n_heads = spec.n_heads;
    let n_kv_heads = spec.n_kv_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let window: i32 = spec.window as i32;
    let softmax_scale: f32 = 1.0;

    // 1. RMSNorm(x_in, attn_norm) + dual Q8_1 quantise (std + MMQ).
    ops.rmsnorm_f16(
        x_in,
        weights.attn_norm,
        scratch.x_norm_f16,
        n_tokens,
        hidden,
        rms_norm_eps,
    )
    .context("prefill attn_norm")?;
    ops.quantize_f16_q8_1(scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden)
        .context("prefill x_norm → Q8_1")?;
    ops.quantize_f16_q8_1_mmq(scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens)
        .context("prefill x_norm → Q8_1 (MMQ)")?;

    // 2. Q projection.
    ops.qmatmul(
        weights.attn_q.ptr,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.mmvq_f32,
        n_tokens,
        hidden,
        q_width,
        weights.attn_q.dtype,
    )
    .context("prefill qmatmul Q")?;
    ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.q_f16, n_tokens * q_width)?;

    // 3+4. K, V projections.
    if !spec.has_kv {
        bail!(
            "forward_layer_prefill: shared-KV tail (layer {}) — prefill of tail layers \
             requires the source layer to be at the same call's KV cache; \
             not supported in S8-B-A",
            spec.index
        );
    }
    let attn_k = weights
        .attn_k
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {}: attn_k missing", spec.index))?;
    let attn_k_norm_w = weights
        .attn_k_norm
        .ok_or_else(|| anyhow::anyhow!("layer {}: attn_k_norm missing", spec.index))?;

    ops.qmatmul(
        attn_k.ptr,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.mmvq_f32,
        n_tokens,
        hidden,
        kv_width,
        attn_k.dtype,
    )
    .context("prefill qmatmul K")?;
    ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.k_f16, n_tokens * kv_width)?;

    if let Some(attn_v) = weights.attn_v.as_ref() {
        ops.qmatmul(
            attn_v.ptr,
            scratch.x_q8_1,
            scratch.x_q8_1_mmq,
            scratch.mmvq_f32,
            n_tokens,
            hidden,
            kv_width,
            attn_v.dtype,
        )
        .context("prefill qmatmul V")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.v_f16, n_tokens * kv_width)?;
    } else {
        // V = K (alt-attention).
        // SAFETY: K and V buffers hold n_tokens * kv_width F16 each.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                scratch.v_f16,
                scratch.k_f16,
                n_tokens * kv_width * 2,
            )?;
        }
    }

    // 5. Per-head Q / K / V RMSNorms (V unlearned via unit weight).
    ops.rmsnorm_f16(
        scratch.q_f16,
        weights.attn_q_norm,
        scratch.q_f16,
        n_tokens * n_heads,
        head_dim,
        rms_norm_eps,
    )?;
    ops.rmsnorm_f16(
        scratch.k_f16,
        attn_k_norm_w,
        scratch.k_f16,
        n_tokens * n_kv_heads,
        head_dim,
        rms_norm_eps,
    )?;
    ops.rmsnorm_f16(
        scratch.v_f16,
        scratch.v_ones_f16,
        scratch.v_f16,
        n_tokens * n_kv_heads,
        head_dim,
        rms_norm_eps,
    )?;

    // 6. Position upload + RoPE on Q + K (V NOT rotated).
    for i in 0..n_tokens {
        scratch.positions_host[i] = (start_position + i) as i32;
    }
    // SAFETY: scratch.positions has n_tokens * 4 valid bytes; positions_host
    // outlives the bounded synchronize.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            scratch.positions,
            DevicePtr(scratch.positions_host.as_ptr() as usize),
            n_tokens * 4,
        )?;
    }
    ops.rope_neox_partial_f16(
        scratch.q_f16,
        scratch.positions,
        spec.rope_freq_base,
        n_tokens,
        n_heads,
        head_dim,
        spec.rope_dim,
    )?;
    ops.rope_neox_partial_f16(
        scratch.k_f16,
        scratch.positions,
        spec.rope_freq_base,
        n_tokens,
        n_kv_heads,
        head_dim,
        spec.rope_dim,
    )?;

    // 7. KV append.
    // SAFETY: K, V buffers hold n_tokens * kv_width F16 each.
    unsafe {
        kv_cache
            .append(device, stream, scratch.k_f16, scratch.v_f16, n_tokens)
            .map_err(|e| anyhow::anyhow!("kv_cache.append: {e}"))?;
    }
    let n_k_tokens = kv_cache.current_tokens();

    // 8. Attention (SWA-aware via window).
    ops.attention_prefill_f16(
        scratch.q_f16,
        kv_cache.k_buffer(),
        kv_cache.v_buffer(),
        scratch.attn_out_f16,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
        n_k_tokens,
        start_position,
        softmax_scale,
        window,
    )
    .context("attention_prefill_f16")?;

    // 9. Output projection.
    ops.quantize_f16_q8_1(scratch.attn_out_f16, scratch.x_q8_1, n_tokens * q_width)?;
    ops.quantize_f16_q8_1_mmq(scratch.attn_out_f16, scratch.x_q8_1_mmq, q_width, n_tokens)?;
    ops.qmatmul(
        weights.attn_output.ptr,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.mmvq_f32,
        n_tokens,
        q_width,
        hidden,
        weights.attn_output.dtype,
    )
    .context("prefill qmatmul attn_output")?;
    ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.attn_out_f16, n_tokens * hidden)?;

    // 10. post_attention_norm + residual.
    ops.rmsnorm_f16(
        scratch.attn_out_f16,
        weights.post_attention_norm,
        scratch.post_attn_norm_f16,
        n_tokens,
        hidden,
        rms_norm_eps,
    )?;
    ops.add_f16(
        x_in,
        scratch.post_attn_norm_f16,
        scratch.attn_residual_f16,
        n_tokens * hidden,
    )?;

    // 11. ffn_norm + dense FFN with GELU.
    ops.rmsnorm_f16(
        scratch.attn_residual_f16,
        weights.ffn_norm,
        scratch.x_norm_f16,
        n_tokens,
        hidden,
        rms_norm_eps,
    )?;
    ops.quantize_f16_q8_1(scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden)?;
    ops.quantize_f16_q8_1_mmq(scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens)?;
    ops.qmatmul(
        weights.ffn_gate.ptr,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.gate_f32,
        n_tokens,
        hidden,
        ff_len,
        weights.ffn_gate.dtype,
    )?;
    ops.qmatmul(
        weights.ffn_up.ptr,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.up_f32,
        n_tokens,
        hidden,
        ff_len,
        weights.ffn_up.dtype,
    )?;
    ops.gelu_f32_to_f16(
        scratch.gate_f32,
        scratch.up_f32,
        scratch.activated_f16,
        n_tokens * ff_len,
    )?;
    ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, n_tokens * ff_len)?;
    ops.quantize_f16_q8_1_mmq(scratch.activated_f16, scratch.activated_q8_1_mmq, ff_len, n_tokens)?;
    ops.qmatmul(
        weights.ffn_down.ptr,
        scratch.activated_q8_1,
        scratch.activated_q8_1_mmq,
        scratch.mmvq_f32,
        n_tokens,
        ff_len,
        hidden,
        weights.ffn_down.dtype,
    )?;
    ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.post_ffw_norm_f16, n_tokens * hidden)?;

    // 12. post_ffw_norm + final residual.
    ops.rmsnorm_f16(
        scratch.post_ffw_norm_f16,
        weights.post_ffw_norm,
        scratch.post_ffw_norm_f16,
        n_tokens,
        hidden,
        rms_norm_eps,
    )?;
    ops.add_f16(
        scratch.attn_residual_f16,
        scratch.post_ffw_norm_f16,
        x_out,
        n_tokens * hidden,
    )?;

    // 13. Optional layer_output_scale (broadcast scalar over all rows).
    if let Some(scale_v) = weights.layer_output_scale {
        if scale_v != 1.0 {
            ops.scale_f16(x_out, x_out, n_tokens * hidden, scale_v)?;
        }
    }
    Ok(())
}
