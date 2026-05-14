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
use flambeau_blocks::{
    StandardAttention, StandardAttentionDecodeScratch, WeightHandle,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
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
    // directly. The `has_kv == true` branch below validates that the
    // layer-load actually populated attn_k / attn_k_norm.

    let head_dim = spec.head_dim;
    let n_heads = spec.n_heads;
    let n_kv_heads = spec.n_kv_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let window: i32 = spec.window as i32;
    // Gemma 4 sets `f_attention_scale = 1.0f` (no pre-attention
    // scaling). See llama.cpp `model.cpp:1638`.
    let softmax_scale: f32 = 1.0;

    // KV layout assertion (the block also enforces this internally;
    // keep an early bail so the error message names the gemma4 layer).
    let kv_layout = L::NAME;
    if kv_layout == Q8Contig::NAME {
        bail!("forward_layer_decode: Q8 KV layout not supported in S5-A");
    }
    if kv_layout != F16Contig::NAME {
        bail!("forward_layer_decode: unsupported KV layout {kv_layout}");
    }

    if spec.has_kv {
        // Standard attention path. Steps 1-10 collapse into
        // `StandardAttention::forward_decode`, which runs:
        //   rmsnorm+Q8_1 → Q proj → K proj → V proj (or alt-V from K)
        //   → Q/K/V per-head norms (V uses the unit-weight buffer)
        //   → RoPE Q+K → KV append → attention → output_proj.
        // Mirrors gemma4 TP's per-rank dispatch at `tp.rs::forward_layer_decode_tp`.
        let attn_k = weights
            .attn_k
            .as_ref()
            .expect("has_kv invariant: attn_k present");
        let attn_k_norm_w = weights
            .attn_k_norm
            .expect("has_kv invariant: attn_k_norm present");
        let attn_v = weights.attn_v.as_ref().map(|v| WeightHandle {
            ptr: v.ptr,
            dtype: v.dtype,
            dims: [kv_width, hidden],
        });
        let attn_q_handle = WeightHandle {
            ptr: weights.attn_q.ptr,
            dtype: weights.attn_q.dtype,
            dims: [q_width, hidden],
        };
        let attn_k_handle = WeightHandle {
            ptr: attn_k.ptr,
            dtype: attn_k.dtype,
            dims: [kv_width, hidden],
        };
        let attn_output_handle = WeightHandle {
            ptr: weights.attn_output.ptr,
            dtype: weights.attn_output.dtype,
            dims: [hidden, q_width],
        };
        let block = StandardAttention::new(
            attn_q_handle,
            attn_k_handle,
            attn_v,
            attn_output_handle,
            weights.attn_norm,
            weights.attn_q_norm,
            attn_k_norm_w,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
            spec.rope_freq_base,
            spec.rope_dim,
            /* gated = */ false,
        )
        .context("StandardAttention::new (gemma4 layer)")?
        .with_softmax_scale(softmax_scale)
        .with_v_norm_w(scratch.v_ones_f16);
        let block = if window > 0 {
            block.with_window_size(window as u32)
        } else {
            block
        };
        let mut std_scratch = StandardAttentionDecodeScratch {
            x_q8_1: scratch.x_q8_1,
            mmvq_f32: scratch.mmvq_f32,
            q_fused_f16: DevicePtr(0),
            q_f16: scratch.q_f16,
            gate_f16: DevicePtr(0),
            k_f16: scratch.k_f16,
            v_f16: scratch.v_f16,
            k_q8_0: DevicePtr(0),
            v_q8_0: DevicePtr(0),
            attn_out_f16: scratch.attn_out_f16,
            gated_out_f16: DevicePtr(0),
            positions: scratch.positions,
            positions_host: scratch.positions_host,
            splitk_partials_m: scratch.splitk_partials_m,
            splitk_partials_s: scratch.splitk_partials_s,
            splitk_partials_o: scratch.splitk_partials_o,
        };
        block
            .forward_decode(
                ops,
                device,
                stream,
                x_in,
                scratch.attn_out_f16,
                kv_cache,
                &mut std_scratch,
                position,
                /* slots = */ None,
            )
            .context("StandardAttention::forward_decode (gemma4 layer)")?;
    } else {
        // Shared-KV tail (has_kv == false): Q runs through this layer's
        // own attn_norm + Q proj + attn_q_norm + RoPE Q, but K/V/append
        // are skipped (the routed cache already holds them from the
        // source layer). Attention reads the routed cache; output_proj
        // runs against this layer's `attn_output`. The
        // `StandardAttention` block always appends, so the tail path
        // stays inline.
        ops.rmsnorm_quant_q8_1(
            x_in,
            weights.attn_norm,
            scratch.x_q8_1,
            1,
            hidden,
            rms_norm_eps,
        )
        .context("attn_norm + quant (shared-KV tail)")?;
        ops.mmvq(
            weights.attn_q.ptr,
            scratch.x_q8_1,
            scratch.mmvq_f32,
            q_width,
            hidden,
            weights.attn_q.dtype,
        )
        .context("mmvq attn_q (shared-KV tail)")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.q_f16, q_width)
            .context("cast Q → f16 (shared-KV tail)")?;
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
        .context("rope Q (shared-KV tail)")?;
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
        .context("attention_decode_f16 (shared-KV tail)")?;
        ops.quantize_f16_q8_1(scratch.attn_out_f16, scratch.x_q8_1, q_width)
            .context("quantize attn_out → Q8_1 (shared-KV tail)")?;
        ops.mmvq(
            weights.attn_output.ptr,
            scratch.x_q8_1,
            scratch.mmvq_f32,
            hidden,
            q_width,
            weights.attn_output.dtype,
        )
        .context("mmvq attn_output (shared-KV tail)")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.attn_out_f16, hidden)
            .context("cast attn_output → f16 (shared-KV tail)")?;
    }

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
    if std::env::var_os("FLAMBEAU_LAYER_PROBE").is_some() && spec.index < 2 {
        use flambeau_core::CopyDirection;
        let mut host = vec![half::f16::from_f32(0.0); hidden];
        // SAFETY: attn_residual_f16 owns hidden*2 bytes.
        unsafe {
            let _ = device.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                flambeau_core::DevicePtr(host.as_mut_ptr() as usize),
                scratch.attn_residual_f16,
                hidden * 2,
            );
        }
        let _ = stream.synchronize();
        let max_abs = host.iter().map(|h| h.to_f32().abs()).fold(0.0f32, f32::max);
        let nans = host.iter().filter(|h| h.to_f32().is_nan()).count();
        eprintln!(
            "  [LAYER_PROBE] L{} attn_residual | max_abs={max_abs:.4} nans={nans} first4={:?}",
            spec.index,
            &host[..4].iter().map(|h| h.to_f32()).collect::<Vec<_>>()
        );
    }

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

    // Steps 1-9 collapse into `StandardAttention::forward_prefill`.
    // Mirrors the decode-side migration: the block runs rmsnorm + Q/K/V
    // proj + per-head Q/K/V norms + RoPE + KV append + attention +
    // output_proj. The block always appends to `kv_cache`, so the
    // (rare) shared-KV-tail prefill case bails up front.
    if !spec.has_kv {
        bail!(
            "forward_layer_prefill: shared-KV tail (layer {}) — prefill of tail layers \
             requires the source layer's KV at the same call site; not supported",
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
    let attn_v = weights.attn_v.as_ref().map(|v| WeightHandle {
        ptr: v.ptr,
        dtype: v.dtype,
        dims: [kv_width, hidden],
    });
    let attn_q_handle = WeightHandle {
        ptr: weights.attn_q.ptr,
        dtype: weights.attn_q.dtype,
        dims: [q_width, hidden],
    };
    let attn_k_handle = WeightHandle {
        ptr: attn_k.ptr,
        dtype: attn_k.dtype,
        dims: [kv_width, hidden],
    };
    let attn_output_handle = WeightHandle {
        ptr: weights.attn_output.ptr,
        dtype: weights.attn_output.dtype,
        dims: [hidden, q_width],
    };
    let block = StandardAttention::new(
        attn_q_handle,
        attn_k_handle,
        attn_v,
        attn_output_handle,
        weights.attn_norm,
        weights.attn_q_norm,
        attn_k_norm_w,
        hidden,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_norm_eps,
        spec.rope_freq_base,
        spec.rope_dim,
        /* gated = */ false,
    )
    .context("StandardAttention::new (gemma4 prefill)")?
    .with_softmax_scale(softmax_scale)
    .with_v_norm_w(scratch.v_ones_f16);
    let block = if window > 0 {
        block.with_window_size(window as u32)
    } else {
        block
    };
    let mut std_scratch = flambeau_blocks::StandardAttentionPrefillScratch {
        max_tokens: scratch.max_tokens,
        x_norm_f16: scratch.x_norm_f16,
        x_q8_1: scratch.x_q8_1,
        x_q8_1_mmq: scratch.x_q8_1_mmq,
        mmvq_f32: scratch.mmvq_f32,
        q_fused_f16: DevicePtr(0),
        q_f16: scratch.q_f16,
        gate_f16: DevicePtr(0),
        k_f16: scratch.k_f16,
        v_f16: scratch.v_f16,
        attn_out_f16: scratch.attn_out_f16,
        gated_out_f16: DevicePtr(0),
        positions: scratch.positions,
        gated_q8_1: scratch.gated_q8_1,
        gated_q8_1_mmq: scratch.gated_q8_1_mmq,
        positions_host: scratch.positions_host,
    };
    block
        .forward_prefill(
            ops,
            device,
            stream,
            x_in,
            scratch.attn_out_f16,
            kv_cache,
            &mut std_scratch,
            n_tokens,
            start_position,
            /* slots = */ None,
        )
        .context("StandardAttention::forward_prefill (gemma4 layer)")?;

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
