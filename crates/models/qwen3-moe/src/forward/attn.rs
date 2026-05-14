//! Full-attention forward: decode (one token) + prefill (L tokens).
//! Composes the attention block — RMSNorm + fused Q|gate projection + K/V
//! projection + partial NeoX RoPE + KV append + softmax attention + output
//! projection. The KV cache is `F16Contig`; Q8-KV is a separate code path
//! in V2+.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::hip::{
    attention::split_q_gate_f16,
    cast::cast_f32_to_f16,
    mlp::sigmoid_mul_f16,
    norm::{quantize_f16_q8_1, quantize_f16_q8_1_mmq, rmsnorm_f16},
    pe::rope_neox_partial_f16,
    qmatmul::qmatmul,
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::BlockQ8_1;
use flambeau_runtime::{CacheLayout, KvCache};

use super::common::{mat_shape, qdtype_of};
use crate::config::Qwen3MoEConfig;
use crate::session::LayerCache;
use crate::weights::{DenseAttnWeights, DeviceTensor, FullAttnWeights};

/// Workspace buffers needed by one decode step of a full-attention layer.
/// Thin wrapper over [`flambeau_blocks::OwnedStandardAttentionDecodeScratch`]:
/// the block owns the device buffers; this struct keeps the per-stage
/// [`flambeau_blocks::RawAllocTracker`] so `dispose(device)` walks every
/// allocation in one place.
pub struct FullAttnScratch {
    inner: flambeau_blocks::OwnedStandardAttentionDecodeScratch,
    tracker: flambeau_blocks::RawAllocTracker,
    disposed: bool,
}

/// Re-export of the block's split-K partial budget so callers that
/// reach into `flambeau_qwen3_moe::forward::attn::MAX_SPLITK_CHUNKS`
/// continue to compile after the migration.
pub use flambeau_blocks::MAX_SPLITK_CHUNKS;

impl FullAttnScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let mut tracker = flambeau_blocks::RawAllocTracker::new();
        let dims = flambeau_blocks::AttentionScratchDims {
            hidden: cfg.hidden_size,
            n_heads: cfg.num_heads,
            n_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        };
        let inner = flambeau_blocks::StandardAttention::alloc_decode_scratch(
            device,
            &mut tracker,
            dims,
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

    /// Build a borrowed view over this scratch shaped to feed into
    /// `flambeau_blocks::StandardAttention::forward_decode`.
    pub fn view_mut(&mut self) -> flambeau_blocks::StandardAttentionDecodeScratch<'_> {
        self.inner.view_mut()
    }
}

// `upload_position`, `qdtype_of`, `mat_shape` moved to `forward::common`.

/// Decode step for one full-attention layer. Consumes `x_in` (F16 `[H]`)
/// and writes the pre-residual output to `delta_out` (F16 `[H]`). The
/// caller is expected to do the residual sum (`out = x_in + delta_out`)
/// outside this function — e adds the fused residual-add kernel
/// for the top-level compose.
/// Appends to `kv_cache` at the current tail. `position` is the 0-based
/// token index used by RoPE and also the `n_tokens_kv` for the attention
/// kernel after the append bumps the cache size by 1.
/// Per-layer slot bundle for graph-captureable decode. Re-exported
/// from `flambeau_blocks` so layer-level wrappers (e.g.
/// `LayerDecodeSlots`) and graph-capture call sites in qwen3-moe
/// keep their existing field names.
pub use flambeau_blocks::AttnDecodeSlots;

/// Build a `flambeau_blocks::StandardAttention` for the gated
/// (qwen35moe / qwen36moe) full-attention path. Q+gate are fused
/// at `2 * n_heads * head_dim` rows; the output goes through a
/// post-attn sigmoid gate.
pub fn build_full_attn_block(
    attn_norm: &DeviceTensor,
    weights: &FullAttnWeights,
    cfg: &Qwen3MoEConfig,
) -> Result<flambeau_blocks::StandardAttention> {
    let q_dtype = qdtype_of(weights.attn_q.dtype)?;
    let k_dtype = qdtype_of(weights.attn_k.dtype)?;
    let v_dtype = qdtype_of(weights.attn_v.dtype)?;
    let o_dtype = qdtype_of(weights.attn_output.dtype)?;
    let (q_rows, q_k) = mat_shape(&weights.attn_q)?;
    let (k_rows, k_k) = mat_shape(&weights.attn_k)?;
    let (v_rows, v_k) = mat_shape(&weights.attn_v)?;
    let (o_rows, o_k) = mat_shape(&weights.attn_output)?;
    flambeau_blocks::StandardAttention::new(
        flambeau_blocks::WeightHandle { ptr: weights.attn_q.ptr, dtype: q_dtype, dims: [q_rows, q_k] },
        flambeau_blocks::WeightHandle { ptr: weights.attn_k.ptr, dtype: k_dtype, dims: [k_rows, k_k] },
        Some(flambeau_blocks::WeightHandle { ptr: weights.attn_v.ptr, dtype: v_dtype, dims: [v_rows, v_k] }),
        flambeau_blocks::WeightHandle {
            ptr: weights.attn_output.ptr,
            dtype: o_dtype,
            dims: [o_rows, o_k],
        },
        attn_norm.ptr,
        weights.attn_q_norm.ptr,
        weights.attn_k_norm.ptr,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
        cfg.rope.freq_base,
        cfg.rope.rotated_dims,
        true, // gated
    )
}

/// Build a `flambeau_blocks::StandardAttention` for the plain Q
/// (qwen3moe) full-attention path. Q is a single matmul at
/// `n_heads * head_dim` rows; the output projection consumes
/// `attn_out_f16` directly with no sigmoid gate.
///
/// V1 doesn't carry Q/K/V biases on the block surface — qwen3moe
/// arches that ship them are rejected at this constructor.
pub fn build_dense_attn_block(
    attn_norm: &DeviceTensor,
    weights: &DenseAttnWeights,
    cfg: &Qwen3MoEConfig,
) -> Result<flambeau_blocks::StandardAttention> {
    if weights.attn_q_bias.is_some()
        || weights.attn_k_bias.is_some()
        || weights.attn_v_bias.is_some()
    {
        bail!(
            "build_dense_attn_block: Q/K/V biases not supported on the block \
             surface (Qwen3-Coder-30B has none; older Qwen3 variants need a \
             bias-add step added to the block)"
        );
    }
    let q_dtype = qdtype_of(weights.attn_q.dtype)?;
    let k_dtype = qdtype_of(weights.attn_k.dtype)?;
    let v_dtype = qdtype_of(weights.attn_v.dtype)?;
    let o_dtype = qdtype_of(weights.attn_output.dtype)?;
    let (q_rows, q_k) = mat_shape(&weights.attn_q)?;
    let (k_rows, k_k) = mat_shape(&weights.attn_k)?;
    let (v_rows, v_k) = mat_shape(&weights.attn_v)?;
    let (o_rows, o_k) = mat_shape(&weights.attn_output)?;
    flambeau_blocks::StandardAttention::new(
        flambeau_blocks::WeightHandle { ptr: weights.attn_q.ptr, dtype: q_dtype, dims: [q_rows, q_k] },
        flambeau_blocks::WeightHandle { ptr: weights.attn_k.ptr, dtype: k_dtype, dims: [k_rows, k_k] },
        Some(flambeau_blocks::WeightHandle { ptr: weights.attn_v.ptr, dtype: v_dtype, dims: [v_rows, v_k] }),
        flambeau_blocks::WeightHandle {
            ptr: weights.attn_output.ptr,
            dtype: o_dtype,
            dims: [o_rows, o_k],
        },
        attn_norm.ptr,
        weights.attn_q_norm.ptr,
        weights.attn_k_norm.ptr,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
        cfg.rope.freq_base,
        cfg.rope.rotated_dims,
        false, // gated
    )
}

pub fn forward_full_attn_decode<L: CacheLayout>(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    post_attn_norm: Option<&DeviceTensor>,
    weights: &FullAttnWeights,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    position: usize,
    slots: Option<AttnDecodeSlots>,
) -> Result<()> {
    // b wires the attention block only; the FFN side of the
    // residual is d. `post_attn_norm` is still unused here; keep
    // the handle so d can call it without a second signature.
    let _ = post_attn_norm;

    // The block carries both the non-slot and the graph-capture-aware
    // (slot-tagged) kernel paths; route both through it.
    let block = build_full_attn_block(attn_norm, weights, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_decode(
        &hipops,
        device,
        stream,
        x_in,
        delta_out,
        kv_cache,
        &mut scratch.view_mut(),
        position,
        slots,
    )
}


/// Route a `LayerCache` entry through the full-attn forward, pulling the
/// correct `KvCache` out of the enum. Fails if the layer is actually a
/// GDN layer (caller dispatch error).
/// dispatches on the cache variant (F16Contig or Q8Contig)
/// to the same generic `forward_full_attn_decode<L>` body; the compiler
/// monomorphises and the runtime branch in the body picks the right
/// quantise + attention kernels.
pub fn forward_full_attn_layer_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    position: usize,
    slots: Option<AttnDecodeSlots>,
) -> Result<()> {
    let crate::weights::AttnWeights::FullAttn(fa) = &layer_weights.attn else {
        bail!(
            "layer {} weights are not FullAttn variant",
            layer_weights.layer_idx
        );
    };
    match layer_cache {
        LayerCache::FullAttn(kv) => forward_full_attn_decode(
            ops, stream, device, cfg,
            &layer_weights.attn_norm, layer_weights.post_attention_norm.as_ref(),
            fa, kv, scratch, x_in, delta_out, position, slots,
        ),
        LayerCache::FullAttnQ8(kv) => forward_full_attn_decode(
            ops, stream, device, cfg,
            &layer_weights.attn_norm, layer_weights.post_attention_norm.as_ref(),
            fa, kv, scratch, x_in, delta_out, position, slots,
        ),
        LayerCache::Gdn(_) => bail!(
            "layer {} is not a full-attn layer (cache variant mismatch)",
            layer_weights.layer_idx
        ),
    }
}


// ---------------------------------------------------------------------------
// f1 — full-attention prefill (L > 1).
// ---------------------------------------------------------------------------

/// Workspace for one prefill chunk of a full-attention layer. Sized once
/// against `(cfg, max_prefill_tokens)` — the caller chunks long prompts
/// to keep scratch VRAM bounded (f4 decides the chunk size).
/// The buffers scale linearly with `max_prefill_tokens` except `x_q8_1`
/// (which scales in blocks of 32 inputs). At hidden=2048 and L=128:
/// activations + scratch < 10 MB total — comfortable even on 16 GB cards.
pub struct FullAttnPrefillScratch {
    pub max_tokens: usize,
    pub x_norm_f16: DevicePtr,      // F16 [max_L, hidden] — rmsnorm output buffer
                                    // (8: split away from the
                                    // D1 fused rmsnorm+quant path so we
                                    // can emit both Q8_1 layouts.)
    pub x_q8_1: DevicePtr,          // Q8_1 blocks [max_L, hidden/32]
    pub x_q8_1_mmq: DevicePtr,      // BlockQ8_1Mmq [hidden/128, max_L] — DS4 layout for MmqLdsX64
    pub mmvq_f32: DevicePtr,        // F32 [max_L, max(2*H*D, H_kv*D, hidden)]
    pub q_fused_f16: DevicePtr,     // F16 [max_L, 2*n_heads*head_dim]
    pub q_f16: DevicePtr,           // F16 [max_L, n_heads*head_dim]
    pub gate_f16: DevicePtr,        // F16 [max_L, n_heads*head_dim]
    pub k_f16: DevicePtr,           // F16 [max_L, n_kv_heads*head_dim]
    pub v_f16: DevicePtr,           // F16 [max_L, n_kv_heads*head_dim]
    pub attn_out_f16: DevicePtr,    // F16 [max_L, n_heads*head_dim]
    pub gated_out_f16: DevicePtr,   // F16 [max_L, n_heads*head_dim]
    pub positions: DevicePtr,       // i32 [max_L]
    pub gated_q8_1: DevicePtr,      // Q8_1 [max_L, n_heads*head_dim/32]
    pub gated_q8_1_mmq: DevicePtr,  // BlockQ8_1Mmq [q_width/128, max_L] — DS4 layout
    /// **#266c** — per-slot K-cache base pointers, uploaded HtoD once
    /// per `forward_full_attn_layer_decode_batched_*` call so the
    /// batched-attention kernel can address each slot's KV. u64 [max_L].
    pub slot_k_ptrs: DevicePtr,
    /// **#266c** — per-slot V-cache base pointers. u64 [max_L].
    pub slot_v_ptrs: DevicePtr,
    /// **#266c** — per-slot KV-tail length post-append. i32 [max_L].
    pub slot_n_tokens_kv: DevicePtr,
    /// Per-slot pre-bump write position (cache tail BEFORE this token's
    /// append). Consumed by `kv_append_f16_batched_slots`. i32 [max_L].
    pub slot_write_pos: DevicePtr,
    /// Persistent host-side staging for the slot tables. Same lifetime
    /// rationale as `positions_host`: stable address for HtoD memcpy.
    pub(crate) slot_k_ptrs_host: Vec<u64>,
    pub(crate) slot_v_ptrs_host: Vec<u64>,
    pub(crate) slot_n_tokens_kv_host: Vec<i32>,
    pub(crate) slot_write_pos_host: Vec<i32>,
    /// 6.a-i5a — persistent host-side position buffer. `positions`
    /// on the device is filled each prefill call via a HtoD memcpy
    /// whose *source* is this Vec's stable address. Keeping it on the
    /// scratch (and therefore alive for the scratch's lifetime) is
    /// what makes the memcpy safe to capture into a `HipGraphExec` —
    /// the previous path used a transient `Vec<i32>` created inside
    /// `upload_positions_range`, whose address becomes invalid once
    /// that function returns and breaks graph replay.
    /// Sized `max_tokens`; writes are in-place via `[..n].copy_from_slice`.
    pub(crate) positions_host: Vec<i32>,
    // Bookkeeping.
    x_norm_f16_bytes: usize,
    x_q8_1_bytes: usize,
    x_q8_1_mmq_bytes: usize,
    mmvq_f32_bytes: usize,
    q_fused_bytes: usize,
    qk_bytes: usize,
    kv_bytes: usize,
    attn_bytes: usize,
    positions_bytes: usize,
    gated_q8_1_bytes: usize,
    gated_q8_1_mmq_bytes: usize,
    slot_k_ptrs_bytes: usize,
    slot_v_ptrs_bytes: usize,
    slot_n_tokens_kv_bytes: usize,
    slot_write_pos_bytes: usize,
    disposed: bool,
}

impl FullAttnPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.num_heads;
        let n_kv_heads = cfg.num_kv_heads;

        let q_fused_width = 2 * n_heads * head_dim;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;

        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(q_width % 32 == 0, "n_heads * head_dim must be multiple of 32");
        assert!(
            hidden % 128 == 0,
            "hidden must be a multiple of QK8_1_MMQ=128 for the DS4 layout"
        );
        assert!(
            q_width % 128 == 0,
            "n_heads * head_dim must be a multiple of QK8_1_MMQ=128 for the DS4 layout"
        );

        let mmq_block = std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let x_norm_f16_bytes = max_tokens * hidden * 2;
        let x_q8_1_bytes = max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let x_q8_1_mmq_bytes = max_tokens * (hidden / 128) * mmq_block;
        let mmvq_f32_bytes = max_tokens * q_fused_width.max(hidden) * 4;
        let q_fused_bytes = max_tokens * q_fused_width * 2;
        let qk_bytes = max_tokens * q_width * 2;
        let kv_bytes = max_tokens * kv_width * 2;
        let attn_bytes = max_tokens * q_width * 2;
        let positions_bytes = max_tokens * 4;
        let gated_q8_1_bytes =
            max_tokens * (q_width / 32) * std::mem::size_of::<BlockQ8_1>();
        let gated_q8_1_mmq_bytes = max_tokens * (q_width / 128) * mmq_block;
        let slot_k_ptrs_bytes = max_tokens * 8;
        let slot_v_ptrs_bytes = max_tokens * 8;
        let slot_n_tokens_kv_bytes = max_tokens * 4;
        let slot_write_pos_bytes = max_tokens * 4;

        let x_norm_f16 = device.alloc(x_norm_f16_bytes)?;
        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let x_q8_1_mmq = device.alloc(x_q8_1_mmq_bytes)?;
        let mmvq_f32 = device.alloc(mmvq_f32_bytes)?;
        let q_fused_f16 = device.alloc(q_fused_bytes)?;
        let q_f16 = device.alloc(qk_bytes)?;
        let gate_f16 = device.alloc(qk_bytes)?;
        let k_f16 = device.alloc(kv_bytes)?;
        let v_f16 = device.alloc(kv_bytes)?;
        let attn_out_f16 = device.alloc(attn_bytes)?;
        let gated_out_f16 = device.alloc(attn_bytes)?;
        let positions = device.alloc(positions_bytes)?;
        let gated_q8_1 = device.alloc(gated_q8_1_bytes)?;
        let gated_q8_1_mmq = device.alloc(gated_q8_1_mmq_bytes)?;
        let slot_k_ptrs = device.alloc(slot_k_ptrs_bytes)?;
        let slot_v_ptrs = device.alloc(slot_v_ptrs_bytes)?;
        let slot_n_tokens_kv = device.alloc(slot_n_tokens_kv_bytes)?;
        let slot_write_pos = device.alloc(slot_write_pos_bytes)?;

        Ok(Self {
            max_tokens,
            x_norm_f16,
            x_q8_1,
            x_q8_1_mmq,
            mmvq_f32,
            q_fused_f16,
            q_f16,
            gate_f16,
            k_f16,
            v_f16,
            attn_out_f16,
            gated_out_f16,
            positions,
            gated_q8_1,
            gated_q8_1_mmq,
            slot_k_ptrs,
            slot_v_ptrs,
            slot_n_tokens_kv,
            slot_write_pos,
            slot_k_ptrs_host: vec![0u64; max_tokens],
            slot_v_ptrs_host: vec![0u64; max_tokens],
            slot_n_tokens_kv_host: vec![0i32; max_tokens],
            slot_write_pos_host: vec![0i32; max_tokens],
            positions_host: vec![0i32; max_tokens],
            x_norm_f16_bytes,
            x_q8_1_bytes,
            x_q8_1_mmq_bytes,
            mmvq_f32_bytes,
            q_fused_bytes,
            qk_bytes,
            kv_bytes,
            attn_bytes,
            positions_bytes,
            gated_q8_1_bytes,
            gated_q8_1_mmq_bytes,
            slot_k_ptrs_bytes,
            slot_v_ptrs_bytes,
            slot_n_tokens_kv_bytes,
            slot_write_pos_bytes,
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
            device.dealloc(self.mmvq_f32, self.mmvq_f32_bytes)?;
            device.dealloc(self.q_fused_f16, self.q_fused_bytes)?;
            device.dealloc(self.q_f16, self.qk_bytes)?;
            device.dealloc(self.gate_f16, self.qk_bytes)?;
            device.dealloc(self.k_f16, self.kv_bytes)?;
            device.dealloc(self.v_f16, self.kv_bytes)?;
            device.dealloc(self.attn_out_f16, self.attn_bytes)?;
            device.dealloc(self.gated_out_f16, self.attn_bytes)?;
            device.dealloc(self.positions, self.positions_bytes)?;
            device.dealloc(self.gated_q8_1, self.gated_q8_1_bytes)?;
            device.dealloc(self.gated_q8_1_mmq, self.gated_q8_1_mmq_bytes)?;
            device.dealloc(self.slot_k_ptrs, self.slot_k_ptrs_bytes)?;
            device.dealloc(self.slot_v_ptrs, self.slot_v_ptrs_bytes)?;
            device.dealloc(self.slot_n_tokens_kv, self.slot_n_tokens_kv_bytes)?;
            device.dealloc(self.slot_write_pos, self.slot_write_pos_bytes)?;
        }
        Ok(())
    }
}

impl Drop for FullAttnPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "FullAttnPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

impl FullAttnPrefillScratch {
    /// Build a borrowed view shaped to feed into
    /// `flambeau_blocks::StandardAttention::forward_prefill`. The
    /// batched-decode-only fields (`slot_k_ptrs`, `slot_v_ptrs`,
    /// `slot_n_tokens_kv`) are not in the block's surface and stay on
    /// `FullAttnPrefillScratch` for callers that need them
    /// (`forward_full_attn_layer_decode_batched`).
    pub fn view_mut(&mut self) -> flambeau_blocks::StandardAttentionPrefillScratch<'_> {
        flambeau_blocks::StandardAttentionPrefillScratch {
            max_tokens: self.max_tokens,
            x_norm_f16: self.x_norm_f16,
            x_q8_1: self.x_q8_1,
            x_q8_1_mmq: self.x_q8_1_mmq,
            mmvq_f32: self.mmvq_f32,
            q_fused_f16: self.q_fused_f16,
            q_f16: self.q_f16,
            gate_f16: self.gate_f16,
            k_f16: self.k_f16,
            v_f16: self.v_f16,
            attn_out_f16: self.attn_out_f16,
            gated_out_f16: self.gated_out_f16,
            positions: self.positions,
            gated_q8_1: self.gated_q8_1,
            gated_q8_1_mmq: self.gated_q8_1_mmq,
            positions_host: &mut self.positions_host,
        }
    }

    /// Builds the batched-decode view (prefill scratch plus the
    /// per-slot tables that `forward_decode_batched_tp` reads).
    pub fn view_mut_batched(
        &mut self,
    ) -> flambeau_blocks::StandardAttentionBatchedDecodeScratch<'_> {
        flambeau_blocks::StandardAttentionBatchedDecodeScratch {
            max_tokens: self.max_tokens,
            x_norm_f16: self.x_norm_f16,
            x_q8_1: self.x_q8_1,
            x_q8_1_mmq: self.x_q8_1_mmq,
            mmvq_f32: self.mmvq_f32,
            q_fused_f16: self.q_fused_f16,
            q_f16: self.q_f16,
            gate_f16: self.gate_f16,
            k_f16: self.k_f16,
            v_f16: self.v_f16,
            attn_out_f16: self.attn_out_f16,
            gated_out_f16: self.gated_out_f16,
            positions: self.positions,
            gated_q8_1: self.gated_q8_1,
            gated_q8_1_mmq: self.gated_q8_1_mmq,
            positions_host: &mut self.positions_host,
            slot_k_ptrs: self.slot_k_ptrs,
            slot_v_ptrs: self.slot_v_ptrs,
            slot_n_tokens_kv: self.slot_n_tokens_kv,
            slot_write_pos: self.slot_write_pos,
            slot_k_ptrs_host: &mut self.slot_k_ptrs_host,
            slot_v_ptrs_host: &mut self.slot_v_ptrs_host,
            slot_n_tokens_kv_host: &mut self.slot_n_tokens_kv_host,
            slot_write_pos_host: &mut self.slot_write_pos_host,
        }
    }
}

/// Upload `L` i32 positions `[start_position, start_position + L)` into
/// the device-side `positions` scratch slot.
/// 6.a-i5a — the host-side source is `scratch.positions_host`, a
/// persistent `Vec<i32>` owned by the scratch. Previously this fn built
/// a transient `Vec<i32>` on the stack, uploaded, and synced to keep
/// the Vec alive across the copy. That works for direct dispatch but
/// breaks graph capture: the memcpy node records the source POINTER;
/// the driver re-reads from it at replay time; if the Vec is gone,
/// replay reads freed memory. Using `positions_host` keeps the source
/// stable for the scratch's lifetime, so the same memcpy replays safely
/// and — critically — picks up new values when we overwrite the host
/// Vec between replays.
/// The internal `stream.synchronize()` is dropped: the memcpy is
/// ordered in-stream with any subsequent RoPE launch, and the host
/// storage (`positions_host`) outlives both the copy and the stream
/// work that reads the device target.
#[allow(dead_code)] // kept available for future batched-decode TP migration
pub(crate) fn upload_positions_range(
    device: &HipDevice,
    stream: &HipStream,
    scratch: &mut FullAttnPrefillScratch,
    start_position: usize,
    n: usize,
) -> Result<()> {
    if n > scratch.max_tokens {
        anyhow::bail!(
            "upload_positions_range: n={n} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    for i in 0..n {
        scratch.positions_host[i] = (start_position + i) as i32;
    }
    // SAFETY: `scratch.positions` has at least `n * 4` valid bytes;
    // `positions_host` has at least `n` valid i32 slots and outlives
    // every in-stream reader of the device dst (bounded by the scratch
    // lifetime, which the caller is responsible for keeping alive
    // until the stream is synced).
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            scratch.positions,
            DevicePtr(scratch.positions_host.as_ptr() as usize),
            n * 4,
        )?;
    }
    Ok(())
}

/// Prefill step for one full-attention layer over `n_tokens = L` inputs.
/// The KV cache is expected to hold `start_position` tokens of history
/// (0 on a fresh sequence); this call appends the L new tokens and
/// computes causal attention for each new Q row against the combined
/// `[history + L]` KV.
/// `x_in` layout: F16 `[L, hidden]`, row-major (rows are tokens).
/// `delta_out` layout: F16 `[L, hidden]`.
/// Optional slot bundle for graph-captureable prefill. Re-exported
/// from `flambeau_blocks` so existing call sites in qwen3-moe keep
/// their field names.
pub use flambeau_blocks::AttnPrefillSlots;

pub fn forward_full_attn_prefill<L: flambeau_runtime::CacheLayout>(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &FullAttnWeights,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut FullAttnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
    slots: Option<AttnPrefillSlots>,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_full_attn_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_full_attn_prefill: n_tokens={n_tokens} > scratch.max_tokens={}; caller must chunk",
            scratch.max_tokens
        );
    }

    // The block carries both the non-slot and the graph-capture-aware
    // (slot-tagged) prefill paths; route both through it.
    let block = build_full_attn_block(attn_norm, weights, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_prefill(
        &hipops,
        device,
        stream,
        x_in,
        delta_out,
        kv_cache,
        &mut scratch.view_mut(),
        n_tokens,
        start_position,
        slots,
    )
}

/// upload arbitrary per-slot positions into
/// `scratch.positions`. Sibling of `upload_positions_range` for the
/// continuous-batching path where each batched row decodes at its own
/// cache offset.
pub(crate) fn upload_positions_arbitrary(
    device: &HipDevice,
    stream: &HipStream,
    scratch: &mut FullAttnPrefillScratch,
    slot_positions: &[usize],
) -> Result<()> {
    let n = slot_positions.len();
    if n > scratch.max_tokens {
        anyhow::bail!(
            "upload_positions_arbitrary: n={n} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    for (i, &p) in slot_positions.iter().enumerate() {
        scratch.positions_host[i] = p as i32;
    }
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            scratch.positions,
            DevicePtr(scratch.positions_host.as_ptr() as usize),
            n * 4,
        )?;
    }
    Ok(())
}

/// **P2.9b-i2-A1** — batched decode for one full-attention layer.
/// Mirrors [`forward_full_attn_prefill`] for the input-side ops
/// (rmsnorm, Q|gate / K / V projections, per-head Q/K rmsnorm, RoPE)
/// at `n_tokens = slot_positions.len()`, so those kernel launches are
/// shared across slots. The KV-append (step 7) and attention (step 8)
/// are split per slot because each slot owns its own KV cache and
/// query history.
/// Layout:
/// - `x_in` / `delta_out` are F16 `[N, hidden]`, row `s` belongs to
/// slot `s` whose index in the per-rank session/cache arrays is
/// also `s`.
/// - `slot_caches[s]` is the layer-local KV cache for slot `s`. All
/// must be `LayerCache::FullAttn` with identical `n_kv_heads`
/// and `head_dim`.
/// - `slot_positions[s]` is the cache tail for slot `s` *before* this
/// token is appended (i.e. the position the new K/V row writes to).
/// - `scratch` is a single shared per-rank `FullAttnPrefillScratch`
/// sized for `max_tokens >= N` — the same buffers that prefill uses,
/// reused as the [N, *] batched workspace.
pub fn forward_full_attn_layer_decode_batched(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    slot_caches: &mut [&mut LayerCache],
    scratch: &mut FullAttnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    slot_positions: &[usize],
) -> Result<()> {
    let n_tokens = slot_positions.len();
    if n_tokens == 0 {
        bail!("forward_full_attn_layer_decode_batched called with 0 slots");
    }
    if n_tokens != slot_caches.len() {
        bail!(
            "slot count mismatch: positions={n_tokens}, caches={}",
            slot_caches.len()
        );
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "n_tokens={n_tokens} > scratch.max_tokens={}; size scratch for ≥N",
            scratch.max_tokens
        );
    }

    let crate::weights::AttnWeights::FullAttn(weights) = &layer_weights.attn else {
        bail!(
            "layer {} weights are not FullAttn (i2-A1 only handles gated full-attn)",
            layer_weights.layer_idx
        );
    };

    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let rope = &cfg.rope;
    let attn_norm = &layer_weights.attn_norm;

    // Steps 1-6: identical to forward_full_attn_prefill body. These all
    // operate on [N, hidden] / [N, q_width] / [N, kv_width] and don't
    // depend on a single contiguous KV cache, so the existing kernels
    // batch across slots naturally.

    // 1. rmsnorm + Q8_1 quantise (both layouts).
    rmsnorm_f16(
        ops, stream, x_in, attn_norm.ptr, scratch.x_norm_f16,
        n_tokens, hidden, cfg.rms_norm_eps,
    )
    .context("batched-decode attn_norm")?;
    quantize_f16_q8_1(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden,
    )
    .context("batched-decode x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens,
    )
    .context("batched-decode x_norm → Q8_1 (MMQ DS4)")?;

    // 2. Q|gate fused projection.
    let dtype_q = qdtype_of(weights.attn_q.dtype)?;
    let (q_rows, q_k) = mat_shape(&weights.attn_q)?;
    if q_rows != 2 * n_heads * head_dim || q_k != hidden {
        bail!(
            "attn_q shape [{q_rows}, {q_k}] != expected [{}, {}]",
            2 * n_heads * head_dim, hidden
        );
    }
    qmatmul(
        ops, stream, weights.attn_q.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.mmvq_f32,
        n_tokens, q_k, q_rows, dtype_q,
    )
    .context("batched-decode qmatmul attn_q")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.q_fused_f16, n_tokens * q_rows,
    )
    .context("batched-decode cast attn_q → f16")?;

    // 3. Split Q | gate.
    split_q_gate_f16(
        ops, stream, scratch.q_fused_f16,
        scratch.q_f16, scratch.gate_f16,
        n_tokens, n_heads, head_dim,
    )
    .context("batched-decode split_q_gate")?;

    // 4. K / V projections.
    let dtype_k = qdtype_of(weights.attn_k.dtype)?;
    let (k_rows, k_k) = mat_shape(&weights.attn_k)?;
    if k_rows != kv_width || k_k != hidden {
        bail!(
            "attn_k shape [{k_rows}, {k_k}] != expected [{kv_width}, {hidden}]"
        );
    }
    qmatmul(
        ops, stream, weights.attn_k.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.mmvq_f32,
        n_tokens, k_k, k_rows, dtype_k,
    )
    .context("batched-decode qmatmul attn_k")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.k_f16, n_tokens * k_rows,
    )
    .context("batched-decode cast attn_k → f16")?;

    let dtype_v = qdtype_of(weights.attn_v.dtype)?;
    let (v_rows, v_k) = mat_shape(&weights.attn_v)?;
    if v_rows != kv_width || v_k != hidden {
        bail!(
            "attn_v shape [{v_rows}, {v_k}] != expected [{kv_width}, {hidden}]"
        );
    }
    qmatmul(
        ops, stream, weights.attn_v.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.mmvq_f32,
        n_tokens, v_k, v_rows, dtype_v,
    )
    .context("batched-decode qmatmul attn_v")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.v_f16, n_tokens * v_rows,
    )
    .context("batched-decode cast attn_v → f16")?;

    // 5. Per-head Q/K rmsnorm.
    let q_norm_dim = weights
        .attn_q_norm.dims.first().copied()
        .context("attn_q_norm missing dim")? as usize;
    if q_norm_dim != head_dim {
        bail!("attn_q_norm dim {q_norm_dim} != head_dim {head_dim}");
    }
    rmsnorm_f16(
        ops, stream, scratch.q_f16, weights.attn_q_norm.ptr, scratch.q_f16,
        n_tokens * n_heads, head_dim, cfg.rms_norm_eps,
    )
    .context("batched-decode attn_q_norm")?;
    rmsnorm_f16(
        ops, stream, scratch.k_f16, weights.attn_k_norm.ptr, scratch.k_f16,
        n_tokens * n_kv_heads, head_dim, cfg.rms_norm_eps,
    )
    .context("batched-decode attn_k_norm")?;

    // 6. RoPE on Q / K with per-slot positions.
    upload_positions_arbitrary(device, stream, scratch, slot_positions)?;
    rope_neox_partial_f16(
        ops, stream, scratch.q_f16, scratch.positions, rope.freq_base,
        n_tokens, n_heads, head_dim, rope.rotated_dims,
    )
    .context("batched-decode rope Q")?;
    rope_neox_partial_f16(
        ops, stream, scratch.k_f16, scratch.positions, rope.freq_base,
        n_tokens, n_kv_heads, head_dim, rope.rotated_dims,
    )
    .context("batched-decode rope K")?;

    // 7. Per-slot KV append. **#275 fix**: write at `current_tokens`
    // (the cache tail) rather than `slot_positions[s]` (which is
    // `prompt_ids.len() + step` = off by 1). Matches legacy
    // `kv_cache.append()` semantics. F16-only (FullAttn cache).
    // **#266c**: in the same pass, populate the per-slot pointer/
    // length tables consumed by `attention_decode_f16_batched`.
    let kv_per_token_bytes = kv_width * 2;
    for (s, cache) in slot_caches.iter_mut().enumerate() {
        let LayerCache::FullAttn(kv) = cache else {
            bail!(
                "i2-A1 batched decode: slot {s} cache is not FullAttn \
                 (Q8 KV path uses sequential single-slot decode)"
            );
        };
        let write_pos = kv.current_tokens();
        let k_src = scratch.k_f16.offset_bytes(s * kv_per_token_bytes);
        let v_src = scratch.v_f16.offset_bytes(s * kv_per_token_bytes);
        let k_dst = kv.k_buffer().offset_bytes(write_pos * kv_per_token_bytes);
        let v_dst = kv.v_buffer().offset_bytes(write_pos * kv_per_token_bytes);
        // SAFETY: src/dst are valid; write_pos < max_seq_len is bounded
        // by the cache.
        unsafe {
            device.memcpy_async(
                stream, CopyDirection::DeviceToDevice,
                k_dst, k_src, kv_per_token_bytes,
            )?;
            device.memcpy_async(
                stream, CopyDirection::DeviceToDevice,
                v_dst, v_src, kv_per_token_bytes,
            )?;
        }
        kv.bump_tail(1)
            .map_err(|e| anyhow::anyhow!("slot {s} bump_tail: {e}"))?;
        // Populate the slot tables for the batched-attention launch
        // below. n_tokens_kv reads post-bump (= includes just-written row).
        scratch.slot_k_ptrs_host[s] = kv.k_buffer().as_usize() as u64;
        scratch.slot_v_ptrs_host[s] = kv.v_buffer().as_usize() as u64;
        scratch.slot_n_tokens_kv_host[s] = kv.current_tokens() as i32;
    }

    // 8. Single-launch batched attention over all N slots
    // (**#266c** — replaces the per-slot loop).
    let scale = (head_dim as f32).sqrt().recip();
    // SAFETY: each `slot_*_host[..n_tokens]` is a Vec<u64|i32> with
    // stable address; the corresponding device buffer is sized to
    // `max_tokens` ≥ n_tokens; HtoD bytes are bounded by the slice.
    unsafe {
        device.memcpy_async(
            stream, CopyDirection::HostToDevice,
            scratch.slot_k_ptrs,
            DevicePtr(scratch.slot_k_ptrs_host.as_ptr() as usize),
            n_tokens * 8,
        )?;
        device.memcpy_async(
            stream, CopyDirection::HostToDevice,
            scratch.slot_v_ptrs,
            DevicePtr(scratch.slot_v_ptrs_host.as_ptr() as usize),
            n_tokens * 8,
        )?;
        device.memcpy_async(
            stream, CopyDirection::HostToDevice,
            scratch.slot_n_tokens_kv,
            DevicePtr(scratch.slot_n_tokens_kv_host.as_ptr() as usize),
            n_tokens * 4,
        )?;
    }
    flambeau_ops::hip::attention::attention_decode_f16_batched(
        ops,
        stream,
        scratch.q_f16,
        scratch.slot_k_ptrs,
        scratch.slot_v_ptrs,
        scratch.attn_out_f16,
        scratch.slot_n_tokens_kv,
        n_heads,
        n_kv_heads,
        head_dim,
        n_tokens,
        scale,
    )
    .context("batched-decode attention (i2-E #266c)")?;

    // 9. Sigmoid-gate over [N, q_width].
    sigmoid_mul_f16(
        ops, stream,
        scratch.gate_f16, scratch.attn_out_f16, scratch.gated_out_f16,
        n_tokens * q_width,
    )
    .context("batched-decode post-attn sigmoid-gate")?;

    // 10. Quantise gated_out to both Q8_1 layouts for output proj.
    quantize_f16_q8_1(
        ops, stream,
        scratch.gated_out_f16, scratch.gated_q8_1, n_tokens * q_width,
    )
    .context("batched-decode quantise gated → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops, stream,
        scratch.gated_out_f16, scratch.gated_q8_1_mmq, q_width, n_tokens,
    )
    .context("batched-decode quantise gated → Q8_1 (MMQ DS4)")?;

    // 11. Output projection over [N, hidden].
    let dtype_o = qdtype_of(weights.attn_output.dtype)?;
    let (o_rows, o_k) = mat_shape(&weights.attn_output)?;
    if o_rows != hidden || o_k != q_width {
        bail!(
            "attn_output shape [{o_rows}, {o_k}] != expected [{hidden}, {q_width}]"
        );
    }
    qmatmul(
        ops, stream, weights.attn_output.ptr,
        scratch.gated_q8_1, scratch.gated_q8_1_mmq, scratch.mmvq_f32,
        n_tokens, o_k, o_rows, dtype_o,
    )
    .context("batched-decode qmatmul attn_output")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, delta_out, n_tokens * hidden,
    )
    .context("batched-decode cast attn_output → f16")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// 8.b-i1 — dense attention (qwen3moe family).
// Differs from `forward_full_attn_*` above:
// - Q projection is plain `[n_heads*head_dim, hidden]` (not 2× fused with
// an input gate).
// - No `split_q_gate`.
// - No post-attn `sigmoid_mul` — `attn_out_f16` is quantised and fed
// directly into the output projection.
// - Q/K/V biases can be present (added after each matmul); Qwen3-Coder-30B
// has none, but the path supports optional biases via a runtime bail if
// the caller passes them (not yet implemented — asserts None).
// Reuses `FullAttnScratch` / `FullAttnPrefillScratch` — `q_fused_f16` and
// `gate_f16` inside them go unused on this path (~128 KiB dead per layer,
// fine at 4.11 GiB/rank for Qwen3-Coder).
// ---------------------------------------------------------------------------

pub fn forward_dense_attn_decode<L: CacheLayout>(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &DenseAttnWeights,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    position: usize,
    slots: Option<AttnDecodeSlots>,
) -> Result<()> {
    let block = build_dense_attn_block(attn_norm, weights, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_decode(
        &hipops,
        device,
        stream,
        x_in,
        delta_out,
        kv_cache,
        &mut scratch.view_mut(),
        position,
        slots,
    )
}

/// Route a `LayerCache` entry through the dense-attn forward, pulling the
/// correct `KvCache` out of the enum. Fails if the layer is GDN.
pub fn forward_dense_attn_layer_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    position: usize,
    slots: Option<AttnDecodeSlots>,
) -> Result<()> {
    let crate::weights::AttnWeights::Dense(d) = &layer_weights.attn else {
        bail!(
            "layer {} weights are not Dense variant (dense attn path)",
            layer_weights.layer_idx
        );
    };
    match layer_cache {
        LayerCache::FullAttn(kv) => forward_dense_attn_decode(
            ops, stream, device, cfg,
            &layer_weights.attn_norm,
            d, kv, scratch, x_in, delta_out,
            position, slots,
        ),
        LayerCache::FullAttnQ8(kv) => forward_dense_attn_decode(
            ops, stream, device, cfg,
            &layer_weights.attn_norm,
            d, kv, scratch, x_in, delta_out,
            position, slots,
        ),
        LayerCache::Gdn(_) => bail!(
            "layer {} is not a full-attn layer (cache variant mismatch)",
            layer_weights.layer_idx
        ),
    }
}

pub fn forward_dense_attn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &DenseAttnWeights,
    kv_cache: &mut KvCache<flambeau_runtime::F16Contig, HipDevice>,
    scratch: &mut FullAttnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
    slots: Option<AttnPrefillSlots>,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_dense_attn_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_dense_attn_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    let block = build_dense_attn_block(attn_norm, weights, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_prefill(
        &hipops,
        device,
        stream,
        x_in,
        delta_out,
        kv_cache,
        &mut scratch.view_mut(),
        n_tokens,
        start_position,
        slots,
    )
}
