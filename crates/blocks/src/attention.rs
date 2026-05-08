//! `StandardAttention` — Qwen3-style full attention block.
//!
//! Composes the attention pipeline (RMSNorm + fused Q|gate + K/V proj +
//! per-head Q/K norm + partial NeoX RoPE + KV append + decode/prefill
//! attention + sigmoid gate + output proj) over the `Ops` trait. The
//! block is HIP-flavored for V1 — it takes `&HipDevice` + `&HipStream`
//! alongside `&O: &impl Ops` because the position upload and KV append
//! are not yet on the trait. Kernel launches always go through `O`.
//!
//! Graph-capture slot variants are reachable through the optional
//! `AttnDecodeSlots` / `AttnPrefillSlots` args on `forward_decode` and
//! `forward_prefill`. When `Some`, the block calls slot-tagged
//! variants of the underlying kernels (`attention_decode_f16_slots`,
//! `attention_prefill_f16_slots`, `kv_cache_append_hip_slot`); the
//! caller updates each slot per replay via `HipGraphExec::set_slot`.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    kv_cache_append_hip_slot, HipDevice, HipStream, MemcpySlot, ScalarSlot,
};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::Ops;
use flambeau_runtime::{CacheLayout, F16Contig, KvCache, Q8Contig};

use flambeau_core::op::QDtype;

/// Graph-capture slot bundle for one decode call. When passed to
/// `StandardAttention::forward_decode` (or via `AttnBlock::Standard`),
/// the block calls slot-tagged kernels so the recorder can bind each
/// slot to a per-replay value via `HipGraphExec::set_slot`.
///
/// Only `AttnBlock::Standard` accepts slots in V1; the kernels for
/// `AttnBlock::DeltaNet` are not graph-capture-aware. Graph-captured
/// callers leave GDN layers uncaptured.
#[derive(Clone, Copy, Debug)]
pub struct AttnDecodeSlots {
    /// Tags `n_tokens_kv` at `attention_decode_f16`. Value per replay
    /// = cache tail post-append (`position + 1`).
    pub n_tokens_kv_slot: ScalarSlot,
    /// Tags the K-tensor dst of `kv_cache.append`.
    pub k_append_slot: MemcpySlot,
    /// Tags the V-tensor dst of `kv_cache.append`.
    pub v_append_slot: MemcpySlot,
}

/// Graph-capture slot bundle for one prefill chunk.
#[derive(Clone, Copy, Debug)]
pub struct AttnPrefillSlots {
    /// Tags `n_k_tokens` at `attention_prefill_f16`.
    pub n_k_slot: ScalarSlot,
    /// Tags `q_offset` at `attention_prefill_f16`.
    pub q_off_slot: ScalarSlot,
    /// Tags the K-tensor dst of `kv_cache.append`.
    pub k_append_slot: MemcpySlot,
    /// Tags the V-tensor dst of `kv_cache.append`.
    pub v_append_slot: MemcpySlot,
}

/// Borrowed handle to one model weight tensor on device. The block
/// holds these instead of owning the underlying alloc — the caller
/// (typically the model crate) owns the buffer and decides its
/// lifetime.
#[derive(Copy, Clone)]
pub struct WeightHandle {
    pub ptr: DevicePtr,
    pub dtype: QDtype,
    /// `[out_rows, in_cols]` for matmul weights. Norm weights are 1-D;
    /// callers store their length in the owning block's shape config.
    pub dims: [usize; 2],
}

/// Borrowed view over a caller-owned attention decode scratch. The
/// `'a` lifetime ties the view to the underlying owning scratch (in
/// V1 this is `qwen3-moe`'s `FullAttnScratch`); call sites build a
/// fresh view per forward call via `view_mut()`. All `DevicePtr`
/// fields are `Copy`; only `positions_host` actually borrows mutably.
pub struct StandardAttentionDecodeScratch<'a> {
    pub x_q8_1: DevicePtr,         // Q8_1 blocks [hidden / 32]
    pub mmvq_f32: DevicePtr,       // F32 [max(2*H*D, H_kv*D, hidden)]
    pub q_fused_f16: DevicePtr,    // F16 [2 * n_heads * head_dim]
    pub q_f16: DevicePtr,          // F16 [n_heads * head_dim]
    pub gate_f16: DevicePtr,       // F16 [n_heads * head_dim]
    pub k_f16: DevicePtr,          // F16 [n_kv_heads * head_dim]
    pub v_f16: DevicePtr,          // F16 [n_kv_heads * head_dim]
    /// Q8_0 staging for `KvCache<Q8Contig>`. Sized for
    /// `n_kv_heads * head_dim / 32` Q8_0 blocks (18 B each). Unused on
    /// the F16-KV path.
    pub k_q8_0: DevicePtr,
    pub v_q8_0: DevicePtr,
    pub attn_out_f16: DevicePtr,
    pub gated_out_f16: DevicePtr,
    pub positions: DevicePtr,      // i32 [1]
    /// Persistent host-side 1-slot position backing — keeps the HtoD
    /// memcpy source stable so the call can drop its sync.
    pub positions_host: &'a mut [i32],
    /// Split-K (flash-decoding) partials for long contexts.
    pub splitk_partials_m: DevicePtr, // F32 [n_heads * MAX_SPLITK_CHUNKS]
    pub splitk_partials_s: DevicePtr, // F32 [n_heads * MAX_SPLITK_CHUNKS]
    pub splitk_partials_o: DevicePtr, // F32 [n_heads * MAX_SPLITK_CHUNKS * head_dim]
}

/// Borrowed view over a caller-owned attention prefill scratch.
pub struct StandardAttentionPrefillScratch<'a> {
    pub max_tokens: usize,
    pub x_norm_f16: DevicePtr,
    pub x_q8_1: DevicePtr,
    pub x_q8_1_mmq: DevicePtr,
    pub mmvq_f32: DevicePtr,
    pub q_fused_f16: DevicePtr,
    pub q_f16: DevicePtr,
    pub gate_f16: DevicePtr,
    pub k_f16: DevicePtr,
    pub v_f16: DevicePtr,
    pub attn_out_f16: DevicePtr,
    pub gated_out_f16: DevicePtr,
    pub positions: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub gated_q8_1_mmq: DevicePtr,
    pub positions_host: &'a mut [i32],
}

/// Split-K partial budget — 32 chunks × 512 tokens covers any decode
/// context up to 16 384 tokens. Bump alongside the dispatch threshold
/// if context ever exceeds. Mirrors qwen3-moe's `MAX_SPLITK_CHUNKS`.
pub const MAX_SPLITK_CHUNKS: usize = 32;

/// Qwen3-style full attention block. Two shapes share one type:
///
/// * `gated = true` (qwen35moe / qwen36moe): Q+gate fused projection
///   into a `[2 * n_heads * head_dim, hidden]` weight, post-attention
///   sigmoid gate `gated_out = sigmoid(gate) * attn_out`. Used by
///   Qwen3.5/3.6 dense + MoE arches.
/// * `gated = false` (qwen3moe): plain Q projection into a
///   `[n_heads * head_dim, hidden]` weight, no output gate
///   (`attn_out` feeds the output projection directly).
///
/// The block carries no backend-specific state; methods take
/// `ops: &O: &impl Ops` at call time.
pub struct StandardAttention {
    pub attn_q: WeightHandle,
    pub attn_k: WeightHandle,        // [n_kv_heads*head_dim, hidden]
    pub attn_v: WeightHandle,        // [n_kv_heads*head_dim, hidden]
    pub attn_output: WeightHandle,   // [hidden, n_heads*head_dim]
    pub attn_norm_w: DevicePtr,      // F16 [hidden]
    pub attn_q_norm_w: DevicePtr,    // F16 [head_dim]
    pub attn_k_norm_w: DevicePtr,    // F16 [head_dim]
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub rope_rotated_dims: usize,
    /// `true` when Q+gate are fused (qwen35moe-style) and the output
    /// passes through `sigmoid(gate) * attn_out`. `false` for the
    /// plain-Q (qwen3moe-style) path.
    pub gated: bool,
}

impl StandardAttention {
    /// Construct a new block. Q-weight rows are
    /// `2 * n_heads * head_dim` when `gated` and `n_heads * head_dim`
    /// otherwise; the constructor asserts the matmul shapes match.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attn_q: WeightHandle,
        attn_k: WeightHandle,
        attn_v: WeightHandle,
        attn_output: WeightHandle,
        attn_norm_w: DevicePtr,
        attn_q_norm_w: DevicePtr,
        attn_k_norm_w: DevicePtr,
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f32,
        rope_freq_base: f32,
        rope_rotated_dims: usize,
        gated: bool,
    ) -> Result<Self> {
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let expected_q_rows = if gated { 2 * q_width } else { q_width };

        if attn_q.dims != [expected_q_rows, hidden] {
            bail!(
                "attn_q dims {:?} != expected [{}, {}]",
                attn_q.dims,
                expected_q_rows,
                hidden
            );
        }
        if attn_k.dims != [kv_width, hidden] {
            bail!(
                "attn_k dims {:?} != expected [{}, {}]",
                attn_k.dims,
                kv_width,
                hidden
            );
        }
        if attn_v.dims != [kv_width, hidden] {
            bail!(
                "attn_v dims {:?} != expected [{}, {}]",
                attn_v.dims,
                kv_width,
                hidden
            );
        }
        if attn_output.dims != [hidden, q_width] {
            bail!(
                "attn_output dims {:?} != expected [{}, {}]",
                attn_output.dims,
                hidden,
                q_width
            );
        }
        Ok(Self {
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_norm_w,
            attn_q_norm_w,
            attn_k_norm_w,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_freq_base,
            rope_rotated_dims,
            gated,
        })
    }

    /// Single-token decode through the attention block. The K/V row
    /// derived from `x_in` is appended to `kv_cache` before the
    /// attention call, so `current_tokens = position + 1` post-call.
    pub fn forward_decode<L: CacheLayout, O: Ops>(
        &self,
        ops: &O,
        device: &HipDevice,
        stream: &HipStream,
        x_in: DevicePtr,
        delta_out: DevicePtr,
        kv_cache: &mut KvCache<L, HipDevice>,
        scratch: &mut StandardAttentionDecodeScratch<'_>,
        position: usize,
        slots: Option<AttnDecodeSlots>,
    ) -> Result<()> {
        let hidden = self.hidden;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let q_proj_rows = if self.gated { 2 * q_width } else { q_width };

        // 1. Fused RMSNorm(x_in) + Q8_1 quantise.
        ops.rmsnorm_quant_q8_1(
            x_in,
            self.attn_norm_w,
            scratch.x_q8_1,
            1,
            hidden,
            self.rms_norm_eps,
        )
        .context("attn_norm + quant")?;

        // 2. Q projection. `gated`: fused Q+gate at `2 * q_width`
        // rows, cast into the fused F16 buffer for the split below.
        // Plain: Q-only at `q_width` rows, cast directly into q_f16
        // (no split, no gate).
        ops.mmvq(
            self.attn_q.ptr,
            scratch.x_q8_1,
            scratch.mmvq_f32,
            q_proj_rows,
            hidden,
            self.attn_q.dtype,
        )
        .context("mmvq attn_q")?;
        if self.gated {
            ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.q_fused_f16, q_proj_rows)
                .context("cast attn_q → f16")?;
            // 3. Split fused [Q | gate] → q_f16, gate_f16.
            ops.split_q_gate_f16(
                scratch.q_fused_f16,
                scratch.q_f16,
                scratch.gate_f16,
                1,
                n_heads,
                head_dim,
            )
            .context("split_q_gate")?;
        } else {
            ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.q_f16, q_proj_rows)
                .context("cast attn_q → f16")?;
        }

        // 4+5. K and V projections. Fuse to one launch when both Q8_0.
        let fuse_kv = self.attn_k.dtype == QDtype::Q8_0 && self.attn_v.dtype == QDtype::Q8_0;
        if fuse_kv {
            let v_f32_offset = scratch.mmvq_f32.offset_bytes(kv_width * 4);
            ops.mmvq_q8_0_gate_up(
                self.attn_k.ptr,
                self.attn_v.ptr,
                scratch.x_q8_1,
                scratch.mmvq_f32,
                v_f32_offset,
                kv_width,
                kv_width,
                hidden,
            )
            .context("attn_k + attn_v fused mmvq_q8_0")?;
            ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.k_f16, kv_width)
                .context("cast attn_k → f16")?;
            ops.cast_f32_to_f16(v_f32_offset, scratch.v_f16, kv_width)
                .context("cast attn_v → f16")?;
        } else {
            ops.mmvq(
                self.attn_k.ptr,
                scratch.x_q8_1,
                scratch.mmvq_f32,
                kv_width,
                hidden,
                self.attn_k.dtype,
            )
            .context("mmvq attn_k")?;
            ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.k_f16, kv_width)
                .context("cast attn_k → f16")?;
            ops.mmvq(
                self.attn_v.ptr,
                scratch.x_q8_1,
                scratch.mmvq_f32,
                kv_width,
                hidden,
                self.attn_v.dtype,
            )
            .context("mmvq attn_v")?;
            ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.v_f16, kv_width)
                .context("cast attn_v → f16")?;
        }

        // 6. Per-head Q / K rmsnorm.
        ops.rmsnorm_f16(
            scratch.q_f16,
            self.attn_q_norm_w,
            scratch.q_f16,
            n_heads,
            head_dim,
            self.rms_norm_eps,
        )
        .context("attn_q_norm")?;
        ops.rmsnorm_f16(
            scratch.k_f16,
            self.attn_k_norm_w,
            scratch.k_f16,
            n_kv_heads,
            head_dim,
            self.rms_norm_eps,
        )
        .context("attn_k_norm")?;

        // 7. Upload position (1-slot) and apply RoPE on Q + K.
        scratch.positions_host[0] = position as i32;
        // SAFETY: scratch.positions has 4 valid bytes; positions_host is a
        // persistent Vec on the scratch whose lifetime exceeds the copy.
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
            self.rope_freq_base,
            1,
            n_heads,
            head_dim,
            self.rope_rotated_dims,
        )
        .context("rope Q")?;
        ops.rope_neox_partial_f16(
            scratch.k_f16,
            scratch.positions,
            self.rope_freq_base,
            1,
            n_kv_heads,
            head_dim,
            self.rope_rotated_dims,
        )
        .context("rope K")?;

        // 8. Append K, V into the KV cache. Slot path uses
        // `kv_cache_append_hip_slot` so the K / V dst memcpys are
        // graph-recorded with their dst tagged for per-replay
        // retargeting. Slots are F16-only — Q8 KV with slots bails.
        let kv_layout = L::NAME;
        if let Some(s) = slots {
            if kv_layout != F16Contig::NAME {
                bail!(
                    "StandardAttention::forward_decode: graph-capture slots are F16-only; \
                     KV layout {kv_layout} not supported under capture"
                );
            }
            // SAFETY: scratch.k_f16 / v_f16 are contiguous F16
            // [n_kv_heads, head_dim] on `device`. The slot variant
            // tags both memcpys' dst arg so the recorder can retarget
            // them per replay via `HipGraphExec::set_memcpy_slot`.
            unsafe {
                kv_cache_append_hip_slot(
                    kv_cache,
                    device,
                    stream,
                    scratch.k_f16,
                    scratch.v_f16,
                    1,
                    s.k_append_slot,
                    s.v_append_slot,
                )?;
            }
        } else if kv_layout == Q8Contig::NAME {
            ops.quantize_f16_q8_0(scratch.k_f16, scratch.k_q8_0, kv_width)
                .context("quantize attn_k → q8_0")?;
            ops.quantize_f16_q8_0(scratch.v_f16, scratch.v_q8_0, kv_width)
                .context("quantize attn_v → q8_0")?;
            // SAFETY: k_q8_0/v_q8_0 hold (n_kv_heads * head_dim / 32) Q8_0
            // blocks (18 B each). KvCache<Q8Contig>::append expects exactly
            // `n_new * n_heads * Q8Contig::bytes_per_row(head_dim)` bytes,
            // matching the buffers above for n_new=1.
            unsafe {
                kv_cache
                    .append(device, stream, scratch.k_q8_0, scratch.v_q8_0, 1)
                    .map_err(|e| anyhow::anyhow!("kv_cache.append (q8): {e}"))?;
            }
        } else {
            // SAFETY: scratch.k_f16 / v_f16 are contiguous F16
            // [n_kv_heads, head_dim].
            unsafe {
                kv_cache
                    .append(device, stream, scratch.k_f16, scratch.v_f16, 1)
                    .map_err(|e| anyhow::anyhow!("kv_cache.append: {e}"))?;
            }
        }

        // 9. Attention decode. Split-K + Q8 KV bypass the slot path
        // (the slot-tagged kernels are F16 single-pass only). Slot
        // callers run F16 KV decodes through `attention_decode_f16_slots`
        // so `n_tokens_kv` is recorded as a tagged kernel arg.
        let n_tokens_kv = kv_cache.current_tokens();
        let scale = (head_dim as f32).sqrt().recip();
        let use_splitk = slots.is_none() && n_tokens_kv > 256;
        if let Some(s) = slots {
            // F16 KV slots-aware decode. Already validated above.
            ops.attention_decode_f16_slots(
                scratch.q_f16,
                kv_cache.k_buffer(),
                kv_cache.v_buffer(),
                scratch.attn_out_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                n_tokens_kv,
                scale,
                Some(s.n_tokens_kv_slot),
            )
            .context("attention_decode_f16_slots")?;
        } else if use_splitk {
            let chunk_size = flambeau_ops::hip::attention::splitk_chunk_size(n_tokens_kv);
            let n_chunks = n_tokens_kv.div_ceil(chunk_size);
            debug_assert!(
                n_chunks <= MAX_SPLITK_CHUNKS,
                "split-K n_chunks={n_chunks} exceeds scratch budget MAX={MAX_SPLITK_CHUNKS}"
            );
            if kv_layout == Q8Contig::NAME {
                ops.attention_decode_q8_kv_splitk(
                    scratch.q_f16,
                    kv_cache.k_buffer(),
                    kv_cache.v_buffer(),
                    scratch.attn_out_f16,
                    scratch.splitk_partials_m,
                    scratch.splitk_partials_s,
                    scratch.splitk_partials_o,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    n_tokens_kv,
                    chunk_size,
                    scale,
                )
                .context("attention_decode_q8_kv_splitk")?;
            } else {
                ops.attention_decode_f16_splitk(
                    scratch.q_f16,
                    kv_cache.k_buffer(),
                    kv_cache.v_buffer(),
                    scratch.attn_out_f16,
                    scratch.splitk_partials_m,
                    scratch.splitk_partials_s,
                    scratch.splitk_partials_o,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    n_tokens_kv,
                    chunk_size,
                    scale,
                )
                .context("attention_decode_f16_splitk")?;
            }
        } else if kv_layout == Q8Contig::NAME {
            ops.attention_decode_q8_kv(
                scratch.q_f16,
                kv_cache.k_buffer(),
                kv_cache.v_buffer(),
                scratch.attn_out_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                n_tokens_kv,
                scale,
            )
            .context("attention_decode_q8_kv")?;
        } else if kv_layout == F16Contig::NAME {
            ops.attention_decode_f16(
                scratch.q_f16,
                kv_cache.k_buffer(),
                kv_cache.v_buffer(),
                scratch.attn_out_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                n_tokens_kv,
                scale,
            )
            .context("attention_decode_f16")?;
        } else {
            bail!("StandardAttention: unsupported KV layout {kv_layout}");
        }

        // 10. Post-attn sigmoid gate (gated path only). Plain path
        // feeds attn_out_f16 straight into the output projection.
        let post_attn_f16 = if self.gated {
            ops.sigmoid_mul_f16(
                scratch.gate_f16,
                scratch.attn_out_f16,
                scratch.gated_out_f16,
                q_width,
            )
            .context("post-attn sigmoid-gate")?;
            scratch.gated_out_f16
        } else {
            scratch.attn_out_f16
        };

        // 11. Quantise the post-attn F16 to Q8_1 for the output proj.
        ops.quantize_f16_q8_1(post_attn_f16, scratch.x_q8_1, q_width)
            .context("quantize post-attn → Q8_1")?;

        // 12. Output projection [hidden, q_width].
        ops.mmvq(
            self.attn_output.ptr,
            scratch.x_q8_1,
            scratch.mmvq_f32,
            hidden,
            q_width,
            self.attn_output.dtype,
        )
        .context("mmvq attn_output")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, delta_out, hidden)
            .context("cast attn_output → f16")?;

        Ok(())
    }

    /// Multi-token prefill. `start_position` is the cache tail length
    /// before this chunk's K/V are appended.
    pub fn forward_prefill<L: CacheLayout, O: Ops>(
        &self,
        ops: &O,
        device: &HipDevice,
        stream: &HipStream,
        x_in: DevicePtr,
        delta_out: DevicePtr,
        kv_cache: &mut KvCache<L, HipDevice>,
        scratch: &mut StandardAttentionPrefillScratch<'_>,
        n_tokens: usize,
        start_position: usize,
        slots: Option<AttnPrefillSlots>,
    ) -> Result<()> {
        if n_tokens == 0 {
            bail!("forward_prefill called with n_tokens = 0");
        }
        if n_tokens > scratch.max_tokens {
            bail!(
                "forward_prefill: n_tokens={n_tokens} > scratch.max_tokens={}; caller must chunk",
                scratch.max_tokens
            );
        }

        let hidden = self.hidden;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let q_proj_rows = if self.gated { 2 * q_width } else { q_width };

        // 1. RMSNorm + dual Q8_1 quantisation (standard + DS4-MMQ).
        ops.rmsnorm_f16(
            x_in,
            self.attn_norm_w,
            scratch.x_norm_f16,
            n_tokens,
            hidden,
            self.rms_norm_eps,
        )
        .context("prefill attn_norm")?;
        ops.quantize_f16_q8_1(scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden)
            .context("prefill x_norm → Q8_1 (std)")?;
        ops.quantize_f16_q8_1_mmq(scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens)
            .context("prefill x_norm → Q8_1 (MMQ DS4)")?;

        // 2. Q projection. `gated`: fused Q+gate at `2*q_width` rows.
        // Plain: Q-only at `q_width` rows.
        ops.qmatmul(
            self.attn_q.ptr,
            scratch.x_q8_1,
            scratch.x_q8_1_mmq,
            scratch.mmvq_f32,
            n_tokens,
            hidden,
            q_proj_rows,
            self.attn_q.dtype,
        )
        .context("prefill qmatmul attn_q")?;
        if self.gated {
            ops.cast_f32_to_f16(
                scratch.mmvq_f32,
                scratch.q_fused_f16,
                n_tokens * q_proj_rows,
            )
            .context("prefill cast attn_q → f16")?;
            // 3. Split Q | gate.
            ops.split_q_gate_f16(
                scratch.q_fused_f16,
                scratch.q_f16,
                scratch.gate_f16,
                n_tokens,
                n_heads,
                head_dim,
            )
            .context("prefill split_q_gate")?;
        } else {
            ops.cast_f32_to_f16(
                scratch.mmvq_f32,
                scratch.q_f16,
                n_tokens * q_proj_rows,
            )
            .context("prefill cast attn_q → f16")?;
        }

        // 4. K projection.
        ops.qmatmul(
            self.attn_k.ptr,
            scratch.x_q8_1,
            scratch.x_q8_1_mmq,
            scratch.mmvq_f32,
            n_tokens,
            hidden,
            kv_width,
            self.attn_k.dtype,
        )
        .context("prefill qmatmul attn_k")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.k_f16, n_tokens * kv_width)
            .context("prefill cast attn_k → f16")?;

        // 5. V projection.
        ops.qmatmul(
            self.attn_v.ptr,
            scratch.x_q8_1,
            scratch.x_q8_1_mmq,
            scratch.mmvq_f32,
            n_tokens,
            hidden,
            kv_width,
            self.attn_v.dtype,
        )
        .context("prefill qmatmul attn_v")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.v_f16, n_tokens * kv_width)
            .context("prefill cast attn_v → f16")?;

        // 6. Per-head Q / K rmsnorm.
        ops.rmsnorm_f16(
            scratch.q_f16,
            self.attn_q_norm_w,
            scratch.q_f16,
            n_tokens * n_heads,
            head_dim,
            self.rms_norm_eps,
        )
        .context("prefill attn_q_norm")?;
        ops.rmsnorm_f16(
            scratch.k_f16,
            self.attn_k_norm_w,
            scratch.k_f16,
            n_tokens * n_kv_heads,
            head_dim,
            self.rms_norm_eps,
        )
        .context("prefill attn_k_norm")?;

        // 7. Upload positions [start_position, start_position + n_tokens).
        for i in 0..n_tokens {
            scratch.positions_host[i] = (start_position + i) as i32;
        }
        // SAFETY: scratch.positions has at least n_tokens * 4 valid bytes;
        // positions_host is a persistent Vec on the scratch with stable
        // address for the duration of this call.
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
            self.rope_freq_base,
            n_tokens,
            n_heads,
            head_dim,
            self.rope_rotated_dims,
        )
        .context("prefill rope Q")?;
        ops.rope_neox_partial_f16(
            scratch.k_f16,
            scratch.positions,
            self.rope_freq_base,
            n_tokens,
            n_kv_heads,
            head_dim,
            self.rope_rotated_dims,
        )
        .context("prefill rope K")?;

        // 8. Append K / V to the KV cache. Slot path uses the
        // graph-capture-tagged variant; F16-only.
        let kv_layout = L::NAME;
        if let Some(s) = slots {
            if kv_layout != F16Contig::NAME {
                bail!(
                    "StandardAttention::forward_prefill: graph-capture slots are F16-only; \
                     KV layout {kv_layout} not supported under capture"
                );
            }
            // SAFETY: scratch.k_f16 / v_f16 hold n_tokens * kv_width
            // F16s. The slot variant tags both memcpys' dst arg so the
            // recorder can retarget per replay.
            unsafe {
                kv_cache_append_hip_slot(
                    kv_cache,
                    device,
                    stream,
                    scratch.k_f16,
                    scratch.v_f16,
                    n_tokens,
                    s.k_append_slot,
                    s.v_append_slot,
                )?;
            }
        } else if kv_layout == Q8Contig::NAME {
            // Q8 path: quantise directly into the cache slot.
            let total_elems = n_tokens * kv_width;
            let (k_dst, v_dst, _) = kv_cache
                .compute_append_dsts(n_tokens)
                .map_err(|e| anyhow::anyhow!("kv_cache.compute_append_dsts q8: {e}"))?;
            ops.quantize_f16_q8_0(scratch.k_f16, k_dst, total_elems)
                .context("quantize prefill K → q8_0 in-place")?;
            ops.quantize_f16_q8_0(scratch.v_f16, v_dst, total_elems)
                .context("quantize prefill V → q8_0 in-place")?;
            kv_cache
                .bump_tail(n_tokens)
                .map_err(|e| anyhow::anyhow!("kv_cache.bump_tail q8: {e}"))?;
        } else {
            // SAFETY: scratch.k_f16 / v_f16 hold n_tokens * kv_width F16s.
            unsafe {
                kv_cache
                    .append(device, stream, scratch.k_f16, scratch.v_f16, n_tokens)
                    .map_err(|e| anyhow::anyhow!("kv_cache.append(L={n_tokens}): {e}"))?;
            }
        }

        // 9. Causal prefill attention. Slot path uses tagged variant.
        let n_k_tokens = kv_cache.current_tokens();
        let scale = (head_dim as f32).sqrt().recip();
        if let Some(s) = slots {
            ops.attention_prefill_f16_slots(
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
                scale,
                Some(s.n_k_slot),
                Some(s.q_off_slot),
            )
            .context("attention_prefill_f16_slots")?;
        } else if kv_layout == Q8Contig::NAME {
            ops.attention_prefill_q8_kv(
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
                scale,
            )
            .context("attention_prefill_q8_kv")?;
        } else if kv_layout == F16Contig::NAME {
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
                scale,
            )
            .context("attention_prefill_f16")?;
        } else {
            bail!("StandardAttention prefill: unsupported KV layout {kv_layout}");
        }

        // 10. Post-attn sigmoid gate (gated path only). Plain path
        // feeds attn_out_f16 straight into the output projection.
        let post_attn_f16 = if self.gated {
            ops.sigmoid_mul_f16(
                scratch.gate_f16,
                scratch.attn_out_f16,
                scratch.gated_out_f16,
                n_tokens * q_width,
            )
            .context("prefill post-attn sigmoid-gate")?;
            scratch.gated_out_f16
        } else {
            scratch.attn_out_f16
        };

        // 11. Quantise the post-attn F16 to BOTH Q8_1 layouts.
        ops.quantize_f16_q8_1(post_attn_f16, scratch.gated_q8_1, n_tokens * q_width)
            .context("prefill quantise post-attn → Q8_1 (std)")?;
        ops.quantize_f16_q8_1_mmq(
            post_attn_f16,
            scratch.gated_q8_1_mmq,
            q_width,
            n_tokens,
        )
        .context("prefill quantise post-attn → Q8_1 (MMQ DS4)")?;

        // 12. Output projection.
        ops.qmatmul(
            self.attn_output.ptr,
            scratch.gated_q8_1,
            scratch.gated_q8_1_mmq,
            scratch.mmvq_f32,
            n_tokens,
            q_width,
            hidden,
            self.attn_output.dtype,
        )
        .context("prefill qmatmul attn_output")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, delta_out, n_tokens * hidden)
            .context("prefill cast attn_output → f16")?;

        Ok(())
    }
}
