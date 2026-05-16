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

use crate::driver_utils::RawAllocTracker;

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

/// Borrowed view over a caller-owned scratch sized for the
/// **batched-decode TP** path (one shared scratch across N concurrent
/// decode slots). Extends the prefill scratch with the per-slot tables
/// the batched KV-append + batched-attention kernels read:
/// - `slot_{k,v}_ptrs`: device `[N] u64` arrays of per-slot KV-cache
///   base pointers.
/// - `slot_n_tokens_kv`: device `[N] i32` of each slot's KV tail
///   length AFTER this token is appended.
/// - `slot_write_pos`: device `[N] i32` of each slot's pre-bump write
///   position (used by `kv_append_f16_batched_slots`).
/// - `slot_*_host`: persistent host mirrors uploaded once per call.
pub struct StandardAttentionBatchedDecodeScratch<'a> {
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
    pub slot_k_ptrs: DevicePtr,
    pub slot_v_ptrs: DevicePtr,
    pub slot_n_tokens_kv: DevicePtr,
    pub slot_write_pos: DevicePtr,
    pub slot_k_ptrs_host: &'a mut [u64],
    pub slot_v_ptrs_host: &'a mut [u64],
    pub slot_n_tokens_kv_host: &'a mut [i32],
    pub slot_write_pos_host: &'a mut [i32],
}

/// Split-K partial budget — 32 chunks × 512 tokens covers any decode
/// context up to 16 384 tokens. Bump alongside the dispatch threshold
/// if context ever exceeds. Mirrors qwen3-moe's `MAX_SPLITK_CHUNKS`.
pub const MAX_SPLITK_CHUNKS: usize = 32;

/// Shape inputs needed to size a `StandardAttention` scratch. Models
/// pass their **max-across-layers** dims here — the scratch is reused
/// across every full-attn layer in a stage. For uniform-arch models
/// (qwen3moe), call [`StandardAttention::scratch_dims`] on any per-call
/// block. For multi-shape archs (gemma4 with `head_dim_swa != head_dim`),
/// compute the max yourself and construct manually.
#[derive(Copy, Clone, Debug)]
pub struct AttentionScratchDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

/// Owned attention decode scratch. The block knows its own size math;
/// models call [`StandardAttention::alloc_decode_scratch`] and use
/// [`OwnedStandardAttentionDecodeScratch::view_mut`] for the borrowed
/// view the forward path consumes.
///
/// All device allocations are routed through the caller's
/// [`RawAllocTracker`], so dispose follows the existing one-tracker-
/// per-stage pattern. This struct holds **no** size or dtype fields —
/// dispose lives on the tracker.
pub struct OwnedStandardAttentionDecodeScratch {
    pub x_q8_1: DevicePtr,
    pub mmvq_f32: DevicePtr,
    pub q_fused_f16: DevicePtr,
    pub q_f16: DevicePtr,
    pub gate_f16: DevicePtr,
    pub k_f16: DevicePtr,
    pub v_f16: DevicePtr,
    pub k_q8_0: DevicePtr,
    pub v_q8_0: DevicePtr,
    pub attn_out_f16: DevicePtr,
    pub gated_out_f16: DevicePtr,
    pub positions: DevicePtr,
    pub positions_host: Vec<i32>,
    pub splitk_partials_m: DevicePtr,
    pub splitk_partials_s: DevicePtr,
    pub splitk_partials_o: DevicePtr,
}

impl OwnedStandardAttentionDecodeScratch {
    /// Build the borrowed view consumed by
    /// `StandardAttention::forward_decode`. All `DevicePtr` fields are
    /// `Copy`; only `positions_host` borrows mutably.
    pub fn view_mut(&mut self) -> StandardAttentionDecodeScratch<'_> {
        StandardAttentionDecodeScratch {
            x_q8_1: self.x_q8_1,
            mmvq_f32: self.mmvq_f32,
            q_fused_f16: self.q_fused_f16,
            q_f16: self.q_f16,
            gate_f16: self.gate_f16,
            k_f16: self.k_f16,
            v_f16: self.v_f16,
            k_q8_0: self.k_q8_0,
            v_q8_0: self.v_q8_0,
            attn_out_f16: self.attn_out_f16,
            gated_out_f16: self.gated_out_f16,
            positions: self.positions,
            positions_host: &mut self.positions_host,
            splitk_partials_m: self.splitk_partials_m,
            splitk_partials_s: self.splitk_partials_s,
            splitk_partials_o: self.splitk_partials_o,
        }
    }
}

/// Owned attention prefill scratch. Sized per `max_tokens` (the upper
/// bound of one prefill chunk; the caller chunks long prompts).
pub struct OwnedStandardAttentionPrefillScratch {
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
    pub positions_host: Vec<i32>,
}

impl OwnedStandardAttentionPrefillScratch {
    pub fn view_mut(&mut self) -> StandardAttentionPrefillScratch<'_> {
        StandardAttentionPrefillScratch {
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
}

/// Owned attention batched-decode scratch. Extends the prefill scratch
/// with per-slot tables that the batched-KV-append + batched-attention
/// kernels read.
pub struct OwnedStandardAttentionBatchedDecodeScratch {
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
    pub positions_host: Vec<i32>,
    pub slot_k_ptrs: DevicePtr,
    pub slot_v_ptrs: DevicePtr,
    pub slot_n_tokens_kv: DevicePtr,
    pub slot_write_pos: DevicePtr,
    pub slot_k_ptrs_host: Vec<u64>,
    pub slot_v_ptrs_host: Vec<u64>,
    pub slot_n_tokens_kv_host: Vec<i32>,
    pub slot_write_pos_host: Vec<i32>,
}

impl OwnedStandardAttentionBatchedDecodeScratch {
    pub fn view_mut(&mut self) -> StandardAttentionBatchedDecodeScratch<'_> {
        StandardAttentionBatchedDecodeScratch {
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
    pub attn_k: WeightHandle,            // [n_kv_heads*head_dim, hidden]
    /// V projection weight. `None` triggers the gemma4
    /// "alt-attention" path where V is the pre-norm K row (a single
    /// DtoD memcpy replaces the V matmul). Standard models (qwen3.x,
    /// llama, etc.) always set this.
    pub attn_v: Option<WeightHandle>,    // [n_kv_heads*head_dim, hidden]
    pub attn_output: WeightHandle,       // [hidden, n_heads*head_dim]
    pub attn_norm_w: DevicePtr,          // F16 [hidden]
    pub attn_q_norm_w: DevicePtr,        // F16 [head_dim]
    pub attn_k_norm_w: DevicePtr,        // F16 [head_dim]
    /// V-norm weight, F16 `[head_dim]`. When `Some`, an extra
    /// per-head RMSNorm is applied to V. Gemma4 uses this with a
    /// caller-owned unit-weight buffer for the "unlearned" V norm
    /// (V-norm with weight=1.0 per lane).
    pub attn_v_norm_w: Option<DevicePtr>,
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
    /// Override the softmax pre-scale applied to QK^T. `None` ⇒ the
    /// canonical `1/sqrt(head_dim)`. Gemma4 sets this to `1.0` (its
    /// `f_attention_scale = 1.0`, no pre-attention scaling).
    pub softmax_scale: Option<f32>,
    /// Sliding-window mask radius. When `Some(w)`, keys older than
    /// `query_pos - w + 1` are masked out (Gemma4 SWA layers). When
    /// `None`, standard causal attention. The SWA-aware kernel
    /// variants land in S3 — the field is stored on the block now so
    /// callers can configure it without a second API change.
    pub window_size: Option<u32>,
    /// When `true`, the output projection skips the final
    /// `cast_f32_to_f16(mmvq_f32, delta_out)` and instead writes the
    /// F32 result directly into `delta_out` (caller must size
    /// `delta_out` as F32 `hidden * 4 bytes`). Used by gemma4 26B-A4B
    /// full-attention layers where V_norm + head_dim=512 produces a
    /// sqrt(head_dim)≈22 spike that overflows the F16 cast on the
    /// row-parallel output projection sum (see
    /// `feedback_gemma4_attn_output_proj_f16_saturate`).
    pub f32_output_proj: bool,
}

impl StandardAttention {
    /// Construct a new block. Q-weight rows are
    /// `2 * n_heads * head_dim` when `gated` and `n_heads * head_dim`
    /// otherwise; the constructor asserts the matmul shapes match.
    ///
    /// `attn_v` is `Option`: pass `Some(handle)` for the standard
    /// (qwen3.x / llama) path with a learned V projection. Pass `None`
    /// for gemma4's "alt-attention" path where V is the pre-norm K
    /// row (a DtoD memcpy from K replaces the V matmul). `attn_v_norm_w`
    /// is set separately via [`with_v_norm_w`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attn_q: WeightHandle,
        attn_k: WeightHandle,
        attn_v: Option<WeightHandle>,
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
        if let Some(v) = attn_v.as_ref() {
            if v.dims != [kv_width, hidden] {
                bail!(
                    "attn_v dims {:?} != expected [{}, {}]",
                    v.dims,
                    kv_width,
                    hidden
                );
            }
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
            attn_v_norm_w: None,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_freq_base,
            rope_rotated_dims,
            gated,
            softmax_scale: None,
            window_size: None,
            f32_output_proj: false,
        })
    }

    /// Attach a V-norm weight. Set this on gemma4 layers — pass a
    /// caller-owned F16 `[head_dim]` buffer prefilled with `1.0`
    /// (unlearned norm; the per-head RMSNorm-with-unit-weights still
    /// rescales lane variance).
    pub fn with_v_norm_w(mut self, ptr: DevicePtr) -> Self {
        self.attn_v_norm_w = Some(ptr);
        self
    }

    /// Override the softmax pre-scale. Gemma4 uses `1.0`; qwen3.x
    /// leaves it `None` and the block falls back to `1/sqrt(head_dim)`.
    pub fn with_softmax_scale(mut self, scale: f32) -> Self {
        self.softmax_scale = Some(scale);
        self
    }

    /// Configure sliding-window attention. `Some(w)` switches the
    /// block to the SWA-aware kernels in S3; `None` keeps the
    /// standard causal path.
    pub fn with_window_size(mut self, window: u32) -> Self {
        self.window_size = Some(window);
        self
    }

    /// Toggle the F32 output-projection path. When enabled,
    /// `forward_decode`'s output projection writes F32 directly into
    /// `delta_out` (caller must size as `hidden * 4 bytes`) and skips
    /// the final `cast_f32_to_f16`. The downstream AR and post-norm
    /// must consume F32. Used by gemma4 26B-A4B full-attention layers.
    pub fn with_f32_output_proj(mut self, enabled: bool) -> Self {
        self.f32_output_proj = enabled;
        self
    }

    /// Snapshot of the block's shape inputs for sizing scratch
    /// buffers. For multi-shape models (e.g. gemma4 SWA / full-attn
    /// interleave) call sites should construct `AttentionScratchDims`
    /// manually with the per-layer max instead.
    pub fn scratch_dims(&self) -> AttentionScratchDims {
        AttentionScratchDims {
            hidden: self.hidden,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
        }
    }

    /// Allocate an [`OwnedStandardAttentionDecodeScratch`] sized for
    /// `dims`. All device buffers are recorded in `tracker`; dispose
    /// happens via `tracker.dispose(device)` when the owning stage
    /// tears down.
    ///
    /// The allocation is uniform in `gated` — `q_fused_f16` is sized
    /// at `2 * q_width` regardless. The non-gated path leaves the
    /// upper half unread; the small over-allocation matches the
    /// pre-existing borrowed-view shape so models that swap from
    /// hand-rolled `*Scratch` types see byte-identical sizes.
    pub fn alloc_decode_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: AttentionScratchDims,
    ) -> Result<OwnedStandardAttentionDecodeScratch> {
        let AttentionScratchDims { hidden, n_heads, n_kv_heads, head_dim } = dims;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let q_fused_width = 2 * q_width;

        let x_q8_1_elems = hidden.max(q_width);
        let mmvq_max = q_fused_width.max(hidden);

        let (x_q8_1, _) = tracker.alloc_q8_1(device, x_q8_1_elems)?;
        let (mmvq_f32, _) = tracker.alloc_f32(device, mmvq_max)?;
        let (q_fused_f16, _) = tracker.alloc_f16(device, q_fused_width)?;
        let (q_f16, _) = tracker.alloc_f16(device, q_width)?;
        let (gate_f16, _) = tracker.alloc_f16(device, q_width)?;
        let (k_f16, _) = tracker.alloc_f16(device, kv_width)?;
        let (v_f16, _) = tracker.alloc_f16(device, kv_width)?;
        let (k_q8_0, _) = tracker.alloc_q8_0(device, kv_width)?;
        let (v_q8_0, _) = tracker.alloc_q8_0(device, kv_width)?;
        let (attn_out_f16, _) = tracker.alloc_f16(device, q_width)?;
        let (gated_out_f16, _) = tracker.alloc_f16(device, q_width)?;
        let (positions, _) = tracker.alloc_i32(device, 1)?;
        let (splitk_partials_m, _) =
            tracker.alloc_f32(device, n_heads * MAX_SPLITK_CHUNKS)?;
        let (splitk_partials_s, _) =
            tracker.alloc_f32(device, n_heads * MAX_SPLITK_CHUNKS)?;
        let (splitk_partials_o, _) =
            tracker.alloc_f32(device, n_heads * MAX_SPLITK_CHUNKS * head_dim)?;

        Ok(OwnedStandardAttentionDecodeScratch {
            x_q8_1,
            mmvq_f32,
            q_fused_f16,
            q_f16,
            gate_f16,
            k_f16,
            v_f16,
            k_q8_0,
            v_q8_0,
            attn_out_f16,
            gated_out_f16,
            positions,
            positions_host: vec![0i32; 1],
            splitk_partials_m,
            splitk_partials_s,
            splitk_partials_o,
        })
    }

    /// Allocate an [`OwnedStandardAttentionPrefillScratch`] sized for
    /// `dims` × `max_tokens`.
    pub fn alloc_prefill_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: AttentionScratchDims,
        max_tokens: usize,
    ) -> Result<OwnedStandardAttentionPrefillScratch> {
        if max_tokens == 0 {
            bail!("alloc_prefill_scratch: max_tokens must be >= 1");
        }
        let AttentionScratchDims { hidden, n_heads, n_kv_heads, head_dim } = dims;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let q_fused_width = 2 * q_width;
        let mmvq_max = q_fused_width.max(hidden);

        let (x_norm_f16, _) = tracker.alloc_f16(device, max_tokens * hidden)?;
        let (x_q8_1, _) = tracker.alloc_q8_1(device, max_tokens * hidden)?;
        let (x_q8_1_mmq, _) = tracker.alloc_q8_1_mmq(device, max_tokens * hidden)?;
        let (mmvq_f32, _) = tracker.alloc_f32(device, max_tokens * mmvq_max)?;
        let (q_fused_f16, _) = tracker.alloc_f16(device, max_tokens * q_fused_width)?;
        let (q_f16, _) = tracker.alloc_f16(device, max_tokens * q_width)?;
        let (gate_f16, _) = tracker.alloc_f16(device, max_tokens * q_width)?;
        let (k_f16, _) = tracker.alloc_f16(device, max_tokens * kv_width)?;
        let (v_f16, _) = tracker.alloc_f16(device, max_tokens * kv_width)?;
        let (attn_out_f16, _) = tracker.alloc_f16(device, max_tokens * q_width)?;
        let (gated_out_f16, _) = tracker.alloc_f16(device, max_tokens * q_width)?;
        let (positions, _) = tracker.alloc_i32(device, max_tokens)?;
        let (gated_q8_1, _) = tracker.alloc_q8_1(device, max_tokens * q_width)?;
        let (gated_q8_1_mmq, _) = tracker.alloc_q8_1_mmq(device, max_tokens * q_width)?;

        Ok(OwnedStandardAttentionPrefillScratch {
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
            positions_host: vec![0i32; max_tokens],
        })
    }

    /// Allocate an [`OwnedStandardAttentionBatchedDecodeScratch`] sized
    /// for `dims` × `max_tokens` (where `max_tokens` is the max number
    /// of concurrent decode slots the scratch will serve).
    pub fn alloc_batched_decode_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: AttentionScratchDims,
        max_tokens: usize,
    ) -> Result<OwnedStandardAttentionBatchedDecodeScratch> {
        let prefill = Self::alloc_prefill_scratch(device, tracker, dims, max_tokens)?;
        let (slot_k_ptrs, _) = tracker.alloc_u64(device, max_tokens)?;
        let (slot_v_ptrs, _) = tracker.alloc_u64(device, max_tokens)?;
        let (slot_n_tokens_kv, _) = tracker.alloc_i32(device, max_tokens)?;
        let (slot_write_pos, _) = tracker.alloc_i32(device, max_tokens)?;
        Ok(OwnedStandardAttentionBatchedDecodeScratch {
            max_tokens: prefill.max_tokens,
            x_norm_f16: prefill.x_norm_f16,
            x_q8_1: prefill.x_q8_1,
            x_q8_1_mmq: prefill.x_q8_1_mmq,
            mmvq_f32: prefill.mmvq_f32,
            q_fused_f16: prefill.q_fused_f16,
            q_f16: prefill.q_f16,
            gate_f16: prefill.gate_f16,
            k_f16: prefill.k_f16,
            v_f16: prefill.v_f16,
            attn_out_f16: prefill.attn_out_f16,
            gated_out_f16: prefill.gated_out_f16,
            positions: prefill.positions,
            gated_q8_1: prefill.gated_q8_1,
            gated_q8_1_mmq: prefill.gated_q8_1_mmq,
            positions_host: prefill.positions_host,
            slot_k_ptrs,
            slot_v_ptrs,
            slot_n_tokens_kv,
            slot_write_pos,
            slot_k_ptrs_host: vec![0u64; max_tokens],
            slot_v_ptrs_host: vec![0u64; max_tokens],
            slot_n_tokens_kv_host: vec![0i32; max_tokens],
            slot_write_pos_host: vec![0i32; max_tokens],
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

        // 4+5. K and V projections. Three paths:
        // - Both K and V are present + Q4_0: single-launch fused
        //   `mmvq_q4_0_kv_f16dst` writes directly into F16 buffers.
        // - Both K and V are present + Q8_0: single-launch fused
        //   `mmvq_q8_0_gate_up` into F32 slab + two casts.
        // - V absent (gemma4 alt-attention): K projection only; V is
        //   a DtoD copy of K's pre-norm row.
        // - Otherwise: two unfused `mmvq` + two casts (universal fallback).
        let v_present = self.attn_v.is_some();
        let attn_v_ref = self.attn_v.as_ref();
        let fuse_q4_0_kv = v_present
            && self.attn_k.dtype == QDtype::Q4_0
            && attn_v_ref.map(|v| v.dtype == QDtype::Q4_0).unwrap_or(false);
        let fuse_q8_0_kv = v_present
            && self.attn_k.dtype == QDtype::Q8_0
            && attn_v_ref.map(|v| v.dtype == QDtype::Q8_0).unwrap_or(false);
        if fuse_q4_0_kv {
            let v = attn_v_ref.expect("v_present");
            ops.mmvq_q4_0_kv_f16dst(
                self.attn_k.ptr,
                v.ptr,
                scratch.x_q8_1,
                scratch.k_f16,
                scratch.v_f16,
                kv_width,
                hidden,
            )
            .context("attn_k + attn_v fused mmvq_q4_0_kv_f16dst")?;
        } else if fuse_q8_0_kv {
            let v = attn_v_ref.expect("v_present");
            let v_f32_offset = scratch.mmvq_f32.offset_bytes(kv_width * 4);
            ops.mmvq_q8_0_gate_up(
                self.attn_k.ptr,
                v.ptr,
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
            if matches!(self.attn_k.dtype, QDtype::Q4_0 | QDtype::Q4_1 | QDtype::Q8_0) {
                ops.mmvq_f16_direct(
                    self.attn_k.ptr,
                    scratch.x_q8_1,
                    scratch.k_f16,
                    kv_width,
                    hidden,
                    self.attn_k.dtype,
                )
                .context("mmvq attn_k → f16 direct")?;
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
            }
            if let Some(v) = attn_v_ref {
                ops.mmvq(
                    v.ptr,
                    scratch.x_q8_1,
                    scratch.mmvq_f32,
                    kv_width,
                    hidden,
                    v.dtype,
                )
                .context("mmvq attn_v")?;
                ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.v_f16, kv_width)
                    .context("cast attn_v → f16")?;
            } else {
                // Alt-attention (gemma4): V = K's pre-norm row. Done
                // BEFORE the per-head norms so V's RMSNorm (when
                // present) runs on the raw projection, not the
                // K-normed one.
                // SAFETY: k_f16 / v_f16 are both contiguous F16
                // `[n_kv_heads, head_dim]` buffers on `device`.
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
        }

        // 6. Per-head Q / K rmsnorm. V-norm (unlearned, gemma4) runs
        // only when `attn_v_norm_w` is set.
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
        if let Some(v_norm_w) = self.attn_v_norm_w {
            ops.rmsnorm_f16(
                scratch.v_f16,
                v_norm_w,
                scratch.v_f16,
                n_kv_heads,
                head_dim,
                self.rms_norm_eps,
            )
            .context("attn_v_norm (unlearned)")?;
        }

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
        let scale = self
            .softmax_scale
            .unwrap_or_else(|| (head_dim as f32).sqrt().recip());
        let window = self.window_size.map(|w| w as i32).unwrap_or(0);
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
                window,
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
                    window,
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
                window,
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

        // 12. Output projection [hidden, q_width]. F32 mmvq output.
        // When `f32_output_proj`, write directly into `delta_out`
        // (caller-sized F32) and skip the saturating F16 cast —
        // gemma4 26B-A4B full-attention V_norm spike + Q8_0 + head_dim=512
        // can push the F32 sum past F16 max.
        let mmvq_out = if self.f32_output_proj {
            delta_out
        } else {
            scratch.mmvq_f32
        };
        ops.mmvq(
            self.attn_output.ptr,
            scratch.x_q8_1,
            mmvq_out,
            hidden,
            q_width,
            self.attn_output.dtype,
        )
        .context("mmvq attn_output")?;
        if !self.f32_output_proj {
            ops.cast_f32_to_f16(scratch.mmvq_f32, delta_out, hidden)
                .context("cast attn_output → f16")?;
        }

        Ok(())
    }

    /// Decode for a shared-KV tail layer — the layer reuses another
    /// layer's KV cache instead of computing its own. Skips K/V
    /// projection + V norm + KV append; runs only:
    /// `attn_norm → Q proj → Q norm → RoPE Q → attention read →
    ///  output_proj`.
    ///
    /// `kv_cache` is the routed source layer's cache; this method
    /// only reads from it. Honours `softmax_scale`, `window_size`,
    /// and `f32_output_proj` like `forward_decode`.
    ///
    /// Used by gemma4 E4B's shared-KV tail layers.
    pub fn forward_decode_shared_kv<L: CacheLayout, O: Ops>(
        &self,
        ops: &O,
        device: &HipDevice,
        stream: &HipStream,
        x_in: DevicePtr,
        delta_out: DevicePtr,
        kv_cache: &KvCache<L, HipDevice>,
        scratch: &mut StandardAttentionDecodeScratch<'_>,
        position: usize,
    ) -> Result<()> {
        let hidden = self.hidden;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let q_width = n_heads * head_dim;

        // 1. attn_norm + Q8_1 quantise.
        ops.rmsnorm_quant_q8_1(
            x_in,
            self.attn_norm_w,
            scratch.x_q8_1,
            1,
            hidden,
            self.rms_norm_eps,
        )
        .context("shared-kv attn_norm + quant")?;

        // 2. Q projection.
        ops.mmvq(
            self.attn_q.ptr,
            scratch.x_q8_1,
            scratch.mmvq_f32,
            q_width,
            hidden,
            self.attn_q.dtype,
        )
        .context("shared-kv mmvq attn_q")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.q_f16, q_width)
            .context("shared-kv cast Q → f16")?;

        // 3. Per-head Q norm.
        ops.rmsnorm_f16(
            scratch.q_f16,
            self.attn_q_norm_w,
            scratch.q_f16,
            n_heads,
            head_dim,
            self.rms_norm_eps,
        )
        .context("shared-kv attn_q_norm")?;

        // 4. Position upload + RoPE Q.
        scratch.positions_host[0] = position as i32;
        // SAFETY: scratch.positions is i32 [1]; positions_host outlives sync.
        unsafe {
            device.memcpy_async(
                stream,
                flambeau_core::CopyDirection::HostToDevice,
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
        .context("shared-kv rope Q")?;

        // 5. Attention read against the routed cache.
        let n_tokens_kv = kv_cache.current_tokens();
        let scale = self
            .softmax_scale
            .unwrap_or_else(|| (head_dim as f32).sqrt().recip());
        let window = self.window_size.map(|w| w as i32).unwrap_or(0);
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
            window,
        )
        .context("shared-kv attention_decode_f16")?;

        // 6. Output projection.
        ops.quantize_f16_q8_1(scratch.attn_out_f16, scratch.x_q8_1, q_width)
            .context("shared-kv quantize attn_out → Q8_1")?;
        let mmvq_out = if self.f32_output_proj {
            delta_out
        } else {
            scratch.mmvq_f32
        };
        ops.mmvq(
            self.attn_output.ptr,
            scratch.x_q8_1,
            mmvq_out,
            hidden,
            q_width,
            self.attn_output.dtype,
        )
        .context("shared-kv mmvq attn_output")?;
        if !self.f32_output_proj {
            ops.cast_f32_to_f16(scratch.mmvq_f32, delta_out, hidden)
                .context("shared-kv cast attn_output → f16")?;
        }
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

        // 5. V projection. Standard path runs `qmatmul attn_v`; gemma4
        // alt-attention (`attn_v == None`) copies the pre-norm K row
        // into V via a single DtoD memcpy. The copy must precede the
        // per-head norms so V's RMSNorm (when present, unlearned for
        // gemma4) runs on the raw K projection, not the K-normed one.
        if let Some(attn_v_pref) = self.attn_v.as_ref() {
            ops.qmatmul(
                attn_v_pref.ptr,
                scratch.x_q8_1,
                scratch.x_q8_1_mmq,
                scratch.mmvq_f32,
                n_tokens,
                hidden,
                kv_width,
                attn_v_pref.dtype,
            )
            .context("prefill qmatmul attn_v")?;
            ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.v_f16, n_tokens * kv_width)
                .context("prefill cast attn_v → f16")?;
        } else {
            // SAFETY: k_f16 / v_f16 each hold n_tokens * kv_width F16 values.
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

        // 6. Per-head Q / K / (optional V) rmsnorm.
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
        if let Some(v_norm_w) = self.attn_v_norm_w {
            ops.rmsnorm_f16(
                scratch.v_f16,
                v_norm_w,
                scratch.v_f16,
                n_tokens * n_kv_heads,
                head_dim,
                self.rms_norm_eps,
            )
            .context("prefill attn_v_norm (unlearned)")?;
        }

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
        let scale = self
            .softmax_scale
            .unwrap_or_else(|| (head_dim as f32).sqrt().recip());
        let window = self.window_size.map(|w| w as i32).unwrap_or(0);
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
                window,
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
                window,
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

    /// **#266c** — batched decode across `N = slot_positions.len()`
    /// concurrent slots, per-rank TP. Front-end (RMSNorm + Q|gate proj
    /// + K proj + V proj + per-head Q/K norm + RoPE) mirrors prefill at
    /// `n_tokens = N`. Back-end diverges:
    /// - **Step 8 (KV append):** per-slot bookkeeping + optional
    ///   single-launch `kv_append_f16_batched_slots` driven by
    ///   `slot_write_pos`. Set `batch_kv_append = false` to fall back
    ///   to the per-slot DtoD memcpy loop (FLAMBEAU_KV_APPEND_BATCHED=0
    ///   at qwen3-moe call sites).
    /// - **Step 9 (attention):** single-launch
    ///   `attention_decode_f16_batched` over all N slots reading the
    ///   `slot_*` tables.
    /// Caller schedules the AR over `partial_out` after this returns.
    ///
    /// Requires `self.gated == true` (qwen3.x batched-decode is always
    /// gated) and the V projection present (no alt-V on this path).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_batched_tp<O: Ops>(
        &self,
        ops: &O,
        device: &HipDevice,
        stream: &HipStream,
        x_in: DevicePtr,
        partial_out: DevicePtr,
        slot_kv_caches: &mut [&mut KvCache<F16Contig, HipDevice>],
        slot_positions: &[usize],
        scratch: &mut StandardAttentionBatchedDecodeScratch<'_>,
        batch_kv_append: bool,
    ) -> Result<()> {
        if !self.gated {
            bail!(
                "StandardAttention::forward_decode_batched_tp: requires gated=true \
                 (qwen3.x batched-decode); plain-Q + alt-V batched path not implemented"
            );
        }
        let attn_v = self.attn_v.as_ref().ok_or_else(|| anyhow::anyhow!(
            "forward_decode_batched_tp: attn_v is None (alt-attention batched path not implemented)"
        ))?;
        let n_tokens = slot_positions.len();
        if n_tokens == 0 {
            bail!("forward_decode_batched_tp called with 0 slots");
        }
        if n_tokens != slot_kv_caches.len() {
            bail!(
                "slot count mismatch: positions={n_tokens}, caches={}",
                slot_kv_caches.len()
            );
        }
        if n_tokens > scratch.max_tokens {
            bail!(
                "n_tokens={n_tokens} > scratch.max_tokens={}",
                scratch.max_tokens
            );
        }

        let hidden = self.hidden;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;
        let q_proj_rows = 2 * q_width;

        // 1. RMSNorm + Q8_1 quantise (both layouts).
        ops.rmsnorm_f16(
            x_in,
            self.attn_norm_w,
            scratch.x_norm_f16,
            n_tokens,
            hidden,
            self.rms_norm_eps,
        )
        .context("batched-decode attn_norm")?;
        ops.quantize_f16_q8_1(scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden)
            .context("batched-decode x_norm → Q8_1 std")?;
        ops.quantize_f16_q8_1_mmq(
            scratch.x_norm_f16,
            scratch.x_q8_1_mmq,
            hidden,
            n_tokens,
        )
        .context("batched-decode x_norm → Q8_1 MMQ")?;

        // 2. Q|gate fused projection.
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
        .context("batched-decode qmatmul attn_q")?;
        ops.cast_f32_to_f16(
            scratch.mmvq_f32,
            scratch.q_fused_f16,
            n_tokens * q_proj_rows,
        )
        .context("batched-decode cast attn_q → f16")?;

        // 3. Split Q | gate.
        ops.split_q_gate_f16(
            scratch.q_fused_f16,
            scratch.q_f16,
            scratch.gate_f16,
            n_tokens,
            n_heads,
            head_dim,
        )
        .context("batched-decode split_q_gate")?;

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
        .context("batched-decode qmatmul attn_k")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.k_f16, n_tokens * kv_width)
            .context("batched-decode cast attn_k → f16")?;

        // 5. V projection.
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
        .context("batched-decode qmatmul attn_v")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.v_f16, n_tokens * kv_width)
            .context("batched-decode cast attn_v → f16")?;

        // 6. Per-head Q/K rmsnorm.
        ops.rmsnorm_f16(
            scratch.q_f16,
            self.attn_q_norm_w,
            scratch.q_f16,
            n_tokens * n_heads,
            head_dim,
            self.rms_norm_eps,
        )
        .context("batched-decode attn_q_norm")?;
        ops.rmsnorm_f16(
            scratch.k_f16,
            self.attn_k_norm_w,
            scratch.k_f16,
            n_tokens * n_kv_heads,
            head_dim,
            self.rms_norm_eps,
        )
        .context("batched-decode attn_k_norm")?;

        // 7. RoPE with per-slot positions.
        for (i, &pos) in slot_positions.iter().enumerate() {
            scratch.positions_host[i] = pos as i32;
        }
        // SAFETY: scratch.positions has at least n_tokens * 4 valid
        // bytes; positions_host is a persistent Vec on the scratch.
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
        .context("batched-decode rope Q")?;
        ops.rope_neox_partial_f16(
            scratch.k_f16,
            scratch.positions,
            self.rope_freq_base,
            n_tokens,
            n_kv_heads,
            head_dim,
            self.rope_rotated_dims,
        )
        .context("batched-decode rope K")?;

        // 8. Per-slot KV append. Host-side bookkeeping always runs
        // (records write_pos, bumps tail, populates slot tables); the
        // K/V row copy is either deferred to a single
        // `kv_append_f16_batched_slots` launch (when
        // `batch_kv_append=true`) or done per-slot via DtoD memcpy.
        let kv_per_token_bytes = kv_width * 2;
        for s in 0..n_tokens {
            let kv = &mut *slot_kv_caches[s];
            let write_pos = kv.current_tokens();
            if !batch_kv_append {
                let k_src = scratch.k_f16.offset_bytes(s * kv_per_token_bytes);
                let v_src = scratch.v_f16.offset_bytes(s * kv_per_token_bytes);
                let k_dst = kv.k_buffer().offset_bytes(write_pos * kv_per_token_bytes);
                let v_dst = kv.v_buffer().offset_bytes(write_pos * kv_per_token_bytes);
                // SAFETY: src/dst sizes match kv_per_token_bytes;
                // write_pos < max_seq_len enforced by cache.
                unsafe {
                    device.memcpy_async(
                        stream,
                        CopyDirection::DeviceToDevice,
                        k_dst,
                        k_src,
                        kv_per_token_bytes,
                    )?;
                    device.memcpy_async(
                        stream,
                        CopyDirection::DeviceToDevice,
                        v_dst,
                        v_src,
                        kv_per_token_bytes,
                    )?;
                }
            }
            kv.bump_tail(1)
                .map_err(|e| anyhow::anyhow!("slot {s} bump_tail: {e}"))?;
            scratch.slot_k_ptrs_host[s] = kv.k_buffer().as_usize() as u64;
            scratch.slot_v_ptrs_host[s] = kv.v_buffer().as_usize() as u64;
            scratch.slot_n_tokens_kv_host[s] = kv.current_tokens() as i32;
            scratch.slot_write_pos_host[s] = write_pos as i32;
        }

        // 9. Upload slot tables + single-launch batched attention.
        // SAFETY: host vecs have stable addresses; device buffers
        // sized for max_tokens ≥ n_tokens.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                scratch.slot_k_ptrs,
                DevicePtr(scratch.slot_k_ptrs_host.as_ptr() as usize),
                n_tokens * 8,
            )?;
            device.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                scratch.slot_v_ptrs,
                DevicePtr(scratch.slot_v_ptrs_host.as_ptr() as usize),
                n_tokens * 8,
            )?;
            device.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                scratch.slot_n_tokens_kv,
                DevicePtr(scratch.slot_n_tokens_kv_host.as_ptr() as usize),
                n_tokens * 4,
            )?;
            if batch_kv_append {
                device.memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    scratch.slot_write_pos,
                    DevicePtr(scratch.slot_write_pos_host.as_ptr() as usize),
                    n_tokens * 4,
                )?;
            }
        }
        if batch_kv_append {
            ops.kv_append_f16_batched_slots(
                scratch.k_f16,
                scratch.v_f16,
                scratch.slot_k_ptrs,
                scratch.slot_v_ptrs,
                scratch.slot_write_pos,
                n_tokens,
                kv_width,
            )
            .context("batched KV-append")?;
        }
        let scale = self
            .softmax_scale
            .unwrap_or_else(|| (head_dim as f32).sqrt().recip());
        ops.attention_decode_f16_batched(
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
        .context("attention_decode_f16_batched")?;

        // 10. Post-attn sigmoid gate over [N, q_width].
        let gated_elems = n_tokens * q_width;
        ops.sigmoid_mul_f16(
            scratch.gate_f16,
            scratch.attn_out_f16,
            scratch.gated_out_f16,
            gated_elems,
        )
        .context("batched-decode sigmoid-gate")?;

        // 11. Quantise gated → both Q8_1 layouts.
        ops.quantize_f16_q8_1(scratch.gated_out_f16, scratch.gated_q8_1, gated_elems)
            .context("batched-decode gated → Q8_1 std")?;
        ops.quantize_f16_q8_1_mmq(
            scratch.gated_out_f16,
            scratch.gated_q8_1_mmq,
            q_width,
            n_tokens,
        )
        .context("batched-decode gated → Q8_1 MMQ")?;

        // 12. Row-parallel output projection.
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
        .context("batched-decode qmatmul attn_output")?;
        ops.cast_f32_to_f16(scratch.mmvq_f32, partial_out, n_tokens * hidden)
            .context("batched-decode cast attn_output → f16")?;

        Ok(())
    }
}
