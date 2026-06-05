//! Op-signature parameter aggregates.
//!
//! Every type in this module is a `Copy + Clone + Debug` POD whose only
//! purpose is to group recurring kernel-launch arguments by semantic
//! identity (launch context, buffers, shape, knobs). They contain no
//! methods, no invariants, and no behaviour — pass them by value, they
//! are pointer-sized stack aggregates.
//!
//! See `doc/CLIPPY_STRUCT_REFACTOR_PLAN.md` for the phased migration.

use crate::OpsRegistry;
use flambeau_backend_hip::HipStream;
use flambeau_core::device::DevicePtr;

/// Launch context handed to every `crates/ops/src/hip/*` free-fn wrapper.
/// Both fields are short-lived borrows shared across every kernel launch
/// in a forward pass — the registry owns the loaded modules, the stream
/// is the queue the launch enqueues onto.
#[derive(Copy, Clone)]
pub struct OpCtx<'a> {
    pub reg: &'a OpsRegistry,
    pub stream: &'a HipStream,
}

// --- matmul / mmvq families -------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct MmvqBuffers {
    pub weights: DevicePtr,
    pub act_q8_1: DevicePtr,
    pub dst: DevicePtr,
}

/// Umbrella `qmatmul` dispatcher buffers. Adds `act_q8_1_mmq` (DS4
/// 144 B layout) for the MmqLdsX64 prefill recipe; decode-path callers
/// can pass `DevicePtr(0)` for it (the dispatcher's F16 / MMVQ short-
/// circuits never read it).
#[derive(Copy, Clone, Debug)]
pub struct QmatmulBuffers {
    pub weights: DevicePtr,
    pub act_q8_1: DevicePtr,
    pub act_q8_1_mmq: DevicePtr,
    pub dst: DevicePtr,
}

/// Fused K+V MMVQ buffers for the `mmvq_q4_0_kv_f16dst` dispatcher.
/// Used on TP full-attn layers where K and V share their projection
/// activation; both outputs land in F16 in one launch.
#[derive(Copy, Clone, Debug)]
pub struct MmvqKvF16Buffers {
    pub k_w: DevicePtr,
    pub v_w: DevicePtr,
    pub y_q8_1: DevicePtr,
    pub k_out_f16: DevicePtr,
    pub v_out_f16: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqGateUpBuffers {
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub act_q8_1: DevicePtr,
    pub gate_out: DevicePtr,
    pub up_out: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MatmulShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqShape {
    pub n_rows: usize,
    pub k: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqBatchShape {
    pub n_rows: usize,
    pub k: usize,
    pub n_slots: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqGateUpShape {
    pub n_rows_gate: usize,
    pub n_rows_up: usize,
    pub k: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqGateUpBatchShape {
    pub n_rows_gate: usize,
    pub n_rows_up: usize,
    pub k: usize,
    pub n_slots: usize,
}

// --- attention family -------------------------------------------------------

use flambeau_backend_hip::ScalarSlot;

#[derive(Copy, Clone, Debug)]
pub struct AttnBuffers {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub out: DevicePtr,
}

/// Batched (single-launch over N slots) decode buffers.
/// `k_cache_ptrs` + `v_cache_ptrs` + `n_tokens_kv_ptrs` are device-side
/// `[n_slots]` arrays of per-slot KV-cache pointers + tail lengths.
#[derive(Copy, Clone, Debug)]
pub struct AttnBatchedBuffers {
    pub q_batched: DevicePtr,
    pub k_cache_ptrs: DevicePtr,
    pub v_cache_ptrs: DevicePtr,
    pub out_batched: DevicePtr,
    pub n_tokens_kv_ptrs: DevicePtr,
}

/// PagedAttention decode buffers. K/V live in shared pools indexed via
/// `block_tables`. `n_tokens_kv_ptrs` is `[n_slots]` i32 device array.
#[derive(Copy, Clone, Debug)]
pub struct AttnPagedDecodeBuffers {
    pub q_batched: DevicePtr,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub block_tables: DevicePtr,
    pub out_batched: DevicePtr,
    pub n_tokens_kv_ptrs: DevicePtr,
}

/// PagedAttention prefill buffers. One slot's prefill walks a single
/// `block_table` row across `n_q_tokens` K/V rows in the pool.
#[derive(Copy, Clone, Debug)]
pub struct AttnPagedPrefillBuffers {
    pub q: DevicePtr,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub block_table: DevicePtr,
    pub out: DevicePtr,
}

/// Extra scratch the split-K decode variant needs alongside
/// [`AttnBuffers`]. Per-`(head, chunk)` running max + softmax denom +
/// partial output vectors get merged in the combine pass.
#[derive(Copy, Clone, Debug)]
pub struct AttnSplitkPartials {
    pub partials_m: DevicePtr,
    pub partials_s: DevicePtr,
    pub partials_o: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnDecodeShape {
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub n_tokens_kv: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnDecodeBatchedShape {
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub n_slots: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnDecodePagedShape {
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub n_slots: usize,
    pub page_size: usize,
    pub max_pages_per_slot: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnSplitkShape {
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub n_tokens_kv: usize,
    pub chunk_size: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnPrefillShape {
    pub n_q_tokens: usize,
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub n_k_tokens: usize,
    pub q_offset: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnPrefillPagedShape {
    pub n_q_tokens: usize,
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub n_k_tokens: usize,
    pub q_offset: usize,
    pub page_size: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnKnobs {
    pub scale: f32,
    pub window_size: i32,
    /// Ring-buffer KV slab depth in rows. `0` = absolute addressing
    /// (full-context contiguous cache, today's default). `>0` = the
    /// attention kernel addresses KV rows modulo this depth, for
    /// window-sized sliding-window-attention slabs. Set from the
    /// per-layer KV slab depth; equals the slab's row count.
    pub ring_depth: i32,
}

/// Graph-capture slot for `attention_decode_f16_slots`. `Some(slot)`
/// tags the `n_tokens_kv` kernel arg for per-replay update.
#[derive(Copy, Clone, Debug)]
pub struct AttnDecodeSlots {
    pub n_tokens_kv: ScalarSlot,
}

/// Graph-capture slots for `attention_prefill_f16_slots`.
#[derive(Copy, Clone, Debug)]
pub struct AttnPrefillSlots {
    pub n_k: ScalarSlot,
    pub q_off: ScalarSlot,
}

// --- MoE family ------------------------------------------------------------

/// Single-weight MoE MMVQ buffers — `weights` is the contiguous
/// expert-block weights array; `expert_ids` is `[n_tokens, top_k] i32`
/// of expert indices; `act` is `[n_tokens] Q8_1`; `dst` is `[n_tokens,
/// top_k, n_rows] F32`.
#[derive(Copy, Clone, Debug)]
pub struct MoeMmvqBuffers {
    pub weights: DevicePtr,
    pub act: DevicePtr,
    pub expert_ids: DevicePtr,
    pub dst: DevicePtr,
}

/// Fused gate + up MoE MMVQ — two weight matrices, two output buffers,
/// shared expert_ids + activation.
#[derive(Copy, Clone, Debug)]
pub struct MoeMmvqGateUpBuffers {
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub act: DevicePtr,
    pub expert_ids: DevicePtr,
    pub gate_out: DevicePtr,
    pub up_out: DevicePtr,
}

/// Single-weight MoE MMVQ with a sorted-pair token-index reorder for
/// expert-major launch ordering — same buffer set as
/// [`MoeMmvqBuffers`] plus the sorted pair index array.
#[derive(Copy, Clone, Debug)]
pub struct MoeMmvqSortedBuffers {
    pub weights: DevicePtr,
    pub act: DevicePtr,
    pub expert_ids: DevicePtr,
    pub sorted_pair_idx: DevicePtr,
    pub dst: DevicePtr,
}

/// Fused gate + up MoE MMVQ with sorted-pair token-index reorder —
/// extends [`MoeMmvqGateUpBuffers`] with the sorted pair index array.
#[derive(Copy, Clone, Debug)]
pub struct MoeMmvqGateUpSortedBuffers {
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub act: DevicePtr,
    pub expert_ids: DevicePtr,
    pub sorted_pair_idx: DevicePtr,
    pub gate_out: DevicePtr,
    pub up_out: DevicePtr,
}

/// Fused gate + up MoE MMQ tile8 buffers — same shape as
/// [`MoeMmvqGateUpSortedBuffers`] plus a `padded_offsets` ptr that
/// encodes per-block starting position within the padded sorted
/// expert-major batch.
#[derive(Copy, Clone, Debug)]
pub struct MoeMmqTile8GateUpBuffers {
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub act: DevicePtr,
    pub expert_ids: DevicePtr,
    pub sorted_pair_idx_padded: DevicePtr,
    pub padded_offsets: DevicePtr,
    pub gate_out: DevicePtr,
    pub up_out: DevicePtr,
}

/// Single-weight MoE MMQ tile8 buffers — used by the `*_down_tile8`
/// down-projection family.
#[derive(Copy, Clone, Debug)]
pub struct MoeMmqTile8DownBuffers {
    pub weights: DevicePtr,
    pub act: DevicePtr,
    pub expert_ids: DevicePtr,
    pub sorted_pair_idx_padded: DevicePtr,
    pub padded_offsets: DevicePtr,
    pub dst: DevicePtr,
}

/// Shape parameters shared across every MoE MMVQ variant.
/// `n_sb_per_row` is super-blocks for K-quants / Q-blocks (k / 32)
/// for `_0` quants — the kernel ABI takes the raw count.
#[derive(Copy, Clone, Debug)]
pub struct MoeMmvqShape {
    pub n_rows: usize,
    pub n_tokens: usize,
    pub top_k: usize,
    pub n_sb_per_row: usize,
}

// --- norm / fused-norm family ----------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct NormBuffers {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub output: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct NormResidualBuffers {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub resid_in: DevicePtr,
    pub resid_out: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct NormFusedAddBuffers {
    pub x_in: DevicePtr,
    pub delta: DevicePtr,
    pub weight: DevicePtr,
    pub mid: DevicePtr,
    pub mid_norm: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct NormShape {
    pub m: usize,
    pub k: usize,
}

// --- rope / positional-encoding family -------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct RopeBuffers {
    pub x: DevicePtr,
    pub positions: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct RopeFusedBuffers {
    pub x: DevicePtr,
    pub norm_w: DevicePtr,
    pub positions: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct RopeShape {
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct RopePartialShape {
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rotated_dims: usize,
}

// --- KV-append family ------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct KvAppendBuffers {
    pub k_src: DevicePtr,
    pub v_src: DevicePtr,
    pub k_dst: DevicePtr,
    pub v_dst: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct KvAppendPagedSlotsBuffers {
    pub k_src: DevicePtr,
    pub v_src: DevicePtr,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub block_tables: DevicePtr,
    pub slot_write_pos: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct KvAppendPagedPrefillBuffers {
    pub k_src: DevicePtr,
    pub v_src: DevicePtr,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub block_table: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct KvAppendBatchedSlotsBuffers {
    pub k_src: DevicePtr,
    pub v_src: DevicePtr,
    pub slot_k_dst_ptrs: DevicePtr,
    pub slot_v_dst_ptrs: DevicePtr,
    pub slot_write_pos: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct KvAppendVUnitShape {
    pub n_tokens: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Ring-buffer slab depth in rows; 0 = absolute addressing.
    pub ring_depth: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct KvAppendPagedSlotsShape {
    pub n_slots: usize,
    pub kv_width: usize,
    pub page_size: usize,
    pub max_pages_per_slot: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct KvAppendPagedPrefillShape {
    pub n_tokens: usize,
    pub kv_width: usize,
    pub start_pos: usize,
    pub page_size: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct KvAppendBatchedSlotsShape {
    pub n_slots: usize,
    pub kv_width: usize,
    /// Ring-buffer slab depth in rows; 0 = absolute addressing.
    pub ring_depth: usize,
}

// --- MoE sort-by-expert family ---------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct MoeSortBuffers {
    pub expert_ids: DevicePtr,
    pub counts: DevicePtr,
    pub offsets: DevicePtr,
    pub cursors: DevicePtr,
    pub sorted_pair_idx: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MoeSortPaddedBuffers {
    pub expert_ids: DevicePtr,
    pub counts: DevicePtr,
    pub offsets: DevicePtr,
    pub cursors: DevicePtr,
    pub sorted_pair_idx: DevicePtr,
    pub padded_offsets: DevicePtr,
    pub sorted_pair_idx_padded: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MoeSortShape {
    pub total: usize,
    pub n_experts: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MoeSortPaddedShape {
    pub total: usize,
    pub n_experts: usize,
    pub max_tokens: usize,
    pub top_k: usize,
}

// --- MoE combine family ----------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct MoeCombineBuffers {
    pub expert_outs: DevicePtr,
    pub weights: DevicePtr,
    pub residual: DevicePtr,
    pub out: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MoeCombineNoResidualBuffers {
    pub expert_outs: DevicePtr,
    pub weights: DevicePtr,
    pub out: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MoeCombineTwoResidualsBuffers {
    pub expert_outs: DevicePtr,
    pub weights: DevicePtr,
    pub residual1: DevicePtr,
    pub residual2: DevicePtr,
    pub out: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MoeCombineShape {
    pub n_tokens: usize,
    pub top_k: usize,
    pub hidden: usize,
}

// ---------------------------------------------------------------------------
// GDN (gated delta-net) recurrent-state aggregates.
// ---------------------------------------------------------------------------

/// Buffers for [`Ops::gdn_state_step_f32_s128`] — the gate/beta variant
/// of the GDN per-token recurrent step.
#[derive(Copy, Clone, Debug)]
pub struct GdnStepBuffers {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub gate: DevicePtr,
    pub beta: DevicePtr,
    pub state_in: DevicePtr,
    pub state_out: DevicePtr,
    pub attn_out: DevicePtr,
}

/// Buffers for [`Ops::gdn_state_step_alphabeta_f32_s128`] — α/β variant
/// that fuses the gate+beta prep inline.
#[derive(Copy, Clone, Debug)]
pub struct GdnStepAlphaBetaBuffers {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub alpha_in: DevicePtr,
    pub beta_in: DevicePtr,
    pub ssm_dt_bias: DevicePtr,
    pub ssm_a: DevicePtr,
    pub state_in: DevicePtr,
    pub state_out: DevicePtr,
    pub attn_out: DevicePtr,
}

/// Buffers for the batched-slots α/β GDN step. Same shape as
/// [`GdnStepAlphaBetaBuffers`] but `state_in_ptrs` / `state_out_ptrs`
/// are `[B] u64` device arrays of per-slot state base pointers.
#[derive(Copy, Clone, Debug)]
pub struct GdnStepAlphaBetaBatchedSlotsBuffers {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub alpha_in: DevicePtr,
    pub beta_in: DevicePtr,
    pub ssm_dt_bias: DevicePtr,
    pub ssm_a: DevicePtr,
    pub state_in_ptrs: DevicePtr,
    pub state_out_ptrs: DevicePtr,
    pub attn_out: DevicePtr,
}

/// Shared shape carrier for the three GDN state-step variants.
#[derive(Copy, Clone, Debug)]
pub struct GdnStepShape {
    pub b: usize,
    pub h_v: usize,
    pub l: usize,
    pub n_rep: usize,
    pub rep_inner_layout: bool,
}

/// Buffers for [`Ops::gdn_alpha_beta_f32`] — the α/β preparation kernel
/// invoked before [`GdnStepAlphaBetaBuffers`]-shaped state steps.
#[derive(Copy, Clone, Debug)]
pub struct GdnAlphaBetaBuffers {
    pub alpha_in: DevicePtr,
    pub beta_in: DevicePtr,
    pub ssm_dt_bias: DevicePtr,
    pub ssm_a: DevicePtr,
    pub gate_out: DevicePtr,
    pub beta_out: DevicePtr,
}

/// Shape for [`Ops::gdn_alpha_beta_f32`].
#[derive(Copy, Clone, Debug)]
pub struct GdnAlphaBetaShape {
    pub num_v_heads: usize,
    pub n_tokens: usize,
}

/// Buffers for [`Ops::gdn_conv_trio_decode_f32_batched_slots`].
#[derive(Copy, Clone, Debug)]
pub struct GdnConvTrioBatchedSlotsBuffers {
    pub slot_history_ptrs: DevicePtr,
    pub qkv_mixed: DevicePtr,
    pub weight: DevicePtr,
    pub conv_out: DevicePtr,
}

/// Shape for [`Ops::gdn_conv_trio_decode_f32_batched_slots`].
#[derive(Copy, Clone, Debug)]
pub struct GdnConvTrioShape {
    pub n_slots: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
}

/// Buffers for [`Ops::gdn_split_qkv_f32`].
#[derive(Copy, Clone, Debug)]
pub struct GdnSplitQkvBuffers {
    pub silu_out: DevicePtr,
    pub q_out: DevicePtr,
    pub k_out: DevicePtr,
    pub v_out: DevicePtr,
}

/// Shape for [`Ops::gdn_split_qkv_f32`].
#[derive(Copy, Clone, Debug)]
pub struct GdnSplitQkvShape {
    pub n_tokens: usize,
    pub qk_size: usize,
    pub v_size: usize,
}

// ---------------------------------------------------------------------------
// Sampling penalty aggregates.
// ---------------------------------------------------------------------------

/// Buffers for [`Ops::apply_penalties_f32`].
#[derive(Copy, Clone, Debug)]
pub struct PenaltyBuffers {
    pub logits: DevicePtr,
    pub token_counts: DevicePtr,
}

/// OpenAI-style penalty knobs.
#[derive(Copy, Clone, Debug)]
pub struct PenaltyKnobs {
    pub repetition: f32,
    pub presence: f32,
    pub frequency: f32,
}

// ---------------------------------------------------------------------------
// MoE Q4_K turbo aggregates — used by `indexed_moe_mmq_q4_k_*_turbo` and the
// bucket-sorted `indexed_moe_mmq_q4_k` paths.
// ---------------------------------------------------------------------------

/// Buffers for [`Ops::indexed_moe_mmq_q4_k_gate_up_turbo`].
#[derive(Copy, Clone, Debug)]
pub struct MoeMmqQ4KGateUpTurboBuffers {
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub y_mmq: DevicePtr,
    pub expert_ids: DevicePtr,
    pub sorted_pair_idx_padded: DevicePtr,
    pub padded_offsets: DevicePtr,
    pub gate_out: DevicePtr,
    pub up_out: DevicePtr,
}

/// Buffers for [`Ops::indexed_moe_mmq_q4_k_down_turbo`].
#[derive(Copy, Clone, Debug)]
pub struct MoeMmqQ4KDownTurboBuffers {
    pub down_w: DevicePtr,
    pub y_mmq: DevicePtr,
    pub expert_ids: DevicePtr,
    pub sorted_pair_idx_padded: DevicePtr,
    pub padded_offsets: DevicePtr,
    pub dst: DevicePtr,
}

/// Buffers for the bucket-sorted [`Ops::indexed_moe_mmq_q4_k`].
#[derive(Copy, Clone, Debug)]
pub struct MoeMmqQ4KBuffers {
    pub w: DevicePtr,
    pub y: DevicePtr,
    pub bucket_expert: DevicePtr,
    pub bucket_slots: DevicePtr,
    pub dst: DevicePtr,
}

/// Shape for the bucket-sorted [`Ops::indexed_moe_mmq_q4_k`].
#[derive(Copy, Clone, Debug)]
pub struct MoeMmqQ4KShape {
    pub n_rows: usize,
    pub n_sb_per_row: usize,
    pub top_k: usize,
    pub n_buckets: usize,
}

// ---------------------------------------------------------------------------
// Per-file small launcher aggregates (Q3d).
// ---------------------------------------------------------------------------

/// Buffers for `attention::split_q_gate_f16` — interleaved (Q, gate)
/// projection split into two contiguous F16 tensors.
#[derive(Copy, Clone, Debug)]
pub struct SplitQGateBuffers {
    pub fused_qg: DevicePtr,
    pub q_out: DevicePtr,
    pub gate_out: DevicePtr,
}

/// Shape for `attention::split_q_gate_f16`.
#[derive(Copy, Clone, Debug)]
pub struct SplitQGateShape {
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

/// Buffers for `conv::causal_conv1d_f32`.
#[derive(Copy, Clone, Debug)]
pub struct ConvCausal1dBuffers {
    pub conv_input: DevicePtr,
    pub weight: DevicePtr,
    pub y: DevicePtr,
}

/// Shape for `conv::causal_conv1d_f32`.
#[derive(Copy, Clone, Debug)]
pub struct ConvCausal1dShape {
    pub n_new: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
}

/// Buffers for `moe::topk_f32` (router top-K).
#[derive(Copy, Clone, Debug)]
pub struct TopkBuffers {
    pub logits: DevicePtr,
    pub idx: DevicePtr,
    pub weights: DevicePtr,
}

/// Shape for `moe::topk_f32`.
#[derive(Copy, Clone, Debug)]
pub struct TopkShape {
    pub n_tokens: usize,
    pub n_experts: usize,
    pub k: usize,
}

/// Buffers shared by both `router::dense_gemv_{f16,f32}_f16_batched`.
#[derive(Copy, Clone, Debug)]
pub struct DenseGemvBatchedBuffers {
    pub w: DevicePtr,
    pub x: DevicePtr,
    pub y: DevicePtr,
}

/// Shape shared by both `router::dense_gemv_{f16,f32}_f16_batched`.
#[derive(Copy, Clone, Debug)]
pub struct DenseGemvBatchedShape {
    pub n_rows: usize,
    pub k: usize,
    pub n_tokens: usize,
}

/// Buffers for `sampling::topk_softmax_f32` (sampler — distinct from
/// `moe::topk_f32`'s router).
#[derive(Copy, Clone, Debug)]
pub struct SamplerTopkSoftmaxBuffers {
    pub logits: DevicePtr,
    pub out_ids: DevicePtr,
    pub out_probs: DevicePtr,
}

/// Shape + temperature inverse for `sampling::topk_softmax_f32`.
#[derive(Copy, Clone, Debug)]
pub struct SamplerTopkSoftmaxKnobs {
    pub vocab: usize,
    pub k: usize,
    pub inv_temp: f32,
}

/// Buffers for `softmax::softmax_masked_f16`.
#[derive(Copy, Clone, Debug)]
pub struct SoftmaxMaskedBuffers {
    pub scores: DevicePtr,
    pub mask: DevicePtr,
    pub out: DevicePtr,
}

/// Shape + scale for `softmax::softmax_masked_f16`.
#[derive(Copy, Clone, Debug)]
pub struct SoftmaxMaskedKnobs {
    pub m: usize,
    pub k: usize,
    pub scale: f32,
}
