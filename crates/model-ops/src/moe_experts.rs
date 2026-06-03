//! `MoeExperts` — Qwen3-MoE routed-experts block (decode + prefill).
//!
//! Decode pipeline:
//!
//! 1. (router, optional) `dense_gemv` over `ffn_gate_inp` + `topk_f32` →
//!    `expert_ids[top_k]`, `expert_weights[top_k]`.
//! 2. `quantize_f16_q8_1(x_norm)`.
//! 3. indexed gate+up per dtype (Q4_K / Q4_0 fused / Q8_0 unfused).
//! 4. `swiglu_f32_to_f16(gate, up)` + `quantize_f16_q8_1(activated)`.
//! 5. indexed down per dtype (each top_k slot as its own effective
//!    token with `top_k=1`).
//! 6. `cast_f32_to_f16(down)`.
//! 7. `moe_combine_f16(...)` (or `moe_combine_two_residuals_f16` if a
//!    shared-expert delta is provided as `extra_residual`).
//!
//! Prefill pipeline. Two dispatch shapes share the same router /
//! swiglu / down / combine bookends:
//!
//! * **Q4_0/Q8_0 short-prompt fast path** (`L < 32`): plain
//!   `indexed_moe_mmvq_*_gate_up` (no sort/pad), shared activations
//!   are reused as Q8_1 across the top_k slots.
//! * **tile8** (default for `L ≥ 32` and any Q4_K layout): expert-
//!   sorted + padded routing → 64×8-tile MMQ for gate+up and down.
//!   Down dtype dispatches to per-dtype tile8 kernels (Q4_K, Q4_0,
//!   Q8_0, Q4_1, Q5_K, Q6_K), each with its own super-block stride.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::device::DevicePtr;
use flambeau_core::op::QDtype;
use flambeau_ops::Ops;

use crate::driver_utils::RawAllocTracker;
use crate::weight_handle::WeightHandle;

/// Borrowed-by-value view of a caller-owned MoE decode scratch.
#[derive(Copy, Clone)]
pub struct MoeExpertsDecodeScratch {
    pub x_q8_1: DevicePtr,
    pub router_logits: DevicePtr,  // F32 [n_experts]
    pub expert_ids: DevicePtr,     // i32 [top_k]
    pub expert_weights: DevicePtr, // F32 [top_k]
    pub gate_out_f32: DevicePtr,   // F32 [top_k, intermediate]
    pub up_out_f32: DevicePtr,     // F32 [top_k, intermediate]
    pub activated_f16: DevicePtr,  // F16 [top_k, intermediate]
    pub activated_q8_1: DevicePtr, // Q8_1 [top_k, intermediate / 32]
    pub down_f32: DevicePtr,       // F32 [top_k, hidden]
    pub down_f16: DevicePtr,       // F16 [top_k, hidden]
}

/// Borrowed-by-value view of a caller-owned MoE prefill scratch.
/// All buffers are sized for `max_tokens` (the chunk's upper bound).
#[derive(Copy, Clone)]
pub struct MoeExpertsPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,                      // Q8_1 [L, hidden / 32]
    pub router_logits: DevicePtr,               // F32 [L, n_experts]
    pub expert_ids: DevicePtr,                  // i32 [L, top_k]
    pub expert_weights: DevicePtr,              // F32 [L, top_k]
    pub gate_out_f32: DevicePtr,                // F32 [L, top_k, intermediate]
    pub up_out_f32: DevicePtr,                  // F32 [L, top_k, intermediate]
    pub activated_f16: DevicePtr,               // F16 [L, top_k, intermediate]
    pub activated_q8_1: DevicePtr,              // Q8_1 [L, top_k, intermediate / 32]
    pub down_f32: DevicePtr,                    // F32 [L, top_k, hidden]
    pub down_f16: DevicePtr,                    // F16 [L, top_k, hidden]
    pub sort_counts: DevicePtr,                 // i32 [n_experts]
    pub sort_offsets: DevicePtr,                // i32 [n_experts + 1]
    pub sort_cursors: DevicePtr,                // i32 [n_experts]
    pub sort_sorted_pair_idx: DevicePtr,        // i32 [L * top_k]
    pub sort_padded_offsets: DevicePtr,         // i32 [n_experts + 1]
    pub sort_sorted_pair_idx_padded: DevicePtr, // i32 [L * top_k + n_experts * 8]
}

const QK_K: usize = 256;

/// Shape inputs needed to size a `MoeExperts` decode scratch.
#[derive(Copy, Clone, Debug)]
pub struct MoeExpertsScratchDims {
    pub hidden: usize,
    pub intermediate: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

/// Owned MoE decode scratch.
pub struct OwnedMoeExpertsDecodeScratch {
    pub x_q8_1: DevicePtr,
    pub router_logits: DevicePtr,
    pub expert_ids: DevicePtr,
    pub expert_weights: DevicePtr,
    pub gate_out_f32: DevicePtr,
    pub up_out_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
}

impl OwnedMoeExpertsDecodeScratch {
    pub fn view(&self) -> MoeExpertsDecodeScratch {
        MoeExpertsDecodeScratch {
            x_q8_1: self.x_q8_1,
            router_logits: self.router_logits,
            expert_ids: self.expert_ids,
            expert_weights: self.expert_weights,
            gate_out_f32: self.gate_out_f32,
            up_out_f32: self.up_out_f32,
            activated_f16: self.activated_f16,
            activated_q8_1: self.activated_q8_1,
            down_f32: self.down_f32,
            down_f16: self.down_f16,
        }
    }
}

/// Owned MoE prefill scratch.
pub struct OwnedMoeExpertsPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    pub router_logits: DevicePtr,
    pub expert_ids: DevicePtr,
    pub expert_weights: DevicePtr,
    pub gate_out_f32: DevicePtr,
    pub up_out_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
    pub sort_counts: DevicePtr,
    pub sort_offsets: DevicePtr,
    pub sort_cursors: DevicePtr,
    pub sort_sorted_pair_idx: DevicePtr,
    pub sort_padded_offsets: DevicePtr,
    pub sort_sorted_pair_idx_padded: DevicePtr,
}

impl OwnedMoeExpertsPrefillScratch {
    pub fn view(&self) -> MoeExpertsPrefillScratch {
        MoeExpertsPrefillScratch {
            max_tokens: self.max_tokens,
            x_q8_1: self.x_q8_1,
            router_logits: self.router_logits,
            expert_ids: self.expert_ids,
            expert_weights: self.expert_weights,
            gate_out_f32: self.gate_out_f32,
            up_out_f32: self.up_out_f32,
            activated_f16: self.activated_f16,
            activated_q8_1: self.activated_q8_1,
            down_f32: self.down_f32,
            down_f16: self.down_f16,
            sort_counts: self.sort_counts,
            sort_offsets: self.sort_offsets,
            sort_cursors: self.sort_cursors,
            sort_sorted_pair_idx: self.sort_sorted_pair_idx,
            sort_padded_offsets: self.sort_padded_offsets,
            sort_sorted_pair_idx_padded: self.sort_sorted_pair_idx_padded,
        }
    }
}

/// Source of the router input. Qwen3.x routes on the post-norm
/// hidden (`Cur`); Gemma4 routes on the residual stream `attn_out`.
/// The block does not pick the buffer — this field documents the
/// arch convention so the caller can pass the correct DevicePtr as
/// `x_norm`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RouterInput {
    /// Post-norm hidden state (qwen3.x).
    Cur,
    /// Pre-norm residual stream `attn_out` (gemma4).
    AttnOut,
}

/// Expert activation function. Used between gate/up and down.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum Activation {
    /// `silu(gate) * up` — qwen3.x convention.
    #[default]
    SwiGLU,
    /// `gelu(gate) * up` — gemma4 convention (ggml tanh-approximation GELU).
    Gelu,
}


/// How router logits become per-expert weights.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RouterNormalize {
    /// Pick top-k highest logits, then softmax over the captured k
    /// raw values (sum to 1). Matches qwen3.x convention and the
    /// current `topk_f32` kernel. Equivalent to llama.cpp's
    /// `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` + `norm_topk_prob`.
    TopkRenorm,
    /// Softmax over all `n_experts`, then take top-k indices keeping
    /// the softmax-over-all probability for each kept slot (no
    /// renorm). Matches gemma4 (`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`
    /// without renorm). The kernel arrives in S3 — selecting this
    /// variant before then errors out at `route_*` call time.
    Softmax,
}

/// Router shape: input source, optional pre-router scale (vector
/// `[hidden]` multiplied element-wise into `x_norm` before the
/// router matmul), optional pre-router scalar, and the gate
/// normalisation policy. Default = qwen3.x convention.
#[derive(Copy, Clone)]
pub struct RouterPolicy {
    pub input: RouterInput,
    /// F32 [hidden] device pointer. When `Some`, the model crate is
    /// expected to broadcast-multiply it into the router input
    /// before calling `route_*`. The handle is stored here so the
    /// block can validate / pass it to a future fused pre-scale
    /// kernel; the current API does no implicit application.
    pub pre_scale: Option<DevicePtr>,
    /// Scalar applied to the router input. Defaults to `1.0`. Gemma4
    /// uses `1.0 / sqrt(n_embd)`. Same delivery contract as
    /// `pre_scale` — the caller applies it.
    pub pre_scalar: f32,
    pub normalize: RouterNormalize,
}

impl Default for RouterPolicy {
    fn default() -> Self {
        Self {
            input: RouterInput::Cur,
            pre_scale: None,
            pre_scalar: 1.0,
            normalize: RouterNormalize::TopkRenorm,
        }
    }
}

/// Qwen3-MoE routed-experts block (decode-only V1).
///
/// Holds: router weight (`ffn_gate_inp`), per-expert gate/up/down
/// indexed weights, and the shape scalars. The block does NOT own the
/// shared-expert path; that stays in the model crate.
pub struct MoeExperts {
    pub ffn_gate_inp: WeightHandle,  // [n_experts, hidden]
    pub ffn_gate_exps: WeightHandle, // [n_experts, intermediate, hidden]
    pub ffn_up_exps: WeightHandle,   // dito
    pub ffn_down_exps: WeightHandle, // [n_experts, hidden, intermediate]
    pub hidden: usize,
    pub intermediate: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub router_policy: RouterPolicy,
    /// Activation between gate/up and down. Default `SwiGLU` (qwen3.x);
    /// gemma4 sets this to `Gelu`.
    pub activation: Activation,
    /// Minimum `prompt_len` at which the tile8 path engages, in
    /// addition to the universal `n_pairs >= TILE8_PAIRS_MIN` gate.
    /// `None` ⇒ no extra minimum (tile8 engages whenever the pair
    /// count clears 8). qwen3-moe TP sets this to `Some(32)` when
    /// `tp_world >= 2` to avoid the AR-vs-tile8 launch serialisation
    /// observed on tp2 / N=4 batched decode (the cert noted -10% wall;
    /// see `crates/models/qwen3-moe/src/forward/moe_tp.rs`).
    pub tile8_min_tokens: Option<usize>,
}

/// Construction-time weight handles for [`MoeExperts::new`].
#[derive(Copy, Clone)]
pub struct MoeExpertsWeights {
    pub ffn_gate_inp: WeightHandle,
    pub ffn_gate_exps: WeightHandle,
    pub ffn_up_exps: WeightHandle,
    pub ffn_down_exps: WeightHandle,
}

/// Construction-time dimension scalars for [`MoeExperts::new`].
#[derive(Copy, Clone, Debug)]
pub struct MoeExpertsDims {
    pub hidden: usize,
    pub intermediate: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

/// Prefill I/O pointers consumed by [`MoeExperts::forward_prefill`].
#[derive(Copy, Clone, Debug)]
pub struct MoeExpertsPrefillBuffers {
    pub x_norm: DevicePtr,
    pub residual: DevicePtr,
    pub extra_residual: Option<DevicePtr>,
    pub out: DevicePtr,
}

impl MoeExperts {
    pub fn new(weights: MoeExpertsWeights, dims: MoeExpertsDims) -> Result<Self> {
        if weights.ffn_gate_inp.dims != [dims.n_experts, dims.hidden] {
            bail!(
                "ffn_gate_inp dims {:?} != [{}, {}]",
                weights.ffn_gate_inp.dims,
                dims.n_experts,
                dims.hidden
            );
        }
        // Indexed expert weights flatten the outer-most `n_experts`
        // dim into the row count: gate/up = n_experts × intermediate
        // × hidden, but stored as a 2-D `[n_experts * intermediate,
        // hidden]` slab. Skip the strict dim assert here — loaders
        // already normalise this.
        Ok(Self {
            ffn_gate_inp: weights.ffn_gate_inp,
            ffn_gate_exps: weights.ffn_gate_exps,
            ffn_up_exps: weights.ffn_up_exps,
            ffn_down_exps: weights.ffn_down_exps,
            hidden: dims.hidden,
            intermediate: dims.intermediate,
            n_experts: dims.n_experts,
            top_k: dims.top_k,
            router_policy: RouterPolicy::default(),
            activation: Activation::default(),
            tile8_min_tokens: None,
        })
    }

    /// Override the router policy. Default keeps the qwen3.x
    /// convention; gemma4 sets `{ input: AttnOut, pre_scale:
    /// Some(ffn_gate_inp_s), pre_scalar: 1.0 / sqrt(hidden),
    /// normalize: Softmax }`.
    pub fn with_router_policy(mut self, policy: RouterPolicy) -> Self {
        self.router_policy = policy;
        self
    }

    /// Override the activation. Default `SwiGLU`; gemma4 uses `Gelu`.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set a minimum `prompt_len` for tile8 prefill engagement (on top
    /// of the universal `n_pairs >= 8` gate). qwen3-moe TP sets this
    /// to 32 when `tp_world >= 2`.
    pub fn with_tile8_min_tokens(mut self, n: usize) -> Self {
        self.tile8_min_tokens = Some(n);
        self
    }

    pub fn scratch_dims(&self) -> MoeExpertsScratchDims {
        MoeExpertsScratchDims {
            hidden: self.hidden,
            intermediate: self.intermediate,
            n_experts: self.n_experts,
            top_k: self.top_k,
        }
    }

    /// Allocate an [`OwnedMoeExpertsPrefillScratch`] sized for `dims`
    /// × `max_tokens`.
    pub fn alloc_prefill_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: MoeExpertsScratchDims,
        max_tokens: usize,
    ) -> Result<OwnedMoeExpertsPrefillScratch> {
        if max_tokens == 0 {
            bail!("alloc_prefill_scratch: max_tokens must be >= 1");
        }
        let MoeExpertsScratchDims {
            hidden,
            intermediate,
            n_experts,
            top_k,
        } = dims;
        let (x_q8_1, _) = tracker.alloc_q8_1(device, max_tokens * hidden)?;
        let (router_logits, _) = tracker.alloc_f32(device, max_tokens * n_experts)?;
        let (expert_ids, _) = tracker.alloc_i32(device, max_tokens * top_k)?;
        let (expert_weights, _) = tracker.alloc_f32(device, max_tokens * top_k)?;
        let (gate_out_f32, _) = tracker.alloc_f32(device, max_tokens * top_k * intermediate)?;
        let (up_out_f32, _) = tracker.alloc_f32(device, max_tokens * top_k * intermediate)?;
        let (activated_f16, _) = tracker.alloc_f16(device, max_tokens * top_k * intermediate)?;
        let (activated_q8_1, _) = tracker.alloc_q8_1(device, max_tokens * top_k * intermediate)?;
        let (down_f32, _) = tracker.alloc_f32(device, max_tokens * top_k * hidden)?;
        let (down_f16, _) = tracker.alloc_f16(device, max_tokens * top_k * hidden)?;
        let (sort_counts, _) = tracker.alloc_i32(device, n_experts)?;
        let (sort_offsets, _) = tracker.alloc_i32(device, n_experts + 1)?;
        let (sort_cursors, _) = tracker.alloc_i32(device, n_experts)?;
        let (sort_sorted_pair_idx, _) = tracker.alloc_i32(device, max_tokens * top_k)?;
        let (sort_padded_offsets, _) = tracker.alloc_i32(device, n_experts + 1)?;
        let (sort_sorted_pair_idx_padded, _) =
            tracker.alloc_i32(device, max_tokens * top_k + n_experts * 16)?;
        Ok(OwnedMoeExpertsPrefillScratch {
            max_tokens,
            x_q8_1,
            router_logits,
            expert_ids,
            expert_weights,
            gate_out_f32,
            up_out_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            down_f16,
            sort_counts,
            sort_offsets,
            sort_cursors,
            sort_sorted_pair_idx,
            sort_padded_offsets,
            sort_sorted_pair_idx_padded,
        })
    }

    /// Allocate an [`OwnedMoeExpertsDecodeScratch`] sized for `dims`.
    /// `x_q8_1` is sized for `hidden` (single-token router input);
    /// per-slot buffers are sized `top_k * intermediate` / `top_k *
    /// hidden`.
    pub fn alloc_decode_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: MoeExpertsScratchDims,
    ) -> Result<OwnedMoeExpertsDecodeScratch> {
        let MoeExpertsScratchDims {
            hidden,
            intermediate,
            n_experts,
            top_k,
        } = dims;
        let (x_q8_1, _) = tracker.alloc_q8_1(device, hidden)?;
        let (router_logits, _) = tracker.alloc_f32(device, n_experts)?;
        let (expert_ids, _) = tracker.alloc_i32(device, top_k)?;
        let (expert_weights, _) = tracker.alloc_f32(device, top_k)?;
        let (gate_out_f32, _) = tracker.alloc_f32(device, top_k * intermediate)?;
        let (up_out_f32, _) = tracker.alloc_f32(device, top_k * intermediate)?;
        let (activated_f16, _) = tracker.alloc_f16(device, top_k * intermediate)?;
        let (activated_q8_1, _) = tracker.alloc_q8_1(device, top_k * intermediate)?;
        let (down_f32, _) = tracker.alloc_f32(device, top_k * hidden)?;
        let (down_f16, _) = tracker.alloc_f16(device, top_k * hidden)?;
        Ok(OwnedMoeExpertsDecodeScratch {
            x_q8_1,
            router_logits,
            expert_ids,
            expert_weights,
            gate_out_f32,
            up_out_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            down_f16,
        })
    }

    /// Run the dense-router GEMV + topk that populates
    /// `scratch.expert_ids` and `scratch.expert_weights`. Caller can
    /// skip this when synthesising routing decisions (test fixtures).
    pub fn route_decode<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        scratch: MoeExpertsDecodeScratch,
    ) -> Result<()> {
        match self.ffn_gate_inp.dtype {
            QDtype::F16 => ops
                .dense_gemv_f16_f16(
                    self.ffn_gate_inp.ptr,
                    x_norm,
                    scratch.router_logits,
                    self.n_experts,
                    self.hidden,
                )
                .context("router dense_gemv_f16_f16")?,
            QDtype::F32 => ops
                .dense_gemv_f32_f16(
                    self.ffn_gate_inp.ptr,
                    x_norm,
                    scratch.router_logits,
                    self.n_experts,
                    self.hidden,
                )
                .context("router dense_gemv_f32_f16")?,
            other => bail!("router expects F32 or F16 ffn_gate_inp; got {other:?}"),
        }
        match self.router_policy.normalize {
            RouterNormalize::TopkRenorm => ops
                .topk_f32(
                    scratch.router_logits,
                    scratch.expert_ids,
                    scratch.expert_weights,
                    1,
                    self.n_experts,
                    self.top_k,
                )
                .context("router topk_f32")?,
            RouterNormalize::Softmax => bail!(
                "RouterNormalize::Softmax requires the softmax-then-topk kernel landing in S3"
            ),
        }
        Ok(())
    }

    fn gate_up<O: Ops>(&self, ops: &O, scratch: MoeExpertsDecodeScratch) -> Result<()> {
        let inter = self.intermediate;
        let n_tokens = 1usize;
        let top_k = self.top_k;
        let hidden = self.hidden;
        let dtype = self.ffn_gate_exps.dtype;
        macro_rules! iq_gate_up_split {
            ($ops:ident, $self:ident, $scratch:ident, $op:ident, $tag:literal,
             $nb:expr, $inter:ident, $n_tokens:ident, $top_k:ident) => {{
                let nb = $nb;
                $ops.$op(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: $self.ffn_gate_exps.ptr,
                        act: $scratch.x_q8_1,
                        expert_ids: $scratch.expert_ids,
                        dst: $scratch.gate_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: $inter, n_tokens: $n_tokens, top_k: $top_k, n_sb_per_row: nb,
                    },
                ).context(concat!("indexed_moe gate ", $tag, " split"))?;
                $ops.$op(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: $self.ffn_up_exps.ptr,
                        act: $scratch.x_q8_1,
                        expert_ids: $scratch.expert_ids,
                        dst: $scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: $inter, n_tokens: $n_tokens, top_k: $top_k, n_sb_per_row: nb,
                    },
                ).context(concat!("indexed_moe up ", $tag, " split"))
            }};
        }
        match dtype {
            QDtype::Q4_K => {
                let nb = hidden / QK_K;
                ops.indexed_moe_mmvq_q4_k_gate_up(
                    flambeau_ops::MoeMmvqGateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe gate+up q4_k")
            }
            QDtype::Q8_0 => {
                let nb = hidden / 32;
                ops.indexed_moe_mmvq_q8_0_gate_up(
                    flambeau_ops::MoeMmvqGateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe gate+up q8_0 fused")
            }
            QDtype::Q4_0 => {
                let nb = hidden / 32;
                ops.indexed_moe_mmvq_q4_0_gate_up(
                    flambeau_ops::MoeMmvqGateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe gate+up q4_0 fused")
            }
            QDtype::Q3_K => {
                let nb = hidden / QK_K;
                ops.indexed_moe_mmvq_q3_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_gate_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.gate_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe gate q3_k split")?;
                ops.indexed_moe_mmvq_q3_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe up q3_k split")
            }
            QDtype::Q5_K => {
                let nb = hidden / QK_K;
                ops.indexed_moe_mmvq_q5_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_gate_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.gate_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe gate q5_k split")?;
                ops.indexed_moe_mmvq_q5_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe up q5_k split")
            }
            QDtype::Q6_K => {
                let nb = hidden / QK_K;
                ops.indexed_moe_mmvq_q6_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_gate_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.gate_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe gate q6_k split")?;
                ops.indexed_moe_mmvq_q6_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: n_tokens,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe up q6_k split")
            }
            QDtype::IQ4_XS  => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq4_xs,  "iq4_xs",  hidden / QK_K, inter, n_tokens, top_k),
            QDtype::IQ4_NL  => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq4_nl,  "iq4_nl",  hidden / 32,   inter, n_tokens, top_k),
            QDtype::IQ3_XXS => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq3_xxs, "iq3_xxs", hidden / QK_K, inter, n_tokens, top_k),
            QDtype::IQ3_S   => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq3_s,   "iq3_s",   hidden / QK_K, inter, n_tokens, top_k),
            QDtype::IQ2_XXS => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq2_xxs, "iq2_xxs", hidden / QK_K, inter, n_tokens, top_k),
            QDtype::IQ2_XS  => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq2_xs,  "iq2_xs",  hidden / QK_K, inter, n_tokens, top_k),
            QDtype::IQ2_S   => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq2_s,   "iq2_s",   hidden / QK_K, inter, n_tokens, top_k),
            QDtype::IQ1_S   => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq1_s,   "iq1_s",   hidden / QK_K, inter, n_tokens, top_k),
            QDtype::IQ1_M   => iq_gate_up_split!(ops, self, scratch, indexed_moe_mmvq_iq1_m,   "iq1_m",   hidden / QK_K, inter, n_tokens, top_k),
            other => {
                bail!("MoeExperts gate dtype {other:?} not supported (expected Q4_K / Q3_K / Q5_K / Q6_K / Q8_0 / Q4_0 / IQ family)")
            }
        }
    }

    fn down<O: Ops>(&self, ops: &O, scratch: MoeExpertsDecodeScratch) -> Result<()> {
        let inter = self.intermediate;
        let hidden = self.hidden;
        let n_tokens_eff = self.top_k;
        let top_k_inner = 1;
        let dtype = self.ffn_down_exps.dtype;
        macro_rules! iq_down {
            ($ops:ident, $self:ident, $scratch:ident, $op:ident, $tag:literal,
             $nb:expr, $hidden:ident, $n_tokens_eff:ident, $top_k_inner:ident) => {{
                let nb = $nb;
                $ops.$op(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: $self.ffn_down_exps.ptr,
                        act: $scratch.activated_q8_1,
                        expert_ids: $scratch.expert_ids,
                        dst: $scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: $hidden, n_tokens: $n_tokens_eff, top_k: $top_k_inner, n_sb_per_row: nb,
                    },
                ).context(concat!("indexed_moe down ", $tag))
            }};
        }
        match dtype {
            QDtype::Q4_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q4_k_r2(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_tokens_eff,
                        top_k: top_k_inner,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe down q4_k r2")
            }
            QDtype::Q5_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q5_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_tokens_eff,
                        top_k: top_k_inner,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe down q5_k")
            }
            QDtype::Q6_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q6_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_tokens_eff,
                        top_k: top_k_inner,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe down q6_k")
            }
            QDtype::Q3_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q3_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_tokens_eff,
                        top_k: top_k_inner,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe down q3_k")
            }
            QDtype::Q8_0 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q8_0(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_tokens_eff,
                        top_k: top_k_inner,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe down q8_0")
            }
            QDtype::Q4_0 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q4_0(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_tokens_eff,
                        top_k: top_k_inner,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe down q4_0")
            }
            QDtype::Q4_1 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q4_1(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_tokens_eff,
                        top_k: top_k_inner,
                        n_sb_per_row: nb,
                    },
                )
                .context("indexed_moe down q4_1")
            }
            QDtype::IQ4_XS  => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq4_xs,  "iq4_xs",  inter / QK_K, hidden, n_tokens_eff, top_k_inner),
            QDtype::IQ4_NL  => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq4_nl,  "iq4_nl",  inter / 32,   hidden, n_tokens_eff, top_k_inner),
            QDtype::IQ3_XXS => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq3_xxs, "iq3_xxs", inter / QK_K, hidden, n_tokens_eff, top_k_inner),
            QDtype::IQ3_S   => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq3_s,   "iq3_s",   inter / QK_K, hidden, n_tokens_eff, top_k_inner),
            QDtype::IQ2_XXS => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq2_xxs, "iq2_xxs", inter / QK_K, hidden, n_tokens_eff, top_k_inner),
            QDtype::IQ2_XS  => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq2_xs,  "iq2_xs",  inter / QK_K, hidden, n_tokens_eff, top_k_inner),
            QDtype::IQ2_S   => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq2_s,   "iq2_s",   inter / QK_K, hidden, n_tokens_eff, top_k_inner),
            QDtype::IQ1_S   => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq1_s,   "iq1_s",   inter / QK_K, hidden, n_tokens_eff, top_k_inner),
            QDtype::IQ1_M   => iq_down!(ops, self, scratch, indexed_moe_mmvq_iq1_m,   "iq1_m",   inter / QK_K, hidden, n_tokens_eff, top_k_inner),
            other => bail!(
                "MoeExperts down dtype {other:?} not supported (expected Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0 / Q4_1 / IQ family)"
            ),
        }
    }

    /// One decode step. Caller has populated `expert_ids` and
    /// `expert_weights` (via [`route_decode`] or a synthetic test
    /// fixture). Writes
    /// `out = residual + (extra_residual?) + Σ weight_k · expert_out_k`.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        residual: DevicePtr,
        extra_residual: Option<DevicePtr>,
        out: DevicePtr,
        scratch: MoeExpertsDecodeScratch,
    ) -> Result<()> {
        let hidden = self.hidden;
        let inter = self.intermediate;
        let top_k = self.top_k;

        // 1. Quantise x_norm → Q8_1.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, hidden)
            .context("moe x_norm → Q8_1")?;

        // 2. indexed gate + up per expert dtype.
        self.gate_up(ops, scratch)?;

        // 3+4. Fused activation → F16 + Q8_1 quantise. SwiGLU at
        // multiples of QK8_1=32 takes the single-launch fused
        // `swiglu_f32_to_q8_1` fast path (saves the F16 cast +
        // quantise pair). GELU has no fused-q8_1 sibling kernel.
        let n_total = top_k * inter;
        let fuse_swiglu_quant = matches!(self.activation, Activation::SwiGLU) && n_total % 32 == 0;
        if fuse_swiglu_quant {
            ops.swiglu_f32_to_q8_1(
                scratch.gate_out_f32,
                scratch.up_out_f32,
                scratch.activated_q8_1,
                n_total,
            )
            .context("moe swiglu_f32_to_q8_1")?;
        } else {
            match self.activation {
                Activation::SwiGLU => ops
                    .swiglu_f32_to_f16(
                        scratch.gate_out_f32,
                        scratch.up_out_f32,
                        scratch.activated_f16,
                        n_total,
                    )
                    .context("moe swiglu_f32_to_f16")?,
                Activation::Gelu => ops
                    .gelu_f32_to_f16(
                        scratch.gate_out_f32,
                        scratch.up_out_f32,
                        scratch.activated_f16,
                        n_total,
                    )
                    .context("moe gelu_f32_to_f16")?,
            }
            ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, n_total)
                .context("moe quantize activated → Q8_1")?;
        }

        // 5. indexed down. Treats each top_k slot as its own effective
        // token with top_k=1 — the kernel's expert lookup collapses to
        // a flat index into `scratch.expert_ids[0..top_k]`.
        self.down(ops, scratch)?;

        // 6. Cast expert outputs to F16 for the combine kernel.
        ops.cast_f32_to_f16(scratch.down_f32, scratch.down_f16, top_k * hidden)
            .context("moe cast down → f16")?;

        // 7. Weighted sum + residual (and optional shared-expert
        // delta as a fused two-residual combine).
        if let Some(extra) = extra_residual {
            ops.moe_combine_two_residuals_f16(
                flambeau_ops::MoeCombineTwoResidualsBuffers {
                    expert_outs: scratch.down_f16,
                    weights: scratch.expert_weights,
                    residual1: residual,
                    residual2: extra,
                    out,
                },
                flambeau_ops::MoeCombineShape {
                    n_tokens: 1,
                    top_k,
                    hidden,
                },
            )
            .context("moe_combine_two_residuals_f16")?;
        } else {
            ops.moe_combine_f16(
                flambeau_ops::MoeCombineBuffers {
                    expert_outs: scratch.down_f16,
                    weights: scratch.expert_weights,
                    residual,
                    out,
                },
                flambeau_ops::MoeCombineShape {
                    n_tokens: 1,
                    top_k,
                    hidden,
                },
            )
            .context("moe_combine_f16")?;
        }
        Ok(())
    }

    /// Per-rank decode for the TP path. Same kernel sequence as
    /// [`Self::forward_decode`] but operates on per-rank sliced expert
    /// weights (`self.intermediate` set to `local_inter`) and emits a
    /// `partial_out` `[hidden]` via `moe_combine_no_residual_f16` — the
    /// caller's AR folds the residual + cross-rank sum together.
    /// `self.intermediate` MUST equal `moe_intermediate_size / tp_world`;
    /// shape checks happen at `new()`.
    pub fn forward_decode_tp<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        partial_out: DevicePtr,
        scratch: MoeExpertsDecodeScratch,
    ) -> Result<()> {
        let hidden = self.hidden;
        let inter = self.intermediate;
        let top_k = self.top_k;

        // 1. Quantise x_norm → Q8_1.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, hidden)
            .context("moe (TP) x_norm → Q8_1")?;

        // 2. Indexed gate + up. `inter` is `local_inter` from the
        // outer perspective; the indexed-MoE kernels are oblivious.
        self.gate_up(ops, scratch)?;

        // 3+4. Fused activation → Q8_1 (SwiGLU @ multiples of QK8_1=32)
        // or unfused (GELU / off-multiples).
        let n_total = top_k * inter;
        let fuse_swiglu_quant = matches!(self.activation, Activation::SwiGLU) && n_total % 32 == 0;
        if fuse_swiglu_quant {
            ops.swiglu_f32_to_q8_1(
                scratch.gate_out_f32,
                scratch.up_out_f32,
                scratch.activated_q8_1,
                n_total,
            )
            .context("moe (TP) swiglu_f32_to_q8_1")?;
        } else {
            match self.activation {
                Activation::SwiGLU => ops
                    .swiglu_f32_to_f16(
                        scratch.gate_out_f32,
                        scratch.up_out_f32,
                        scratch.activated_f16,
                        n_total,
                    )
                    .context("moe (TP) swiglu_f32_to_f16")?,
                Activation::Gelu => ops
                    .gelu_f32_to_f16(
                        scratch.gate_out_f32,
                        scratch.up_out_f32,
                        scratch.activated_f16,
                        n_total,
                    )
                    .context("moe (TP) gelu_f32_to_f16")?,
            }
            ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, n_total)
                .context("moe (TP) quantize activated → Q8_1")?;
        }

        // 5. Indexed down on the sliced ffn_down_exps.
        self.down(ops, scratch)?;

        // 6. Cast expert outputs to F16 for the combine kernel.
        ops.cast_f32_to_f16(scratch.down_f32, scratch.down_f16, top_k * hidden)
            .context("moe (TP) cast down → f16")?;

        // 7. Weighted sum WITHOUT residual: partial = Σ w_k · down_k.
        // Residual is folded by the AR that follows.
        ops.moe_combine_no_residual_f16(
            flambeau_ops::MoeCombineNoResidualBuffers {
                expert_outs: scratch.down_f16,
                weights: scratch.expert_weights,
                out: partial_out,
            },
            flambeau_ops::MoeCombineShape {
                n_tokens: 1,
                top_k,
                hidden,
            },
        )
        .context("moe (TP) combine_no_residual_f16")?;
        Ok(())
    }

    /// F32-output sibling of [`Self::forward_decode_tp`]. Skips the
    /// `cast_f32_to_f16(down_f32, down_f16)` step and emits a F32 partial
    /// via `moe_combine_no_residual_f32`. Required by the gemma4
    /// head_dim=512 + Q8_0 path where V-norm spikes propagate into down
    /// outputs and the F16 cast saturates. Caller pairs this with an F32
    /// AllReduce (`tp_allreduce_sum_f32`) and a F32 post-norm cascade.
    pub fn forward_decode_tp_f32<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        partial_out_f32: DevicePtr,
        scratch: MoeExpertsDecodeScratch,
    ) -> Result<()> {
        let hidden = self.hidden;
        let inter = self.intermediate;
        let top_k = self.top_k;

        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, hidden)
            .context("moe (TP-F32) x_norm → Q8_1")?;
        self.gate_up(ops, scratch)?;
        let n_total = top_k * inter;
        let fuse_swiglu_quant = matches!(self.activation, Activation::SwiGLU) && n_total % 32 == 0;
        if fuse_swiglu_quant {
            ops.swiglu_f32_to_q8_1(
                scratch.gate_out_f32,
                scratch.up_out_f32,
                scratch.activated_q8_1,
                n_total,
            )
            .context("moe (TP-F32) swiglu_f32_to_q8_1")?;
        } else {
            match self.activation {
                Activation::SwiGLU => ops
                    .swiglu_f32_to_f16(
                        scratch.gate_out_f32,
                        scratch.up_out_f32,
                        scratch.activated_f16,
                        n_total,
                    )
                    .context("moe (TP-F32) swiglu_f32_to_f16")?,
                Activation::Gelu => ops
                    .gelu_f32_to_f16(
                        scratch.gate_out_f32,
                        scratch.up_out_f32,
                        scratch.activated_f16,
                        n_total,
                    )
                    .context("moe (TP-F32) gelu_f32_to_f16")?,
            }
            ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, n_total)
                .context("moe (TP-F32) quantize activated → Q8_1")?;
        }
        self.down(ops, scratch)?;
        // F32 combine: read F32 down outputs directly, write F32 partial.
        ops.moe_combine_no_residual_f32(
            flambeau_ops::MoeCombineNoResidualBuffers {
                expert_outs: scratch.down_f32,
                weights: scratch.expert_weights,
                out: partial_out_f32,
            },
            flambeau_ops::MoeCombineShape {
                n_tokens: 1,
                top_k,
                hidden,
            },
        )
        .context("moe (TP-F32) combine_no_residual_f32")?;
        Ok(())
    }

    /// Multi-token router. Same shape as `route_decode` but uses the
    /// batched dense_gemv kernel + L-aware topk.
    pub fn route_prefill<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        prompt_len: usize,
        scratch: MoeExpertsPrefillScratch,
    ) -> Result<()> {
        match self.ffn_gate_inp.dtype {
            QDtype::F16 => ops
                .dense_gemv_f16_f16_batched(
                    self.ffn_gate_inp.ptr,
                    x_norm,
                    scratch.router_logits,
                    self.n_experts,
                    self.hidden,
                    prompt_len,
                )
                .context("router prefill dense_gemv_f16_f16_batched")?,
            QDtype::F32 => ops
                .dense_gemv_f32_f16_batched(
                    self.ffn_gate_inp.ptr,
                    x_norm,
                    scratch.router_logits,
                    self.n_experts,
                    self.hidden,
                    prompt_len,
                )
                .context("router prefill dense_gemv_f32_f16_batched")?,
            other => bail!("router expects F32 or F16 ffn_gate_inp; got {other:?}"),
        }
        match self.router_policy.normalize {
            RouterNormalize::TopkRenorm => ops
                .topk_f32(
                    scratch.router_logits,
                    scratch.expert_ids,
                    scratch.expert_weights,
                    prompt_len,
                    self.n_experts,
                    self.top_k,
                )
                .context("router prefill topk_f32")?,
            RouterNormalize::Softmax => bail!(
                "RouterNormalize::Softmax requires the softmax-then-topk kernel landing in S3"
            ),
        }
        Ok(())
    }

    /// Threshold above which the tile8 path beats the plain MMVQ path
    /// for Q4_0 / Q8_0 prefill. Below this, tile8's grid overhead
    /// dominates and the per-token MMVQ wins.
    /// Historically `prompt_len >= 32` (a prefill-only condition). At
    /// batched-decode `prompt_len = N` is small but `n_pairs = N * top_k`
    /// lands in the tile8 sweet spot (top_k=4 → N=2 hits 8, N=4 hits 16).
    /// Both prefill and batched-decode now share the
    /// `n_pairs >= TILE8_PAIRS_MIN` gate.
    const TILE8_PAIRS_MIN: usize = 8;

    /// Multi-token routed-experts prefill. Caller has already
    /// populated `expert_ids` / `expert_weights` (via
    /// `route_prefill`). Writes
    /// `out = residual + extra_residual? + Σ w_k · expert_k(x_norm)`
    /// at every prompt position.
    pub fn forward_prefill<O: Ops>(
        &self,
        ops: &O,
        buffers: MoeExpertsPrefillBuffers,
        prompt_len: usize,
        scratch: MoeExpertsPrefillScratch,
    ) -> Result<()> {
        self.prefill_compute_expert_outs(ops, buffers.x_norm, prompt_len, scratch)?;
        // 7. Weighted sum + residual (+ optional shared-expert delta).
        self.combine_prefill(
            ops,
            scratch,
            buffers.residual,
            buffers.extra_residual,
            buffers.out,
            prompt_len,
        )
    }

    /// Per-rank routed-experts prefill (TP). Same kernel sequence as
    /// [`Self::forward_prefill`] but writes the per-rank partial via
    /// `moe_combine_no_residual_f16`; caller's AR folds the residual
    /// + cross-rank sum together. `self.intermediate` must equal
    ///   `moe_intermediate_size / tp_world`. Set
    ///   [`Self::with_tile8_min_tokens`] to gate the tile8 path on a
    ///   prompt-length floor when running at `tp_world >= 2`.
    pub fn forward_prefill_tp<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        partial_out: DevicePtr,
        prompt_len: usize,
        scratch: MoeExpertsPrefillScratch,
    ) -> Result<()> {
        self.prefill_compute_expert_outs(ops, x_norm, prompt_len, scratch)?;
        ops.moe_combine_no_residual_f16(
            flambeau_ops::MoeCombineNoResidualBuffers {
                expert_outs: scratch.down_f16,
                weights: scratch.expert_weights,
                out: partial_out,
            },
            flambeau_ops::MoeCombineShape {
                n_tokens: prompt_len,
                top_k: self.top_k,
                hidden: self.hidden,
            },
        )
        .context("prefill (TP) combine_no_residual_f16")
    }

    /// F32-output sibling of [`Self::forward_prefill_tp`]. Reads
    /// `scratch.down_f32` directly (skipping the cast that
    /// `prefill_compute_expert_outs` performs at the end into
    /// `down_f16`) and emits a `[prompt_len, hidden]` F32 partial via
    /// `moe_combine_no_residual_f32`. Caller pairs this with an F32
    /// AllReduce (`tp_allreduce_sum_f32` on SD/PP no-op) and a F32→F16
    /// cast for the residual path. Matches the v2 forward composite's
    /// `forward_decode_tp_f32` shape.
    pub fn forward_prefill_tp_f32<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        partial_out_f32: DevicePtr,
        prompt_len: usize,
        scratch: MoeExpertsPrefillScratch,
    ) -> Result<()> {
        self.prefill_compute_expert_outs(ops, x_norm, prompt_len, scratch)?;
        ops.moe_combine_no_residual_f32(
            flambeau_ops::MoeCombineNoResidualBuffers {
                expert_outs: scratch.down_f32,
                weights: scratch.expert_weights,
                out: partial_out_f32,
            },
            flambeau_ops::MoeCombineShape {
                n_tokens: prompt_len,
                top_k: self.top_k,
                hidden: self.hidden,
            },
        )
        .context("prefill (TP-F32) combine_no_residual_f32")
    }

    /// Steps 1-6 of the prefill pipeline. Writes the per-pair F16
    /// expert outputs into `scratch.down_f16` (`[prompt_len * top_k,
    /// hidden]`). Routing decisions in `scratch.expert_ids` /
    /// `scratch.expert_weights` must already be populated by the
    /// caller (`route_prefill` typically). The helper picks the mmvq
    /// short-prompt fallback vs. tile8 dispatch per the configured
    /// thresholds.
    fn prefill_compute_expert_outs<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        prompt_len: usize,
        scratch: MoeExpertsPrefillScratch,
    ) -> Result<()> {
        if prompt_len == 0 {
            bail!("MoeExperts::forward_prefill called with prompt_len = 0");
        }
        if prompt_len > scratch.max_tokens {
            bail!(
                "MoeExperts::forward_prefill: prompt_len={prompt_len} > scratch.max_tokens={}",
                scratch.max_tokens
            );
        }
        let hidden = self.hidden;
        let inter = self.intermediate;
        let top_k = self.top_k;
        let n_experts = self.n_experts;
        let n_pairs = prompt_len * top_k;
        let n_sb_per_row_inter_kk = inter / QK_K;
        let n_sb_per_row_hidden_kk = hidden / QK_K;
        let n_sb_per_row_inter_32 = inter / 32;
        let n_sb_per_row_hidden_32 = hidden / 32;

        let gate_dt = self.ffn_gate_exps.dtype;
        let up_dt = self.ffn_up_exps.dtype;
        let down_dt = self.ffn_down_exps.dtype;

        // 1. Quantise x_norm [L, hidden] → Q8_1.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, prompt_len * hidden)
            .context("prefill moe x_norm → Q8_1")?;

        // 2. Q4_0 / Q8_0 short-prompt fast path. Uses plain indexed
        // MoE MMVQ (no sort/pad). Allowed combos: Q4_0 gate+up with
        // Q4_0 / Q8_0 / Q4_1 down; Q8_0 gate+up with Q8_0 down.
        let allow_tile8_decode = std::env::var("FLAMBEAU_MOE_TILE8_DECODE").as_deref() != Ok("0");
        let tile8_tokens_ok = self
            .tile8_min_tokens
            .map(|m| prompt_len >= m)
            .unwrap_or(true);
        let q4_0_use_tile8 = gate_dt == QDtype::Q4_0
            && (down_dt == QDtype::Q4_0 || down_dt == QDtype::Q8_0 || down_dt == QDtype::Q4_1)
            && n_pairs >= Self::TILE8_PAIRS_MIN
            && tile8_tokens_ok
            && allow_tile8_decode;
        let q8_0_use_tile8 = gate_dt == QDtype::Q8_0
            && down_dt == QDtype::Q8_0
            && n_pairs >= Self::TILE8_PAIRS_MIN
            && tile8_tokens_ok
            && allow_tile8_decode;
        if (gate_dt == QDtype::Q4_0 && !q4_0_use_tile8)
            || (gate_dt == QDtype::Q8_0 && !q8_0_use_tile8)
        {
            self.gate_up_prefill_mmvq(ops, prompt_len, scratch)?;
            ops.swiglu_f32_to_f16(
                scratch.gate_out_f32,
                scratch.up_out_f32,
                scratch.activated_f16,
                n_pairs * inter,
            )
            .context("prefill moe swiglu_f32_to_f16 (q4_0/q8_0 short path)")?;
            ops.quantize_f16_q8_1(
                scratch.activated_f16,
                scratch.activated_q8_1,
                n_pairs * inter,
            )
            .context("prefill moe quantize activated (q4_0/q8_0 short path)")?;
            self.down_prefill_mmvq(ops, prompt_len, scratch)?;
            ops.cast_f32_to_f16(scratch.down_f32, scratch.down_f16, n_pairs * hidden)
                .context("prefill cast down → f16 (q4_0/q8_0 short path)")?;
            return Ok(());
        }

        // 3. tile8 path. Sort + pad routing decisions, then 64×8-tile
        // MMQ for gate+up.
        let padded_total_ub = n_pairs + n_experts * 8;
        ops.moe_sort_by_expert_padded(
            flambeau_ops::MoeSortPaddedBuffers {
                expert_ids: scratch.expert_ids,
                counts: scratch.sort_counts,
                offsets: scratch.sort_offsets,
                cursors: scratch.sort_cursors,
                sorted_pair_idx: scratch.sort_sorted_pair_idx,
                padded_offsets: scratch.sort_padded_offsets,
                sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
            },
            flambeau_ops::MoeSortPaddedShape {
                total: n_pairs,
                n_experts,
                max_tokens: scratch.max_tokens,
                top_k,
            },
        )
        .context("prefill moe_sort_by_expert_padded")?;

        match gate_dt {
            QDtype::Q4_0 => ops
                .indexed_moe_mmq_q4_0_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_32,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up q4_0 tile8")?,
            QDtype::Q8_0 => ops
                .indexed_moe_mmq_q8_0_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_32,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up q8_0 tile8")?,
            QDtype::IQ4_NL => ops
                .indexed_moe_mmq_iq4_nl_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_32,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq4_nl tile8")?,
            QDtype::IQ4_XS => ops
                .indexed_moe_mmq_iq4_xs_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq4_xs tile8")?,
            QDtype::IQ3_XXS => ops
                .indexed_moe_mmq_iq3_xxs_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq3_xxs tile8")?,
            QDtype::IQ3_S => ops
                .indexed_moe_mmq_iq3_s_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq3_s tile8")?,
            QDtype::IQ2_XXS => ops
                .indexed_moe_mmq_iq2_xxs_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq2_xxs tile8")?,
            QDtype::IQ2_XS => ops
                .indexed_moe_mmq_iq2_xs_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq2_xs tile8")?,
            QDtype::IQ2_S => ops
                .indexed_moe_mmq_iq2_s_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq2_s tile8")?,
            QDtype::IQ1_S => ops
                .indexed_moe_mmq_iq1_s_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq1_s tile8")?,
            QDtype::IQ1_M => ops
                .indexed_moe_mmq_iq1_m_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up iq1_m tile8")?,
            QDtype::Q4_K => ops
                .indexed_moe_mmq_q4_k_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up q4_k tile8")?,
            QDtype::Q3_K => ops
                .indexed_moe_mmq_q3_k_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up q3_k tile8")?,
            QDtype::Q5_K => ops
                .indexed_moe_mmq_q5_k_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up q5_k tile8")?,
            QDtype::Q6_K => ops
                .indexed_moe_mmq_q6_k_gate_up_tile8(
                    flambeau_ops::MoeMmqTile8GateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k,
                        n_sb_per_row: n_sb_per_row_hidden_kk,
                        n_experts,
                        padded_total_upper_bound: padded_total_ub,
                    },
                )
                .context("prefill indexed_moe gate+up q6_k tile8")?,
            other => bail!(
                "MoeExperts prefill: gate_dt {other:?} not on the tile8 surface (Q4_K / Q3_K / Q5_K / Q6_K / Q4_0 / Q8_0 / IQ family)"
            ),
        }
        let _ = up_dt;

        // 4. Fused SwiGLU → F16 + Q8_1 quantise.
        ops.swiglu_f32_to_f16(
            scratch.gate_out_f32,
            scratch.up_out_f32,
            scratch.activated_f16,
            n_pairs * inter,
        )
        .context("prefill moe swiglu_f32_to_f16")?;
        ops.quantize_f16_q8_1(
            scratch.activated_f16,
            scratch.activated_q8_1,
            n_pairs * inter,
        )
        .context("prefill quantise activated → Q8_1")?;

        // 5. Down dispatch — per-dtype tile8 / MMVQ choice.
        let down_shape_tile8_kk = flambeau_ops::MoeShape {
            n_rows: hidden,
            n_tokens: n_pairs,
            top_k: 1,
            n_sb_per_row: n_sb_per_row_inter_kk,
            n_experts,
            padded_total_upper_bound: padded_total_ub,
        };
        let down_shape_tile8_32 = flambeau_ops::MoeShape {
            n_rows: hidden,
            n_tokens: n_pairs,
            top_k: 1,
            n_sb_per_row: n_sb_per_row_inter_32,
            n_experts,
            padded_total_upper_bound: padded_total_ub,
        };
        match down_dt {
            QDtype::Q4_K => ops
                .indexed_moe_mmq_q4_k_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down q4_k tile8")?,
            QDtype::Q3_K => ops
                .indexed_moe_mmq_q3_k_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down q3_k tile8")?,
            QDtype::Q5_K => ops
                .indexed_moe_mmq_q5_k_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down q5_k tile8")?,
            QDtype::Q6_K => ops
                .indexed_moe_mmq_q6_k_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down q6_k tile8")?,
            QDtype::Q4_0 => ops
                .indexed_moe_mmq_q4_0_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_32,
                )
                .context("prefill indexed_moe down q4_0 tile8")?,
            QDtype::Q8_0 => ops
                .indexed_moe_mmq_q8_0_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_32,
                )
                .context("prefill indexed_moe down q8_0 tile8")?,
            QDtype::Q4_1 => ops
                .indexed_moe_mmq_q4_1_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_32,
                )
                .context("prefill indexed_moe down q4_1 tile8")?,
            QDtype::IQ4_NL => ops
                .indexed_moe_mmq_iq4_nl_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_32,
                )
                .context("prefill indexed_moe down iq4_nl tile8")?,
            QDtype::IQ4_XS => ops
                .indexed_moe_mmq_iq4_xs_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down iq4_xs tile8")?,
            QDtype::IQ3_XXS => ops
                .indexed_moe_mmq_iq3_xxs_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down iq3_xxs tile8")?,
            QDtype::IQ3_S => ops
                .indexed_moe_mmq_iq3_s_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down iq3_s tile8")?,
            QDtype::IQ2_XXS => ops
                .indexed_moe_mmq_iq2_xxs_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down iq2_xxs tile8")?,
            QDtype::IQ2_XS => ops
                .indexed_moe_mmq_iq2_xs_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down iq2_xs tile8")?,
            QDtype::IQ2_S => ops
                .indexed_moe_mmq_iq2_s_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down iq2_s tile8")?,
            QDtype::IQ1_S => ops
                .indexed_moe_mmq_iq1_s_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down iq1_s tile8")?,
            QDtype::IQ1_M => ops
                .indexed_moe_mmq_iq1_m_down_tile8(
                    flambeau_ops::MoeMmqTile8DownBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        sorted_pair_idx_padded: scratch.sort_sorted_pair_idx_padded,
                        padded_offsets: scratch.sort_padded_offsets,
                        dst: scratch.down_f32,
                    },
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down iq1_m tile8")?,
            other => bail!("MoeExperts prefill: down_dt {other:?} not on the tile8 surface"),
        }

        // 6. Cast expert outputs to F16.
        ops.cast_f32_to_f16(scratch.down_f32, scratch.down_f16, n_pairs * hidden)
            .context("prefill cast down → f16")?;

        Ok(())
    }

    fn gate_up_prefill_mmvq<O: Ops>(
        &self,
        ops: &O,
        prompt_len: usize,
        scratch: MoeExpertsPrefillScratch,
    ) -> Result<()> {
        let inter = self.intermediate;
        let hidden = self.hidden;
        let top_k = self.top_k;
        match self.ffn_gate_exps.dtype {
            QDtype::Q4_K => {
                let nb = hidden / QK_K;
                ops.indexed_moe_mmvq_q4_k_gate_up(
                    flambeau_ops::MoeMmvqGateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe gate+up q4_k mmvq")
            }
            QDtype::Q8_0 => {
                let nb = hidden / 32;
                ops.indexed_moe_mmvq_q8_0(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_gate_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.gate_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe gate q8_0 mmvq")?;
                ops.indexed_moe_mmvq_q8_0(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe up q8_0 mmvq")
            }
            QDtype::Q4_0 => {
                let nb = hidden / 32;
                ops.indexed_moe_mmvq_q4_0_gate_up(
                    flambeau_ops::MoeMmvqGateUpBuffers {
                        gate_w: self.ffn_gate_exps.ptr,
                        up_w: self.ffn_up_exps.ptr,
                        act: scratch.x_q8_1,
                        expert_ids: scratch.expert_ids,
                        gate_out: scratch.gate_out_f32,
                        up_out: scratch.up_out_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: inter,
                        n_tokens: prompt_len,
                        top_k: top_k,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe gate+up q4_0 mmvq")
            }
            other => bail!("MoeExperts prefill MMVQ short path: gate_dt {other:?} not supported"),
        }
    }

    fn down_prefill_mmvq<O: Ops>(
        &self,
        ops: &O,
        prompt_len: usize,
        scratch: MoeExpertsPrefillScratch,
    ) -> Result<()> {
        let inter = self.intermediate;
        let hidden = self.hidden;
        let top_k = self.top_k;
        let n_pairs = prompt_len * top_k;
        match self.ffn_down_exps.dtype {
            QDtype::Q4_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q4_k_r2(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_pairs,
                        top_k: 1,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe down q4_k r2")
            }
            QDtype::Q5_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q5_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_pairs,
                        top_k: 1,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe down q5_k mmvq")
            }
            QDtype::Q6_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q6_k(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_pairs,
                        top_k: 1,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe down q6_k mmvq")
            }
            QDtype::Q8_0 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q8_0(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_pairs,
                        top_k: 1,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe down q8_0 mmvq")
            }
            QDtype::Q4_0 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q4_0(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_pairs,
                        top_k: 1,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe down q4_0 mmvq")
            }
            QDtype::Q4_1 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q4_1(
                    flambeau_ops::MoeMmvqBuffers {
                        weights: self.ffn_down_exps.ptr,
                        act: scratch.activated_q8_1,
                        expert_ids: scratch.expert_ids,
                        dst: scratch.down_f32,
                    },
                    flambeau_ops::MoeMmvqShape {
                        n_rows: hidden,
                        n_tokens: n_pairs,
                        top_k: 1,
                        n_sb_per_row: nb,
                    },
                )
                .context("prefill indexed_moe down q4_1 mmvq")
            }
            other => bail!("MoeExperts prefill MMVQ short path: down_dt {other:?} not supported"),
        }
    }

    fn combine_prefill<O: Ops>(
        &self,
        ops: &O,
        scratch: MoeExpertsPrefillScratch,
        residual: DevicePtr,
        extra_residual: Option<DevicePtr>,
        out: DevicePtr,
        prompt_len: usize,
    ) -> Result<()> {
        if let Some(extra) = extra_residual {
            ops.moe_combine_two_residuals_f16(
                flambeau_ops::MoeCombineTwoResidualsBuffers {
                    expert_outs: scratch.down_f16,
                    weights: scratch.expert_weights,
                    residual1: residual,
                    residual2: extra,
                    out,
                },
                flambeau_ops::MoeCombineShape {
                    n_tokens: prompt_len,
                    top_k: self.top_k,
                    hidden: self.hidden,
                },
            )
            .context("prefill moe_combine_two_residuals_f16")
        } else {
            ops.moe_combine_f16(
                flambeau_ops::MoeCombineBuffers {
                    expert_outs: scratch.down_f16,
                    weights: scratch.expert_weights,
                    residual,
                    out,
                },
                flambeau_ops::MoeCombineShape {
                    n_tokens: prompt_len,
                    top_k: self.top_k,
                    hidden: self.hidden,
                },
            )
            .context("prefill moe_combine_f16")
        }
    }
}
