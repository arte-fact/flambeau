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
use flambeau_core::device::DevicePtr;
use flambeau_core::op::QDtype;
use flambeau_ops::Ops;

use crate::WeightHandle;

/// Borrowed-by-value view of a caller-owned MoE decode scratch.
#[derive(Copy, Clone)]
pub struct MoeExpertsDecodeScratch {
    pub x_q8_1: DevicePtr,
    pub router_logits: DevicePtr,   // F32 [n_experts]
    pub expert_ids: DevicePtr,      // i32 [top_k]
    pub expert_weights: DevicePtr,  // F32 [top_k]
    pub gate_out_f32: DevicePtr,    // F32 [top_k, intermediate]
    pub up_out_f32: DevicePtr,      // F32 [top_k, intermediate]
    pub activated_f16: DevicePtr,   // F16 [top_k, intermediate]
    pub activated_q8_1: DevicePtr,  // Q8_1 [top_k, intermediate / 32]
    pub down_f32: DevicePtr,        // F32 [top_k, hidden]
    pub down_f16: DevicePtr,        // F16 [top_k, hidden]
}

/// Borrowed-by-value view of a caller-owned MoE prefill scratch.
/// All buffers are sized for `max_tokens` (the chunk's upper bound).
#[derive(Copy, Clone)]
pub struct MoeExpertsPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,                       // Q8_1 [L, hidden / 32]
    pub router_logits: DevicePtr,                // F32 [L, n_experts]
    pub expert_ids: DevicePtr,                   // i32 [L, top_k]
    pub expert_weights: DevicePtr,               // F32 [L, top_k]
    pub gate_out_f32: DevicePtr,                 // F32 [L, top_k, intermediate]
    pub up_out_f32: DevicePtr,                   // F32 [L, top_k, intermediate]
    pub activated_f16: DevicePtr,                // F16 [L, top_k, intermediate]
    pub activated_q8_1: DevicePtr,               // Q8_1 [L, top_k, intermediate / 32]
    pub down_f32: DevicePtr,                     // F32 [L, top_k, hidden]
    pub down_f16: DevicePtr,                     // F16 [L, top_k, hidden]
    pub sort_counts: DevicePtr,                  // i32 [n_experts]
    pub sort_offsets: DevicePtr,                 // i32 [n_experts + 1]
    pub sort_cursors: DevicePtr,                 // i32 [n_experts]
    pub sort_sorted_pair_idx: DevicePtr,         // i32 [L * top_k]
    pub sort_padded_offsets: DevicePtr,          // i32 [n_experts + 1]
    pub sort_sorted_pair_idx_padded: DevicePtr,  // i32 [L * top_k + n_experts * 8]
}

const QK_K: usize = 256;

/// Qwen3-MoE routed-experts block (decode-only V1).
///
/// Holds: router weight (`ffn_gate_inp`), per-expert gate/up/down
/// indexed weights, and the shape scalars. The block does NOT own the
/// shared-expert path; that stays in the model crate.
pub struct MoeExperts {
    pub ffn_gate_inp: WeightHandle,    // [n_experts, hidden]
    pub ffn_gate_exps: WeightHandle,   // [n_experts, intermediate, hidden]
    pub ffn_up_exps: WeightHandle,     // dito
    pub ffn_down_exps: WeightHandle,   // [n_experts, hidden, intermediate]
    pub hidden: usize,
    pub intermediate: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

impl MoeExperts {
    pub fn new(
        ffn_gate_inp: WeightHandle,
        ffn_gate_exps: WeightHandle,
        ffn_up_exps: WeightHandle,
        ffn_down_exps: WeightHandle,
        hidden: usize,
        intermediate: usize,
        n_experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        if ffn_gate_inp.dims != [n_experts, hidden] {
            bail!(
                "ffn_gate_inp dims {:?} != [{}, {}]",
                ffn_gate_inp.dims,
                n_experts,
                hidden
            );
        }
        // Indexed expert weights flatten the outer-most `n_experts`
        // dim into the row count: gate/up = n_experts × intermediate
        // × hidden, but stored as a 2-D `[n_experts * intermediate,
        // hidden]` slab. Skip the strict dim assert here — loaders
        // already normalise this.
        Ok(Self {
            ffn_gate_inp,
            ffn_gate_exps,
            ffn_up_exps,
            ffn_down_exps,
            hidden,
            intermediate,
            n_experts,
            top_k,
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
        ops.topk_f32(
            scratch.router_logits,
            scratch.expert_ids,
            scratch.expert_weights,
            1,
            self.n_experts,
            self.top_k,
        )
        .context("router topk_f32")?;
        Ok(())
    }

    fn gate_up<O: Ops>(
        &self,
        ops: &O,
        scratch: MoeExpertsDecodeScratch,
    ) -> Result<()> {
        let inter = self.intermediate;
        let n_tokens = 1usize;
        let top_k = self.top_k;
        let hidden = self.hidden;
        let dtype = self.ffn_gate_exps.dtype;
        match dtype {
            QDtype::Q4_K => {
                let nb = hidden / QK_K;
                ops.indexed_moe_mmvq_q4_k_gate_up(
                    self.ffn_gate_exps.ptr,
                    self.ffn_up_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.gate_out_f32,
                    scratch.up_out_f32,
                    inter,
                    n_tokens,
                    top_k,
                    nb,
                )
                .context("indexed_moe gate+up q4_k")
            }
            QDtype::Q8_0 => {
                let nb = hidden / 32;
                ops.indexed_moe_mmvq_q8_0(
                    self.ffn_gate_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.gate_out_f32,
                    inter,
                    n_tokens,
                    top_k,
                    nb,
                )
                .context("indexed_moe gate q8_0")?;
                ops.indexed_moe_mmvq_q8_0(
                    self.ffn_up_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.up_out_f32,
                    inter,
                    n_tokens,
                    top_k,
                    nb,
                )
                .context("indexed_moe up q8_0")
            }
            QDtype::Q4_0 => {
                let nb = hidden / 32;
                ops.indexed_moe_mmvq_q4_0_gate_up(
                    self.ffn_gate_exps.ptr,
                    self.ffn_up_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.gate_out_f32,
                    scratch.up_out_f32,
                    inter,
                    n_tokens,
                    top_k,
                    nb,
                )
                .context("indexed_moe gate+up q4_0 fused")
            }
            other => bail!(
                "MoeExperts gate dtype {other:?} not supported (expected Q4_K / Q8_0 / Q4_0)"
            ),
        }
    }

    fn down<O: Ops>(
        &self,
        ops: &O,
        scratch: MoeExpertsDecodeScratch,
    ) -> Result<()> {
        let inter = self.intermediate;
        let hidden = self.hidden;
        let n_tokens_eff = self.top_k;
        let top_k_inner = 1;
        let dtype = self.ffn_down_exps.dtype;
        match dtype {
            QDtype::Q4_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q4_k_r2(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_tokens_eff,
                    top_k_inner,
                    nb,
                )
                .context("indexed_moe down q4_k r2")
            }
            QDtype::Q5_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q5_k(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_tokens_eff,
                    top_k_inner,
                    nb,
                )
                .context("indexed_moe down q5_k")
            }
            QDtype::Q6_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q6_k(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_tokens_eff,
                    top_k_inner,
                    nb,
                )
                .context("indexed_moe down q6_k")
            }
            QDtype::Q8_0 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q8_0(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_tokens_eff,
                    top_k_inner,
                    nb,
                )
                .context("indexed_moe down q8_0")
            }
            QDtype::Q4_0 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q4_0(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_tokens_eff,
                    top_k_inner,
                    nb,
                )
                .context("indexed_moe down q4_0")
            }
            QDtype::Q4_1 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q4_1(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_tokens_eff,
                    top_k_inner,
                    nb,
                )
                .context("indexed_moe down q4_1")
            }
            other => bail!(
                "MoeExperts down dtype {other:?} not supported (expected Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0 / Q4_1)"
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

        // 3+4. Fused SwiGLU → F16 + Q8_1 quantise.
        ops.swiglu_f32_to_f16(
            scratch.gate_out_f32,
            scratch.up_out_f32,
            scratch.activated_f16,
            top_k * inter,
        )
        .context("moe swiglu_f32_to_f16")?;
        ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, top_k * inter)
            .context("moe quantize activated → Q8_1")?;

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
                scratch.down_f16,
                scratch.expert_weights,
                residual,
                extra,
                out,
                1,
                top_k,
                hidden,
            )
            .context("moe_combine_two_residuals_f16")?;
        } else {
            ops.moe_combine_f16(
                scratch.down_f16,
                scratch.expert_weights,
                residual,
                out,
                1,
                top_k,
                hidden,
            )
            .context("moe_combine_f16")?;
        }
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
        ops.topk_f32(
            scratch.router_logits,
            scratch.expert_ids,
            scratch.expert_weights,
            prompt_len,
            self.n_experts,
            self.top_k,
        )
        .context("router prefill topk_f32")?;
        Ok(())
    }

    /// Threshold above which the tile8 path beats the plain MMVQ path
    /// for Q4_0 / Q8_0 prefill. Below this, tile8's grid overhead
    /// dominates and the per-token MMVQ wins.
    const TILE8_THRESHOLD: usize = 32;

    /// Multi-token routed-experts prefill. Caller has already
    /// populated `expert_ids` / `expert_weights` (via
    /// `route_prefill`). Writes
    /// `out = residual + extra_residual? + Σ w_k · expert_k(x_norm)`
    /// at every prompt position.
    pub fn forward_prefill<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        residual: DevicePtr,
        extra_residual: Option<DevicePtr>,
        out: DevicePtr,
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
        let q4_0_use_tile8 = gate_dt == QDtype::Q4_0
            && (down_dt == QDtype::Q4_0 || down_dt == QDtype::Q8_0 || down_dt == QDtype::Q4_1)
            && prompt_len >= Self::TILE8_THRESHOLD;
        let q8_0_use_tile8 = gate_dt == QDtype::Q8_0
            && down_dt == QDtype::Q8_0
            && prompt_len >= Self::TILE8_THRESHOLD;
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
            return self.combine_prefill(ops, scratch, residual, extra_residual, out, prompt_len);
        }

        // 3. tile8 path. Sort + pad routing decisions, then 64×8-tile
        // MMQ for gate+up.
        let padded_total_ub = n_pairs + n_experts * 8;
        ops.moe_sort_by_expert_padded(
            scratch.expert_ids,
            scratch.sort_counts,
            scratch.sort_offsets,
            scratch.sort_cursors,
            scratch.sort_sorted_pair_idx,
            scratch.sort_padded_offsets,
            scratch.sort_sorted_pair_idx_padded,
            n_pairs,
            n_experts,
            scratch.max_tokens,
            top_k,
        )
        .context("prefill moe_sort_by_expert_padded")?;

        match gate_dt {
            QDtype::Q4_0 => ops
                .indexed_moe_mmq_q4_0_gate_up_tile8(
                    self.ffn_gate_exps.ptr,
                    self.ffn_up_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.sort_sorted_pair_idx_padded,
                    scratch.sort_padded_offsets,
                    scratch.gate_out_f32,
                    scratch.up_out_f32,
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
                    self.ffn_gate_exps.ptr,
                    self.ffn_up_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.sort_sorted_pair_idx_padded,
                    scratch.sort_padded_offsets,
                    scratch.gate_out_f32,
                    scratch.up_out_f32,
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
            QDtype::Q4_K => {
                let shape = flambeau_ops::MoeShape {
                    n_rows: inter,
                    n_tokens: prompt_len,
                    top_k,
                    n_sb_per_row: n_sb_per_row_hidden_kk,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                };
                let split = std::env::var("FLAMBEAU_MOE_Q4K_SPLIT_GATEUP")
                    .ok()
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false);
                if split {
                    ops.indexed_moe_mmq_q4_k_gate_only_tile8(
                        self.ffn_gate_exps.ptr,
                        scratch.x_q8_1,
                        scratch.expert_ids,
                        scratch.sort_sorted_pair_idx_padded,
                        scratch.sort_padded_offsets,
                        scratch.gate_out_f32,
                        shape,
                    )
                    .context("prefill indexed_moe gate-only q4_k tile8")?;
                    ops.indexed_moe_mmq_q4_k_up_only_tile8(
                        self.ffn_up_exps.ptr,
                        scratch.x_q8_1,
                        scratch.expert_ids,
                        scratch.sort_sorted_pair_idx_padded,
                        scratch.sort_padded_offsets,
                        scratch.up_out_f32,
                        shape,
                    )
                    .context("prefill indexed_moe up-only q4_k tile8")?;
                } else {
                    ops.indexed_moe_mmq_q4_k_gate_up_tile8(
                        self.ffn_gate_exps.ptr,
                        self.ffn_up_exps.ptr,
                        scratch.x_q8_1,
                        scratch.expert_ids,
                        scratch.sort_sorted_pair_idx_padded,
                        scratch.sort_padded_offsets,
                        scratch.gate_out_f32,
                        scratch.up_out_f32,
                        shape,
                    )
                    .context("prefill indexed_moe gate+up q4_k tile8")?;
                }
            }
            other => bail!(
                "MoeExperts prefill: gate_dt {other:?} not on the tile8 surface (Q4_K / Q4_0 / Q8_0)"
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
        ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, n_pairs * inter)
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
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.sort_sorted_pair_idx_padded,
                    scratch.sort_padded_offsets,
                    scratch.down_f32,
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down q4_k tile8")?,
            QDtype::Q5_K => ops
                .indexed_moe_mmq_q5_k_down_tile8(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.sort_sorted_pair_idx_padded,
                    scratch.sort_padded_offsets,
                    scratch.down_f32,
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down q5_k tile8")?,
            QDtype::Q6_K => ops
                .indexed_moe_mmq_q6_k_down_tile8(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.sort_sorted_pair_idx_padded,
                    scratch.sort_padded_offsets,
                    scratch.down_f32,
                    down_shape_tile8_kk,
                )
                .context("prefill indexed_moe down q6_k tile8")?,
            QDtype::Q4_0 => ops
                .indexed_moe_mmq_q4_0_down_tile8(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.sort_sorted_pair_idx_padded,
                    scratch.sort_padded_offsets,
                    scratch.down_f32,
                    down_shape_tile8_32,
                )
                .context("prefill indexed_moe down q4_0 tile8")?,
            QDtype::Q8_0 => ops
                .indexed_moe_mmq_q8_0_down_tile8(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.sort_sorted_pair_idx_padded,
                    scratch.sort_padded_offsets,
                    scratch.down_f32,
                    down_shape_tile8_32,
                )
                .context("prefill indexed_moe down q8_0 tile8")?,
            QDtype::Q4_1 => ops
                .indexed_moe_mmq_q4_1_down_tile8(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.sort_sorted_pair_idx_padded,
                    scratch.sort_padded_offsets,
                    scratch.down_f32,
                    down_shape_tile8_32,
                )
                .context("prefill indexed_moe down q4_1 tile8")?,
            other => bail!(
                "MoeExperts prefill: down_dt {other:?} not on the tile8 surface"
            ),
        }

        // 6. Cast expert outputs to F16.
        ops.cast_f32_to_f16(scratch.down_f32, scratch.down_f16, n_pairs * hidden)
            .context("prefill cast down → f16")?;

        // 7. Weighted sum + residual (+ optional shared-expert delta).
        self.combine_prefill(ops, scratch, residual, extra_residual, out, prompt_len)
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
                    self.ffn_gate_exps.ptr,
                    self.ffn_up_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.gate_out_f32,
                    scratch.up_out_f32,
                    inter,
                    prompt_len,
                    top_k,
                    nb,
                )
                .context("prefill indexed_moe gate+up q4_k mmvq")
            }
            QDtype::Q8_0 => {
                let nb = hidden / 32;
                ops.indexed_moe_mmvq_q8_0(
                    self.ffn_gate_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.gate_out_f32,
                    inter,
                    prompt_len,
                    top_k,
                    nb,
                )
                .context("prefill indexed_moe gate q8_0 mmvq")?;
                ops.indexed_moe_mmvq_q8_0(
                    self.ffn_up_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.up_out_f32,
                    inter,
                    prompt_len,
                    top_k,
                    nb,
                )
                .context("prefill indexed_moe up q8_0 mmvq")
            }
            QDtype::Q4_0 => {
                let nb = hidden / 32;
                ops.indexed_moe_mmvq_q4_0_gate_up(
                    self.ffn_gate_exps.ptr,
                    self.ffn_up_exps.ptr,
                    scratch.x_q8_1,
                    scratch.expert_ids,
                    scratch.gate_out_f32,
                    scratch.up_out_f32,
                    inter,
                    prompt_len,
                    top_k,
                    nb,
                )
                .context("prefill indexed_moe gate+up q4_0 mmvq")
            }
            other => bail!(
                "MoeExperts prefill MMVQ short path: gate_dt {other:?} not supported"
            ),
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
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_pairs,
                    1,
                    nb,
                )
                .context("prefill indexed_moe down q4_k r2")
            }
            QDtype::Q5_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q5_k(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_pairs,
                    1,
                    nb,
                )
                .context("prefill indexed_moe down q5_k mmvq")
            }
            QDtype::Q6_K => {
                let nb = inter / QK_K;
                ops.indexed_moe_mmvq_q6_k(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_pairs,
                    1,
                    nb,
                )
                .context("prefill indexed_moe down q6_k mmvq")
            }
            QDtype::Q8_0 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q8_0(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_pairs,
                    1,
                    nb,
                )
                .context("prefill indexed_moe down q8_0 mmvq")
            }
            QDtype::Q4_0 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q4_0(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_pairs,
                    1,
                    nb,
                )
                .context("prefill indexed_moe down q4_0 mmvq")
            }
            QDtype::Q4_1 => {
                let nb = inter / 32;
                ops.indexed_moe_mmvq_q4_1(
                    self.ffn_down_exps.ptr,
                    scratch.activated_q8_1,
                    scratch.expert_ids,
                    scratch.down_f32,
                    hidden,
                    n_pairs,
                    1,
                    nb,
                )
                .context("prefill indexed_moe down q4_1 mmvq")
            }
            other => bail!(
                "MoeExperts prefill MMVQ short path: down_dt {other:?} not supported"
            ),
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
                scratch.down_f16,
                scratch.expert_weights,
                residual,
                extra,
                out,
                prompt_len,
                self.top_k,
                self.hidden,
            )
            .context("prefill moe_combine_two_residuals_f16")
        } else {
            ops.moe_combine_f16(
                scratch.down_f16,
                scratch.expert_weights,
                residual,
                out,
                prompt_len,
                self.top_k,
                self.hidden,
            )
            .context("prefill moe_combine_f16")
        }
    }
}
