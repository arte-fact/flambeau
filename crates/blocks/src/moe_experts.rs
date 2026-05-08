//! `MoeExperts` — Qwen3-MoE routed-experts block (decode).
//!
//! Pipeline (decode):
//!
//! 1. (router, optional) `dense_gemv` over `ffn_gate_inp` + `topk_f32` →
//!    `expert_ids[top_k]`, `expert_weights[top_k]`.
//! 2. `quantize_f16_q8_1(x_norm)`.
//! 3. indexed gate+up per dtype (Q4_K / Q4_0 fused / Q8_0 unfused).
//! 4. `swiglu_f32_to_f16(gate, up)` + `quantize_f16_q8_1(activated)`.
//! 5. indexed down per dtype (treat each top_k slot as its own
//!    effective token with `top_k=1`).
//! 6. `cast_f32_to_f16(down)`.
//! 7. `moe_combine_f16(...)` (or `moe_combine_two_residuals_f16` if a
//!    shared-expert delta is provided as `extra_residual`).
//!
//! Prefill stays on the existing `qwen3-moe::forward::moe`
//! free-fn — its multi-path tile8 / turbo MMQ dispatch needs a
//! separate design pass and isn't included in this block. Shared
//! expert + dense-FFN-with-sigmoid likewise stay in the model crate
//! for V1.

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
}
