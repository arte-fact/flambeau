//! MoE forward: routed experts + shared expert + router, decode + prefill.
//!
//! All MoE forward paths live here. Dense layers bypass this module and
//! route through `forward::dense_ffn` instead. The split between routed /
//! shared / router reflects the Qwen3 MoE recipe:
//! - router: dense GEMV producing per-expert logits → topk.
//! - routed experts: indexed MMVQ/MMQ selecting the top-k experts per token.
//! - shared expert: a dense FFN added to every token's output, gated by a
//!   learned sigmoid.

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
    cast::{cast_f16_to_f32, cast_f32_to_f16},
    mlp::swiglu_f32,
    moe::{
        indexed_moe_mmq_q4_k_down_tile8, indexed_moe_mmq_q4_k_down_turbo,
        indexed_moe_mmq_q4_k_gate_up_tile8, indexed_moe_mmq_q4_k_gate_up_turbo,
        indexed_moe_mmq_q6_k_down_tile8,
        indexed_moe_mmvq_q4_k_gate_up, indexed_moe_mmvq_q4_k_gate_up_sorted,
        indexed_moe_mmvq_q4_k_r2, indexed_moe_mmvq_q4_k_r2_sorted, indexed_moe_mmvq_q6_k,
        moe_combine_f16, moe_sort_by_expert,
        moe_sort_by_expert_padded, shared_expert_scale_f32, topk_f32,
    },
    norm::{quantize_f16_q8_1, quantize_f16_q8_1_mmq},
    qmatmul::mmvq_q8_0_gate_up,
    router::dense_gemv_f32_f16,
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::{BlockQ8_1, GgmlDType};

use super::common::{
    cast_and_quantize_f32_to_q8_1, mat_shape, run_indexed_moe_down,
    run_indexed_moe_gate_up, run_mmvq_from_tensor, run_qmatmul_from_tensor, validate_moe_dtypes,
    QK_K,
};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

// ---------------------------------------------------------------------------
// V1.7.3-d1 — routed MoE FFN decode step.
// ---------------------------------------------------------------------------

/// Workspace for one decode step of the routed MoE FFN (no shared expert —
/// that lands in V1.7.3-d2, no router — V1.7.3-d3). Sized against
/// `(hidden, moe_intermediate_size, num_experts_per_tok=top_k)`.
pub struct MoeScratch {
    // Q8_1 of layer input, shared across all top_k experts' gate/up matmuls.
    pub x_q8_1: DevicePtr,
    // Router logits (F32 [n_experts]) → populated by `forward_router_decode`.
    pub router_logits: DevicePtr,
    // Expert ids / weights — populated by the router (or the caller).
    pub expert_ids: DevicePtr,        // i32 [top_k]
    pub expert_weights: DevicePtr,    // F32 [top_k]
    // Fused gate+up MMVQ outputs: F32 [top_k, moe_inter] each.
    pub gate_out_f32: DevicePtr,
    pub up_out_f32: DevicePtr,
    // swiglu(gate, up) result: F32 [top_k, moe_inter], then cast to F16,
    // then quantised to Q8_1 (flat [top_k, moe_inter/32]) for the down step.
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    // Down MMVQ output: F32 [top_k, hidden], then cast to F16 for combine.
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
    // Bookkeeping.
    x_q8_1_bytes: usize,
    router_logits_bytes: usize,
    expert_ids_bytes: usize,
    expert_weights_bytes: usize,
    gate_up_bytes: usize,
    activated_f32_bytes: usize,
    activated_f16_bytes: usize,
    activated_q8_1_bytes: usize,
    down_f32_bytes: usize,
    down_f16_bytes: usize,
    disposed: bool,
}

impl MoeScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let top_k = cfg.num_experts_per_tok;
        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(inter % 32 == 0, "moe_intermediate_size must be a multiple of QK8_1=32");

        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let router_logits_bytes = cfg.num_experts * 4;
        let expert_ids_bytes = top_k * 4;
        let expert_weights_bytes = top_k * 4;
        let gate_up_bytes = top_k * inter * 4;
        let activated_f32_bytes = top_k * inter * 4;
        let activated_f16_bytes = top_k * inter * 2;
        let activated_q8_1_bytes = top_k * (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let down_f32_bytes = top_k * hidden * 4;
        let down_f16_bytes = top_k * hidden * 2;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let router_logits = device.alloc(router_logits_bytes)?;
        let expert_ids = device.alloc(expert_ids_bytes)?;
        let expert_weights = device.alloc(expert_weights_bytes)?;
        let gate_out_f32 = device.alloc(gate_up_bytes)?;
        let up_out_f32 = device.alloc(gate_up_bytes)?;
        let activated_f32 = device.alloc(activated_f32_bytes)?;
        let activated_f16 = device.alloc(activated_f16_bytes)?;
        let activated_q8_1 = device.alloc(activated_q8_1_bytes)?;
        let down_f32 = device.alloc(down_f32_bytes)?;
        let down_f16 = device.alloc(down_f16_bytes)?;

        Ok(Self {
            x_q8_1,
            router_logits,
            expert_ids,
            expert_weights,
            gate_out_f32,
            up_out_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            down_f16,
            x_q8_1_bytes,
            router_logits_bytes,
            expert_ids_bytes,
            expert_weights_bytes,
            gate_up_bytes,
            activated_f32_bytes,
            activated_f16_bytes,
            activated_q8_1_bytes,
            down_f32_bytes,
            down_f16_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.router_logits, self.router_logits_bytes)?;
            device.dealloc(self.expert_ids, self.expert_ids_bytes)?;
            device.dealloc(self.expert_weights, self.expert_weights_bytes)?;
            device.dealloc(self.gate_out_f32, self.gate_up_bytes)?;
            device.dealloc(self.up_out_f32, self.gate_up_bytes)?;
            device.dealloc(self.activated_f32, self.activated_f32_bytes)?;
            device.dealloc(self.activated_f16, self.activated_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.activated_q8_1_bytes)?;
            device.dealloc(self.down_f32, self.down_f32_bytes)?;
            device.dealloc(self.down_f16, self.down_f16_bytes)?;
        }
        Ok(())
    }
}

impl Drop for MoeScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "MoeScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One decode step of the routed MoE FFN. Assumes the caller has:
/// - Run `post_attention_norm` on the residual stream (so `x_norm` is the
///   norm output).
/// - Already filled `scratch.expert_ids` and `scratch.expert_weights` with
///   the router's output. V1.7.3-d3 will land the router; until then the
///   caller is synthetic (test fixture or hand-rolled top-k).
///
/// The final `out` is computed as `residual + Σ_k weight_k · expert_out_k`,
/// matching `moe_combine_f16`'s semantics — so `out` already has the
/// residual fused in and the outer loop can skip a second residual add.
pub fn forward_moe_ffn_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn: &crate::weights::FfnWeights,
    scratch: &mut MoeScratch,
    x_norm: DevicePtr,
    residual: DevicePtr,
    extra_residual: Option<DevicePtr>,
    out: DevicePtr,
) -> Result<()> {
    let ffn_gate_exps = ffn
        .ffn_gate_exps
        .as_ref()
        .context("forward_moe_ffn_decode: ffn.ffn_gate_exps missing (loader should have rejected a non-MoE layer routed here)")?;
    let ffn_up_exps = ffn
        .ffn_up_exps
        .as_ref()
        .context("forward_moe_ffn_decode: ffn.ffn_up_exps missing")?;
    let ffn_down_exps = ffn
        .ffn_down_exps
        .as_ref()
        .context("forward_moe_ffn_decode: ffn.ffn_down_exps missing")?;
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let top_k = cfg.num_experts_per_tok;
    let _ = cfg.num_experts; // presently only asserted by dims on ffn_gate_exps

    // 1. Quantise x_norm to Q8_1 for the gate/up matmul.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
        .context("moe x_norm → Q8_1")?;

    // 2. Fused gate + up matmul across top_k selected experts in one launch.
    // Weight shape (outermost-first): `[n_experts, inter, hidden]`. The
    // indexed_moe kernels take n_sb_per_row = hidden / QK_K.
    // gate+up may both be Q4_K (standard UD-Q4_K_S) or Q8_0 (V2.22.a:
    // UD-Q8_K_XL). down may be Q4_K, Q6_K (UD-Q4_K_S ffn_down promotion),
    // or Q8_0 (UD-Q8_K_XL). BF16 layers in UD-Q8_K_XL aren't handled here
    // yet — loader converts them to Q8_0 on host.
    let gate_dt = ffn_gate_exps.dtype;
    let up_dt = ffn_up_exps.dtype;
    validate_moe_dtypes(
        "indexed-MoE",
        gate_dt,
        up_dt,
        ffn_down_exps.dtype,
        hidden,
        inter,
    )?;
    run_indexed_moe_gate_up(
        ops,
        stream,
        gate_dt,
        ffn_gate_exps.ptr,
        ffn_up_exps.ptr,
        scratch.x_q8_1,
        scratch.expert_ids,
        scratch.gate_out_f32,
        scratch.up_out_f32,
        inter,
        1,
        top_k,
        hidden,
    )?;

    // 3. SwiGLU(gate, up) → activated, F32.
    swiglu_f32(
        ops,
        stream,
        scratch.gate_out_f32,
        scratch.up_out_f32,
        scratch.activated_f32,
        top_k * inter,
    )
    .context("moe swiglu_f32")?;

    // 4. Cast activated F32 → F16, then quantise F16 → Q8_1. Layout is
    // `[top_k, inter]` flat — each row of `activated_q8_1` is one expert's
    // input to the down matmul (re-used as one "effective token" below).
    cast_and_quantize_f32_to_q8_1(
        ops,
        stream,
        scratch.activated_f32,
        scratch.activated_f16,
        scratch.activated_q8_1,
        top_k * inter,
        "moe decode activated",
    )?;

    // 5. Down matmul. Indexed MoE MMVQ dispatches by
    // `expert_ids[token * top_k + slot]`. Treat each of our top_k routed
    // experts as its own "effective token" with `top_k = 1` and its own
    // expert id. The scratch already holds `expert_ids[0..top_k]` which
    // doubles as the flat expert lookup (`flat[i] = expert_ids[0 * 1 + i]`).
    run_indexed_moe_down(
        ops,
        stream,
        ffn_down_exps.dtype,
        ffn_down_exps.ptr,
        scratch.activated_q8_1,
        scratch.expert_ids,
        scratch.down_f32,
        hidden,
        top_k, // n_tokens_effective
        1,     // top_k=1 in this re-indexed view
        inter,
    )?;

    // 6. Cast expert outputs to F16 for the combine kernel.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.down_f32,
        scratch.down_f16,
        top_k * hidden,
    )
    .context("cast down → f16")?;

    // 7. Weighted sum + residual. V2.23.a.2 — if caller provides a second
    // residual (shared-expert delta), fuse it into the combine step so
    // we skip the standalone add_f16 between shared expert and combine.
    if let Some(extra) = extra_residual {
        flambeau_ops::hip::moe::moe_combine_two_residuals_f16(
            ops,
            stream,
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
        moe_combine_f16(
            ops,
            stream,
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

// ---------------------------------------------------------------------------
// V1.7.3-d2 — shared expert decode step.
// ---------------------------------------------------------------------------

/// Workspace for one decode step of the shared expert (dense FFN +
/// per-token sigmoid gate scaling). Sized against
/// `(hidden, shared_expert_intermediate_size)`.
pub struct SharedExpertScratch {
    // Q8_1 of `x_norm`, shared across gate/up matmuls.
    pub x_q8_1: DevicePtr,
    // Dense gate/up matmul outputs, F32 [shared_inter].
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    // SwiGLU output + F16 round-trip for the down matmul input.
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    // Down matmul output (F32), scaled in place by `shared_expert_scale_f32`.
    pub down_f32: DevicePtr,
    // F32 view of `x_norm` — the gate-scale kernel dots it against
    // `ffn_gate_inp_shexp` to produce the per-token gate scalar.
    pub x_norm_f32: DevicePtr,
    // Bookkeeping.
    x_q8_1_bytes: usize,
    inter_f32_bytes: usize,
    inter_f16_bytes: usize,
    inter_q8_1_bytes: usize,
    hidden_f32_bytes: usize,
    disposed: bool,
}

impl SharedExpertScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg
            .shared_expert_intermediate_size
            .context("SharedExpertScratch requires cfg.shared_expert_intermediate_size")?;
        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(
            inter % 32 == 0,
            "shared_expert_intermediate_size must be a multiple of QK8_1=32"
        );

        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let inter_f32_bytes = inter * 4;
        let inter_f16_bytes = inter * 2;
        let inter_q8_1_bytes = (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let hidden_f32_bytes = hidden * 4;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let gate_f32 = device.alloc(inter_f32_bytes)?;
        let up_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f16 = device.alloc(inter_f16_bytes)?;
        let activated_q8_1 = device.alloc(inter_q8_1_bytes)?;
        let down_f32 = device.alloc(hidden_f32_bytes)?;
        let x_norm_f32 = device.alloc(hidden_f32_bytes)?;

        Ok(Self {
            x_q8_1,
            gate_f32,
            up_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            x_norm_f32,
            x_q8_1_bytes,
            inter_f32_bytes,
            inter_f16_bytes,
            inter_q8_1_bytes,
            hidden_f32_bytes,
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
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.gate_f32, self.inter_f32_bytes)?;
            device.dealloc(self.up_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f16, self.inter_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.inter_q8_1_bytes)?;
            device.dealloc(self.down_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.x_norm_f32, self.hidden_f32_bytes)?;
        }
        Ok(())
    }
}

impl Drop for SharedExpertScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "SharedExpertScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One decode step of the shared expert (always-on dense FFN), composed
/// with the learned per-token sigmoid-gate scaling:
///
///   gate_scalar[t] = sigmoid(⟨ ffn_gate_inp_shexp, x_norm[t] ⟩)
///   dense[t]       = down_shexp(swiglu(gate_shexp(x_norm[t]), up_shexp(x_norm[t])))
///   shared_out[t]  = gate_scalar[t] * dense[t]
///
/// Output (`shared_out`) is a standalone F16 **delta** — the caller is
/// expected to sum it with the routed-MoE output and the residual in
/// V1.7.3-e. Keeping this delta-only keeps the composition orthogonal:
/// routed and shared contributions flow through the same combine layer in
/// V1.7.3-e without re-using `residual` for a second purpose.
pub fn forward_shared_expert_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    shared: &crate::weights::SharedExpertWeights,
    scratch: &mut SharedExpertScratch,
    x_norm: DevicePtr,
    shared_out: DevicePtr,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let inter = cfg
        .shared_expert_intermediate_size
        .context("forward_shared_expert_decode requires cfg.shared_expert_intermediate_size")?;

    // 1. Quantise x_norm → Q8_1 for gate/up matmuls.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
        .context("shexp x_norm → Q8_1")?;

    // 2+3. Dense gate + up matmuls. Fuse when FLAMBEAU_VARIANT=dp4a_vdr2 so
    // the shared Q8_1 activation is read once, saving one kernel launch per
    // layer per forward. Both weights must be Q8_0 for the fused path.
    let fuse_gate_up = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && shared.ffn_gate_shexp.dtype == flambeau_quant::GgmlDType::Q8_0
        && shared.ffn_up_shexp.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_gate_up {
        let (g_rows, g_k) = mat_shape(&shared.ffn_gate_shexp)?;
        let (u_rows, u_k) = mat_shape(&shared.ffn_up_shexp)?;
        if g_rows != inter || g_k != hidden || u_rows != inter || u_k != hidden {
            bail!(
                "fused shexp gate/up shape mismatch: gate=[{g_rows},{g_k}] up=[{u_rows},{u_k}] expected=[{inter},{hidden}]"
            );
        }
        mmvq_q8_0_gate_up(
            ops,
            stream,
            shared.ffn_gate_shexp.ptr,
            shared.ffn_up_shexp.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            scratch.up_f32,
            inter,
            inter,
            hidden,
        )
        .context("shexp mmvq_q8_0_gate_up (fused)")?;
    } else {
        run_mmvq_from_tensor(
            ops,
            stream,
            &shared.ffn_gate_shexp,
            scratch.x_q8_1,
            scratch.gate_f32,
            inter,
            hidden,
            "ffn_gate_shexp",
        )?;
        run_mmvq_from_tensor(
            ops,
            stream,
            &shared.ffn_up_shexp,
            scratch.x_q8_1,
            scratch.up_f32,
            inter,
            hidden,
            "ffn_up_shexp",
        )?;
    }

    // 4. swiglu(gate, up) → activated_f32.
    swiglu_f32(
        ops,
        stream,
        scratch.gate_f32,
        scratch.up_f32,
        scratch.activated_f32,
        inter,
    )
    .context("shexp swiglu_f32")?;

    // 5. Cast + quantise activated for the down matmul input.
    // Tried F32→Q8_1 direct (skip F16 intermediate) — regressed ~1% because
    // the F32 quantize kernel is slower per-element than the F16 one (no
    // packed fp16 max-reduction). Two small kernels beat one big one here.
    cast_and_quantize_f32_to_q8_1(
        ops,
        stream,
        scratch.activated_f32,
        scratch.activated_f16,
        scratch.activated_q8_1,
        inter,
        "shexp decode activated",
    )?;

    // 6. Dense down matmul → down_f32 [hidden].
    run_mmvq_from_tensor(
        ops,
        stream,
        &shared.ffn_down_shexp,
        scratch.activated_q8_1,
        scratch.down_f32,
        hidden,
        inter,
        "ffn_down_shexp",
    )?;

    // 7. Apply the learned per-token sigmoid gate scaling in place.
    // The kernel needs F32 views of both `shared_out` (the dense FFN
    // result) and `x_norm` (the layer input the gate learns from).
    cast_f16_to_f32(ops, stream, x_norm, scratch.x_norm_f32, hidden)
        .context("shexp cast x_norm → f32")?;
    shared_expert_scale_f32(
        ops,
        stream,
        scratch.down_f32,
        scratch.x_norm_f32,
        shared.ffn_gate_inp_shexp.ptr,
        1,
        hidden,
    )
    .context("shared_expert_scale_f32")?;

    // 8. Cast the scaled output back to F16 for the outer composition.
    cast_f32_to_f16(ops, stream, scratch.down_f32, shared_out, hidden)
        .context("shexp cast → f16")?;

    Ok(())
}

// Dense-FFN decode + prefill (DenseFfnScratch, forward_dense_ffn_decode,
// DenseFfnPrefillScratch, forward_dense_ffn_prefill) moved to `forward::dense_ffn`.

// ---------------------------------------------------------------------------
// V1.7.3-d3 — MoE router.
// ---------------------------------------------------------------------------

/// Run the MoE router for one decode token. Reads `x_norm` and the FFN's
/// `ffn_gate_inp` weight; writes the top-k selected expert ids + their
/// softmaxed weights into the MoE scratch buffers that
/// `forward_moe_ffn_decode` consumes.
///
/// Two-stage path:
///   1. `dense_gemv_f32_f16(ffn_gate_inp, x_norm)` → `router_logits` F32 [n_experts]
///   2. `topk_f32(router_logits, expert_ids, expert_weights, 1, n_experts, top_k)`
///
/// The router weight must be F32 — Qwen3.x GGUFs don't quantise this
/// particular tensor (`ffn_gate_inp.weight`) since it's tiny.
pub fn forward_router_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_inp: &DeviceTensor,
    scratch: &mut MoeScratch,
    x_norm: DevicePtr,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let n_experts = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok;

    if ffn_gate_inp.dtype != GgmlDType::F32 {
        bail!(
            "router expects F32 ffn_gate_inp; got {:?}",
            ffn_gate_inp.dtype
        );
    }
    // Weight dims (outermost-first): `[n_experts, hidden]`.
    if ffn_gate_inp.dims.len() != 2 {
        bail!(
            "ffn_gate_inp: expected 2D weight, got dims {:?}",
            ffn_gate_inp.dims
        );
    }
    let w_rows = ffn_gate_inp.dims[0] as usize;
    let w_k = ffn_gate_inp.dims[1] as usize;
    if w_rows != n_experts || w_k != hidden {
        bail!(
            "ffn_gate_inp shape [{w_rows}, {w_k}] != expected [{n_experts}, {hidden}]"
        );
    }

    dense_gemv_f32_f16(
        ops,
        stream,
        ffn_gate_inp.ptr,
        x_norm,
        scratch.router_logits,
        n_experts,
        hidden,
    )
    .context("router dense_gemv_f32_f16")?;

    topk_f32(
        ops,
        stream,
        scratch.router_logits,
        scratch.expert_ids,
        scratch.expert_weights,
        1,
        n_experts,
        top_k,
    )
    .context("router topk_f32")?;

    Ok(())
}


// ---------------------------------------------------------------------------
// V1.7.3-f3 — MoE + shared expert + router prefill.
// ---------------------------------------------------------------------------

/// Workspace for one prefill chunk of the routed MoE FFN. Sized against
/// `(cfg, max_tokens)`.
pub struct MoePrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    pub router_logits: DevicePtr,      // F32 [L, n_experts]
    pub expert_ids: DevicePtr,         // i32 [L, top_k]
    pub expert_weights: DevicePtr,     // F32 [L, top_k]
    pub gate_out_f32: DevicePtr,       // F32 [L, top_k, inter]
    pub up_out_f32: DevicePtr,         // F32 [L, top_k, inter]
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    // V2.14.c DS4 Q8_1 activation buffers (turbo MoE variant only).
    // `x_q8_1_mmq`: hidden activation in DS4 layout — [hidden/128, n_tokens].
    // `activated_q8_1_mmq`: per-pair SwiGLU'd activation in DS4 layout — [inter/128, n_pairs].
    pub x_q8_1_mmq: DevicePtr,
    pub activated_q8_1_mmq: DevicePtr,
    pub down_f32: DevicePtr,           // F32 [L, top_k, hidden]
    pub down_f16: DevicePtr,
    // V2.5.a sort-by-expert state. Only populated / used when
    // FLAMBEAU_MOE_SORTED=1 is set on the gate+up path.
    pub sort_counts: DevicePtr,        // i32 [n_experts]
    pub sort_offsets: DevicePtr,       // i32 [n_experts + 1]
    pub sort_cursors: DevicePtr,       // i32 [n_experts]
    pub sort_sorted_pair_idx: DevicePtr, // i32 [L * top_k]
    // V2.6.a padded sort outputs (only touched when tile8 path is on).
    pub sort_padded_offsets: DevicePtr,   // i32 [n_experts + 1]
    pub sort_sorted_pair_idx_padded: DevicePtr, // i32 [max_tokens * top_k + n_experts * 8]
    x_q8_1_bytes: usize,
    router_logits_bytes: usize,
    expert_ids_bytes: usize,
    expert_weights_bytes: usize,
    gate_up_bytes: usize,
    activated_f32_bytes: usize,
    activated_f16_bytes: usize,
    activated_q8_1_bytes: usize,
    x_q8_1_mmq_bytes: usize,
    activated_q8_1_mmq_bytes: usize,
    down_f32_bytes: usize,
    down_f16_bytes: usize,
    sort_counts_bytes: usize,
    sort_offsets_bytes: usize,
    sort_cursors_bytes: usize,
    sort_sorted_pair_idx_bytes: usize,
    sort_padded_offsets_bytes: usize,
    sort_sorted_pair_idx_padded_bytes: usize,
    disposed: bool,
}

impl MoePrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let top_k = cfg.num_experts_per_tok;
        let n_experts = cfg.num_experts;
        assert!(hidden % 32 == 0);
        assert!(inter % 32 == 0);

        let x_q8_1_bytes =
            max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let router_logits_bytes = max_tokens * n_experts * 4;
        let expert_ids_bytes = max_tokens * top_k * 4;
        let expert_weights_bytes = max_tokens * top_k * 4;
        let gate_up_bytes = max_tokens * top_k * inter * 4;
        let activated_f32_bytes = max_tokens * top_k * inter * 4;
        let activated_f16_bytes = max_tokens * top_k * inter * 2;
        let activated_q8_1_bytes =
            max_tokens * top_k * (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        // V2.14.c DS4 activation buffers. 144 bytes per MMQ block (128 elements).
        // hidden/128 big_blocks × max_tokens rows for gate+up (per-token);
        // inter/128 big_blocks × max_tokens*top_k rows for down (per-pair).
        let x_q8_1_mmq_bytes =
            max_tokens * (hidden / 128) * std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let activated_q8_1_mmq_bytes =
            max_tokens * top_k * (inter / 128) * std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let down_f32_bytes = max_tokens * top_k * hidden * 4;
        let down_f16_bytes = max_tokens * top_k * hidden * 2;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let router_logits = device.alloc(router_logits_bytes)?;
        let expert_ids = device.alloc(expert_ids_bytes)?;
        let expert_weights = device.alloc(expert_weights_bytes)?;
        let gate_out_f32 = device.alloc(gate_up_bytes)?;
        let up_out_f32 = device.alloc(gate_up_bytes)?;
        let activated_f32 = device.alloc(activated_f32_bytes)?;
        let activated_f16 = device.alloc(activated_f16_bytes)?;
        let activated_q8_1 = device.alloc(activated_q8_1_bytes)?;
        let x_q8_1_mmq = device.alloc(x_q8_1_mmq_bytes)?;
        let activated_q8_1_mmq = device.alloc(activated_q8_1_mmq_bytes)?;
        let down_f32 = device.alloc(down_f32_bytes)?;
        let down_f16 = device.alloc(down_f16_bytes)?;

        // V2.5.a sort-by-expert scratch
        let sort_counts_bytes = n_experts * 4;
        let sort_offsets_bytes = (n_experts + 1) * 4;
        let sort_cursors_bytes = n_experts * 4;
        let sort_sorted_pair_idx_bytes = max_tokens * top_k * 4;
        let sort_counts = device.alloc(sort_counts_bytes)?;
        let sort_offsets = device.alloc(sort_offsets_bytes)?;
        let sort_cursors = device.alloc(sort_cursors_bytes)?;
        let sort_sorted_pair_idx = device.alloc(sort_sorted_pair_idx_bytes)?;
        // V2.6.a padded sort outputs. Upper bound on padded total: the
        // real total plus up to 15 padding entries per expert (V2.31.b
        // bumped from 7 to accommodate pad-to-16 for tile16 MMQ; tile8
        // path uses ≤ 7 slack and still fits).
        let sort_padded_offsets_bytes = (n_experts + 1) * 4;
        let sort_sorted_pair_idx_padded_bytes =
            (max_tokens * top_k + n_experts * 16) * 4;
        let sort_padded_offsets = device.alloc(sort_padded_offsets_bytes)?;
        let sort_sorted_pair_idx_padded = device.alloc(sort_sorted_pair_idx_padded_bytes)?;

        Ok(Self {
            max_tokens,
            x_q8_1,
            router_logits,
            expert_ids,
            expert_weights,
            gate_out_f32,
            up_out_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            x_q8_1_mmq,
            activated_q8_1_mmq,
            down_f32,
            down_f16,
            sort_counts,
            sort_offsets,
            sort_cursors,
            sort_sorted_pair_idx,
            sort_padded_offsets,
            sort_sorted_pair_idx_padded,
            x_q8_1_bytes,
            router_logits_bytes,
            expert_ids_bytes,
            expert_weights_bytes,
            gate_up_bytes,
            activated_f32_bytes,
            activated_f16_bytes,
            activated_q8_1_bytes,
            x_q8_1_mmq_bytes,
            activated_q8_1_mmq_bytes,
            down_f32_bytes,
            down_f16_bytes,
            sort_counts_bytes,
            sort_offsets_bytes,
            sort_cursors_bytes,
            sort_sorted_pair_idx_bytes,
            sort_padded_offsets_bytes,
            sort_sorted_pair_idx_padded_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.router_logits, self.router_logits_bytes)?;
            device.dealloc(self.expert_ids, self.expert_ids_bytes)?;
            device.dealloc(self.expert_weights, self.expert_weights_bytes)?;
            device.dealloc(self.gate_out_f32, self.gate_up_bytes)?;
            device.dealloc(self.up_out_f32, self.gate_up_bytes)?;
            device.dealloc(self.activated_f32, self.activated_f32_bytes)?;
            device.dealloc(self.activated_f16, self.activated_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.activated_q8_1_bytes)?;
            device.dealloc(self.x_q8_1_mmq, self.x_q8_1_mmq_bytes)?;
            device.dealloc(self.activated_q8_1_mmq, self.activated_q8_1_mmq_bytes)?;
            device.dealloc(self.down_f32, self.down_f32_bytes)?;
            device.dealloc(self.down_f16, self.down_f16_bytes)?;
            device.dealloc(self.sort_counts, self.sort_counts_bytes)?;
            device.dealloc(self.sort_offsets, self.sort_offsets_bytes)?;
            device.dealloc(self.sort_cursors, self.sort_cursors_bytes)?;
            device.dealloc(self.sort_sorted_pair_idx, self.sort_sorted_pair_idx_bytes)?;
            device.dealloc(self.sort_padded_offsets, self.sort_padded_offsets_bytes)?;
            device.dealloc(self.sort_sorted_pair_idx_padded, self.sort_sorted_pair_idx_padded_bytes)?;
        }
        Ok(())
    }
}

impl Drop for MoePrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "MoePrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Run the MoE router for a prefill chunk. Produces L × top_k expert ids and
/// softmaxed weights. `dense_gemv_f32_f16` is currently 1-row; we loop L
/// times (launch overhead ≈ L µs, negligible at typical chunk sizes).
/// V2 fusion candidate: a true M-dimension variant.
pub fn forward_router_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_inp: &DeviceTensor,
    scratch: &mut MoePrefillScratch,
    x_norm: DevicePtr,
    n_tokens: usize,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let n_experts = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok;

    if ffn_gate_inp.dtype != GgmlDType::F32 {
        bail!("router expects F32 ffn_gate_inp; got {:?}", ffn_gate_inp.dtype);
    }
    if ffn_gate_inp.dims.len() != 2
        || ffn_gate_inp.dims[0] as usize != n_experts
        || ffn_gate_inp.dims[1] as usize != hidden
    {
        bail!(
            "ffn_gate_inp shape {:?} != expected [{n_experts}, {hidden}]",
            ffn_gate_inp.dims
        );
    }

    let x_row_bytes = hidden * 2;
    let logits_row_bytes = n_experts * 4;
    for t in 0..n_tokens {
        dense_gemv_f32_f16(
            ops,
            stream,
            ffn_gate_inp.ptr,
            x_norm.offset_bytes(t * x_row_bytes),
            scratch.router_logits.offset_bytes(t * logits_row_bytes),
            n_experts,
            hidden,
        )
        .with_context(|| format!("prefill router dense_gemv token {t}"))?;
    }

    topk_f32(
        ops,
        stream,
        scratch.router_logits,
        scratch.expert_ids,
        scratch.expert_weights,
        n_tokens,
        n_experts,
        top_k,
    )
    .context("prefill router topk_f32")?;

    Ok(())
}

/// Routed MoE FFN prefill. Mirrors `forward_moe_ffn_decode` but parametrised
/// by `n_tokens`; every indexed-MoE op already takes an `n_tokens` arg.
/// Resolve `FLAMBEAU_MOE_VARIANT` (falling back to `FLAMBEAU_MOE_SORTED=0 → r4`,
/// else `tile8`). Cached on first call for the lifetime of the process —
/// env vars don't change under our runtime, and the prefill hot path hit
/// this twice per layer per token.
fn moe_variant_cached() -> &'static str {
    use std::sync::OnceLock;
    static CACHED: OnceLock<String> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            std::env::var("FLAMBEAU_MOE_VARIANT").ok().unwrap_or_else(|| {
                if std::env::var("FLAMBEAU_MOE_SORTED").as_deref() == Ok("0") {
                    "r4".to_string()
                } else {
                    "tile8".to_string()
                }
            })
        })
        .as_str()
}

pub fn forward_moe_ffn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn: &crate::weights::FfnWeights,
    scratch: &mut MoePrefillScratch,
    x_norm: DevicePtr,
    residual: DevicePtr,
    out: DevicePtr,
    n_tokens: usize,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_moe_ffn_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_moe_ffn_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }

    let ffn_gate_exps = ffn
        .ffn_gate_exps
        .as_ref()
        .context("forward_moe_ffn_prefill: ffn.ffn_gate_exps missing (loader should have rejected a non-MoE layer routed here)")?;
    let ffn_up_exps = ffn
        .ffn_up_exps
        .as_ref()
        .context("forward_moe_ffn_prefill: ffn.ffn_up_exps missing")?;
    let ffn_down_exps = ffn
        .ffn_down_exps
        .as_ref()
        .context("forward_moe_ffn_prefill: ffn.ffn_down_exps missing")?;

    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let top_k = cfg.num_experts_per_tok;
    let n_experts = cfg.num_experts;

    let gate_dt_pre = ffn_gate_exps.dtype;
    let up_dt_pre = ffn_up_exps.dtype;
    let down_dt_pre = ffn_down_exps.dtype;
    validate_moe_dtypes(
        "indexed-MoE prefill",
        gate_dt_pre,
        up_dt_pre,
        down_dt_pre,
        hidden,
        inter,
    )?;
    let nb_per_row_hidden = hidden / QK_K;
    let nb_per_row_inter = inter / QK_K;

    // 1. Quantise x_norm [L, hidden] → Q8_1.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("prefill moe x_norm → Q8_1")?;

    // V2.22.a / V2.23.a — Q8_0 / Q4_0 fast path. Skip sort/pad + MMQ tile8
    // (not ported yet); use plain indexed MoE MMVQ with n_tokens > 1. Slower
    // than tile8 at prefill but structurally correct — unblocks UD-Q8_K_XL
    // and Qwen3.6-35B-A3B-Q4_0 load-and-run. V2.22.b will add MMQ tile8 for
    // Q8_0 to recover prefill throughput.
    //
    // Allowed combos: Q8_0 gate+up requires Q8_0 down; Q4_0 gate+up allows
    // either Q4_0 or Q8_0 down (V2.23.a ffn_down promotion).
    //
    // V2.28.c — at n_tokens >= 32, route Q4_0 through the tile8 MMQ path
    // below (sort+pad+fused-tile kernels). MMVQ fallback stays for small
    // n_tokens where the tile8 kernel's grid overhead dominates.
    const Q4_0_TILE8_THRESHOLD: usize = 32;
    // V2.22.b — tile8 handles three shape classes:
    //   (Q4_0, Q4_0): pure Q4_0 → Q4_0 gate_up + Q4_0 down tile8
    //   (Q4_0, Q8_0): mixed 5-layer case (V2.23.a Q4_1→Q8_0 conversion for
    //                 35B-A3B-Q4_0) → Q4_0 gate_up + Q8_0 down tile8
    //   (Q8_0, Q8_0): pure Q8_0 (UD-Q8_K_XL) → Q8_0 gate_up + Q8_0 down tile8
    // MMVQ fallback stays for n_tokens < 32 where tile8 grid overhead dominates.
    let q4_0_use_tile8 = gate_dt_pre == GgmlDType::Q4_0
        && (down_dt_pre == GgmlDType::Q4_0 || down_dt_pre == GgmlDType::Q8_0)
        && n_tokens >= Q4_0_TILE8_THRESHOLD;
    let q8_0_use_tile8 = gate_dt_pre == GgmlDType::Q8_0
        && down_dt_pre == GgmlDType::Q8_0
        && n_tokens >= Q4_0_TILE8_THRESHOLD;
    if (gate_dt_pre == GgmlDType::Q4_0 && !q4_0_use_tile8)
        || (gate_dt_pre == GgmlDType::Q8_0 && !q8_0_use_tile8)
    {
        match (gate_dt_pre, down_dt_pre) {
            (GgmlDType::Q4_0, GgmlDType::Q4_0)
            | (GgmlDType::Q4_0, GgmlDType::Q8_0)
            | (GgmlDType::Q8_0, GgmlDType::Q8_0) => {}
            _ => bail!(
                "Q4_0/Q8_0 prefill fast path: down_dt must pair with gate_dt as (Q4_0,Q4_0), (Q4_0,Q8_0) or (Q8_0,Q8_0); got ({:?}, {:?})",
                gate_dt_pre, down_dt_pre
            ),
        }
        run_indexed_moe_gate_up(
            ops, stream, gate_dt_pre, ffn_gate_exps.ptr, ffn_up_exps.ptr,
            scratch.x_q8_1, scratch.expert_ids, scratch.gate_out_f32,
            scratch.up_out_f32, inter, n_tokens, top_k, hidden,
        )?;
        swiglu_f32(
            ops, stream,
            scratch.gate_out_f32, scratch.up_out_f32, scratch.activated_f32,
            n_tokens * top_k * inter,
        ).context("prefill moe swiglu_f32 (q4_0/q8_0 path)")?;
        cast_and_quantize_f32_to_q8_1(
            ops,
            stream,
            scratch.activated_f32,
            scratch.activated_f16,
            scratch.activated_q8_1,
            n_tokens * top_k * inter,
            "prefill moe activated (q4_0/q8_0 path)",
        )?;
        // Down matmul: each (token, slot) pair is its own "effective token"
        // with top_k_inner = 1, same pattern as decode-path down.
        run_indexed_moe_down(
            ops, stream, down_dt_pre, ffn_down_exps.ptr, scratch.activated_q8_1,
            scratch.expert_ids, scratch.down_f32,
            hidden, n_tokens * top_k, 1, inter,
        )?;
        cast_f32_to_f16(
            ops, stream, scratch.down_f32, scratch.down_f16,
            n_tokens * top_k * hidden,
        ).context("prefill cast down → f16 (q4_0/q8_0 path)")?;
        moe_combine_f16(
            ops, stream,
            scratch.down_f16, scratch.expert_weights, residual, out,
            n_tokens, top_k, hidden,
        ).context("prefill moe_combine (q4_0/q8_0 path)")?;
        return Ok(());
    }

    // 2. Path selection:
    //   tile8  (V2.6.b, default): sort+pad + 64×8-tile MMQ kernel
    //   sorted (V2.5.b): sort + r4 block reorder
    //   none   (V2.4): raw r4
    // FLAMBEAU_MOE_VARIANT in {tile8, sorted, r4}. Default = tile8.
    // FLAMBEAU_MOE_SORTED=0 still works as a shortcut to force r4.
    // Env vars are process-static — resolve once per process (T3.4), avoid
    // one getenv per layer per token.
    let moe_variant = moe_variant_cached();
    let total_pairs = n_tokens * top_k;
    if moe_variant == "turbo" || moe_variant == "tile8" {
        moe_sort_by_expert_padded(
            ops,
            stream,
            scratch.expert_ids,
            scratch.sort_counts,
            scratch.sort_offsets,
            scratch.sort_cursors,
            scratch.sort_sorted_pair_idx,
            scratch.sort_padded_offsets,
            scratch.sort_sorted_pair_idx_padded,
            total_pairs,
            n_experts,
            scratch.max_tokens,
            top_k,
        )
        .context("prefill moe_sort_by_expert_padded")?;
        // Upper bound on padded_total: real total plus up to 7 padding entries
        // per expert. Kernel early-exits blocks past the actual count.
        let padded_total_ub = total_pairs + n_experts * 8;
    if moe_variant == "turbo" {
        // V2.14.c: DS4 Q8_1 activation for turbo gate_up. Per-TOKEN layout
        // — hidden activation shared across the top_k slots of each token.
        quantize_f16_q8_1_mmq(ops, stream, x_norm, scratch.x_q8_1_mmq, hidden, n_tokens)
            .context("prefill turbo quantize x_norm → Q8_1_MMQ")?;
        indexed_moe_mmq_q4_k_gate_up_turbo(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1_mmq,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx_padded,
            scratch.sort_padded_offsets,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            flambeau_ops::hip::moe::MoeShape {
                n_rows: inter,
                n_tokens,
                top_k,
                n_sb_per_row: nb_per_row_hidden,
                n_experts,
                padded_total_upper_bound: padded_total_ub,
            },
        )
        .context("prefill indexed_moe gate+up turbo")?;
    } else if gate_dt_pre == GgmlDType::Q4_0 {
        // V2.28.c — Q4_0 variant of the tile8 gate+up MMQ. n_sb_per_row for
        // Q4_0 is hidden/32 (no super-block), not hidden/QK_K.
        flambeau_ops::hip::moe::indexed_moe_mmq_q4_0_gate_up_tile8(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx_padded,
            scratch.sort_padded_offsets,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            flambeau_ops::hip::moe::MoeShape {
                n_rows: inter,
                n_tokens,
                top_k,
                n_sb_per_row: hidden / 32,
                n_experts,
                padded_total_upper_bound: padded_total_ub,
            },
        )
        .context("prefill indexed_moe gate+up q4_0 tile8")?;
    } else if gate_dt_pre == GgmlDType::Q8_0 {
        // V2.22.b — Q8_0 variant of the tile8 gate+up MMQ. Same Q4_0 shape
        // conventions (hidden/32 blocks/row).
        flambeau_ops::hip::moe::indexed_moe_mmq_q8_0_gate_up_tile8(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx_padded,
            scratch.sort_padded_offsets,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            flambeau_ops::hip::moe::MoeShape {
                n_rows: inter,
                n_tokens,
                top_k,
                n_sb_per_row: hidden / 32,
                n_experts,
                padded_total_upper_bound: padded_total_ub,
            },
        )
        .context("prefill indexed_moe gate+up q8_0 tile8")?;
    } else {
        indexed_moe_mmq_q4_k_gate_up_tile8(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx_padded,
            scratch.sort_padded_offsets,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            flambeau_ops::hip::moe::MoeShape {
                n_rows: inter,
                n_tokens,
                top_k,
                n_sb_per_row: nb_per_row_hidden,
                n_experts,
                padded_total_upper_bound: padded_total_ub,
            },
        )
        .context("prefill indexed_moe gate+up tile8")?;
    }
    } else if moe_variant == "sorted" {
        moe_sort_by_expert(
            ops,
            stream,
            scratch.expert_ids,
            scratch.sort_counts,
            scratch.sort_offsets,
            scratch.sort_cursors,
            scratch.sort_sorted_pair_idx,
            total_pairs,
            n_experts,
        )
        .context("prefill moe_sort_by_expert")?;
        indexed_moe_mmvq_q4_k_gate_up_sorted(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            inter,
            n_tokens,
            top_k,
            nb_per_row_hidden,
        )
        .context("prefill indexed_moe gate+up (sorted)")?;
    } else {
        indexed_moe_mmvq_q4_k_gate_up(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            inter,
            n_tokens,
            top_k,
            nb_per_row_hidden,
        )
        .context("prefill indexed_moe gate+up")?;
    }

    // 3. SwiGLU over [L, top_k, inter] flat.
    swiglu_f32(
        ops,
        stream,
        scratch.gate_out_f32,
        scratch.up_out_f32,
        scratch.activated_f32,
        n_tokens * top_k * inter,
    )
    .context("prefill moe swiglu_f32")?;

    // 4. Cast + Q8_1-quantise activated. Each (token, slot) pair is one
    // "effective token" in the down matmul's input layout.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.activated_f32,
        scratch.activated_f16,
        n_tokens * top_k * inter,
    )
    .context("prefill cast activated → f16")?;
    if moe_variant == "turbo" {
        // V2.14.c turbo path: DS4 Q8_1 activation for down matmul, per-PAIR layout.
        quantize_f16_q8_1_mmq(
            ops,
            stream,
            scratch.activated_f16,
            scratch.activated_q8_1_mmq,
            inter,
            n_tokens * top_k,
        )
        .context("prefill turbo quantise activated → Q8_1_MMQ")?;
    } else {
        quantize_f16_q8_1(
            ops,
            stream,
            scratch.activated_f16,
            scratch.activated_q8_1,
            n_tokens * top_k * inter,
        )
        .context("prefill quantise activated → Q8_1")?;
    }

    // 5. Down matmul: treat each of L × top_k activations as one
    // "effective token" with top_k_inner = 1 and its own expert id. The
    // scratch's flat `expert_ids` [L, top_k] doubles as the flat lookup
    // [L * top_k] when viewed with stride 1.
    match ffn_down_exps.dtype {
        GgmlDType::Q4K if moe_variant == "turbo" => {
            let padded_total_ub = total_pairs + n_experts * 8;
            indexed_moe_mmq_q4_k_down_turbo(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1_mmq,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                flambeau_ops::hip::moe::MoeShape {
                    n_rows: hidden,
                    n_tokens: n_tokens * top_k,
                    top_k: 1,
                    n_sb_per_row: nb_per_row_inter,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                },
            )
            .context("prefill indexed_moe down q4_k turbo")?;
        }
        GgmlDType::Q4K if moe_variant == "tile8" => {
            let padded_total_ub = total_pairs + n_experts * 8;
            indexed_moe_mmq_q4_k_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                flambeau_ops::hip::moe::MoeShape {
                    n_rows: hidden,
                    n_tokens: n_tokens * top_k,
                    top_k: 1,
                    n_sb_per_row: nb_per_row_inter,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                },
            )
            .context("prefill indexed_moe down q4_k tile8")?;
        }
        GgmlDType::Q4K if moe_variant == "sorted" => indexed_moe_mmvq_q4_k_r2_sorted(
            ops,
            stream,
            ffn_down_exps.ptr,
            scratch.activated_q8_1,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx,
            scratch.down_f32,
            hidden,
            n_tokens * top_k,
            1,
            nb_per_row_inter,
        )
        .context("prefill indexed_moe down q4_k r2 sorted")?,
        GgmlDType::Q4K => indexed_moe_mmvq_q4_k_r2(
            ops,
            stream,
            ffn_down_exps.ptr,
            scratch.activated_q8_1,
            scratch.expert_ids,
            scratch.down_f32,
            hidden,
            n_tokens * top_k, // n_tokens_effective
            1,                // top_k_inner
            nb_per_row_inter,
        )
        .context("prefill indexed_moe down q4_k r2")?,
        GgmlDType::Q6K if moe_variant == "tile8" => {
            let padded_total_ub = total_pairs + n_experts * 8;
            indexed_moe_mmq_q6_k_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                flambeau_ops::hip::moe::MoeShape {
                    n_rows: hidden,
                    n_tokens: n_tokens * top_k,
                    top_k: 1,
                    n_sb_per_row: nb_per_row_inter,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                },
            )
            .context("prefill indexed_moe down q6_k tile8")?;
        }
        GgmlDType::Q6K => indexed_moe_mmvq_q6_k(
            ops,
            stream,
            ffn_down_exps.ptr,
            scratch.activated_q8_1,
            scratch.expert_ids,
            scratch.down_f32,
            hidden,
            n_tokens * top_k,
            1,
            nb_per_row_inter,
        )
        .context("prefill indexed_moe down q6_k")?,
        GgmlDType::Q4_0 if moe_variant == "tile8" => {
            // V2.28.c — Q4_0 down tile8 for MoE prefill.
            let padded_total_ub = total_pairs + n_experts * 8;
            flambeau_ops::hip::moe::indexed_moe_mmq_q4_0_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                flambeau_ops::hip::moe::MoeShape {
                    n_rows: hidden,
                    n_tokens: n_tokens * top_k,
                    top_k: 1,
                    n_sb_per_row: inter / 32,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                },
            )
            .context("prefill indexed_moe down q4_0 tile8")?;
        }
        GgmlDType::Q8_0 if moe_variant == "tile8" => {
            // V2.22.b — Q8_0 down tile8 for MoE prefill. Serves pure Q8_0
            // (UD-Q8_K_XL) AND the mixed Q4_0/Q8_0 layer case (35B-A3B-Q4_0
            // V2.23.a-converted layers).
            let padded_total_ub = total_pairs + n_experts * 8;
            flambeau_ops::hip::moe::indexed_moe_mmq_q8_0_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                flambeau_ops::hip::moe::MoeShape {
                    n_rows: hidden,
                    n_tokens: n_tokens * top_k,
                    top_k: 1,
                    n_sb_per_row: inter / 32,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                },
            )
            .context("prefill indexed_moe down q8_0 tile8")?;
        }
        other => bail!("unreachable: ffn_down_exps dtype {other:?} should have been rejected"),
    }

    // 6. Cast expert outputs to F16.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.down_f32,
        scratch.down_f16,
        n_tokens * top_k * hidden,
    )
    .context("prefill cast down → f16")?;

    // 7. Weighted sum + residual. `moe_combine_f16` handles L natively.
    moe_combine_f16(
        ops,
        stream,
        scratch.down_f16,
        scratch.expert_weights,
        residual,
        out,
        n_tokens,
        top_k,
        hidden,
    )
    .context("prefill moe_combine_f16")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared-expert prefill.
// ---------------------------------------------------------------------------

pub struct SharedExpertPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    pub down_f32: DevicePtr,
    pub x_norm_f32: DevicePtr,
    x_q8_1_bytes: usize,
    inter_f32_bytes: usize,
    inter_f16_bytes: usize,
    inter_q8_1_bytes: usize,
    hidden_f32_bytes: usize,
    disposed: bool,
}

impl SharedExpertPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1);
        let hidden = cfg.hidden_size;
        let inter = cfg
            .shared_expert_intermediate_size
            .context("SharedExpertPrefillScratch requires cfg.shared_expert_intermediate_size")?;
        assert!(hidden % 32 == 0);
        assert!(inter % 32 == 0);

        let x_q8_1_bytes = max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let inter_f32_bytes = max_tokens * inter * 4;
        let inter_f16_bytes = max_tokens * inter * 2;
        let inter_q8_1_bytes =
            max_tokens * (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let hidden_f32_bytes = max_tokens * hidden * 4;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let gate_f32 = device.alloc(inter_f32_bytes)?;
        let up_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f16 = device.alloc(inter_f16_bytes)?;
        let activated_q8_1 = device.alloc(inter_q8_1_bytes)?;
        let down_f32 = device.alloc(hidden_f32_bytes)?;
        let x_norm_f32 = device.alloc(hidden_f32_bytes)?;

        Ok(Self {
            max_tokens,
            x_q8_1,
            gate_f32,
            up_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            x_norm_f32,
            x_q8_1_bytes,
            inter_f32_bytes,
            inter_f16_bytes,
            inter_q8_1_bytes,
            hidden_f32_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.gate_f32, self.inter_f32_bytes)?;
            device.dealloc(self.up_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f16, self.inter_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.inter_q8_1_bytes)?;
            device.dealloc(self.down_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.x_norm_f32, self.hidden_f32_bytes)?;
        }
        Ok(())
    }
}

impl Drop for SharedExpertPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "SharedExpertPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Shared-expert prefill. Mirrors `forward_shared_expert_decode` with
/// `n_tokens = L`; every op (`qmatmul`, `swiglu_f32`, `shared_expert_scale_f32`)
/// already accepts L natively.
pub fn forward_shared_expert_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    shared: &crate::weights::SharedExpertWeights,
    scratch: &mut SharedExpertPrefillScratch,
    x_norm: DevicePtr,
    shared_out: DevicePtr,
    n_tokens: usize,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_shared_expert_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_shared_expert_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }

    let hidden = cfg.hidden_size;
    let inter = cfg
        .shared_expert_intermediate_size
        .context("forward_shared_expert_prefill requires cfg.shared_expert_intermediate_size")?;

    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("prefill shexp x_norm → Q8_1")?;

    // Shared-expert FFN weights are Q4_K on all V1 targets (Qwen3.6 MoE);
    // Q4_K has no MmqLdsX64 row in dispatch, so the DS4 Q8_1 buffer is
    // semantically unused here — DevicePtr(0) is honest, not a placeholder.
    // If a future Q4_K turbo kernel lands, add x_q8_1_mmq to
    // SharedExpertPrefillScratch and populate it alongside x_q8_1.
    run_qmatmul_from_tensor(
        ops, stream, &shared.ffn_gate_shexp, scratch.x_q8_1, DevicePtr(0), scratch.gate_f32,
        n_tokens, hidden, inter, "ffn_gate_shexp",
    )?;
    run_qmatmul_from_tensor(
        ops, stream, &shared.ffn_up_shexp, scratch.x_q8_1, DevicePtr(0), scratch.up_f32,
        n_tokens, hidden, inter, "ffn_up_shexp",
    )?;
    swiglu_f32(
        ops, stream, scratch.gate_f32, scratch.up_f32, scratch.activated_f32,
        n_tokens * inter,
    )
    .context("prefill shexp swiglu_f32")?;
    cast_and_quantize_f32_to_q8_1(
        ops,
        stream,
        scratch.activated_f32,
        scratch.activated_f16,
        scratch.activated_q8_1,
        n_tokens * inter,
        "prefill shexp activated",
    )?;
    run_qmatmul_from_tensor(
        ops, stream, &shared.ffn_down_shexp, scratch.activated_q8_1, DevicePtr(0),
        scratch.down_f32,
        n_tokens, inter, hidden, "ffn_down_shexp",
    )?;
    cast_f16_to_f32(ops, stream, x_norm, scratch.x_norm_f32, n_tokens * hidden)
        .context("prefill shexp cast x_norm → f32")?;
    shared_expert_scale_f32(
        ops, stream, scratch.down_f32, scratch.x_norm_f32,
        shared.ffn_gate_inp_shexp.ptr, n_tokens, hidden,
    )
    .context("prefill shared_expert_scale_f32")?;
    cast_f32_to_f16(ops, stream, scratch.down_f32, shared_out, n_tokens * hidden)
        .context("prefill shexp cast → f16")?;

    Ok(())
}

