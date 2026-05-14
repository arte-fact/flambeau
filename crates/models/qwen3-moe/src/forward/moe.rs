//! MoE forward: routed experts + shared expert + router, decode + prefill.
//! All MoE forward paths live here. Dense layers bypass this module and
//! route through `forward::dense_ffn` instead. The split between routed /
//! shared / router reflects the Qwen3 MoE recipe:
//! - router: dense GEMV producing per-expert logits → topk.
//! - routed experts: indexed MMVQ/MMQ selecting the top-k experts per token.
//! - shared expert: a dense FFN added to every token's output, gated by a
//! learned sigmoid.

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
    moe::{
        shared_expert_scale_f32, topk_f32,
    },
    norm::quantize_f16_q8_1,
    router::dense_gemv_f32_f16,
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::{BlockQ8_1, GgmlDType};

use super::common::{
    run_qmatmul_from_tensor, validate_moe_dtypes,
};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

// ---------------------------------------------------------------------------
// d1 — routed MoE FFN decode step.
// ---------------------------------------------------------------------------

/// Workspace for one decode step of the routed MoE FFN. Wraps
/// [`flambeau_blocks::OwnedMoeExpertsDecodeScratch`]; field access
/// (`scratch.x_q8_1`, `scratch.expert_ids`, …) flows through `Deref` to
/// the inner block scratch.
pub struct MoeScratch {
    inner: flambeau_blocks::OwnedMoeExpertsDecodeScratch,
    tracker: flambeau_blocks::RawAllocTracker,
    disposed: bool,
}

impl MoeScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let mut tracker = flambeau_blocks::RawAllocTracker::new();
        let dims = flambeau_blocks::MoeExpertsScratchDims {
            hidden: cfg.hidden_size,
            intermediate: cfg.moe_intermediate_size,
            n_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
        };
        let inner =
            flambeau_blocks::MoeExperts::alloc_decode_scratch(device, &mut tracker, dims)?;
        Ok(Self { inner, tracker, disposed: false })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        self.tracker.dispose(device)
    }

    /// View shaped for `flambeau_blocks::MoeExperts` decode methods.
    pub fn view(&self) -> flambeau_blocks::MoeExpertsDecodeScratch {
        self.inner.view()
    }
}

impl std::ops::Deref for MoeScratch {
    type Target = flambeau_blocks::OwnedMoeExpertsDecodeScratch;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for MoeScratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Build a `flambeau_blocks::MoeExperts` from already-unpacked
/// MoE-FFN weights + the model config.
pub fn build_moe_experts_block(
    ffn_gate_inp: &DeviceTensor,
    ffn: &crate::weights::FfnWeights,
    cfg: &Qwen3MoEConfig,
) -> Result<flambeau_blocks::MoeExperts> {
    use super::common::qdtype_of;
    let ffn_gate_exps = ffn
        .ffn_gate_exps
        .as_ref()
        .context("build_moe_experts_block: ffn.ffn_gate_exps missing")?;
    let ffn_up_exps = ffn
        .ffn_up_exps
        .as_ref()
        .context("build_moe_experts_block: ffn.ffn_up_exps missing")?;
    let ffn_down_exps = ffn
        .ffn_down_exps
        .as_ref()
        .context("build_moe_experts_block: ffn.ffn_down_exps missing")?;
    let router_dtype = qdtype_of(ffn_gate_inp.dtype)?;
    let gate_dtype = qdtype_of(ffn_gate_exps.dtype)?;
    let up_dtype = qdtype_of(ffn_up_exps.dtype)?;
    let down_dtype = qdtype_of(ffn_down_exps.dtype)?;
    // Router + expert weights are stored as multi-D piles and may not
    // strictly be 2-D in fixtures; compute dims from cfg so the block
    // doesn't depend on the exact upload-side flattening.
    flambeau_blocks::MoeExperts::new(
        flambeau_blocks::WeightHandle {
            ptr: ffn_gate_inp.ptr,
            dtype: router_dtype,
            dims: [cfg.num_experts, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_gate_exps.ptr,
            dtype: gate_dtype,
            dims: [cfg.num_experts * cfg.moe_intermediate_size, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_up_exps.ptr,
            dtype: up_dtype,
            dims: [cfg.num_experts * cfg.moe_intermediate_size, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_down_exps.ptr,
            dtype: down_dtype,
            dims: [cfg.num_experts * cfg.hidden_size, cfg.moe_intermediate_size],
        },
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        cfg.num_experts,
        cfg.num_experts_per_tok,
    )
}

/// One decode step of the routed MoE FFN. Assumes the caller has:
/// - Run `post_attention_norm` on the residual stream (so `x_norm` is the
/// norm output).
/// - Already filled `scratch.expert_ids` and `scratch.expert_weights` with
/// the router's output. d3 will land the router; until then the
/// caller is synthetic (test fixture or hand-rolled top-k).
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
    let ffn_gate_inp = ffn.ffn_gate_inp.as_ref().context(
        "forward_moe_ffn_decode: ffn.ffn_gate_inp missing (router weight); routing must precede expert forward",
    )?;
    let ffn_gate_exps = ffn
        .ffn_gate_exps
        .as_ref()
        .context("forward_moe_ffn_decode: ffn.ffn_gate_exps missing")?;
    let ffn_up_exps = ffn
        .ffn_up_exps
        .as_ref()
        .context("forward_moe_ffn_decode: ffn.ffn_up_exps missing")?;
    let ffn_down_exps = ffn
        .ffn_down_exps
        .as_ref()
        .context("forward_moe_ffn_decode: ffn.ffn_down_exps missing")?;
    validate_moe_dtypes(
        "indexed-MoE",
        ffn_gate_exps.dtype,
        ffn_up_exps.dtype,
        ffn_down_exps.dtype,
        cfg.hidden_size,
        cfg.moe_intermediate_size,
    )?;
    let block = build_moe_experts_block(ffn_gate_inp, ffn, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_decode(&hipops, x_norm, residual, extra_residual, out, scratch.view())
}

// ---------------------------------------------------------------------------
// d2 — shared expert decode step.
// ---------------------------------------------------------------------------

/// Workspace for one decode step of the shared expert (dense FFN +
/// per-token sigmoid gate scaling). Wraps
/// [`flambeau_blocks::OwnedSharedExpertDecodeScratch`]; field access
/// flows through `Deref` to the inner block scratch.
pub struct SharedExpertScratch {
    inner: flambeau_blocks::OwnedSharedExpertDecodeScratch,
    tracker: flambeau_blocks::RawAllocTracker,
    disposed: bool,
}

impl SharedExpertScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let intermediate = cfg
            .shared_expert_intermediate_size
            .context("SharedExpertScratch requires cfg.shared_expert_intermediate_size")?;
        let mut tracker = flambeau_blocks::RawAllocTracker::new();
        let dims = flambeau_blocks::SharedExpertScratchDims {
            hidden: cfg.hidden_size,
            intermediate,
        };
        let inner =
            flambeau_blocks::SharedExpert::alloc_decode_scratch(device, &mut tracker, dims)?;
        Ok(Self { inner, tracker, disposed: false })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        self.tracker.dispose(device)
    }

    /// View shaped for `flambeau_blocks::SharedExpert::forward_decode`.
    pub fn view(&self) -> flambeau_blocks::SharedExpertDecodeScratch {
        self.inner.view()
    }
}

impl std::ops::Deref for SharedExpertScratch {
    type Target = flambeau_blocks::OwnedSharedExpertDecodeScratch;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for SharedExpertScratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Build a `flambeau_blocks::SharedExpert` from already-unpacked
/// shared-expert weights + the model config.
pub fn build_shared_expert_block(
    shared: &crate::weights::SharedExpertWeights,
    cfg: &Qwen3MoEConfig,
) -> Result<flambeau_blocks::SharedExpert> {
    use super::common::qdtype_of;
    let inter = cfg
        .shared_expert_intermediate_size
        .context("build_shared_expert_block requires cfg.shared_expert_intermediate_size")?;
    let g_dt = qdtype_of(shared.ffn_gate_shexp.dtype)?;
    let u_dt = qdtype_of(shared.ffn_up_shexp.dtype)?;
    let d_dt = qdtype_of(shared.ffn_down_shexp.dtype)?;
    flambeau_blocks::SharedExpert::new(
        Some(shared.ffn_gate_inp_shexp.ptr),
        flambeau_blocks::WeightHandle {
            ptr: shared.ffn_gate_shexp.ptr,
            dtype: g_dt,
            dims: [inter, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: shared.ffn_up_shexp.ptr,
            dtype: u_dt,
            dims: [inter, cfg.hidden_size],
        },
        flambeau_blocks::WeightHandle {
            ptr: shared.ffn_down_shexp.ptr,
            dtype: d_dt,
            dims: [cfg.hidden_size, inter],
        },
        cfg.hidden_size,
        inter,
    )
}

/// One decode step of the shared expert (always-on dense FFN), composed
/// with the learned per-token sigmoid-gate scaling:
/// gate_scalar[t] = sigmoid(⟨ ffn_gate_inp_shexp, x_norm[t] ⟩)
/// dense[t] = down_shexp(swiglu(gate_shexp(x_norm[t]), up_shexp(x_norm[t])))
/// shared_out[t] = gate_scalar[t] * dense[t]
/// Output (`shared_out`) is a standalone F16 **delta** — the caller is
/// expected to sum it with the routed-MoE output and the residual in
/// e. Keeping this delta-only keeps the composition orthogonal:
/// routed and shared contributions flow through the same combine layer in
/// e without re-using `residual` for a second purpose.
pub fn forward_shared_expert_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    shared: &crate::weights::SharedExpertWeights,
    scratch: &mut SharedExpertScratch,
    x_norm: DevicePtr,
    shared_out: DevicePtr,
) -> Result<()> {
    let block = build_shared_expert_block(shared, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_decode(&hipops, x_norm, shared_out, scratch.view())
}

// Dense-FFN decode + prefill (DenseFfnScratch, forward_dense_ffn_decode,
// DenseFfnPrefillScratch, forward_dense_ffn_prefill) moved to `forward::dense_ffn`.

// ---------------------------------------------------------------------------
// d3 — MoE router.
// ---------------------------------------------------------------------------

/// Run the MoE router for one decode token. Reads `x_norm` and the FFN's
/// `ffn_gate_inp` weight; writes the top-k selected expert ids + their
/// softmaxed weights into the MoE scratch buffers that
/// `forward_moe_ffn_decode` consumes.
/// Two-stage path:
/// 1. `dense_gemv_f32_f16(ffn_gate_inp, x_norm)` → `router_logits` F32 [n_experts]
/// 2. `topk_f32(router_logits, expert_ids, expert_weights, 1, n_experts, top_k)`
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

    // accept either F32 or F16 router weight. F16
    // is the iter-3 default (loader converts F32→F16 at upload — see
    // sharded.rs); F32 stays as a fallback when a model legitimately
    // ships an F32 router (older Qwen3.5 GGUFs predate the conversion).
    if ffn_gate_inp.dtype != GgmlDType::F32 && ffn_gate_inp.dtype != GgmlDType::F16 {
        bail!(
            "router expects F32 or F16 ffn_gate_inp; got {:?}",
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

    if ffn_gate_inp.dtype == GgmlDType::F16 {
        flambeau_ops::hip::router::dense_gemv_f16_f16(
            ops,
            stream,
            ffn_gate_inp.ptr,
            x_norm,
            scratch.router_logits,
            n_experts,
            hidden,
        )
        .context("router dense_gemv_f16_f16")?;
    } else {
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
    }

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
// f3 — MoE + shared expert + router prefill.
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
    // 4.c DS4 Q8_1 activation buffers (turbo MoE variant only).
    // `x_q8_1_mmq`: hidden activation in DS4 layout — [hidden/128, n_tokens].
    // `activated_q8_1_mmq`: per-pair SwiGLU'd activation in DS4 layout — [inter/128, n_pairs].
    pub x_q8_1_mmq: DevicePtr,
    pub activated_q8_1_mmq: DevicePtr,
    pub down_f32: DevicePtr,           // F32 [L, top_k, hidden]
    pub down_f16: DevicePtr,
    // sort-by-expert state. Only populated / used when
    // FLAMBEAU_MOE_SORTED=1 is set on the gate+up path.
    pub sort_counts: DevicePtr,        // i32 [n_experts]
    pub sort_offsets: DevicePtr,       // i32 [n_experts + 1]
    pub sort_cursors: DevicePtr,       // i32 [n_experts]
    pub sort_sorted_pair_idx: DevicePtr, // i32 [L * top_k]
    // padded sort outputs (only touched when tile8 path is on).
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
        // 4.c DS4 activation buffers. 144 bytes per MMQ block (128 elements).
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

        // sort-by-expert scratch
        let sort_counts_bytes = n_experts * 4;
        let sort_offsets_bytes = (n_experts + 1) * 4;
        let sort_cursors_bytes = n_experts * 4;
        let sort_sorted_pair_idx_bytes = max_tokens * top_k * 4;
        let sort_counts = device.alloc(sort_counts_bytes)?;
        let sort_offsets = device.alloc(sort_offsets_bytes)?;
        let sort_cursors = device.alloc(sort_cursors_bytes)?;
        let sort_sorted_pair_idx = device.alloc(sort_sorted_pair_idx_bytes)?;
        // padded sort outputs. Upper bound on padded total: the
        // real total plus up to 15 padding entries per expert (1.b
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

impl MoePrefillScratch {
    /// View shaped for `flambeau_blocks::MoeExperts` prefill methods.
    pub fn view(&self) -> flambeau_blocks::MoeExpertsPrefillScratch {
        flambeau_blocks::MoeExpertsPrefillScratch {
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
    if ffn_gate_inp.dtype != GgmlDType::F32 && ffn_gate_inp.dtype != GgmlDType::F16 {
        bail!("router expects F32 or F16 ffn_gate_inp; got {:?}", ffn_gate_inp.dtype);
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
    if ffn_gate_inp.dtype == GgmlDType::F16 {
        flambeau_ops::hip::router::dense_gemv_f16_f16_batched(
            ops,
            stream,
            ffn_gate_inp.ptr,
            x_norm,
            scratch.router_logits,
            n_experts,
            hidden,
            n_tokens,
        )
        .context("router dense_gemv_f16_f16_batched (prefill)")?;
    } else {
        flambeau_ops::hip::router::dense_gemv_f32_f16_batched(
            ops,
            stream,
            ffn_gate_inp.ptr,
            x_norm,
            scratch.router_logits,
            n_experts,
            hidden,
            n_tokens,
        )
        .context("prefill router dense_gemv batched")?;
    }
    flambeau_ops::hip::moe::topk_f32(
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
    let ffn_gate_inp = ffn.ffn_gate_inp.as_ref().context(
        "forward_moe_ffn_prefill: ffn.ffn_gate_inp missing (router weight)",
    )?;
    let block = build_moe_experts_block(ffn_gate_inp, ffn, cfg)?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_prefill(
        &hipops,
        x_norm,
        residual,
        None,
        out,
        n_tokens,
        scratch.view(),
    )
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

impl SharedExpertPrefillScratch {
    /// View shaped for `flambeau_blocks::SharedExpert::forward_prefill`.
    pub fn view(&self) -> flambeau_blocks::SharedExpertPrefillScratch {
        flambeau_blocks::SharedExpertPrefillScratch {
            max_tokens: self.max_tokens,
            x_q8_1: self.x_q8_1,
            gate_f32: self.gate_f32,
            up_f32: self.up_f32,
            activated_f16: self.activated_f16,
            activated_q8_1: self.activated_q8_1,
            down_f32: self.down_f32,
            x_norm_f32: self.x_norm_f32,
        }
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
    // 3.d.2 — fused swiglu_f32_to_f16 + quantize (prefill shared expert).
    flambeau_ops::hip::mlp::swiglu_f32_to_f16(
        ops, stream, scratch.gate_f32, scratch.up_f32, scratch.activated_f16,
        n_tokens * inter,
    )
    .context("prefill shexp swiglu_f32_to_f16")?;
    flambeau_ops::hip::norm::quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        n_tokens * inter,
    )
    .context("prefill shexp quantize activated → Q8_1")?;
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

