//! rmsnorm → quantise → router qmatmul → host top-k + softmax-over-k →
//! `flambeau_model_ops::MoeExperts::forward_decode_tp_f32` (indexed
//! batched-expert gate+up / activate / down / weighted-sum) → AR →
//! cast → optional shared expert → delta.
//!
//! Indexed-MoE dispatch collapses 256+ per-expert MMVQ launches per
//! token (Qwen3.6 top_k=8 × 32 layers) into 32 batched
//! `indexed_moe_mmvq_*` launches, closing the legacy/v2 perf gap
//! documented in `certs/perf/d4_batched_decode/v2_vs_legacy_profile.md`.

use anyhow::{bail, Result};
use flambeau_core::DevicePtr;
use flambeau_model_ops::moe_experts::{MoeExperts, MoeExpertsDecodeScratch, RouterPolicy};
use flambeau_model_ops::WeightHandle;
use flambeau_model_ops::{Tensor, F16, F32, I32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::{Activation, MoeWeights};
use flambeau_model_ops::moe_experts::Activation as BlockActivation;

pub fn moe_ffn_local<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &MoeWeights,
    n_tokens: usize,
    next_norm: Option<&Tensor<F16>>,
) -> Result<Option<Tensor<F16>>> {
    let input_pre_normed = state.pool.input_pre_normed;
    state.pool.input_pre_normed = false;
    let hidden = state.hidden();
    // gemma4 cascade: 5-norm F32 path with shared MLP + routed MoE +
    // per-expert scale fold. Detected by the cascade fields being Some
    // (gemma4 loader populates them all together).
    if weights.pre_router_weight_f16.is_some() {
        return gemma4_moe_cascade_local(state, hooks, input, weights, n_tokens);
    }
    if n_tokens > 1 {
        return moe_ffn_loop(state, hooks, input, weights, n_tokens, next_norm);
    }
    let m = state.pool.config.intermediate;
    let n_experts = weights.n_experts;
    let k_top = weights.experts_per_tok;
    if n_experts > state.pool.config.max_experts {
        bail!(
            "moe_ffn: n_experts {n_experts} > pool.max_experts {}",
            state.pool.config.max_experts
        );
    }
    if k_top == 0 || k_top > n_experts {
        bail!("moe_ffn: experts_per_tok {k_top} must be in 1..={n_experts}");
    }
    if k_top > state.pool.config.max_experts_per_tok {
        bail!(
            "moe_ffn: experts_per_tok {k_top} > pool.max_experts_per_tok {} — \
             update ScratchConfig.max_experts_per_tok",
            state.pool.config.max_experts_per_tok
        );
    }
    if weights.experts_gate.len() != n_experts
        || weights.experts_up.len() != n_experts
        || weights.experts_down.len() != n_experts
    {
        bail!(
            "moe_ffn: per-expert weight slices must have len {n_experts}; got gate={}/up={}/down={}",
            weights.experts_gate.len(),
            weights.experts_up.len(),
            weights.experts_down.len()
        );
    }

    let ops = state.ops();
    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };

    // 1. rmsnorm to F16 + quantise to Q8_1. The Q8_1 is reused by the
    //    block's gate_up indexed kernel AND by the shared expert when
    //    present. The block does its own quantise too (idempotent on
    //    the same buffer) — one extra ~4µs launch per layer; acceptable.
    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, hidden) };
    let x_norm_f16_ptr = state.pool.norm;
    if !input_pre_normed {
        let mut x_norm_f16 = unsafe { Tensor::<F16>::from_raw(x_norm_f16_ptr, hidden) };
        flambeau_model_ops::rmsnorm_f16(
            input,
            &weights.ffn_norm,
            &mut x_norm_f16,
            1,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
    }
    let x_norm_f16 = unsafe { Tensor::<F16>::from_raw(x_norm_f16_ptr, hidden) };
    flambeau_model_ops::quantize_f16_to_q8_1(&x_norm_f16, &mut norm_q8_1, hidden, &ops)?;

    // 2. Router via v2 qmatmul (works for both F16/F32 and quantised
    //    router weights; the block's route_decode only handles F16/F32).
    let mut router_logits =
        unsafe { Tensor::<F32>::from_raw(state.pool.router_logits_f32, n_experts) };
    weights.router.qmatmul(
        &norm_q8_1,
        &act_mmq_null,
        &mut router_logits,
        flambeau_ops::MatmulShape {
            m: 1,
            k: hidden,
            n: n_experts,
        },
        &ops,
    )?;

    // 3. GPU-side top-k + softmax-over-k (algebraically equivalent
    //    to softmax-over-V → top-k → renormalise; see topk_f32.cu's
    //    proof comment). Replaces DtoH + CPU topk + HtoD with a
    //    single in-block kernel.
    let mut expert_ids = unsafe { Tensor::<I32>::from_raw(state.pool.moe_expert_ids, k_top) };
    let mut expert_weights =
        unsafe { Tensor::<F32>::from_raw(state.pool.moe_expert_weights, k_top) };
    flambeau_model_ops::moe_router_topk_f32(
        &router_logits,
        &mut expert_ids,
        &mut expert_weights,
        1,
        n_experts,
        k_top,
        &ops,
    )?;

    // 5. Indexed-MoE expert dispatch via flambeau_model_ops::MoeExperts.
    //    `experts_gate[0].ptr` is the base of the stacked packed-expert
    //    tensor (loader::upload_moe_experts_stacked guarantees this);
    //    block treats it as a 2D `[n_experts * inter, hidden]` view.
    let gate_dt = weights.experts_gate[0].dtype;
    let up_dt = weights.experts_up[0].dtype;
    let down_dt = weights.experts_down[0].dtype;
    let router_dt = weights.router.dtype;
    let block = MoeExperts::new(
        flambeau_model_ops::MoeExpertsWeights {
            ffn_gate_inp: WeightHandle {
                ptr: weights.router.ptr,
                dtype: router_dt,
                dims: [n_experts, hidden],
            },
            ffn_gate_exps: WeightHandle {
                ptr: weights.experts_gate[0].ptr,
                dtype: gate_dt,
                dims: [n_experts * m, hidden],
            },
            ffn_up_exps: WeightHandle {
                ptr: weights.experts_up[0].ptr,
                dtype: up_dt,
                dims: [n_experts * m, hidden],
            },
            ffn_down_exps: WeightHandle {
                ptr: weights.experts_down[0].ptr,
                dtype: down_dt,
                dims: [n_experts * hidden, m],
            },
        },
        flambeau_model_ops::MoeExpertsDims {
            hidden,
            intermediate: m,
            n_experts,
            top_k: k_top,
        },
    )?
    .with_router_policy(RouterPolicy::default())
    .with_activation(match weights.activation {
        Activation::SwiGLU => BlockActivation::SwiGLU,
        Activation::GeluTanh => BlockActivation::Gelu,
    });

    let scratch_view = MoeExpertsDecodeScratch {
        x_q8_1: state.pool.norm_q8_1,
        router_logits: state.pool.router_logits_f32,
        expert_ids: state.pool.moe_expert_ids,
        expert_weights: state.pool.moe_expert_weights,
        gate_out_f32: state.pool.moe_gate_out_f32,
        up_out_f32: state.pool.moe_up_out_f32,
        activated_f16: state.pool.moe_activated_f16,
        activated_q8_1: state.pool.moe_activated_q8_1,
        down_f32: state.pool.moe_down_f32,
        down_f16: state.pool.moe_down_f16,
    };

    // F32 partial → AR (no-op on SD/PP) → cast F32→F16 → delta.
    // `pool.down_f32` is sized hidden×F32; perfect for partial output.
    block.forward_decode_tp_f32(&ops, x_norm_f16_ptr, state.pool.down_f32, scratch_view)?;

    // Fast path: BAR1 TP=2 — fuse AR + residual-add for the routed
    // MoE partial AND (if present) the shared-expert partial.
    // Returns None so the model skips its `residual_add`.
    // Disabled when post_ffn_norm is set (gemma4 MoE): the post-norm
    // must be applied to the delta before residual_add, which the
    // AR-fold fast path doesn't expose. Slow path below handles it.
    if hooks.supports_ar_residual_f16() && weights.post_ffn_norm.is_none() {
        let down_f32_t = unsafe { Tensor::<F32>::from_raw(state.pool.down_f32, hidden) };
        let mut moe_partial_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
        flambeau_model_ops::cast_f32_to_f16(&down_f32_t, &mut moe_partial_f16, hidden, &ops)?;

        if let Some(sh) = weights.shared.as_ref() {
            if state.pool.shared_x_norm_f32.as_usize() == 0 && sh.gate_inp.is_some() {
                bail!(
                    "moe_ffn: shared expert with per-token gate present but \
                     pool.shared_x_norm_f32 unallocated — set \
                     ScratchConfig.shared_intermediate"
                );
            }
            let shared_block = build_shared_expert_block(sh, hidden, weights.activation)?;
            let scratch_view = flambeau_model_ops::SharedExpertDecodeScratch {
                x_q8_1: state.pool.norm_q8_1,
                gate_f32: state.pool.gate_f32,
                up_f32: state.pool.up_f32,
                activated_f16: state.pool.gated_f16,
                activated_q8_1: state.pool.gated_q8_1,
                down_f32: state.pool.down_f32,
                x_norm_f32: state.pool.shared_x_norm_f32,
            };
            let shared_out = state.pool.q_f16;
            shared_block.forward_decode(&ops, x_norm_f16_ptr, shared_out, scratch_view)?;
            let moe_view = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
            let shared_view = unsafe { Tensor::<F16>::from_raw(shared_out, hidden) };
            let mut merged = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
            flambeau_model_ops::add_f16(&moe_view, &shared_view, &mut merged, hidden, &ops)?;
        }

        let fuse_into_norm = next_norm.is_some() && hooks.supports_ar_residual_rmsnorm_f16();
        if fuse_into_norm {
            let next_w = next_norm.unwrap();
            hooks.ar_residual_rmsnorm_f16(
                crate::core::ArResidualRmsNormHookBuffers {
                    residual_inout: input.ptr,
                    partial_f16: state.pool.delta,
                    rms_weight: next_w.ptr,
                    out_norm: state.pool.norm,
                },
                hidden,
                weights.rms_eps,
                state.device,
                state.stream,
            )?;
            state.pool.input_pre_normed = true;
        } else {
            hooks.ar_residual_f16(
                input.ptr,
                state.pool.delta,
                hidden,
                state.device,
                state.stream,
            )?;
        }
        let _ = (act_mmq_null, gate_dt, up_dt, down_dt, router_dt);
        return Ok(None);
    }

    hooks.ar_sum_f32(state.pool.down_f32, hidden, state.device, state.stream)?;
    let down_f32_t = unsafe { Tensor::<F32>::from_raw(state.pool.down_f32, hidden) };
    let mut delta_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
    flambeau_model_ops::cast_f32_to_f16(&down_f32_t, &mut delta_f16, hidden, &ops)?;

    // 6. Shared expert (qwen3-moe family): adds into the same delta.
    if let Some(sh) = weights.shared.as_ref() {
        if state.pool.shared_x_norm_f32.as_usize() == 0 && sh.gate_inp.is_some() {
            bail!(
                "moe_ffn: shared expert with per-token gate present but \
                 pool.shared_x_norm_f32 unallocated — set \
                 ScratchConfig.shared_intermediate"
            );
        }
        let shared_block = build_shared_expert_block(sh, hidden, weights.activation)?;
        let scratch_view = flambeau_model_ops::SharedExpertDecodeScratch {
            x_q8_1: state.pool.norm_q8_1,
            gate_f32: state.pool.gate_f32,
            up_f32: state.pool.up_f32,
            activated_f16: state.pool.gated_f16,
            activated_q8_1: state.pool.gated_q8_1,
            down_f32: state.pool.down_f32,
            x_norm_f32: state.pool.shared_x_norm_f32,
        };
        // shared_out target: pool.q_f16 is sized [q_width] F16 which
        // is ≥ hidden on every supported MoE arch.
        let shared_out = state.pool.q_f16;
        shared_block.forward_decode(&ops, x_norm_f16_ptr, shared_out, scratch_view)?;
        // Under TP, down_shexp is row-parallel and shared_out is a
        // rank-local partial. Cast → F32 → ar_sum_f32 → F16 to fold
        // the cross-rank sum (no-op on SD/PP).
        let shared = unsafe { Tensor::<F16>::from_raw(shared_out, hidden) };
        let mut shared_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.attn_proj_f32, hidden) };
        flambeau_model_ops::cast_f16_to_f32(&shared, &mut shared_f32, hidden, &ops)?;
        hooks.ar_sum_f32(shared_f32.ptr, hidden, state.device, state.stream)?;
        let mut shared_synced = unsafe { Tensor::<F16>::from_raw(shared_out, hidden) };
        flambeau_model_ops::cast_f32_to_f16(&shared_f32, &mut shared_synced, hidden, &ops)?;
        let shared_view = unsafe { Tensor::<F16>::from_raw(shared_out, hidden) };
        let delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
        let mut delta_out = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
        flambeau_model_ops::add_f16(&delta, &shared_view, &mut delta_out, hidden, &ops)?;
    }

    // Optional post-ffn rmsnorm on the F16 delta (gemma4 MoE pattern).
    if let Some(post_norm) = weights.post_ffn_norm.as_ref() {
        let delta_in = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
        let mut delta_out = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
        flambeau_model_ops::rmsnorm_f16(
            &delta_in,
            post_norm,
            &mut delta_out,
            1,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
    }
    let _ = (act_mmq_null, gate_dt, up_dt, down_dt, router_dt);
    Ok(Some(unsafe {
        Tensor::<F16>::from_raw(state.pool.delta, hidden)
    }))
}

/// Batched prefill MoE path. Used whenever `n_tokens > 1`. Mirrors the
/// decode path in `moe_ffn_local` but at N tokens: one rmsnorm + one
/// router qmatmul + one batched DtoH of router logits + per-token host
/// topk (cheap CPU loop) + one HtoD of expert_ids/weights + one block
/// `forward_prefill_tp_f32` call. The block dispatches MMQ tile8
/// kernels at `n_pairs ≥ 8`.
///
/// Shared expert (when present): looped per token after the routed
/// path. SharedExpert block has no prefill variant; the dense FFN
/// inside still fires at n_tokens=1 per loop iteration, but it's a
/// small per-layer add — the dominant cost is the routed path which
/// is now batched.
fn moe_ffn_loop<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &MoeWeights,
    n_tokens: usize,
    _next_norm: Option<&Tensor<F16>>,
) -> Result<Option<Tensor<F16>>> {
    let hidden = state.hidden();
    let m = state.pool.config.intermediate;
    let n_experts = weights.n_experts;
    let k_top = weights.experts_per_tok;
    if n_experts > state.pool.config.max_experts {
        bail!(
            "moe_ffn: n_experts {n_experts} > pool.max_experts {}",
            state.pool.config.max_experts
        );
    }
    if k_top == 0 || k_top > n_experts {
        bail!("moe_ffn: experts_per_tok {k_top} must be in 1..={n_experts}");
    }
    if n_tokens > state.pool.config.max_prefill_tokens {
        bail!(
            "moe_ffn: n_tokens {n_tokens} > pool.max_prefill_tokens {}",
            state.pool.config.max_prefill_tokens
        );
    }
    let prefill_scratch = state
        .pool
        .moe_prefill_scratch
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "moe_ffn batched prefill: pool not configured (need max_prefill_tokens > 1)"
            )
        })?
        .view();

    let ops = state.ops();
    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };

    // 1. rmsnorm at n_tokens=N → F16 norm scratch (pool.norm).
    let x_norm_f16_ptr = state.pool.norm;
    let input_view = unsafe { Tensor::<F16>::from_raw(input.ptr, n_tokens * hidden) };
    let mut x_norm_f16 = unsafe { Tensor::<F16>::from_raw(x_norm_f16_ptr, n_tokens * hidden) };
    flambeau_model_ops::rmsnorm_f16(
        &input_view,
        &weights.ffn_norm,
        &mut x_norm_f16,
        n_tokens,
        hidden,
        weights.rms_eps,
        &ops,
    )?;
    // Quantise [N, hidden] → block's prefill x_q8_1 buffer.
    let mut x_q8_1_n =
        unsafe { Tensor::<Q8_1>::from_raw(prefill_scratch.x_q8_1, n_tokens * hidden) };
    flambeau_model_ops::quantize_f16_to_q8_1(&x_norm_f16, &mut x_q8_1_n, n_tokens * hidden, &ops)?;

    // 2. Router qmatmul at n_tokens=N → router_logits [N, n_experts] F32.
    let mut router_logits =
        unsafe { Tensor::<F32>::from_raw(prefill_scratch.router_logits, n_tokens * n_experts) };
    weights.router.qmatmul(
        &x_q8_1_n,
        &act_mmq_null,
        &mut router_logits,
        flambeau_ops::MatmulShape {
            m: n_tokens,
            k: hidden,
            n: n_experts,
        },
        &ops,
    )?;

    // 3. GPU-side per-token top-k + softmax-over-k (replaces DtoH
    //    + per-row CPU topk + HtoD).
    let mut expert_ids =
        unsafe { Tensor::<I32>::from_raw(prefill_scratch.expert_ids, n_tokens * k_top) };
    let mut expert_weights =
        unsafe { Tensor::<F32>::from_raw(prefill_scratch.expert_weights, n_tokens * k_top) };
    flambeau_model_ops::moe_router_topk_f32(
        &router_logits,
        &mut expert_ids,
        &mut expert_weights,
        n_tokens,
        n_experts,
        k_top,
        &ops,
    )?;

    // 5. Batched indexed-MoE forward + F32 partial AR.
    let gate_dt = weights.experts_gate[0].dtype;
    let up_dt = weights.experts_up[0].dtype;
    let down_dt = weights.experts_down[0].dtype;
    let router_dt = weights.router.dtype;
    let block = MoeExperts::new(
        flambeau_model_ops::MoeExpertsWeights {
            ffn_gate_inp: WeightHandle {
                ptr: weights.router.ptr,
                dtype: router_dt,
                dims: [n_experts, hidden],
            },
            ffn_gate_exps: WeightHandle {
                ptr: weights.experts_gate[0].ptr,
                dtype: gate_dt,
                dims: [n_experts * m, hidden],
            },
            ffn_up_exps: WeightHandle {
                ptr: weights.experts_up[0].ptr,
                dtype: up_dt,
                dims: [n_experts * m, hidden],
            },
            ffn_down_exps: WeightHandle {
                ptr: weights.experts_down[0].ptr,
                dtype: down_dt,
                dims: [n_experts * hidden, m],
            },
        },
        flambeau_model_ops::MoeExpertsDims {
            hidden,
            intermediate: m,
            n_experts,
            top_k: k_top,
        },
    )?
    .with_router_policy(RouterPolicy::default())
    .with_activation(match weights.activation {
        Activation::SwiGLU => BlockActivation::SwiGLU,
        Activation::GeluTanh => BlockActivation::Gelu,
    });

    // `pool.down_f32` is sized `n * h * f32` = max_prefill_tokens × hidden,
    // sufficient for N×hidden F32 partial out.
    block.forward_prefill_tp_f32(
        &ops,
        x_norm_f16_ptr,
        state.pool.down_f32,
        n_tokens,
        prefill_scratch,
    )?;
    hooks.ar_sum_f32(
        state.pool.down_f32,
        n_tokens * hidden,
        state.device,
        state.stream,
    )?;
    let down_full = unsafe { Tensor::<F32>::from_raw(state.pool.down_f32, n_tokens * hidden) };
    let mut delta_full = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden) };
    flambeau_model_ops::cast_f32_to_f16(&down_full, &mut delta_full, n_tokens * hidden, &ops)?;

    // 6. Shared expert (per-token; block has no prefill variant). Adds
    // each token's shared_out into delta[t].
    if let Some(sh) = weights.shared.as_ref() {
        if state.pool.shared_x_norm_f32.as_usize() == 0 && sh.gate_inp.is_some() {
            bail!(
                "moe_ffn: shared expert with per-token gate present but \
                 pool.shared_x_norm_f32 unallocated — set ScratchConfig.shared_intermediate"
            );
        }
        let shared_block = build_shared_expert_block(sh, hidden, weights.activation)?;
        let shared_out = state.pool.q_f16;
        let shared_view = flambeau_model_ops::SharedExpertPrefillScratch {
            max_tokens: state.pool.config.max_prefill_tokens,
            x_q8_1: state.pool.norm_q8_1,
            x_q8_1_mmq: state.pool.norm_q8_1_mmq,
            gate_f32: state.pool.gate_f32,
            up_f32: state.pool.up_f32,
            activated_f16: state.pool.gated_f16,
            activated_q8_1: state.pool.gated_q8_1,
            activated_q8_1_mmq: state.pool.gated_q8_1_mmq,
            down_f32: state.pool.down_f32,
            x_norm_f32: state.pool.shared_x_norm_f32,
        };
        shared_block.forward_prefill(&ops, x_norm_f16_ptr, shared_out, n_tokens, shared_view)?;
        // Single batched AR over the whole [n_tokens, hidden] partial,
        // not n_tokens scalar ARs. Same cast-up / AR / cast-back
        // pattern as the per-token decode path.
        let shared_all = unsafe { Tensor::<F16>::from_raw(shared_out, n_tokens * hidden) };
        let mut shared_f32 =
            unsafe { Tensor::<F32>::from_raw(state.pool.attn_proj_f32, n_tokens * hidden) };
        flambeau_model_ops::cast_f16_to_f32(&shared_all, &mut shared_f32, n_tokens * hidden, &ops)?;
        hooks.ar_sum_f32(
            shared_f32.ptr,
            n_tokens * hidden,
            state.device,
            state.stream,
        )?;
        let mut shared_synced = unsafe { Tensor::<F16>::from_raw(shared_out, n_tokens * hidden) };
        flambeau_model_ops::cast_f32_to_f16(
            &shared_f32,
            &mut shared_synced,
            n_tokens * hidden,
            &ops,
        )?;
        let shared_view_f16 = unsafe { Tensor::<F16>::from_raw(shared_out, n_tokens * hidden) };
        let delta_in = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden) };
        let mut delta_out = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden) };
        flambeau_model_ops::add_f16(
            &delta_in,
            &shared_view_f16,
            &mut delta_out,
            n_tokens * hidden,
            &ops,
        )?;
    }

    // Optional post-ffn rmsnorm on the batched F16 delta (gemma4 MoE).
    if let Some(post_norm) = weights.post_ffn_norm.as_ref() {
        let delta_in = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden) };
        let mut delta_out = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden) };
        flambeau_model_ops::rmsnorm_f16(
            &delta_in,
            post_norm,
            &mut delta_out,
            n_tokens,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
    }
    let _ = (act_mmq_null, gate_dt, up_dt, down_dt, router_dt);
    Ok(Some(unsafe {
        Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden)
    }))
}

fn build_shared_expert_block(
    sh: &crate::ctx::SharedExpertWeights,
    hidden: usize,
    activation: Activation,
) -> Result<flambeau_model_ops::SharedExpert> {
    let block = flambeau_model_ops::SharedExpert::new(
        sh.gate_inp.as_ref().map(|t| t.ptr),
        flambeau_model_ops::WeightHandle {
            ptr: sh.gate.ptr,
            dtype: sh.gate.dtype,
            dims: [sh.intermediate, hidden],
        },
        flambeau_model_ops::WeightHandle {
            ptr: sh.up.ptr,
            dtype: sh.up.dtype,
            dims: [sh.intermediate, hidden],
        },
        flambeau_model_ops::WeightHandle {
            ptr: sh.down.ptr,
            dtype: sh.down.dtype,
            dims: [hidden, sh.intermediate],
        },
        hidden,
        sh.intermediate,
    )?;
    Ok(block.with_activation(match activation {
        Activation::SwiGLU => BlockActivation::SwiGLU,
        Activation::GeluTanh => BlockActivation::Gelu,
    }))
}

/// Gemma4 MoE forward. Dispatches the single-token cascade for
/// `n_tokens == 1` (decode) and the batched cascade otherwise
/// (prefill). Math: legacy `crates/models/gemma4/src/moe.rs::
/// forward_ffn_moe`:
///
/// ```text
/// router_input  = rmsnorm_f16(input, pre_router_weight)
/// {ids, w}      = MoE::route(router_input); w *= expert_down_scale
/// rmsnorm_f16(input, ffn_norm) → SharedExpert::f32_partial → ar_sum
/// cur_mlp_f32   = rmsnorm_f32(partial_shared, post_ffw_norm_1)
/// rmsnorm_f16(input, pre_ffw_norm_2) → MoeExperts::tp_f32 → ar_sum
/// cur_moe_f32   = rmsnorm_f32(partial_moe, post_ffw_norm_2)
/// cur_combined  = cur_mlp + cur_moe
/// delta_f16     = cast_f32_to_f16(rmsnorm_f32(cur_combined, post_ffn_norm))
/// ```
fn gemma4_moe_cascade_local<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &MoeWeights,
    n_tokens: usize,
) -> Result<Option<Tensor<F16>>> {
    let hidden = state.hidden();
    if n_tokens == 0 {
        bail!("gemma4_moe_cascade: empty n_tokens");
    }
    if n_tokens == 1 {
        gemma4_moe_cascade_one_token(state, hooks, input.ptr, weights, state.pool.delta)?;
    } else {
        gemma4_moe_cascade_batched(state, hooks, input, weights, n_tokens)?;
    }
    Ok(Some(unsafe {
        Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden)
    }))
}

fn gemma4_moe_cascade_batched<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &MoeWeights,
    n_tokens: usize,
) -> Result<()> {
    let hidden = state.hidden();
    let n_experts = weights.n_experts;
    let k_top = weights.experts_per_tok;
    if n_tokens > state.pool.config.max_prefill_tokens {
        bail!(
            "gemma4_moe_cascade_batched: n_tokens {n_tokens} > pool.max_prefill_tokens {}",
            state.pool.config.max_prefill_tokens
        );
    }
    if weights.shared.is_none() {
        bail!("gemma4_moe_cascade_batched: shared MLP is required");
    }
    let pre_router_w = weights.pre_router_weight_f16.as_ref().ok_or_else(|| {
        anyhow::anyhow!("gemma4_moe_cascade_batched: pre_router_weight_f16 missing")
    })?;
    let pre_ffw_norm_2 = weights
        .pre_ffw_norm_2_f16
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gemma4_moe_cascade_batched: pre_ffw_norm_2_f16 missing"))?;
    let post_ffw_norm_1 = weights.post_ffw_norm_1_f32.as_ref().ok_or_else(|| {
        anyhow::anyhow!("gemma4_moe_cascade_batched: post_ffw_norm_1_f32 missing")
    })?;
    let post_ffw_norm_2 = weights.post_ffw_norm_2_f32.as_ref().ok_or_else(|| {
        anyhow::anyhow!("gemma4_moe_cascade_batched: post_ffw_norm_2_f32 missing")
    })?;
    let post_norm_f32 = weights
        .post_ffn_norm_f32
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gemma4_moe_cascade_batched: post_ffn_norm_f32 missing"))?;
    let expert_scale = weights.expert_down_scale_f32.as_ref().ok_or_else(|| {
        anyhow::anyhow!("gemma4_moe_cascade_batched: expert_down_scale_f32 missing")
    })?;
    let prefill_scratch = state
        .pool
        .moe_prefill_scratch
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "gemma4_moe_cascade_batched: pool.moe_prefill_scratch missing — \
                 ScratchConfig.max_prefill_tokens > 1 required"
            )
        })?
        .view();
    let _m_shared = state.pool.config.shared_intermediate;
    let m_routed = state.pool.config.intermediate;
    if state.pool.shared_x_norm_f32.as_usize() == 0 {
        bail!("gemma4_moe_cascade_batched: pool.shared_x_norm_f32 unallocated");
    }

    let ops = state.ops();

    let router_input_ptr = state.pool.q_f16;
    let partial_shared_mlp_ptr = state.pool.down_f32;
    let moe_input_ptr = state.pool.norm;
    let partial_moe_ptr = state.pool.attn_proj_f32;
    let cur_combined_ptr = state.pool.shared_x_norm_f32;

    let input_view = unsafe { Tensor::<F16>::from_raw(input.ptr, n_tokens * hidden) };

    let mut router_input = unsafe { Tensor::<F16>::from_raw(router_input_ptr, n_tokens * hidden) };
    flambeau_model_ops::rmsnorm_f16(
        &input_view,
        pre_router_w,
        &mut router_input,
        n_tokens,
        hidden,
        weights.rms_eps,
        &ops,
    )?;

    let gate_dt = weights.experts_gate[0].dtype;
    let up_dt = weights.experts_up[0].dtype;
    let down_dt = weights.experts_down[0].dtype;
    let router_dt = weights.router.dtype;
    let block = MoeExperts::new(
        flambeau_model_ops::MoeExpertsWeights {
            ffn_gate_inp: WeightHandle {
                ptr: weights.router.ptr,
                dtype: router_dt,
                dims: [n_experts, hidden],
            },
            ffn_gate_exps: WeightHandle {
                ptr: weights.experts_gate[0].ptr,
                dtype: gate_dt,
                dims: [n_experts * m_routed, hidden],
            },
            ffn_up_exps: WeightHandle {
                ptr: weights.experts_up[0].ptr,
                dtype: up_dt,
                dims: [n_experts * m_routed, hidden],
            },
            ffn_down_exps: WeightHandle {
                ptr: weights.experts_down[0].ptr,
                dtype: down_dt,
                dims: [n_experts * hidden, m_routed],
            },
        },
        flambeau_model_ops::MoeExpertsDims {
            hidden,
            intermediate: m_routed,
            n_experts,
            top_k: k_top,
        },
    )?
    .with_router_policy(RouterPolicy::default())
    .with_activation(match weights.activation {
        Activation::SwiGLU => BlockActivation::SwiGLU,
        Activation::GeluTanh => BlockActivation::Gelu,
    });
    block.route_prefill(&ops, router_input_ptr, n_tokens, prefill_scratch)?;
    use flambeau_ops::Ops;
    ops.apply_per_expert_scale_f32(
        prefill_scratch.expert_weights,
        prefill_scratch.expert_ids,
        expert_scale.ptr,
        n_tokens,
        k_top,
    )?;

    let sh_block =
        build_shared_expert_block(weights.shared.as_ref().unwrap(), hidden, weights.activation)?;
    let mut x_norm_shared = unsafe { Tensor::<F16>::from_raw(router_input_ptr, n_tokens * hidden) };
    flambeau_model_ops::rmsnorm_f16(
        &input_view,
        &weights.ffn_norm,
        &mut x_norm_shared,
        n_tokens,
        hidden,
        weights.rms_eps,
        &ops,
    )?;
    let shared_scratch = flambeau_model_ops::SharedExpertPrefillScratch {
        max_tokens: state.pool.config.max_prefill_tokens,
        x_q8_1: state.pool.norm_q8_1,
        x_q8_1_mmq: state.pool.norm_q8_1_mmq,
        gate_f32: state.pool.gate_f32,
        up_f32: state.pool.up_f32,
        activated_f16: state.pool.gated_f16,
        activated_q8_1: state.pool.gated_q8_1,
        activated_q8_1_mmq: state.pool.gated_q8_1_mmq,
        down_f32: state.pool.down_f32,
        x_norm_f32: state.pool.shared_x_norm_f32,
    };
    sh_block.forward_prefill_f32_partial(
        &ops,
        router_input_ptr,
        partial_shared_mlp_ptr,
        n_tokens,
        shared_scratch,
    )?;
    hooks.ar_sum_f32(
        partial_shared_mlp_ptr,
        n_tokens * hidden,
        state.device,
        state.stream,
    )?;
    let post_ffw_norm_1_view = unsafe { Tensor::<F32>::from_raw(post_ffw_norm_1.ptr, hidden) };
    let mut cur_mlp = unsafe { Tensor::<F32>::from_raw(partial_shared_mlp_ptr, n_tokens * hidden) };
    flambeau_model_ops::rmsnorm_f32(
        &unsafe { Tensor::<F32>::from_raw(partial_shared_mlp_ptr, n_tokens * hidden) },
        &post_ffw_norm_1_view,
        &mut cur_mlp,
        n_tokens,
        hidden,
        weights.rms_eps,
        &ops,
    )?;

    let mut moe_input_f16 = unsafe { Tensor::<F16>::from_raw(moe_input_ptr, n_tokens * hidden) };
    flambeau_model_ops::rmsnorm_f16(
        &input_view,
        pre_ffw_norm_2,
        &mut moe_input_f16,
        n_tokens,
        hidden,
        weights.rms_eps,
        &ops,
    )?;
    block.forward_prefill_tp_f32(
        &ops,
        moe_input_ptr,
        partial_moe_ptr,
        n_tokens,
        prefill_scratch,
    )?;
    hooks.ar_sum_f32(
        partial_moe_ptr,
        n_tokens * hidden,
        state.device,
        state.stream,
    )?;
    let post_ffw_norm_2_view = unsafe { Tensor::<F32>::from_raw(post_ffw_norm_2.ptr, hidden) };
    let mut cur_moe = unsafe { Tensor::<F32>::from_raw(partial_moe_ptr, n_tokens * hidden) };
    flambeau_model_ops::rmsnorm_f32(
        &unsafe { Tensor::<F32>::from_raw(partial_moe_ptr, n_tokens * hidden) },
        &post_ffw_norm_2_view,
        &mut cur_moe,
        n_tokens,
        hidden,
        weights.rms_eps,
        &ops,
    )?;

    let cur_mlp_view =
        unsafe { Tensor::<F32>::from_raw(partial_shared_mlp_ptr, n_tokens * hidden) };
    let cur_moe_view = unsafe { Tensor::<F32>::from_raw(partial_moe_ptr, n_tokens * hidden) };
    let mut cur_combined = unsafe { Tensor::<F32>::from_raw(cur_combined_ptr, n_tokens * hidden) };
    flambeau_model_ops::add_f32(
        &cur_mlp_view,
        &cur_moe_view,
        &mut cur_combined,
        n_tokens * hidden,
        &ops,
    )?;

    let mut tmp_inout = cur_combined;
    flambeau_model_ops::rmsnorm_f32(
        &unsafe { Tensor::<F32>::from_raw(cur_combined_ptr, n_tokens * hidden) },
        post_norm_f32,
        &mut tmp_inout,
        n_tokens,
        hidden,
        weights.rms_eps,
        &ops,
    )?;
    let mut delta_all = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden) };
    flambeau_model_ops::cast_f32_to_f16(&tmp_inout, &mut delta_all, n_tokens * hidden, &ops)?;

    let _ = (gate_dt, up_dt, down_dt, router_dt);
    Ok(())
}

fn gemma4_moe_cascade_one_token<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    hooks: &mut H,
    input_ptr: DevicePtr,
    weights: &MoeWeights,
    output_ptr: DevicePtr,
) -> Result<()> {
    let hidden = state.hidden();
    let n_experts = weights.n_experts;
    let k_top = weights.experts_per_tok;
    if n_experts > state.pool.config.max_experts {
        bail!(
            "gemma4_moe_cascade: n_experts {n_experts} > pool.max_experts {}",
            state.pool.config.max_experts
        );
    }
    if k_top == 0 || k_top > n_experts {
        bail!("gemma4_moe_cascade: experts_per_tok {k_top} must be in 1..={n_experts}");
    }
    if k_top > state.pool.config.max_experts_per_tok {
        bail!(
            "gemma4_moe_cascade: experts_per_tok {k_top} > pool.max_experts_per_tok {}",
            state.pool.config.max_experts_per_tok
        );
    }
    if weights.shared.is_none() {
        bail!("gemma4_moe_cascade: shared MLP is required but `MoeWeights::shared` is None");
    }
    let pre_router_w = weights
        .pre_router_weight_f16
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gemma4_moe_cascade: pre_router_weight_f16 must be Some"))?;
    let pre_ffw_norm_2 = weights
        .pre_ffw_norm_2_f16
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gemma4_moe_cascade: pre_ffw_norm_2_f16 must be Some"))?;
    let post_ffw_norm_1 = weights
        .post_ffw_norm_1_f32
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gemma4_moe_cascade: post_ffw_norm_1_f32 must be Some"))?;
    let post_ffw_norm_2 = weights
        .post_ffw_norm_2_f32
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gemma4_moe_cascade: post_ffw_norm_2_f32 must be Some"))?;
    let post_norm_f32 = weights
        .post_ffn_norm_f32
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gemma4_moe_cascade: post_ffn_norm_f32 must be Some"))?;
    let expert_scale = weights
        .expert_down_scale_f32
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gemma4_moe_cascade: expert_down_scale_f32 must be Some"))?;

    let m_shared = state.pool.config.shared_intermediate;
    let m_routed = state.pool.config.intermediate;
    if m_shared == 0 {
        bail!(
            "gemma4_moe_cascade: pool.shared_intermediate must be > 0 (got 0); \
             set ScratchConfig.shared_intermediate to the per-rank dense FFN width"
        );
    }
    if state.pool.shared_x_norm_f32.as_usize() == 0 {
        bail!(
            "gemma4_moe_cascade: pool.shared_x_norm_f32 unallocated — needs hidden×F32 \
             buffer for the cur_combined / tmp F32 cascade"
        );
    }

    let ops = state.ops();

    let router_input_ptr = state.pool.q_f16;
    let partial_shared_mlp_ptr = state.pool.down_f32;
    let moe_input_ptr = state.pool.norm;
    let partial_moe_ptr = state.pool.attn_proj_f32;
    let cur_combined_ptr = state.pool.shared_x_norm_f32;

    let input_view = unsafe { Tensor::<F16>::from_raw(input_ptr, hidden) };

    let mut router_input = unsafe { Tensor::<F16>::from_raw(router_input_ptr, hidden) };
    flambeau_model_ops::rmsnorm_f16(
        &input_view,
        pre_router_w,
        &mut router_input,
        1,
        hidden,
        weights.rms_eps,
        &ops,
    )?;

    let gate_dt = weights.experts_gate[0].dtype;
    let up_dt = weights.experts_up[0].dtype;
    let down_dt = weights.experts_down[0].dtype;
    let router_dt = weights.router.dtype;
    let block = MoeExperts::new(
        flambeau_model_ops::MoeExpertsWeights {
            ffn_gate_inp: WeightHandle {
                ptr: weights.router.ptr,
                dtype: router_dt,
                dims: [n_experts, hidden],
            },
            ffn_gate_exps: WeightHandle {
                ptr: weights.experts_gate[0].ptr,
                dtype: gate_dt,
                dims: [n_experts * m_routed, hidden],
            },
            ffn_up_exps: WeightHandle {
                ptr: weights.experts_up[0].ptr,
                dtype: up_dt,
                dims: [n_experts * m_routed, hidden],
            },
            ffn_down_exps: WeightHandle {
                ptr: weights.experts_down[0].ptr,
                dtype: down_dt,
                dims: [n_experts * hidden, m_routed],
            },
        },
        flambeau_model_ops::MoeExpertsDims {
            hidden,
            intermediate: m_routed,
            n_experts,
            top_k: k_top,
        },
    )?
    .with_router_policy(RouterPolicy::default())
    .with_activation(match weights.activation {
        Activation::SwiGLU => BlockActivation::SwiGLU,
        Activation::GeluTanh => BlockActivation::Gelu,
    });
    let decode_scratch = MoeExpertsDecodeScratch {
        x_q8_1: state.pool.norm_q8_1,
        router_logits: state.pool.router_logits_f32,
        expert_ids: state.pool.moe_expert_ids,
        expert_weights: state.pool.moe_expert_weights,
        gate_out_f32: state.pool.moe_gate_out_f32,
        up_out_f32: state.pool.moe_up_out_f32,
        activated_f16: state.pool.moe_activated_f16,
        activated_q8_1: state.pool.moe_activated_q8_1,
        down_f32: state.pool.moe_down_f32,
        down_f16: state.pool.moe_down_f16,
    };
    block.route_decode(&ops, router_input_ptr, decode_scratch)?;
    use flambeau_ops::Ops;
    ops.apply_per_expert_scale_f32(
        state.pool.moe_expert_weights,
        state.pool.moe_expert_ids,
        expert_scale.ptr,
        1,
        k_top,
    )?;

    let sh_block =
        build_shared_expert_block(weights.shared.as_ref().unwrap(), hidden, weights.activation)?;
    let mut x_norm_shared = unsafe { Tensor::<F16>::from_raw(router_input_ptr, hidden) };
    flambeau_model_ops::rmsnorm_f16(
        &input_view,
        &weights.ffn_norm,
        &mut x_norm_shared,
        1,
        hidden,
        weights.rms_eps,
        &ops,
    )?;
    let shared_scratch = flambeau_model_ops::SharedExpertDecodeScratch {
        x_q8_1: state.pool.norm_q8_1,
        gate_f32: state.pool.gate_f32,
        up_f32: state.pool.up_f32,
        activated_f16: state.pool.gated_f16,
        activated_q8_1: state.pool.gated_q8_1,
        down_f32: state.pool.down_f32,
        x_norm_f32: state.pool.shared_x_norm_f32,
    };
    sh_block.forward_decode_f32_partial(
        &ops,
        router_input_ptr,
        partial_shared_mlp_ptr,
        shared_scratch,
    )?;
    hooks.ar_sum_f32(partial_shared_mlp_ptr, hidden, state.device, state.stream)?;
    let post_ffw_norm_1_view = unsafe { Tensor::<F32>::from_raw(post_ffw_norm_1.ptr, hidden) };
    let mut cur_mlp = unsafe { Tensor::<F32>::from_raw(partial_shared_mlp_ptr, hidden) };
    flambeau_model_ops::rmsnorm_f32(
        &unsafe { Tensor::<F32>::from_raw(partial_shared_mlp_ptr, hidden) },
        &post_ffw_norm_1_view,
        &mut cur_mlp,
        1,
        hidden,
        weights.rms_eps,
        &ops,
    )?;

    let mut moe_input_f16 = unsafe { Tensor::<F16>::from_raw(moe_input_ptr, hidden) };
    flambeau_model_ops::rmsnorm_f16(
        &input_view,
        pre_ffw_norm_2,
        &mut moe_input_f16,
        1,
        hidden,
        weights.rms_eps,
        &ops,
    )?;
    block.forward_decode_tp_f32(&ops, moe_input_ptr, partial_moe_ptr, decode_scratch)?;
    hooks.ar_sum_f32(partial_moe_ptr, hidden, state.device, state.stream)?;
    let post_ffw_norm_2_view = unsafe { Tensor::<F32>::from_raw(post_ffw_norm_2.ptr, hidden) };
    let mut cur_moe_inout = unsafe { Tensor::<F32>::from_raw(partial_moe_ptr, hidden) };
    flambeau_model_ops::rmsnorm_f32(
        &unsafe { Tensor::<F32>::from_raw(partial_moe_ptr, hidden) },
        &post_ffw_norm_2_view,
        &mut cur_moe_inout,
        1,
        hidden,
        weights.rms_eps,
        &ops,
    )?;

    let cur_mlp_view = unsafe { Tensor::<F32>::from_raw(partial_shared_mlp_ptr, hidden) };
    let cur_moe_view = unsafe { Tensor::<F32>::from_raw(partial_moe_ptr, hidden) };
    let mut cur_combined = unsafe { Tensor::<F32>::from_raw(cur_combined_ptr, hidden) };
    flambeau_model_ops::add_f32(
        &cur_mlp_view,
        &cur_moe_view,
        &mut cur_combined,
        hidden,
        &ops,
    )?;

    let mut tmp_inout = cur_combined;
    flambeau_model_ops::rmsnorm_f32(
        &unsafe { Tensor::<F32>::from_raw(cur_combined_ptr, hidden) },
        post_norm_f32,
        &mut tmp_inout,
        1,
        hidden,
        weights.rms_eps,
        &ops,
    )?;
    let mut delta = unsafe { Tensor::<F16>::from_raw(output_ptr, hidden) };
    flambeau_model_ops::cast_f32_to_f16(&tmp_inout, &mut delta, hidden, &ops)?;

    let _ = (gate_dt, up_dt, down_dt, router_dt, input_view);
    Ok(())
}
