//! rmsnorm → quantise → router qmatmul → host top-k + softmax-over-k →
//! `flambeau_blocks::MoeExperts::forward_decode_tp_f32` (indexed
//! batched-expert gate+up / activate / down / weighted-sum) → AR →
//! cast → optional shared expert → delta.
//!
//! Indexed-MoE dispatch collapses 256+ per-expert MMVQ launches per
//! token (Qwen3.6 top_k=8 × 32 layers) into 32 batched
//! `indexed_moe_mmvq_*` launches, closing the legacy/v2 perf gap
//! documented in `certs/perf/d4_batched_decode/v2_vs_legacy_profile.md`.

use anyhow::{bail, Context, Result};
use flambeau_blocks::{
    MoeExperts, MoeExpertsDecodeScratch, RouterPolicy, WeightHandle,
};
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::{Activation, MoeWeights};
use flambeau_blocks::{Activation as BlockActivation};

pub fn moe_ffn_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &MoeWeights,
    n_tokens: usize,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    if n_tokens > 1 {
        return moe_ffn_loop(state, hooks, input, weights, n_tokens);
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
    flambeau_model_ops::quantize_f16_to_q8_1(&x_norm_f16, &mut norm_q8_1, hidden, &ops)?;

    // 2. Router via v2 qmatmul (works for both F16/F32 and quantised
    //    router weights; the block's route_decode only handles F16/F32).
    let mut router_logits =
        unsafe { Tensor::<F32>::from_raw(state.pool.router_logits_f32, n_experts) };
    weights.router.qmatmul(
        &norm_q8_1,
        &act_mmq_null,
        &mut router_logits,
        1,
        hidden,
        n_experts,
        &ops,
    )?;

    // 3. DtoH router logits + host top-k + softmax-over-k.
    let mut logits_host = vec![0.0_f32; n_experts];
    let logits_bytes = n_experts * 4;
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::DeviceToHost,
                DevicePtr(logits_host.as_mut_ptr() as usize),
                router_logits.ptr,
                logits_bytes,
            )
            .context("moe_ffn: router logits DtoH")?;
    }
    flambeau_core::Stream::synchronize(state.stream)?;

    let mut topk: Vec<(usize, f32)> = logits_host.iter().copied().enumerate().collect();
    topk.select_nth_unstable_by(k_top - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    topk.truncate(k_top);
    topk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_logit = topk[0].1;
    let mut exps: Vec<f32> = topk.iter().map(|(_, v)| (v - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in &mut exps {
        *e /= sum;
    }
    let host_ids: Vec<i32> = topk.iter().map(|(e, _)| *e as i32).collect();

    // 4. Upload expert_ids + expert_weights to device scratch.
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                state.pool.moe_expert_ids,
                DevicePtr(host_ids.as_ptr() as usize),
                k_top * 4,
            )
            .context("moe_ffn: expert_ids HtoD")?;
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                state.pool.moe_expert_weights,
                DevicePtr(exps.as_ptr() as usize),
                k_top * 4,
            )
            .context("moe_ffn: expert_weights HtoD")?;
    }
    flambeau_core::Stream::synchronize(state.stream)?;

    // 5. Indexed-MoE expert dispatch via flambeau_blocks::MoeExperts.
    //    `experts_gate[0].ptr` is the base of the stacked packed-expert
    //    tensor (loader::upload_moe_experts_stacked guarantees this);
    //    block treats it as a 2D `[n_experts * inter, hidden]` view.
    let gate_dt = weights.experts_gate[0].dtype;
    let up_dt = weights.experts_up[0].dtype;
    let down_dt = weights.experts_down[0].dtype;
    let router_dt = weights.router.dtype;
    let block = MoeExperts::new(
        WeightHandle { ptr: weights.router.ptr, dtype: router_dt, dims: [n_experts, hidden] },
        WeightHandle { ptr: weights.experts_gate[0].ptr, dtype: gate_dt, dims: [n_experts * m, hidden] },
        WeightHandle { ptr: weights.experts_up[0].ptr, dtype: up_dt, dims: [n_experts * m, hidden] },
        WeightHandle { ptr: weights.experts_down[0].ptr, dtype: down_dt, dims: [n_experts * hidden, m] },
        hidden,
        m,
        n_experts,
        k_top,
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
        let shared_block = build_shared_expert_block(sh, hidden)?;
        let scratch_view = flambeau_blocks::SharedExpertDecodeScratch {
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
        let shared = unsafe { Tensor::<F16>::from_raw(shared_out, hidden) };
        let delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
        let mut delta_out = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
        flambeau_model_ops::add_f16(&delta, &shared, &mut delta_out, hidden, &ops)?;
    }

    let _ = (act_mmq_null, gate_dt, up_dt, down_dt, router_dt);
    Ok(unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) })
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
fn moe_ffn_loop<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &MoeWeights,
    n_tokens: usize,
) -> Result<Tensor<F16>> {
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
    let input_view =
        unsafe { Tensor::<F16>::from_raw(input.ptr, n_tokens * hidden) };
    let mut x_norm_f16 =
        unsafe { Tensor::<F16>::from_raw(x_norm_f16_ptr, n_tokens * hidden) };
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
    let mut x_q8_1_n = unsafe {
        Tensor::<Q8_1>::from_raw(prefill_scratch.x_q8_1, n_tokens * hidden)
    };
    flambeau_model_ops::quantize_f16_to_q8_1(
        &x_norm_f16,
        &mut x_q8_1_n,
        n_tokens * hidden,
        &ops,
    )?;

    // 2. Router qmatmul at n_tokens=N → router_logits [N, n_experts] F32.
    let mut router_logits = unsafe {
        Tensor::<F32>::from_raw(prefill_scratch.router_logits, n_tokens * n_experts)
    };
    weights.router.qmatmul(
        &x_q8_1_n,
        &act_mmq_null,
        &mut router_logits,
        n_tokens,
        hidden,
        n_experts,
        &ops,
    )?;

    // 3. DtoH all N×n_experts logits in one transfer; host topk per row.
    let mut logits_host = vec![0.0_f32; n_tokens * n_experts];
    let logits_bytes = n_tokens * n_experts * 4;
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::DeviceToHost,
                DevicePtr(logits_host.as_mut_ptr() as usize),
                router_logits.ptr,
                logits_bytes,
            )
            .context("moe_ffn prefill: router logits DtoH")?;
    }
    flambeau_core::Stream::synchronize(state.stream)?;
    let mut host_ids: Vec<i32> = Vec::with_capacity(n_tokens * k_top);
    let mut host_weights: Vec<f32> = Vec::with_capacity(n_tokens * k_top);
    for t in 0..n_tokens {
        let row = &logits_host[t * n_experts..(t + 1) * n_experts];
        let mut topk: Vec<(usize, f32)> =
            row.iter().copied().enumerate().collect();
        topk.select_nth_unstable_by(k_top - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        topk.truncate(k_top);
        topk.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let max_logit = topk[0].1;
        let mut exps: Vec<f32> = topk.iter().map(|(_, v)| (v - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for e in &mut exps {
            *e /= sum;
        }
        for (slot, &(e, _)) in topk.iter().enumerate() {
            host_ids.push(e as i32);
            host_weights.push(exps[slot]);
        }
    }

    // 4. HtoD upload [N, top_k] ids + weights into the prefill scratch.
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                prefill_scratch.expert_ids,
                DevicePtr(host_ids.as_ptr() as usize),
                n_tokens * k_top * 4,
            )
            .context("moe_ffn prefill: expert_ids HtoD")?;
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                prefill_scratch.expert_weights,
                DevicePtr(host_weights.as_ptr() as usize),
                n_tokens * k_top * 4,
            )
            .context("moe_ffn prefill: expert_weights HtoD")?;
    }
    flambeau_core::Stream::synchronize(state.stream)?;

    // 5. Batched indexed-MoE forward + F32 partial AR.
    let gate_dt = weights.experts_gate[0].dtype;
    let up_dt = weights.experts_up[0].dtype;
    let down_dt = weights.experts_down[0].dtype;
    let router_dt = weights.router.dtype;
    let block = MoeExperts::new(
        WeightHandle { ptr: weights.router.ptr, dtype: router_dt, dims: [n_experts, hidden] },
        WeightHandle {
            ptr: weights.experts_gate[0].ptr,
            dtype: gate_dt,
            dims: [n_experts * m, hidden],
        },
        WeightHandle {
            ptr: weights.experts_up[0].ptr,
            dtype: up_dt,
            dims: [n_experts * m, hidden],
        },
        WeightHandle {
            ptr: weights.experts_down[0].ptr,
            dtype: down_dt,
            dims: [n_experts * hidden, m],
        },
        hidden,
        m,
        n_experts,
        k_top,
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
    hooks.ar_sum_f32(state.pool.down_f32, n_tokens * hidden, state.device, state.stream)?;
    let down_full = unsafe { Tensor::<F32>::from_raw(state.pool.down_f32, n_tokens * hidden) };
    let mut delta_full =
        unsafe { Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden) };
    flambeau_model_ops::cast_f32_to_f16(
        &down_full,
        &mut delta_full,
        n_tokens * hidden,
        &ops,
    )?;

    // 6. Shared expert (per-token; block has no prefill variant). Adds
    // each token's shared_out into delta[t].
    if let Some(sh) = weights.shared.as_ref() {
        if state.pool.shared_x_norm_f32.as_usize() == 0 && sh.gate_inp.is_some() {
            bail!(
                "moe_ffn: shared expert with per-token gate present but \
                 pool.shared_x_norm_f32 unallocated — set ScratchConfig.shared_intermediate"
            );
        }
        let shared_block = build_shared_expert_block(sh, hidden)?;
        let row_bytes = hidden * 2;
        for t in 0..n_tokens {
            let x_row = x_norm_f16_ptr.offset_bytes(t * row_bytes);
            let shared_view = flambeau_blocks::SharedExpertDecodeScratch {
                x_q8_1: state.pool.norm_q8_1,
                gate_f32: state.pool.gate_f32,
                up_f32: state.pool.up_f32,
                activated_f16: state.pool.gated_f16,
                activated_q8_1: state.pool.gated_q8_1,
                down_f32: state.pool.down_f32,
                x_norm_f32: state.pool.shared_x_norm_f32,
            };
            let shared_out = state.pool.q_f16;
            shared_block.forward_decode(&ops, x_row, shared_out, shared_view)?;
            let shared_t = unsafe { Tensor::<F16>::from_raw(shared_out, hidden) };
            let delta_row = state.pool.delta.offset_bytes(t * row_bytes);
            let delta_t_in = unsafe { Tensor::<F16>::from_raw(delta_row, hidden) };
            let mut delta_t_out = unsafe { Tensor::<F16>::from_raw(delta_row, hidden) };
            flambeau_model_ops::add_f16(&delta_t_in, &shared_t, &mut delta_t_out, hidden, &ops)?;
        }
    }

    let _ = (act_mmq_null, gate_dt, up_dt, down_dt, router_dt);
    Ok(unsafe { Tensor::<F16>::from_raw(state.pool.delta, n_tokens * hidden) })
}

fn build_shared_expert_block(
    sh: &crate::ctx::SharedExpertWeights,
    hidden: usize,
) -> Result<flambeau_blocks::SharedExpert> {
    flambeau_blocks::SharedExpert::new(
        sh.gate_inp.as_ref().map(|t| t.ptr),
        flambeau_blocks::WeightHandle {
            ptr: sh.gate.ptr,
            dtype: sh.gate.dtype,
            dims: [sh.intermediate, hidden],
        },
        flambeau_blocks::WeightHandle {
            ptr: sh.up.ptr,
            dtype: sh.up.dtype,
            dims: [sh.intermediate, hidden],
        },
        flambeau_blocks::WeightHandle {
            ptr: sh.down.ptr,
            dtype: sh.down.dtype,
            dims: [hidden, sh.intermediate],
        },
        hidden,
        sh.intermediate,
    )
}
