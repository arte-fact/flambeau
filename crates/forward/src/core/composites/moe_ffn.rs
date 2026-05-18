//! rmsnorm-quant → router qmatmul → host top-k + softmax-over-k → for
//! each selected expert: gate / up / activate / quantise / down (with
//! TP AR) / cast / scale-and-accumulate → delta. Routing follows the
//! TopkRenorm convention (top-k by raw logit, then softmax over the
//! kept k).

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::{Activation, MoeWeights};

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

    // 1. rmsnorm input. Two paths:
    //    - shared expert: rmsnorm_f16 → `pool.norm` (F16 reused by
    //      shared expert later), then quantize_f16_to_q8_1 → `norm_q8_1`.
    //    - else: fused `rmsnorm_quant_q8_1` direct to Q8_1.
    // The unfused split adds one HBM round-trip when a shared expert
    // is present; the routed loop and the shared-expert block both
    // consume `norm_q8_1` so we only normalise once.
    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, hidden) };
    let x_norm_f16_ptr = state.pool.norm;
    if weights.shared.is_some() {
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
    } else {
        flambeau_model_ops::rmsnorm_quant_q8_1(
            input,
            &weights.ffn_norm,
            &mut norm_q8_1,
            1,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
    }

    // 2. Router logits: [n_experts] F32.
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
    let bytes = n_experts * 4;
    // SAFETY: logits_host has n_experts*4 bytes; router_logits owns the same.
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::DeviceToHost,
                DevicePtr(logits_host.as_mut_ptr() as usize),
                router_logits.ptr,
                bytes,
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

    // 4. Zero accumulator on device.
    let accum_bytes = hidden * 2;
    let zero_host = vec![half::f16::ZERO; hidden];
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                state.pool.moe_accum_f16,
                DevicePtr(zero_host.as_ptr() as usize),
                accum_bytes,
            )
            .context("moe_ffn: zero accumulator HtoD")?;
    }
    flambeau_core::Stream::synchronize(state.stream)?;

    // 5. Per-selected-expert FFN, scaled-and-accumulated.
    for (slot, &(expert_idx, _)) in topk.iter().enumerate() {
        let weight = exps[slot];

        let mut gate_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.gate_f32, m) };
        weights.experts_gate[expert_idx].qmatmul(
            &norm_q8_1,
            &act_mmq_null,
            &mut gate_f32,
            1,
            hidden,
            m,
            &ops,
        )?;
        let mut up_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.up_f32, m) };
        weights.experts_up[expert_idx].qmatmul(
            &norm_q8_1,
            &act_mmq_null,
            &mut up_f32,
            1,
            hidden,
            m,
            &ops,
        )?;

        let mut gated_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.gated_f16, m) };
        match weights.activation {
            Activation::SwiGLU => {
                flambeau_model_ops::swiglu_f32_to_f16(&gate_f32, &up_f32, &mut gated_f16, m, &ops)?;
            }
            Activation::GeluTanh => {
                flambeau_model_ops::gelu_mul_f32_to_f16(
                    &gate_f32, &up_f32, &mut gated_f16, m, &ops,
                )?;
            }
        }

        let mut gated_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.gated_q8_1, m) };
        flambeau_model_ops::quantize_f16_to_q8_1(&gated_f16, &mut gated_q8_1, m, &ops)?;
        let mut down_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.down_f32, hidden) };
        weights.experts_down[expert_idx].qmatmul(
            &gated_q8_1,
            &act_mmq_null,
            &mut down_f32,
            1,
            m,
            hidden,
            &ops,
        )?;
        // Row-parallel down — AR partials across ranks (no-op on SD/PP).
        hooks.ar_sum_f32(down_f32.ptr, hidden, state.device, state.stream)?;

        // Scale down_f32 by `weight`, cast to F16, accumulate into moe_accum.
        let mut down_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
        flambeau_model_ops::cast_f32_to_f16(&down_f32, &mut down_f16, hidden, &ops)?;
        let mut scaled = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, hidden) };
        flambeau_model_ops::scale_f16(&down_f16, &mut scaled, hidden, weight, &ops)?;
        let accum = unsafe { Tensor::<F16>::from_raw(state.pool.moe_accum_f16, hidden) };
        let mut accum_out = unsafe { Tensor::<F16>::from_raw(state.pool.moe_accum_f16, hidden) };
        flambeau_model_ops::add_f16(&accum, &scaled, &mut accum_out, hidden, &ops)?;
    }

    // 6. Shared expert (qwen3-moe family): one always-on dense FFN
    //    with optional per-token sigmoid-gated output. Adds into the
    //    same accumulator as the routed experts; final delta = sum.
    if let Some(sh) = weights.shared.as_ref() {
        if state.pool.shared_x_norm_f32.as_usize() == 0 && sh.gate_inp.is_some() {
            bail!(
                "moe_ffn: shared expert with per-token gate present but \
                 pool.shared_x_norm_f32 unallocated — set \
                 ScratchConfig.shared_intermediate"
            );
        }
        let block = build_shared_expert_block(sh, hidden)?;
        // Reuse routed-loop scratch slots that are free again post-loop.
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
        // is ≥ hidden on every supported MoE arch (attn n_heads ×
        // head_dim ≥ hidden); using delta as the temp is unsafe because
        // we'd then read+write it for the final accumulate.
        let shared_out = state.pool.q_f16;
        block.forward_decode(&ops, x_norm_f16_ptr, shared_out, scratch_view)?;
        let shared = unsafe { Tensor::<F16>::from_raw(shared_out, hidden) };
        let accum = unsafe { Tensor::<F16>::from_raw(state.pool.moe_accum_f16, hidden) };
        let mut accum_out =
            unsafe { Tensor::<F16>::from_raw(state.pool.moe_accum_f16, hidden) };
        flambeau_model_ops::add_f16(&accum, &shared, &mut accum_out, hidden, &ops)?;
    }

    // 7. delta = accumulated F16.
    let bytes = hidden * 2;
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::DeviceToDevice,
                state.pool.delta,
                state.pool.moe_accum_f16,
                bytes,
            )
            .context("moe_ffn: accumulator → delta")?;
    }
    Ok(unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) })
}

fn moe_ffn_loop<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &MoeWeights,
    n_tokens: usize,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let row_bytes = hidden * 2;
    let delta_ptr = state.pool.delta;
    // Iterate in reverse so token 0's result lands naturally at delta[0..hidden]
    // (the inner call always writes there); every other iteration's result
    // gets copied to delta[i*hidden..(i+1)*hidden] before the next iteration
    // overwrites delta[0..hidden].
    for i in (0..n_tokens).rev() {
        let in_i = unsafe {
            Tensor::<F16>::from_raw(input.ptr.offset_bytes(i * row_bytes), hidden)
        };
        let _ = moe_ffn_local(state, hooks, &in_i, weights, 1)?;
        if i > 0 {
            let dst = delta_ptr.offset_bytes(i * row_bytes);
            // SAFETY: delta is sized max_prefill_tokens * hidden * F16; src and
            // dst rows are non-overlapping for i > 0 on the same stream.
            unsafe {
                state
                    .device
                    .memcpy_async(
                        state.stream,
                        CopyDirection::DeviceToDevice,
                        dst,
                        delta_ptr,
                        row_bytes,
                    )
                    .context("moe_ffn_loop: per-token delta DtoD fanout")?;
            }
        }
    }
    Ok(unsafe { Tensor::<F16>::from_raw(delta_ptr, n_tokens * hidden) })
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
