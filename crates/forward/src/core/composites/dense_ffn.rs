//! rmsnorm-quant → gate/up → activate → quantise → down (with TP AR) → cast.

use anyhow::Result;
use flambeau_core::DevicePtr;
use flambeau_model_ops::{Tensor, F16, F32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::{Activation, FfnWeights};

pub fn dense_ffn_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &FfnWeights,
    n_tokens: usize,
    next_norm: Option<&Tensor<F16>>,
) -> Result<Option<Tensor<F16>>> {
    let input_pre_normed = state.pool.input_pre_normed;
    state.pool.input_pre_normed = false;
    let hidden = state.hidden();
    let m = state.pool.config.intermediate;
    let ops = state.ops();
    let n = n_tokens;

    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, n * hidden) };
    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
    let norm_mmq_t;
    let act_norm_mmq: &Tensor<Q8_1> = if input_pre_normed && n == 1 {
        let norm_view = unsafe { Tensor::<F16>::from_raw(state.pool.norm, n * hidden) };
        flambeau_model_ops::quantize_f16_to_q8_1(&norm_view, &mut norm_q8_1, n * hidden, &ops)?;
        &act_mmq_null
    } else if n > 1 {
        // Unfused at N>1 so we can also produce the MMQ-layout activation.
        let mut norm_f16 =
            unsafe { Tensor::<F16>::from_raw(state.pool.norm, n * hidden) };
        flambeau_model_ops::rmsnorm_f16(
            input,
            &weights.ffn_norm,
            &mut norm_f16,
            n,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
        flambeau_model_ops::quantize_f16_to_q8_1(&norm_f16, &mut norm_q8_1, n * hidden, &ops)?;
        let mut norm_mmq =
            unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1_mmq, n * hidden) };
        flambeau_model_ops::quantize_f16_to_q8_1_mmq(&norm_f16, &mut norm_mmq, hidden, n, &ops)?;
        norm_mmq_t = norm_mmq;
        &norm_mmq_t
    } else {
        flambeau_model_ops::rmsnorm_quant_q8_1(
            input,
            &weights.ffn_norm,
            &mut norm_q8_1,
            n,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
        &act_mmq_null
    };

    let mut gate_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.gate_f32, n * m) };
    weights
        .ffn_gate
        .qmatmul(&norm_q8_1, act_norm_mmq, &mut gate_f32, n, hidden, m, &ops)?;
    let mut up_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.up_f32, n * m) };
    weights
        .ffn_up
        .qmatmul(&norm_q8_1, act_norm_mmq, &mut up_f32, n, hidden, m, &ops)?;

    let mut gated_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.gated_f16, n * m) };
    match weights.activation {
        Activation::SwiGLU => {
            flambeau_model_ops::swiglu_f32_to_f16(&gate_f32, &up_f32, &mut gated_f16, n * m, &ops)?;
        }
        Activation::GeluTanh => {
            flambeau_model_ops::gelu_mul_f32_to_f16(&gate_f32, &up_f32, &mut gated_f16, n * m, &ops)?;
        }
    }

    let mut gated_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.gated_q8_1, n * m) };
    flambeau_model_ops::quantize_f16_to_q8_1(&gated_f16, &mut gated_q8_1, n * m, &ops)?;
    let gated_mmq_t;
    let act_gated_mmq: &Tensor<Q8_1> = if n > 1 {
        let mut gated_mmq =
            unsafe { Tensor::<Q8_1>::from_raw(state.pool.gated_q8_1_mmq, n * m) };
        flambeau_model_ops::quantize_f16_to_q8_1_mmq(&gated_f16, &mut gated_mmq, m, n, &ops)?;
        gated_mmq_t = gated_mmq;
        &gated_mmq_t
    } else {
        &act_mmq_null
    };
    // Decode F16-direct fast path: at n=1 under AR-fold the down
    // projection writes straight to `pool.delta` via mmvq's saturating
    // F16-cast — skips one `cast_f32_to_f16` launch per layer per token.
    let f16_fast = n == 1
        && weights.post_ffn_norm.is_none()
        && weights.ffn_down.supports_decode_to_f16()
        && (hooks.supports_ar_residual_rmsnorm_f16()
            || hooks.supports_ar_residual_f16());
    let mut down_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.down_f32, n * hidden) };
    if !f16_fast {
        weights
            .ffn_down
            .qmatmul(&gated_q8_1, act_gated_mmq, &mut down_f32, n, m, hidden, &ops)?;
    }
    // Fused AR + residual fast path when post_ffn_norm is None.
    if n == 1
        && weights.post_ffn_norm.is_none()
        && next_norm.is_some()
        && hooks.supports_ar_residual_rmsnorm_f16()
    {
        let next_w = next_norm.unwrap();
        let mut partial_f16 =
            unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        if f16_fast {
            weights.ffn_down.qmatmul_decode_to_f16(
                &gated_q8_1,
                &mut partial_f16,
                m,
                hidden,
                &ops,
            )?;
        } else {
            flambeau_model_ops::cast_f32_to_f16(&down_f32, &mut partial_f16, n * hidden, &ops)?;
        }
        hooks.ar_residual_rmsnorm_f16(
            input.ptr,
            partial_f16.ptr,
            next_w.ptr,
            state.pool.norm,
            n * hidden,
            weights.rms_eps,
            state.device,
            state.stream,
        )?;
        state.pool.input_pre_normed = true;
        return Ok(None);
    }
    if hooks.supports_ar_residual_f16() && weights.post_ffn_norm.is_none() {
        let mut partial_f16 =
            unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        if f16_fast {
            weights.ffn_down.qmatmul_decode_to_f16(
                &gated_q8_1,
                &mut partial_f16,
                m,
                hidden,
                &ops,
            )?;
        } else {
            flambeau_model_ops::cast_f32_to_f16(&down_f32, &mut partial_f16, n * hidden, &ops)?;
        }
        hooks.ar_residual_f16(
            input.ptr,
            partial_f16.ptr,
            n * hidden,
            state.device,
            state.stream,
        )?;
        return Ok(None);
    }
    hooks.ar_sum_f32(down_f32.ptr, n * hidden, state.device, state.stream)?;
    let delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
    if let Some(post_norm) = weights.post_ffn_norm.as_ref() {
        // Fused F32→F16 + rmsnorm: skip the cast_f32_to_f16 launch.
        let mut delta_out = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        flambeau_model_ops::rmsnorm_f32_to_f16(
            &down_f32,
            post_norm,
            &mut delta_out,
            n,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
    } else {
        let mut delta_mut = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        flambeau_model_ops::cast_f32_to_f16(&down_f32, &mut delta_mut, n * hidden, &ops)?;
    }
    Ok(Some(delta))
}
