//! rmsnorm-quant → gate/up → activate → quantise → down (with TP AR) → cast.

use anyhow::Result;
use flambeau_core::DevicePtr;
use flambeau_ops::{ActLayout, ActLayouts};
use flambeau_model_ops::{Tensor, F16, F32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::{Activation, FfnWeights};

pub fn dense_ffn_local<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
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

    // The activation layouts the gate/up projections read at this shape.
    let gateup_shape = flambeau_ops::MatmulShape { m: n, k: hidden, n: m };
    let norm_layouts: ActLayouts = [&weights.ffn_gate, &weights.ffn_up]
        .into_iter()
        .map(|w| w.act_layout(gateup_shape, &ops))
        .collect();

    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, n * hidden) };
    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
    let norm_mmq_t;
    let act_norm_mmq: &Tensor<Q8_1> = if input_pre_normed && n == 1 {
        let norm_view = unsafe { Tensor::<F16>::from_raw(state.pool.norm, n * hidden) };
        flambeau_model_ops::quantize_f16_to_q8_1(&norm_view, &mut norm_q8_1, n * hidden, &ops)?;
        &act_mmq_null
    } else if n > 1 {
        // Unfused at N>1 so we can also produce the MMQ-layout activation.
        let mut norm_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.norm, n * hidden) };
        flambeau_model_ops::rmsnorm_f16(
            input,
            &weights.ffn_norm,
            &mut norm_f16,
            n,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
        let mut norm_mmq =
            unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1_mmq, n * hidden) };
        flambeau_model_ops::quantize_f16_for_layouts(
            &norm_f16,
            norm_layouts,
            &mut norm_q8_1,
            &mut norm_mmq,
            hidden,
            n,
            &ops,
        )?;
        if norm_layouts.reads(ActLayout::Mmq) {
            norm_mmq_t = norm_mmq;
            &norm_mmq_t
        } else {
            &act_mmq_null
        }
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
    let mut up_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.up_f32, n * m) };
    let fuse_q4_0_decode = n == 1
        && weights.ffn_gate.dtype == flambeau_core::op::QDtype::Q4_0
        && weights.ffn_up.dtype == flambeau_core::op::QDtype::Q4_0;
    let fuse_q8_0_decode = n == 1
        && weights.ffn_gate.dtype == flambeau_core::op::QDtype::Q8_0
        && weights.ffn_up.dtype == flambeau_core::op::QDtype::Q8_0;
    if fuse_q4_0_decode {
        let gate_w = unsafe {
            Tensor::<flambeau_model_ops::Q4_0>::from_raw(
                weights.ffn_gate.ptr,
                weights.ffn_gate.n_elems,
            )
        };
        let up_w = unsafe {
            Tensor::<flambeau_model_ops::Q4_0>::from_raw(
                weights.ffn_up.ptr,
                weights.ffn_up.n_elems,
            )
        };
        flambeau_model_ops::mmvq_q4_0_gate_up_t128_decode(
            &gate_w,
            &up_w,
            &norm_q8_1,
            &mut gate_f32,
            &mut up_f32,
            flambeau_ops::MmvqShape { n_rows: m, k: hidden },
            &ops,
        )?;
    } else if fuse_q8_0_decode {
        let gate_w = unsafe {
            Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                weights.ffn_gate.ptr,
                weights.ffn_gate.n_elems,
            )
        };
        let up_w = unsafe {
            Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                weights.ffn_up.ptr,
                weights.ffn_up.n_elems,
            )
        };
        flambeau_model_ops::mmvq_q8_0_gate_up_t128_decode(
            &gate_w,
            &up_w,
            &norm_q8_1,
            &mut gate_f32,
            &mut up_f32,
            flambeau_ops::MmvqShape { n_rows: m, k: hidden },
            &ops,
        )?;
    } else {
        weights.ffn_gate.qmatmul(
            &norm_q8_1,
            act_norm_mmq,
            &mut gate_f32,
            flambeau_ops::MatmulShape {
                m: n,
                k: hidden,
                n: m,
            },
            &ops,
        )?;
        weights.ffn_up.qmatmul(
            &norm_q8_1,
            act_norm_mmq,
            &mut up_f32,
            flambeau_ops::MatmulShape {
                m: n,
                k: hidden,
                n: m,
            },
            &ops,
        )?;
    }

    // The activation layout the down projection reads. The F16 `gated`
    // intermediate is only needed when we don't fuse straight to DS4.
    let down_shape = flambeau_ops::MatmulShape { m: n, k: m, n: hidden };
    let down_layout = weights.ffn_down.act_layout(down_shape, &ops);
    let use_fused_gelu_mmq =
        down_layout == ActLayout::Mmq && matches!(weights.activation, Activation::GeluTanh);

    let mut gated_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.gated_q8_1, n * m) };
    let gated_mmq_t;
    let act_gated_mmq: &Tensor<Q8_1>;
    if use_fused_gelu_mmq {
        // Gemma-4 prefill: GELU(gate)*up → DS4 MMQ activation in one launch,
        // skipping the F16 gated write + re-read.
        let mut gated_mmq = unsafe { Tensor::<Q8_1>::from_raw(state.pool.gated_q8_1_mmq, n * m) };
        flambeau_model_ops::gelu_mul_f32_to_q8_1_mmq(
            &gate_f32, &up_f32, &mut gated_mmq, m, n, &ops,
        )?;
        gated_mmq_t = gated_mmq;
        act_gated_mmq = &gated_mmq_t;
    } else {
        let mut gated_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.gated_f16, n * m) };
        match weights.activation {
            Activation::SwiGLU => {
                flambeau_model_ops::swiglu_f32_to_f16(
                    &gate_f32, &up_f32, &mut gated_f16, n * m, &ops,
                )?;
            }
            Activation::GeluTanh => {
                flambeau_model_ops::gelu_mul_f32_to_f16(
                    &gate_f32, &up_f32, &mut gated_f16, n * m, &ops,
                )?;
            }
        }
        let mut gated_mmq = unsafe { Tensor::<Q8_1>::from_raw(state.pool.gated_q8_1_mmq, n * m) };
        flambeau_model_ops::quantize_f16_for_layouts(
            &gated_f16,
            down_layout.into(),
            &mut gated_q8_1,
            &mut gated_mmq,
            m,
            n,
            &ops,
        )?;
        if down_layout == ActLayout::Mmq {
            gated_mmq_t = gated_mmq;
            act_gated_mmq = &gated_mmq_t;
        } else {
            act_gated_mmq = &act_mmq_null;
        }
    }
    // Decode F16-direct fast path: at n=1 under AR-fold the down
    // projection writes straight to `pool.delta` via mmvq's saturating
    // F16-cast — skips one `cast_f32_to_f16` launch per layer per token.
    let f16_fast = n == 1
        && weights.post_ffn_norm.is_none()
        && weights.ffn_down.supports_decode_to_f16(&ops)
        && (hooks.supports_ar_residual_rmsnorm_f16() || hooks.supports_ar_residual_f16());
    let mut down_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.down_f32, n * hidden) };
    if !f16_fast {
        weights.ffn_down.qmatmul(
            &gated_q8_1,
            act_gated_mmq,
            &mut down_f32,
            flambeau_ops::MatmulShape {
                m: n,
                k: m,
                n: hidden,
            },
            &ops,
        )?;
    }
    // Fused AR + residual fast path when post_ffn_norm is None.
    if let Some(next_w) = next_norm
        .filter(|_| n == 1
            && weights.post_ffn_norm.is_none()
            && hooks.supports_ar_residual_rmsnorm_f16())
    {
        let mut partial_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
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
            crate::core::ArResidualRmsNormHookBuffers {
                residual_inout: input.ptr,
                partial_f16: partial_f16.ptr,
                rms_weight: next_w.ptr,
                out_norm: state.pool.norm,
            },
            n * hidden,
            weights.rms_eps,
            state.device,
            state.stream,
        )?;
        state.pool.input_pre_normed = true;
        return Ok(None);
    }
    if hooks.supports_ar_residual_f16() && weights.post_ffn_norm.is_none() {
        let mut partial_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
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
    if let Some(post_norm) = weights.post_ffn_norm.as_ref() {
        // Fused F32→F16 rmsnorm + residual add. See standard_attn's
        // matching block for the slot-advance + flag pattern.
        let resid_in_ptr = input.ptr;
        let new_resid_ptr = state.pool.next_residual_slot();
        use flambeau_ops::Ops;
        ops.rmsnorm_f32_to_f16_add_residual(
            flambeau_ops::NormResidualBuffers {
                input: down_f32.ptr,
                weight: post_norm.ptr,
                resid_in: resid_in_ptr,
                resid_out: new_resid_ptr,
            },
            flambeau_ops::NormShape { m: n, k: hidden },
            weights.rms_eps,
        )?;
        state.pool.fused_residual_already_done = true;
        let new_resid = unsafe { Tensor::<F16>::from_raw(new_resid_ptr, n * hidden) };
        Ok(Some(new_resid))
    } else {
        let mut delta_mut = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        flambeau_model_ops::cast_f32_to_f16(&down_f32, &mut delta_mut, n * hidden, &ops)?;
        let delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        Ok(Some(delta))
    }
}
