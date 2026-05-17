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
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let m = state.pool.config.intermediate;
    let ops = state.ops();

    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, hidden) };
    flambeau_model_ops::rmsnorm_quant_q8_1(
        input,
        &weights.ffn_norm,
        &mut norm_q8_1,
        1,
        hidden,
        weights.rms_eps,
        &ops,
    )?;
    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };

    let mut gate_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.gate_f32, m) };
    weights
        .ffn_gate
        .qmatmul(&norm_q8_1, &act_mmq_null, &mut gate_f32, 1, hidden, m, &ops)?;
    let mut up_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.up_f32, m) };
    weights
        .ffn_up
        .qmatmul(&norm_q8_1, &act_mmq_null, &mut up_f32, 1, hidden, m, &ops)?;

    let mut gated_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.gated_f16, m) };
    match weights.activation {
        Activation::SwiGLU => {
            flambeau_model_ops::swiglu_f32_to_f16(&gate_f32, &up_f32, &mut gated_f16, m, &ops)?;
        }
        Activation::GeluTanh => {
            flambeau_model_ops::gelu_mul_f32_to_f16(&gate_f32, &up_f32, &mut gated_f16, m, &ops)?;
        }
    }

    let mut gated_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.gated_q8_1, m) };
    flambeau_model_ops::quantize_f16_to_q8_1(&gated_f16, &mut gated_q8_1, m, &ops)?;
    let mut down_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.down_f32, hidden) };
    weights
        .ffn_down
        .qmatmul(&gated_q8_1, &act_mmq_null, &mut down_f32, 1, m, hidden, &ops)?;
    hooks.ar_sum_f32(down_f32.ptr, hidden, state.device, state.stream)?;
    let mut delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
    flambeau_model_ops::cast_f32_to_f16(&down_f32, &mut delta, hidden, &ops)?;
    Ok(delta)
}
