//! rmsnorm-quant → LM-head matmul → DtoH into `state.logits_host`.

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::LmHeadWeights;

pub fn output_head_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    _hooks: &mut H,
    input: &Tensor<F16>,
    lm_head: &LmHeadWeights,
) -> Result<()> {
    let hidden = state.hidden();
    if hidden != lm_head.hidden {
        bail!(
            "output_head: ctx hidden {hidden} != lm_head.hidden {}",
            lm_head.hidden
        );
    }
    let vocab = lm_head.vocab_size;
    let ops = state.ops();

    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, hidden) };
    flambeau_model_ops::rmsnorm_quant_q8_1(
        input,
        &lm_head.output_norm,
        &mut norm_q8_1,
        1,
        hidden,
        lm_head.rms_eps,
        &ops,
    )?;

    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
    let mut logits_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.logits_f32_dev, vocab) };
    lm_head
        .lm_head
        .qmatmul(&norm_q8_1, &act_mmq_null, &mut logits_f32, 1, hidden, vocab, &ops)?;

    if let Some(cap) = lm_head.final_logit_softcap {
        let mut logits_inplace = unsafe {
            Tensor::<F32>::from_raw(state.pool.logits_f32_dev, vocab)
        };
        flambeau_model_ops::apply_softcap_f32(
            &logits_f32,
            &mut logits_inplace,
            vocab,
            cap,
            &ops,
        )?;
    }

    if state.logits_host.len() != vocab {
        state.logits_host = vec![0.0_f32; vocab];
    }
    let bytes = vocab * 4;
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::DeviceToHost,
                DevicePtr(state.logits_host.as_mut_ptr() as usize),
                state.pool.logits_f32_dev,
                bytes,
            )
            .context("output_head: logits DtoH")?;
    }
    flambeau_core::Stream::synchronize(state.stream)?;
    Ok(())
}
