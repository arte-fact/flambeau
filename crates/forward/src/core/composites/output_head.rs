//! rmsnorm + LM-head matmul + DtoH into `state.logits_host`. Decode /
//! prefill emits 1 row (last token); batched-decode (distinct slot
//! ids) emits N rows in row-major `[N, vocab]` order.

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
    slot_ids: &[usize],
) -> Result<()> {
    let hidden = state.hidden();
    if hidden != lm_head.hidden {
        bail!(
            "output_head: ctx hidden {hidden} != lm_head.hidden {}",
            lm_head.hidden
        );
    }
    let n_tokens = slot_ids.len();
    if n_tokens == 0 {
        bail!("output_head: empty slot_ids");
    }
    let vocab = lm_head.vocab_size;
    let ops = state.ops();
    let emit_all = n_tokens > 1 && slot_ids.iter().any(|&s| s != slot_ids[0]);
    let n_emit = if emit_all { n_tokens } else { 1 };
    if n_emit > state.pool.config.max_slots.max(1) {
        bail!(
            "output_head: n_emit {n_emit} > max_slots {} (pool.logits_f32_dev too small)",
            state.pool.config.max_slots.max(1)
        );
    }

    let input_slice_ptr = if emit_all {
        input.ptr
    } else {
        input.ptr.offset_bytes((n_tokens - 1) * hidden * 2)
    };
    let input_slice =
        unsafe { Tensor::<F16>::from_raw(input_slice_ptr, n_emit * hidden) };

    let mut norm_q8_1 =
        unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, n_emit * hidden) };
    flambeau_model_ops::rmsnorm_quant_q8_1(
        &input_slice,
        &lm_head.output_norm,
        &mut norm_q8_1,
        n_emit,
        hidden,
        lm_head.rms_eps,
        &ops,
    )?;

    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
    let mut logits_f32 =
        unsafe { Tensor::<F32>::from_raw(state.pool.logits_f32_dev, n_emit * vocab) };
    // Q4_K LM-head at decode (n_emit=1) gets r4 — halves block count over
    // r2, which is the dispatch default for typical hidden-size matmuls.
    // At vocab=131072+ the per-block work is small enough that r4's 4-row
    // amortisation pays off where it doesn't at hidden-sized n. Measured
    // +2.6% prefill / +1.1% decode on E4B-Q4_0.
    use flambeau_core::op::QDtype;
    let use_r4_lmhead = n_emit == 1
        && lm_head.lm_head.dtype == QDtype::Q4_K
        && vocab >= 131072
        && hidden % 256 == 0;
    if use_r4_lmhead {
        let n_superblocks = hidden / 256;
        flambeau_ops::hip::qmatmul::mmvq_simple_launch(
            state.reg,
            state.stream,
            "mmvq_q4_k_r4",
            "flambeau_mmvq_q4_k_r4_q8_1",
            lm_head.lm_head.ptr,
            state.pool.norm_q8_1,
            state.pool.logits_f32_dev,
            vocab,
            n_superblocks,
            64,
            4,
        )?;
    } else {
        lm_head.lm_head.qmatmul(
            &norm_q8_1,
            &act_mmq_null,
            &mut logits_f32,
            n_emit,
            hidden,
            vocab,
            &ops,
        )?;
    }

    if let Some(cap) = lm_head.final_logit_softcap {
        let mut logits_inplace = unsafe {
            Tensor::<F32>::from_raw(state.pool.logits_f32_dev, n_emit * vocab)
        };
        flambeau_model_ops::apply_softcap_f32(
            &logits_f32,
            &mut logits_inplace,
            n_emit * vocab,
            cap,
            &ops,
        )?;
    }

    let n_elems = n_emit * vocab;
    if state.logits_host.len() != n_elems {
        state.logits_host = vec![0.0_f32; n_elems];
    }
    let bytes = n_elems * 4;
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
