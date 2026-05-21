//! Typed, individually-testable model ops. One op per file, flat
//! public namespace, co-located CPU-reference parity tests. See
//! `README.md` + `CLAUDE.md`.

#![cfg(feature = "hip")]

pub mod dtype;
pub mod error;
pub mod tensor;

#[cfg(test)]
pub(crate) mod testing;

pub mod ops;

pub use dtype::{ElemType, F16, F32, I32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1};
pub use error::{Error, Result};
pub use tensor::Tensor;

pub use ops::activation::{gelu_mul_f32_to_f16, swiglu_f16, swiglu_f32_to_f16};
pub use ops::add::{add_f16, add_f32};
pub use ops::attn_decode::attn_decode_f16;
pub use ops::attn_decode_batched::{attn_decode_f16_batched, kv_append_f16_batched_slots};
pub use ops::attn_decode_splitk::{attn_decode_f16_splitk, splitk_chunk_size};
// pub use ops::attn_decode_splitk_h2::attn_decode_f16_splitk_h2;
pub use ops::attn_prefill::attn_prefill_f16;
pub use ops::cast::{cast_f16_to_f32, cast_f32_to_f16};
pub use ops::gated_attn::{sigmoid_mul_f16, split_q_gate_f16};
pub use ops::kv_append::kv_append_f16;
pub use ops::moe_router::topk_f32 as moe_router_topk_f32;
pub use ops::qmatmul::{qmatmul_q4_0, qmatmul_q4_1, qmatmul_q5_0, qmatmul_q5_1, qmatmul_q8_0};
pub use ops::quantize::{quantize_f16_to_q8_1, quantize_f16_to_q8_1_mmq, quantize_f32_to_q8_1};
pub use ops::rmsnorm::{rmsnorm_f16, rmsnorm_f32, rmsnorm_f32_to_f16, rmsnorm_quant_q8_1};
pub use ops::rope::{rope_f16, rope_neox_partial_f16};
pub use ops::scale::scale_f16;
pub use ops::softcap::apply_softcap_f32;
pub use ops::topk_softmax::{topk_softmax_f32, SAMPLER_K_OUT_MAX};
