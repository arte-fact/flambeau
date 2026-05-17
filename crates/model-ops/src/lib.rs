//! Typed, individually-testable model ops.
//!
//! Each op is a free function over typed tensors with a co-located
//! mock-data parity test against a CPU reference. See `README.md` for
//! the design rationale and `CLAUDE.md` for the discipline rules.
//!
//! Consumers `use flambeau_model_ops::{rmsnorm_f16, qmatmul_q4_0,
//! ...};` — flat namespace, no `ops::` prefix. Module structure is an
//! implementation detail.

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
pub use ops::cast::{cast_f16_to_f32, cast_f32_to_f16};
pub use ops::qmatmul::{qmatmul_q4_0, qmatmul_q4_1, qmatmul_q5_0, qmatmul_q5_1, qmatmul_q8_0};
pub use ops::quantize::{quantize_f16_to_q8_1, quantize_f32_to_q8_1};
pub use ops::rmsnorm::{rmsnorm_f16, rmsnorm_f32, rmsnorm_quant_q8_1};
pub use ops::rope::{rope_f16, rope_neox_partial_f16};
pub use ops::scale::scale_f16;
pub use ops::topk_softmax::{topk_softmax_f32, SAMPLER_K_OUT_MAX};
