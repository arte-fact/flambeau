//! Typed kernel-launch wrappers around the [`flambeau_ops::Ops`] trait.
//!
//! Each fn here takes [`Buffer<T, D>`] arguments instead of raw
//! [`DevicePtr`]s and delegates to the matching `Ops::method` via
//! `.ptr()`. The legacy `Ops::method(ptr, ptr, ptr, ...)` API stays
//! unchanged; arch code that wants compile-time data-flow tracking
//! opts in to the typed wrappers per call site.
//!
//! The wrappers are intentionally minimal — they enforce *distribution
//! invariants* (e.g. rmsnorm preserves distribution, AR transitions
//! consume RowParallel and produce Replicated) but stay zero-cost at
//! runtime because [`Buffer`] is a `Copy` wrapper around `DevicePtr` +
//! a couple of phantom markers.
//!
//! When a kernel changes distribution (e.g. column-parallel matmul
//! goes `Buffer<I32, Replicated>` → `Buffer<F32, ColParallel<0>>`),
//! the wrapper enforces it via the generic signature. Distribution
//! mismatches at call sites are compile errors.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_ops::Ops;

use crate::tensor_view::{Buffer, Distribution, ElemType, Replicated, F16, F32};

/// `y[i] = rmsnorm(x[i]) * weight` — preserves distribution. The
/// weights are always [`Replicated`] (norm weights are global). Used
/// for `attn_norm`, `post_attention_norm`, `ffn_norm`,
/// `post_ffw_norm`, per-head Q/K norms, output_norm.
#[inline]
pub fn rmsnorm_f16<O: Ops, D: Distribution>(
    ops: &O,
    x: Buffer<F16, D>,
    weight: Buffer<F16, Replicated>,
    y: Buffer<F16, D>,
    m: usize,
    k: usize,
    eps: f32,
) -> Result<()> {
    ops.rmsnorm_f16(x.ptr(), weight.ptr(), y.ptr(), m, k, eps)
}

/// `y[i] = a[i] + b[i]` over `n` F16 lanes — preserves distribution.
/// Residual-add inside a layer; both operands must share distribution
/// (residual stream + post-norm output are both Replicated; partial-
/// accumulator + partial-output would both be RowParallel).
#[inline]
pub fn add_f16<O: Ops, D: Distribution>(
    ops: &O,
    a: Buffer<F16, D>,
    b: Buffer<F16, D>,
    y: Buffer<F16, D>,
    n: usize,
) -> Result<()> {
    ops.add_f16(a.ptr(), b.ptr(), y.ptr(), n)
}

/// `y[i] = scale * x[i]` over `n` F16 lanes — preserves distribution.
/// Used for gemma4 `layer_output_scale` + softmax pre-scaling.
#[inline]
pub fn scale_f16<O: Ops, D: Distribution>(
    ops: &O,
    x: Buffer<F16, D>,
    y: Buffer<F16, D>,
    n: usize,
    scale: f32,
) -> Result<()> {
    ops.scale_f16(x.ptr(), y.ptr(), n, scale)
}

/// `y_q8_1 = quantize(x_f16)` block-wise — preserves distribution. The
/// Q8_1 output's logical element count matches `x_f16`'s; storage size
/// is `(n / 32) * sizeof(BlockQ8_1)` = `(n / 32) * 36` bytes. Use the
/// `F16` marker on the Q8_1 buffer (it's not literally F16, but the
/// element-count semantics line up).
#[inline]
pub fn quantize_f16_q8_1<O: Ops, D: Distribution>(
    ops: &O,
    x_f16: Buffer<F16, D>,
    y_q8_1: Buffer<F16, D>,
    n_elems: usize,
) -> Result<()> {
    ops.quantize_f16_q8_1(x_f16.ptr(), y_q8_1.ptr(), n_elems)
}

/// `y_f16[i] = f16(x_f32[i])` — preserves distribution. Used for
/// post-matmul cast (F32 mmvq output → F16 attention input).
#[inline]
pub fn cast_f32_to_f16<O: Ops, D: Distribution>(
    ops: &O,
    x_f32: Buffer<F32, D>,
    y_f16: Buffer<F16, D>,
    n: usize,
) -> Result<()> {
    ops.cast_f32_to_f16(x_f32.ptr(), y_f16.ptr(), n)
}

/// `y_f16[i] = f16(gelu(a[i]) * b[i])` — preserves distribution.
/// Gemma 4 dense FFN's gate × up fusion (GELU = ggml tanh form).
#[inline]
pub fn gelu_f32_to_f16<O: Ops, D: Distribution>(
    ops: &O,
    a: Buffer<F32, D>,
    b: Buffer<F32, D>,
    y: Buffer<F16, D>,
    n: usize,
) -> Result<()> {
    ops.gelu_f32_to_f16(a.ptr(), b.ptr(), y.ptr(), n)
}

/// `y_f16[i] = f16(silu(a[i]) * b[i])` — preserves distribution.
/// Qwen3 + Llama-family FFN's SwiGLU gate × up fusion.
#[inline]
pub fn swiglu_f32_to_f16<O: Ops, D: Distribution>(
    ops: &O,
    a: Buffer<F32, D>,
    b: Buffer<F32, D>,
    y: Buffer<F16, D>,
    n: usize,
) -> Result<()> {
    ops.swiglu_f32_to_f16(a.ptr(), b.ptr(), y.ptr(), n)
}

// ---------------------------------------------------------------------------
// Convenience: untyped element-count helpers.
// ---------------------------------------------------------------------------

/// Force an untyped element-count read, used at boundaries where the
/// typed buffer needs to be inspected for size without dropping into
/// the raw `DevicePtr` API.
#[inline]
pub fn n_elems<T: ElemType, D: Distribution>(b: Buffer<T, D>) -> usize {
    b.n_elems()
}
