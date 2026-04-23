//! `Op` trait — the typed contract every model-visible operation implements.
//!
//! Architectural rule 3 from CLAUDE.md: **one contract per op, many impls.**
//! Models compose ops from `flambeau-ops`; each op carries a typed signature
//! (`Input`, `Output`, `Cfg`) and a contract struct (`OpContract`) describing
//! what the operation *promises* downstream (output shape, dtype, tolerance
//! envelope). Concrete kernels implement [`KernelImpl`] against the same op
//! trait; the dispatcher selects an impl at runtime using a shape-predicate
//! table (`dispatch/<backend>/<arch>.toml`).
//!
//! V1.3+ surface: [`QMatMul`] as the first op. The four MMVQ + two MMQ
//! kernels we've landed register as `KernelImpl<QMatMul, HipDevice>`. This
//! trait surface is deliberately minimal — it replaces bare impl-id strings
//! with typed references, without pulling in async / gradient / scheduler
//! concerns (those are V1.6+ territory).

use crate::device::Device;

/// A tensor-algebra operation with a fixed input/output signature.
///
/// Implementors are usually zero-size marker types; all behaviour lives on
/// `KernelImpl<Self, D>` impls registered in backend crates.
pub trait Op: 'static {
    /// The named input tuple (usually a struct with borrowed tensors).
    type Input<'a, D: Device + 'a>;
    /// The named output tuple.
    type Output<'a, D: Device + 'a>;
    /// Shape / dtype / predicate information the dispatcher uses to pick
    /// an impl. Cheap to construct.
    type Cfg;
}

/// Static tolerance envelope a given op's kernel impls promise to respect.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Maximum `|got - ref|` at `|ref|=0`. Absolute error floor.
    pub abs: f32,
    /// Maximum `|got - ref| / max(|ref|, abs_floor)`. Relative error cap.
    pub rel: f32,
}

impl Tolerance {
    pub const fn new(abs: f32, rel: f32) -> Self {
        Self { abs, rel }
    }
}

/// Human-readable contract derived from an `Op::Cfg`. The dispatcher logs
/// this on kernel selection so mismatches between the cert grid and a live
/// request are visible without rebuilding.
#[derive(Debug, Clone)]
pub struct OpContract {
    pub op_name: &'static str,
    pub output_shape: Vec<usize>,
    pub output_dtype: &'static str,
    pub tolerance: Tolerance,
}

/// A concrete kernel implementation for an op on one device.
///
/// `KernelImpl` is the analog of a `dispatch/*.toml` row plus its cert:
/// `ID` matches the `impl` column, `applies` is the runtime evaluation of
/// the row's `shape` predicate, and `cert()` points at the `certs/` file.
/// V1.3+ only wires the lookup — registration happens in backend crates
/// and is pulled together by the dispatcher in V1.7 (model forward-pass).
pub trait KernelImpl<O: Op, D: Device>: Send + Sync + 'static {
    /// Stable identifier. Used in `dispatch/*.toml` rows + `certs/*.json`.
    const ID: &'static str;

    /// True iff this impl can handle the given `(input, cfg)`. Typical form:
    /// "m in 1..512 and k is a multiple of the block size".
    fn applies(input: &O::Input<'_, D>, cfg: &O::Cfg) -> bool;

    /// Path to the cert JSON, relative to the repo root.
    fn cert_path() -> &'static str;

    /// Human-readable name for logging (e.g. `"mmvq_q4_K_nw1_r2"`).
    fn short_name() -> &'static str {
        Self::ID
    }
}

/// QMatMul — quantised weight × activation matrix-multiply, producing F32.
///
/// Used for both MMVQ (M=1 decode path) and MMQ (M≥128 prefill path) —
/// same op, different impls. The dispatcher picks between them based on
/// the `m` predicate in `dispatch/hip/gfx906.toml`.
#[derive(Debug)]
pub struct QMatMul;

/// RMSNorm — root-mean-square layer normalisation with a learnable weight
/// vector. Same `Op` at both F16 and Q8_1 outputs; impls select via dtype.
#[derive(Debug)]
pub struct RmsNorm;

/// SwiGLU — `silu(gate) * up`, pointwise. Used post-FFN-gate+up.
#[derive(Debug)]
pub struct SwiGLU;

/// `QMatMul` config — dtype + shape extents known at dispatch time.
#[derive(Debug, Clone, Copy)]
pub struct QMatMulCfg {
    pub dtype_weight: QDtype,
    pub dtype_activation: QDtype,
    /// Output rows (= weight rows).
    pub n: usize,
    /// Batch rows (= activation rows; 1 for MMVQ).
    pub m: usize,
    /// Contracted dim.
    pub k: usize,
}

/// Narrow dtype tag for `QMatMulCfg`. Mirrors `flambeau-quant::GgmlDType`
/// but kept here without the quant-crate dep so `core` stays leaf-level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    non_camel_case_types,
    reason = "mirror GGUF / ggml dtype names (Q4_K, Q5_K, Q6_K) so logs/errors/dispatch \
              rows read the same as the external file format and llama.cpp references"
)]
pub enum QDtype {
    F32,
    F16,
    BF16,
    Q8_0,
    Q8_1,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl QDtype {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
        }
    }
}

impl Op for QMatMul {
    // `Input` / `Output` are placeholders until backend-hip lands the real
    // tensor types. We keep the GATs so downstream code can start typing
    // against the op without a second trait bump later.
    type Input<'a, D: Device + 'a> = QMatMulInput<'a, D>;
    type Output<'a, D: Device + 'a> = QMatMulOutput<'a, D>;
    type Cfg = QMatMulCfg;
}

/// RMSNorm config — per-call. `k` is the contracted-then-normalised dim;
/// `m` is the number of rows (= batch × seq_len for a transformer block).
#[derive(Debug, Clone, Copy)]
pub struct RmsNormCfg {
    pub dtype_in: QDtype,
    pub dtype_out: QDtype,
    pub m: usize,
    pub k: usize,
    pub eps: f32,
}

#[derive(Debug)]
pub struct RmsNormInput<'a, D: Device> {
    pub x: crate::device::DevicePtr,
    pub weight: crate::device::DevicePtr,
    pub _marker: std::marker::PhantomData<&'a D>,
}
#[derive(Debug)]
pub struct RmsNormOutput<'a, D: Device> {
    pub y: crate::device::DevicePtr,
    pub _marker: std::marker::PhantomData<&'a D>,
}

impl Op for RmsNorm {
    type Input<'a, D: Device + 'a> = RmsNormInput<'a, D>;
    type Output<'a, D: Device + 'a> = RmsNormOutput<'a, D>;
    type Cfg = RmsNormCfg;
}

#[derive(Debug, Clone, Copy)]
pub struct SwiGLUCfg {
    pub dtype: QDtype,
    /// Flat length = `m * hidden`.
    pub n: usize,
}

#[derive(Debug)]
pub struct SwiGLUInput<'a, D: Device> {
    pub gate: crate::device::DevicePtr,
    pub up: crate::device::DevicePtr,
    pub _marker: std::marker::PhantomData<&'a D>,
}
#[derive(Debug)]
pub struct SwiGLUOutput<'a, D: Device> {
    pub y: crate::device::DevicePtr,
    pub _marker: std::marker::PhantomData<&'a D>,
}

impl Op for SwiGLU {
    type Input<'a, D: Device + 'a> = SwiGLUInput<'a, D>;
    type Output<'a, D: Device + 'a> = SwiGLUOutput<'a, D>;
    type Cfg = SwiGLUCfg;
}

/// Placeholder input view — carries device pointers to (weights, activation,
/// output). Real `Tensor<'a, D>` / `QTensor<'a, D>` shapes replace this in
/// V1.7 when the model forward-pass lands.
#[derive(Debug)]
pub struct QMatMulInput<'a, D: Device> {
    pub weights_bytes: usize,
    pub weights: crate::device::DevicePtr,
    pub activation: crate::device::DevicePtr,
    pub _marker: std::marker::PhantomData<&'a D>,
}

#[derive(Debug)]
pub struct QMatMulOutput<'a, D: Device> {
    pub dst: crate::device::DevicePtr,
    pub _marker: std::marker::PhantomData<&'a D>,
}

/// Runtime registry: every `KernelImpl<O, D>` registers a
/// [`KernelDescriptor`] record the dispatcher queries by `(op, dtype,
/// shape)`. A single static registry per process is enough for V1 (we don't
/// hot-swap impls yet).
#[derive(Debug, Clone)]
pub struct KernelDescriptor {
    pub op_name: &'static str,
    pub impl_id: &'static str,
    pub backend: &'static str,
    pub arch: &'static str,
    pub dtype_weight: QDtype,
    pub dtype_activation: QDtype,
    /// Shape predicate as a `(m_min, m_max)` tuple. Dispatch only looks at
    /// `m` in V1 because K and N are always "any" in the committed table.
    /// Extend as needed.
    pub m_range: (usize, usize),
    pub cert_rel_path: &'static str,
}

impl KernelDescriptor {
    pub const fn matches(
        &self,
        dtype_w: QDtype,
        dtype_a: QDtype,
        m: usize,
    ) -> bool {
        let (lo, hi) = self.m_range;
        m >= lo && m <= hi && eq_qdtype(self.dtype_weight, dtype_w) && eq_qdtype(self.dtype_activation, dtype_a)
    }
}

/// Registry entry for a kernel that is invoked directly by a call site
/// (`reg.expect_module("stem")`) rather than through shape-based dispatch.
///
/// Use this for ops where the implementation choice is fixed per dtype / GGUF
/// layer type rather than dependent on an `m_range` predicate — e.g.
/// `indexed_moe_mmvq_q8_0`, `mmvq_q4_0`, `attention_decode_f16_splitk`. These
/// kernels still need a `dispatch/*.toml` row and a cert, but there is
/// exactly one implementation per (op, dtype) tuple so the shape-dispatch
/// machinery adds no value.
///
/// The `dispatch_toml_roundtrip` test in `backend-hip` asserts every TOML
/// `impl = "..."` entry is covered by a `KernelDescriptor` OR a
/// `DirectCallKernel`, closing the V2.8-class drift window.
#[derive(Debug, Clone, Copy)]
pub struct DirectCallKernel {
    pub impl_id: &'static str,
    pub cert_rel_path: &'static str,
}

// const fn helper — QDtype's PartialEq isn't const-callable on stable Rust.
const fn eq_qdtype(a: QDtype, b: QDtype) -> bool {
    a as u8 == b as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    const DESC: KernelDescriptor = KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q4_K_mmvq_nw1_r2_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 512),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_K_mmvq_nw1_r2_gfx906.json",
    };

    #[test]
    fn descriptor_matches_on_dtype_and_m() {
        assert!(DESC.matches(QDtype::Q4_K, QDtype::Q8_1, 1));
        assert!(DESC.matches(QDtype::Q4_K, QDtype::Q8_1, 256));
        assert!(DESC.matches(QDtype::Q4_K, QDtype::Q8_1, 512));
        assert!(!DESC.matches(QDtype::Q4_K, QDtype::Q8_1, 1024)); // m > 512
        assert!(!DESC.matches(QDtype::Q6_K, QDtype::Q8_1, 1));    // dtype_w mismatch
        assert!(!DESC.matches(QDtype::Q4_K, QDtype::F16, 1));     // dtype_a mismatch
    }
}
