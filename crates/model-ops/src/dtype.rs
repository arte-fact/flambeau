//! Element-type markers.
//!
//! Each supported on-device element kind is a zero-sized struct that
//! parameterises `Tensor<T>`. Ops constrain their inputs/outputs by
//! the marker type, so the type-checker enforces "you can't pass an
//! `F32` tensor to a kernel that expects `F16`".
//!
//! Markers are intentionally bare. They expose `bytes_for_n_elems(n)`
//! (for buffer sizing — handles block-aligned quant dtypes) and a
//! stable name (for diagnostics). They do NOT carry trait machinery
//! for arithmetic — ops own the per-dtype kernel choice, not the
//! type system.

/// Element type marker. Implemented by the zero-sized structs below.
pub trait ElemType: sealed::Sealed + 'static {
    /// Bytes required to hold `n` logical elements of this type on
    /// the device. Fixed-width dtypes return `n * elem_bytes`;
    /// block-quantised dtypes round up to a block boundary and return
    /// `ceil(n / block_size) * type_size`.
    fn bytes_for_n_elems(n: usize) -> usize;

    /// Short stable name for diagnostics (`"f16"`, `"q4_0"`, …).
    fn name() -> &'static str;
}

macro_rules! fixed_elem_type {
    ($name:ident, $bytes:expr, $label:literal) => {
        /// `ElemType` marker. See module docs.
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl ElemType for $name {
            fn bytes_for_n_elems(n: usize) -> usize {
                n * $bytes
            }
            fn name() -> &'static str {
                $label
            }
        }
    };
}

macro_rules! block_elem_type {
    ($name:ident, $block_size:expr, $type_size:expr, $label:literal) => {
        /// `ElemType` marker (block-quantised; bytes round up to a
        /// block boundary). See module docs.
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl ElemType for $name {
            fn bytes_for_n_elems(n: usize) -> usize {
                let blocks = n.div_ceil($block_size);
                blocks * $type_size
            }
            fn name() -> &'static str {
                $label
            }
        }
    };
}

fixed_elem_type!(F32, 4, "f32");
fixed_elem_type!(F16, 2, "f16");
fixed_elem_type!(I32, 4, "i32");

// GGML quant block sizes (elements / bytes) — match
// `flambeau-quant::GgmlDType::{block_size, type_size}`. If those
// change upstream, update here.
block_elem_type!(Q4_0, 32, 18, "q4_0");
block_elem_type!(Q4_1, 32, 20, "q4_1");
block_elem_type!(Q5_0, 32, 22, "q5_0");
block_elem_type!(Q5_1, 32, 24, "q5_1");
block_elem_type!(Q8_0, 32, 34, "q8_0");
block_elem_type!(Q8_1, 32, 36, "q8_1");

mod sealed {
    pub trait Sealed {}
}
