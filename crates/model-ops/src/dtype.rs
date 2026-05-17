//! Element-type markers.
//!
//! Each supported on-device element kind is a zero-sized struct that
//! parameterises `Tensor<T>`. Ops constrain their inputs/outputs by
//! the marker type, so the type-checker enforces "you can't pass an
//! `F32` tensor to a kernel that expects `F16`".
//!
//! Markers are intentionally bare. They expose `bytes_per_elem()` (for
//! buffer sizing) and a stable name (for diagnostics). They do NOT
//! carry trait machinery for arithmetic — ops own the per-dtype kernel
//! choice, not the type system.

/// Element type marker. Implemented by the zero-sized structs below.
pub trait ElemType: sealed::Sealed + 'static {
    /// Bytes occupied by one element on the device.
    fn bytes_per_elem() -> usize;
    /// Short stable name for diagnostics (`"f16"`, `"q4_0"`, …).
    fn name() -> &'static str;
}

macro_rules! elem_type {
    ($name:ident, $bytes:expr, $label:literal) => {
        /// `ElemType` marker. See module docs.
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl ElemType for $name {
            fn bytes_per_elem() -> usize {
                $bytes
            }
            fn name() -> &'static str {
                $label
            }
        }
    };
}

elem_type!(F32, 4, "f32");
elem_type!(F16, 2, "f16");
elem_type!(I32, 4, "i32");

// GGML quant families. Bytes-per-elem here is "bytes per logical
// element" averaged over a block (block_size_in_bytes / block_n_elems);
// callers that care about block alignment route through
// `flambeau-quant` to compute exact buffer sizes.
elem_type!(Q4_0, 1, "q4_0"); // 18 bytes / 32 elems = 0.5625 → rounded
elem_type!(Q4_1, 1, "q4_1");
elem_type!(Q5_0, 1, "q5_0");
elem_type!(Q5_1, 1, "q5_1");
elem_type!(Q8_0, 1, "q8_0");
elem_type!(Q8_1, 1, "q8_1");

mod sealed {
    pub trait Sealed {}
}
