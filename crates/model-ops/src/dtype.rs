//! ZST element-type markers for `Tensor<T>`. Block-quant dtypes round
//! byte sizes up to a block boundary; fixed-width dtypes don't.

pub trait ElemType: sealed::Sealed + 'static {
    fn bytes_for_n_elems(n: usize) -> usize;
    fn name() -> &'static str;
}

macro_rules! fixed_elem_type {
    ($name:ident, $bytes:expr, $label:literal) => {
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

// (block_elems, type_bytes) must match `flambeau_quant::GgmlDType::{block_size, type_size}`.
block_elem_type!(Q4_0, 32, 18, "q4_0");
block_elem_type!(Q4_1, 32, 20, "q4_1");
block_elem_type!(Q5_0, 32, 22, "q5_0");
block_elem_type!(Q5_1, 32, 24, "q5_1");
block_elem_type!(Q8_0, 32, 34, "q8_0");
block_elem_type!(Q8_1, 32, 36, "q8_1");

mod sealed {
    pub trait Sealed {}
}
