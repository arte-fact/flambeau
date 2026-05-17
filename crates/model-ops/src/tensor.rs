//! `Tensor<T>` — typed device-buffer view. Borrows; doesn't own the
//! underlying allocation (the caller's pool / KV / weight store does).

use std::marker::PhantomData;

use flambeau_core::DevicePtr;

use crate::dtype::ElemType;

/// `T` is a ZST dtype marker (`dtype.rs`). `Tensor<F16>` and
/// `Tensor<F32>` are distinct types.
pub struct Tensor<T: ElemType> {
    /// May be `DevicePtr::NULL` for placeholder / optional outputs.
    pub ptr: DevicePtr,
    pub n_elems: usize,
    _phantom: PhantomData<fn() -> T>,
}

impl<T: ElemType> Tensor<T> {
    /// # Safety
    /// `ptr` is `DevicePtr::NULL` or valid for `T::bytes_for_n_elems(n_elems)`
    /// bytes on the device the eventual kernel binds, and outlives the
    /// returned `Tensor`.
    pub unsafe fn from_raw(ptr: DevicePtr, n_elems: usize) -> Self {
        Self {
            ptr,
            n_elems,
            _phantom: PhantomData,
        }
    }

    pub fn bytes(&self) -> usize {
        T::bytes_for_n_elems(self.n_elems)
    }

    /// # Safety
    /// Caller guarantees the device memory contains a valid layout
    /// for `U`. Common case: a Q8_1 scratch buffer re-tagged as Q8_1
    /// input to a quantised matmul.
    pub unsafe fn retag<U: ElemType>(self) -> Tensor<U> {
        Tensor {
            ptr: self.ptr,
            n_elems: self.n_elems,
            _phantom: PhantomData,
        }
    }
}

impl<T: ElemType> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dtype", &T::name())
            .field("ptr", &self.ptr)
            .field("n_elems", &self.n_elems)
            .finish()
    }
}
