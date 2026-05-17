//! `Tensor<T>` — typed view over a device buffer.
//!
//! Holds a `DevicePtr` + element count + a marker for dtype. Ops
//! consume `&Tensor<T>` for reads and `&mut Tensor<T>` for writes;
//! the borrow checker is the only correctness layer we get for free,
//! so we use it liberally.
//!
//! `Tensor` does NOT own the device memory. The model crate that
//! constructed the tensor (via its own scratch / KV / weight allocator)
//! holds the lifetime. This crate's ops are stateless: they receive
//! borrowed tensors, launch a kernel, and return.

use std::marker::PhantomData;

use flambeau_core::DevicePtr;

use crate::dtype::ElemType;

/// Typed view over a device buffer.
///
/// The `T` parameter is a marker zst (see `dtype.rs`). `Tensor<F16>`
/// and `Tensor<F32>` are distinct types; the type-checker prevents
/// passing one where the other is expected.
///
/// Layout (parallel vs replicated) is NOT modelled in V1. Ops that
/// act across ranks take `&[&Tensor<T>]` slices (one per rank) and
/// the per-topology executor lives in the model layer. Adding a
/// `Layout` typestate later is possible but explicitly out of scope
/// here; see `CLAUDE.md`.
pub struct Tensor<T: ElemType> {
    /// Device-side pointer. May be `DevicePtr::NULL` to represent an
    /// "optional output" or a placeholder; ops that read from a
    /// tensor must validate non-null before launch.
    pub ptr: DevicePtr,
    /// Number of `T` elements addressable through `ptr`.
    pub n_elems: usize,
    _phantom: PhantomData<fn() -> T>,
}

impl<T: ElemType> Tensor<T> {
    /// Wrap a `(ptr, n_elems)` pair as a typed tensor.
    ///
    /// # Safety
    /// Caller guarantees:
    /// - `ptr` is either `DevicePtr::NULL` or a valid device address
    ///   for at least `n_elems * T::bytes_per_elem()` bytes on the
    ///   device the eventual kernel launch will bind.
    /// - The pointer outlives the returned `Tensor`.
    pub unsafe fn from_raw(ptr: DevicePtr, n_elems: usize) -> Self {
        Self {
            ptr,
            n_elems,
            _phantom: PhantomData,
        }
    }

    /// Byte length of the addressable region. Equals
    /// `n_elems * elem_bytes` for fixed-width dtypes; for block-
    /// quantised dtypes rounds up to a block boundary.
    pub fn bytes(&self) -> usize {
        T::bytes_for_n_elems(self.n_elems)
    }

    /// Re-tag a tensor with a different element marker.
    ///
    /// # Safety
    /// Caller guarantees the underlying device memory contains a valid
    /// layout for `U`. Common case: re-tagging a Q8_1-quantised
    /// scratch buffer as Q8_1 input to a quantised matmul.
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
