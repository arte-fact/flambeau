//! Typed activation buffers with compile-time distribution tracking.
//!
//! Every device-resident activation in a TP / pp+tp forward pass has a
//! *distribution*: it is private to one rank ([`Local`]), byte-identical
//! across all ranks ([`Replicated`]), column-sliced along a known dim
//! ([`ColParallel`]), row-sliced ([`RowParallel`] — a per-rank partial
//! that *must* be AllReduced before it represents a full-hidden value),
//! or stage-local within a hybrid sub-cluster ([`SubClusterPartial`]).
//!
//! `Buffer<T, D>` wraps a raw [`DevicePtr`] with two phantom-typed
//! markers — element type and distribution. Ops that read a full-hidden
//! tensor declare `Buffer<F16, Replicated>` in their signature; the
//! compiler rejects an attempt to feed in a `Buffer<F16,
//! RowParallel<DIM>>` because the partial hasn't been reduced yet.
//!
//! Transitions between distributions live in [`Buffer`] methods (or
//! free helpers like `tp_allreduce_sum_into`, Phase 4d). Each typed
//! transition consumes the source `Buffer` and returns the new
//! distribution. The `project_tp4d_i3_gdn_kq_replicated` bug — a
//! row-parallel partial being read as full-hidden without an AR — is a
//! compile error under this scheme.

#![cfg(feature = "hip")]

use std::marker::PhantomData;

use flambeau_core::DevicePtr;

// ---------------------------------------------------------------------------
// Element type markers.
// ---------------------------------------------------------------------------

/// Marker trait for the kinds of element a [`Buffer`] holds.
/// Implementations are zero-sized.
pub trait ElemType: 'static {
    const NAME: &'static str;
    /// Element size in bytes.
    const BYTES: usize;
}

/// F16 activation element (2 bytes).
pub struct F16;
impl ElemType for F16 {
    const NAME: &'static str = "F16";
    const BYTES: usize = 2;
}

/// F32 activation element (4 bytes).
pub struct F32;
impl ElemType for F32 {
    const NAME: &'static str = "F32";
    const BYTES: usize = 4;
}

/// I32 element (used for positions, mask scratch, etc.).
pub struct I32;
impl ElemType for I32 {
    const NAME: &'static str = "I32";
    const BYTES: usize = 4;
}

// ---------------------------------------------------------------------------
// Distribution typestate.
// ---------------------------------------------------------------------------

/// Marker trait for [`Buffer`] distribution. The compiler picks the
/// correct AR / peer-copy / gather between types via the typestate.
pub trait Distribution: 'static {
    const NAME: &'static str;
    /// `true` iff the buffer's contents are byte-identical across all
    /// ranks. Helpers can use this to skip cross-rank operations.
    const IS_REPLICATED: bool = false;
    /// `true` iff the buffer is a row-parallel partial waiting for an
    /// AllReduce.
    const REQUIRES_REDUCE: bool = false;
}

/// Per-rank private buffer. No cross-rank semantics — every rank has
/// its own data, the contents may differ arbitrarily. Default for
/// per-rank scratch (KV append targets, intermediate Q8_1 quant
/// staging, etc.).
pub struct Local;
impl Distribution for Local {
    const NAME: &'static str = "Local";
}

/// Byte-identical across every rank in the TP mesh. The full-hidden
/// activation between layers (after the post-FFW residual + AR) lives
/// here.
pub struct Replicated;
impl Distribution for Replicated {
    const NAME: &'static str = "Replicated";
    const IS_REPLICATED: bool = true;
}

/// Column-sliced across ranks along `DIM`. Per-rank shape on `DIM` is
/// `full / world`. Produced by column-parallel matmuls (Q / K / V
/// projections in Megatron-style TP).
pub struct ColParallel<const DIM: usize>;
impl<const DIM: usize> Distribution for ColParallel<DIM> {
    const NAME: &'static str = "ColParallel";
}

/// Row-sliced across ranks along `DIM`. The buffer holds one rank's
/// *partial* contribution to a full-hidden value; it must be
/// AllReduce-summed (`tp_allreduce_sum_into`) to become
/// [`Replicated`].
///
/// Ops that need a full-hidden input declare `Buffer<T, Replicated>`,
/// not `Buffer<T, RowParallel<DIM>>` — passing the latter is a
/// compile error.
pub struct RowParallel<const DIM: usize>;
impl<const DIM: usize> Distribution for RowParallel<DIM> {
    const NAME: &'static str = "RowParallel";
    const REQUIRES_REDUCE: bool = true;
}

/// Hybrid stage-local: byte-identical within one PP stage's TP
/// sub-cluster, but different across stages. The PP cross-stage
/// hand-off `peer_copy_via_host` migrates a `SubClusterPartial`
/// activation from stage `s` to a [`Replicated`] (or
/// `SubClusterPartial` on the next stage if intermediate).
pub struct SubClusterPartial;
impl Distribution for SubClusterPartial {
    const NAME: &'static str = "SubClusterPartial";
}

// ---------------------------------------------------------------------------
// Buffer wrapper.
// ---------------------------------------------------------------------------

/// A typed view into a caller-owned device buffer.
///
/// `Buffer<T, D>` is a thin wrapper over [`DevicePtr`] with two phantom
/// markers: the element type `T: ElemType` and the distribution `D:
/// Distribution`. It carries no lifetime — the underlying allocation
/// is owned elsewhere (typically a `RawAllocTracker`), and `Buffer`
/// just tags it with the typestate the type system uses for
/// distribution checks.
///
/// Construction at boundaries:
/// - Use [`Buffer::from_raw_unchecked`] when wrapping a raw allocation
///   the caller knows the distribution of (uploader output, scratch
///   alloc).
/// - Distribution transitions (AR, peer-copy, gather/scatter) consume
///   the source buffer and return a freshly-tagged buffer of the new
///   distribution.
#[derive(Debug)]
pub struct Buffer<T: ElemType, D: Distribution> {
    ptr: DevicePtr,
    n_elems: usize,
    _t: PhantomData<T>,
    _d: PhantomData<D>,
}

impl<T: ElemType, D: Distribution> Buffer<T, D> {
    /// Wrap a raw device pointer with the given distribution. The
    /// caller asserts the distribution invariant — the type system
    /// will enforce it from then on.
    pub fn from_raw_unchecked(ptr: DevicePtr, n_elems: usize) -> Self {
        Self {
            ptr,
            n_elems,
            _t: PhantomData,
            _d: PhantomData,
        }
    }

    /// Raw underlying pointer. Use sparingly — every callsite that
    /// reads `.ptr()` is bypassing the typestate.
    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    /// Element count along the local dim. For [`Replicated`] this is
    /// the full count; for `ColParallel<DIM>` / `RowParallel<DIM>` it
    /// is the per-rank shard count.
    pub fn n_elems(&self) -> usize {
        self.n_elems
    }

    /// Allocation size in bytes (n_elems * T::BYTES).
    pub fn bytes(&self) -> usize {
        self.n_elems * T::BYTES
    }

    /// Drop the distribution typestate and return the underlying
    /// `DevicePtr`. Use only at FFI / kernel-launch boundaries where
    /// the kernel takes a raw pointer.
    pub fn into_raw(self) -> DevicePtr {
        self.ptr
    }

    /// Re-tag the buffer with a different distribution without
    /// changing the underlying data. Used by typed-transition methods
    /// (AR, peer-copy) that finished their work — they call this
    /// internally to produce the next typestate. **Not for general
    /// use** — bypasses the safety the typestate provides.
    pub fn retag<D2: Distribution>(self) -> Buffer<T, D2> {
        Buffer {
            ptr: self.ptr,
            n_elems: self.n_elems,
            _t: PhantomData,
            _d: PhantomData,
        }
    }
}

// `Buffer<T, D>` is `Copy` so it composes ergonomically through
// function calls (it's a `DevicePtr` + count + zero-sized markers).
impl<T: ElemType, D: Distribution> Clone for Buffer<T, D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ElemType, D: Distribution> Copy for Buffer<T, D> {}

// ---------------------------------------------------------------------------
// Convenience aliases for the most common buffer shapes.
// ---------------------------------------------------------------------------

/// F16 buffer holding a *per-rank partial* row-parallel hidden vector
/// (`hidden / world` elems? No — for row-parallel attn_output /
/// ffn_down the per-rank output is full-hidden; the partial nature is
/// in the values, not the shape). Pre-AR.
pub type RowPartialF16<const DIM: usize> = Buffer<F16, RowParallel<DIM>>;

/// F16 buffer that is byte-identical across ranks. Post-AR, the
/// canonical "hidden state between layers" type.
pub type ReplicatedF16 = Buffer<F16, Replicated>;

/// F16 buffer private to one rank (per-rank scratch, KV append, etc.).
pub type LocalF16 = Buffer<F16, Local>;

/// F32 column-parallel buffer (gate / up projection outputs in
/// Megatron TP).
pub type ColParallelF32<const DIM: usize> = Buffer<F32, ColParallel<DIM>>;
