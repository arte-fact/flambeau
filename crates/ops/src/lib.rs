//! flambeau-ops — typed fused building blocks used by model crates.
//! Everything here is `Mesh<N>`-generic: `Mesh<1>` is a degenerate single-GPU
//! instance routed through the same trait. Model crates import only from
//! here, never from `backend-*` or `kernels-*`.
//! ## surface
//! - [`OpsRegistry`] — one-shot loader for every HIP kernel module the model
//!   touches. Model code holds `&OpsRegistry` for the life of the session.
//! - `qmatmul` — quantised weight × activation matmul. Runtime-dispatched to
//!   MMVQ or MMQ by `(dtype, m)`; all K-quants routed through the existing
//!   `dispatch/hip/gfx906.toml` rows.
//! - `norm` — RMSNorm (F16 in, F16 out) + fused RMSNorm+Quantize<Q8_1>.
//! - `pe` — RoPE in-place (F16, interleaved-pair).
//! - `mlp` — SwiGLU pointwise.
//! - `softmax` — masked-and-scaled softmax.
//! - `attention` — decode (F16 KV, Q8 KV) + prefill (F16).
//! - `moe` — TopK router, IndexedMoE MMVQ + MMQ + fused gate+up, combine.
//!   Each op module exposes stateless free functions that accept `&OpsRegistry`,
//!   device pointers, and shape scalars. The registry owns the `HipModule`s;
//!   each launch looks up the kernel by name (cheap) and fires. No per-call
//!   hsaco parsing.
//!   Mesh-genericity: the HIP-specialised surface below takes a single
//!   `HipDevice` / `HipStream`. wraps this behind a `MeshOps<M: Mesh>`
//!   trait that lives alongside the collective ops in `flambeau-runtime`.

#[cfg(feature = "hip")]
pub mod hip;

#[cfg(feature = "cuda")]
pub mod cuda;

// Backend-generic op-launch bodies (one `fn op<B: KernelBackend>(…)` per op).
// Always compiled — HIP and CUDA share the same bodies; the `hip` / `cuda`
// modules only add the per-backend `Registry` / `Ops` type aliases.
pub mod launch;

// Backend-generic kernel-module registry over the `core::KernelBackend` seam.
// Always compiled so a CUDA `Ops` impl shares it; `hip::OpsRegistry` is the
// `Registry<HipKernels>` alias.
pub mod registry;
pub use registry::{Registry, RegistryError, KERNEL_STEMS};

// `sig` is the backend-portable op-signature surface (POD buffers / shapes /
// knobs over `DevicePtr` + `core::ScalarSlot`). Always compiled so the `Ops`
// trait and a future `CudaOps` impl share it. The one HIP-specific member —
// `OpCtx` (holds `&HipStream`) — is gated within the module.
pub mod sig;

#[cfg(feature = "hip")]
pub use hip::{OpsRegistry, OpsRegistryError};

pub use sig::{
    AttnBatchedBuffers, AttnBuffers, AttnDecodeBatchedShape, AttnDecodeBatchedSplitkShape,
    AttnDecodePagedShape,
    AttnDecodeQ8Slots, AttnDecodeQ8SplitkSlots,
    AttnDecodeShape, AttnDecodeSlots, AttnDposKnobs, AttnKnobs, AttnPagedDecodeBuffers,
    AttnPagedPrefillBuffers,
    AttnPrefillPagedShape, AttnPrefillShape, AttnPrefillSlots, AttnSplitkDposShape,
    AttnSplitkPartials,
    AttnSplitkShape, AttnSplitkSlots, KvAppendSlots,
    MatmulShape, MmvqBatchShape, MmvqBuffers, MmvqGateUpBatchShape,
    MmvqGateUpBuffers, MmvqGateUpShape, MmvqShape, MoeMmvqBuffers, MoeMmvqGateUpBuffers,
    MoeMmqTile8DownBuffers, MoeMmqTile8GateUpBuffers, MoeMmvqShape,
    KvAppendBatchedSlotsBuffers, KvAppendBatchedSlotsShape, KvAppendBuffers,
    KvAppendPagedPrefillBuffers, KvAppendPagedPrefillShape, KvAppendPagedSlotsBuffers,
    KvAppendPagedSlotsShape, KvAppendVUnitShape, MoeCombineBuffers,
    MoeCombineNoResidualBuffers, MoeCombineShape, MoeCombineTwoResidualsBuffers,
    MoeSortBuffers, MoeSortPaddedBuffers, MoeSortPaddedShape, MoeSortShape,
    MmvqKvF16Buffers, MoeShape, NormBuffers, NormFusedAddBuffers, NormResidualBuffers, NormShape,
    QmatmulBuffers, RopeBuffers, RopeFusedBuffers, RopePartialShape, RopeShape,
    GdnStepBuffers, GdnStepAlphaBetaBuffers, GdnStepAlphaBetaBatchedSlotsBuffers, GdnStepShape,
    GdnAlphaBetaBuffers, GdnAlphaBetaShape, GdnConvTrioBatchedSlotsBuffers, GdnConvTrioShape,
    GdnSplitQkvBuffers, GdnSplitQkvShape,
    PenaltyBuffers, PenaltyKnobs,
    SplitQGateBuffers, SplitQGateShape,
    ConvCausal1dBuffers, ConvCausal1dShape,
    TopkBuffers, TopkShape,
    DenseGemvBatchedBuffers, DenseGemvBatchedShape,
    SamplerTopkSoftmaxBuffers, SamplerTopkSoftmaxKnobs,
    SoftmaxMaskedBuffers, SoftmaxMaskedKnobs,
};

#[cfg(feature = "hip")]
pub use sig::OpCtx;

// The op-launch bodies are backend-generic; re-export them at the historical
// crate-root paths so callers say `flambeau_ops::qmatmul::…` for either backend.
pub use launch::{attention, conv, mlp, moe, norm, pe, qmatmul, softmax};

// The portable op contract. Backend-neutral: every method takes `sig` POD +
// `core` types only (the `Ops` impl — `OpsImpl<B>`, aliased per backend as
// `HipOps` / `CudaOps` — holds the backend `(registry, stream)` pair
// internally). Always compiled.
mod ops_impl;
mod ops_trait;
pub use ops_impl::OpsImpl;
pub use ops_trait::Ops;

#[cfg(feature = "hip")]
pub use hip::HipOps;

#[cfg(feature = "cuda")]
pub use cuda::{CudaOps, CudaOpsRegistry};
