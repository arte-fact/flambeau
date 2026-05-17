//! Composite-op free functions shared across every topology.
//!
//! One file per composite — same discipline as `flambeau-model-ops`'s
//! `src/ops/`. Each function takes `&mut CoreState + &mut H: TopologyHooks`
//! plus the composite's typed inputs.
//!
//! Trait method bodies on `SingleDeviceForwardCtx` /
//! `PpForwardCtx` / `TpForwardCtx` / `HybridForwardCtx` delegate here.
//! As PP / TP land, AR / peer-copy / rank-guard hook calls are
//! inserted at the right points inside these bodies (not in the model
//! and not via wholesale per-topology re-implementations).

pub mod dense_ffn;
pub mod embed;
pub mod moe_ffn;
pub mod output_head;
pub mod residual_add;
pub mod rmsnorm;
pub mod standard_attn;

pub use dense_ffn::dense_ffn_local;
pub use embed::embed_local;
pub use moe_ffn::moe_ffn_local;
pub use output_head::output_head_local;
pub use residual_add::residual_add_local;
pub use rmsnorm::rmsnorm_local;
pub use standard_attn::standard_attn_local;

use flambeau_core::DevicePtr;
use flambeau_model_ops::{Tensor, F16};

/// Wrap a pool slot pointer as `Tensor<F16>` with the given length.
///
/// # Safety
/// `ptr` must be a `ScratchPool` slot sized for at least `n_elems`
/// F16 elements. Callers in this module satisfy that by construction.
pub(super) fn slot_f16(ptr: DevicePtr, n_elems: usize) -> Tensor<F16> {
    // SAFETY: caller-side invariant — see doc.
    unsafe { Tensor::<F16>::from_raw(ptr, n_elems) }
}
