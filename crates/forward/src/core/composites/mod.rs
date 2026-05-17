//! One file per composite. Trait method bodies delegate here;
//! topology customisation lands via `TopologyHooks` call sites
//! inside each body.

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

/// # Safety
/// `ptr` must be a `ScratchPool` slot sized for ≥ `n_elems` F16
/// elements. Callers in this module satisfy that by construction.
pub(super) fn slot_f16(ptr: DevicePtr, n_elems: usize) -> Tensor<F16> {
    unsafe { Tensor::<F16>::from_raw(ptr, n_elems) }
}
