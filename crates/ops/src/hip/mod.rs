//! HIP-specialised op surface.
//! Every model-visible op lands here as a stateless free function taking
//! `&OpsRegistry` + `&HipStream` + device pointers + shape. The registry
//! pre-loads one module per kernel stem at session init; per-call cost is one
//! symbol lookup (`hipModuleGetFunction`) + `hipModuleLaunchKernel`.
//! Architectural placement:
//! - Op wrappers here **never** parse hsaco, never write dispatch predicates
//!   from env flags. They resolve variant selection through the committed
//!   `KernelDescriptor` tables in `flambeau-backend-hip::impls` (architectural
//!   rule 1: dispatch lives in `dispatch/<backend>/<arch>.toml`, mirrored by
//!   the Rust table).
//! - The lifetime story: each op call borrows the registry and a stream; all
//!   kernel args are locals in the launch function so they live until the
//!   launch returns. Matches the pattern in `flambeau-bench::sweep_*`.

use flambeau_backend_hip::HipKernels;
pub use flambeau_backend_hip::{HipDevice, HipStream};

pub mod attention;
pub mod cast;
pub mod conv;
pub mod mlp;
pub mod moe;
pub mod norm;
mod ops_impl;
pub mod pe;
pub mod qmatmul;
pub mod recurrent;
pub mod router;
pub mod sampling;
pub mod softcap;
pub mod softmax;

pub use ops_impl::HipOps;

// The kernel-stem catalogue + the registry are backend-generic now; re-export
// them under the historical `hip::` names.
pub use crate::registry::{Registry, RegistryError as OpsRegistryError, KERNEL_STEMS};

/// HIP single-session op registry — [`Registry`] specialised to the HIP launch
/// backend. Op bodies hold `&OpsRegistry`.
pub type OpsRegistry = Registry<HipKernels>;
