//! Topology executor for model forward passes.
//!
//! See `README.md` for the design and `CLAUDE.md` for the discipline
//! rules. Public surface is intentionally narrow: the `ForwardCtx`
//! trait, three concrete impls, and a recording-mock for tests.
//!
//! A model is `pub fn forward_one_token<C: ForwardCtx>(model, ctx,
//! token, position) -> Result<u32>` — one function, monomorphised per
//! topology by the compiler.

#![cfg(feature = "hip")]

pub mod ctx;
pub mod hybrid;
pub mod layer_range;
pub mod pp;
pub mod tp;

#[cfg(test)]
pub mod testing;

pub use ctx::ForwardCtx;
pub use hybrid::HybridForwardCtx;
pub use pp::PpForwardCtx;
pub use tp::TpForwardCtx;
