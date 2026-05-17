//! Shared composite engine.
//!
//! `ForwardCtx` trait impls are thin wrappers around this module:
//!
//! - `CoreState` carries the per-request scratch / KV / logits buffer.
//! - `TopologyHooks` is a trait of small primitives (AR, peer-copy,
//!   rank guards) that each topology impl provides; defaults are
//!   no-ops, which is what SingleDevice uses.
//! - `composites::*` are free functions taking `&mut CoreState +
//!   &mut H: TopologyHooks` and contain the bulk of every composite's
//!   body. The trait impl methods just delegate.
//!
//! This split means new topologies are written as a small `Hooks` impl
//! plus ~8 trait-method delegates — no per-topology copy of the 600+
//! LOC composite bodies.

pub mod composites;
pub mod hooks;
pub mod scratch;
pub mod state;

pub use hooks::{NoopHooks, TopologyHooks};
pub use scratch::{ScratchConfig, ScratchPool};
pub use state::CoreState;
