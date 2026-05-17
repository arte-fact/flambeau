//! Topology-customisation hooks.
//!
//! `TopologyHooks` is the small surface each topology impl provides on
//! top of the shared composite engine. Every method has a sensible
//! default that matches **single-device** semantics (no AR, no
//! peer-copy, every rank owns everything). PP / TP / Hybrid override
//! the methods they need.
//!
//! The trait stays small on purpose: we resist the urge to push entire
//! composite bodies through hooks. Hooks are for the discrete points
//! where topologies actually differ — AllReduce after a row-parallel
//! matmul, peer-copy at a PP stage boundary, rank guards for embed /
//! output_head. If you find yourself adding a hook that one topology
//! returns `None` from, the abstraction is wrong — push back.

/// Per-topology hook surface. Default impls match SingleDevice semantics.
pub trait TopologyHooks {
    // PP/TP-specific methods will land in P4/P5 with sensible defaults.
    // P3.5 keeps the trait empty — SingleDevice needs zero hooks.
}

/// All-default hook bundle used by SingleDevice (and by tests that want
/// to exercise composites with no topology behaviour).
pub struct NoopHooks;

impl TopologyHooks for NoopHooks {}
