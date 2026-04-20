//! flambeau-ops — typed fused building blocks used by model crates.
//!
//! Everything here is `Mesh<N>`-generic: `Mesh<1>` is a degenerate single-GPU instance
//! routed through the same trait. Model crates import only from here, never from
//! backend-* or kernels-*.
