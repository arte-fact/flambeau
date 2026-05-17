//! Shared helpers for the per-topology `layer_range` iterators.
//!
//! Each topology's `layer_range` method returns an iterator over the
//! global layer indices THIS context processes in a single forward
//! call. The iterator's drop / final-yield site is where stage
//! handoff lives on PP / Hybrid — the model sees only
//! `for layer_idx in ctx.layer_range(layout) { ... }`.
//!
//! This module hosts the iterator types (still to be implemented as
//! composites land). The trait method on `ForwardCtx` returns
//! `Box<dyn Iterator<Item = usize>>` to keep the signature simple;
//! the executor can box a concrete iterator that owns the handoff
//! closure.

// Iterator implementations land alongside the first model-v2 that
// drives them. Until then, this module is a placeholder so callers
// know where to look.
