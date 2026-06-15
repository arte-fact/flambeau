//! Generic embedding-model handle: pooled-embedding forward + token cap.
//!
//! Routes.rs holds the handle as `Box<dyn EmbeddingHandle>` so the
//! `/v1/embeddings` handler doesn't reach into arch-specific code
//! directly. Concrete impls live in their respective model crates.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};

pub trait EmbeddingHandle: Send {
    fn max_tokens(&self) -> usize;
    fn compute_pooled_embedding(
        &mut self,
        device: &HipDevice,
        stream: &HipStream,
        tokens: &[u32],
    ) -> Result<Vec<f32>>;
}
