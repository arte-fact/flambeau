//! Generic embedding-model handle: pooled-embedding forward + token cap.
//!
//! Routes.rs holds the handle as `Box<dyn EmbeddingHandle>` so the
//! `/v1/embeddings` handler doesn't reach into `flambeau_qwen3_moe`
//! directly. Today the only impl is qwen3-moe's `EmbeddingModel`; a
//! gemma4 or CUDA embedding backend would impl the same trait.

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

impl EmbeddingHandle for flambeau_qwen3_moe::EmbeddingModel {
    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    fn compute_pooled_embedding(
        &mut self,
        device: &HipDevice,
        stream: &HipStream,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        flambeau_qwen3_moe::EmbeddingModel::compute_pooled_embedding(self, device, stream, tokens)
    }
}
