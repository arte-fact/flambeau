//! `RecordingCtx` — a `ForwardCtx` impl that records each composite
//! call as an `OpCall` instead of running it. Used by model unit
//! tests to assert "this arch calls embed → rmsnorm → standard_attn
//! → ... in the right order with the right layer indices" without
//! requiring a HIP device.
//!
//! This is structural, not numerical. Real correctness comes from the
//! topology-parity tests on real devices (one per concrete topology
//! ctx, exercising every composite).
//!
//! Method bodies fill in the matching `OpCall` variant and return a
//! dummy `Tensor<F16>` (null ptr, n_elems = 0 by default; tests can
//! seed non-zero shapes if they care). The dummy tensor is never
//! dereferenced — only passed back as ctx-relative tokens that the
//! recording impl ignores.

#![cfg(test)]

use anyhow::Result;
use flambeau_core::DevicePtr;
use flambeau_model_ops::{Tensor, F16};

use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

/// Recorded composite call. One variant per `ForwardCtx` method.
/// `Eq` for assert_eq! in tests; `Debug` for failure messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpCall {
    Embed { token: u32 },
    Rmsnorm,
    ResidualAdd,
    StandardAttn { layer_idx: usize, position: usize },
    DenseFfn { layer_idx: usize },
    MoeFfn { layer_idx: usize },
    OutputHead,
}

/// Recording-mode forward context. Maintains the ordered list of
/// composite calls.
pub struct RecordingCtx {
    pub ops_called: Vec<OpCall>,
    /// Layer indices to yield from `layer_range`. Tests configure
    /// this to match the topology they're simulating (PP: a slice of
    /// layer indices; TP: 0..num_layers; Hybrid: a stage's slice).
    pub layer_range_yield: Vec<usize>,
    /// Logits slot returned by `logits()`. Test fixture.
    pub logits_buf: Vec<f32>,
}

impl RecordingCtx {
    pub fn new() -> Self {
        Self {
            ops_called: Vec::new(),
            layer_range_yield: Vec::new(),
            logits_buf: Vec::new(),
        }
    }

    /// Configure `layer_range` to yield exactly these layer indices.
    pub fn with_layers(mut self, layers: impl IntoIterator<Item = usize>) -> Self {
        self.layer_range_yield = layers.into_iter().collect();
        self
    }

    fn dummy_tensor() -> Tensor<F16> {
        // SAFETY: pointer is NULL + n_elems is 0; this tensor is only
        // ever passed back to a `RecordingCtx` which never dereferences it.
        unsafe { Tensor::<F16>::from_raw(DevicePtr::NULL, 0) }
    }
}

impl Default for RecordingCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl ForwardCtx for RecordingCtx {
    fn embed(&mut self, _token_embd: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::Embed { token: token_id });
        Ok(Self::dummy_tensor())
    }

    fn rmsnorm(
        &mut self,
        _input: &Tensor<F16>,
        _weight: &Tensor<F16>,
        _eps: f32,
    ) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::Rmsnorm);
        Ok(Self::dummy_tensor())
    }

    fn residual_add(&mut self, _a: Tensor<F16>, _b: Tensor<F16>) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::ResidualAdd);
        Ok(Self::dummy_tensor())
    }

    fn standard_attn(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &AttnWeights,
        layer_idx: usize,
        position: usize,
    ) -> Result<Tensor<F16>> {
        self.ops_called
            .push(OpCall::StandardAttn { layer_idx, position });
        Ok(Self::dummy_tensor())
    }

    fn dense_ffn(&mut self, _input: &Tensor<F16>, _weights: &FfnWeights) -> Result<Tensor<F16>> {
        // layer_idx isn't a parameter of dense_ffn today; if a model
        // needs it for routing, the trait method picks it up and this
        // variant captures it.
        self.ops_called.push(OpCall::DenseFfn { layer_idx: 0 });
        Ok(Self::dummy_tensor())
    }

    fn moe_ffn(&mut self, _input: &Tensor<F16>, _weights: &MoeWeights) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::MoeFfn { layer_idx: 0 });
        Ok(Self::dummy_tensor())
    }

    fn output_head(
        &mut self,
        _input: &Tensor<F16>,
        _lm_head: &LmHeadWeights,
    ) -> Result<()> {
        self.ops_called.push(OpCall::OutputHead);
        Ok(())
    }

    fn layer_range<'a>(
        &'a mut self,
        _layout: &'a ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(self.layer_range_yield.clone().into_iter())
    }

    fn logits(&self) -> &[f32] {
        &self.logits_buf
    }
}
