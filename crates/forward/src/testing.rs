//! Device-free `ForwardCtx` that records the composite call sequence.
//! Lets model unit tests assert op order + per-layer indices without
//! a HIP device. Returned tensors are NULL/0 placeholders and never
//! dereferenced by the recording impl.

#![cfg(test)]

use anyhow::Result;
use flambeau_core::DevicePtr;
use flambeau_model_ops::{Tensor, F16};

use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpCall {
    Embed { token: u32 },
    Rmsnorm,
    ResidualAdd,
    StandardAttn { layer_idx: usize, position: usize },
    GdnLayer { layer_idx: usize },
    DenseFfn { layer_idx: usize },
    MoeFfn { layer_idx: usize },
    OutputHead,
}

pub struct RecordingCtx {
    pub ops_called: Vec<OpCall>,
    pub layer_range_yield: Vec<usize>,
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

    pub fn with_layers(mut self, layers: impl IntoIterator<Item = usize>) -> Self {
        self.layer_range_yield = layers.into_iter().collect();
        self
    }

    fn dummy_tensor() -> Tensor<F16> {
        // SAFETY: NULL ptr / 0 len — never dereferenced by the recording impl.
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

    fn gdn_layer(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &crate::ctx::GdnWeights,
        layer_idx: usize,
    ) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::GdnLayer { layer_idx });
        Ok(Self::dummy_tensor())
    }

    fn dense_ffn(&mut self, _input: &Tensor<F16>, _weights: &FfnWeights) -> Result<Tensor<F16>> {
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
