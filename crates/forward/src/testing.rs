//! Device-free `ForwardCtx` that records the composite call sequence.
//! Lets model unit tests assert op order + per-layer indices without
//! a HIP device. Returned tensors are NULL/0 placeholders and never
//! dereferenced by the recording impl.

#![cfg(test)]

use anyhow::Result;
use flambeau_core::DevicePtr;
use flambeau_model_ops::{Tensor, F16};

use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, GdnWeights, LmHeadWeights, ModelLayout,
    MoeWeights,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpCall {
    Embed {
        tokens: Vec<u32>,
    },
    Rmsnorm {
        n_tokens: usize,
    },
    ResidualAdd {
        n_tokens: usize,
    },
    ScaleInplace {
        n_tokens: usize,
    },
    StandardAttn {
        layer_idx: usize,
        positions: Vec<usize>,
        slot_ids: Vec<usize>,
        next_norm: bool,
    },
    GdnLayer {
        layer_idx: usize,
        slot_ids: Vec<usize>,
        next_norm: bool,
    },
    DenseFfn {
        n_tokens: usize,
        next_norm: bool,
    },
    MoeFfn {
        n_tokens: usize,
        next_norm: bool,
    },
    OutputHead {
        slot_ids: Vec<usize>,
    },
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
    fn embed(&mut self, _token_embd: &EmbeddingWeights, tokens: &[u32]) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::Embed {
            tokens: tokens.to_vec(),
        });
        Ok(Self::dummy_tensor())
    }

    fn rmsnorm(
        &mut self,
        _input: &Tensor<F16>,
        _weight: &Tensor<F16>,
        _eps: f32,
        n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::Rmsnorm { n_tokens });
        Ok(Self::dummy_tensor())
    }

    fn residual_add(
        &mut self,
        _a: Tensor<F16>,
        _b: Tensor<F16>,
        n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::ResidualAdd { n_tokens });
        Ok(Self::dummy_tensor())
    }

    fn scale_inplace_f16(
        &mut self,
        _buf: Tensor<F16>,
        _scale: f32,
        n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        self.ops_called.push(OpCall::ScaleInplace { n_tokens });
        Ok(Self::dummy_tensor())
    }

    fn standard_attn(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &AttnWeights,
        layer_idx: usize,
        positions: &[usize],
        slot_ids: &[usize],
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>> {
        self.ops_called.push(OpCall::StandardAttn {
            layer_idx,
            positions: positions.to_vec(),
            slot_ids: slot_ids.to_vec(),
            next_norm: next_norm.is_some(),
        });
        Ok(Some(Self::dummy_tensor()))
    }

    fn gdn_layer(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &GdnWeights,
        layer_idx: usize,
        slot_ids: &[usize],
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>> {
        self.ops_called.push(OpCall::GdnLayer {
            layer_idx,
            slot_ids: slot_ids.to_vec(),
            next_norm: next_norm.is_some(),
        });
        Ok(Some(Self::dummy_tensor()))
    }

    fn dense_ffn(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &FfnWeights,
        n_tokens: usize,
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>> {
        self.ops_called.push(OpCall::DenseFfn {
            n_tokens,
            next_norm: next_norm.is_some(),
        });
        Ok(Some(Self::dummy_tensor()))
    }

    fn moe_ffn(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &MoeWeights,
        n_tokens: usize,
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>> {
        self.ops_called.push(OpCall::MoeFfn {
            n_tokens,
            next_norm: next_norm.is_some(),
        });
        Ok(Some(Self::dummy_tensor()))
    }

    fn output_head(
        &mut self,
        _input: &Tensor<F16>,
        _lm_head: &LmHeadWeights,
        slot_ids: &[usize],
    ) -> Result<()> {
        self.ops_called.push(OpCall::OutputHead {
            slot_ids: slot_ids.to_vec(),
        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_basic_op_sequence() {
        let mut ctx = RecordingCtx::new().with_layers([0, 1]);
        let embd = EmbeddingWeights::placeholder(32, 8);
        let lm_head = LmHeadWeights::placeholder(32, 8, 1e-5);
        let layout = ModelLayout {
            num_layers: 2,
            hidden: 8,
            kv_max_seq_len: 64,
        };

        let resid = ctx.embed(&embd, &[1u32]).unwrap();
        let layers: Vec<usize> = ctx.layer_range(&layout).collect();
        assert_eq!(layers, vec![0, 1]);
        ctx.output_head(&resid, &lm_head, &[0]).unwrap();

        assert_eq!(
            ctx.ops_called,
            vec![
                OpCall::Embed { tokens: vec![1] },
                OpCall::OutputHead {
                    slot_ids: vec![0]
                },
            ]
        );
    }
}
