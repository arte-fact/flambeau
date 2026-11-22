//! Dense FFN loader. Under `ShardMode::Tp`, gate / up are col-sharded
//! along the intermediate axis; down is row-sharded along its input
//! intermediate.

use anyhow::{bail, Result};
use flambeau_core::{Device, DevicePtr};
use flambeau_quant::GgufFile;

use crate::ctx::{Activation, FfnWeights};

use super::primitives::upload_dequant_to_f16;
use super::shard::{upload_col, upload_row};
use super::ShardMode;

pub struct DenseFfnLayerSpec<'a> {
    pub ffn_norm_name: &'a str,
    /// Optional name of the F16 norm tensor applied to the FFN delta
    /// BEFORE the outer residual add. Gemma4 sets this to
    /// `post_ffw_norm.weight`; other arches pass `None`.
    pub post_ffn_norm_name: Option<&'a str>,
    pub ffn_gate_name: &'a str,
    pub ffn_up_name: &'a str,
    pub ffn_down_name: &'a str,
    pub hidden: usize,
    pub intermediate: usize,
    pub activation: Activation,
    pub rms_eps: f32,
}

pub fn load_dense_ffn_layer(
    file: &GgufFile,
    device: &impl Device,
    spec: &DenseFfnLayerSpec,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<FfnWeights> {
    let n_ranks = shard.n_ranks();
    if spec.intermediate % n_ranks != 0 {
        bail!(
            "dense_ffn: intermediate {} not divisible by n_ranks {n_ranks}",
            spec.intermediate
        );
    }
    let ffn_norm = upload_dequant_to_f16(file, device, spec.ffn_norm_name, spec.hidden, allocs)?;
    let post_ffn_norm = spec
        .post_ffn_norm_name
        .map(|n| upload_dequant_to_f16(file, device, n, spec.hidden, allocs))
        .transpose()?;
    let ffn_gate = upload_col(
        file,
        device,
        spec.ffn_gate_name,
        spec.intermediate,
        spec.hidden,
        shard,
        allocs,
    )?;
    let ffn_up = upload_col(
        file,
        device,
        spec.ffn_up_name,
        spec.intermediate,
        spec.hidden,
        shard,
        allocs,
    )?;
    let ffn_down = upload_row(
        file,
        device,
        spec.ffn_down_name,
        spec.hidden,
        spec.intermediate,
        shard,
        allocs,
    )?;
    Ok(FfnWeights {
        ffn_norm,
        post_ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        activation: spec.activation,
        rms_eps: spec.rms_eps,
    })
}
