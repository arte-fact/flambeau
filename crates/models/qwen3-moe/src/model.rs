//! `Qwen3MoEModel` — config + device weights, the read-only half of
//! inference state. A model is loaded once per rank and used across many
//! requests; each active request owns a `Qwen3MoESession` alongside it.
//! Forward passes (`forward_one_token`, `forward_prefill`) are not yet
//! wired — a lands only the load + teardown scaffold so later
//! chunks (`b`..`e`) can focus on kernel composition.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_ops::hip::{HipDevice, OpsRegistry};
use flambeau_quant::GgufFile;

use crate::config::Qwen3MoEConfig;
use crate::layout::ModelLayout;
use crate::weights::ModelWeights;

/// Loaded Qwen3.x MoE model. Owns the op registry + device weights.
/// Created via [`Qwen3MoEModel::load`]. Call [`Qwen3MoEModel::dispose`]
/// before the `HipDevice` is reclaimed — otherwise the inner weights leak
/// and you'll see a warn on drop.
pub struct Qwen3MoEModel {
    pub config: Qwen3MoEConfig,
    pub layout: ModelLayout,
    pub weights: ModelWeights,
    pub ops: OpsRegistry,
}

impl Qwen3MoEModel {
    /// Parse config + tensor layout from `file`, allocate + upload all
    /// weights to `device`, and build a kernel registry ready for forward
    /// passes. Synchronises the device before returning.
    pub fn load(file: &GgufFile, device: &HipDevice) -> Result<Self> {
        let config = Qwen3MoEConfig::from_gguf(file)?;
        let layout = ModelLayout::from_gguf(file, &config)?;
        let weights = ModelWeights::upload(file, &layout, device)?;
        let ops = OpsRegistry::new(device)
            .map_err(|e| anyhow::anyhow!("OpsRegistry init: {e}"))?;
        Ok(Self {
            config,
            layout,
            weights,
            ops,
        })
    }

    /// Free device weights. Mandatory — see [`ModelWeights::dispose`].
    pub fn dispose(self, device: &HipDevice) -> Result<()> {
        let Self { weights, .. } = self;
        weights.dispose(device)
    }
}
