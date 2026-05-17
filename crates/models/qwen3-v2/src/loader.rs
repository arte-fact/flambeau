//! qwen3 GGUF → device weight handles. Knows tensor names + layout;
//! all upload primitives come from `flambeau_forward::loader`.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, LmHeadWeights, ModelLayout,
};
use flambeau_forward::loader::{upload_dequant_to_f16, upload_quant_weight};
use flambeau_quant::GgufFile;

use crate::config::Qwen3V2Config;

/// Same struct for SD + TP-sharded loaders — only the field values
/// (sharded vs full weights) differ.
pub struct Qwen3V2Model {
    pub config: Qwen3V2Config,
    pub layout: ModelLayout,
    pub embedding: EmbeddingWeights,
    pub attn: Vec<AttnWeights>,
    pub ffn: Vec<FfnWeights>,
    pub lm_head: LmHeadWeights,

    pub(crate) allocs: Vec<(DevicePtr, usize)>,
    pub(crate) device_id: i32,
}

impl Qwen3V2Model {
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if device.default_stream().device_id() != self.device_id {
            bail!(
                "Qwen3V2Model::dispose: device mismatch (model on {}, called on {})",
                self.device_id,
                device.default_stream().device_id()
            );
        }
        for (ptr, bytes) in self.allocs.drain(..) {
            // SAFETY: ptr returned by `device.alloc(bytes)` in the loader.
            unsafe { device.dealloc(ptr, bytes) }
                .with_context(|| format!("dealloc {bytes} bytes"))?;
        }
        Ok(())
    }
}

/// Full (non-sharded) weights for SingleDevice/PP.
pub fn load_from_gguf(file: &GgufFile, device: &HipDevice) -> Result<Qwen3V2Model> {
    let config = Qwen3V2Config::from_gguf(file).context("parse qwen3 config")?;
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();

    let hidden = config.hidden;
    let q_width = config.n_heads * config.head_dim;
    let kv_width = config.n_kv_heads * config.head_dim;
    let m = config.intermediate;
    let v = config.vocab_size;

    let token_embd =
        upload_dequant_to_f16(file, device, "token_embd.weight", v * hidden, &mut allocs)?;
    let embedding = EmbeddingWeights {
        token_embd,
        vocab_size: v,
        hidden,
    };

    let mut attn: Vec<AttnWeights> = Vec::with_capacity(config.num_layers);
    let mut ffn: Vec<FfnWeights> = Vec::with_capacity(config.num_layers);
    for li in 0..config.num_layers {
        let prefix = format!("blk.{li}");

        let attn_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{prefix}.attn_norm.weight"),
            hidden,
            &mut allocs,
        )?;
        let attn_q = upload_quant_weight(
            file,
            device,
            &format!("{prefix}.attn_q.weight"),
            q_width * hidden,
            &mut allocs,
        )?;
        let attn_k = upload_quant_weight(
            file,
            device,
            &format!("{prefix}.attn_k.weight"),
            kv_width * hidden,
            &mut allocs,
        )?;
        let attn_v = upload_quant_weight(
            file,
            device,
            &format!("{prefix}.attn_v.weight"),
            kv_width * hidden,
            &mut allocs,
        )?;
        let attn_output = upload_quant_weight(
            file,
            device,
            &format!("{prefix}.attn_output.weight"),
            hidden * q_width,
            &mut allocs,
        )?;
        let attn_q_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{prefix}.attn_q_norm.weight"),
            config.head_dim,
            &mut allocs,
        )
        .ok();
        let attn_k_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{prefix}.attn_k_norm.weight"),
            config.head_dim,
            &mut allocs,
        )
        .ok();
        attn.push(AttnWeights {
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm,
            attn_k_norm,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            rotated_dims: config.rotated_dims,
            rope_theta: config.rope_theta,
            window_size: 0,
            rms_eps: config.rms_eps,
            softmax_scale: None,
        });

        let ffn_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{prefix}.ffn_norm.weight"),
            hidden,
            &mut allocs,
        )?;
        let ffn_gate = upload_quant_weight(
            file,
            device,
            &format!("{prefix}.ffn_gate.weight"),
            m * hidden,
            &mut allocs,
        )?;
        let ffn_up = upload_quant_weight(
            file,
            device,
            &format!("{prefix}.ffn_up.weight"),
            m * hidden,
            &mut allocs,
        )?;
        let ffn_down = upload_quant_weight(
            file,
            device,
            &format!("{prefix}.ffn_down.weight"),
            hidden * m,
            &mut allocs,
        )?;
        ffn.push(FfnWeights {
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            activation: Activation::SwiGLU,
            rms_eps: config.rms_eps,
        });
    }

    let output_norm = upload_dequant_to_f16(
        file,
        device,
        "output_norm.weight",
        hidden,
        &mut allocs,
    )?;
    let lm_head_quant = if config.tied_lm_head {
        upload_quant_weight(file, device, "token_embd.weight", v * hidden, &mut allocs)?
    } else {
        upload_quant_weight(file, device, "output.weight", v * hidden, &mut allocs)?
    };
    let lm_head = LmHeadWeights {
        output_norm,
        lm_head: lm_head_quant,
        final_logit_softcap: None,
        vocab_size: v,
        hidden,
        rms_eps: config.rms_eps,
    };

    let layout = ModelLayout {
        num_layers: config.num_layers,
        hidden,
        kv_max_seq_len: config.context_length,
    };

    let device_id = device.default_stream().device_id();
    Ok(Qwen3V2Model {
        config,
        layout,
        embedding,
        attn,
        ffn,
        lm_head,
        allocs,
        device_id,
    })
}
