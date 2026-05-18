//! Gemma-4 dense GGUF → device. Every layer is FullAttn (no GDN/MoE
//! in this first cut); per-layer SWA dims come from `config.attn[li]`.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, LmHeadWeights, ModelLayout,
};
use flambeau_forward::loader::{upload_dequant_to_f16, upload_quant_weight};
use flambeau_quant::GgufFile;

use crate::config::Gemma4V2Config;

pub struct Gemma4V2Model {
    pub config: Gemma4V2Config,
    pub layout: ModelLayout,
    pub embedding: EmbeddingWeights,
    pub attn: Vec<AttnWeights>,
    pub ffn: Vec<FfnWeights>,
    pub lm_head: LmHeadWeights,

    pub(crate) allocs: Vec<(DevicePtr, usize)>,
    pub(crate) device_id: i32,
}

impl Gemma4V2Model {
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if device.default_stream().device_id() != self.device_id {
            bail!(
                "Gemma4V2Model::dispose: device mismatch (model on {}, called on {})",
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

pub fn load_from_gguf(file: &GgufFile, device: &HipDevice) -> Result<Gemma4V2Model> {
    let config = Gemma4V2Config::from_gguf(file).context("parse gemma4 config")?;
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();

    let hidden = config.hidden;
    let m = config.intermediate;
    let v = config.vocab_size;

    let token_embd =
        upload_dequant_to_f16(file, device, "token_embd.weight", v * hidden, &mut allocs)?;
    // Gemma4 multiplies the embedding row by `sqrt(n_embd)` before
    // the first layer (memory: `parity_vs_argmax_in_vocab`).
    let post_scale = Some((hidden as f32).sqrt());
    let embedding = EmbeddingWeights {
        token_embd,
        vocab_size: v,
        hidden,
        post_scale,
    };

    let mut attn: Vec<AttnWeights> = Vec::with_capacity(config.num_layers);
    let mut ffn: Vec<FfnWeights> = Vec::with_capacity(config.num_layers);
    for li in 0..config.num_layers {
        let p = format!("blk.{li}");
        let dims = config.attn[li];
        let n_kv_heads = config.num_kv_heads[li];
        let q_width = config.num_heads * dims.head_dim;
        let kv_width = n_kv_heads * dims.head_dim;

        let attn_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{p}.attn_norm.weight"),
            hidden,
            &mut allocs,
        )?;
        let attn_q = upload_quant_weight(
            file,
            device,
            &format!("{p}.attn_q.weight"),
            q_width * hidden,
            &mut allocs,
        )?;
        let attn_k = upload_quant_weight(
            file,
            device,
            &format!("{p}.attn_k.weight"),
            kv_width * hidden,
            &mut allocs,
        )?;
        // Gemma4: no attn_v on disk — V is K verbatim (composite branches).
        let attn_v: Option<flambeau_forward::ctx::QuantWeight> = None;
        let attn_output = upload_quant_weight(
            file,
            device,
            &format!("{p}.attn_output.weight"),
            hidden * q_width,
            &mut allocs,
        )?;
        // Gemma4 carries per-layer q_norm + k_norm (head_dim length).
        let attn_q_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{p}.attn_q_norm.weight"),
            dims.head_dim,
            &mut allocs,
        )
        .ok();
        let attn_k_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{p}.attn_k_norm.weight"),
            dims.head_dim,
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
            n_heads: config.num_heads,
            n_kv_heads,
            head_dim: dims.head_dim,
            rotated_dims: dims.rotated_dims,
            rope_theta: dims.rope_theta,
            window_size: dims.window_size,
            rms_eps: config.rms_eps,
            // Gemma4 uses softmax_scale = 1.0 (not 1/sqrt(head_dim)).
            // Memory: `gemma4_attn_output_proj_f16_saturate` explains the
            // unit-scale convention.
            softmax_scale: Some(1.0),
        });

        // Gemma4 has post-attention + post-FFN norms (extra rmsnorms).
        // For the first cut we use `post_attention_norm` as the pre-FFN
        // norm; the additional post-residual norms are TODO if outputs
        // diverge from oracle.
        let ffn_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{p}.post_attention_norm.weight"),
            hidden,
            &mut allocs,
        )?;
        let ffn_gate = upload_quant_weight(
            file,
            device,
            &format!("{p}.ffn_gate.weight"),
            m * hidden,
            &mut allocs,
        )?;
        let ffn_up = upload_quant_weight(
            file,
            device,
            &format!("{p}.ffn_up.weight"),
            m * hidden,
            &mut allocs,
        )?;
        let ffn_down = upload_quant_weight(
            file,
            device,
            &format!("{p}.ffn_down.weight"),
            hidden * m,
            &mut allocs,
        )?;
        ffn.push(FfnWeights {
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            activation: Activation::GeluTanh,
            rms_eps: config.rms_eps,
        });
    }

    let output_norm =
        upload_dequant_to_f16(file, device, "output_norm.weight", hidden, &mut allocs)?;
    let lm_head_quant = if config.tied_lm_head {
        upload_quant_weight(file, device, "token_embd.weight", v * hidden, &mut allocs)?
    } else {
        upload_quant_weight(file, device, "output.weight", v * hidden, &mut allocs)?
    };
    let final_logit_softcap = if config.final_logit_softcap > 0.0 {
        Some(config.final_logit_softcap)
    } else {
        None
    };
    let lm_head = LmHeadWeights {
        output_norm,
        lm_head: lm_head_quant,
        final_logit_softcap,
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
    Ok(Gemma4V2Model {
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
