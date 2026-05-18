//! TP loader: col-shard Q/K/V/gate/up (dim-0), row-shard
//! attn_output/ffn_down (dim-1); norm + embedding + LM head replicated.
//! Sharding primitives live in `flambeau_forward::loader`.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, LmHeadWeights, ModelLayout,
};
use flambeau_forward::loader::{
    upload_col_sharded_quant, upload_dequant_to_f16, upload_quant_weight,
    upload_row_sharded_quant,
};
use flambeau_quant::GgufFile;

use crate::config::Qwen3V2Config;
use crate::loader::Qwen3V2Model;

/// Load this rank's TP shard. `n_heads` / `n_kv_heads` / `intermediate`
/// must each be divisible by `n_ranks`.
pub fn load_tp_shard_from_gguf(
    file: &GgufFile,
    device: &HipDevice,
    rank: usize,
    n_ranks: usize,
) -> Result<Qwen3V2Model> {
    let config = Qwen3V2Config::from_gguf(file).context("parse qwen3 config")?;
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();

    if rank >= n_ranks {
        bail!("tp_shard: rank {rank} >= n_ranks {n_ranks}");
    }
    if config.n_heads % n_ranks != 0 || config.n_kv_heads % n_ranks != 0 {
        bail!(
            "tp_shard: n_heads {} and n_kv_heads {} must both be divisible by n_ranks {n_ranks}",
            config.n_heads,
            config.n_kv_heads
        );
    }
    if config.intermediate % n_ranks != 0 {
        bail!(
            "tp_shard: intermediate {} not divisible by n_ranks {n_ranks}",
            config.intermediate
        );
    }

    let hidden = config.hidden;
    let q_width = config.n_heads * config.head_dim;
    let kv_width = config.n_kv_heads * config.head_dim;
    let m = config.intermediate;
    let v = config.vocab_size;
    let n_heads_local = config.n_heads / n_ranks;
    let n_kv_heads_local = config.n_kv_heads / n_ranks;

    // Embedding + LM head: replicated.
    let token_embd =
        upload_dequant_to_f16(file, device, "token_embd.weight", v * hidden, &mut allocs)?;
    let embedding = EmbeddingWeights {
        token_embd,
        vocab_size: v,
        hidden,
        post_scale: None,
    };

    let mut attn: Vec<AttnWeights> = Vec::with_capacity(config.num_layers);
    let mut ffn: Vec<FfnWeights> = Vec::with_capacity(config.num_layers);
    for li in 0..config.num_layers {
        let p = format!("blk.{li}");

        let attn_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{p}.attn_norm.weight"),
            hidden,
            &mut allocs,
        )?;
        let attn_q = upload_col_sharded_quant(
            file,
            device,
            &format!("{p}.attn_q.weight"),
            q_width,
            hidden,
            rank,
            n_ranks,
            &mut allocs,
        )?;
        let attn_k = upload_col_sharded_quant(
            file,
            device,
            &format!("{p}.attn_k.weight"),
            kv_width,
            hidden,
            rank,
            n_ranks,
            &mut allocs,
        )?;
        let attn_v = upload_col_sharded_quant(
            file,
            device,
            &format!("{p}.attn_v.weight"),
            kv_width,
            hidden,
            rank,
            n_ranks,
            &mut allocs,
        )?;
        let attn_output = upload_row_sharded_quant(
            file,
            device,
            &format!("{p}.attn_output.weight"),
            hidden,
            q_width,
            rank,
            n_ranks,
            &mut allocs,
        )?;
        let attn_q_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{p}.attn_q_norm.weight"),
            config.head_dim,
            &mut allocs,
        )
        .ok();
        let attn_k_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{p}.attn_k_norm.weight"),
            config.head_dim,
            &mut allocs,
        )
        .ok();
        attn.push(AttnWeights {
            attn_norm,
            attn_q,
            attn_k,
            attn_v: Some(attn_v),
            attn_output,
            attn_q_norm,
            attn_k_norm,
            n_heads: n_heads_local,
            n_kv_heads: n_kv_heads_local,
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
            &format!("{p}.ffn_norm.weight"),
            hidden,
            &mut allocs,
        )?;
        let ffn_gate = upload_col_sharded_quant(
            file,
            device,
            &format!("{p}.ffn_gate.weight"),
            m,
            hidden,
            rank,
            n_ranks,
            &mut allocs,
        )?;
        let ffn_up = upload_col_sharded_quant(
            file,
            device,
            &format!("{p}.ffn_up.weight"),
            m,
            hidden,
            rank,
            n_ranks,
            &mut allocs,
        )?;
        let ffn_down = upload_row_sharded_quant(
            file,
            device,
            &format!("{p}.ffn_down.weight"),
            hidden,
            m,
            rank,
            n_ranks,
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
