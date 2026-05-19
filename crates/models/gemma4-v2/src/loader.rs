//! Gemma-4 GGUF → device. Per-layer SWA alternation; every layer is
//! FullAttn. Dense variant (31B) populates `ffn`; MoE variant
//! (26B-A4B) populates `moe` with the routed experts + shared MLP
//! sibling (gemma4 MoE layer = routed MoE + dense MLP in parallel,
//! one pre-norm, one post-norm). Single entry parametrised by
//! `ShardMode`.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, LmHeadWeights, ModelLayout,
    MoeWeights, SharedExpertWeights,
};
use flambeau_forward::loader::{
    load_dense_attn_layer, load_dense_ffn_layer, load_embedding, load_lm_head,
    upload_col_sharded_quant, upload_dequant_to_f16, upload_f32_tensor,
    upload_gemma4_pre_router_weight_f16, upload_moe_experts_fused_gate_up_stacked,
    upload_moe_experts_stacked_row_sharded, upload_quant_weight, upload_row_sharded_quant,
    upload_router_f16, DenseAttnLayerSpec, DenseFfnLayerSpec, EmbeddingSpec, LmHeadSpec,
    ShardMode,
};
use flambeau_quant::GgufFile;

use crate::config::Gemma4V2Config;

pub struct Gemma4V2Model {
    pub config: Gemma4V2Config,
    pub layout: ModelLayout,
    pub embedding: EmbeddingWeights,
    pub attn: Vec<Option<AttnWeights>>,
    /// Dense FFN per layer for the 31B / 9B variants. Empty Vec when
    /// the model is MoE (use `moe` instead).
    pub ffn: Vec<Option<FfnWeights>>,
    /// Routed MoE + shared MLP per layer for the 26B-A4B variant.
    /// Empty Vec for dense variants.
    pub moe: Vec<Option<MoeWeights>>,
    pub lm_head: LmHeadWeights,
    /// Per-layer `blk.N.layer_output_scale.weight` F32 scalar applied
    /// to the residual after both attn + FFN residuals. `None` when
    /// the tensor is absent on disk for that layer (older gemma4
    /// variants might skip it; main-line gemma4 ships one per layer).
    pub layer_output_scale: Vec<Option<f32>>,

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

fn load_with_shard(
    file: &GgufFile,
    device: &HipDevice,
    shard: ShardMode,
    layer_range: Option<(usize, usize)>,
    ctx_cap: Option<usize>,
) -> Result<Gemma4V2Model> {
    let mut config = Gemma4V2Config::from_gguf(file).context("parse gemma4 config")?;
    if let Some(cap) = ctx_cap {
        if cap > 0 && cap < config.context_length {
            config.context_length = cap;
        }
    }
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();
    let owns_embed = layer_range.map_or(true, |(s, _)| s == 0);
    let owns_lm_head = layer_range.map_or(true, |(_, e)| e == config.num_layers);
    let in_range = |li: usize| -> bool {
        layer_range.map_or(true, |(s, e)| li >= s && li < e)
    };

    // Gemma4: inpL *= sqrt(n_embd) post-embed.
    let embedding = if owns_embed {
        load_embedding(
            file,
            device,
            &EmbeddingSpec {
                token_embd_name: "token_embd.weight",
                vocab_size: config.vocab_size,
                hidden: config.hidden,
                post_scale: Some((config.hidden as f32).sqrt()),
            },
            &mut allocs,
        )?
    } else {
        let mut placeholder = EmbeddingWeights::placeholder(config.vocab_size, config.hidden);
        placeholder.post_scale = Some((config.hidden as f32).sqrt());
        placeholder
    };

    let is_moe = config.moe.is_some();
    let mut attn = Vec::with_capacity(config.num_layers);
    let mut ffn: Vec<Option<FfnWeights>> =
        if is_moe { Vec::new() } else { Vec::with_capacity(config.num_layers) };
    let mut moe_layers: Vec<Option<MoeWeights>> =
        if is_moe { Vec::with_capacity(config.num_layers) } else { Vec::new() };
    let mut layer_output_scale: Vec<Option<f32>> = Vec::with_capacity(config.num_layers);
    for li in 0..config.num_layers {
        if !in_range(li) {
            attn.push(None);
            if is_moe {
                moe_layers.push(None);
            } else {
                ffn.push(None);
            }
            layer_output_scale.push(None);
            continue;
        }
        let p = format!("blk.{li}");
        let dims = config.attn[li];
        let n_kv_heads = config.num_kv_heads[li];

        let (norm, post_attn_norm, q, k, v, output, q_norm, k_norm) = (
            format!("{p}.attn_norm.weight"),
            format!("{p}.post_attention_norm.weight"),
            format!("{p}.attn_q.weight"),
            format!("{p}.attn_k.weight"),
            format!("{p}.attn_v.weight"),
            format!("{p}.attn_output.weight"),
            format!("{p}.attn_q_norm.weight"),
            format!("{p}.attn_k_norm.weight"),
        );
        // Gemma4's `shared_kv_layers` (E2B/E4B variants) makes the
        // tail-N layers reuse a shared V. Probe tensor existence to
        // decide per-layer; 31B / 26B-A4B / 9B have shared_kv_layers=0
        // so every layer has its own attn_v.weight.
        let has_attn_v = file.info(&v).is_ok();
        attn.push(Some(load_dense_attn_layer(
            file,
            device,
            &DenseAttnLayerSpec {
                attn_norm_name: &norm,
                post_attn_norm_name: Some(&post_attn_norm),
                attn_q_name: &q,
                attn_k_name: &k,
                attn_v_name: if has_attn_v { Some(&v) } else { None },
                attn_output_name: &output,
                attn_q_norm_name: Some(&q_norm),
                attn_k_norm_name: Some(&k_norm),
                attn_v_unit_norm: true,
                n_heads: config.num_heads,
                n_kv_heads,
                head_dim: dims.head_dim,
                hidden: config.hidden,
                rotated_dims: dims.rotated_dims,
                rope_theta: dims.rope_theta,
                rope_variant: flambeau_forward::ctx::RopeVariant::Interleaved,
                window_size: dims.window_size,
                rms_eps: config.rms_eps,
                // Gemma4: softmax_scale = 1.0, not 1/sqrt(head_dim).
                softmax_scale: Some(1.0),
                attn_q_gated: false,
            },
            shard,
            &mut allocs,
        )?));

        let (ffn_norm_name, post_ffn_norm_name, ffn_gate, ffn_up, ffn_down) = (
            format!("{p}.ffn_norm.weight"),
            format!("{p}.post_ffw_norm.weight"),
            format!("{p}.ffn_gate.weight"),
            format!("{p}.ffn_up.weight"),
            format!("{p}.ffn_down.weight"),
        );
        if let Some(mdims) = config.moe {
            // Gemma4 MoE: routed experts + shared dense MLP per layer.
            // `ffn_norm` is shared by both paths; `post_ffw_norm` is
            // applied to the summed delta before the residual_add.
            let router_name = format!("{p}.ffn_gate_inp.weight");
            let gate_up_exps = format!("{p}.ffn_gate_up_exps.weight");
            let down_exps = format!("{p}.ffn_down_exps.weight");
            let ffn_norm_t = upload_dequant_to_f16(
                file,
                device,
                &ffn_norm_name,
                config.hidden,
                &mut allocs,
            )?;
            // Final F16 post-norm (legacy keeps both F16 and F32 copies;
            // v2 cascade uses the F32 variant, but we keep the F16
            // tensor too for the qwen-path fallback / future use).
            let post_ffn_norm_t = upload_dequant_to_f16(
                file,
                device,
                &post_ffn_norm_name,
                config.hidden,
                &mut allocs,
            )?;
            // F32 cascade norms (kept as F32 on device for the
            // F32-precision rmsnorm steps in the gemma4 MoE path).
            let post_ffn_norm_f32_name = post_ffn_norm_name.clone();
            let post_ffn_norm_f32_t = upload_f32_tensor(
                file,
                device,
                &post_ffn_norm_f32_name,
                config.hidden,
                &mut allocs,
            )?;
            let pre_ffw_norm_2_name = format!("{p}.pre_ffw_norm_2.weight");
            let pre_ffw_norm_2_f16 = upload_dequant_to_f16(
                file,
                device,
                &pre_ffw_norm_2_name,
                config.hidden,
                &mut allocs,
            )?;
            let post_ffw_norm_1_name = format!("{p}.post_ffw_norm_1.weight");
            let post_ffw_norm_1_f32 = upload_f32_tensor(
                file,
                device,
                &post_ffw_norm_1_name,
                config.hidden,
                &mut allocs,
            )?;
            let post_ffw_norm_2_name = format!("{p}.post_ffw_norm_2.weight");
            let post_ffw_norm_2_f32 = upload_f32_tensor(
                file,
                device,
                &post_ffw_norm_2_name,
                config.hidden,
                &mut allocs,
            )?;
            // pre_router_weight is derived from `ffn_gate_inp.scale`
            // (F32 [hidden]) × 1/sqrt(hidden) → F16 [hidden].
            let ffn_gate_inp_scale_name = format!("{p}.ffn_gate_inp.scale");
            let pre_router_weight_f16 = upload_gemma4_pre_router_weight_f16(
                file,
                device,
                &ffn_gate_inp_scale_name,
                config.hidden,
                &mut allocs,
            )?;
            // Per-expert F32 scale array [n_experts] folded into the
            // router top-k weights.
            let expert_down_scale_name = format!("{p}.ffn_down_exps.scale");
            let expert_down_scale_f32 = upload_f32_tensor(
                file,
                device,
                &expert_down_scale_name,
                mdims.num_experts,
                &mut allocs,
            )?;
            let router = upload_router_f16(
                file,
                device,
                &router_name,
                mdims.num_experts * config.hidden,
                &mut allocs,
            )?;
            // gate_up_exps: `[n_experts, 2*moe_inter, hidden]` fused on
            // disk → split into two stacked tensors.
            let (experts_gate, experts_up) = upload_moe_experts_fused_gate_up_stacked(
                file,
                device,
                &gate_up_exps,
                mdims.num_experts,
                mdims.moe_intermediate,
                config.hidden,
                &mut allocs,
            )?;
            // down_exps: `[n_experts, hidden, moe_inter]` — row-shard
            // the inner moe_inter dim for TP.
            let experts_down = upload_moe_experts_stacked_row_sharded(
                file,
                device,
                &down_exps,
                mdims.num_experts,
                config.hidden,
                mdims.moe_intermediate,
                shard,
                &mut allocs,
            )?;
            // Shared MLP: dense FFN at `config.intermediate`, summed
            // into the routed MoE delta. Same TP shard plan as the
            // qwen35moe shared expert.
            let inter = config.intermediate;
            let (gate_sh, up_sh, down_sh) = match shard {
                ShardMode::Replicated => (
                    upload_quant_weight(file, device, &ffn_gate, inter * config.hidden, &mut allocs)?,
                    upload_quant_weight(file, device, &ffn_up, inter * config.hidden, &mut allocs)?,
                    upload_quant_weight(file, device, &ffn_down, config.hidden * inter, &mut allocs)?,
                ),
                ShardMode::Tp { rank, n_ranks } => (
                    upload_col_sharded_quant(file, device, &ffn_gate, inter, config.hidden, rank, n_ranks, &mut allocs)?,
                    upload_col_sharded_quant(file, device, &ffn_up, inter, config.hidden, rank, n_ranks, &mut allocs)?,
                    upload_row_sharded_quant(file, device, &ffn_down, config.hidden, inter, rank, n_ranks, &mut allocs)?,
                ),
            };
            let shared = Some(SharedExpertWeights {
                gate: gate_sh,
                up: up_sh,
                down: down_sh,
                gate_inp: None,
                intermediate: inter / shard.n_ranks(),
            });
            moe_layers.push(Some(MoeWeights {
                ffn_norm: ffn_norm_t,
                post_ffn_norm: Some(post_ffn_norm_t),
                router,
                experts_gate,
                experts_up,
                experts_down,
                n_experts: mdims.num_experts,
                experts_per_tok: mdims.experts_per_tok,
                activation: Activation::GeluTanh,
                rms_eps: config.rms_eps,
                shared,
                pre_router_weight_f16: Some(pre_router_weight_f16),
                pre_ffw_norm_2_f16: Some(pre_ffw_norm_2_f16),
                post_ffw_norm_1_f32: Some(post_ffw_norm_1_f32),
                post_ffw_norm_2_f32: Some(post_ffw_norm_2_f32),
                post_ffn_norm_f32: Some(post_ffn_norm_f32_t),
                expert_down_scale_f32: Some(expert_down_scale_f32),
            }));
        } else {
            ffn.push(Some(load_dense_ffn_layer(
                file,
                device,
                &DenseFfnLayerSpec {
                    ffn_norm_name: &ffn_norm_name,
                    post_ffn_norm_name: Some(&post_ffn_norm_name),
                    ffn_gate_name: &ffn_gate,
                    ffn_up_name: &ffn_up,
                    ffn_down_name: &ffn_down,
                    hidden: config.hidden,
                    intermediate: config.intermediate,
                    activation: Activation::GeluTanh,
                    rms_eps: config.rms_eps,
                },
                shard,
                &mut allocs,
            )?));
        }

        // Per-layer F32 scalar applied after both residuals (gemma4
        // trained behavior). Tensor is `[1]` F32 on disk; read raw +
        // reinterpret. Absence = leave as None (legacy treats absent
        // as scale=1).
        let scale_name = format!("{p}.layer_output_scale.weight");
        let scale = if file.info(&scale_name).is_ok() {
            let raw = file
                .tensor_raw(&scale_name)
                .with_context(|| format!("read {scale_name}"))?;
            if raw.len() < 4 {
                bail!("{scale_name}: raw len {} < 4", raw.len());
            }
            Some(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
        } else {
            None
        };
        layer_output_scale.push(scale);
    }

    let lm_head = if owns_lm_head {
        let lm_head_name = if config.tied_lm_head {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        let final_logit_softcap = if config.final_logit_softcap > 0.0 {
            Some(config.final_logit_softcap)
        } else {
            None
        };
        load_lm_head(
            file,
            device,
            &LmHeadSpec {
                output_norm_name: "output_norm.weight",
                lm_head_name,
                vocab_size: config.vocab_size,
                hidden: config.hidden,
                rms_eps: config.rms_eps,
                final_logit_softcap,
            },
            &mut allocs,
        )?
    } else {
        LmHeadWeights::placeholder(config.vocab_size, config.hidden, config.rms_eps)
    };

    let layout = ModelLayout {
        num_layers: config.num_layers,
        hidden: config.hidden,
        kv_max_seq_len: config.context_length,
    };
    let device_id = device.default_stream().device_id();
    Ok(Gemma4V2Model {
        config,
        layout,
        embedding,
        attn,
        ffn,
        moe: moe_layers,
        lm_head,
        layer_output_scale,
        allocs,
        device_id,
    })
}

pub fn load_from_gguf(
    file: &GgufFile,
    device: &HipDevice,
    layer_range: Option<(usize, usize)>,
    ctx_cap: Option<usize>,
) -> Result<Gemma4V2Model> {
    load_with_shard(file, device, ShardMode::Replicated, layer_range, ctx_cap)
}

pub fn load_tp_shard_from_gguf(
    file: &GgufFile,
    device: &HipDevice,
    rank: usize,
    n_ranks: usize,
    layer_range: Option<(usize, usize)>,
    ctx_cap: Option<usize>,
) -> Result<Gemma4V2Model> {
    load_with_shard(
        file,
        device,
        ShardMode::Tp { rank, n_ranks },
        layer_range,
        ctx_cap,
    )
}
