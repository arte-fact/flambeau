//! qwen35 GGUF → device. Per-layer dispatch on `is_recurrent` decides
//! whether to upload `FullAttn` (attn_q/k/v/output + q/k norms) or
//! `Gdn` (attn_qkv/attn_gate/ssm_alpha/beta/out + ssm_dt_bias/a/conv1d/norm).

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, GdnWeights, LayerKind, LmHeadWeights,
    ModelLayout,
};
use flambeau_forward::loader::{upload_dequant_to_f16, upload_quant_weight};
use flambeau_model_ops::{Tensor, F32};
use flambeau_quant::GgufFile;

use crate::config::Qwen35V2Config;

pub struct Qwen35V2Model {
    pub config: Qwen35V2Config,
    pub layout: ModelLayout,
    pub embedding: EmbeddingWeights,
    /// Per-layer attention family selector.
    pub layer_kinds: Vec<LayerKind>,
    /// `full_attn[li] = Some` iff `layer_kinds[li] == FullAttn`.
    pub full_attn: Vec<Option<AttnWeights>>,
    /// `gdn[li] = Some` iff `layer_kinds[li] == Gdn`.
    pub gdn: Vec<Option<GdnWeights>>,
    pub ffn: Vec<FfnWeights>,
    pub lm_head: LmHeadWeights,

    pub(crate) allocs: Vec<(DevicePtr, usize)>,
    pub(crate) device_id: i32,
}

impl Qwen35V2Model {
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if device.default_stream().device_id() != self.device_id {
            bail!(
                "Qwen35V2Model::dispose: device mismatch (model on {}, called on {})",
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

/// HtoD-upload a host F32 slice as `Tensor<F32>` (no dequant —
/// GDN's `ssm_dt_bias` / `ssm_a` / `ssm_conv1d` are stored F32 on disk).
fn upload_f32_tensor(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    expected_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F32>> {
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != expected_elems {
        bail!(
            "loader: {name} dequant produced {} elems, expected {expected_elems}",
            f32_vec.len()
        );
    }
    let bytes = f32_vec.len() * 4;
    let ptr = device.alloc(bytes).context("alloc F32 tensor")?;
    let stream = device.default_stream();
    use flambeau_core::CopyDirection;
    // SAFETY: ptr owns bytes; f32_vec has matching bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(f32_vec.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after F32 upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F32>::from_raw(ptr, f32_vec.len()) })
}

pub fn load_from_gguf(file: &GgufFile, device: &HipDevice) -> Result<Qwen35V2Model> {
    let config = Qwen35V2Config::from_gguf(file).context("parse qwen35 config")?;
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();

    let hidden = config.hidden;
    let q_width = config.n_heads * config.head_dim;
    let kv_width = config.n_kv_heads * config.head_dim;
    let m = config.intermediate;
    let v = config.vocab_size;
    let g = config.gdn;

    let token_embd =
        upload_dequant_to_f16(file, device, "token_embd.weight", v * hidden, &mut allocs)?;
    let embedding = EmbeddingWeights {
        token_embd,
        vocab_size: v,
        hidden,
    };

    let mut layer_kinds: Vec<LayerKind> = Vec::with_capacity(config.num_layers);
    let mut full_attn: Vec<Option<AttnWeights>> = Vec::with_capacity(config.num_layers);
    let mut gdn: Vec<Option<GdnWeights>> = Vec::with_capacity(config.num_layers);
    let mut ffn: Vec<FfnWeights> = Vec::with_capacity(config.num_layers);

    for li in 0..config.num_layers {
        let p = format!("blk.{li}");

        // qwen35 uses `post_attention_norm` as the pre-FFN norm on hybrid arches.
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
            activation: Activation::SwiGLU,
            rms_eps: config.rms_eps,
        });

        let attn_norm = upload_dequant_to_f16(
            file,
            device,
            &format!("{p}.attn_norm.weight"),
            hidden,
            &mut allocs,
        )?;

        if config.is_recurrent(li) {
            // GDN layer.
            let attn_qkv = upload_quant_weight(
                file,
                device,
                &format!("{p}.attn_qkv.weight"),
                g.conv_channels * hidden,
                &mut allocs,
            )?;
            let attn_gate = upload_quant_weight(
                file,
                device,
                &format!("{p}.attn_gate.weight"),
                g.d_inner * hidden,
                &mut allocs,
            )?;
            let ssm_alpha = upload_quant_weight(
                file,
                device,
                &format!("{p}.ssm_alpha.weight"),
                g.num_v_heads * hidden,
                &mut allocs,
            )?;
            let ssm_beta = upload_quant_weight(
                file,
                device,
                &format!("{p}.ssm_beta.weight"),
                g.num_v_heads * hidden,
                &mut allocs,
            )?;
            let ssm_out = upload_quant_weight(
                file,
                device,
                &format!("{p}.ssm_out.weight"),
                hidden * g.d_inner,
                &mut allocs,
            )?;
            let ssm_dt_bias =
                upload_f32_tensor(file, device, &format!("{p}.ssm_dt.bias"), g.num_v_heads, &mut allocs)?;
            let ssm_a =
                upload_f32_tensor(file, device, &format!("{p}.ssm_a"), g.num_v_heads, &mut allocs)?;
            let ssm_conv1d = upload_f32_tensor(
                file,
                device,
                &format!("{p}.ssm_conv1d.weight"),
                g.conv_kernel * g.conv_channels,
                &mut allocs,
            )?;
            let ssm_norm_w = upload_dequant_to_f16(
                file,
                device,
                &format!("{p}.ssm_norm.weight"),
                g.head_v_dim,
                &mut allocs,
            )?;

            full_attn.push(None);
            gdn.push(Some(GdnWeights {
                attn_norm,
                attn_qkv,
                attn_gate,
                ssm_alpha,
                ssm_beta,
                ssm_out,
                ssm_dt_bias,
                ssm_a,
                ssm_conv1d,
                ssm_norm_w,
                dims: g,
                rms_eps: config.rms_eps,
                rep_inner_layout: false,
            }));
            layer_kinds.push(LayerKind::Gdn);
        } else {
            // Full-attention layer.
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
            let attn_v = upload_quant_weight(
                file,
                device,
                &format!("{p}.attn_v.weight"),
                kv_width * hidden,
                &mut allocs,
            )?;
            let attn_output = upload_quant_weight(
                file,
                device,
                &format!("{p}.attn_output.weight"),
                hidden * q_width,
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
            full_attn.push(Some(AttnWeights {
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
            }));
            gdn.push(None);
            layer_kinds.push(LayerKind::FullAttn);
        }
    }

    let output_norm =
        upload_dequant_to_f16(file, device, "output_norm.weight", hidden, &mut allocs)?;
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
    Ok(Qwen35V2Model {
        config,
        layout,
        embedding,
        layer_kinds,
        full_attn,
        gdn,
        ffn,
        lm_head,
        allocs,
        device_id,
    })
}
