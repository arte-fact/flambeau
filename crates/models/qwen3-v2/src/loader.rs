//! GGUF → device-side weight handles.
//!
//! Walks the qwen3 tensor table and produces the per-layer `AttnWeights`
//! + `FfnWeights` + the top-level `EmbeddingWeights` / `LmHeadWeights`
//! that `flambeau-forward` composites consume.
//!
//! Norm weights ship as F32 in qwen3 GGUFs; the loader dequantises on
//! host (a no-op for F32) and casts to F16 before HtoD upload, since
//! the model-ops norm kernels expect F16 weights.
//!
//! `token_embd.weight` ships as Q8_0 in most qwen3 builds. The loader
//! dequantises the full table to F16 once at load (the device-side
//! `embed` composite does a single F16 row DtoD memcpy). When the LM
//! head is tied (`output.weight` absent), the same Q8_0 bytes are also
//! uploaded as a `QuantWeight::Q8_0` LM head — two device copies of
//! the same data, ~475 MB total at qwen3-0.6B / Q8_0. Larger qwen3
//! checkpoints can opt into a host-roundtrip embed path later.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, LmHeadWeights, ModelLayout,
    QuantWeight,
};
use flambeau_model_ops::{Tensor, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};
use flambeau_quant::{GgmlDType, GgufFile};
use half::f16;

use crate::config::Qwen3V2Config;

/// Loaded qwen3 model on device. Holds every weight + a record of every
/// allocation so `drop` can release them.
pub struct Qwen3V2Model {
    pub config: Qwen3V2Config,
    pub layout: ModelLayout,
    pub embedding: EmbeddingWeights,
    pub attn: Vec<AttnWeights>,
    pub ffn: Vec<FfnWeights>,
    pub lm_head: LmHeadWeights,

    /// Every (ptr, bytes) returned by `device.alloc` during loading.
    /// `dispose` walks this once.
    allocs: Vec<(DevicePtr, usize)>,
    device_id: i32,
}

impl Qwen3V2Model {
    /// Release every device buffer. Idempotent — repeated calls drain
    /// the allocs list and become no-ops.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if device.default_stream().device_id() != self.device_id {
            bail!(
                "Qwen3V2Model::dispose: device mismatch (model on {}, called on {})",
                self.device_id,
                device.default_stream().device_id()
            );
        }
        for (ptr, bytes) in self.allocs.drain(..) {
            // SAFETY: ptr produced by `device.alloc(bytes)` below;
            // not freed elsewhere, not aliased after the field reads
            // (the caller invariant for `dispose` is "no further
            // forward calls").
            unsafe { device.dealloc(ptr, bytes) }
                .with_context(|| format!("dealloc {bytes} bytes"))?;
        }
        Ok(())
    }
}

/// Read a tensor's raw bytes, allocate device storage of the same size,
/// and HtoD upload. Returns the device pointer for caller-side wrapping.
fn upload_raw(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<DevicePtr> {
    let bytes = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let len = bytes.len();
    let ptr = device.alloc(len).with_context(|| format!("alloc {name}"))?;
    let stream = device.default_stream();
    // SAFETY: ptr owns `len` bytes; bytes has `len` host bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(bytes.as_ptr() as usize),
            len,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after upload")?;
    allocs.push((ptr, len));
    Ok(ptr)
}

/// Dequant a tensor to F32 on host, cast to F16, upload as F16.
/// Used for norm weights (F32 → F16) and for the embedding table when
/// the GGUF stores it quantised (Q8_0 → F16 in the qwen3 case).
fn upload_dequant_to_f16(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    expected_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F16>> {
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != expected_elems {
        bail!(
            "loader: {name} dequant produced {} elems, expected {}",
            f32_vec.len(),
            expected_elems
        );
    }
    let f16_vec: Vec<f16> = f32_vec.iter().map(|&v| f16::from_f32(v)).collect();
    let bytes = f16_vec.len() * 2;
    let ptr = device
        .alloc(bytes)
        .with_context(|| format!("alloc {name} F16"))?;
    let stream = device.default_stream();
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(f16_vec.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F16>::from_raw(ptr, expected_elems) })
}

/// Upload a quant tensor's raw bytes and wrap in `QuantWeight`.
fn upload_quant_weight(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let info = file
        .info(name)
        .with_context(|| format!("tensor info {name}"))?;
    let ptr = upload_raw(file, device, name, allocs)?;
    let qw = match info.dtype {
        GgmlDType::Q4_0 => QuantWeight::Q4_0(unsafe { Tensor::<Q4_0>::from_raw(ptr, n_elems) }),
        GgmlDType::Q4_1 => QuantWeight::Q4_1(unsafe { Tensor::<Q4_1>::from_raw(ptr, n_elems) }),
        GgmlDType::Q5_0 => QuantWeight::Q5_0(unsafe { Tensor::<Q5_0>::from_raw(ptr, n_elems) }),
        GgmlDType::Q5_1 => QuantWeight::Q5_1(unsafe { Tensor::<Q5_1>::from_raw(ptr, n_elems) }),
        GgmlDType::Q8_0 => QuantWeight::Q8_0(unsafe { Tensor::<Q8_0>::from_raw(ptr, n_elems) }),
        other => bail!(
            "loader: {name} dtype {other:?} not in model-ops's qmatmul set (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0); \
             K-quant + IQ + MXFP4 wrappers in model-ops land in a later phase"
        ),
    };
    Ok(qw)
}

/// Load a qwen3 GGUF onto `device`.
pub fn load_from_gguf(file: &GgufFile, device: &HipDevice) -> Result<Qwen3V2Model> {
    let config = Qwen3V2Config::from_gguf(file).context("parse qwen3 config")?;
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();

    let hidden = config.hidden;
    let q_width = config.n_heads * config.head_dim;
    let kv_width = config.n_kv_heads * config.head_dim;
    let m = config.intermediate;
    let v = config.vocab_size;

    // Embedding: dequant Q8_0 → F16 row table.
    let token_embd =
        upload_dequant_to_f16(file, device, "token_embd.weight", v * hidden, &mut allocs)?;
    let embedding = EmbeddingWeights {
        token_embd,
        vocab_size: v,
        hidden,
    };

    // Per-layer attention + FFN.
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

    // LM head: tied or separate.
    let output_norm = upload_dequant_to_f16(
        file,
        device,
        "output_norm.weight",
        hidden,
        &mut allocs,
    )?;
    let lm_head_quant = if config.tied_lm_head {
        // Tied: upload `token_embd.weight` again as Q8_0 (separate device
        // copy because the F16 dequant above isn't a `QuantWeight`).
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
