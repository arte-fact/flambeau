//! MTP-3 — Multi-token-prediction head loader for Qwen3.6-27B.
//!
//! Reads the 15 mtp.* tensors from a thin sibling GGUF emitted by
//! `tools/convert_qwen36_mtp.py` (or from the integrated form where the
//! mtp.* tensors live in the same file as the base). Holds them on a
//! single device — the LM-head rank in PP/TP topologies.
//!
//! Forward composition is intentionally kept out of this module: it
//! lives next to the existing `forward/full_attn.rs` once the activation-
//! precision plumbing is in place (MTP-3.5).
//!
//! Pairing semantics:
//!   1. base GGUF carries `mtp.*` tensors → use those (integrated mode).
//!   2. else look for `<basename>-mtp.gguf` next to the base, verify
//!      `mtp.target_arch` / `mtp.target_hidden_size` /
//!      `mtp.target_vocab_size` against base.
//!   3. mismatch → fail loud.
//!   4. neither → load without MTP (caller must handle Option<None>).

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_quant::{GgmlDType, GgufFile};

use crate::layout::ResolvedTensor;
use crate::weights::DeviceTensor;

/// Names of the 15 MTP tensors that must be present.
pub const MTP_TENSOR_NAMES: &[&str] = &[
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
];

/// All MTP-head weights for a Qwen3.6-27B base model. Lives on one
/// device — typically the LM-head rank.
#[derive(Debug)]
pub struct MtpHeadWeights {
    /// Concat([norm_h, norm_e]) → hidden projection. Shape `[hidden, 2*hidden]`.
    pub fc: DeviceTensor,
    /// Final norm before LM head. Shape `[hidden]` F32.
    pub norm: DeviceTensor,
    /// Norm of base hidden state h_t. Shape `[hidden]` F32.
    pub pre_fc_norm_hidden: DeviceTensor,
    /// Norm of next-token embedding e_{t+1}. Shape `[hidden]` F32.
    pub pre_fc_norm_embedding: DeviceTensor,

    /// MTP transformer block (single layer; `mtp_num_hidden_layers=1`).
    pub block: MtpBlockWeights,
}

/// Weights for the single transformer block inside the MTP head. Looks
/// like a standard pre-norm Qwen3 block with two notable details:
/// 1. `q_proj` is `[hidden, 2*num_q_heads*head_dim]` because the
///    output is `[Q ‖ gate]` (gated attention; `output_gate_type =
///    "swish"` per Qwen3.6 config).
/// 2. There is no `attn_gate` weight — the gate signal lives in the
///    second half of `q_proj`'s output.
#[derive(Debug)]
pub struct MtpBlockWeights {
    pub input_layernorm: DeviceTensor,
    pub post_attention_layernorm: DeviceTensor,
    pub q_proj: DeviceTensor,    // [in=hidden, out=2*num_q_heads*head_dim]
    pub k_proj: DeviceTensor,    // [in=hidden, out=num_kv_heads*head_dim]
    pub v_proj: DeviceTensor,    // [in=hidden, out=num_kv_heads*head_dim]
    pub o_proj: DeviceTensor,    // [in=num_q_heads*head_dim, out=hidden]
    pub q_norm: DeviceTensor,    // [head_dim] F32 — per-head norm
    pub k_norm: DeviceTensor,    // [head_dim] F32
    pub gate_proj: DeviceTensor, // [in=hidden, out=intermediate]
    pub up_proj: DeviceTensor,   // [in=hidden, out=intermediate]
    pub down_proj: DeviceTensor, // [in=intermediate, out=hidden]
}

impl MtpHeadWeights {
    /// Total bytes uploaded to the device.
    pub fn total_bytes(&self) -> usize {
        let b = &self.block;
        self.fc.bytes
            + self.norm.bytes
            + self.pre_fc_norm_hidden.bytes
            + self.pre_fc_norm_embedding.bytes
            + b.input_layernorm.bytes
            + b.post_attention_layernorm.bytes
            + b.q_proj.bytes
            + b.k_proj.bytes
            + b.v_proj.bytes
            + b.o_proj.bytes
            + b.q_norm.bytes
            + b.k_norm.bytes
            + b.gate_proj.bytes
            + b.up_proj.bytes
            + b.down_proj.bytes
    }
}

/// Verify the pair-up metadata in `mtp_file` matches the base config.
///
/// The thin sibling format stamps `mtp.target_arch`,
/// `mtp.target_hidden_size`, and `mtp.target_vocab_size`. The
/// integrated form does NOT stamp these (the base IS the target by
/// definition); we skip the check there.
pub fn verify_pairing(
    mtp_file: &GgufFile,
    base_arch: &str,
    base_hidden_size: usize,
    base_vocab_size: usize,
) -> Result<()> {
    // For thin-mode files: `general.architecture == "qwen35-mtp"` (sentinel).
    let mtp_arch = mtp_file.metadata_str("general.architecture").unwrap_or("");
    if mtp_arch != format!("{base_arch}-mtp") && mtp_arch != base_arch {
        bail!(
            "MTP file arch `{mtp_arch}` does not match base `{base_arch}` \
             (expected `{base_arch}-mtp` for thin sibling, or `{base_arch}` for integrated)"
        );
    }

    // Pairing keys (only present in thin sibling files).
    if let Some(target_arch) = mtp_file.metadata_str("mtp.target_arch") {
        if target_arch != base_arch {
            bail!(
                "MTP target_arch `{target_arch}` mismatches base arch `{base_arch}` — \
                 re-run convert_qwen36_mtp.py with the matching base"
            );
        }
    }
    if let Some(target_hidden) = mtp_file.metadata_u32("mtp.target_hidden_size") {
        if target_hidden as usize != base_hidden_size {
            bail!(
                "MTP target_hidden_size {target_hidden} != base hidden_size {base_hidden_size}"
            );
        }
    }
    if let Some(target_vocab) = mtp_file.metadata_u32("mtp.target_vocab_size") {
        if target_vocab as usize != base_vocab_size {
            bail!(
                "MTP target_vocab_size {target_vocab} != base vocab_size {base_vocab_size}"
            );
        }
    }
    Ok(())
}

/// Resolve a `mtp.*` tensor from a GGUF file. Errors if missing.
fn resolve_mtp_tensor(file: &GgufFile, name: &str) -> Result<ResolvedTensor> {
    let info = file
        .tensors
        .get(name)
        .ok_or_else(|| anyhow!("MTP tensor `{name}` not present in GGUF"))?;
    Ok(ResolvedTensor {
        name: name.to_string(),
        dtype: info.dtype,
        dims: info.dims.clone(),
        size_bytes: info.size_in_bytes(),
    })
}

/// Upload one MTP tensor to `device`. Linears (Q8_0) go up as-is.
/// Norms (F32 1D in the converter output) are CAST to F16 before
/// upload — flambeau's `rmsnorm_f16` / `rmsnorm_quant_q8_1` kernels
/// expect F16 weight, matching the existing convention used by base
/// model's `output_norm` / `attn_norm` etc.
fn upload_mtp_tensor(
    file: &GgufFile,
    name: &str,
    device: &HipDevice,
) -> Result<DeviceTensor> {
    let r = resolve_mtp_tensor(file, name)?;
    if !matches!(r.dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::Q8_0) {
        bail!("MTP tensor `{name}` has unexpected dtype {:?}", r.dtype);
    }
    use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

    // Norms (F32) → cast to F16 on host before upload so downstream
    // ops can use them with F16 inputs.
    if r.dtype == GgmlDType::F32 {
        let raw = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw `{name}`"))?;
        let n_elems = (r.size_bytes / 4) as usize;
        if raw.len() < n_elems * 4 {
            bail!(
                "MTP tensor `{name}` F32 mmap slice {} < {n_elems}*4 B",
                raw.len()
            );
        }
        // SAFETY: tensor_raw guarantees the slice is large enough; the
        // tensor's GGUF dtype declares F32, so the bytes are F32 LE.
        let f32_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_elems)
        };
        let f16_buf: Vec<half::f16> = f32_slice
            .iter()
            .map(|x| half::f16::from_f32(*x))
            .collect();
        let bytes = n_elems * 2;
        let ptr = device
            .alloc(bytes)
            .map_err(|e| anyhow!("hipMalloc {bytes} B `{name}` (F32→F16): {e}"))?;
        // SAFETY: `ptr` is a fresh device alloc of `bytes`; `f16_buf`
        // owns `n_elems * 2` host bytes for the duration of memcpy +
        // sync below.
        unsafe {
            device
                .memcpy_async(
                    device.default_stream(),
                    CopyDirection::HostToDevice,
                    ptr,
                    DevicePtr(f16_buf.as_ptr() as usize),
                    bytes,
                )
                .map_err(|e| anyhow!("memcpy `{name}` (F32→F16): {e}"))?;
        }
        device
            .default_stream()
            .synchronize()
            .map_err(|e| anyhow!("stream sync after `{name}` (F32→F16): {e}"))?;
        return Ok(DeviceTensor {
            ptr,
            dtype: GgmlDType::F16,
            dims: r.dims,
            bytes,
            name: std::sync::Arc::from(name),
        });
    }

    // Q8_0 (and any other non-F32) → upload verbatim.
    let bytes = r.size_bytes as usize;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw `{name}`"))?;
    if raw.len() < bytes {
        bail!(
            "MTP tensor `{name}` mmap slice {} < declared {bytes}",
            raw.len()
        );
    }
    let ptr = device
        .alloc(bytes)
        .map_err(|e| anyhow!("hipMalloc {bytes} B `{name}`: {e}"))?;
    // SAFETY: `ptr` is a fresh device alloc of `bytes`; `raw` is a
    // host mmap view of at least `bytes` bytes.
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(raw.as_ptr() as usize),
                bytes,
            )
            .map_err(|e| anyhow!("memcpy `{name}`: {e}"))?;
    }
    device
        .default_stream()
        .synchronize()
        .map_err(|e| anyhow!("stream sync after `{name}`: {e}"))?;
    Ok(DeviceTensor {
        ptr,
        dtype: r.dtype,
        dims: r.dims,
        bytes,
        name: std::sync::Arc::from(name),
    })
}

/// MTP-4-C-6: BF16 sibling of `upload_mtp_tensor` for the linear
/// weights. F32 (norms) → F16 unchanged; F16 (linears) → BF16 via
/// host-side cast. Q8_0 linears would need a dequant pass first and
/// are rejected — re-run the converter with `MTP_LINEAR_DTYPE=f16`
/// (the default since MTP-4-C-1) for BF16 forward.
fn upload_mtp_tensor_bf16(
    file: &GgufFile,
    name: &str,
    device: &HipDevice,
) -> Result<DeviceTensor> {
    let r = resolve_mtp_tensor(file, name)?;
    use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

    if r.dtype == GgmlDType::F32 {
        // Norm — same F32 → F16 cast as the F16 path; rmsnorm_bf16
        // takes F16 weight (decided in MTP-4-C-3).
        return upload_mtp_tensor(file, name, device);
    }

    if r.dtype == GgmlDType::F16 {
        let raw = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw `{name}`"))?;
        let n_elems = (r.size_bytes / 2) as usize;
        if raw.len() < n_elems * 2 {
            bail!(
                "MTP tensor `{name}` F16 mmap slice {} < {n_elems}*2 B",
                raw.len()
            );
        }
        // SAFETY: tensor_raw covers n_elems * 2 bytes; GGUF dtype declares F16.
        let f16_slice: &[half::f16] = unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const half::f16, n_elems)
        };
        // Round-trip via F32 — half::bf16::from_f32 is RNE.
        let bf16_buf: Vec<half::bf16> = f16_slice
            .iter()
            .map(|x| half::bf16::from_f32(x.to_f32()))
            .collect();
        let bytes = n_elems * 2;
        let ptr = device
            .alloc(bytes)
            .map_err(|e| anyhow!("hipMalloc {bytes} B `{name}` (F16→BF16): {e}"))?;
        // SAFETY: ptr is a fresh alloc of `bytes`; bf16_buf lives
        // through the synchronize() below.
        unsafe {
            device
                .memcpy_async(
                    device.default_stream(),
                    CopyDirection::HostToDevice,
                    ptr,
                    DevicePtr(bf16_buf.as_ptr() as usize),
                    bytes,
                )
                .map_err(|e| anyhow!("memcpy `{name}` (F16→BF16): {e}"))?;
        }
        device
            .default_stream()
            .synchronize()
            .map_err(|e| anyhow!("stream sync after `{name}` (F16→BF16): {e}"))?;
        return Ok(DeviceTensor {
            ptr,
            dtype: GgmlDType::BF16,
            dims: r.dims,
            bytes,
            name: std::sync::Arc::from(name),
        });
    }

    bail!(
        "MTP linear `{name}` dtype {:?} not supported by load_mtp_head_bf16 \
         (need F16 — re-run convert_qwen36_mtp.py with MTP_LINEAR_DTYPE=f16)",
        r.dtype
    );
}

/// MTP-4-C-6: load MTP head with BF16 linear weights. Norms stay
/// F16 (matches `rmsnorm_bf16`'s F16-weight signature).
pub fn load_mtp_head_bf16(
    mtp_file: &GgufFile,
    device: &HipDevice,
) -> Result<MtpHeadWeights> {
    device.bind()?;
    let fc = upload_mtp_tensor_bf16(mtp_file, "mtp.fc.weight", device)?;
    let norm = upload_mtp_tensor_bf16(mtp_file, "mtp.norm.weight", device)?;
    let pre_fc_norm_hidden =
        upload_mtp_tensor_bf16(mtp_file, "mtp.pre_fc_norm_hidden.weight", device)?;
    let pre_fc_norm_embedding =
        upload_mtp_tensor_bf16(mtp_file, "mtp.pre_fc_norm_embedding.weight", device)?;

    let block = MtpBlockWeights {
        input_layernorm: upload_mtp_tensor_bf16(
            mtp_file,
            "mtp.layers.0.input_layernorm.weight",
            device,
        )?,
        post_attention_layernorm: upload_mtp_tensor_bf16(
            mtp_file,
            "mtp.layers.0.post_attention_layernorm.weight",
            device,
        )?,
        q_proj: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.self_attn.q_proj.weight", device)?,
        k_proj: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.self_attn.k_proj.weight", device)?,
        v_proj: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.self_attn.v_proj.weight", device)?,
        o_proj: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.self_attn.o_proj.weight", device)?,
        q_norm: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.self_attn.q_norm.weight", device)?,
        k_norm: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.self_attn.k_norm.weight", device)?,
        gate_proj: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.mlp.gate_proj.weight", device)?,
        up_proj: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.mlp.up_proj.weight", device)?,
        down_proj: upload_mtp_tensor_bf16(mtp_file, "mtp.layers.0.mlp.down_proj.weight", device)?,
    };

    Ok(MtpHeadWeights {
        fc,
        norm,
        pre_fc_norm_hidden,
        pre_fc_norm_embedding,
        block,
    })
}

/// Load all 15 MTP tensors from `mtp_file` to `device`.
///
/// The caller is responsible for validating `mtp_file`'s pairing
/// metadata against the base config (use `verify_pairing`).
pub fn load_mtp_head(mtp_file: &GgufFile, device: &HipDevice) -> Result<MtpHeadWeights> {
    device.bind()?;
    let fc = upload_mtp_tensor(mtp_file, "mtp.fc.weight", device)?;
    let norm = upload_mtp_tensor(mtp_file, "mtp.norm.weight", device)?;
    let pre_fc_norm_hidden =
        upload_mtp_tensor(mtp_file, "mtp.pre_fc_norm_hidden.weight", device)?;
    let pre_fc_norm_embedding =
        upload_mtp_tensor(mtp_file, "mtp.pre_fc_norm_embedding.weight", device)?;

    let block = MtpBlockWeights {
        input_layernorm: upload_mtp_tensor(
            mtp_file,
            "mtp.layers.0.input_layernorm.weight",
            device,
        )?,
        post_attention_layernorm: upload_mtp_tensor(
            mtp_file,
            "mtp.layers.0.post_attention_layernorm.weight",
            device,
        )?,
        q_proj: upload_mtp_tensor(mtp_file, "mtp.layers.0.self_attn.q_proj.weight", device)?,
        k_proj: upload_mtp_tensor(mtp_file, "mtp.layers.0.self_attn.k_proj.weight", device)?,
        v_proj: upload_mtp_tensor(mtp_file, "mtp.layers.0.self_attn.v_proj.weight", device)?,
        o_proj: upload_mtp_tensor(mtp_file, "mtp.layers.0.self_attn.o_proj.weight", device)?,
        q_norm: upload_mtp_tensor(mtp_file, "mtp.layers.0.self_attn.q_norm.weight", device)?,
        k_norm: upload_mtp_tensor(mtp_file, "mtp.layers.0.self_attn.k_norm.weight", device)?,
        gate_proj: upload_mtp_tensor(mtp_file, "mtp.layers.0.mlp.gate_proj.weight", device)?,
        up_proj: upload_mtp_tensor(mtp_file, "mtp.layers.0.mlp.up_proj.weight", device)?,
        down_proj: upload_mtp_tensor(mtp_file, "mtp.layers.0.mlp.down_proj.weight", device)?,
    };

    Ok(MtpHeadWeights {
        fc,
        norm,
        pre_fc_norm_hidden,
        pre_fc_norm_embedding,
        block,
    })
}

/// Optional persistent KV cache for accumulated MTP attention.
/// `kcache` / `vcache` are F16 `[max_tokens, n_kv_heads * head_dim]`
/// buffers owned by the caller. `cache_position` is where the
/// current step's K/V will be written; `n_tokens_kv` is the count
/// (1..=cache_position+1) the attention call should attend over.
///
/// vLLM/sglang's `spec_info.hidden_states` flow primes MTP's KV from
/// every prefill position; passive harnesses without priming get
/// ~0% acceptance because MTP's attention sees only the current
/// step. The caller is responsible for ensuring the prior positions
/// (0..cache_position) are populated, either via a prefill loop or
/// from a prior decode step's append.
#[derive(Clone, Copy, Debug)]
pub struct MtpKvCache {
    pub kcache: flambeau_core::DevicePtr,
    pub vcache: flambeau_core::DevicePtr,
    pub cache_position: usize,
    pub n_tokens_kv: usize,
}

/// Reusable per-call scratch for `forward_mtp_step` and
/// `forward_mtp_step_with_lm_head`.
///
/// MTP-4-CLEAN: previously each call did ~28 `hipMalloc` + `hipFree`
/// pairs. The struct is allocated once at session init and reused
/// for every MTP step. Sizes are derived from the model config
/// (`hidden_size`, `num_heads`, `num_kv_heads`, `head_dim`,
/// `moe_intermediate_size`, `vocab_size`) — re-allocate if any of
/// these change.
///
/// Buffers split into three groups:
///   1. step buffers — used by the body of `forward_mtp_step`
///   2. transient KV slot — used only when the caller passes
///      `kv = None` (legacy 1-slot path)
///   3. lm-head buffers — used only by `forward_mtp_step_with_lm_head`
#[derive(Debug)]
pub struct MtpForwardScratch {
    // ── Sizes (kept around for `dispose` so it doesn't need cfg).
    bytes_2h_f16: usize,
    bytes_h_f16: usize,
    bytes_h_f32: usize,
    bytes_qfull_f32: usize,
    bytes_qfull_f16: usize,
    bytes_qhalf_f16: usize,
    bytes_kv_f32: usize,
    bytes_kv_f16: usize,
    bytes_2h_q8_1: usize,
    bytes_h_q8_1: usize,
    bytes_qhalf_q8_1: usize,
    bytes_inter_f32: usize,
    bytes_inter_q8_1: usize,
    bytes_inter_bf16: usize,
    bytes_vocab_f32: usize,

    // ── Step buffers.
    pub fc_in_f16: flambeau_core::DevicePtr,
    pub fc_in_q8_1: flambeau_core::DevicePtr,
    pub h0_f32: flambeau_core::DevicePtr,
    pub h0_f16: flambeau_core::DevicePtr,
    pub h0n_q8_1: flambeau_core::DevicePtr,
    pub q_full_f32: flambeau_core::DevicePtr,
    pub q_full_f16: flambeau_core::DevicePtr,
    pub q_f16: flambeau_core::DevicePtr,
    pub gate_f16: flambeau_core::DevicePtr,
    pub k_f32: flambeau_core::DevicePtr,
    pub v_f32: flambeau_core::DevicePtr,
    pub k_f16: flambeau_core::DevicePtr,
    pub v_f16: flambeau_core::DevicePtr,
    /// Transient 1-slot KV (used when caller passes `kv = None`).
    pub kcache_slot_f16: flambeau_core::DevicePtr,
    pub vcache_slot_f16: flambeau_core::DevicePtr,
    pub attn_out_f16: flambeau_core::DevicePtr,
    pub gated_out_f16: flambeau_core::DevicePtr,
    pub gated_q8_1: flambeau_core::DevicePtr,
    pub attn_proj_f32: flambeau_core::DevicePtr,
    pub h1_f16: flambeau_core::DevicePtr,
    pub h1n_q8_1: flambeau_core::DevicePtr,
    pub gate_mlp_f32: flambeau_core::DevicePtr,
    pub up_mlp_f32: flambeau_core::DevicePtr,
    pub mlp_q8_1: flambeau_core::DevicePtr,
    /// MTP-4-C-6: BF16 staging buffer for the MLP `silu(gate)*up` →
    /// down_proj input on the BF16 forward path. Q8_1 buffer is too
    /// small (`inter * 1.125 B` vs BF16 needs `inter * 2 B`).
    pub mlp_bf16: flambeau_core::DevicePtr,
    pub down_f32: flambeau_core::DevicePtr,
    pub h2_f16: flambeau_core::DevicePtr,

    // ── lm-head wrapper buffers.
    pub h_t_post_norm: flambeau_core::DevicePtr,
    pub mtp_h_final: flambeau_core::DevicePtr,
    pub x_q8_1: flambeau_core::DevicePtr,
    pub logits_f32: flambeau_core::DevicePtr,
}

impl MtpForwardScratch {
    /// Allocate every scratch buffer needed for a forward MTP step
    /// against the given config.
    pub fn new(device: &HipDevice, cfg: &crate::Qwen3MoEConfig) -> Result<Self> {
        use flambeau_core::Device;

        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let n_q = cfg.num_heads;
        let n_kv = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let vocab = cfg.vocab_size;

        // Q8_1 block bytes: F16 d + F16 s + 32 int8 quants = 36 bytes.
        let q8_1_block_bytes = std::mem::size_of::<flambeau_quant::BlockQ8_1>();
        let q8_1_blocks = |n_elems: usize| n_elems / 32 * q8_1_block_bytes;

        let bytes_2h_f16 = 2 * h * 2;
        let bytes_h_f16 = h * 2;
        let bytes_h_f32 = h * 4;
        let bytes_qfull_f32 = 2 * n_q * head_dim * 4;
        let bytes_qfull_f16 = 2 * n_q * head_dim * 2;
        let bytes_qhalf_f16 = n_q * head_dim * 2;
        let bytes_kv_f32 = n_kv * head_dim * 4;
        let bytes_kv_f16 = n_kv * head_dim * 2;
        let bytes_2h_q8_1 = q8_1_blocks(2 * h);
        let bytes_h_q8_1 = q8_1_blocks(h);
        let bytes_qhalf_q8_1 = q8_1_blocks(n_q * head_dim);
        let bytes_inter_f32 = inter * 4;
        let bytes_inter_q8_1 = q8_1_blocks(inter);
        let bytes_inter_bf16 = inter * 2;
        let bytes_vocab_f32 = vocab * 4;

        let alloc = |bytes: usize| -> Result<flambeau_core::DevicePtr> {
            device
                .alloc(bytes)
                .map_err(|e| anyhow!("hipMalloc {bytes} B: {e}"))
        };

        Ok(Self {
            bytes_2h_f16,
            bytes_h_f16,
            bytes_h_f32,
            bytes_qfull_f32,
            bytes_qfull_f16,
            bytes_qhalf_f16,
            bytes_kv_f32,
            bytes_kv_f16,
            bytes_2h_q8_1,
            bytes_h_q8_1,
            bytes_qhalf_q8_1,
            bytes_inter_f32,
            bytes_inter_q8_1,
            bytes_inter_bf16,
            bytes_vocab_f32,

            fc_in_f16: alloc(bytes_2h_f16)?,
            fc_in_q8_1: alloc(bytes_2h_q8_1)?,
            h0_f32: alloc(bytes_h_f32)?,
            h0_f16: alloc(bytes_h_f16)?,
            h0n_q8_1: alloc(bytes_h_q8_1)?,
            q_full_f32: alloc(bytes_qfull_f32)?,
            q_full_f16: alloc(bytes_qfull_f16)?,
            q_f16: alloc(bytes_qhalf_f16)?,
            gate_f16: alloc(bytes_qhalf_f16)?,
            k_f32: alloc(bytes_kv_f32)?,
            v_f32: alloc(bytes_kv_f32)?,
            k_f16: alloc(bytes_kv_f16)?,
            v_f16: alloc(bytes_kv_f16)?,
            kcache_slot_f16: alloc(bytes_kv_f16)?,
            vcache_slot_f16: alloc(bytes_kv_f16)?,
            attn_out_f16: alloc(bytes_qhalf_f16)?,
            gated_out_f16: alloc(bytes_qhalf_f16)?,
            gated_q8_1: alloc(bytes_qhalf_q8_1)?,
            attn_proj_f32: alloc(bytes_h_f32)?,
            h1_f16: alloc(bytes_h_f16)?,
            h1n_q8_1: alloc(bytes_h_q8_1)?,
            gate_mlp_f32: alloc(bytes_inter_f32)?,
            up_mlp_f32: alloc(bytes_inter_f32)?,
            mlp_q8_1: alloc(bytes_inter_q8_1)?,
            mlp_bf16: alloc(bytes_inter_bf16)?,
            down_f32: alloc(bytes_h_f32)?,
            h2_f16: alloc(bytes_h_f16)?,

            h_t_post_norm: alloc(bytes_h_f16)?,
            mtp_h_final: alloc(bytes_h_f16)?,
            x_q8_1: alloc(bytes_h_q8_1)?,
            logits_f32: alloc(bytes_vocab_f32)?,
        })
    }

    /// Free every buffer. Consumes `self` so no use-after-free.
    pub fn dispose(self, device: &HipDevice) -> Result<()> {
        use flambeau_core::Device;
        // SAFETY: every pointer here was returned by `device.alloc`
        // in `Self::new` and has not been freed since; sizes match.
        unsafe {
            device.dealloc(self.fc_in_f16, self.bytes_2h_f16)?;
            device.dealloc(self.fc_in_q8_1, self.bytes_2h_q8_1)?;
            device.dealloc(self.h0_f32, self.bytes_h_f32)?;
            device.dealloc(self.h0_f16, self.bytes_h_f16)?;
            device.dealloc(self.h0n_q8_1, self.bytes_h_q8_1)?;
            device.dealloc(self.q_full_f32, self.bytes_qfull_f32)?;
            device.dealloc(self.q_full_f16, self.bytes_qfull_f16)?;
            device.dealloc(self.q_f16, self.bytes_qhalf_f16)?;
            device.dealloc(self.gate_f16, self.bytes_qhalf_f16)?;
            device.dealloc(self.k_f32, self.bytes_kv_f32)?;
            device.dealloc(self.v_f32, self.bytes_kv_f32)?;
            device.dealloc(self.k_f16, self.bytes_kv_f16)?;
            device.dealloc(self.v_f16, self.bytes_kv_f16)?;
            device.dealloc(self.kcache_slot_f16, self.bytes_kv_f16)?;
            device.dealloc(self.vcache_slot_f16, self.bytes_kv_f16)?;
            device.dealloc(self.attn_out_f16, self.bytes_qhalf_f16)?;
            device.dealloc(self.gated_out_f16, self.bytes_qhalf_f16)?;
            device.dealloc(self.gated_q8_1, self.bytes_qhalf_q8_1)?;
            device.dealloc(self.attn_proj_f32, self.bytes_h_f32)?;
            device.dealloc(self.h1_f16, self.bytes_h_f16)?;
            device.dealloc(self.h1n_q8_1, self.bytes_h_q8_1)?;
            device.dealloc(self.gate_mlp_f32, self.bytes_inter_f32)?;
            device.dealloc(self.up_mlp_f32, self.bytes_inter_f32)?;
            device.dealloc(self.mlp_q8_1, self.bytes_inter_q8_1)?;
            device.dealloc(self.mlp_bf16, self.bytes_inter_bf16)?;
            device.dealloc(self.down_f32, self.bytes_h_f32)?;
            device.dealloc(self.h2_f16, self.bytes_h_f16)?;

            device.dealloc(self.h_t_post_norm, self.bytes_h_f16)?;
            device.dealloc(self.mtp_h_final, self.bytes_h_f16)?;
            device.dealloc(self.x_q8_1, self.bytes_h_q8_1)?;
            device.dealloc(self.logits_f32, self.bytes_vocab_f32)?;
        }
        Ok(())
    }
}

/// MTP-3.5 forward — one MTP step on F16 hidden state.
///
/// Composes existing flambeau ops; no new kernels. Caller owns
/// `scratch` (one alloc per session via `MtpForwardScratch::new`)
/// and the optional persistent KV cache; passes `kv = None` for the
/// legacy transient-1-slot path.
///
/// Inputs:
///   `h_t` F16 [hidden]      — base model hidden (post `output_norm`)
///   `e_token` F16 [hidden]  — embedding of the token sampled from h_t
///   `position` usize        — base-model position (drives MROPE)
///   `kv` Option             — Some = persistent multi-slot cache
///                             (vLLM/sglang flow); None = transient
/// Output:
///   `h_final_out` F16 [hidden] — pre-LM-head MTP-block output. Caller
///                                runs the (shared) lm_head matmul to
///                                produce draft logits.
///
/// `position == 0` skips MROPE entirely (rotation is identity at
/// position 0 — useful for the parity test against the Python ref).
/// At `position > 0` we apply `rope_neox_partial_f16` with
/// `cfg.rope.rotated_dims` (matches base full-attn convention).
#[allow(clippy::too_many_arguments)]
pub fn forward_mtp_step(
    ops: &flambeau_ops::OpsRegistry,
    stream: &flambeau_backend_hip::HipStream,
    device: &HipDevice,
    cfg: &crate::Qwen3MoEConfig,
    mtp: &MtpHeadWeights,
    scratch: &MtpForwardScratch,
    h_t: flambeau_core::DevicePtr,
    e_token: flambeau_core::DevicePtr,
    position: usize,
    h_final_out: flambeau_core::DevicePtr,
    kv: Option<MtpKvCache>,
) -> Result<()> {
    use flambeau_core::{Device, DevicePtr, Stream};
    use flambeau_core::op::QDtype;
    use flambeau_ops::hip::attention::{attention_decode_f16_slots, split_q_gate_f16};
    use flambeau_ops::hip::cast::cast_f32_to_f16;
    use flambeau_ops::hip::mlp::{add_f32, sigmoid_mul_f16, swiglu_f32_to_q8_1};
    use flambeau_ops::hip::norm::{quantize_f16_q8_1, rmsnorm_f16, rmsnorm_quant_q8_1};
    use flambeau_ops::hip::qmatmul::mmvq;

    fn qdtype_for(w: &DeviceTensor) -> Result<QDtype> {
        Ok(match w.dtype {
            GgmlDType::Q8_0 => QDtype::Q8_0,
            GgmlDType::F16  => QDtype::F16,
            d => bail!("MTP linear `{}` has unsupported dtype {d:?}", w.name),
        })
    }

    let h = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size; // dense FFN size (set from feed_forward_length for qwen35 dense)
    let n_q = cfg.num_heads;
    let n_kv = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let eps = cfg.rms_norm_eps;
    let rope = &cfg.rope;

    let bytes_kv_f16 = scratch.bytes_kv_f16;

    let fc_in_f16 = scratch.fc_in_f16;
    let fc_in_q8_1 = scratch.fc_in_q8_1;
    let h0_f32 = scratch.h0_f32;
    let h0_f16 = scratch.h0_f16;
    let h0n_q8_1 = scratch.h0n_q8_1;
    let q_full_f32 = scratch.q_full_f32;
    let q_full_f16 = scratch.q_full_f16;
    let q_f16 = scratch.q_f16;
    let gate_f16 = scratch.gate_f16;
    let k_f32 = scratch.k_f32;
    let v_f32 = scratch.v_f32;
    let k_f16 = scratch.k_f16;
    let v_f16 = scratch.v_f16;
    let kcache_slot_f16 = scratch.kcache_slot_f16;
    let vcache_slot_f16 = scratch.vcache_slot_f16;
    let attn_out_f16 = scratch.attn_out_f16;
    let gated_out_f16 = scratch.gated_out_f16;
    let gated_q8_1 = scratch.gated_q8_1;
    let attn_proj_f32 = scratch.attn_proj_f32;
    let h1_f16 = scratch.h1_f16;
    let h1n_q8_1 = scratch.h1n_q8_1;
    let gate_mlp_f32 = scratch.gate_mlp_f32;
    let up_mlp_f32 = scratch.up_mlp_f32;
    let mlp_q8_1 = scratch.mlp_q8_1;
    let down_f32 = scratch.down_f32;
    let h2_f16 = scratch.h2_f16;

    // ── 2. Pre-FC norms + concat. Order is [embedding, hidden] per vLLM
    // Qwen3NextMTP source (concat happens at vllm/.../qwen3_next_mtp.py:86).
    // Reversing this order silently produces wrong projections — the
    // fc.weight rows expect [embedding | hidden] in that order.
    let fc_in_h_offset = fc_in_f16.offset_bytes(h * 2); // F16 = 2 B
    rmsnorm_f16(
        ops, stream, e_token, mtp.pre_fc_norm_embedding.ptr, fc_in_f16,
        1, h, eps,
    )
    .context("mtp pre_fc_norm_embedding (first half)")?;
    rmsnorm_f16(
        ops, stream, h_t, mtp.pre_fc_norm_hidden.ptr, fc_in_h_offset,
        1, h, eps,
    )
    .context("mtp pre_fc_norm_hidden (second half)")?;

    // ── 3. fc matmul (h0 = concat([norm_h, norm_e]) @ fc.T)
    quantize_f16_q8_1(ops, stream, fc_in_f16, fc_in_q8_1, 2 * h)
        .context("mtp fc_in → Q8_1")?;
    mmvq(
        ops, stream, mtp.fc.ptr, fc_in_q8_1, h0_f32,
        h, 2 * h, qdtype_for(&mtp.fc)?,
    )
    .context("mtp mmvq fc")?;
    cast_f32_to_f16(ops, stream, h0_f32, h0_f16, h)
        .context("mtp cast h0 → F16")?;

    // ── 4. Block input_layernorm + Q8_1 quantize (fused)
    rmsnorm_quant_q8_1(
        ops, stream, h0_f16, mtp.block.input_layernorm.ptr, h0n_q8_1,
        1, h, eps,
    )
    .context("mtp input_layernorm + quant")?;

    // ── 5. q/k/v projections from h0n_q8_1
    mmvq(
        ops, stream, mtp.block.q_proj.ptr, h0n_q8_1, q_full_f32,
        2 * n_q * head_dim, h, qdtype_for(&mtp.block.q_proj)?,
    )
    .context("mtp mmvq q_proj")?;
    cast_f32_to_f16(ops, stream, q_full_f32, q_full_f16, 2 * n_q * head_dim)
        .context("mtp cast q_full → F16")?;
    split_q_gate_f16(
        ops, stream, q_full_f16, q_f16, gate_f16,
        1, n_q, head_dim,
    )
    .context("mtp split q_gate")?;

    mmvq(
        ops, stream, mtp.block.k_proj.ptr, h0n_q8_1, k_f32,
        n_kv * head_dim, h, qdtype_for(&mtp.block.k_proj)?,
    )
    .context("mtp mmvq k_proj")?;
    cast_f32_to_f16(ops, stream, k_f32, k_f16, n_kv * head_dim)
        .context("mtp cast k → F16")?;
    mmvq(
        ops, stream, mtp.block.v_proj.ptr, h0n_q8_1, v_f32,
        n_kv * head_dim, h, qdtype_for(&mtp.block.v_proj)?,
    )
    .context("mtp mmvq v_proj")?;
    cast_f32_to_f16(ops, stream, v_f32, v_f16, n_kv * head_dim)
        .context("mtp cast v → F16")?;

    // ── 6. Per-head q_norm/k_norm
    rmsnorm_f16(
        ops, stream, q_f16, mtp.block.q_norm.ptr, q_f16,
        n_q, head_dim, eps,
    )
    .context("mtp q_norm")?;
    rmsnorm_f16(
        ops, stream, k_f16, mtp.block.k_norm.ptr, k_f16,
        n_kv, head_dim, eps,
    )
    .context("mtp k_norm")?;

    // ── 7. Partial RoPE (skipped at position=0)
    if position != 0 {
        // Stack-buffered position; sync after copy.
        let position_host: [i32; 1] = [position as i32];
        let position_dev = device
            .alloc(4)
            .map_err(|e| anyhow!("hipMalloc 4 B (position): {e}"))?;
        // SAFETY: position_dev is fresh alloc of 4 bytes; position_host
        // lives on stack across the synchronize() below.
        unsafe {
            device.memcpy_async(
                stream,
                flambeau_core::CopyDirection::HostToDevice,
                position_dev,
                DevicePtr(position_host.as_ptr() as usize),
                4,
            )?;
        }
        stream.synchronize()?;
        flambeau_ops::hip::pe::rope_neox_partial_f16(
            ops, stream, q_f16, position_dev, rope.freq_base,
            1, n_q, head_dim, rope.rotated_dims,
        )
        .context("mtp rope Q")?;
        flambeau_ops::hip::pe::rope_neox_partial_f16(
            ops, stream, k_f16, position_dev, rope.freq_base,
            1, n_kv, head_dim, rope.rotated_dims,
        )
        .context("mtp rope K")?;
        // SAFETY: position_dev is uniquely owned by this scope.
        unsafe { device.dealloc(position_dev, 4)?; }
    }

    // ── 8. Attention. Either:
    //   (transient KV branch, kv=None) — use the scratch 1-slot KV.
    //   (persistent KV branch, kv=Some) — append new K/V to
    //     caller-provided cache at `cache_position`, run with
    //     `n_tokens_kv` for accumulated history (this is what
    //     vLLM/sglang's spec_info flow does — primed by the
    //     prefill walk).
    use flambeau_core::CopyDirection;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let (k_buf, v_buf, n_tokens_kv) = if let Some(c) = kv {
        // Append new K/V to position `cache_position` in the caller's cache.
        let row_bytes = bytes_kv_f16; // n_kv * head_dim * 2
        let k_dst = c.kcache.offset_bytes(c.cache_position * row_bytes);
        let v_dst = c.vcache.offset_bytes(c.cache_position * row_bytes);
        // SAFETY: caller asserts cache size ≥ (cache_position+1)*row_bytes;
        // k_f16/v_f16 are scratch allocs of row_bytes.
        unsafe {
            device.memcpy_async(stream, CopyDirection::DeviceToDevice, k_dst, k_f16, row_bytes)?;
            device.memcpy_async(stream, CopyDirection::DeviceToDevice, v_dst, v_f16, row_bytes)?;
        }
        (c.kcache, c.vcache, c.n_tokens_kv)
    } else {
        // SAFETY: scratch slot KV buffers each hold n_kv*head_dim*2 B.
        unsafe {
            device.memcpy_async(stream, CopyDirection::DeviceToDevice, kcache_slot_f16, k_f16, bytes_kv_f16)?;
            device.memcpy_async(stream, CopyDirection::DeviceToDevice, vcache_slot_f16, v_f16, bytes_kv_f16)?;
        }
        (kcache_slot_f16, vcache_slot_f16, 1usize)
    };
    attention_decode_f16_slots(
        ops, stream, q_f16, k_buf, v_buf, attn_out_f16,
        n_q, n_kv, head_dim, n_tokens_kv, scale,
        /*n_tokens_kv_slot=*/ None,
    )
    .context("mtp attention_decode_f16")?;

    // ── 9. Output gate: sigmoid(gate) * attn_out (matches V1.7.4.b finding)
    sigmoid_mul_f16(
        ops, stream, gate_f16, attn_out_f16, gated_out_f16,
        n_q * head_dim,
    )
    .context("mtp sigmoid_mul")?;

    // ── 10. o_proj
    quantize_f16_q8_1(ops, stream, gated_out_f16, gated_q8_1, n_q * head_dim)
        .context("mtp gated → Q8_1")?;
    mmvq(
        ops, stream, mtp.block.o_proj.ptr, gated_q8_1, attn_proj_f32,
        h, n_q * head_dim, qdtype_for(&mtp.block.o_proj)?,
    )
    .context("mtp mmvq o_proj")?;
    // MTP-4-A: residual add in F32. h0_f32 (from fc mmvq) and
    // attn_proj_f32 (from o_proj mmvq) are both already F32 — keep them
    // F32 for the add, cast to F16 only for the post-attn norm input.
    add_f32(ops, stream, h0_f32, attn_proj_f32, attn_proj_f32, h)
        .context("mtp residual h1 (F32)")?;
    // attn_proj_f32 now holds h1_f32 = h0 + attn_proj. Reuse buffer.
    let h1_f32 = attn_proj_f32;
    cast_f32_to_f16(ops, stream, h1_f32, h1_f16, h)
        .context("mtp cast h1 F32 → F16 for post_attn norm")?;

    // ── 12. post_attention_layernorm + Q8_1 quant for MLP
    rmsnorm_quant_q8_1(
        ops, stream, h1_f16, mtp.block.post_attention_layernorm.ptr, h1n_q8_1,
        1, h, eps,
    )
    .context("mtp post_attention_layernorm + quant")?;

    // ── 13. MLP: gate + up matmuls, swiglu+quant fused, down matmul
    mmvq(
        ops, stream, mtp.block.gate_proj.ptr, h1n_q8_1, gate_mlp_f32,
        inter, h, qdtype_for(&mtp.block.gate_proj)?,
    )
    .context("mtp mmvq gate_proj")?;
    mmvq(
        ops, stream, mtp.block.up_proj.ptr, h1n_q8_1, up_mlp_f32,
        inter, h, qdtype_for(&mtp.block.up_proj)?,
    )
    .context("mtp mmvq up_proj")?;
    swiglu_f32_to_q8_1(ops, stream, gate_mlp_f32, up_mlp_f32, mlp_q8_1, inter)
        .context("mtp swiglu_f32_to_q8_1")?;
    mmvq(
        ops, stream, mtp.block.down_proj.ptr, mlp_q8_1, down_f32,
        h, inter, qdtype_for(&mtp.block.down_proj)?,
    )
    .context("mtp mmvq down_proj")?;
    // MTP-4-A: residual add in F32. h1_f32 (= h0 + attn_proj from above)
    // and down_f32 (from down_proj mmvq) are both F32 — keep F32 through
    // the add, cast to F16 only for the final mtp.norm input.
    add_f32(ops, stream, h1_f32, down_f32, down_f32, h)
        .context("mtp residual h2 (F32)")?;
    let h2_f32 = down_f32;  // reused buffer holds h2 = h1 + down
    cast_f32_to_f16(ops, stream, h2_f32, h2_f16, h)
        .context("mtp cast h2 F32 → F16 for final norm")?;

    // ── 15. Final norm: h_final = rmsnorm(h2, mtp.norm)
    rmsnorm_f16(
        ops, stream, h2_f16, mtp.norm.ptr, h_final_out,
        1, h, eps,
    )
    .context("mtp final norm")?;

    stream.synchronize()?;
    Ok(())
}

/// MTP-4-C-6: BF16-throughout MTP forward.
///
/// Same residual/attention/MLP structure as `forward_mtp_step`, but
/// activations stay BF16 across matmul → matmul (no Q8_1 activation
/// quantize). F32 mmvq accumulators are cast to BF16 directly. The
/// dominant per-mmvq Q8_1 noise (~0.78 % per-block-of-32, compounded
/// across 8 sequential matmuls) is eliminated.
///
/// `mtp` weights must be loaded via `load_mtp_head_bf16` so the
/// linears are BF16 (`mmvq_bf16_bf16` requires BF16 weight). Norms
/// stay F16 (small + needs more mantissa than BF16).
///
/// `h_t` / `e_token` are F16 (caller convention from
/// `forward_one_token_pp` / token_embd lookup); cast to BF16 at
/// entry. `h_final_out` is F16 (caller's convention into the LM
/// head); cast back at exit.
///
/// Scratch reuse: every "_f16" buffer in `MtpForwardScratch` is the
/// same byte size as its BF16 counterpart, so the existing scratch
/// is reused without growth. The Q8_1 buffers go unused on this
/// path (harmless).
#[allow(clippy::too_many_arguments)]
pub fn forward_mtp_step_bf16(
    ops: &flambeau_ops::OpsRegistry,
    stream: &flambeau_backend_hip::HipStream,
    device: &HipDevice,
    cfg: &crate::Qwen3MoEConfig,
    mtp: &MtpHeadWeights,
    scratch: &MtpForwardScratch,
    h_t_f16: flambeau_core::DevicePtr,
    e_token_f16: flambeau_core::DevicePtr,
    position: usize,
    h_final_out_f16: flambeau_core::DevicePtr,
    kv: Option<MtpKvCache>,
) -> Result<()> {
    use flambeau_core::{Device, DevicePtr, Stream};
    use flambeau_ops::hip::attention::{
        attention_decode_bf16, split_q_gate_bf16,
    };
    use flambeau_ops::hip::cast::{
        cast_bf16_to_f16, cast_f16_to_bf16, cast_f32_to_bf16,
    };
    use flambeau_ops::hip::mlp::{
        add_f32, sigmoid_mul_bf16, swiglu_f32_to_bf16,
    };
    use flambeau_ops::hip::norm::rmsnorm_bf16;
    use flambeau_ops::hip::pe::rope_neox_partial_bf16;
    use flambeau_ops::hip::qmatmul::mmvq_bf16_bf16;

    let h = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let n_q = cfg.num_heads;
    let n_kv = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let eps = cfg.rms_norm_eps;
    let rope = &cfg.rope;

    // Buffer aliases — every "_f16" scratch ptr is reused as BF16
    // (same 2 B/elem footprint). Comment notes the BF16 role.
    let fc_in_bf16     = scratch.fc_in_f16;     // [2*h] BF16
    let h0_f32         = scratch.h0_f32;        // mmvq accumulator
    let h0_bf16        = scratch.h0_f16;        // [h] BF16
    let q_full_f32     = scratch.q_full_f32;
    let q_full_bf16    = scratch.q_full_f16;    // [2*n_q*head_dim] BF16
    let q_bf16         = scratch.q_f16;         // [n_q*head_dim] BF16
    let gate_bf16      = scratch.gate_f16;      // [n_q*head_dim] BF16
    let k_f32          = scratch.k_f32;
    let v_f32          = scratch.v_f32;
    let k_bf16         = scratch.k_f16;         // [n_kv*head_dim] BF16
    let v_bf16         = scratch.v_f16;         // [n_kv*head_dim] BF16
    let kcache_slot    = scratch.kcache_slot_f16;
    let vcache_slot    = scratch.vcache_slot_f16;
    let attn_out_bf16  = scratch.attn_out_f16;
    let gated_out_bf16 = scratch.gated_out_f16;
    let attn_proj_f32  = scratch.attn_proj_f32;
    let h1_bf16        = scratch.h1_f16;        // [h] BF16
    let gate_mlp_f32   = scratch.gate_mlp_f32;
    let up_mlp_f32     = scratch.up_mlp_f32;
    let mlp_bf16       = scratch.mlp_bf16;      // [inter] BF16 (dedicated; Q8_1 buf too small)
    let down_f32       = scratch.down_f32;
    let h2_bf16        = scratch.h2_f16;        // [h] BF16

    debug_assert_eq!(
        scratch.bytes_inter_bf16,
        inter * 2,
        "MtpForwardScratch.mlp_bf16 was allocated for a different inter"
    );

    // ── 1. Cast h_t and e_token (F16) → BF16 into the fc input slots
    // (concat order: [embedding, hidden] per vLLM source).
    let fc_in_h_offset = fc_in_bf16.offset_bytes(h * 2); // BF16 = 2 B
    cast_f16_to_bf16(ops, stream, e_token_f16, fc_in_bf16, h)
        .context("mtp bf16 cast e_token F16→BF16")?;
    cast_f16_to_bf16(ops, stream, h_t_f16, fc_in_h_offset, h)
        .context("mtp bf16 cast h_t F16→BF16")?;

    // ── 2. pre_fc norms (BF16). Read each half, write to its half.
    // rmsnorm_bf16 normalises in-place over [m, k]; we run two 1-row calls,
    // one per half, with the corresponding norm weight.
    rmsnorm_bf16(
        ops, stream, fc_in_bf16, mtp.pre_fc_norm_embedding.ptr, fc_in_bf16,
        1, h, eps,
    )
    .context("mtp bf16 pre_fc_norm_embedding (first half)")?;
    rmsnorm_bf16(
        ops, stream, fc_in_h_offset, mtp.pre_fc_norm_hidden.ptr, fc_in_h_offset,
        1, h, eps,
    )
    .context("mtp bf16 pre_fc_norm_hidden (second half)")?;

    // ── 3. fc matmul: BF16 weight × BF16 act → F32.
    mmvq_bf16_bf16(ops, stream, mtp.fc.ptr, fc_in_bf16, h0_f32, h, 2 * h)
        .context("mtp bf16 mmvq fc")?;
    cast_f32_to_bf16(ops, stream, h0_f32, h0_bf16, h)
        .context("mtp bf16 cast h0 F32→BF16")?;

    // ── 4. input_layernorm (BF16, in-place into h0_bf16).
    rmsnorm_bf16(
        ops, stream, h0_bf16, mtp.block.input_layernorm.ptr, h0_bf16,
        1, h, eps,
    )
    .context("mtp bf16 input_layernorm")?;

    // ── 5. q/k/v projections — BF16 weight × BF16 act → F32; cast → BF16.
    mmvq_bf16_bf16(ops, stream, mtp.block.q_proj.ptr, h0_bf16, q_full_f32, 2 * n_q * head_dim, h)
        .context("mtp bf16 mmvq q_proj")?;
    cast_f32_to_bf16(ops, stream, q_full_f32, q_full_bf16, 2 * n_q * head_dim)
        .context("mtp bf16 cast q_full F32→BF16")?;
    split_q_gate_bf16(ops, stream, q_full_bf16, q_bf16, gate_bf16, 1, n_q, head_dim)
        .context("mtp bf16 split q_gate")?;

    mmvq_bf16_bf16(ops, stream, mtp.block.k_proj.ptr, h0_bf16, k_f32, n_kv * head_dim, h)
        .context("mtp bf16 mmvq k_proj")?;
    cast_f32_to_bf16(ops, stream, k_f32, k_bf16, n_kv * head_dim)
        .context("mtp bf16 cast k F32→BF16")?;
    mmvq_bf16_bf16(ops, stream, mtp.block.v_proj.ptr, h0_bf16, v_f32, n_kv * head_dim, h)
        .context("mtp bf16 mmvq v_proj")?;
    cast_f32_to_bf16(ops, stream, v_f32, v_bf16, n_kv * head_dim)
        .context("mtp bf16 cast v F32→BF16")?;

    // ── 6. q/k_norm per head (BF16).
    rmsnorm_bf16(ops, stream, q_bf16, mtp.block.q_norm.ptr, q_bf16, n_q, head_dim, eps)
        .context("mtp bf16 q_norm")?;
    rmsnorm_bf16(ops, stream, k_bf16, mtp.block.k_norm.ptr, k_bf16, n_kv, head_dim, eps)
        .context("mtp bf16 k_norm")?;

    // ── 7. Partial RoPE (skipped at position 0).
    if position != 0 {
        let position_host: [i32; 1] = [position as i32];
        let position_dev = device
            .alloc(4)
            .map_err(|e| anyhow!("hipMalloc 4 B (position): {e}"))?;
        // SAFETY: position_dev is fresh alloc; position_host outlives sync.
        unsafe {
            device.memcpy_async(
                stream,
                flambeau_core::CopyDirection::HostToDevice,
                position_dev,
                DevicePtr(position_host.as_ptr() as usize),
                4,
            )?;
        }
        stream.synchronize()?;
        rope_neox_partial_bf16(
            ops, stream, q_bf16, position_dev, rope.freq_base,
            1, n_q, head_dim, rope.rotated_dims,
        )
        .context("mtp bf16 rope Q")?;
        rope_neox_partial_bf16(
            ops, stream, k_bf16, position_dev, rope.freq_base,
            1, n_kv, head_dim, rope.rotated_dims,
        )
        .context("mtp bf16 rope K")?;
        // SAFETY: position_dev is uniquely owned by this scope.
        unsafe { device.dealloc(position_dev, 4)?; }
    }

    // ── 8. Attention. Either persistent (kv=Some) or transient slot.
    use flambeau_core::CopyDirection;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let bytes_kv = scratch.bytes_kv_f16;
    let (k_buf, v_buf, n_tokens_kv) = if let Some(c) = kv {
        let k_dst = c.kcache.offset_bytes(c.cache_position * bytes_kv);
        let v_dst = c.vcache.offset_bytes(c.cache_position * bytes_kv);
        // SAFETY: caller asserts cache size; k_bf16/v_bf16 hold bytes_kv each.
        unsafe {
            device.memcpy_async(stream, CopyDirection::DeviceToDevice, k_dst, k_bf16, bytes_kv)?;
            device.memcpy_async(stream, CopyDirection::DeviceToDevice, v_dst, v_bf16, bytes_kv)?;
        }
        (c.kcache, c.vcache, c.n_tokens_kv)
    } else {
        // SAFETY: scratch slot KV buffers each hold bytes_kv.
        unsafe {
            device.memcpy_async(stream, CopyDirection::DeviceToDevice, kcache_slot, k_bf16, bytes_kv)?;
            device.memcpy_async(stream, CopyDirection::DeviceToDevice, vcache_slot, v_bf16, bytes_kv)?;
        }
        (kcache_slot, vcache_slot, 1usize)
    };
    attention_decode_bf16(
        ops, stream, q_bf16, k_buf, v_buf, attn_out_bf16,
        n_q, n_kv, head_dim, n_tokens_kv, scale,
    )
    .context("mtp bf16 attention_decode")?;

    // ── 9. Output gate: sigmoid(gate) * attn_out (V1.7.4.b math).
    sigmoid_mul_bf16(ops, stream, gate_bf16, attn_out_bf16, gated_out_bf16, n_q * head_dim)
        .context("mtp bf16 sigmoid_mul")?;

    // ── 10. o_proj: BF16 → F32 attn_proj.
    mmvq_bf16_bf16(
        ops, stream, mtp.block.o_proj.ptr, gated_out_bf16, attn_proj_f32,
        h, n_q * head_dim,
    )
    .context("mtp bf16 mmvq o_proj")?;

    // ── 11. F32 residual (h0_f32 + attn_proj_f32 → attn_proj_f32).
    add_f32(ops, stream, h0_f32, attn_proj_f32, attn_proj_f32, h)
        .context("mtp bf16 residual h1 (F32)")?;
    let h1_f32 = attn_proj_f32;
    cast_f32_to_bf16(ops, stream, h1_f32, h1_bf16, h)
        .context("mtp bf16 cast h1 F32→BF16")?;

    // ── 12. post_attention_layernorm in BF16.
    rmsnorm_bf16(
        ops, stream, h1_bf16, mtp.block.post_attention_layernorm.ptr, h1_bf16,
        1, h, eps,
    )
    .context("mtp bf16 post_attention_layernorm")?;

    // ── 13. MLP — BF16 mmvq for gate/up; fused swiglu+cast to BF16 mlp; BF16 mmvq down.
    mmvq_bf16_bf16(ops, stream, mtp.block.gate_proj.ptr, h1_bf16, gate_mlp_f32, inter, h)
        .context("mtp bf16 mmvq gate_proj")?;
    mmvq_bf16_bf16(ops, stream, mtp.block.up_proj.ptr, h1_bf16, up_mlp_f32, inter, h)
        .context("mtp bf16 mmvq up_proj")?;
    swiglu_f32_to_bf16(ops, stream, gate_mlp_f32, up_mlp_f32, mlp_bf16, inter)
        .context("mtp bf16 swiglu_f32_to_bf16")?;
    mmvq_bf16_bf16(ops, stream, mtp.block.down_proj.ptr, mlp_bf16, down_f32, h, inter)
        .context("mtp bf16 mmvq down_proj")?;

    // ── 14. F32 residual h2 = h1 + down.
    add_f32(ops, stream, h1_f32, down_f32, down_f32, h)
        .context("mtp bf16 residual h2 (F32)")?;
    let h2_f32 = down_f32;
    cast_f32_to_bf16(ops, stream, h2_f32, h2_bf16, h)
        .context("mtp bf16 cast h2 F32→BF16")?;

    // ── 15. Final norm (BF16).
    rmsnorm_bf16(
        ops, stream, h2_bf16, mtp.norm.ptr, h2_bf16,
        1, h, eps,
    )
    .context("mtp bf16 final norm")?;

    // ── 16. Cast BF16 → F16 for caller-visible h_final_out.
    cast_bf16_to_f16(ops, stream, h2_bf16, h_final_out_f16, h)
        .context("mtp bf16 cast h_final BF16→F16")?;

    stream.synchronize()?;
    Ok(())
}

/// MTP helper: run `forward_mtp_step` followed by the LM-head
/// matmul, returning the argmax-predicted token id.
///
/// vLLM convention (verified from
/// `vllm/model_executor/models/qwen3_next.py:531`): the base model's
/// forward applies `self.norm(hidden, residual)` BEFORE returning the
/// hidden state to the caller. So when the spec-decode caller
/// invokes `Qwen3NextMultiTokenPredictor.forward(hidden_states, ...)`,
/// `hidden_states` is **post-`model.norm`**.
///
/// flambeau's existing `forward_one_token_pp` writes the pre-norm
/// hidden into `scratch.hidden_a` (since output_norm is folded into
/// `forward_output_head_decode` via `rmsnorm_quant_q8_1`). For MTP we
/// re-apply `output_norm` standalone here to match vLLM's convention.
///
/// Inputs:
///   `scratch`              — reusable per-session scratch buffers
///   `output_norm_weight`   — base model's `output_norm.weight`
///   `lm_head_weight`       — base model's `output.weight` (any GGML quant)
///   `token_embd_row_f16`   — F16 [hidden] embedding of the token whose
///                            successor we're predicting
///   `hidden_pre_norm`      — F16 [hidden] base hidden state pre-output_norm
///   `position`             — base-model position (drives MROPE)
///   `kv`                   — Some = persistent multi-slot cache; None = transient
///
/// Returns the predicted next-token id.
#[allow(clippy::too_many_arguments)]
pub fn forward_mtp_step_with_lm_head(
    ops: &flambeau_ops::OpsRegistry,
    stream: &flambeau_backend_hip::HipStream,
    device: &HipDevice,
    cfg: &crate::Qwen3MoEConfig,
    mtp: &MtpHeadWeights,
    scratch: &MtpForwardScratch,
    output_norm_weight: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    hidden_pre_norm: flambeau_core::DevicePtr,
    token_embd_row_f16: flambeau_core::DevicePtr,
    position: usize,
    kv: Option<MtpKvCache>,
) -> Result<u32> {
    use flambeau_core::{Device, DevicePtr, Stream};
    use flambeau_ops::hip::norm::{quantize_f16_q8_1, rmsnorm_f16};

    device.bind()?;
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    // 1. Apply base output_norm to get the post-norm hidden vLLM's MTP
    //    convention expects.
    rmsnorm_f16(
        ops, stream,
        hidden_pre_norm, output_norm_weight.ptr, scratch.h_t_post_norm,
        1, hidden, cfg.rms_norm_eps,
    )
    .context("base output_norm for MTP h_t")?;

    // 2. Run MTP on post-norm hidden via the F16/Q8_1 forward path.
    //    The BF16 alt (FLAMBEAU_MTP_BF16=1) was deleted in S3 — anchors
    //    leave SPEC_MTP unset, so MTP is loaded but never invoked here.
    forward_mtp_step(
        ops, stream, device, cfg, mtp, scratch,
        scratch.h_t_post_norm,
        token_embd_row_f16,
        position,
        scratch.mtp_h_final,
        kv,
    )?;

    // 3. LM head: quantize MTP output → Q8_1 → mmvq → F32 logits → argmax.
    quantize_f16_q8_1(ops, stream, scratch.mtp_h_final, scratch.x_q8_1, hidden)
        .context("mtp lm_head quantize")?;

    let dtype = match lm_head_weight.dtype {
        flambeau_quant::GgmlDType::F16  => flambeau_core::op::QDtype::F16,
        flambeau_quant::GgmlDType::Q8_0 => flambeau_core::op::QDtype::Q8_0,
        flambeau_quant::GgmlDType::Q4_0 => flambeau_core::op::QDtype::Q4_0,
        flambeau_quant::GgmlDType::Q4_1 => flambeau_core::op::QDtype::Q4_1,
        flambeau_quant::GgmlDType::Q4K  => flambeau_core::op::QDtype::Q4_K,
        flambeau_quant::GgmlDType::Q5_0 => flambeau_core::op::QDtype::Q5_0,
        flambeau_quant::GgmlDType::Q5K  => flambeau_core::op::QDtype::Q5_K,
        flambeau_quant::GgmlDType::Q6K  => flambeau_core::op::QDtype::Q6_K,
        d => bail!("unsupported lm_head dtype {d:?} for MTP probe"),
    };
    flambeau_ops::hip::qmatmul::mmvq(
        ops, stream, lm_head_weight.ptr, scratch.x_q8_1, scratch.logits_f32,
        vocab, hidden, dtype,
    )
    .context("mtp lm_head mmvq")?;

    // 4. Argmax host-side (cheap; vocab × 4 bytes ≈ 1 MB).
    let mut host = vec![0.0f32; vocab];
    // SAFETY: logits_f32 holds vocab*4 bytes; host has vocab*4 bytes.
    unsafe {
        device.memcpy_async(
            stream,
            flambeau_core::CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            scratch.logits_f32,
            vocab * 4,
        )?;
    }
    stream.synchronize()?;
    let mut best_idx = 0usize;
    let mut best_val = host[0];
    for (i, &v) in host.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

/// MTP-5h-Lever-B — async variant of [`forward_mtp_step_with_lm_head`]
/// that issues every device-side kernel (output_norm, MTP forward, LM
/// head quant + mmvq) on the supplied `stream` but does NOT download
/// logits or host-argmax. Returns once kernels are queued; logits live
/// in `scratch.logits_f32` on device until the caller pairs this with
/// [`mtp_logits_argmax_host`].
///
/// Used by the spec-decode driver to overlap the MTP draft with
/// `save_gdn_snapshot` on per-rank default streams (run on a different
/// stream of the head device, executes concurrently with the snap
/// memcpys).
#[allow(clippy::too_many_arguments)]
pub fn forward_mtp_step_with_lm_head_async(
    ops: &flambeau_ops::OpsRegistry,
    stream: &flambeau_backend_hip::HipStream,
    device: &HipDevice,
    cfg: &crate::Qwen3MoEConfig,
    mtp: &MtpHeadWeights,
    scratch: &MtpForwardScratch,
    output_norm_weight: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    hidden_pre_norm: flambeau_core::DevicePtr,
    token_embd_row_f16: flambeau_core::DevicePtr,
    position: usize,
    kv: Option<MtpKvCache>,
) -> Result<()> {
    use flambeau_ops::hip::norm::{quantize_f16_q8_1, rmsnorm_f16};

    device.bind()?;
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    rmsnorm_f16(
        ops, stream,
        hidden_pre_norm, output_norm_weight.ptr, scratch.h_t_post_norm,
        1, hidden, cfg.rms_norm_eps,
    )
    .context("base output_norm for MTP h_t (async)")?;

    forward_mtp_step(
        ops, stream, device, cfg, mtp, scratch,
        scratch.h_t_post_norm, token_embd_row_f16, position,
        scratch.mtp_h_final, kv,
    )?;

    quantize_f16_q8_1(ops, stream, scratch.mtp_h_final, scratch.x_q8_1, hidden)
        .context("mtp lm_head quantize (async)")?;

    let dtype = match lm_head_weight.dtype {
        flambeau_quant::GgmlDType::F16  => flambeau_core::op::QDtype::F16,
        flambeau_quant::GgmlDType::Q8_0 => flambeau_core::op::QDtype::Q8_0,
        flambeau_quant::GgmlDType::Q4_0 => flambeau_core::op::QDtype::Q4_0,
        flambeau_quant::GgmlDType::Q4_1 => flambeau_core::op::QDtype::Q4_1,
        flambeau_quant::GgmlDType::Q4K  => flambeau_core::op::QDtype::Q4_K,
        flambeau_quant::GgmlDType::Q5_0 => flambeau_core::op::QDtype::Q5_0,
        flambeau_quant::GgmlDType::Q5K  => flambeau_core::op::QDtype::Q5_K,
        flambeau_quant::GgmlDType::Q6K  => flambeau_core::op::QDtype::Q6_K,
        d => bail!("unsupported lm_head dtype {d:?} for MTP probe"),
    };
    flambeau_ops::hip::qmatmul::mmvq(
        ops, stream, lm_head_weight.ptr, scratch.x_q8_1, scratch.logits_f32,
        vocab, hidden, dtype,
    )
    .context("mtp lm_head mmvq (async)")?;

    Ok(())
}

/// MTP-5h-Lever-B — host-side download + argmax pair for
/// [`forward_mtp_step_with_lm_head_async`]. Syncs `stream` (waits for
/// the queued lm-head mmvq to complete), downloads
/// `scratch.logits_f32` to host, and returns the argmax token id.
pub fn mtp_logits_argmax_host(
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    scratch: &MtpForwardScratch,
    vocab: usize,
) -> Result<u32> {
    use flambeau_core::{Device, DevicePtr, Stream};

    let mut host = vec![0.0f32; vocab];
    // SAFETY: logits_f32 holds vocab*4 bytes; host has vocab*4 bytes.
    unsafe {
        device.memcpy_async(
            stream,
            flambeau_core::CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            scratch.logits_f32,
            vocab * 4,
        )?;
    }
    stream.synchronize()?;
    let mut best_idx = 0usize;
    let mut best_val = host[0];
    for (i, &v) in host.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

/// Variant of [`forward_mtp_step_with_lm_head`] that returns the full
/// F32 logit row instead of just the argmax token. Used by the
/// rejection-sampling spec-decode path (MTP-5g) which needs MTP's
/// distribution `q(·)` to compute `min(1, p(t)/q(t))` against the
/// base verifier's `p(·)`.
#[allow(clippy::too_many_arguments)]
pub fn forward_mtp_step_with_lm_head_logits(
    ops: &flambeau_ops::OpsRegistry,
    stream: &flambeau_backend_hip::HipStream,
    device: &HipDevice,
    cfg: &crate::Qwen3MoEConfig,
    mtp: &MtpHeadWeights,
    scratch: &MtpForwardScratch,
    output_norm_weight: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    hidden_pre_norm: flambeau_core::DevicePtr,
    token_embd_row_f16: flambeau_core::DevicePtr,
    position: usize,
    kv: Option<MtpKvCache>,
    logits_out: &mut Vec<f32>,
) -> Result<u32> {
    use flambeau_core::{Device, DevicePtr, Stream};
    use flambeau_ops::hip::norm::{quantize_f16_q8_1, rmsnorm_f16};

    device.bind()?;
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    rmsnorm_f16(
        ops, stream,
        hidden_pre_norm, output_norm_weight.ptr, scratch.h_t_post_norm,
        1, hidden, cfg.rms_norm_eps,
    )
    .context("base output_norm for MTP h_t (logits variant)")?;

    forward_mtp_step(
        ops, stream, device, cfg, mtp, scratch,
        scratch.h_t_post_norm, token_embd_row_f16, position,
        scratch.mtp_h_final, kv,
    )?;

    quantize_f16_q8_1(ops, stream, scratch.mtp_h_final, scratch.x_q8_1, hidden)
        .context("mtp lm_head quantize (logits variant)")?;

    let dtype = match lm_head_weight.dtype {
        flambeau_quant::GgmlDType::F16  => flambeau_core::op::QDtype::F16,
        flambeau_quant::GgmlDType::Q8_0 => flambeau_core::op::QDtype::Q8_0,
        flambeau_quant::GgmlDType::Q4_0 => flambeau_core::op::QDtype::Q4_0,
        flambeau_quant::GgmlDType::Q4_1 => flambeau_core::op::QDtype::Q4_1,
        flambeau_quant::GgmlDType::Q4K  => flambeau_core::op::QDtype::Q4_K,
        flambeau_quant::GgmlDType::Q5_0 => flambeau_core::op::QDtype::Q5_0,
        flambeau_quant::GgmlDType::Q5K  => flambeau_core::op::QDtype::Q5_K,
        flambeau_quant::GgmlDType::Q6K  => flambeau_core::op::QDtype::Q6_K,
        d => bail!("unsupported lm_head dtype {d:?} for MTP probe"),
    };
    flambeau_ops::hip::qmatmul::mmvq(
        ops, stream, lm_head_weight.ptr, scratch.x_q8_1, scratch.logits_f32,
        vocab, hidden, dtype,
    )
    .context("mtp lm_head mmvq (logits variant)")?;

    logits_out.clear();
    logits_out.resize(vocab, 0.0);
    // SAFETY: logits_f32 holds vocab*4 bytes; logits_out has vocab*4 bytes.
    unsafe {
        device.memcpy_async(
            stream,
            flambeau_core::CopyDirection::DeviceToHost,
            DevicePtr(logits_out.as_mut_ptr() as usize),
            scratch.logits_f32,
            vocab * 4,
        )?;
    }
    stream.synchronize()?;
    let mut best_idx = 0usize;
    let mut best_val = logits_out[0];
    for (i, &v) in logits_out.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

/// Derive the conventional sibling MTP file path from a base GGUF
/// path: strip the trailing quant suffix and `.gguf` extension, then
/// append `-mtp.gguf`. Returns the candidate path; the caller checks
/// `.exists()`.
///
/// Examples:
///   `Qwen3.6-27B-Q4_0.gguf`         → `Qwen3.6-27B-mtp.gguf`
///   `Qwen3.6-27B-UD-Q4_K_XL.gguf`   → `Qwen3.6-27B-mtp.gguf`
///   `Qwen3.6-27B-Q8_0.gguf`         → `Qwen3.6-27B-mtp.gguf`
///
/// The heuristic: split the stem on `-`, drop trailing components
/// that look like quant tags (start with `Q`, `UD`, `IQ`, `F`, `BF`,
/// or are pure-numeric), keep the prefix.
pub fn derive_sibling_mtp_path(base_path: &std::path::Path) -> std::path::PathBuf {
    let parent = base_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let stem = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let parts: Vec<&str> = stem.split('-').collect();
    let mut keep_n = parts.len();
    while keep_n > 0 {
        let p = parts[keep_n - 1];
        let is_quant_tag = p.starts_with('Q')
            || p.starts_with("UD")
            || p.starts_with("IQ")
            || p == "F16"
            || p == "F32"
            || p == "BF16"
            || p == "XL"
            || p == "M"
            || p == "S"
            || p == "K"
            || p == "L";
        if is_quant_tag {
            keep_n -= 1;
        } else {
            break;
        }
    }
    let basename = parts[..keep_n].join("-");
    parent.join(format!("{basename}-mtp.gguf"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_path_strips_quant_suffix() {
        let base = std::path::Path::new("/m/Qwen3.6-27B-Q4_0.gguf");
        assert_eq!(
            derive_sibling_mtp_path(base),
            std::path::PathBuf::from("/m/Qwen3.6-27B-mtp.gguf")
        );
        let base = std::path::Path::new("/m/Qwen3.6-27B-UD-Q4_K_XL.gguf");
        assert_eq!(
            derive_sibling_mtp_path(base),
            std::path::PathBuf::from("/m/Qwen3.6-27B-mtp.gguf")
        );
        let base = std::path::Path::new("/m/Qwen3.6-27B-Q8_0.gguf");
        assert_eq!(
            derive_sibling_mtp_path(base),
            std::path::PathBuf::from("/m/Qwen3.6-27B-mtp.gguf")
        );
    }
}
