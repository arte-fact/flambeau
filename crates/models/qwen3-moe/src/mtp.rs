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
//! Pairing semantics (see `certs/research/mtp_2_converter_landed.md`):
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
    if !matches!(r.dtype, GgmlDType::F32 | GgmlDType::Q8_0) {
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

/// MTP-3.5 forward — one MTP step on F16 hidden state.
///
/// Composes existing flambeau ops; no new kernels. Allocates scratch
/// buffers internally; this is the smoke / parity-test version, not
/// the spec-decode hot path. MTP-4 wraps this in a reusable scratch
/// struct.
///
/// Inputs:
///   `h_t` F16 [hidden]      — base model hidden (post `output_norm`)
///   `e_token` F16 [hidden]  — embedding of the token sampled from h_t
///   `position` usize        — base-model position (drives MROPE)
/// Output:
///   `h_final_out` F16 [hidden] — pre-LM-head MTP-block output. Caller
///                                runs the (shared) lm_head matmul to
///                                produce draft logits.
///
/// `position == 0` skips MROPE entirely (rotation is identity at
/// position 0 — useful for the parity test against the Python ref).
/// At `position > 0` we apply `rope_neox_partial_f16` with
/// `cfg.rope.rotated_dims` (matches base full-attn convention).
pub fn forward_mtp_step(
    ops: &flambeau_ops::OpsRegistry,
    stream: &flambeau_backend_hip::HipStream,
    device: &HipDevice,
    cfg: &crate::Qwen3MoEConfig,
    mtp: &MtpHeadWeights,
    h_t: flambeau_core::DevicePtr,
    e_token: flambeau_core::DevicePtr,
    position: usize,
    h_final_out: flambeau_core::DevicePtr,
) -> Result<()> {
    use flambeau_backend_hip::HipDevice as _Dev;
    use flambeau_core::{Device, DevicePtr, Stream};
    use flambeau_core::op::QDtype;
    use flambeau_ops::hip::attention::{attention_decode_f16_slots, split_q_gate_f16};
    use flambeau_ops::hip::cast::{cast_f16_to_f32, cast_f32_to_f16};
    use flambeau_ops::hip::mlp::{add_f16, sigmoid_mul_f16, swiglu_f32_to_q8_1};
    use flambeau_ops::hip::norm::{quantize_f16_q8_1, rmsnorm_f16, rmsnorm_quant_q8_1};
    use flambeau_ops::hip::qmatmul::mmvq;

    let _ = _Dev::new; // silence unused-import warning in some configs

    let h = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size; // dense FFN size (set from feed_forward_length for qwen35 dense)
    let n_q = cfg.num_heads;
    let n_kv = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let eps = cfg.rms_norm_eps;
    let rope = &cfg.rope;

    // Q8_1 block bytes: F16 d + F16 s + 32 int8 quants = 36 bytes.
    let q8_1_block_bytes = std::mem::size_of::<flambeau_quant::BlockQ8_1>();
    let q8_1_blocks = |n_elems: usize| n_elems / 32 * q8_1_block_bytes;

    // ── 1. Allocate scratch (released at end of function via Drop).
    // Sizes for all the intermediate buffers. F16 = 2 B, F32 = 4 B.
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

    let alloc = |bytes: usize| -> Result<DevicePtr> {
        device
            .alloc(bytes)
            .map_err(|e| anyhow!("hipMalloc {bytes} B: {e}"))
    };

    let fc_in_f16 = alloc(bytes_2h_f16)?;            // [norm_h ‖ norm_e] F16 [10240]
    let fc_in_q8_1 = alloc(bytes_2h_q8_1)?;          // Q8_1 input to fc matmul
    let h0_f32 = alloc(bytes_h_f32)?;                // fc output
    let h0_f16 = alloc(bytes_h_f16)?;                // cast to F16
    let h0n_q8_1 = alloc(bytes_h_q8_1)?;             // norm + quant for q/k/v matmuls
    let q_full_f32 = alloc(bytes_qfull_f32)?;        // q_proj output [12288]
    let q_full_f16 = alloc(bytes_qfull_f16)?;        // cast to F16
    let q_f16 = alloc(bytes_qhalf_f16)?;             // Q half of q_full
    let gate_f16 = alloc(bytes_qhalf_f16)?;          // gate half of q_full
    let k_f32 = alloc(bytes_kv_f32)?;                // k_proj output
    let v_f32 = alloc(bytes_kv_f32)?;                // v_proj output
    let k_f16 = alloc(bytes_kv_f16)?;                // cast to F16
    let v_f16 = alloc(bytes_kv_f16)?;                // cast to F16
    // 1-token KV cache: just stages K/V for the single attention call.
    let kcache_f16 = alloc(bytes_kv_f16)?;
    let vcache_f16 = alloc(bytes_kv_f16)?;
    let attn_out_f16 = alloc(bytes_qhalf_f16)?;
    let gated_out_f16 = alloc(bytes_qhalf_f16)?;
    let gated_q8_1 = alloc(bytes_qhalf_q8_1)?;
    let attn_proj_f32 = alloc(bytes_h_f32)?;
    let attn_proj_f16 = alloc(bytes_h_f16)?;
    let h1_f16 = alloc(bytes_h_f16)?;
    let h1n_q8_1 = alloc(bytes_h_q8_1)?;
    let gate_mlp_f32 = alloc(bytes_inter_f32)?;
    let up_mlp_f32 = alloc(bytes_inter_f32)?;
    let mlp_q8_1 = alloc(bytes_inter_q8_1)?;
    let down_f32 = alloc(bytes_h_f32)?;
    let down_f16 = alloc(bytes_h_f16)?;
    let h2_f16 = alloc(bytes_h_f16)?;

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
        h, 2 * h, QDtype::Q8_0,
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
        2 * n_q * head_dim, h, QDtype::Q8_0,
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
        n_kv * head_dim, h, QDtype::Q8_0,
    )
    .context("mtp mmvq k_proj")?;
    cast_f32_to_f16(ops, stream, k_f32, k_f16, n_kv * head_dim)
        .context("mtp cast k → F16")?;
    mmvq(
        ops, stream, mtp.block.v_proj.ptr, h0n_q8_1, v_f32,
        n_kv * head_dim, h, QDtype::Q8_0,
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
        let position_dev = alloc(4)?;
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

    // ── 8. Single-token attention. Stage K/V at position 0 of a
    //     length-1 KV cache and run attention_decode_f16_slots.
    use flambeau_core::CopyDirection;
    // SAFETY: kcache_f16 / vcache_f16 are fresh allocs of n_kv*head_dim*2 B.
    unsafe {
        device.memcpy_async(stream, CopyDirection::DeviceToDevice, kcache_f16, k_f16, bytes_kv_f16)?;
        device.memcpy_async(stream, CopyDirection::DeviceToDevice, vcache_f16, v_f16, bytes_kv_f16)?;
    }
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    attention_decode_f16_slots(
        ops, stream, q_f16, kcache_f16, vcache_f16, attn_out_f16,
        n_q, n_kv, head_dim, /*n_tokens_kv=*/ 1, scale,
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
        h, n_q * head_dim, QDtype::Q8_0,
    )
    .context("mtp mmvq o_proj")?;
    cast_f32_to_f16(ops, stream, attn_proj_f32, attn_proj_f16, h)
        .context("mtp cast o_proj → F16")?;

    // ── 11. Residual add: h1 = h0 + attn_proj
    add_f16(ops, stream, h0_f16, attn_proj_f16, h1_f16, h)
        .context("mtp residual h1")?;

    // ── 12. post_attention_layernorm + Q8_1 quant for MLP
    rmsnorm_quant_q8_1(
        ops, stream, h1_f16, mtp.block.post_attention_layernorm.ptr, h1n_q8_1,
        1, h, eps,
    )
    .context("mtp post_attention_layernorm + quant")?;

    // ── 13. MLP: gate + up matmuls, swiglu+quant fused, down matmul
    mmvq(
        ops, stream, mtp.block.gate_proj.ptr, h1n_q8_1, gate_mlp_f32,
        inter, h, QDtype::Q8_0,
    )
    .context("mtp mmvq gate_proj")?;
    mmvq(
        ops, stream, mtp.block.up_proj.ptr, h1n_q8_1, up_mlp_f32,
        inter, h, QDtype::Q8_0,
    )
    .context("mtp mmvq up_proj")?;
    swiglu_f32_to_q8_1(ops, stream, gate_mlp_f32, up_mlp_f32, mlp_q8_1, inter)
        .context("mtp swiglu_f32_to_q8_1")?;
    mmvq(
        ops, stream, mtp.block.down_proj.ptr, mlp_q8_1, down_f32,
        h, inter, QDtype::Q8_0,
    )
    .context("mtp mmvq down_proj")?;
    cast_f32_to_f16(ops, stream, down_f32, down_f16, h)
        .context("mtp cast down → F16")?;

    // ── 14. Residual: h2 = h1 + down
    add_f16(ops, stream, h1_f16, down_f16, h2_f16, h)
        .context("mtp residual h2")?;

    // ── 15. Final norm: h_final = rmsnorm(h2, mtp.norm)
    rmsnorm_f16(
        ops, stream, h2_f16, mtp.norm.ptr, h_final_out,
        1, h, eps,
    )
    .context("mtp final norm")?;

    stream.synchronize()?;

    // ── Free scratch
    let _ = cast_f16_to_f32; // silence unused (kept for future debug)
    // SAFETY: each allocation is uniquely owned by this scope.
    unsafe {
        device.dealloc(fc_in_f16, bytes_2h_f16)?;
        device.dealloc(fc_in_q8_1, bytes_2h_q8_1)?;
        device.dealloc(h0_f32, bytes_h_f32)?;
        device.dealloc(h0_f16, bytes_h_f16)?;
        device.dealloc(h0n_q8_1, bytes_h_q8_1)?;
        device.dealloc(q_full_f32, bytes_qfull_f32)?;
        device.dealloc(q_full_f16, bytes_qfull_f16)?;
        device.dealloc(q_f16, bytes_qhalf_f16)?;
        device.dealloc(gate_f16, bytes_qhalf_f16)?;
        device.dealloc(k_f32, bytes_kv_f32)?;
        device.dealloc(v_f32, bytes_kv_f32)?;
        device.dealloc(k_f16, bytes_kv_f16)?;
        device.dealloc(v_f16, bytes_kv_f16)?;
        device.dealloc(kcache_f16, bytes_kv_f16)?;
        device.dealloc(vcache_f16, bytes_kv_f16)?;
        device.dealloc(attn_out_f16, bytes_qhalf_f16)?;
        device.dealloc(gated_out_f16, bytes_qhalf_f16)?;
        device.dealloc(gated_q8_1, bytes_qhalf_q8_1)?;
        device.dealloc(attn_proj_f32, bytes_h_f32)?;
        device.dealloc(attn_proj_f16, bytes_h_f16)?;
        device.dealloc(h1_f16, bytes_h_f16)?;
        device.dealloc(h1n_q8_1, bytes_h_q8_1)?;
        device.dealloc(gate_mlp_f32, bytes_inter_f32)?;
        device.dealloc(up_mlp_f32, bytes_inter_f32)?;
        device.dealloc(mlp_q8_1, bytes_inter_q8_1)?;
        device.dealloc(down_f32, bytes_h_f32)?;
        device.dealloc(down_f16, bytes_h_f16)?;
        device.dealloc(h2_f16, bytes_h_f16)?;
    }
    Ok(())
}

/// MTP-4 helper: run `forward_mtp_step` followed by the LM-head
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
///   `output_norm_weight` — base model's `output_norm.weight` (F16,
///                          on the same device as MTP)
///   `lm_head_weight`     — base model's `output.weight` (any
///                          GGML quant, on the same device as MTP)
///   `token_embd_row_f16` — F16 [hidden] embedding of the token whose
///                          successor we're predicting (caller is
///                          responsible for cross-rank copy if needed)
///   `hidden_pre_norm`    — F16 [hidden] base hidden state pre-output_norm
///                          (= flambeau's `scratch.hidden_a` after a
///                          forward pass)
///   `position`           — base-model position (drives MROPE)
///
/// Returns the predicted next-token id.
#[allow(clippy::too_many_arguments)]
pub fn forward_mtp_step_with_lm_head(
    ops: &flambeau_ops::OpsRegistry,
    stream: &flambeau_backend_hip::HipStream,
    device: &HipDevice,
    cfg: &crate::Qwen3MoEConfig,
    mtp: &MtpHeadWeights,
    output_norm_weight: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    hidden_pre_norm: flambeau_core::DevicePtr,
    token_embd_row_f16: flambeau_core::DevicePtr,
    position: usize,
) -> Result<u32> {
    use flambeau_core::{Device, DevicePtr, Stream};
    use flambeau_ops::hip::norm::{quantize_f16_q8_1, rmsnorm_f16};

    device.bind()?;
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    // 1. Apply base output_norm to get the post-norm hidden vLLM's MTP
    //    convention expects.
    let h_t_post_norm = device
        .alloc(hidden * 2)
        .map_err(|e| anyhow!("hipMalloc h_t_post_norm: {e}"))?;
    rmsnorm_f16(
        ops, stream,
        hidden_pre_norm, output_norm_weight.ptr, h_t_post_norm,
        1, hidden, cfg.rms_norm_eps,
    )
    .context("base output_norm for MTP h_t")?;

    // 2. Run MTP on post-norm hidden.
    let mtp_h_final = device
        .alloc(hidden * 2)
        .map_err(|e| anyhow!("hipMalloc mtp_h_final: {e}"))?;
    forward_mtp_step(
        ops, stream, device, cfg, mtp,
        h_t_post_norm,
        token_embd_row_f16,
        position,
        mtp_h_final,
    )?;

    // 3. LM head: quantize MTP output → Q8_1 → mmvq → F32 logits → argmax.
    let q8_1_block_bytes = std::mem::size_of::<flambeau_quant::BlockQ8_1>();
    let n_blocks = hidden / 32;
    let x_q8_1 = device
        .alloc(n_blocks * q8_1_block_bytes)
        .map_err(|e| anyhow!("hipMalloc x_q8_1: {e}"))?;
    let logits_f32 = device
        .alloc(vocab * 4)
        .map_err(|e| anyhow!("hipMalloc logits_f32: {e}"))?;

    quantize_f16_q8_1(ops, stream, mtp_h_final, x_q8_1, hidden)
        .context("mtp lm_head quantize")?;

    let dtype = match lm_head_weight.dtype {
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
        ops, stream, lm_head_weight.ptr, x_q8_1, logits_f32,
        vocab, hidden, dtype,
    )
    .context("mtp lm_head mmvq")?;

    // 4. Argmax host-side (cheap; vocab × 4 bytes = ~1 MB).
    let mut host = vec![0.0f32; vocab];
    // SAFETY: logits_f32 has vocab*4 bytes; host has vocab*4 bytes.
    unsafe {
        device.memcpy_async(
            stream,
            flambeau_core::CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            logits_f32,
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

    // 5. Free per-call scratch.
    // SAFETY: each alloc is uniquely owned by this call.
    unsafe {
        device.dealloc(h_t_post_norm, hidden * 2)?;
        device.dealloc(mtp_h_final, hidden * 2)?;
        device.dealloc(x_q8_1, n_blocks * q8_1_block_bytes)?;
        device.dealloc(logits_f32, vocab * 4)?;
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
