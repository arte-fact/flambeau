//! **#230 P2.11a** — minimal Qwen3 dense-architecture embedding-model
//! loader. Hosts a single `general.architecture = "qwen3"` GGUF on a
//! single HIP device for use by the `/v1/embeddings` endpoint
//! (#231 wires the forward path).
//!
//! Scope (V1):
//! - Single-device only (TP / PP for the embedding head is V2 work; a
//!   600 M / 4 B / 8 B Qwen3-Embedding fits on one MI50 even alongside
//!   a 27 B chat model).
//! - `qwen3` arch only (no Qwen3.5/3.6 hybrid GDN; no MoE).
//! - Tensors uploaded verbatim (Q8_0 stays Q8_0; F32 norms stay F32 —
//!   downstream forward casts to F16 in scratch).
//!
//! Scope is intentionally narrow: the existing `Qwen3MoEShardedModel`
//! loader assumes `family == Hybrid` with `ssm.*` keys present, which
//! Qwen3-Embedding GGUFs do not carry. Forking the loader is cheaper
//! than threading "no GDN, no MoE, no full_attention_interval" through
//! every call site of the existing code.

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;
use flambeau_quant::GgufFile;

use crate::weights::DeviceTensor;

/// Lightweight config for an embedding model. Read from
/// `general.architecture = "qwen3"` GGUF metadata; intentionally a
/// subset of the full `Qwen3MoEConfig` since embedding models don't
/// carry MoE / GDN keys.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// `general.architecture` — must be `"qwen3"` in V1.
    pub arch: String,
    /// `qwen3.embedding_length`.
    pub hidden_size: usize,
    /// Outer dim of `token_embd.weight`.
    pub vocab_size: usize,
    /// `qwen3.block_count`.
    pub num_layers: usize,
    /// `qwen3.attention.head_count`.
    pub num_heads: usize,
    /// `qwen3.attention.head_count_kv`.
    pub num_kv_heads: usize,
    /// `qwen3.attention.key_length`, falling back to `hidden / heads`.
    pub head_dim: usize,
    /// `qwen3.context_length`.
    pub context_length: usize,
    /// `qwen3.attention.layer_norm_rms_epsilon`.
    pub rms_norm_eps: f32,
    /// `qwen3.feed_forward_length` — dense GLU FFN width.
    pub ffn_inner: usize,
    /// `qwen3.rope.freq_base`, default 10_000.
    pub rope_freq_base: f32,
    /// `qwen3.pooling_type` (llama.cpp convention): 0=none, 1=mean,
    /// 2=cls, 3=last. Qwen3-Embedding ships with `3` (last-token
    /// pooling). Stored for #231; loader doesn't act on it.
    pub pooling_type: u32,
}

impl EmbeddingConfig {
    /// Parse from an opened embedding GGUF. Errors when the arch tag
    /// isn't `qwen3` or when a required metadata key is missing.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file
            .architecture()
            .ok_or_else(|| anyhow!("missing general.architecture"))?
            .to_string();
        if arch != "qwen3" {
            bail!(
                "embedding loader supports `qwen3` arch only (got `{arch}`); \
                 see #230 design doc for the V1 scope decision"
            );
        }
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let req_u32 = |suffix: &str| -> Result<usize> {
            file.metadata_u32(&key(suffix))
                .map(|v| v as usize)
                .ok_or_else(|| anyhow!("missing key `{}`", key(suffix)))
        };
        let opt_u32 = |suffix: &str| file.metadata_u32(&key(suffix)).map(|v| v as usize);
        let req_f32 = |suffix: &str| -> Result<f32> {
            file.metadata_f32(&key(suffix))
                .ok_or_else(|| anyhow!("missing key `{}`", key(suffix)))
        };
        let opt_f32 = |suffix: &str| file.metadata_f32(&key(suffix));

        let hidden_size = req_u32("embedding_length")?;
        let num_heads = req_u32("attention.head_count")?;
        let num_kv_heads = req_u32("attention.head_count_kv")?;
        let head_dim = opt_u32("attention.key_length").unwrap_or(hidden_size / num_heads);
        let num_layers = req_u32("block_count")?;
        let context_length = req_u32("context_length")?;
        let rms_norm_eps = req_f32("attention.layer_norm_rms_epsilon")?;
        let ffn_inner = req_u32("feed_forward_length")?;
        let rope_freq_base = opt_f32("rope.freq_base").unwrap_or(10_000.0);
        let pooling_type = file.metadata_u32(&key("pooling_type")).unwrap_or(0);
        let vocab_size = file
            .info("token_embd.weight")
            .ok()
            .and_then(|ti| ti.dims.first().copied())
            .map(|v| v as usize)
            .ok_or_else(|| anyhow!("missing tensor `token_embd.weight`"))?;

        Ok(Self {
            arch,
            hidden_size,
            vocab_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            context_length,
            rms_norm_eps,
            ffn_inner,
            rope_freq_base,
            pooling_type,
        })
    }
}

/// Per-layer device-resident weights for one transformer block of a
/// Qwen3-style dense embedding model. Mirrors the tensor naming
/// produced by llama.cpp's GGUF converter for `qwen3` arch:
/// `blk.{L}.attn_{q,k,v,output,norm,q_norm,k_norm}.weight` and
/// `blk.{L}.ffn_{gate,up,down,norm}.weight`.
#[derive(Debug, Clone)]
pub struct EmbeddingLayer {
    pub attn_q: DeviceTensor,
    pub attn_k: DeviceTensor,
    pub attn_v: DeviceTensor,
    pub attn_output: DeviceTensor,
    pub attn_norm: DeviceTensor,
    pub attn_q_norm: DeviceTensor,
    pub attn_k_norm: DeviceTensor,
    pub ffn_gate: DeviceTensor,
    pub ffn_up: DeviceTensor,
    pub ffn_down: DeviceTensor,
    pub ffn_norm: DeviceTensor,
}

/// Loaded embedding model + per-tensor device pointers. Owns all
/// allocations; teardown via `dispose`.
#[derive(Debug)]
pub struct EmbeddingModel {
    pub config: EmbeddingConfig,
    /// Logical device ID this model lives on (caller-supplied; just
    /// echoed back for diagnostics).
    pub device_id: i32,
    pub token_embd: DeviceTensor,
    pub output_norm: DeviceTensor,
    pub layers: Vec<EmbeddingLayer>,
    /// Total bytes uploaded to the device. Logged at boot.
    pub total_bytes: usize,
    disposed: bool,
}

impl EmbeddingModel {
    /// Load all weights from `file` onto `device`. One synchronise at
    /// the end covers the whole upload; per-tensor synchronises are
    /// unnecessary for a one-shot batched HtoD copy.
    pub fn load(file: &GgufFile, device: &HipDevice, device_id: i32) -> Result<Self> {
        device.bind()?;
        let stream = device.default_stream();
        let config = EmbeddingConfig::from_gguf(file).context("parse embedding config")?;

        let mut total_bytes: usize = 0;
        let mut upload = |name: &str| -> Result<DeviceTensor> {
            let info = file
                .info(name)
                .with_context(|| format!("tensor info `{name}`"))?
                .clone();
            let bytes = info.size_in_bytes() as usize;
            let raw = file
                .tensor_raw(name)
                .with_context(|| format!("tensor_raw `{name}`"))?;
            if raw.len() < bytes {
                bail!(
                    "embedding tensor `{name}` mmap slice {} < declared {bytes}",
                    raw.len()
                );
            }
            let ptr = device
                .alloc(bytes)
                .map_err(|e| anyhow!("hipMalloc {bytes} B `{name}`: {e}"))?;
            // SAFETY: `ptr` is a fresh device allocation of `bytes` bytes;
            // `raw` is an mmap view of at least `bytes` host bytes.
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ptr,
                        DevicePtr(raw.as_ptr() as usize),
                        bytes,
                    )
                    .map_err(|e| anyhow!("memcpy `{name}`: {e}"))?;
            }
            total_bytes += bytes;
            Ok(DeviceTensor {
                ptr,
                dtype: info.dtype,
                dims: info.dims,
                bytes,
                name: Arc::from(name),
            })
        };

        let token_embd = upload("token_embd.weight")?;
        let output_norm = upload("output_norm.weight")?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for il in 0..config.num_layers {
            let l = EmbeddingLayer {
                attn_q: upload(&format!("blk.{il}.attn_q.weight"))?,
                attn_k: upload(&format!("blk.{il}.attn_k.weight"))?,
                attn_v: upload(&format!("blk.{il}.attn_v.weight"))?,
                attn_output: upload(&format!("blk.{il}.attn_output.weight"))?,
                attn_norm: upload(&format!("blk.{il}.attn_norm.weight"))?,
                attn_q_norm: upload(&format!("blk.{il}.attn_q_norm.weight"))?,
                attn_k_norm: upload(&format!("blk.{il}.attn_k_norm.weight"))?,
                ffn_gate: upload(&format!("blk.{il}.ffn_gate.weight"))?,
                ffn_up: upload(&format!("blk.{il}.ffn_up.weight"))?,
                ffn_down: upload(&format!("blk.{il}.ffn_down.weight"))?,
                ffn_norm: upload(&format!("blk.{il}.ffn_norm.weight"))?,
            };
            layers.push(l);
        }

        stream.synchronize().context("stream sync after embedding upload")?;

        Ok(Self {
            config,
            device_id,
            token_embd,
            output_norm,
            layers,
            total_bytes,
            disposed: false,
        })
    }

    /// Free all device allocations. Pair with a `&HipDevice` that
    /// addresses the same physical card the model was loaded onto.
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        device.bind()?;
        let mut first_err: Option<anyhow::Error> = None;
        let mut free = |t: &DeviceTensor| {
            // SAFETY: pointer came from `device.alloc` in `load` and
            // hasn't been aliased.
            unsafe {
                if let Err(e) = device.dealloc(t.ptr, t.bytes) {
                    if first_err.is_none() {
                        first_err = Some(anyhow!("dealloc `{}`: {e}", t.name));
                    }
                }
            }
        };
        free(&self.token_embd);
        free(&self.output_norm);
        for l in self.layers.drain(..) {
            free(&l.attn_q);
            free(&l.attn_k);
            free(&l.attn_v);
            free(&l.attn_output);
            free(&l.attn_norm);
            free(&l.attn_q_norm);
            free(&l.attn_k_norm);
            free(&l.ffn_gate);
            free(&l.ffn_up);
            free(&l.ffn_down);
            free(&l.ffn_norm);
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for EmbeddingModel {
    fn drop(&mut self) {
        if !self.disposed && !self.layers.is_empty() {
            tracing::warn!(
                target: "flambeau_qwen3_moe::embedding",
                arch = %self.config.arch,
                layers = self.layers.len(),
                bytes = self.total_bytes,
                "EmbeddingModel dropped without dispose(device); device buffers leaked"
            );
        }
    }
}
