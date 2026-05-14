//! **#230 P2.11a / #231 P2.11b** — embedding model loader + pooled
//! forward for `general.architecture = "qwen3"` GGUFs (Qwen3-Embedding
//! family: 0.6B / 4B / 8B).
//! Architectural reuse: the qwen3 dense arch maps cleanly to the
//! existing `AttentionFamily::Dense` + dense-FFN path that the
//! layout / loader / forward stack already supports. `qwen35moe` and
//! `qwen36moe` (Hybrid GDN) chat models stay on the `Hybrid` family
//! and are unaffected. We just thread `qwen3` as a third arch in
//! `Qwen3MoEConfig::from_gguf` and reuse `Qwen3MoEModel` end-to-end
//! for storage. The piece this module owns is the pooled-embedding
//! forward (`compute_pooled_embedding`): mirrors `forward_prefill`
//! through the per-layer loop, then replaces the lm-head with a
//! final RMSNorm + L2-normalize on the LAST token's hidden vector
//! and downloads the result as `Vec<f32>`.
//! V1 scope:
//! - Single-device only. Qwen3-Embedding-0.6B fits trivially next to
//! a 27B chat shard on 16 GB MI50; multi-device sharding is V2.
//! - Single text per request — batch over multiple inputs is V2.
//! - Last-token pooling (`pooling_type = 3`, what Qwen3-Embedding ships
//! with). Mean / CLS pooling are V2.
//! - Output is L2-normalised F32 (the canonical embedding format).

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::cast::cast_f16_to_f32;
use flambeau_ops::hip::norm::{l2_norm_f32, rmsnorm_f16};
use flambeau_ops::hip::{HipDevice, HipStream};
use flambeau_quant::{GgmlDType, GgufFile};

use crate::weights::DeviceTensor;

use crate::forward::{
    forward_embed_decode_host, forward_layer_prefill, ForwardPrefillScratch,
};
use crate::model::Qwen3MoEModel;
use crate::session::Qwen3MoESession;

/// Loaded embedding model, single-device.
/// Wraps a `Qwen3MoEModel` (which handles `qwen3` arch via
/// `AttentionFamily::Dense` after the #231 config extension) plus
/// per-request session + scratch state allocated lazily on the first
/// inference call.
/// Concurrent requests must serialise externally — the inner session
/// + scratch are not thread-safe and one call mutates KV cache
/// counters during the per-layer loop (we discard the cache between
/// requests, but the slots themselves are shared scratch).
pub struct EmbeddingModel {
    /// Inner model is `Option` so dispose can move it out without
    /// fighting `Drop`. Always `Some` outside of `dispose()`.
    inner: Option<Qwen3MoEModel>,
    /// Logical device id this model lives on; echoed in logs.
    pub device_id: i32,
    /// Reusable pooled-forward scratch sized for `max_tokens` input
    /// length. Lazy on first `compute_pooled_embedding` to avoid
    /// up-front VRAM cost when the operator loaded the model but
    /// hasn't called the endpoint yet.
    scratch: Option<EmbeddingScratch>,
    /// Reusable session — KV cache is unused (we reset between
    /// requests) but the per-layer `LayerCache` allocations are
    /// needed for the `forward_layer_prefill` signature.
    session: Option<Qwen3MoESession>,
    /// Maximum input tokens accepted by `compute_pooled_embedding`.
    /// Sized at construction; `compute` errors if exceeded.
    pub max_tokens: usize,
    disposed: bool,
}

impl EmbeddingModel {
    pub fn inner(&self) -> &Qwen3MoEModel {
        self.inner
            .as_ref()
            .expect("EmbeddingModel.inner is None — dispose() called")
    }

    /// Bytes uploaded to the device — reads from the inner model's
    /// weights total. Logged at boot for diagnostics.
    pub fn total_bytes(&self) -> usize {
        self.inner().weights.total_bytes()
    }

    /// Architecture string from the GGUF (`qwen3`).
    pub fn arch(&self) -> &str {
        &self.inner().config.arch
    }

    /// Hidden size = embedding dimension returned to the client.
    pub fn hidden_size(&self) -> usize {
        self.inner().config.hidden_size
    }

    /// Vocabulary size — informational; the tokenizer for a separate
    /// embedding model lives in the GGUF itself but #231 reuses the
    /// chat tokenizer (Qwen3-Embedding shares the Qwen3 tokenizer).
    pub fn vocab_size(&self) -> usize {
        self.inner().config.vocab_size
    }

    /// Pooling type tag from the GGUF (`qwen3.pooling_type`). 3 =
    /// last-token pooling, which is what `compute_pooled_embedding`
    /// implements. Other values currently fall through to last-token
    /// behaviour with a warn.
    pub fn pooling_type(&self) -> u32 {
        self.inner().config.pooling_type.unwrap_or(3)
    }
}

/// Reusable scratch for pooled-embedding inference.
pub struct EmbeddingScratch {
    /// `Option` so dispose can move it out without leaving a half-
    /// disposed value that triggers `ForwardPrefillScratch::Drop`'s
    /// warn.
    prefill: Option<ForwardPrefillScratch>,
    /// F16 [hidden] — RMSNorm output for the last-token hidden state.
    pooled_norm_f16: DevicePtr,
    /// F32 [hidden] — `cast_f16_to_f32` of the above.
    pooled_f32: DevicePtr,
    /// F32 [hidden] — L2-normalised result (DtoH'd to the response).
    pooled_normed_f32: DevicePtr,
    hidden_bytes_f16: usize,
    hidden_bytes_f32: usize,
    disposed: bool,
}

impl EmbeddingScratch {
    pub fn new(model: &Qwen3MoEModel, device: &HipDevice, max_tokens: usize) -> Result<Self> {
        let hidden = model.config.hidden_size;
        let prefill = ForwardPrefillScratch::new(&model.config, device, max_tokens)
            .context("EmbeddingScratch: alloc ForwardPrefillScratch")?;
        let hidden_bytes_f16 = hidden * 2;
        let hidden_bytes_f32 = hidden * 4;
        let pooled_norm_f16 = device
            .alloc(hidden_bytes_f16)
            .map_err(|e| anyhow!("alloc pooled_norm_f16: {e}"))?;
        let pooled_f32 = device
            .alloc(hidden_bytes_f32)
            .map_err(|e| anyhow!("alloc pooled_f32: {e}"))?;
        let pooled_normed_f32 = device
            .alloc(hidden_bytes_f32)
            .map_err(|e| anyhow!("alloc pooled_normed_f32: {e}"))?;
        Ok(Self {
            prefill: Some(prefill),
            pooled_norm_f16,
            pooled_f32,
            pooled_normed_f32,
            hidden_bytes_f16,
            hidden_bytes_f32,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.pooled_norm_f16, self.hidden_bytes_f16)?;
            device.dealloc(self.pooled_f32, self.hidden_bytes_f32)?;
            device.dealloc(self.pooled_normed_f32, self.hidden_bytes_f32)?;
        }
        if let Some(prefill) = self.prefill.take() {
            prefill.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for EmbeddingScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::embedding",
                "EmbeddingScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

impl EmbeddingModel {
    /// Load all weights from `file` onto `device`.
    /// `max_tokens` caps the input length the scratch will support;
    /// the scratch + session are allocated lazily on first inference,
    /// so this method just uploads weights + builds the OpsRegistry.
    pub fn load(
        file: &GgufFile,
        device: &HipDevice,
        device_id: i32,
        max_tokens: usize,
    ) -> Result<Self> {
        device.bind()?;
        let mut inner =
            Qwen3MoEModel::load(file, device).context("Qwen3MoEModel::load (embedding)")?;
        // **#231 quality fix** — `Qwen3MoEModel::load` (the single-
        // device loader) uploads norm weights verbatim, but
        // `rmsnorm_f16` reinterprets the buffer as F16. Qwen3-Embedding
        // GGUFs ship every norm as F32; without the cast, byte-level
        // reinterpret turns half the F16 lookups into 0.0 and the
        // model's output picks up an alternating-zeros pattern (every
        // other dim ends up exactly 0). The chat path's sharded loader
        // (`upload_as_f16` in sharded.rs) handles this; the single-
        // device loader doesn't. Mirror that cast post-load here.
        cast_f32_norms_to_f16(&mut inner.weights, file, device).context(
            "embedding load: cast F32 norm weights to F16 (matches sharded loader's up_f16)",
        )?;
        if inner.config.arch != "qwen3" {
            // Defensive: the embedding API is only useful on a
            // qwen3-arch model. A chat GGUF (qwen35moe / etc) would
            // technically load (post #231 the Hybrid family path
            // handles it) but semantically isn't an embedding model.
            let arch = inner.config.arch.clone();
            inner.dispose(device).ok();
            bail!(
                "embedding model rejected: arch `{arch}` is not `qwen3`. \
                 Use a Qwen3-Embedding GGUF (general.architecture=qwen3) here."
            );
        }
        Ok(Self {
            inner: Some(inner),
            device_id,
            scratch: None,
            session: None,
            max_tokens,
            disposed: false,
        })
    }

    /// Lazy-init or reuse the scratch + session on `device`. Both
    /// must live on the same device the model was loaded on (the
    /// chat cluster's HipDevice handle, in the server's case).
    fn ensure_scratch(&mut self, device: &HipDevice) -> Result<()> {
        if self.scratch.is_none() {
            let model = self.inner();
            self.scratch = Some(EmbeddingScratch::new(model, device, self.max_tokens)?);
        }
        if self.session.is_none() {
            let mut cfg_for_session = self.inner().config.clone();
            // Cap session ctx at max_tokens — saves VRAM vs the
            // GGUF-native 32k.
            if cfg_for_session.context_length > self.max_tokens {
                cfg_for_session.context_length = self.max_tokens;
            }
            // Embedding pooling never reads KV; F16 layout is fine here.
            let session = Qwen3MoESession::new(
                &cfg_for_session,
                device,
                crate::session::KvLayout::F16,
            )
            .context("EmbeddingModel::ensure_scratch: alloc Qwen3MoESession")?;
            self.session = Some(session);
        }
        Ok(())
    }

    /// **#231** — run pooled-embedding inference on a single sequence
    /// of token ids. Returns an L2-normalised F32 vector of length
    /// `hidden_size`.
    /// Errors when `tokens.len() > max_tokens` or `tokens` is empty.
    /// Implementation:
    /// 1. Reset session (clear KV `current_tokens`).
    /// 2. Embed L tokens row-by-row into `prefill.hidden_a.ptr()`.
    /// 3. Per-layer loop: `forward_layer_prefill` ping-ponging
    /// `(hidden_a, hidden_b)`.
    /// 4. Take the LAST token's F16 hidden vector (offset
    /// `(L-1) * hidden * 2` into the final ping-pong buffer).
    /// 5. RMSNorm with `output_norm` → `pooled_norm_f16`.
    /// 6. `cast_f16_to_f32` → `pooled_f32`.
    /// 7. `l2_norm_f32` → `pooled_normed_f32`.
    /// 8. DtoH the F32 vector and return.
    pub fn compute_pooled_embedding(
        &mut self,
        device: &HipDevice,
        stream: &HipStream,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            bail!("compute_pooled_embedding: empty tokens");
        }
        if tokens.len() > self.max_tokens {
            bail!(
                "compute_pooled_embedding: L={} > max_tokens={}",
                tokens.len(),
                self.max_tokens
            );
        }
        device.bind()?;
        self.ensure_scratch(device)?;

        // Take immutable handles to the inner model first so we don't
        // double-borrow `self` when destructuring scratch + session.
        let inner_ref = self
            .inner
            .as_ref()
            .expect("compute_pooled_embedding called after dispose");
        let cfg = inner_ref.config.clone();
        let hidden = cfg.hidden_size;
        let row_bytes = hidden * 2;
        let l = tokens.len();

        let scratch = self
            .scratch
            .as_mut()
            .context("ensure_scratch left scratch=None")?;
        let session = self
            .session
            .as_mut()
            .context("ensure_scratch left session=None")?;

        // 1. Reset session.
        session
            .reset_for_next_request(device)
            .context("reset session for embedding request")?;

        let prefill = scratch
            .prefill
            .as_mut()
            .context("EmbeddingScratch.prefill missing (disposed?)")?;

        // 2. Embed L tokens row-by-row.
        for (t, &token_id) in tokens.iter().enumerate() {
            forward_embed_decode_host(
                device,
                stream,
                &inner_ref.weights.token_embd,
                token_id,
                prefill.hidden_a.offset_bytes(t * row_bytes),
                hidden,
            )?;
        }

        // 3. Per-layer loop with ping-pong.
        let layer_scratch = prefill
            .layer
            .as_mut()
            .context("ForwardPrefillScratch.layer missing")?;
        let (mut x_in, mut x_out) = (prefill.hidden_a.ptr(), prefill.hidden_b.ptr());
        for (il, layer_weights) in inner_ref.weights.layers.iter().enumerate() {
            let layer_cache = &mut session.layers_mut()[il];
            forward_layer_prefill(
                &inner_ref.ops,
                stream,
                device,
                &cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                l,
                0,    // start_position
                None, // no per-position-id override
                None, // no GDN-state event (Dense family has no GDN)
            )?;
            std::mem::swap(&mut x_in, &mut x_out);
            let _ = il;
        }
        // x_in now holds the final F16 [L, hidden].

        // 4-7. Last-token slice → RMSNorm → cast → L2 norm.
        let last_token_hidden = x_in.offset_bytes((l - 1) * row_bytes);
        rmsnorm_f16(
            &inner_ref.ops,
            stream,
            last_token_hidden,
            inner_ref.weights.output_norm.ptr,
            scratch.pooled_norm_f16,
            1,
            hidden,
            cfg.rms_norm_eps,
        )
        .context("output_norm rmsnorm on pooled hidden")?;
        cast_f16_to_f32(
            &inner_ref.ops,
            stream,
            scratch.pooled_norm_f16,
            scratch.pooled_f32,
            hidden,
        )
        .context("cast pooled F16 -> F32")?;
        l2_norm_f32(
            &inner_ref.ops,
            stream,
            scratch.pooled_f32,
            scratch.pooled_normed_f32,
            1,
            hidden,
            1e-12,
        )
        .context("l2_norm_f32 pooled vector")?;

        // 8. DtoH.
        let mut host = vec![0f32; hidden];
        // SAFETY: device buffer sized to `hidden * 4` bytes (F32);
        // host buffer sized to match. DtoH async + sync.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                DevicePtr(host.as_mut_ptr() as usize),
                scratch.pooled_normed_f32,
                hidden * 4,
            )?;
        }
        stream.synchronize()?;
        Ok(host)
    }

    /// Free device weights + scratch + session.
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        device.bind()?;
        if let Some(s) = self.scratch.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.session.take() {
            s.dispose(device)?;
        }
        if let Some(inner) = self.inner.take() {
            inner.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for EmbeddingModel {
    fn drop(&mut self) {
        if !self.disposed && self.inner.is_some() {
            tracing::warn!(
                target: "flambeau_qwen3_moe::embedding",
                "EmbeddingModel dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// **#231** — re-upload every F32 norm weight as F16 in place. The
/// single-device `Qwen3MoEModel::load` skips this cast (the sharded
/// loader doesn't), but `rmsnorm_f16` byte-reinterprets the buffer
/// as F16 and silently corrupts every other lookup when the source
/// is F32. Walks: `output_norm`, per-layer `attn_norm`, `ffn_norm`,
/// `attn.Dense.{attn_q_norm, attn_k_norm}`. No-op for tensors already
/// F16.
fn cast_f32_norms_to_f16(
    weights: &mut crate::weights::ModelWeights,
    file: &GgufFile,
    device: &HipDevice,
) -> Result<()> {
    device.bind()?;
    let stream = device.default_stream();
    cast_one_norm(&mut weights.output_norm, file, device, stream)
        .context("output_norm")?;
    for (il, layer) in weights.layers.iter_mut().enumerate() {
        cast_one_norm(&mut layer.attn_norm, file, device, stream)
            .with_context(|| format!("layer {il} attn_norm"))?;
        if let Some(t) = layer.ffn_norm.as_mut() {
            cast_one_norm(t, file, device, stream)
                .with_context(|| format!("layer {il} ffn_norm"))?;
        }
        if let Some(t) = layer.post_attention_norm.as_mut() {
            cast_one_norm(t, file, device, stream)
                .with_context(|| format!("layer {il} post_attention_norm"))?;
        }
        if let crate::weights::AttnWeights::Dense(d) = &mut layer.attn {
            cast_one_norm(&mut d.attn_q_norm, file, device, stream)
                .with_context(|| format!("layer {il} attn_q_norm"))?;
            cast_one_norm(&mut d.attn_k_norm, file, device, stream)
                .with_context(|| format!("layer {il} attn_k_norm"))?;
        }
    }
    stream
        .synchronize()
        .context("sync after norm F32→F16 fixup")?;
    Ok(())
}

/// Cast one F32 `DeviceTensor` to F16 in place: alloc fresh F16
/// buffer, host-side convert, upload, free old buffer, swap pointers.
/// Pass-through on F16/non-F32 tensors.
fn cast_one_norm(
    t: &mut DeviceTensor,
    file: &GgufFile,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    if t.dtype != GgmlDType::F32 {
        return Ok(());
    }
    let raw = file
        .tensor_raw(&t.name)
        .with_context(|| format!("tensor_raw `{}`", t.name))?;
    let elems: usize = t.dims.iter().product::<u64>() as usize;
    if raw.len() < elems * 4 {
        bail!(
            "cast_one_norm: `{}` mmap slice {} < expected {}",
            t.name,
            raw.len(),
            elems * 4
        );
    }
    // SAFETY: tensor.dtype == F32 means the GGUF metadata declares F32
    // bytes; bytemuck::cast_slice will only reinterpret if alignment
    // is satisfied. The mmap is page-aligned which exceeds 4-byte
    // alignment.
    let src: &[f32] = bytemuck::cast_slice(&raw[..elems * 4]);
    let host: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
    let new_bytes = elems * 2;
    let new_ptr = device
        .alloc(new_bytes)
        .map_err(|e| anyhow!("alloc F16 norm `{}`: {e}", t.name))?;
    // SAFETY: `new_ptr` is a fresh device alloc of `new_bytes`; `host`
    // owns `new_bytes` host bytes for the duration of memcpy + the
    // outer caller's stream sync.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                new_ptr,
                DevicePtr(host.as_ptr() as usize),
                new_bytes,
            )
            .map_err(|e| anyhow!("memcpy F32→F16 `{}`: {e}", t.name))?;
    }
    stream.synchronize()?;
    drop(host);
    // Free the old F32 buffer.
    let old_ptr = t.ptr;
    let old_bytes = t.bytes;
    // SAFETY: `old_ptr` came from `device.alloc` in the original load;
    // we've owned exclusive access since.
    unsafe {
        device
            .dealloc(old_ptr, old_bytes)
            .map_err(|e| anyhow!("dealloc F32 norm `{}`: {e}", t.name))?;
    }
    // Replace pointer + dtype + bytes — dims unchanged.
    t.ptr = new_ptr;
    t.dtype = GgmlDType::F16;
    t.bytes = new_bytes;
    Ok(())
}
