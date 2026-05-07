//! I/O boundary for the forward pass: token embedding gather, output head
//! (RMSNorm + LM-head MMVQ), and host-side argmax.
//!
//! Three entry points:
//! - [`forward_embed_decode_host`] — look up a token embedding row. Host-
//!   side dequantise + upload; per-token cost is negligible vs the layer
//!   stack.
//! - [`forward_output_head_decode`] — final `rmsnorm + mmvq` producing the
//!   logits row.
//! - [`argmax_token_host`] — download the logit row and pick its maximum.
//!   Greedy-only in V1; temperature/top-p land alongside a logit-returning
//!   variant of the forward.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition — every unsafe block is a kernel.launch or \
              memcpy_async over DevicePtrs owned by the session's scratch / weights / \
              KV cache. Buffers live for the whole session; sync is driven by the top- \
              level forward_*_decode/prefill caller."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{
    norm::rmsnorm_quant_q8_1,
    qmatmul::mmvq,
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::{BlockQ8_1, GgmlDType};

use super::common::{mat_shape, qdtype_of, row_bytes_for_dtype};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

#[cfg(feature = "dev_trace")]
fn dev_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}
#[cfg(not(feature = "dev_trace"))]
#[inline(always)]
fn dev_flag(_name: &str) -> bool {
    false
}

// ---------------------------------------------------------------------------
// V1.7.3-e2 — token embedding gather.
// ---------------------------------------------------------------------------

// `row_bytes_for_dtype` moved to `forward::common`.

/// V2.26.a-i7b — persistent host-side scratch for the prefill embed
/// batch path (`forward_embed_prefill_batch`). Sized once per scratch
/// construction to the largest ubatch the session will ever use;
/// grown on demand if a caller asks for more.
///
/// The Rust Vec's address is read by the batched HtoD memcpy AFTER
/// the last host-side dequant iteration completes — dropping the
/// per-token `stream.synchronize()` calls that the legacy per-token
/// path needed to keep its stack-local buffers alive.
pub struct EmbedPrefillHostScratch {
    pub raw: Vec<u8>,
    pub f16: Vec<half::f16>,
}

impl EmbedPrefillHostScratch {
    pub fn with_capacity(max_tokens: usize, row_bytes: usize, hidden: usize) -> Self {
        Self {
            raw: vec![0u8; max_tokens * row_bytes],
            f16: vec![half::f16::ZERO; max_tokens * hidden],
        }
    }
}

/// V2.26.a-i7b — batched embed for `L` prefill tokens. Replaces the
/// per-token loop `for tid in tokens { forward_embed_decode_host(...) }`
/// which under async-PP dispatch issued 2L host-blocking
/// `stream.synchronize()` calls on rank 0 — each of which starved
/// other-lane / other-rank dispatches of the single Rust driver
/// thread (a cross-lane barrier in the same shape as V2.26.a-i5a's
/// `upload_positions_range` fix).
///
/// The batched path:
///   1. Async DtoH each of `L` rows into one persistent host buffer
///      (caller-owned — usually on `FullAttnPrefillScratch` /
///      per-rank scratch).
///   2. Sync ONCE — the downloads all belong to the caller's stream.
///   3. Dequant all `L` rows on the host (straight-line CPU work;
///      no kernel launches).
///   4. Single HtoD copy of the whole batched F16 block to
///      `out_f16`. No sync needed — stream-ordered with the
///      subsequent layer-chain kernels; the host scratch's
///      lifetime is bounded by the caller's `&mut`.
///
/// Result on 9B Q4_1 Mesh<4> L=4096 ubatch=128 u_lanes=2: per-pass
/// embed syncs drop from 2·4096 = 8192 to 1. Freed driver time
/// lets rank 0 issue other-rank work sooner, tightening 1F1B
/// pipeline fill.
pub fn forward_embed_prefill_batch(
    device: &HipDevice,
    stream: &HipStream,
    token_embd: &DeviceTensor,
    token_ids: &[u32],
    out_f16: DevicePtr,
    hidden: usize,
    host_scratch: &mut EmbedPrefillHostScratch,
) -> Result<()> {
    let l = token_ids.len();
    if l == 0 {
        return Ok(());
    }
    if token_embd.dims.len() != 2 {
        bail!(
            "token_embd: expected 2D weight [vocab, hidden], got dims {:?}",
            token_embd.dims
        );
    }
    let vocab = token_embd.dims[0] as usize;
    let w_k = token_embd.dims[1] as usize;
    if w_k != hidden {
        bail!("token_embd inner dim {w_k} != config hidden {hidden}");
    }
    let row_bytes = row_bytes_for_dtype(token_embd.dtype, hidden)?;
    // Grow host scratch on demand.
    let raw_need = l * row_bytes;
    let f16_need = l * hidden;
    if host_scratch.raw.len() < raw_need {
        host_scratch.raw.resize(raw_need, 0);
    }
    if host_scratch.f16.len() < f16_need {
        host_scratch.f16.resize(f16_need, half::f16::ZERO);
    }

    // 1. Async DtoH each row into the batched host buffer. All
    // issued on `stream` back-to-back — no intermediate sync.
    for (i, &tid) in token_ids.iter().enumerate() {
        let t = tid as usize;
        if t >= vocab {
            bail!("token_id {tid} >= vocab {vocab}");
        }
        let src = token_embd.ptr.offset_bytes(t * row_bytes);
        let dst_host = DevicePtr(
            (host_scratch.raw.as_mut_ptr() as usize) + i * row_bytes,
        );
        // SAFETY: src is a valid device offset (bounds-checked via
        // `t < vocab`); host_scratch.raw has >= (i+1)*row_bytes
        // elements; stream is live.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                dst_host,
                src,
                row_bytes,
            )?;
        }
    }
    stream.synchronize()?;  // ONE sync for all L downloads.

    // 2. Dequant all rows on the host. F16 fast-path avoids the
    // F32 round-trip.
    if token_embd.dtype == GgmlDType::F16 {
        // raw[..] IS the F16 representation (just u8 bytes); cast.
        let raw_f16 = bytemuck::cast_slice::<u8, half::f16>(
            &host_scratch.raw[..raw_need],
        );
        host_scratch.f16[..f16_need].copy_from_slice(raw_f16);
    } else {
        for i in 0..l {
            let row_raw = &host_scratch.raw[i * row_bytes..(i + 1) * row_bytes];
            let row_f32 = flambeau_quant::dequantize_to_vec(
                token_embd.dtype,
                row_raw,
                hidden,
            )
            .map_err(|e| anyhow::anyhow!("dequant token_embd row {}: {e}", token_ids[i]))?;
            for (j, v) in row_f32.into_iter().enumerate() {
                host_scratch.f16[i * hidden + j] = half::f16::from_f32(v);
            }
        }
    }

    // 3. Single HtoD upload of the whole batched F16 block.
    let upload_bytes = f16_need * 2;
    // SAFETY: out_f16 has at least `upload_bytes` valid device bytes
    // (caller contract); host_scratch.f16 is >= f16_need elems; the
    // scratch outlives the stream's consumption of this copy per
    // caller's `&mut` guarantee.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            out_f16,
            DevicePtr(host_scratch.f16.as_ptr() as usize),
            upload_bytes,
        )?;
    }
    // No stream.synchronize — stream ordering guarantees the
    // subsequent layer-chain kernels see the upload complete.
    Ok(())
}

/// Look up a single token's embedding row and write it as F16 into
/// `out_f16`. Decode-path only (one token). Path: download the row's raw
/// bytes from `token_embd` on device → dequantise on host → cast to F16 →
/// upload to `out_f16`.
///
/// Per-token cost at Qwen3.6 dims: ~1 KB download + 256-block dequant +
/// ~4 KB upload + two stream syncs. Negligible at any realistic tg
/// throughput. Future optimisation: an on-device `gather_q4_k_to_f16`
/// kernel that bypasses the host roundtrip.
pub fn forward_embed_decode_host(
    device: &HipDevice,
    stream: &HipStream,
    token_embd: &DeviceTensor,
    token_id: u32,
    out_f16: DevicePtr,
    hidden: usize,
) -> Result<()> {
    if token_embd.dims.len() != 2 {
        bail!(
            "token_embd: expected 2D weight [vocab, hidden], got dims {:?}",
            token_embd.dims
        );
    }
    let vocab = token_embd.dims[0] as usize;
    let w_k = token_embd.dims[1] as usize;
    if w_k != hidden {
        bail!(
            "token_embd inner dim {w_k} != config hidden {hidden}"
        );
    }
    if (token_id as usize) >= vocab {
        bail!("token_id {token_id} >= vocab {vocab}");
    }

    let row_bytes = row_bytes_for_dtype(token_embd.dtype, hidden)?;
    let offset = token_id as usize * row_bytes;
    if offset + row_bytes > token_embd.bytes {
        bail!(
            "token_embd row out of bounds: token_id={token_id} row_bytes={row_bytes} \
             total_bytes={}",
            token_embd.bytes
        );
    }

    // 1. Download the row's raw bytes.
    let mut row_raw = vec![0u8; row_bytes];
    let src = token_embd.ptr.offset_bytes(offset);
    // SAFETY: `src` points to at least `row_bytes` valid device bytes
    // (checked above); `row_raw` is a host vec of the same length.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(row_raw.as_mut_ptr() as usize),
            src,
            row_bytes,
        )?;
    }
    stream.synchronize()?;

    // 2. Dequantise on host. F16 fast-path avoids the F32 round-trip.
    let row_f16: Vec<half::f16> = if token_embd.dtype == GgmlDType::F16 {
        bytemuck::cast_slice::<u8, half::f16>(&row_raw).to_vec()
    } else {
        let row_f32 =
            flambeau_quant::dequantize_to_vec(token_embd.dtype, &row_raw, hidden)
                .map_err(|e| anyhow::anyhow!("dequant token_embd row {token_id}: {e}"))?;
        row_f32
            .into_iter()
            .map(half::f16::from_f32)
            .collect()
    };
    drop(row_raw);

    // 3. Upload to the F16 scratch slot.
    let upload_bytes = hidden * 2;
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            out_f16,
            DevicePtr(row_f16.as_ptr() as usize),
            upload_bytes,
        )?;
    }
    stream.synchronize()?;
    drop(row_f16);
    Ok(())
}

// ---------------------------------------------------------------------------
// V1.7.3-e3 — output norm + LM head + argmax sampling.
// ---------------------------------------------------------------------------

/// Workspace for the output / LM head path. Sized against
/// `(hidden, vocab_size)`.
pub struct OutputHeadScratch {
    pub x_norm_f16: DevicePtr,   // [hidden] F16, rmsnorm(output_norm, x_final)
    pub x_q8_1: DevicePtr,       // Q8_1 of x_norm for the LM head mmvq
    pub logits_f32: DevicePtr,   // [vocab] F32
    // Bookkeeping.
    x_norm_bytes: usize,
    x_q8_1_bytes: usize,
    logits_bytes: usize,
    disposed: bool,
}

impl OutputHeadScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");

        let x_norm_bytes = hidden * 2;
        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let logits_bytes = vocab * 4;

        let x_norm_f16 = device.alloc(x_norm_bytes)?;
        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let logits_f32 = device.alloc(logits_bytes)?;

        Ok(Self {
            x_norm_f16,
            x_q8_1,
            logits_f32,
            x_norm_bytes,
            x_q8_1_bytes,
            logits_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_norm_f16, self.x_norm_bytes)?;
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.logits_f32, self.logits_bytes)?;
        }
        Ok(())
    }
}

impl Drop for OutputHeadScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "OutputHeadScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Run the post-last-layer tail: output rmsnorm → LM head mmvq → logits F32.
///
/// `lm_head_weight`: either the untied `output.weight` (when present) or
/// the tied `token_embd.weight`. Expected shape (outermost-first):
/// `[vocab, hidden]`.
///
/// On return, `scratch.logits_f32` holds `[vocab]` F32 logits.
pub fn forward_output_head_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    output_norm: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    scratch: &mut OutputHeadScratch,
    x_final: DevicePtr,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    // 1. Final rmsnorm + Q8_1 quantise in one fused launch.
    rmsnorm_quant_q8_1(
        ops,
        stream,
        x_final,
        output_norm.ptr,
        scratch.x_q8_1,
        1,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("output_norm + quant")?;

    // 2. LM head mmvq → F32 logits.
    let dtype = qdtype_of(lm_head_weight.dtype)?;
    let (rows, k) = mat_shape(lm_head_weight)?;
    if rows != vocab || k != hidden {
        bail!(
            "lm_head weight shape [{rows}, {k}] != expected [{vocab}, {hidden}]"
        );
    }
    mmvq(
        ops,
        stream,
        lm_head_weight.ptr,
        scratch.x_q8_1,
        scratch.logits_f32,
        rows,
        k,
        dtype,
    )
    .context("lm_head mmvq")?;

    Ok(())
}

/// Host-side argmax sampler over the F32 logits produced by
/// [`forward_output_head_decode`]. Downloads `[vocab]` F32s to host and
/// scans for the maximum.
///
/// V1 intentionally keeps sampling CPU-side: vocab × 4 bytes is tiny
/// (Qwen3.6: 248320 × 4 ≈ 970 KB — a single PCIe memcpy + ~1 ms argmax).
/// Temperature / top-p sampling is V2.
pub fn argmax_token_host(
    device: &HipDevice,
    stream: &HipStream,
    logits: DevicePtr,
    vocab: usize,
) -> Result<u32> {
    let mut host = vec![0.0f32; vocab];
    // SAFETY: `logits` points to at least `vocab * 4` valid device bytes
    // (caller's contract — OutputHeadScratch sizes it against cfg.vocab_size).
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            logits,
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
    // V1.7.4.a diagnostic: set FLAMBEAU_PARITY_TOPK_LOGITS to dump the
    // top-20 argmax + rank of llama.cpp's top-7 baseline tokens. Lets us
    // tell "F16 noise, llama's #1 is in our top 20" from "systematic bug,
    // llama's #1 is rank 200k+". Left in because the parity gap isn't
    // closed yet and the harness is cheap.
    if dev_flag("FLAMBEAU_PARITY_TOPK_LOGITS") {
        let mut idxs: Vec<usize> = (0..host.len()).collect();
        idxs.sort_by(|&a, &b| host[b].partial_cmp(&host[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = idxs.iter().take(20).map(|&i| (i, host[i])).collect();
        eprintln!("[argmax-topk] top-20 = {top:?}");
        // Show where llama.cpp's top-7 ids at pos-0 land in our ranking.
        // Reference generated via `llama-server` POST /completion on the
        // same GGUF with `return_tokens: true, n_probs: 10`.
        let lcpp_top = [11u32, 4858, 0, 1017, 660, 13, 353];
        for tok in lcpp_top {
            let rank = idxs.iter().position(|&i| i == tok as usize).unwrap_or(usize::MAX);
            eprintln!(
                "[argmax-topk] llama.cpp token {tok} → our logit={} rank={}",
                host[tok as usize], rank
            );
        }
    }
    Ok(best_idx as u32)
}

/// Host-side download of the F32 logit row produced by
/// [`forward_output_head_decode`]. Fills `out` with exactly `vocab` F32
/// values; any prior contents are replaced. `out.capacity() >= vocab`
/// avoids a reallocation on the hot path.
///
/// Used by the sampler path in the HTTP server (`/v1/chat/completions`
/// with `temperature > 0` / `top_p < 1`). Greedy callers should keep
/// using [`argmax_token_host`] to avoid the vocab-sized memcpy + clone.
pub fn download_logits_host(
    device: &HipDevice,
    stream: &HipStream,
    logits: DevicePtr,
    vocab: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    out.resize(vocab, 0.0);
    // SAFETY: `logits` points to at least `vocab * 4` valid device bytes
    // (caller's contract — OutputHeadScratch sizes it against cfg.vocab_size).
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            logits,
            vocab * 4,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

