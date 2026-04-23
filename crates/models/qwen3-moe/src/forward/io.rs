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

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{
    norm::{quantize_f16_q8_1, rmsnorm_f16, rmsnorm_quant_q8_1},
    qmatmul::mmvq,
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::{BlockQ8_1, GgmlDType};
use half::f16;

use super::common::{mat_shape, qdtype_of, row_bytes_for_dtype};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

// ---------------------------------------------------------------------------
// V1.7.3-e2 — token embedding gather.
// ---------------------------------------------------------------------------

// `row_bytes_for_dtype` moved to `forward::common`.

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
    if std::env::var("FLAMBEAU_PARITY_TOPK_LOGITS").is_ok() {
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

