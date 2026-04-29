//! Sampler-D3 Phase A — server-side hook that runs the GPU top-K +
//! softmax kernel on the head-rank logits in place after the existing
//! forward path, then DtoH only the K-tuple.
//!
//! This still pays the 600 KB host-logits DtoH that the existing
//! `decode_logits` / `prefill_logits` does (Phase A's marginal-win
//! tradeoff). Phase B will add a `forward_*_topk` variant that skips
//! that DtoH.
//!
//! Gated behind `FLAMBEAU_GPU_SAMPLER=1` so the path is opt-in until
//! the cert + bench-A/B confirm parity at the server level.

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::HipCluster;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::sampling::{topk_softmax_f32, SAMPLER_K_OUT_MAX};

use crate::model::{Inflight, LoadedModel};

/// Per-request scratch for the GPU sampler. Allocates two K-element
/// buffers on the head rank's device (8 bytes total) reused across
/// every decode step in the request.
pub struct GpuSamplerScratch {
    head_dev_idx: usize,
    d_ids: DevicePtr,
    d_probs: DevicePtr,
    pub k: usize,
    pub host_ids: Vec<u32>,
    pub host_probs: Vec<f32>,
    disposed: bool,
}

impl GpuSamplerScratch {
    /// Allocate on the head rank of `cluster`. `k` must be in
    /// `[1, SAMPLER_K_OUT_MAX]`.
    pub fn new(cluster: &HipCluster, head_rank: usize, k: usize) -> Result<Self> {
        if k == 0 || k > SAMPLER_K_OUT_MAX {
            bail!(
                "GpuSamplerScratch::new: k={k} must be in [1, {SAMPLER_K_OUT_MAX}]"
            );
        }
        let dev = cluster.device(head_rank);
        dev.bind()?;
        let d_ids = dev.alloc(k * 4)?;
        let d_probs = dev.alloc(k * 4)?;
        Ok(Self {
            head_dev_idx: head_rank,
            d_ids,
            d_probs,
            k,
            host_ids: vec![0u32; k],
            host_probs: vec![0.0f32; k],
            disposed: false,
        })
    }

    /// Free the device buffers. Must pair with `new(cluster, ...)`.
    pub fn dispose(mut self, cluster: &HipCluster) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let dev = cluster.device(self.head_dev_idx);
        dev.bind()?;
        // SAFETY: d_ids / d_probs were allocated by `new` for `k * 4`
        // bytes each on `dev`; we deallocate the same regions exactly
        // once (guarded by `disposed`).
        unsafe {
            dev.dealloc(self.d_ids, self.k * 4)?;
            dev.dealloc(self.d_probs, self.k * 4)?;
        }
        Ok(())
    }
}

impl Drop for GpuSamplerScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_server::gpu_sampler",
                "GpuSamplerScratch dropped without dispose(cluster); device buffers leaked"
            );
        }
    }
}

/// Run `topk_softmax_f32` on the head rank's `logits_f32` buffer (the
/// device-resident output of the existing forward pass), then DtoH the
/// K-tuple into `scratch.host_ids` / `scratch.host_probs`.
///
/// Caller must invoke this immediately after `decode_logits` /
/// `prefill_logits` returns — the device pointer to logits is only
/// guaranteed-valid until the next forward call clobbers it.
///
/// # Errors
/// - Topology unsupported (Phase A wires TP only).
/// - Topk kernel-launch / DtoH failure.
pub fn run_gpu_topk(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &Inflight,
    scratch: &mut GpuSamplerScratch,
    inv_temp: f32,
) -> Result<()> {
    match (model, inflight) {
        (LoadedModel::Tp { model: m, .. }, Inflight::Tp { decode, .. }) => {
            let head = decode.head_rank.0 as usize;
            if head != scratch.head_dev_idx {
                bail!(
                    "run_gpu_topk: scratch head_dev_idx={} != decode.head_rank={head}",
                    scratch.head_dev_idx
                );
            }
            let dev = cluster.device(head);
            let ops = &m.ops[head];
            let head_scratch = decode.per_rank[head]
                .output_head
                .as_ref()
                .ok_or_else(|| anyhow!("head rank missing OutputHeadScratch"))?;
            let logits_f32 = head_scratch.logits_f32;
            let vocab = m.config.vocab_size;
            dev.bind()?;
            let stream = dev.default_stream();
            topk_softmax_f32(
                ops,
                stream,
                logits_f32,
                scratch.d_ids,
                scratch.d_probs,
                vocab,
                scratch.k,
                inv_temp,
            )
            .context("GPU topk_softmax_f32")?;
            // SAFETY: d_ids/d_probs were allocated by GpuSamplerScratch::new
            // for `k * 4` bytes each on this device; host_ids/host_probs
            // are vecs of matching capacity.
            unsafe {
                <flambeau_backend_hip::HipDevice as Device>::memcpy_async(
                    dev,
                    stream,
                    CopyDirection::DeviceToHost,
                    DevicePtr(scratch.host_ids.as_mut_ptr() as usize),
                    scratch.d_ids,
                    scratch.k * 4,
                )?;
                <flambeau_backend_hip::HipDevice as Device>::memcpy_async(
                    dev,
                    stream,
                    CopyDirection::DeviceToHost,
                    DevicePtr(scratch.host_probs.as_mut_ptr() as usize),
                    scratch.d_probs,
                    scratch.k * 4,
                )?;
            }
            Stream::synchronize(stream)?;
            Ok(())
        }
        _ => bail!(
            "GPU sampler is wired for TP topology only in Phase A (got {})",
            model.topology()
        ),
    }
}

/// Apply the stop-token mask to `(host_ids, host_probs)` AFTER the GPU
/// topk has populated them. Sets matching entries' probs to 0.0; the
/// caller's `Sampler::sample_from_topk` renormalises when it sums for
/// the multinomial draw.
pub fn apply_stop_mask(
    host_ids: &[u32],
    host_probs: &mut [f32],
    stop_ids: &[u32],
) {
    if stop_ids.is_empty() {
        return;
    }
    for (i, &id) in host_ids.iter().enumerate() {
        if stop_ids.contains(&id) {
            host_probs[i] = 0.0;
        }
    }
}
