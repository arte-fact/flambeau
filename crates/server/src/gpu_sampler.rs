//! Sampler-D3 Phase A — server-side hook that runs the GPU top-K +
//! softmax kernel on the head-rank logits in place after the existing
//! forward path, then DtoH only the K-tuple.
//! This still pays the 600 KB host-logits DtoH that the existing
//! `decode_logits` / `prefill_logits` does (Phase A's marginal-win
//! tradeoff). Phase B will add a `forward_*_topk` variant that skips
//! that DtoH.
//! Gated behind `FLAMBEAU_GPU_SAMPLER=1` so the path is opt-in until
//! the cert + bench-A/B confirm parity at the server level.

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipCluster, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::sampling::{apply_penalties_f32, topk_softmax_f32, SAMPLER_K_OUT_MAX};
use flambeau_runtime::Sampling;

use crate::qwen3moe_handle::{Inflight, LoadedModel, Qwen3MoeModelExt};

/// **Sampler-D4 (#212)** — upper bound on the number of unique tokens
/// the GPU penalty kernel can apply per call. The deduped history
/// `(tok, count)` pair count must fit. 8192 covers any realistic chat
/// turn (max_tokens caps at 8192 already; uniqueness usually pushes
/// this much lower).
pub const SAMPLER_HISTORY_MAX: usize = 8192;

/// Per-request scratch for the GPU sampler. Allocates two K-element
/// buffers on the head rank's device for the topk output (~2 KB) plus
/// (lazily, only when penalties are active) a SAMPLER_HISTORY_MAX-sized
/// `(tok, count)` u32 pair buffer (~64 KB). All buffers are reused
/// across every decode step in the request.
pub struct GpuSamplerScratch {
    head_dev_idx: usize,
    d_ids: DevicePtr,
    d_probs: DevicePtr,
    pub k: usize,
    pub host_ids: Vec<u32>,
    pub host_probs: Vec<f32>,
    /// Sampler-D4 — device buffer for `(tok, count)` u32 pairs,
    /// 8 bytes each. Allocated only when penalties are active.
    /// Capacity = SAMPLER_HISTORY_MAX pairs.
    d_history_counts: Option<DevicePtr>,
    /// Reused host scratch for sort+dedup (mirrors Sampler-F's
    /// per-session scratch but lives here so server-side can drive it).
    host_history_sorted: Vec<u32>,
    host_history_counts: Vec<(u32, u32)>,
    /// Pre-flattened upload staging buffer: `[tok0, count0, tok1, ...]`.
    host_history_pairs: Vec<u32>,
    disposed: bool,
}

impl GpuSamplerScratch {
    /// Allocate on `device`. `k` must be in `[1, SAMPLER_K_OUT_MAX]`.
    /// The penalty-path device buffer is allocated lazily on the first
    /// `run_gpu_topk_with_penalties` call so penalty-free requests
    /// don't pay for the 64 KB.
    /// The caller picks the device — TP path passes the global cluster's
    /// `head_rank` device; Hybrid passes the head stage's sub_cluster's
    /// `head_rank` device. The scratch records the device's HIP id so
    /// dispose / penalty-buffer ensure can rebind without re-resolving
    /// from a cluster handle.
    pub fn new(device: &HipDevice, k: usize) -> Result<Self> {
        if k == 0 || k > SAMPLER_K_OUT_MAX {
            bail!(
                "GpuSamplerScratch::new: k={k} must be in [1, {SAMPLER_K_OUT_MAX}]"
            );
        }
        device.bind()?;
        let d_ids = device.alloc(k * 4)?;
        let d_probs = device.alloc(k * 4)?;
        Ok(Self {
            head_dev_idx: device.id() as usize,
            d_ids,
            d_probs,
            k,
            host_ids: vec![0u32; k],
            host_probs: vec![0.0f32; k],
            d_history_counts: None,
            host_history_sorted: Vec::new(),
            host_history_counts: Vec::new(),
            host_history_pairs: Vec::new(),
            disposed: false,
        })
    }

    /// HIP device id this scratch was allocated on.
    pub fn device_id(&self) -> i32 {
        self.head_dev_idx as i32
    }

    /// Allocate the penalty-path device buffer if not already allocated.
    fn ensure_history_buffer(&mut self, device: &HipDevice) -> Result<DevicePtr> {
        if let Some(p) = self.d_history_counts {
            return Ok(p);
        }
        device.bind()?;
        // 2 u32 (tok, count) per pair × SAMPLER_HISTORY_MAX × 4 bytes/u32.
        let bytes = SAMPLER_HISTORY_MAX * 2 * 4;
        let p = device.alloc(bytes)?;
        self.d_history_counts = Some(p);
        Ok(p)
    }

    /// Free the device buffers. Caller must pass the same device the
    /// scratch was allocated on (same HIP id).
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        device.bind()?;
        // SAFETY: d_ids / d_probs were allocated by `new` for `k * 4`
        // bytes each on `device`; d_history_counts (if Some) was
        // allocated by ensure_history_buffer for SAMPLER_HISTORY_MAX*8
        // bytes. We deallocate the same regions exactly once.
        unsafe {
            device.dealloc(self.d_ids, self.k * 4)?;
            device.dealloc(self.d_probs, self.k * 4)?;
            if let Some(p) = self.d_history_counts.take() {
                device.dealloc(p, SAMPLER_HISTORY_MAX * 2 * 4)?;
            }
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
/// Caller must invoke this immediately after `decode_logits` /
/// `prefill_logits` returns — the device pointer to logits is only
/// guaranteed-valid until the next forward call clobbers it.
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
    let (dev, ops, logits_f32, vocab) = resolve_head_logits(model, cluster, inflight, scratch)?;
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

/// Resolve `(device, ops, logits_f32_ptr, vocab_size)` for either TP or
/// Hybrid topologies. Centralises the head-rank lookup. The TP arm
/// pulls the device from the global cluster; the Hybrid arm pulls it
/// from the head stage's sub_cluster (the global cluster is unused).
/// All Option/Result branches use `ok_or_else` rather than `.unwrap()`
/// — a missing scratch field is a real configuration mismatch and
/// must error rather than panic.
fn resolve_head_logits<'a>(
    model: &'a LoadedModel,
    cluster: &'a HipCluster,
    inflight: &'a Inflight,
    scratch: &GpuSamplerScratch,
) -> Result<(
    &'a flambeau_backend_hip::HipDevice,
    &'a flambeau_ops::hip::OpsRegistry,
    DevicePtr,
    usize,
)> {
    if let (Some(pp_model), Some(pp_session)) = (model.as_pp(), inflight.as_pp()) {
        let m = &pp_model.model;
        let n_ranks = m.shards.len();
        if n_ranks == 0 {
            bail!("run_gpu_topk: PP model has zero ranks");
        }
        let head = n_ranks - 1;
        if head >= cluster.ranks() {
            bail!(
                "run_gpu_topk: PP head_rank={head} >= cluster ranks {}",
                cluster.ranks()
            );
        }
        let dev = cluster.device(head);
        if dev.id() != scratch.device_id() {
            bail!(
                "run_gpu_topk: scratch device_id={} != PP head device id={}",
                scratch.device_id(),
                dev.id()
            );
        }
        let ops = &m.shards[head].ops;
        let head_scratch = pp_session
            .decode
            .per_rank
            .get(head)
            .ok_or_else(|| anyhow!("PP scratch per_rank missing rank {head}"))?
            .output_head
            .as_ref()
            .ok_or_else(|| anyhow!("PP head rank missing OutputHeadScratch"))?;
        return Ok((dev, ops, head_scratch.logits_f32, m.config.vocab_size));
    }
    if let (Some(tp_model), Some(tp_session)) = (model.as_tp(), inflight.as_tp()) {
        let m = &tp_model.model;
        let decode = &tp_session.decode;
        let head = decode.head_rank.0 as usize;
        if head >= cluster.ranks() {
            bail!(
                "run_gpu_topk: TP head_rank={head} >= cluster ranks {}",
                cluster.ranks()
            );
        }
        let dev = cluster.device(head);
        if dev.id() != scratch.device_id() {
            bail!(
                "run_gpu_topk: scratch device_id={} != TP head device id={}",
                scratch.device_id(),
                dev.id()
            );
        }
        let ops = m
            .ops
            .get(head)
            .ok_or_else(|| anyhow!("TP ops registry missing rank {head}"))?;
        let head_scratch = decode
            .per_rank
            .get(head)
            .ok_or_else(|| anyhow!("TP scratch per_rank missing rank {head}"))?
            .output_head
            .as_ref()
            .ok_or_else(|| anyhow!("TP head rank missing OutputHeadScratch"))?;
        return Ok((dev, ops, head_scratch.logits_f32, m.config.vocab_size));
    }
    if let (Some(hybrid_model), Some(hybrid_session)) =
        (model.as_hybrid(), inflight.as_hybrid())
    {
        let hm = &hybrid_model.model;
        let decode = &hybrid_session.decode;
        let head_stage = decode.head_stage as usize;
        let stage_model = hm
            .stages
            .get(head_stage)
            .ok_or_else(|| anyhow!("Hybrid model missing head_stage {head_stage}"))?;
        let stage_scratch = decode
            .per_stage
            .get(head_stage)
            .ok_or_else(|| anyhow!("Hybrid decode missing head_stage {head_stage}"))?;
        let head_rank = stage_scratch.head_rank.0 as usize;
        if head_rank >= stage_model.sub_cluster.ranks() {
            bail!(
                "run_gpu_topk: hybrid head_rank={head_rank} >= stage sub-cluster ranks {}",
                stage_model.sub_cluster.ranks()
            );
        }
        let dev = stage_model.sub_cluster.device(head_rank);
        if dev.id() != scratch.device_id() {
            bail!(
                "run_gpu_topk: scratch device_id={} != hybrid head device id={}",
                scratch.device_id(),
                dev.id()
            );
        }
        let ops = stage_model
            .tp_model
            .ops
            .get(head_rank)
            .ok_or_else(|| anyhow!("Hybrid stage ops missing rank {head_rank}"))?;
        let head_scratch = stage_scratch
            .per_rank
            .get(head_rank)
            .ok_or_else(|| anyhow!("Hybrid scratch per_rank missing rank {head_rank}"))?
            .output_head
            .as_ref()
            .ok_or_else(|| anyhow!("Hybrid head rank missing OutputHeadScratch"))?;
        return Ok((dev, ops, head_scratch.logits_f32, hm.config.vocab_size));
    }
    bail!(
        "GPU sampler is wired for TP and Hybrid topologies (got {})",
        model.topology()
    )
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

/// **P0.1** — apply a JSON-grammar mask to `(host_ids, host_probs)`.
/// For each candidate, simulate `state.feed_slice(decoded_bytes(id))`
/// on a *clone*; zero out probs for candidates that would invalidate
/// the running JSON. The token finally sampled gets fed into the
/// caller's actual `state` after the multinomial draw.
/// Decoding 2048 candidates per token is non-trivial — only call this
/// when `params.json_mode == true`. At top_k=2048, vocab=151424,
/// the cost is dominated by the BPE decode loop on the JSON-relevant
/// subset (~1-2 ms/token typical).
pub fn apply_json_mask(
    state: &flambeau_runtime::json_grammar::JsonState,
    tokenizer: &flambeau_quant::GgufTokenizer,
    host_ids: &[u32],
    host_probs: &mut [f32],
) {
    for (i, &id) in host_ids.iter().enumerate() {
        if host_probs[i] <= 0.0 {
            continue;
        }
        // Decode just this single token. Stop tokens & specials get
        // empty bytes from the BPE decoder; treat empty bytes as
        // "always allowed" so the model can still pick `<|im_end|>`
        // when the running JSON is `is_complete()`.
        let bytes = match tokenizer.decode(&[id]) {
            Ok(s) => s,
            Err(_) => {
                host_probs[i] = 0.0;
                continue;
            }
        };
        if bytes.is_empty() {
            // Specials/EOS — only allow if the JSON is currently
            // structurally complete; otherwise the model would emit
            // EOS mid-value and the response would be invalid.
            if !state.is_complete() {
                host_probs[i] = 0.0;
            }
            continue;
        }
        let mut probe = state.clone();
        if !probe.feed_slice(bytes.as_bytes()) {
            host_probs[i] = 0.0;
        }
    }
}

/// **#236 P0.1b** — host-sampler-path JSON mask. Operates directly on
/// a full-vocab logits buffer, NOT on a top-K extract. Bounded by
/// `max_candidates` (default 2048): the function picks the top-N
/// candidates by logit, decodes each, probes the JSON state, and
/// sets the logit of any candidate that would invalidate the running
/// JSON to `f32::NEG_INFINITY`. Tokens outside the top-N stay
/// untouched — they're below the threshold any sane sampler will
/// pick from anyway.
/// Differs from [`apply_json_mask`] (which masks an *already extracted*
/// `(host_ids, host_probs)` top-K from the GPU-sampler path) in that
/// it does the candidate selection itself. Cost is dominated by the
/// per-candidate `tokenizer.decode` (BPE byte reconstruction); ~1–2 ms
/// at `max_candidates=2048` on a 151 k-vocab Qwen tokenizer.
/// Caller must ensure `state` reflects the bytes already emitted /
/// primed; this function does not mutate `state`.
pub fn apply_json_mask_to_logits(
    state: &flambeau_runtime::json_grammar::JsonState,
    tokenizer: &flambeau_quant::GgufTokenizer,
    logits: &mut [f32],
    max_candidates: usize,
) {
    if logits.is_empty() {
        return;
    }
    let k = max_candidates.min(logits.len());
    if k == 0 {
        return;
    }

    // Pick top-K candidate ids by logit value via partial sort
    // (`select_nth_unstable_by`). O(V) per call — same trick as
    // `runtime::sampling::build_distribution`.
    let mut pairs: Vec<(u32, f32)> = (0..logits.len() as u32)
        .map(|i| (i, logits[i as usize]))
        .collect();
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.truncate(k);
    }

    let complete = state.is_complete();
    let already_started = state.has_started();
    for &(id, _) in pairs.iter() {
        let bytes = match tokenizer.decode(&[id]) {
            Ok(s) => s,
            Err(_) => {
                logits[id as usize] = f32::NEG_INFINITY;
                continue;
            }
        };
        if bytes.is_empty() {
            // Specials / EOS — only allowed once the JSON is
            // structurally closed; otherwise the model would emit EOS
            // mid-value and the response would be invalid.
            if !complete {
                logits[id as usize] = f32::NEG_INFINITY;
            }
            continue;
        }
        let mut probe = state.clone();
        if !probe.feed_slice(bytes.as_bytes()) {
            logits[id as usize] = f32::NEG_INFINITY;
            continue;
        }
        // **#236 P0.1b** — when the response hasn't started yet, the
        // OpenAI `response_format: json_object` contract requires the
        // top-level value to be an object. Reject any candidate that
        // doesn't drive the state into an object frame: this kills
        // both the infinite-leading-whitespace stall (model picks
        // `\n` forever) AND the dead-end-number stall (model picks
        // `1`, then nothing in the top-K can extend the number
        // legally and the rest of decode burns tokens on masked
        // junk).
        if !already_started && !probe.has_started() {
            logits[id as usize] = f32::NEG_INFINITY;
            continue;
        }
        if !already_started
            && probe.has_started()
            && bytes.as_bytes().iter().all(|&b| b != b'{')
        {
            // First token started a non-object top-level (a number,
            // string, array, or `true`/`false`/`null`). That's valid
            // JSON but invalid `json_object`.
            logits[id as usize] = f32::NEG_INFINITY;
        }
    }
}

/// **Sampler-D4 (#212)** — penalty-aware variant of [`run_gpu_topk`].
/// Builds `(tok, count)` pairs from `history` via Sampler-F's
/// sort+dedup, uploads to device, runs the GPU penalty kernel
/// in-place on the head-rank's `logits_f32`, then runs topk + DtoH
/// the K-tuple as in [`run_gpu_topk`].
/// `mode.has_penalties()` MUST be true — caller is responsible for
/// short-circuiting to [`run_gpu_topk`] otherwise (the penalty kernel
/// errors on n_pairs=0 and the upload+launch overhead is unwanted).
/// # Errors
/// - Topology unsupported (Phase A wires TP only).
/// - History exceeds [`SAMPLER_HISTORY_MAX`] unique tokens.
/// - Kernel-launch / DtoH failure.
pub fn run_gpu_topk_with_penalties(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &Inflight,
    scratch: &mut GpuSamplerScratch,
    history: &[u32],
    mode: &Sampling,
    inv_temp: f32,
) -> Result<()> {
    debug_assert!(
        mode.has_penalties(),
        "run_gpu_topk_with_penalties called with no penalties active"
    );
    // Build sort+dedup pairs on host.
    flambeau_runtime::sampling::build_history_counts(
        history,
        &mut scratch.host_history_sorted,
        &mut scratch.host_history_counts,
    );
    let n_pairs = scratch.host_history_counts.len();
    if n_pairs == 0 {
        // No active penalty entries (history empty after dedup is
        // impossible — only happens when history is empty, in which
        // case the caller should route to run_gpu_topk).
        return run_gpu_topk(model, cluster, inflight, scratch, inv_temp);
    }
    if n_pairs > SAMPLER_HISTORY_MAX {
        bail!(
            "run_gpu_topk_with_penalties: n_pairs {n_pairs} > SAMPLER_HISTORY_MAX \
             {SAMPLER_HISTORY_MAX} — bump the constant in gpu_sampler.rs if a longer \
             chat turn is genuinely expected"
        );
    }

    // Flatten (tok, count) into u32 pairs.
    scratch.host_history_pairs.clear();
    scratch.host_history_pairs.reserve(n_pairs * 2);
    for &(t, c) in &scratch.host_history_counts {
        scratch.host_history_pairs.push(t);
        scratch.host_history_pairs.push(c);
    }

    let (dev, ops, logits_f32, vocab) = resolve_head_logits(model, cluster, inflight, scratch)?;
    let d_history_counts = scratch.ensure_history_buffer(dev)?;
    dev.bind()?;
    let stream = dev.default_stream();

    // 1. Upload (tok, count) pairs HtoD. SAFETY: d_history_counts was
    // allocated for SAMPLER_HISTORY_MAX*2*4 bytes; we copy n_pairs*2*4
    // bytes which is bounded above.
    unsafe {
        <flambeau_backend_hip::HipDevice as Device>::memcpy_async(
            dev,
            stream,
            CopyDirection::HostToDevice,
            d_history_counts,
            DevicePtr(scratch.host_history_pairs.as_ptr() as usize),
            n_pairs * 2 * 4,
        )?;
    }
    // 2. Apply penalties in place on logits_f32.
    apply_penalties_f32(
        ops,
        stream,
        logits_f32,
        d_history_counts,
        n_pairs,
        vocab,
        mode.repetition_penalty,
        mode.presence_penalty,
        mode.frequency_penalty,
    )
    .context("GPU apply_penalties_f32")?;
    // 3. topk + DtoH small tuple.
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
    .context("GPU topk_softmax_f32 (post-penalty)")?;
    // SAFETY: d_ids/d_probs sized for k*4 each on `dev`; host vecs match.
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
