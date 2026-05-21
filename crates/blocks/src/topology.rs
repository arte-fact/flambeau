//! Pipeline-parallel topology driver.
//!
//! `PpDecodeDriver` is the portable surface a model crate implements
//! to drive a one-token decode through a `HipCluster`. The model
//! crate owns its weights, sessions, and scratches; this module just
//! orchestrates the rank loop (embed, peer-copy, per-layer iteration,
//! output head, argmax).
//!
//! `forward_one_token_pp` is the entry point. It calls back into the
//! driver for every model-specific step.

use anyhow::{bail, Result};
use flambeau_backend_hip::HipCluster;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

/// Per-rank handles and per-call hooks the topology orchestrator
/// needs from the model crate.
///
/// Lifetime: a driver instance is built once per forward call, holds
/// mutable references to the model + session + scratch, and is
/// consumed by `forward_one_token_pp`.
pub trait PpDecodeDriver {
    /// Number of ranks in the pipeline.
    fn n_ranks(&self) -> usize;

    /// Number of transformer layers owned by `rank`.
    fn layers_per_rank(&self, rank: usize) -> usize;

    /// HIP cluster handle. Used for `peer_copy_via_host` between
    /// stages and for per-rank `bind`.
    fn cluster(&self) -> &HipCluster;

    /// Per-rank "current hidden state" buffer in F16 [hidden]. The
    /// orchestrator reads + writes through this between stages.
    fn hidden_a(&self, rank: usize) -> DevicePtr;

    /// Per-rank "scratch" hidden buffer for layer ping-pong.
    fn hidden_b(&self, rank: usize) -> DevicePtr;

    /// Hidden width in bytes (= hidden_size * 2 for F16).
    fn hidden_bytes(&self) -> usize;

    /// Embed `token_id` and write the F16 hidden vector to
    /// `hidden_a(0)`. Called on rank 0 after `bind`.
    fn embed_token(&mut self, token_id: u32) -> Result<()>;

    /// Run one layer's decode on `rank`. Reads from `x_in`, writes
    /// the post-residual output to `x_out`. Both pointers are on the
    /// rank's device. `local_idx` is `0..layers_per_rank(rank)`.
    fn forward_layer_decode(
        &mut self,
        rank: usize,
        local_idx: usize,
        x_in: DevicePtr,
        x_out: DevicePtr,
        position: usize,
    ) -> Result<()>;

    /// Run the output norm + LM-head matmul on the last rank. Reads
    /// from `hidden_a(last)`, writes F32 logits to a driver-owned
    /// buffer the caller can then sample.
    fn output_head(&mut self) -> Result<()>;

    /// Read out the argmax of the F32 logits computed by
    /// `output_head`. Last rank only.
    fn argmax(&self) -> Result<u32>;
}

/// Single-token decode through a pipeline-parallel mesh.
///
/// Sequence:
/// 1. Bind rank 0; `embed_token` writes the F16 hidden vector to
///    `hidden_a(0)`.
/// 2. For each rank in order:
///    a. If `rank > 0`, `peer_copy_via_host` from `hidden_a(rank-1)`
///       on the previous device into `hidden_a(rank)` on this one.
///    b. Bind this rank's device.
///    c. Iterate `layers_per_rank(rank)` calling
///       `forward_layer_decode` with ping-pong between `hidden_a` /
///       `hidden_b`.
///    d. Copy the final per-layer output back into `hidden_a(rank)`
///       so the next stage's `peer_copy_via_host` reads from a
///       stable slot.
/// 3. Bind the last rank; `output_head` runs the LM head.
/// 4. `argmax` returns the chosen token id.
pub fn forward_one_token_pp<D: PpDecodeDriver>(
    driver: &mut D,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    let n_ranks = driver.n_ranks();
    if n_ranks == 0 {
        bail!("forward_one_token_pp: zero-rank driver");
    }

    {
        let rank0 = driver.cluster().device(0);
        rank0.bind()?;
    }
    driver.embed_token(token_id)?;

    let hidden_bytes = driver.hidden_bytes();
    for rank_idx in 0..n_ranks {
        if rank_idx > 0 {
            // Async event-bridged peer copy: src stream issues DtoH +
            // records bridge event; dst stream stream-waits on bridge
            // event + issues HtoD. No host sync, no
            // `hipStreamSynchronize` round-trip in the per-token
            // critical path. The helper also records an `htod_done`
            // event and threads it through `peer_copy_htod_event[src]`
            // so the next call's DtoH on src waits for the prior HtoD
            // to drain the pinned bounce buffer — prevents bounce
            // aliasing without a host barrier.
            //
            // `consumer_stream = None`: PP keeps all subsequent
            // compute on the dst device's default stream — same
            // stream as the HtoD — so stream ordering is implicit.
            //
            // SAFETY: hidden_a buffers on each rank are sized
            // `hidden_bytes`; the helper's internal interlock prevents
            // concurrent peer-copy aliasing of the pinned bounce.
            unsafe {
                driver.cluster().peer_copy_via_host_event(
                    driver.hidden_a(rank_idx),
                    rank_idx,
                    driver.hidden_a(rank_idx - 1),
                    rank_idx - 1,
                    hidden_bytes,
                    None,
                )?;
            }
        }

        driver.cluster().device(rank_idx).bind()?;

        let n_layers = driver.layers_per_rank(rank_idx);
        let (mut x_in, mut x_out) = (driver.hidden_a(rank_idx), driver.hidden_b(rank_idx));
        for local_idx in 0..n_layers {
            driver.forward_layer_decode(rank_idx, local_idx, x_in, x_out, position)?;
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Land the final hidden in `hidden_a(rank)` so the next
        // stage's peer-copy reads from a stable slot. No sync — the
        // peer-copy + downstream kernels all run on the same default
        // stream and stream ordering preserves correctness.
        let dst = driver.hidden_a(rank_idx);
        if x_in != dst {
            let device = driver.cluster().device(rank_idx);
            // SAFETY: `dst` and `x_in` are both `hidden_bytes` long
            // on `device` (driver invariant); the device-to-device
            // memcpy is async on `device.default_stream()`.
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    dst,
                    x_in,
                    hidden_bytes,
                )?;
            }
        }
    }

    let last = n_ranks - 1;
    driver.cluster().device(last).bind()?;
    driver.output_head()?;
    driver.argmax()
}

/// Per-rank handles and per-call hooks for a multi-token prefill
/// chunk. Symmetric to `PpDecodeDriver` but every kernel step
/// processes `n_tokens` rows.
///
/// Finalization (argmax of the last row, download to host, spec-decode
/// paired-L2) stays caller-side — `forward_prefill_pp` runs through
/// `output_head_last_token` and returns; the caller reads logits or
/// argmax through its own driver methods.
pub trait PpPrefillDriver {
    fn n_ranks(&self) -> usize;

    fn layers_per_rank(&self, rank: usize) -> usize;

    fn cluster(&self) -> &HipCluster;

    fn hidden_a(&self, rank: usize) -> DevicePtr;

    fn hidden_b(&self, rank: usize) -> DevicePtr;

    /// Bytes per F16 hidden row (`hidden_size * 2`).
    fn hidden_row_bytes(&self) -> usize;

    /// Maximum chunk size the driver's scratches can hold. Callers
    /// pass `tokens.len() <= max_tokens()`; the orchestrator chunks
    /// longer prompts.
    fn max_tokens(&self) -> usize;

    /// Embed `tokens.len()` rows starting at offset 0 of `hidden_a(0)`.
    fn embed_tokens(&mut self, tokens: &[u32]) -> Result<()>;

    /// Run one layer prefill on `rank`. Reads `n_tokens` rows from
    /// `x_in`; writes the post-residual output to `x_out`.
    fn forward_layer_prefill(
        &mut self,
        rank: usize,
        local_idx: usize,
        x_in: DevicePtr,
        x_out: DevicePtr,
        n_tokens: usize,
        start_position: usize,
    ) -> Result<()>;

    /// Run the output norm + LM-head matmul on the last rank,
    /// reading the final-token row at offset `(l - 1) * row_bytes` of
    /// `hidden_a(last_rank)`. Caller reads the resulting F32 logits
    /// via its own driver methods after the orchestrator returns.
    fn output_head_last_token(&mut self, l: usize) -> Result<()>;
}

/// Multi-token prefill across a pipeline-parallel mesh.
///
/// Sequence (per chunk of up to `max_tokens()`):
/// 1. Bind rank 0; `embed_tokens` writes L F16 rows into `hidden_a(0)`.
/// 2. For each rank in order:
///    a. If `rank > 0`, `peer_copy_via_host` carries `L * row_bytes`
///       bytes of F16 hidden state from `hidden_a(rank-1)` on the
///       previous device into `hidden_a(rank)` on this one.
///    b. Bind this rank's device.
///    c. Iterate `layers_per_rank(rank)` calling
///       `forward_layer_prefill` with ping-pong between `hidden_a` /
///       `hidden_b`.
///    d. Copy the final per-layer output back into `hidden_a(rank)`
///       so the next stage's `peer_copy_via_host` reads from a
///       stable slot. Synchronises before the copy returns.
/// 3. Bind the last rank; `output_head_last_token` runs the LM head
///    on the final token's hidden row. Caller finalises (argmax,
///    logits download, etc.) after the orchestrator returns.
///
/// Inputs longer than `max_tokens()` are split into sequential chunks
/// of that size; KV-cache + GDN state thread per-chunk via
/// `start_position`.
pub fn forward_prefill_pp<D: PpPrefillDriver>(
    driver: &mut D,
    tokens: &[u32],
    start_position: usize,
) -> Result<()> {
    if driver.n_ranks() == 0 {
        bail!("forward_prefill_pp: zero-rank driver");
    }
    if tokens.is_empty() {
        bail!("forward_prefill_pp: empty tokens");
    }
    let max_tokens = driver.max_tokens();
    if max_tokens == 0 {
        bail!("forward_prefill_pp: driver max_tokens=0");
    }

    let mut chunk_start = 0usize;
    while chunk_start < tokens.len() {
        let chunk_end = (chunk_start + max_tokens).min(tokens.len());
        forward_prefill_pp_chunk(
            driver,
            &tokens[chunk_start..chunk_end],
            start_position + chunk_start,
        )?;
        chunk_start = chunk_end;
    }
    Ok(())
}

/// Single-chunk variant for callers that already chunk (e.g. the
/// spec-decode paired-L=2 path) or that download logits per chunk.
/// Bails when `tokens.len() > driver.max_tokens()`.
pub fn forward_prefill_pp_chunk<D: PpPrefillDriver>(
    driver: &mut D,
    tokens: &[u32],
    start_position: usize,
) -> Result<()> {
    let n_ranks = driver.n_ranks();
    if n_ranks == 0 {
        bail!("forward_prefill_pp_chunk: zero-rank driver");
    }
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill_pp_chunk: empty tokens");
    }
    let max_tokens = driver.max_tokens();
    if l > max_tokens {
        bail!("forward_prefill_pp_chunk: tokens.len()={l} > driver.max_tokens()={max_tokens}");
    }
    let row_bytes = driver.hidden_row_bytes();
    let chunk_bytes = l * row_bytes;

    driver.cluster().device(0).bind()?;
    driver.embed_tokens(tokens)?;

    for rank_idx in 0..n_ranks {
        if rank_idx > 0 {
            // SAFETY: hidden_a buffers on each rank are sized to at
            // least `l * row_bytes` (driver invariant via
            // `max_tokens()`); not concurrently touched.
            unsafe {
                driver.cluster().peer_copy_via_host(
                    driver.hidden_a(rank_idx),
                    rank_idx,
                    driver.hidden_a(rank_idx - 1),
                    rank_idx - 1,
                    chunk_bytes,
                )?;
            }
        }

        driver.cluster().device(rank_idx).bind()?;

        let n_layers = driver.layers_per_rank(rank_idx);
        let (mut x_in, mut x_out) = (driver.hidden_a(rank_idx), driver.hidden_b(rank_idx));
        for local_idx in 0..n_layers {
            driver.forward_layer_prefill(rank_idx, local_idx, x_in, x_out, l, start_position)?;
            std::mem::swap(&mut x_in, &mut x_out);
        }

        let dst = driver.hidden_a(rank_idx);
        if x_in != dst {
            let device = driver.cluster().device(rank_idx);
            // SAFETY: dst and x_in are both `chunk_bytes` long on
            // `device`; the dtod copy runs on the default stream.
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    dst,
                    x_in,
                    chunk_bytes,
                )?;
            }
            device.default_stream().synchronize()?;
        }
    }

    let last = n_ranks - 1;
    driver.cluster().device(last).bind()?;
    driver.output_head_last_token(l)
}

/// Per-rank handles + per-call hooks for tensor-parallel single-token
/// decode. TP holds every layer on every rank (sliced via Megatron
/// splits); the topology orchestrator just iterates layers, the
/// driver handles the per-layer intra-rank slicing + BAR1 P2P
/// AllReduces.
///
/// Finalisation (argmax / logits download / keep-on-device) stays
/// caller-side.
pub trait TpDecodeDriver {
    fn cluster(&self) -> &HipCluster;

    fn n_layers(&self) -> usize;

    /// Rank that holds the LM head + final norm. The orchestrator
    /// binds this device before `output_head`.
    fn head_rank(&self) -> usize;

    /// Embed `token_id` on `rank`. token_embd is replicated across
    /// ranks in the V1 TP layout, so each rank dequantises into its
    /// own hidden buffer.
    fn embed_token(&mut self, rank: usize, token_id: u32) -> Result<()>;

    /// Run layer `il`'s decode across all ranks. The driver dispatches
    /// to its own full-attn / GDN / dense / MoE TP kernels per
    /// `cfg.is_recurrent(il)` (or whichever predicate the model uses).
    fn forward_layer_decode(&mut self, il: usize, position: usize) -> Result<()>;

    /// Run final norm + LM head on the head rank, leaving F32 logits
    /// in the driver-owned head scratch. Caller finalises through
    /// driver-specific methods after the orchestrator returns.
    fn output_head(&mut self) -> Result<()>;
}

/// Single-token decode through a tensor-parallel mesh.
///
/// Sequence:
/// 1. Bind each rank in turn; `embed_token` writes the F16 hidden
///    vector to the rank's hidden buffer.
/// 2. Iterate `0..n_layers()` calling `forward_layer_decode` — the
///    driver runs whatever TP-specific intra-layer composition the
///    model needs (BAR1 P2P AllReduce after attn / FFN, etc.).
/// 3. Bind `head_rank`; `output_head` runs the LM head on that
///    rank's hidden vector.
pub fn forward_one_token_tp<D: TpDecodeDriver>(
    driver: &mut D,
    token_id: u32,
    position: usize,
) -> Result<()> {
    let n_ranks = driver.cluster().ranks();
    if n_ranks == 0 {
        bail!("forward_one_token_tp: zero-rank driver");
    }

    for r in 0..n_ranks {
        driver.cluster().device(r).bind()?;
        driver.embed_token(r, token_id)?;
    }

    let n_layers = driver.n_layers();
    for il in 0..n_layers {
        driver.forward_layer_decode(il, position)?;
    }

    let head = driver.head_rank();
    if head >= n_ranks {
        bail!("forward_one_token_tp: head_rank={head} out of range (n_ranks={n_ranks})");
    }
    driver.cluster().device(head).bind()?;
    driver.output_head()
}

/// Per-rank handles + per-call hooks for tensor-parallel multi-token
/// prefill. Same all-ranks-run-all-layers shape as `TpDecodeDriver`,
/// just with multi-token args.
///
/// The driver's `forward_layers_prefill` runs the whole layer chain
/// in one call — TP's L-batched prefill kernels (e.g.
/// `attn_tp::forward_full_attn_prefill_tp`) operate on the full
/// `[L, hidden]` strip per layer. Per-token-loop fallbacks (e.g. on
/// Q8 KV that has no batched-prefill kernel) live at the model
/// wrapper level: the wrapper picks between this orchestrator and a
/// per-token loop through `forward_one_token_tp`.
pub trait TpPrefillDriver {
    fn cluster(&self) -> &HipCluster;

    /// Rank that holds the LM head + final norm.
    fn head_rank(&self) -> usize;

    /// Embed `tokens` on `rank`. token_embd is replicated; each rank
    /// writes the same `[L, hidden]` F16 strip into its hidden buffer.
    fn embed_prompt_on_rank(&mut self, rank: usize, tokens: &[u32]) -> Result<()>;

    /// Run the entire layer chain for the prompt at L tokens. Driver
    /// handles per-rank weight slicing + intra-layer AllReduces.
    fn forward_layers_prefill(&mut self, prompt_len: usize, start_position: usize) -> Result<()>;

    /// Run final norm + LM head on `head_rank`'s last-token row,
    /// leaving F32 logits in the driver-owned head scratch.
    fn output_head_last_token(&mut self, l: usize) -> Result<()>;
}

/// Multi-token prefill across a tensor-parallel mesh.
///
/// Sequence:
/// 1. For each rank, bind + `embed_prompt_on_rank` writes all L F16
///    rows into the rank's hidden buffer (token_embd is replicated).
/// 2. `forward_layers_prefill` runs the whole layer chain at L
///    tokens. Driver handles per-rank slicing + AllReduces.
/// 3. Bind `head_rank`; `output_head_last_token` runs the LM head on
///    the last-token row.
pub fn forward_prefill_tp<D: TpPrefillDriver>(
    driver: &mut D,
    prompt_ids: &[u32],
    start_position: usize,
) -> Result<()> {
    if prompt_ids.is_empty() {
        bail!("forward_prefill_tp: empty prompt");
    }
    let n_ranks = driver.cluster().ranks();
    if n_ranks == 0 {
        bail!("forward_prefill_tp: zero-rank driver");
    }

    for r in 0..n_ranks {
        driver.cluster().device(r).bind()?;
        driver.embed_prompt_on_rank(r, prompt_ids)?;
    }

    driver.forward_layers_prefill(prompt_ids.len(), start_position)?;

    let head = driver.head_rank();
    if head >= n_ranks {
        bail!("forward_prefill_tp: head_rank={head} out of range (n_ranks={n_ranks})");
    }
    driver.cluster().device(head).bind()?;
    driver.output_head_last_token(prompt_ids.len())
}

/// Per-stage handles + per-call hooks for hybrid (PP-of-TP) single-
/// token decode. The mesh is `n_stages` contiguous layer stages, each
/// owning a TP subgroup. Stage 0 holds `token_embd`; the head stage
/// (last in V1) holds the LM head.
///
/// The driver knows about per-stage rank counts + layer ranges; the
/// orchestrator just iterates stages and layers, calling back into
/// the driver for embed, per-layer dispatch, inter-stage handoff,
/// and output head.
pub trait HybridDecodeDriver {
    fn n_stages(&self) -> usize;

    fn ranks_per_stage(&self, stage: usize) -> usize;

    fn n_layers_in_stage(&self, stage: usize) -> usize;

    /// Stage that holds the LM head (last stage in V1).
    fn head_stage(&self) -> usize;

    /// Rank within `head_stage` that holds the head weights / scratch.
    fn head_rank_in_head_stage(&self) -> usize;

    /// Bind `(stage, rank)`'s device on the calling thread. Takes
    /// `&self` so callers can interleave bind with `&mut self`
    /// dispatch calls without borrow conflicts.
    fn bind(&self, stage: usize, rank: usize) -> Result<()>;

    /// Embed `token_id` on `(stage, rank)`. Stage 0 is where this is
    /// called; later stages receive hidden state via
    /// `handoff_stage_to_next`.
    fn embed_token(&mut self, stage: usize, rank: usize, token_id: u32) -> Result<()>;

    /// Run one layer's decode within `stage`. `il_in_stage` is
    /// `0..n_layers_in_stage(stage)`. Driver handles intra-stage TP
    /// per-rank dispatch + BAR1 P2P AllReduces.
    fn forward_layer_decode(
        &mut self,
        stage: usize,
        il_in_stage: usize,
        position: usize,
    ) -> Result<()>;

    /// Carry the residual stream from `stage`'s rank 0 to every rank
    /// of `stage + 1`. Implementor uses the global cluster's
    /// `peer_copy_via_host`.
    fn handoff_stage_to_next(&mut self, stage: usize) -> Result<()>;

    /// Run final norm + LM head on the head rank, leaving F32 logits
    /// in the driver-owned head scratch.
    fn output_head(&mut self) -> Result<()>;
}

/// Single-token decode through a hybrid PP-of-TP mesh.
///
/// Sequence:
/// 1. Bind every rank of stage 0; `embed_token` writes the F16 hidden
///    vector on each.
/// 2. For each stage in order:
///    a. Iterate `n_layers_in_stage(stage)` calling
///       `forward_layer_decode`.
///    b. If `stage + 1 < n_stages`, call `handoff_stage_to_next`.
/// 3. Bind the head rank; `output_head` runs the LM head.
pub fn forward_one_token_hybrid<D: HybridDecodeDriver>(
    driver: &mut D,
    token_id: u32,
    position: usize,
) -> Result<()> {
    let n_stages = driver.n_stages();
    if n_stages == 0 {
        bail!("forward_one_token_hybrid: zero-stage driver");
    }

    for r in 0..driver.ranks_per_stage(0) {
        driver.bind(0, r)?;
        driver.embed_token(0, r, token_id)?;
    }

    for s in 0..n_stages {
        let n_layers = driver.n_layers_in_stage(s);
        for il in 0..n_layers {
            driver.forward_layer_decode(s, il, position)?;
        }
        if s + 1 < n_stages {
            driver.handoff_stage_to_next(s)?;
        }
    }

    let head_s = driver.head_stage();
    let head_r = driver.head_rank_in_head_stage();
    driver.bind(head_s, head_r)?;
    driver.output_head()
}

/// Per-stage handles + per-call hooks for hybrid (PP-of-TP) multi-
/// token prefill. Same stage-aware shape as `HybridDecodeDriver` but
/// every step processes `prompt_len` rows and the inter-stage handoff
/// carries `prompt_len * hidden * 2` bytes.
pub trait HybridPrefillDriver {
    fn n_stages(&self) -> usize;

    fn ranks_per_stage(&self, stage: usize) -> usize;

    fn head_stage(&self) -> usize;

    fn head_rank_in_head_stage(&self) -> usize;

    fn bind(&self, stage: usize, rank: usize) -> Result<()>;

    /// Embed the whole prompt on `(stage, rank)` — token_embd is
    /// replicated within a stage; each rank writes `[L, hidden]` F16
    /// rows into its hidden buffer.
    fn embed_prompt_on_rank(&mut self, stage: usize, rank: usize, tokens: &[u32]) -> Result<()>;

    /// Run `stage`'s entire layer slice at `prompt_len` tokens with
    /// `start_position` as the position the first prompt token lands
    /// at. Driver handles intra-stage TP slicing + AllReduces.
    fn forward_layers_prefill_in_stage(
        &mut self,
        stage: usize,
        prompt_len: usize,
        start_position: usize,
    ) -> Result<()>;

    /// Carry the L-row residual stream from `stage`'s rank 0 to every
    /// rank of `stage + 1`. `prompt_len * hidden * 2` bytes per dst.
    fn handoff_stage_to_next(&mut self, stage: usize, prompt_len: usize) -> Result<()>;

    /// Run final norm + LM head on `head_rank`'s last-token row of
    /// the L-row hidden buffer.
    fn output_head_last_token(&mut self, prompt_len: usize) -> Result<()>;
}

/// Multi-token prefill across a hybrid PP-of-TP mesh.
///
/// Sequence:
/// 1. Bind every rank of stage 0; `embed_prompt_on_rank` writes all L
///    F16 rows into each rank's hidden buffer.
/// 2. For each stage in order:
///    a. `forward_layers_prefill_in_stage` runs the stage's layer
///       chain at L tokens.
///    b. If `stage + 1 < n_stages`, call `handoff_stage_to_next`.
/// 3. Bind the head rank; `output_head_last_token` runs the LM head
///    on the last-token row.
pub fn forward_prefill_hybrid<D: HybridPrefillDriver>(
    driver: &mut D,
    prompt_ids: &[u32],
    start_position: usize,
) -> Result<()> {
    if prompt_ids.is_empty() {
        bail!("forward_prefill_hybrid: empty prompt");
    }
    let n_stages = driver.n_stages();
    if n_stages == 0 {
        bail!("forward_prefill_hybrid: zero-stage driver");
    }
    let l = prompt_ids.len();

    for r in 0..driver.ranks_per_stage(0) {
        driver.bind(0, r)?;
        driver.embed_prompt_on_rank(0, r, prompt_ids)?;
    }

    for s in 0..n_stages {
        driver.forward_layers_prefill_in_stage(s, l, start_position)?;
        if s + 1 < n_stages {
            driver.handoff_stage_to_next(s, l)?;
        }
    }

    let head_s = driver.head_stage();
    let head_r = driver.head_rank_in_head_stage();
    driver.bind(head_s, head_r)?;
    driver.output_head_last_token(l)
}
