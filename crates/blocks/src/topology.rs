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
            // SAFETY: hidden_a buffers on each rank are sized
            // `hidden_bytes` and not concurrently touched by other
            // streams during this peer-copy.
            unsafe {
                driver.cluster().peer_copy_via_host(
                    driver.hidden_a(rank_idx),
                    rank_idx,
                    driver.hidden_a(rank_idx - 1),
                    rank_idx - 1,
                    hidden_bytes,
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
        bail!(
            "forward_prefill_pp_chunk: tokens.len()={l} > driver.max_tokens()={max_tokens}"
        );
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
            driver.forward_layer_prefill(
                rank_idx,
                local_idx,
                x_in,
                x_out,
                l,
                start_position,
            )?;
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
