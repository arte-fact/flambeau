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
use flambeau_core::{CopyDirection, Device, DevicePtr};

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
