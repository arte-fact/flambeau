//! V2.26.a-i5b2 — graph-captureable KvCache::append for HIP.
//!
//! The stock `KvCache::append` (in `flambeau-runtime`) drives two
//! `Device::memcpy_async` calls through the `Device` trait surface,
//! which is agnostic to graph-capture slot tagging. This module owns
//! the HIP-specific variant that issues the K + V memcpys tagged with
//! [`MemcpySlot`]s so their dst pointers can be retargeted per ubatch
//! at replay via `HipGraphExec::set_memcpy_slot`.
//!
//! Split from the runtime crate to keep `flambeau-runtime` backend-
//! agnostic — HipDevice / HipStream / MemcpySlot all live in
//! `flambeau-backend-hip` and would introduce a circular dep.

use flambeau_core::{CopyDirection, DevicePtr};
use flambeau_runtime::{F16Contig, KvCache};

use crate::graph_capture::MemcpySlot;
use crate::{HipDevice, HipStream};

/// Captureable K/V append to an `F16Contig` KV cache.
///
/// Computes the dst offsets for the current tail (via
/// `KvCache::compute_append_dsts`), issues the two memcpys tagged with
/// `k_slot` / `v_slot`, and bumps the cache's logical tail. Under a
/// [`crate::HipGraphExec::capture`] scope the memcpys are recorded;
/// post-capture the bound slots can be updated per ubatch via
/// `HipGraphExec::set_memcpy_slot`, retargeting the dsts to the new
/// pos offset before each replay.
///
/// Callers that aren't capturing should still prefer the stock
/// `KvCache::append` — this variant is slightly slower because it
/// doesn't fold the capacity check with the memcpy and requires slots
/// be pre-allocated.
///
/// # Safety
/// Same contract as `KvCache::append`: `k_new` and `v_new` must point
/// to at least `n_new * n_heads * head_dim * 2` valid device bytes on
/// the same device as the cache.
pub unsafe fn kv_cache_append_hip_slot(
    cache: &mut KvCache<F16Contig, HipDevice>,
    device: &HipDevice,
    stream: &HipStream,
    k_new: DevicePtr,
    v_new: DevicePtr,
    n_new: usize,
    k_slot: MemcpySlot,
    v_slot: MemcpySlot,
) -> anyhow::Result<()> {
    let (k_dst, v_dst, total_bytes) = cache
        .compute_append_dsts(n_new)
        .map_err(|e| anyhow::anyhow!("kv_cache_append_hip_slot: {e}"))?;
    // SAFETY: caller's outer contract covers source validity; k_dst /
    // v_dst are valid per the capacity check in compute_append_dsts.
    unsafe {
        device.memcpy_async_slot(
            stream,
            CopyDirection::DeviceToDevice,
            k_dst,
            k_new,
            total_bytes,
            k_slot,
        )?;
        device.memcpy_async_slot(
            stream,
            CopyDirection::DeviceToDevice,
            v_dst,
            v_new,
            total_bytes,
            v_slot,
        )?;
    }
    cache
        .bump_tail(n_new)
        .map_err(|e| anyhow::anyhow!("kv_cache_append_hip_slot bump_tail: {e}"))?;
    Ok(())
}
