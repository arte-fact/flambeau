//! Graph-captureable KvCache::append for HIP.
//! The stock `KvCache::append` (in `flambeau-runtime`) drives two
//! `Device::memcpy_async` calls through the `Device` trait surface,
//! which is agnostic to graph-capture slot tagging. This module owns
//! the HIP-specific variant that issues the K + V memcpys tagged with
//! [`MemcpySlot`]s so their dst pointers can be retargeted per ubatch
//! at replay via `HipGraphExec::set_memcpy_slot`.
//! Split from the runtime crate to keep `flambeau-runtime` backend-
//! agnostic — HipDevice / HipStream / MemcpySlot all live in
//! `flambeau-backend-hip` and would introduce a circular dep.

use flambeau_core::{CopyDirection, DevicePtr};
use flambeau_runtime::{CacheLayout, KvCache};

use crate::graph_capture::MemcpySlot;
use crate::{HipDevice, HipStream};

/// Captureable K/V append to an `F16Contig` KV cache.
/// Computes the dst offsets for the current tail (via
/// `KvCache::compute_append_dsts`), issues the two memcpys tagged with
/// `k_slot` / `v_slot`, and bumps the cache's logical tail. Under a
/// [`crate::HipGraphExec::capture`] scope the memcpys are recorded;
/// post-capture the bound slots can be updated per ubatch via
/// `HipGraphExec::set_memcpy_slot`, retargeting the dsts to the new
/// pos offset before each replay.
/// Callers that aren't capturing should still prefer the stock
/// `KvCache::append` — this variant is slightly slower because it
/// doesn't fold the capacity check with the memcpy and requires slots
/// be pre-allocated.
/// # Safety
/// Same contract as `KvCache::append`: `k_new` and `v_new` must point
/// to at least `n_new * n_heads * head_dim * 2` valid device bytes on
/// the same device as the cache.
/// Source pointers + token count for a captureable KV append.
#[derive(Copy, Clone, Debug)]
pub struct KvAppendSrc {
    pub k_new: DevicePtr,
    pub v_new: DevicePtr,
    pub n_new: usize,
}

/// Captured-memcpy slot pair, one per K/V leg.
#[derive(Copy, Clone, Debug)]
pub struct KvAppendSlots {
    pub k: MemcpySlot,
    pub v: MemcpySlot,
}

/// # Safety
/// `src.k_new` / `src.v_new` must each point to at least
/// `n_new * n_heads * head_dim * 2` valid device bytes on the same
/// device as `cache`. `device` and `stream` must be the device + stream
/// that own `cache`'s storage; `slots` must be pre-allocated for the
/// caller's `HipGraphExec::capture` scope (or freshly minted via
/// `MemcpySlot::null` outside capture).
pub unsafe fn kv_cache_append_hip_slot<L: CacheLayout>(
    cache: &mut KvCache<L, HipDevice>,
    device: &HipDevice,
    stream: &HipStream,
    src: KvAppendSrc,
    slots: KvAppendSlots,
) -> anyhow::Result<()> {
    let KvAppendSrc { k_new, v_new, n_new } = src;
    let KvAppendSlots { k: k_slot, v: v_slot } = slots;
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
