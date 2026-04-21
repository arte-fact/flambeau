//! Typed KV cache layouts — architectural rule 6: "KV cache layout is a
//! type, not a runtime enum."
//!
//! V1.6.4 ships `KvCache<F16Contig, D>` — the baseline F16 layout that
//! mirrors llama.cpp's default (K and V stored separately, both contiguous
//! in `[max_tokens, n_kv_heads, head_dim]` order). The attention decode /
//! prefill kernels consume `k_buffer()` / `v_buffer()` at `current_tokens`.
//!
//! Follow-ups (own files, own types):
//!   * `KvCache<F16Transposed, D>` — K stored with `[head_dim, max_tokens]`
//!     inner-most order to let flash-attn-v2 prefill stream without an
//!     extra transpose.
//!   * `KvCache<Q8Contig, D>` + `KvCache<Q8Transposed, D>` — 2× HBM
//!     saving; requires the quality cert gate.
//!   * `KvCache<TurboQ4Contig, D>` / `KvCache<TurboQ5Contig, D>` (V2 —
//!     deliberately out of V1 scope per CLAUDE.md).
//!
//! The dispatcher resolves attention impls against the *concrete* layout
//! type, so an "F16 attention" kernel is statically-disallowed from
//! dispatching on a `KvCache<Q8Contig>`. No runtime polymorphism, no
//! downcasts.

use std::marker::PhantomData;

use flambeau_core::{CopyDirection, Device, DevicePtr, DeviceError, DeviceResult, Stream};

/// Marker trait for a KV cache layout. The actual element type + memory
/// order are encoded by the concrete marker struct; this trait just
/// groups them. Different layouts have different per-slot byte costs
/// (F16 is flat, block-quant layouts have `sizeof(block) / block_elems`),
/// so the rule is "bytes per row of `head_dim` elements" — the minimum
/// unit every layout commits to.
pub trait CacheLayout: Send + Sync + 'static {
    /// Short tag used in error messages and dispatch keys.
    const NAME: &'static str;

    /// Bytes consumed by one `(token, head)` row of `head_dim` elements.
    /// `head_dim` must be a multiple of the layout's block size (see
    /// `head_dim_multiple`).
    fn bytes_per_row(head_dim: usize) -> usize;

    /// Any multiple-of-this `head_dim` is supported. F16/BF16 are 1
    /// (arbitrary). Q8_0 is 32 (block size).
    fn head_dim_multiple() -> usize {
        1
    }
}

/// F16, K and V both stored in `[max_tokens, n_heads, head_dim]` order.
/// This is the baseline; matches llama.cpp's `-gwo default` K/V layout.
pub struct F16Contig;

impl CacheLayout for F16Contig {
    const NAME: &'static str = "f16_contig";
    fn bytes_per_row(head_dim: usize) -> usize {
        head_dim * 2
    }
}

/// Q8_0 quantised KV cache: each `head_dim` vector is stored as Q8_0
/// blocks (one `f16` scale per 32 elements + 32 int8 values). At
/// `head_dim=128` that's `128/32 = 4` Q8_0 blocks per (token, head) at
/// `4 × 18 = 72 B` — ~3.6× HBM saving vs F16's `128 × 2 = 256 B`.
///
/// Roadmap requires a **quality cert** (delta-perplexity ≤ 0.5% on
/// wikitext-2 + chat smoke) before this layout is dispatched on a given
/// model. The correctness cert alone (K/V round-trip matches the F16
/// reference within Q8 quant noise) is what V1.6.6 gates on — the
/// per-model quality cert comes with the V1.7 model loader.
pub struct Q8Contig;

impl CacheLayout for Q8Contig {
    const NAME: &'static str = "q8_contig";
    fn bytes_per_row(head_dim: usize) -> usize {
        debug_assert_eq!(head_dim % 32, 0, "Q8Contig needs head_dim % 32 == 0");
        // Each Q8_0 block: 2-byte fp16 scale + 32 int8 values = 18 bytes.
        let n_blocks = head_dim / 32;
        n_blocks * 18
    }
    fn head_dim_multiple() -> usize {
        32
    }
}

#[derive(Debug, thiserror::Error)]
pub enum KvCacheError {
    #[error(transparent)]
    Device(#[from] DeviceError),

    #[error("capacity exceeded: appending {add} tokens to cache at {current}/{cap}")]
    CapacityExceeded {
        current: usize,
        add: usize,
        cap: usize,
    },

    #[error("shape mismatch: cache has n_heads={have_heads} head_dim={have_dim}, caller gave n_heads={given_heads} head_dim={given_dim}")]
    ShapeMismatch {
        have_heads: usize,
        have_dim: usize,
        given_heads: usize,
        given_dim: usize,
    },
}

pub type KvCacheResult<T> = std::result::Result<T, KvCacheError>;

/// Two-buffer KV cache parameterised by layout and device.
///
/// Allocation is eager: `new()` reserves the full `max_tokens * n_heads *
/// head_dim * bytes_per_element()` for both K and V. No realloc on append.
pub struct KvCache<L: CacheLayout, D: Device> {
    k: DevicePtr,
    v: DevicePtr,
    max_tokens: usize,
    current_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    bytes_per_tensor: usize,
    _layout: PhantomData<L>,
    _device: PhantomData<D>,
}

impl<L: CacheLayout, D: Device> KvCache<L, D> {
    /// Allocate K and V buffers sized for `max_tokens` tokens of
    /// `n_heads × head_dim` each. `L::bytes_per_element()` picks the per-slot
    /// byte count.
    pub fn new(
        device: &D,
        n_heads: usize,
        head_dim: usize,
        max_tokens: usize,
    ) -> KvCacheResult<Self> {
        let mul = L::head_dim_multiple();
        assert_eq!(
            head_dim % mul,
            0,
            "KvCache<{}>: head_dim {} must be a multiple of {}",
            L::NAME,
            head_dim,
            mul
        );
        let row_bytes = L::bytes_per_row(head_dim);
        let bytes_per_tensor = max_tokens * n_heads * row_bytes;
        let k = device.alloc(bytes_per_tensor)?;
        let v = device.alloc(bytes_per_tensor)?;
        Ok(Self {
            k,
            v,
            max_tokens,
            current_tokens: 0,
            n_heads,
            head_dim,
            bytes_per_tensor,
            _layout: PhantomData,
            _device: PhantomData,
        })
    }

    /// Append `n_new` tokens' worth of K and V data from device buffers
    /// `k_new` / `v_new`. Both source buffers are expected to be in the
    /// same `[n_new, n_heads, head_dim]` layout as the cache.
    ///
    /// # Safety
    /// `k_new` / `v_new` must point to at least `n_new * n_heads *
    /// head_dim * bytes_per_element()` valid device bytes on the same
    /// device this cache was allocated on.
    pub unsafe fn append(
        &mut self,
        device: &D,
        stream: &D::Stream,
        k_new: DevicePtr,
        v_new: DevicePtr,
        n_new: usize,
    ) -> KvCacheResult<()> {
        if self.current_tokens + n_new > self.max_tokens {
            return Err(KvCacheError::CapacityExceeded {
                current: self.current_tokens,
                add: n_new,
                cap: self.max_tokens,
            });
        }
        let per_token_bytes = self.n_heads * L::bytes_per_row(self.head_dim);
        let offset_bytes = self.current_tokens * per_token_bytes;
        let total_bytes = n_new * per_token_bytes;
        let k_dst = self.k.offset_bytes(offset_bytes);
        let v_dst = self.v.offset_bytes(offset_bytes);
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                k_dst,
                k_new,
                total_bytes,
            )?;
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                v_dst,
                v_new,
                total_bytes,
            )?;
        }
        self.current_tokens += n_new;
        Ok(())
    }

    /// Reset the cache to empty. Does NOT zero the backing memory; the
    /// attention kernel is expected to respect `current_tokens()`.
    pub fn clear(&mut self) {
        self.current_tokens = 0;
    }

    pub fn k_buffer(&self) -> DevicePtr {
        self.k
    }
    pub fn v_buffer(&self) -> DevicePtr {
        self.v
    }
    pub fn current_tokens(&self) -> usize {
        self.current_tokens
    }
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn layout_name(&self) -> &'static str {
        L::NAME
    }
    /// Bytes occupied on-device per tensor (K or V separately).
    pub fn bytes_per_tensor(&self) -> usize {
        self.bytes_per_tensor
    }

    /// Free the underlying allocations. Separate from `Drop` because we
    /// need a `&D` to call `dealloc`; `Drop` can't require a device.
    ///
    /// Callers that forget to call this leak the device buffers — logged
    /// as a warn in V1.7 when session teardown becomes formalised.
    pub fn dispose(mut self, device: &D) -> KvCacheResult<()> {
        // SAFETY: pointers returned by `device.alloc()` on construction
        // have never been aliased elsewhere; no outstanding stream work
        // is pending (caller's contract).
        unsafe {
            device.dealloc(self.k, self.bytes_per_tensor)?;
            device.dealloc(self.v, self.bytes_per_tensor)?;
        }
        self.k = DevicePtr::NULL;
        self.v = DevicePtr::NULL;
        Ok(())
    }
}

impl<L: CacheLayout, D: Device> Drop for KvCache<L, D> {
    fn drop(&mut self) {
        // We can't safely call `device.dealloc` here (no device handle),
        // so detect the common foot-gun: leaking without `dispose()`.
        // Only fire if the pointers are still live; `dispose()` nulls them.
        if !self.k.is_null() || !self.v.is_null() {
            tracing::warn!(
                target: "flambeau_runtime::kv_cache",
                layout = L::NAME,
                bytes = self.bytes_per_tensor * 2,
                "KvCache dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_contig_row_bytes() {
        assert_eq!(F16Contig::bytes_per_row(128), 256);
        assert_eq!(F16Contig::NAME, "f16_contig");
        assert_eq!(F16Contig::head_dim_multiple(), 1);
    }

    #[test]
    fn q8_contig_row_bytes() {
        // head_dim=128 → 4 Q8_0 blocks × 18B = 72B per (token, head).
        assert_eq!(Q8Contig::bytes_per_row(128), 72);
        assert_eq!(Q8Contig::bytes_per_row(256), 144);
        assert_eq!(Q8Contig::head_dim_multiple(), 32);
    }
}
