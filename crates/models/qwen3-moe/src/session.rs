//! Per-sequence inference state for a Qwen3.x MoE model.
//!
//! The model weights are read-only; everything that mutates during decode
//! lives here. V1.7.3-a allocates:
//!
//! - `KvCache<F16Contig, HipDevice>` for each full-attention layer (10 of
//!   40 layers in Qwen3.6-35B). Sized for `max_tokens = config.context_length`.
//! - A GDN state buffer for each recurrent layer (30 of 40), shape
//!   `[num_v_heads, head_k_dim, head_v_dim]` F32, zero-initialised so the
//!   first-token recurrence is a clean slate.
//! - A GDN conv1d "history" buffer per recurrent layer, shape
//!   `[conv_kernel - 1, conv_channels]` F32 — holds the last `conv_kernel-1`
//!   input rows needed by the causal conv at decode-time.
//!
//! Dispose mirrors weights.rs: explicit `dispose(device)` frees everything.

#![cfg(feature = "hip")]

use anyhow::{anyhow, Context, Result};
use flambeau_core::{Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;
use flambeau_runtime::{F16Contig, KvCache, Q8Contig};

use crate::config::Qwen3MoEConfig;

/// Per-layer mutable state. Exactly one of the variants is populated,
/// matching the layer's attention family + chosen KV layout.
///
/// V1-BENCH-#116 — `FullAttnQ8` adds a Q8_0-quantised KV variant. Selected
/// at session-construction time (env: `FLAMBEAU_KV=q8` or
/// `kv_layout: KvLayout` ctor param). Per CLAUDE.md rule #6, layouts are
/// distinct types — the enum here is the dispatch boundary; the inner
/// `KvCache<L>` is monomorphic and the attention-decode kernel pick is
/// type-driven via `L::NAME`.
pub enum LayerCache {
    FullAttn(KvCache<F16Contig, HipDevice>),
    FullAttnQ8(KvCache<Q8Contig, HipDevice>),
    Gdn(GdnLayerState),
}

/// V1-BENCH-#116 — runtime selector for the Q8 KV path. Lives at session
/// construction; once chosen, every full-attn layer in this session uses
/// that layout (mixed F16/Q8 across layers is not supported).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvLayout {
    F16,
    Q8,
}

impl KvLayout {
    /// Read `FLAMBEAU_KV` env (values: "f16" default, "q8" / "q8_0").
    /// Used by every Session ctor so the choice propagates through the
    /// stack without threading another argument.
    pub fn from_env() -> Self {
        match std::env::var("FLAMBEAU_KV").as_deref() {
            Ok("q8") | Ok("q8_0") | Ok("Q8") | Ok("Q8_0") => Self::Q8,
            _ => Self::F16,
        }
    }
}

/// F32 device buffers for one GDN layer. Shapes keyed off `GdnDims`.
pub struct GdnLayerState {
    /// `[num_v_heads, head_k_dim, head_v_dim]` — the linear-attention state
    /// matrix. Row-major; kernel loads one column per warp. Zero on init.
    pub state: DevicePtr,
    pub state_bytes: usize,
    /// `[conv_kernel - 1, conv_channels]` — the last `kernel-1` input rows
    /// needed by the causal conv1d. Zero on init; caller concats a new row
    /// on top each step.
    pub conv_history: DevicePtr,
    pub conv_history_bytes: usize,
    /// Cached shape parameters so the forward pass doesn't re-derive them.
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel: usize,
    pub conv_channels: usize,
}

/// All per-sequence mutable state. One instance per active request on the
/// mesh; never shared between requests.
pub struct Qwen3MoESession {
    caches: Vec<LayerCache>,
    device_id: i32,
    disposed: bool,
}

impl Qwen3MoESession {
    /// Allocate every per-layer cache sized for `cfg.context_length` tokens.
    /// Full-attention layers get a `KvCache<F16Contig>`; recurrent layers
    /// get zero-initialised GDN state + conv-history buffers.
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        device.bind()?;
        let mut caches = Vec::with_capacity(cfg.num_layers);
        for il in 0..cfg.num_layers {
            caches.push(alloc_layer_cache(cfg, device, il)?);
        }
        device.default_stream().synchronize()?;
        Ok(Self {
            caches,
            device_id: device.id(),
            disposed: false,
        })
    }

    /// HIP device id this session was allocated on.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// V1-BENCH-#116 — true if any full-attn layer uses the Q8_0 KV layout.
    /// Callers (forward_prefill_pp, forward_prefill_tp_logits) gate the
    /// batched prefill paths on this; Q8 KV currently has no batched
    /// prefill kernel, so prefill falls back to per-token decode.
    pub fn is_q8_kv(&self) -> bool {
        self.caches.iter().any(|c| matches!(c, LayerCache::FullAttnQ8(_)))
    }

    /// Total bytes allocated across every per-layer cache.
    pub fn total_bytes(&self) -> usize {
        let mut total = 0usize;
        for c in &self.caches {
            match c {
                LayerCache::FullAttn(kv) => {
                    total += kv.bytes_per_tensor() * 2;
                }
                LayerCache::FullAttnQ8(kv) => {
                    total += kv.bytes_per_tensor() * 2;
                }
                LayerCache::Gdn(g) => {
                    total += g.state_bytes + g.conv_history_bytes;
                }
            }
        }
        total
    }

    pub fn layers(&self) -> &[LayerCache] {
        &self.caches
    }

    pub fn layers_mut(&mut self) -> &mut [LayerCache] {
        &mut self.caches
    }

    /// Free every per-layer allocation. Required — see [`ModelWeights::dispose`].
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut first_err: Option<anyhow::Error> = None;
        for cache in self.caches.drain(..) {
            if let Err(e) = dispose_layer_cache(cache, device) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Qwen3MoESession {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::session",
                device_id = self.device_id,
                layers = self.caches.len(),
                "Qwen3MoESession dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Allocate a single layer's cache on `device`. Dispatches on `cfg.is_recurrent(il)`.
/// Shared between the single-device `Qwen3MoESession::new` and the sharded
/// session's per-rank initialisation.
pub(crate) fn alloc_layer_cache(
    cfg: &Qwen3MoEConfig,
    device: &HipDevice,
    il: usize,
) -> Result<LayerCache> {
    if cfg.is_recurrent(il) {
        let gdn = cfg
            .gdn
            .as_ref()
            .context("recurrent layer requires cfg.gdn to be Some")?;
        let head_v_dim = gdn.head_v_dim();
        let state_elems = gdn.num_v_heads * gdn.head_k_dim * head_v_dim;
        let state_bytes = state_elems * std::mem::size_of::<f32>();
        let state = device
            .alloc(state_bytes)
            .map_err(|e| anyhow!("alloc GDN state for layer {il}: {e}"))?;
        zero_f32(device, state, state_bytes)?;

        let conv_channels = gdn.conv_channels();
        let conv_history_elems = (gdn.conv_kernel - 1) * conv_channels;
        let conv_history_bytes = conv_history_elems * std::mem::size_of::<f32>();
        let conv_history = device
            .alloc(conv_history_bytes)
            .map_err(|e| anyhow!("alloc GDN conv-history for layer {il}: {e}"))?;
        zero_f32(device, conv_history, conv_history_bytes)?;

        Ok(LayerCache::Gdn(GdnLayerState {
            state,
            state_bytes,
            conv_history,
            conv_history_bytes,
            num_v_heads: gdn.num_v_heads,
            head_k_dim: gdn.head_k_dim,
            head_v_dim,
            conv_kernel: gdn.conv_kernel,
            conv_channels,
        }))
    } else {
        match KvLayout::from_env() {
            KvLayout::F16 => {
                let kv = KvCache::<F16Contig, HipDevice>::new(
                    device,
                    cfg.num_kv_heads,
                    cfg.head_dim,
                    cfg.context_length,
                )
                .map_err(|e| anyhow!("alloc KvCache<F16> for layer {il}: {e}"))?;
                Ok(LayerCache::FullAttn(kv))
            }
            KvLayout::Q8 => {
                let kv = KvCache::<Q8Contig, HipDevice>::new(
                    device,
                    cfg.num_kv_heads,
                    cfg.head_dim,
                    cfg.context_length,
                )
                .map_err(|e| anyhow!("alloc KvCache<Q8> for layer {il}: {e}"))?;
                Ok(LayerCache::FullAttnQ8(kv))
            }
        }
    }
}

/// **TP-2e** — per-rank `LayerCache` allocator for tensor-parallel
/// decode. Uses `local_num_v_heads = num_v_heads / tp_world` and
/// `local_num_kv_heads = num_kv_heads / tp_world` so each rank's
/// `KvCache` and `GdnLayerState` are sized for the local head subset
/// only. Without this, kernels parameterised by `local_*` head counts
/// would walk into uninitialised slabs in oversized PP-shape allocations.
///
/// Caller (`Qwen3MoETpSession::new`) loops over ranks × layers and
/// binds the device per rank.
pub(crate) fn alloc_layer_cache_tp(
    cfg: &Qwen3MoEConfig,
    device: &HipDevice,
    il: usize,
    tp_world: u32,
) -> Result<LayerCache> {
    if tp_world == 0 {
        anyhow::bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    if cfg.is_recurrent(il) {
        let gdn = cfg
            .gdn
            .as_ref()
            .context("recurrent layer requires cfg.gdn to be Some")?;
        if gdn.num_v_heads % world != 0 {
            anyhow::bail!(
                "alloc_layer_cache_tp: gdn.num_v_heads {} not divisible by tp_world {tp_world}",
                gdn.num_v_heads
            );
        }
        if gdn.num_k_heads % world != 0 {
            anyhow::bail!(
                "alloc_layer_cache_tp: gdn.num_k_heads {} not divisible by tp_world {tp_world}",
                gdn.num_k_heads
            );
        }
        let head_v_dim = gdn.head_v_dim();
        let local_num_v_heads = gdn.num_v_heads / world;
        let local_num_k_heads = gdn.num_k_heads / world;
        let state_elems = local_num_v_heads * gdn.head_k_dim * head_v_dim;
        let state_bytes = state_elems * std::mem::size_of::<f32>();
        let state = device
            .alloc(state_bytes)
            .map_err(|e| anyhow!("alloc GDN state (TP) for layer {il}: {e}"))?;
        zero_f32(device, state, state_bytes)?;

        // local_conv_channels = local_d_inner + 2 · local_qk_size.
        let local_d_inner = local_num_v_heads * head_v_dim;
        let local_qk_size = local_num_k_heads * gdn.head_k_dim;
        let local_conv_channels = local_d_inner + 2 * local_qk_size;
        let conv_history_elems = (gdn.conv_kernel - 1) * local_conv_channels;
        let conv_history_bytes = conv_history_elems * std::mem::size_of::<f32>();
        let conv_history = device
            .alloc(conv_history_bytes)
            .map_err(|e| anyhow!("alloc GDN conv-history (TP) for layer {il}: {e}"))?;
        zero_f32(device, conv_history, conv_history_bytes)?;

        Ok(LayerCache::Gdn(GdnLayerState {
            state,
            state_bytes,
            conv_history,
            conv_history_bytes,
            num_v_heads: local_num_v_heads,
            head_k_dim: gdn.head_k_dim,
            head_v_dim,
            conv_kernel: gdn.conv_kernel,
            conv_channels: local_conv_channels,
        }))
    } else {
        // **TP-4d-i2** — KV-replication fallback: when nKV doesn't
        // divide world, allocate the full KvCache on every rank.
        // Caller (forward_full_attn_decode_tp) passes kv_replicated=true
        // and the K/V projections run with full nKV per rank.
        let local_n_kv_heads = if cfg.num_kv_heads % world == 0 {
            cfg.num_kv_heads / world
        } else {
            cfg.num_kv_heads
        };
        match KvLayout::from_env() {
            KvLayout::F16 => {
                let kv = KvCache::<F16Contig, HipDevice>::new(
                    device,
                    local_n_kv_heads,
                    cfg.head_dim,
                    cfg.context_length,
                )
                .map_err(|e| anyhow!("alloc KvCache<F16> (TP) for layer {il}: {e}"))?;
                Ok(LayerCache::FullAttn(kv))
            }
            KvLayout::Q8 => {
                let kv = KvCache::<Q8Contig, HipDevice>::new(
                    device,
                    local_n_kv_heads,
                    cfg.head_dim,
                    cfg.context_length,
                )
                .map_err(|e| anyhow!("alloc KvCache<Q8> (TP) for layer {il}: {e}"))?;
                Ok(LayerCache::FullAttnQ8(kv))
            }
        }
    }
}

/// Free a single `LayerCache` on `device`. Mirrors `alloc_layer_cache`.
pub(crate) fn dispose_layer_cache(cache: LayerCache, device: &HipDevice) -> Result<()> {
    match cache {
        LayerCache::FullAttn(kv) => kv
            .dispose(device)
            .map_err(|e| anyhow!("KvCache<F16> dispose: {e}")),
        LayerCache::FullAttnQ8(kv) => kv
            .dispose(device)
            .map_err(|e| anyhow!("KvCache<Q8> dispose: {e}")),
        LayerCache::Gdn(g) => {
            // SAFETY: both pointers came from `device.alloc(..)` in alloc_layer_cache.
            unsafe {
                device
                    .dealloc(g.state, g.state_bytes)
                    .map_err(|e| anyhow!("hipFree GDN state: {e}"))?;
                device
                    .dealloc(g.conv_history, g.conv_history_bytes)
                    .map_err(|e| anyhow!("hipFree GDN conv-history: {e}"))?;
            }
            Ok(())
        }
    }
}

/// Zero-fill `bytes` of device memory at `ptr`. Uses a host-side zero buffer
/// and `memcpy_async` — the V1 HIP backend doesn't expose `hipMemset` through
/// its `Device` trait yet, and this path is on the cold (session-init) path
/// so a ~2 MB staging buffer is not a concern.
fn zero_f32(device: &HipDevice, ptr: DevicePtr, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let host_zero = vec![0u8; bytes];
    let stream = device.default_stream();
    // SAFETY: `ptr` points to `bytes` fresh HIP bytes; `host_zero` is a
    // host vec of length `bytes`.
    unsafe {
        device.memcpy_async(
            stream,
            flambeau_core::CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host_zero.as_ptr() as usize),
            bytes,
        )?;
    }
    // host_zero must outlive the memcpy until the stream has consumed it —
    // synchronise before dropping it.
    stream.synchronize()?;
    drop(host_zero);
    Ok(())
}
