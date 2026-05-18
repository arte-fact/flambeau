//! Pre-allocated device scratch + per-layer KV cache.

use anyhow::{Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};

use crate::ctx::GdnDims;

/// `q_width` / `kv_width` are PER-RANK under TP (caller divides by
/// tp_size). `num_layers` is the count of owned KV slots — PP rank
/// owning a layer slice passes the slice length, not the global total.
/// `max_experts` sizes the MoE router logits slot; 0 for dense-only.
/// `gdn` is Some for hybrid arches; sizes the per-layer state +
/// conv-history slots (allocated once per owned layer).
#[derive(Clone, Copy, Debug, Default)]
pub struct ScratchConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub q_width: usize,
    pub kv_width: usize,
    pub vocab: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub max_experts: usize,
    pub gdn: Option<GdnDims>,
}

#[derive(Clone, Copy)]
pub struct KvCache {
    pub k: DevicePtr,
    pub v: DevicePtr,
}

/// Caller must invoke `dispose(device)` before drop to release HBM.
pub struct ScratchPool {
    pub config: ScratchConfig,

    pub resid_a: DevicePtr,
    pub resid_b: DevicePtr,
    pub norm: DevicePtr,
    pub delta: DevicePtr,

    pub norm_q8_1: DevicePtr,
    pub q_f16: DevicePtr,
    pub k_f16: DevicePtr,
    pub v_f16: DevicePtr,
    pub attn_out_f16: DevicePtr,
    pub attn_out_q8_1: DevicePtr,
    pub attn_proj_f32: DevicePtr,

    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub gated_f16: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub down_f32: DevicePtr,

    pub logits_f32_dev: DevicePtr,
    pub position_i32: DevicePtr,

    /// `[max_experts]` F32. NULL when `config.max_experts == 0`.
    pub router_logits_f32: DevicePtr,
    /// `[hidden]` F16 MoE per-expert accumulator. NULL when `max_experts == 0`.
    pub moe_accum_f16: DevicePtr,

    pub kv_caches: Vec<KvCache>,

    /// Per-owned-layer recurrent state + conv history. Empty when
    /// `config.gdn` is None.
    pub gdn_state: Vec<GdnLayerState>,
    /// Shared GDN per-decode scratch wrapping blocks's owned scratch.
    /// `None` when `config.gdn` is None.
    pub gdn_decode_scratch: Option<flambeau_blocks::OwnedDeltaNetLayerDecodeScratch>,

    pub current_residual_is_a: bool,

    allocs: Vec<(DevicePtr, usize)>,
}

#[derive(Clone, Copy)]
pub struct GdnLayerState {
    /// `[num_v_heads, head_k_dim, head_v_dim]` F32 recurrent state.
    pub state: DevicePtr,
    /// `[conv_kernel - 1, conv_channels]` F32 conv1d history.
    pub conv_history: DevicePtr,
}

impl ScratchPool {
    pub fn new(device: &HipDevice, config: ScratchConfig) -> Result<Self> {
        let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();
        let mut alloc_bytes = |bytes: usize| -> Result<DevicePtr> {
            let p = device.alloc(bytes).context("alloc")?;
            allocs.push((p, bytes));
            Ok(p)
        };

        let f16 = 2;
        let f32 = 4;
        let i32_b = 4;
        let q8_1 = |n: usize| n.div_ceil(32) * 36;

        let h = config.hidden;
        let m = config.intermediate;
        let qw = config.q_width;
        let kvw = config.kv_width;

        let resid_a = alloc_bytes(h * f16)?;
        let resid_b = alloc_bytes(h * f16)?;
        let norm = alloc_bytes(h * f16)?;
        let delta = alloc_bytes(h * f16)?;

        let norm_q8_1 = alloc_bytes(q8_1(h))?;
        let q_f16 = alloc_bytes(qw * f16)?;
        let k_f16 = alloc_bytes(kvw * f16)?;
        let v_f16 = alloc_bytes(kvw * f16)?;
        let attn_out_f16 = alloc_bytes(qw * f16)?;
        let attn_out_q8_1 = alloc_bytes(q8_1(qw))?;
        // Reused for Q / K / V / output-proj F32 — size to the max.
        let attn_proj_f32 = alloc_bytes(qw.max(kvw).max(h) * f32)?;

        let gate_f32 = alloc_bytes(m * f32)?;
        let up_f32 = alloc_bytes(m * f32)?;
        let gated_f16 = alloc_bytes(m * f16)?;
        let gated_q8_1 = alloc_bytes(q8_1(m))?;
        let down_f32 = alloc_bytes(h * f32)?;

        let logits_f32_dev = alloc_bytes(config.vocab * f32)?;
        let position_i32 = alloc_bytes(i32_b)?;

        let (router_logits_f32, moe_accum_f16) = if config.max_experts > 0 {
            let r = alloc_bytes(config.max_experts * f32)?;
            let a = alloc_bytes(h * f16)?;
            (r, a)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        let mut kv_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let k = alloc_bytes(config.max_seq_len * kvw * f16)?;
            let v = alloc_bytes(config.max_seq_len * kvw * f16)?;
            kv_caches.push(KvCache { k, v });
        }

        let (gdn_state, gdn_decode_scratch) = if let Some(g) = config.gdn {
            let mut state_vec = Vec::with_capacity(config.num_layers);
            let state_bytes = g.num_v_heads * g.head_k_dim * g.head_v_dim * f32;
            let hist_bytes = (g.conv_kernel - 1) * g.conv_channels * f32;
            for _ in 0..config.num_layers {
                let state = alloc_bytes(state_bytes)?;
                let conv_history = alloc_bytes(hist_bytes)?;
                state_vec.push(GdnLayerState { state, conv_history });
            }
            // Per-decode scratch lives in a blocks-owned RawAllocTracker
            // we then drain into our `allocs` list for a single dispose path.
            let mut tracker = flambeau_blocks::RawAllocTracker::new();
            let dims = flambeau_blocks::DeltaNetScratchDims {
                hidden: h,
                d_inner: g.d_inner,
                num_v_heads: g.num_v_heads,
                num_k_heads: g.num_k_heads,
                head_k_dim: g.head_k_dim,
                head_v_dim: g.head_v_dim,
                conv_channels: g.conv_channels,
                conv_kernel: g.conv_kernel,
            };
            let owned = flambeau_blocks::DeltaNetLayer::alloc_decode_scratch(
                device,
                &mut tracker,
                dims,
            )?;
            // Fold the blocks-tracker allocs into our own list (the
            // tracker's `allocs` field is public). Replace with empty
            // so `RawAllocTracker::Drop` doesn't double-dispose.
            allocs.extend(std::mem::take(&mut tracker.allocs));
            (state_vec, Some(owned))
        } else {
            (Vec::new(), None)
        };

        Ok(Self {
            config,
            resid_a,
            resid_b,
            norm,
            delta,
            norm_q8_1,
            q_f16,
            k_f16,
            v_f16,
            attn_out_f16,
            attn_out_q8_1,
            attn_proj_f32,
            gate_f32,
            up_f32,
            gated_f16,
            gated_q8_1,
            down_f32,
            logits_f32_dev,
            position_i32,
            router_logits_f32,
            moe_accum_f16,
            kv_caches,
            gdn_state,
            gdn_decode_scratch,
            current_residual_is_a: true,
            allocs,
        })
    }

    /// Idempotent.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        for (ptr, bytes) in self.allocs.drain(..) {
            // SAFETY: `ptr` came from `device.alloc(bytes)`; caller
            // contract: no further forward calls past dispose.
            unsafe { device.dealloc(ptr, bytes) }.context("dealloc")?;
        }
        Ok(())
    }

    /// Flip residual ping-pong and return the live slot. Called by
    /// `embed` and every `residual_add`.
    pub fn next_residual_slot(&mut self) -> DevicePtr {
        self.current_residual_is_a = !self.current_residual_is_a;
        if self.current_residual_is_a {
            self.resid_a
        } else {
            self.resid_b
        }
    }
}
