//! Pre-allocated device scratch + per-layer KV cache.

use anyhow::{Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};

use crate::ctx::GdnDims;

/// `q_width` / `kv_width` are PER-RANK under TP (caller divides by
/// tp_size) AND act as upper bounds across layers — they size the
/// shared (ephemeral) Q / K / V / projection scratch slots, which
/// every layer reuses. For uniform arches they equal the per-layer
/// values; for gemma4-style SWA/global alternation they are the
/// per-layer max.
///
/// `num_layers` is the count of owned KV slots — PP rank owning a
/// layer slice passes the slice length, not the global total.
///
/// `per_layer_kv_widths`: when `Some`, length must equal `num_layers`
/// and each entry sets that slot's KV-cache stride (used by gemma4
/// SWA layers, which need half the cache of global-attention layers).
/// When `None`, every slot is sized at `kv_width`.
///
/// `max_experts` sizes the MoE router logits slot; 0 for dense-only.
/// `gdn` is Some for hybrid arches; sizes the per-layer state +
/// conv-history slots (allocated once per owned layer).
#[derive(Clone, Debug)]
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
    pub per_layer_kv_widths: Option<Vec<usize>>,
    /// `true` when the arch has gated full-attention (qwen3.5 /
    /// qwen3.6 / qwen3-Next). Drives allocation of the extra
    /// `q_fused_f16` (2·q_width F16) + `gate_f16` (q_width F16)
    /// scratch slots the split-then-sigmoid-gate path needs.
    pub attn_q_gated: bool,
    /// Per-layer shared-expert FFN intermediate size. `0` when the
    /// MoE arch has no shared expert; positive when one is present
    /// (Qwen3.6-35B-A3B = 512, qwen3next = ...). Drives `shared_x_norm_f32`
    /// scratch sizing.
    pub shared_intermediate: usize,
    /// Upper bound on tokens-per-forward. 1 for decode-only; >1 for
    /// chunked prefill. Every per-token scratch slot (resid, norm,
    /// q/k/v, gate, attn_out, gate_f32/up_f32/gated, down, moe_accum,
    /// shared_x_norm, router_logits, position_i32) is sized at
    /// `max_prefill_tokens * <per-token width>`.
    pub max_prefill_tokens: usize,
    /// Number of independent inflight slots this pool reserves KV/GDN
    /// state for. 1 = single-request decode + chunked prefill. >1 =
    /// batched-decode across N concurrent slots. Each layer's
    /// `kv_caches[li].k/v` is sized `[max_slots, max_seq_len, kv_width]`;
    /// GDN per-layer state + conv_history multiply by `max_slots`.
    pub max_slots: usize,
}

impl Default for ScratchConfig {
    fn default() -> Self {
        Self {
            hidden: 0,
            intermediate: 0,
            q_width: 0,
            kv_width: 0,
            vocab: 0,
            max_seq_len: 0,
            num_layers: 0,
            max_experts: 0,
            gdn: None,
            per_layer_kv_widths: None,
            attn_q_gated: false,
            shared_intermediate: 0,
            max_prefill_tokens: 1,
            max_slots: 1,
        }
    }
}

#[derive(Clone, Copy)]
pub struct KvCache {
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub kv_width: usize,
}

/// Caller must invoke `dispose(device)` before drop to release HBM.
pub struct ScratchPool {
    pub config: ScratchConfig,

    pub resid_a: DevicePtr,
    pub resid_b: DevicePtr,
    pub norm: DevicePtr,
    pub delta: DevicePtr,

    pub norm_q8_1: DevicePtr,
    pub norm_q8_1_mmq: DevicePtr,
    pub q_f16: DevicePtr,
    pub k_f16: DevicePtr,
    pub v_f16: DevicePtr,
    /// `[2 * q_width]` F16 — fused `[Q | gate]` projection output for
    /// gated full-attention arches. `DevicePtr::NULL` otherwise.
    pub q_fused_f16: DevicePtr,
    /// `[q_width]` F16 — per-head sigmoid gate for gated full-attn.
    /// `DevicePtr::NULL` otherwise.
    pub gate_f16: DevicePtr,
    pub attn_out_f16: DevicePtr,
    pub attn_out_q8_1: DevicePtr,
    pub attn_out_q8_1_mmq: DevicePtr,
    pub attn_proj_f32: DevicePtr,

    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub gated_f16: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub gated_q8_1_mmq: DevicePtr,
    pub down_f32: DevicePtr,

    pub logits_f32_dev: DevicePtr,
    pub position_i32: DevicePtr,

    /// `[max_experts]` F32. NULL when `config.max_experts == 0`.
    pub router_logits_f32: DevicePtr,
    /// `[hidden]` F16 MoE per-expert accumulator. NULL when `max_experts == 0`.
    pub moe_accum_f16: DevicePtr,
    /// `[hidden]` F32 — F32 cast of x_norm for the shared expert's
    /// per-token gate scale step. NULL when no shared expert.
    pub shared_x_norm_f32: DevicePtr,

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
        // MMQ block: 144 B per 128 elements, row-major over (ncols/128, total_b).
        let q8_1_mmq = |cols: usize, rows: usize| cols.div_ceil(128) * rows * 144;

        let h = config.hidden;
        let m = config.intermediate;
        let qw = config.q_width;
        let kvw = config.kv_width;
        let n = config.max_prefill_tokens.max(1);

        let resid_a = alloc_bytes(n * h * f16)?;
        let resid_b = alloc_bytes(n * h * f16)?;
        let norm = alloc_bytes(n * h * f16)?;
        let delta = alloc_bytes(n * h * f16)?;

        let norm_q8_1 = alloc_bytes(q8_1(n * h))?;
        let norm_q8_1_mmq = if n > 1 {
            alloc_bytes(q8_1_mmq(h, n))?
        } else {
            DevicePtr::NULL
        };
        let q_f16 = alloc_bytes(n * qw * f16)?;
        let k_f16 = alloc_bytes(n * kvw * f16)?;
        let v_f16 = alloc_bytes(n * kvw * f16)?;
        let (q_fused_f16, gate_f16) = if config.attn_q_gated {
            (alloc_bytes(n * 2 * qw * f16)?, alloc_bytes(n * qw * f16)?)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let attn_out_f16 = alloc_bytes(n * qw * f16)?;
        let attn_out_q8_1 = alloc_bytes(q8_1(n * qw))?;
        let attn_out_q8_1_mmq = if n > 1 {
            alloc_bytes(q8_1_mmq(qw, n))?
        } else {
            DevicePtr::NULL
        };
        let q_or_fused = if config.attn_q_gated { 2 * qw } else { qw };
        let attn_proj_f32 = alloc_bytes(n * q_or_fused.max(kvw).max(h) * f32)?;

        let gate_f32 = alloc_bytes(n * m * f32)?;
        let up_f32 = alloc_bytes(n * m * f32)?;
        let gated_f16 = alloc_bytes(n * m * f16)?;
        let gated_q8_1 = alloc_bytes(q8_1(n * m))?;
        let gated_q8_1_mmq = if n > 1 {
            alloc_bytes(q8_1_mmq(m, n))?
        } else {
            DevicePtr::NULL
        };
        let down_f32 = alloc_bytes(n * h * f32)?;

        let logits_f32_dev = alloc_bytes(config.max_slots.max(1) * config.vocab * f32)?;
        let position_i32 = alloc_bytes(n * i32_b)?;

        let (router_logits_f32, moe_accum_f16) = if config.max_experts > 0 {
            let r = alloc_bytes(n * config.max_experts * f32)?;
            let a = alloc_bytes(n * h * f16)?;
            (r, a)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let shared_x_norm_f32 = if config.shared_intermediate > 0 {
            alloc_bytes(n * h * f32)?
        } else {
            DevicePtr::NULL
        };

        if let Some(per) = config.per_layer_kv_widths.as_ref() {
            if per.len() != config.num_layers {
                anyhow::bail!(
                    "per_layer_kv_widths.len() {} != num_layers {}",
                    per.len(),
                    config.num_layers
                );
            }
            for (li, &w) in per.iter().enumerate() {
                if w > kvw {
                    anyhow::bail!(
                        "per_layer_kv_widths[{li}] = {w} > kv_width {kvw} \
                         (kv_width must be >= max per-layer kv_width — \
                         it sizes the shared K/V scratch)",
                    );
                }
            }
        }
        let n_slots = config.max_slots.max(1);
        let mut kv_caches = Vec::with_capacity(config.num_layers);
        for li in 0..config.num_layers {
            let slot_kvw = config
                .per_layer_kv_widths
                .as_ref()
                .map(|p| p[li])
                .unwrap_or(kvw);
            let k = alloc_bytes(n_slots * config.max_seq_len * slot_kvw * f16)?;
            let v = alloc_bytes(n_slots * config.max_seq_len * slot_kvw * f16)?;
            kv_caches.push(KvCache {
                k,
                v,
                kv_width: slot_kvw,
            });
        }

        let (gdn_state, gdn_decode_scratch) = if let Some(g) = config.gdn {
            let mut state_vec = Vec::with_capacity(config.num_layers);
            let state_bytes = n_slots * g.num_v_heads * g.head_k_dim * g.head_v_dim * f32;
            let hist_bytes = n_slots * (g.conv_kernel - 1) * g.conv_channels * f32;
            // Recurrent state + conv history must start at zero —
            // the step kernel reads them every call, including
            // position=0. `device.alloc` is uninitialised.
            let zero_buf = vec![0u8; state_bytes.max(hist_bytes)];
            let stream = device.default_stream();
            for _ in 0..config.num_layers {
                let state = alloc_bytes(state_bytes)?;
                let conv_history = alloc_bytes(hist_bytes)?;
                // SAFETY: state owns state_bytes, conv_history owns
                // hist_bytes, zero_buf has >= max(state_bytes, hist_bytes).
                unsafe {
                    device
                        .memcpy_async(
                            stream,
                            CopyDirection::HostToDevice,
                            state,
                            DevicePtr(zero_buf.as_ptr() as usize),
                            state_bytes,
                        )
                        .context("zero gdn state")?;
                    device
                        .memcpy_async(
                            stream,
                            CopyDirection::HostToDevice,
                            conv_history,
                            DevicePtr(zero_buf.as_ptr() as usize),
                            hist_bytes,
                        )
                        .context("zero gdn conv_history")?;
                }
                state_vec.push(GdnLayerState { state, conv_history });
            }
            flambeau_core::Stream::synchronize(stream).context("sync gdn zero")?;
            // Drain the blocks-owned RawAllocTracker into our list
            // so a single dispose walks every alloc.
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
            // `mem::take` so `RawAllocTracker::Drop` doesn't double-dispose.
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
            norm_q8_1_mmq,
            q_f16,
            k_f16,
            v_f16,
            q_fused_f16,
            gate_f16,
            attn_out_f16,
            attn_out_q8_1,
            attn_out_q8_1_mmq,
            attn_proj_f32,
            gate_f32,
            up_f32,
            gated_f16,
            gated_q8_1,
            gated_q8_1_mmq,
            down_f32,
            logits_f32_dev,
            position_i32,
            router_logits_f32,
            moe_accum_f16,
            shared_x_norm_f32,
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

    /// Zero per-layer GDN recurrent state + conv-history slabs so the
    /// next request starts fresh. No-op when the arch is non-GDN.
    /// KV-cache positions are caller-supplied (no counter on the
    /// pool), so this is the only stateful slot that needs reset.
    pub fn reset_gdn_state(&self, device: &HipDevice) -> Result<()> {
        let Some(g) = self.config.gdn else {
            return Ok(());
        };
        if self.gdn_state.is_empty() {
            return Ok(());
        }
        let f32 = 4;
        let n_slots = self.config.max_slots.max(1);
        let state_bytes = n_slots * g.num_v_heads * g.head_k_dim * g.head_v_dim * f32;
        let hist_bytes = n_slots * (g.conv_kernel - 1) * g.conv_channels * f32;
        let zero_buf = vec![0u8; state_bytes.max(hist_bytes)];
        let stream = device.default_stream();
        for ls in &self.gdn_state {
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ls.state,
                        DevicePtr(zero_buf.as_ptr() as usize),
                        state_bytes,
                    )
                    .context("reset gdn state")?;
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ls.conv_history,
                        DevicePtr(zero_buf.as_ptr() as usize),
                        hist_bytes,
                    )
                    .context("reset gdn conv_history")?;
            }
        }
        flambeau_core::Stream::synchronize(stream).context("sync gdn reset")?;
        Ok(())
    }

    /// Zero one slot's GDN state + conv-history across every layer.
    /// Used by the server's shared-Session pool to reset a single
    /// conversation slot without touching others. No-op for non-GDN
    /// archs or when the slot index is out of range.
    pub fn reset_gdn_state_slot(&self, slot_id: usize, device: &HipDevice) -> Result<()> {
        let Some(g) = self.config.gdn else {
            return Ok(());
        };
        if self.gdn_state.is_empty() {
            return Ok(());
        }
        let n_slots = self.config.max_slots.max(1);
        if slot_id >= n_slots {
            anyhow::bail!(
                "reset_gdn_state_slot: slot_id {slot_id} >= max_slots {n_slots}"
            );
        }
        let f32 = 4;
        let state_bytes_per_slot = g.num_v_heads * g.head_k_dim * g.head_v_dim * f32;
        let hist_bytes_per_slot = (g.conv_kernel - 1) * g.conv_channels * f32;
        let zero_buf = vec![0u8; state_bytes_per_slot.max(hist_bytes_per_slot)];
        let stream = device.default_stream();
        for ls in &self.gdn_state {
            let state_off = ls.state.offset_bytes(slot_id * state_bytes_per_slot);
            let hist_off = ls.conv_history.offset_bytes(slot_id * hist_bytes_per_slot);
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        state_off,
                        DevicePtr(zero_buf.as_ptr() as usize),
                        state_bytes_per_slot,
                    )
                    .context("reset gdn state slot")?;
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        hist_off,
                        DevicePtr(zero_buf.as_ptr() as usize),
                        hist_bytes_per_slot,
                    )
                    .context("reset gdn conv_history slot")?;
            }
        }
        flambeau_core::Stream::synchronize(stream).context("sync gdn reset slot")?;
        Ok(())
    }
}
