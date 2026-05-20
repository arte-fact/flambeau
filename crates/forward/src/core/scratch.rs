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
    /// Top-k value (e.g. 8 for Qwen3.6 MoE, 4 for gemma4-MoE). Sizes
    /// the per-token `expert_ids` / `expert_weights` slots and the
    /// per-slot indexed-MoE scratch (`gate_out` / `up_out` /
    /// `activated` / `down`). `0` when no MoE.
    pub max_experts_per_tok: usize,
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
    /// Per-layer side-channel embedding width (gemma 4n / E2B / E4B,
    /// 256 on E4B). `0` when the arch has no per-layer side-channel;
    /// drives the 6 small F32 / F16 scratch slots for the apply block.
    pub per_layer_embd: usize,
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
            max_experts_per_tok: 0,
            gdn: None,
            per_layer_kv_widths: None,
            attn_q_gated: false,
            shared_intermediate: 0,
            max_prefill_tokens: 1,
            max_slots: 1,
            per_layer_embd: 0,
        }
    }
}

/// Maximum split-K chunk count. Sized so that `n_chunks` per
/// `splitk_chunk_size` stays ≤ 32 for any `n_tokens_kv` ≤ 16 384;
/// callers must keep ctx within that bound.
pub const MAX_SPLITK_CHUNKS: usize = 32;

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

    /// `[n_heads_q * MAX_SPLITK_CHUNKS]` F32 — split-K online-softmax
    /// per-chunk running max. NULL until `q_width > 0`.
    pub splitk_partials_m: DevicePtr,
    /// `[n_heads_q * MAX_SPLITK_CHUNKS]` F32 — split-K per-chunk
    /// running denom.
    pub splitk_partials_s: DevicePtr,
    /// `[n_heads_q * MAX_SPLITK_CHUNKS * head_dim]` F32 — split-K
    /// per-chunk numerator outputs. Sized as `q_width * MAX_SPLITK_CHUNKS`
    /// (= n_heads_q * head_dim * MAX_SPLITK_CHUNKS).
    pub splitk_partials_o: DevicePtr,

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

    /// `[max_slots]` U64 — per-slot K-cache base pointers, filled
    /// host-side per forward call and consumed by
    /// `attn_decode_f16_batched`. NULL when `max_slots == 1`.
    pub attn_slot_k_dst_ptrs: DevicePtr,
    pub attn_slot_v_dst_ptrs: DevicePtr,
    /// `[max_slots]` I32 — per-slot write position (= positions[i]).
    pub attn_slot_write_pos: DevicePtr,
    /// `[max_slots]` I32 — per-slot KV length (= positions[i] + 1).
    pub attn_slot_n_kv: DevicePtr,

    /// Shared MoE prefill scratch (`flambeau_blocks` owned, sized
    /// `max_prefill_tokens × top_k × intermediate`). `None` when no MoE
    /// or `max_prefill_tokens <= 1`.
    pub moe_prefill_scratch: Option<flambeau_blocks::OwnedMoeExpertsPrefillScratch>,

    /// `[max_experts_per_tok]` I32 — top-k expert indices. NULL when no MoE.
    pub moe_expert_ids: DevicePtr,
    /// `[max_experts_per_tok]` F32 — top-k normalised expert weights. NULL when no MoE.
    pub moe_expert_weights: DevicePtr,
    /// `[max_experts_per_tok * intermediate]` F32 — indexed gate output.
    pub moe_gate_out_f32: DevicePtr,
    pub moe_up_out_f32: DevicePtr,
    pub moe_activated_f16: DevicePtr,
    pub moe_activated_q8_1: DevicePtr,
    /// `[max_experts_per_tok * hidden]` F32 — indexed down output before combine.
    pub moe_down_f32: DevicePtr,
    pub moe_down_f16: DevicePtr,

    /// Per-layer side-channel embedding scratch (E2B / E4B).
    /// NULL when `config.per_layer_embd == 0`.
    pub ple_gate_out_f32: DevicePtr,
    pub ple_activated_f32: DevicePtr,
    pub ple_activated_f16: DevicePtr,
    pub ple_proj_out_f32: DevicePtr,
    pub ple_proj_out_f16: DevicePtr,
    pub ple_normed_f16: DevicePtr,

    pub kv_caches: Vec<KvCache>,

    /// Per-owned-layer recurrent state + conv history. Empty when
    /// `config.gdn` is None.
    pub gdn_state: Vec<GdnLayerState>,
    /// Shared GDN per-decode scratch wrapping blocks's owned scratch.
    /// `None` when `config.gdn` is None.
    pub gdn_decode_scratch: Option<flambeau_blocks::OwnedDeltaNetLayerDecodeScratch>,
    /// Shared GDN prefill scratch sized for `max_prefill_tokens`. Used
    /// by the composite when the call is single-slot, contiguous, and
    /// `n_tokens > 1` (the prefill-shape path). `None` when GDN is off
    /// or `max_prefill_tokens <= 1`.
    pub gdn_prefill_scratch: Option<flambeau_blocks::OwnedDeltaNetLayerPrefillScratch>,

    pub current_residual_is_a: bool,

    /// When true, `pool.norm` holds an already-rmsnormed F16 buffer
    /// written by the previous composite's BAR1 fused
    /// `residual_rmsnorm_tp2` call. The next composite must consume it
    /// (skip its initial rmsnorm) and clear the flag, OR clear the
    /// flag and ignore the stale value if its rmsnorm uses a different
    /// weight than the one folded in upstream.
    pub input_pre_normed: bool,

    /// When true, the previous composite fused its post-norm+residual
    /// add into the next residual slot directly (gemma4 paths with
    /// post_attn_norm / post_ffn_norm). The matching `residual_add`
    /// must skip its own advance + add and return the incoming `b`
    /// (which IS the new residual) as-is. Set by `standard_attn_local`
    /// / `dense_ffn_local` when they take the fused path; cleared by
    /// `residual_add_local`.
    pub fused_residual_already_done: bool,

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

        // Split-K decode-attn partials. Over-allocate `partials_m/s`
        // at q_width elems (= n_heads_q * head_dim) rather than the
        // exact n_heads_q — saves storing head_dim in ScratchConfig
        // and the waste is sub-MB.
        let (splitk_partials_m, splitk_partials_s, splitk_partials_o) = if qw > 0 {
            let m = alloc_bytes(qw * MAX_SPLITK_CHUNKS * f32)?;
            let s = alloc_bytes(qw * MAX_SPLITK_CHUNKS * f32)?;
            let o = alloc_bytes(qw * MAX_SPLITK_CHUNKS * f32)?;
            (m, s, o)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL)
        };

        // FFN gate/up/activated buffers are shared between the dense
        // FFN path (intermediate=ffn_inter), the qwen-shared-expert
        // path (intermediate=shared_expert_inter, usually =
        // expert_inter), and the gemma4 MoE shared MLP path
        // (shared_intermediate > routed expert_intermediate). Size for
        // the max so all callers fit.
        let m_buf = m.max(config.shared_intermediate);
        let gate_f32 = alloc_bytes(n * m_buf * f32)?;
        let up_f32 = alloc_bytes(n * m_buf * f32)?;
        let gated_f16 = alloc_bytes(n * m_buf * f16)?;
        let gated_q8_1 = alloc_bytes(q8_1(n * m_buf))?;
        let gated_q8_1_mmq = if n > 1 {
            alloc_bytes(q8_1_mmq(m_buf, n))?
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

        // Per-layer side-channel embedding scratch (gemma 4n / E2B / E4B).
        // Six small per-token buffers; pe is typically 256 so total is
        // ~10 KB. Skip alloc when the arch has no per-layer side channel.
        let pe = config.per_layer_embd;
        let (
            ple_gate_out_f32,
            ple_activated_f32,
            ple_activated_f16,
            ple_proj_out_f32,
            ple_proj_out_f16,
            ple_normed_f16,
        ) = if pe > 0 {
            (
                alloc_bytes(n * pe * f32)?,
                alloc_bytes(n * pe * f32)?,
                alloc_bytes(n * pe * f16)?,
                alloc_bytes(n * h * f32)?,
                alloc_bytes(n * h * f16)?,
                alloc_bytes(n * h * f16)?,
            )
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
        };

        // Batched-decode attention scratch. Per-slot pointer/scalar
        // arrays consumed by `attn_decode_f16_batched` and
        // `kv_append_f16_batched_slots`. Allocated only when N > 1
        // (single-slot decode goes through the unbatched kernel).
        let n_slots = config.max_slots.max(1);
        let (attn_slot_k_dst_ptrs, attn_slot_v_dst_ptrs, attn_slot_write_pos, attn_slot_n_kv) =
            if n_slots > 1 {
                let kd = alloc_bytes(n_slots * 8)?;
                let vd = alloc_bytes(n_slots * 8)?;
                let wp = alloc_bytes(n_slots * i32_b)?;
                let nk = alloc_bytes(n_slots * i32_b)?;
                (kd, vd, wp, nk)
            } else {
                (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL)
            };

        let topk = config.max_experts_per_tok;
        let (
            moe_expert_ids,
            moe_expert_weights,
            moe_gate_out_f32,
            moe_up_out_f32,
            moe_activated_f16,
            moe_activated_q8_1,
            moe_down_f32,
            moe_down_f16,
        ) = if topk > 0 && config.max_experts > 0 {
            let ids = alloc_bytes(topk * i32_b)?;
            let weights = alloc_bytes(topk * f32)?;
            let gate = alloc_bytes(topk * m * f32)?;
            let up = alloc_bytes(topk * m * f32)?;
            let act_f16 = alloc_bytes(topk * m * f16)?;
            let act_q8 = alloc_bytes(q8_1(topk * m))?;
            let dn_f32 = alloc_bytes(topk * h * f32)?;
            let dn_f16 = alloc_bytes(topk * h * f16)?;
            (ids, weights, gate, up, act_f16, act_q8, dn_f32, dn_f16)
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
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

        let (gdn_state, gdn_decode_scratch, gdn_prefill_scratch) = if let Some(g) = config.gdn {
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
            allocs.extend(std::mem::take(&mut tracker.allocs));
            let prefill_owned = if config.max_prefill_tokens > 1 {
                let mut p_tracker = flambeau_blocks::RawAllocTracker::new();
                let p = flambeau_blocks::DeltaNetLayer::alloc_prefill_scratch(
                    device,
                    &mut p_tracker,
                    dims,
                    config.max_prefill_tokens,
                )?;
                allocs.extend(std::mem::take(&mut p_tracker.allocs));
                Some(p)
            } else {
                None
            };
            (state_vec, Some(owned), prefill_owned)
        } else {
            (Vec::new(), None, None)
        };

        // MoE prefill scratch — must happen AFTER the `alloc_bytes`
        // closure's last use (allocs is captured-mutably; extending it
        // here is the only safe spot once the closure has been
        // dropped from the borrow checker's perspective).
        let moe_prefill_scratch = if topk > 0
            && config.max_experts > 0
            && config.max_prefill_tokens > 1
        {
            let mut p_tracker = flambeau_blocks::RawAllocTracker::new();
            let dims = flambeau_blocks::MoeExpertsScratchDims {
                hidden: h,
                intermediate: m,
                n_experts: config.max_experts,
                top_k: topk,
            };
            let owned = flambeau_blocks::MoeExperts::alloc_prefill_scratch(
                device,
                &mut p_tracker,
                dims,
                config.max_prefill_tokens,
            )?;
            allocs.extend(std::mem::take(&mut p_tracker.allocs));
            Some(owned)
        } else {
            None
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
            splitk_partials_m,
            splitk_partials_s,
            splitk_partials_o,
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
            attn_slot_k_dst_ptrs,
            attn_slot_v_dst_ptrs,
            attn_slot_write_pos,
            attn_slot_n_kv,
            moe_prefill_scratch,
            moe_expert_ids,
            moe_expert_weights,
            moe_gate_out_f32,
            moe_up_out_f32,
            moe_activated_f16,
            moe_activated_q8_1,
            moe_down_f32,
            moe_down_f16,
            ple_gate_out_f32,
            ple_activated_f32,
            ple_activated_f16,
            ple_proj_out_f32,
            ple_proj_out_f16,
            ple_normed_f16,
            kv_caches,
            gdn_state,
            gdn_decode_scratch,
            gdn_prefill_scratch,
            current_residual_is_a: true,
            input_pre_normed: false,
            fused_residual_already_done: false,
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
