//! Single-device Gemma 4 session. Holds device weights, per-layer KV
//! caches, persistent scratch device buffers, and the host-side
//! position buffer. The driver functions in [`crate::single_device`]
//! consume `&mut Gemma4Session`.

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::{Context, Result};
use flambeau_core::{Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;
use flambeau_quant::GgufFile;
use flambeau_runtime::{F16Contig, KvCache};
use half::f16;

use crate::config::Gemma4Config;
use crate::layout::ModelLayout;
use crate::moe::Gemma4MoeScratch;
use crate::output_head::OutputHeadScratch;
use crate::weights_hip::Gemma4DeviceWeights;
use flambeau_blocks::{MoeExpertsDecodeScratch, RawAllocTracker};

/// Persistent device scratch for one decode step. Allocated once at
/// session-init; the per-call [`crate::scratch::LayerDecodeScratch`]
/// view is reconstructed each forward call from these pointers.
struct LayerScratchPtrs {
    x_q8_1: DevicePtr,
    mmvq_f32: DevicePtr,
    q_f16: DevicePtr,
    k_f16: DevicePtr,
    v_f16: DevicePtr,
    attn_out_f16: DevicePtr,
    post_attn_norm_f16: DevicePtr,
    attn_residual_f16: DevicePtr,
    ffn_norm_f16: DevicePtr,
    gate_f32: DevicePtr,
    up_f32: DevicePtr,
    activated_f16: DevicePtr,
    activated_q8_1: DevicePtr,
    down_f32: DevicePtr,
    post_ffw_norm_f16: DevicePtr,
    positions: DevicePtr,
    v_ones_f16: DevicePtr,
    splitk_partials_m: DevicePtr,
    splitk_partials_s: DevicePtr,
    splitk_partials_o: DevicePtr,
}

struct OuterScratchPtrs {
    /// F16 [hidden] — residual stream (input of next layer, output of last).
    x_residual_f16: DevicePtr,
    /// F16 [hidden] — destination of the current layer (swapped with residual each step).
    x_next_f16: DevicePtr,
}

/// Persistent device scratch for the MoE branch (26B-A4B only).
/// Allocated when `cfg.moe.is_some()`. `zero_hidden_f16` is uploaded
/// once at session init and never written — fed as `residual=0` to
/// `MoeExperts::forward_decode`.
struct MoeScratchPtrs {
    router_input_f16: DevicePtr,
    cur_mlp_f16: DevicePtr,
    cur_moe_f16: DevicePtr,
    cur_combined_f16: DevicePtr,
    zero_hidden_f16: DevicePtr,
    /// Sized for `[top_k, intermediate]`-shaped expert buffers.
    x_q8_1: DevicePtr,
    router_logits: DevicePtr,
    expert_ids: DevicePtr,
    expert_weights: DevicePtr,
    gate_out_f32: DevicePtr,
    up_out_f32: DevicePtr,
    activated_f16: DevicePtr,
    activated_q8_1: DevicePtr,
    down_f32: DevicePtr,
    down_f16: DevicePtr,
}

pub struct Gemma4Session {
    pub cfg: Gemma4Config,
    pub layout: ModelLayout,
    pub weights: Gemma4DeviceWeights,
    /// Per-layer KV cache slot. `Some` iff the layer owns its KV
    /// (most layers). `None` for shared-KV tail layers; their
    /// `LayerSpec.kv_share_src` points the driver at another slot.
    pub kv_caches: Vec<Option<KvCache<F16Contig, HipDevice>>>,
    layer_scratch: LayerScratchPtrs,
    outer_scratch: OuterScratchPtrs,
    /// Every device alloc the session made — scratch + MoE + per-layer-
    /// embd table — tracked here for `dispose()`.
    raw_alloc: RawAllocTracker,
    /// `Some` when `cfg.moe.is_some()`; the MoE composer reads it via
    /// [`Gemma4Session::moe_scratch_view`].
    moe_scratch: Option<MoeScratchPtrs>,
    output_head: OutputHeadScratch,
    /// 1-slot position buffer (host).
    positions_host: Vec<i32>,
    /// E2B/E4B only: F32 `[n_layer, pe]` device buffer holding the
    /// `inp_per_layer_table` for the current token. Rebuilt host-side
    /// per token from the GGUF mmap and uploaded once per forward
    /// call. `None` when `cfg.per_layer_embed.is_none()`.
    pub(crate) inp_per_layer_table_buf: Option<DevicePtr>,
    /// E2B/E4B only: GGUF mmap reference so we can dequant the
    /// `per_layer_token_embd` row for the input token + read the
    /// `per_layer_model_proj` / `per_layer_proj_norm` tensors per
    /// forward call without copying the entire 1.4 GB table to host
    /// RAM. Cloned into [`Gemma4Session::new_with_gguf`] from the
    /// caller's mmap.
    pub(crate) gguf: Option<Arc<GgufFile>>,
    pub device_id: i32,
    pub max_tokens: usize,
    disposed: bool,
}

impl Gemma4Session {
    pub fn new(
        device: &HipDevice,
        weights: Gemma4DeviceWeights,
        cfg: Gemma4Config,
        layout: ModelLayout,
        max_tokens: usize,
    ) -> Result<Self> {
        device.bind()?;
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;

        // KV cache allocation per layer. Shared-KV tail layers
        // (`has_kv == false`) skip alloc and route to
        // `kv_share_src` at call time.
        let mut kv_caches = Vec::with_capacity(cfg.num_layers);
        for spec in &layout.layers {
            if spec.has_kv {
                let kv = KvCache::<F16Contig, HipDevice>::new(
                    device,
                    spec.n_kv_heads,
                    spec.head_dim,
                    max_tokens,
                )
                .map_err(|e| anyhow::anyhow!("kv alloc layer {}: {e}", spec.index))?;
                kv_caches.push(Some(kv));
            } else {
                if spec.kv_share_src.is_none() {
                    return Err(anyhow::anyhow!(
                        "layer {}: has_kv=false but kv_share_src unresolved; \
                         caller must run `ModelLayout::resolve_kv_sharing` first",
                        spec.index
                    ));
                }
                kv_caches.push(None);
            }
        }

        // Layer scratch sized for the widest per-call shape across all layers.
        let q_width_max = layout
            .layers
            .iter()
            .map(|s| s.n_heads * s.head_dim)
            .max()
            .unwrap_or(hidden);
        let kv_width_max = layout
            .layers
            .iter()
            .map(|s| s.n_kv_heads * s.head_dim)
            .max()
            .unwrap_or(hidden);
        let head_dim_max = layout
            .layers
            .iter()
            .map(|s| s.head_dim)
            .max()
            .unwrap_or(64);
        let ff_len = cfg.feed_forward_length;
        let mmvq_max = q_width_max.max(kv_width_max).max(hidden).max(ff_len);
        let x_q8_1_n = hidden.max(ff_len).div_ceil(32) * 32;
        let activated_q8_1_n = ff_len.div_ceil(32) * 32;

        let mut raw_alloc = RawAllocTracker::new();

        let n_heads_max = layout
            .layers
            .iter()
            .map(|s| s.n_heads)
            .max()
            .unwrap_or(1);
        let splitk_chunks = flambeau_blocks::MAX_SPLITK_CHUNKS;
        let layer_scratch = LayerScratchPtrs {
            x_q8_1: raw_alloc.alloc_q8_1(device, x_q8_1_n)?.0,
            mmvq_f32: raw_alloc.alloc_f32(device, mmvq_max)?.0,
            q_f16: raw_alloc.alloc_f16(device, q_width_max)?.0,
            k_f16: raw_alloc.alloc_f16(device, kv_width_max)?.0,
            v_f16: raw_alloc.alloc_f16(device, kv_width_max)?.0,
            attn_out_f16: raw_alloc.alloc_f16(device, q_width_max.max(hidden))?.0,
            post_attn_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
            attn_residual_f16: raw_alloc.alloc_f16(device, hidden)?.0,
            ffn_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
            gate_f32: raw_alloc.alloc_f32(device, ff_len)?.0,
            up_f32: raw_alloc.alloc_f32(device, ff_len)?.0,
            activated_f16: raw_alloc.alloc_f16(device, ff_len)?.0,
            activated_q8_1: raw_alloc.alloc_q8_1(device, activated_q8_1_n)?.0,
            down_f32: raw_alloc.alloc_f32(device, hidden)?.0,
            post_ffw_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
            positions: raw_alloc.alloc_i32(device, 1)?.0,
            v_ones_f16: raw_alloc.alloc_f16(device, head_dim_max)?.0,
            splitk_partials_m: raw_alloc.alloc_f32(device, n_heads_max * splitk_chunks)?.0,
            splitk_partials_s: raw_alloc.alloc_f32(device, n_heads_max * splitk_chunks)?.0,
            splitk_partials_o: raw_alloc
                .alloc_f32(device, n_heads_max * splitk_chunks * head_dim_max)?
                .0,
        };

        // Fill the v_ones buffer with 1.0 in F16.
        let ones: Vec<f16> = vec![f16::from_f32(1.0); head_dim_max];
        let bytes = ones.len() * 2;
        // SAFETY: ones lives until the bounded synchronize below; `v_ones_f16` has bytes capacity.
        unsafe {
            device.memcpy_async(
                device.default_stream(),
                flambeau_core::CopyDirection::HostToDevice,
                layer_scratch.v_ones_f16,
                DevicePtr(ones.as_ptr() as usize),
                bytes,
            )?;
        }
        device.default_stream().synchronize()?;
        drop(ones);

        let outer_scratch = OuterScratchPtrs {
            x_residual_f16: raw_alloc.alloc_f16(device, hidden)?.0,
            x_next_f16: raw_alloc.alloc_f16(device, hidden)?.0,
        };

        // MoE composer scratch (26B-A4B only). `zero_hidden_f16` is
        // uploaded once with zeros and fed read-only as the `residual=0`
        // input to `MoeExperts::forward_decode`.
        let moe_scratch = if let Some(moe_dims) = cfg.moe {
            let top_k = moe_dims.num_experts_per_tok;
            let n_experts = moe_dims.num_experts;
            let intermediate = moe_dims.moe_intermediate_size;

            let router_input_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
            let cur_mlp_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
            let cur_moe_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
            let cur_combined_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
            let zero_hidden_f16 = raw_alloc.alloc_f16(device, hidden)?.0;

            let x_q8_1 = raw_alloc.alloc_q8_1(device, hidden.div_ceil(32) * 32)?.0;
            let router_logits = raw_alloc.alloc_f32(device, n_experts)?.0;
            let expert_ids = raw_alloc.alloc_i32(device, top_k)?.0;
            let expert_weights = raw_alloc.alloc_f32(device, top_k)?.0;
            let gate_out_f32 = raw_alloc.alloc_f32(device, top_k * intermediate)?.0;
            let up_out_f32 = raw_alloc.alloc_f32(device, top_k * intermediate)?.0;
            let activated_f16 = raw_alloc.alloc_f16(device, top_k * intermediate)?.0;
            let activated_q8_1 = raw_alloc
                .alloc_q8_1(device, top_k * intermediate.div_ceil(32) * 32)?
                .0;
            let down_f32 = raw_alloc.alloc_f32(device, top_k * hidden)?.0;
            let down_f16 = raw_alloc.alloc_f16(device, top_k * hidden)?.0;

            // Upload zeros to zero_hidden_f16.
            let zeros: Vec<f16> = vec![f16::from_f32(0.0); hidden];
            // SAFETY: zeros lives until the bounded synchronize below;
            // zero_hidden_f16 owns hidden*2 bytes.
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    flambeau_core::CopyDirection::HostToDevice,
                    zero_hidden_f16,
                    DevicePtr(zeros.as_ptr() as usize),
                    hidden * 2,
                )?;
            }
            device.default_stream().synchronize()?;
            drop(zeros);

            Some(MoeScratchPtrs {
                router_input_f16,
                cur_mlp_f16,
                cur_moe_f16,
                cur_combined_f16,
                zero_hidden_f16,
                x_q8_1,
                router_logits,
                expert_ids,
                expert_weights,
                gate_out_f32,
                up_out_f32,
                activated_f16,
                activated_q8_1,
                down_f32,
                down_f16,
            })
        } else {
            None
        };

        let output_head = OutputHeadScratch {
            x_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
            x_q8_1: raw_alloc.alloc_q8_1(device, x_q8_1_n)?.0,
            logits_f32: raw_alloc.alloc_f32(device, vocab)?.0,
        };

        // Per-layer-embd table buffer: F32 [n_layer × pe]. Allocated
        // only when the variant has the side-channel; otherwise None.
        let inp_per_layer_table_buf = if let Some(ple) = cfg.per_layer_embed {
            Some(raw_alloc.alloc_f32(device, cfg.num_layers * ple.n_embd_per_layer)?.0)
        } else {
            None
        };

        Ok(Self {
            cfg,
            layout,
            weights,
            kv_caches,
            layer_scratch,
            outer_scratch,
            moe_scratch,
            output_head,
            positions_host: vec![0i32; 1],
            inp_per_layer_table_buf,
            gguf: None,
            device_id: device.id(),
            max_tokens,
            raw_alloc,
            disposed: false,
        })
    }

    /// Sister of [`Self::new`] that retains an [`Arc<GgufFile>`] so
    /// per-token `build_inp_per_layer_table` can dequant the input
    /// token's `per_layer_token_embd` row from the mmap without copying
    /// the whole table to host RAM. Required for E2B/E4B variants;
    /// other variants can use [`Self::new`].
    pub fn new_with_gguf(
        device: &HipDevice,
        weights: Gemma4DeviceWeights,
        cfg: Gemma4Config,
        layout: ModelLayout,
        max_tokens: usize,
        gguf: Arc<GgufFile>,
    ) -> Result<Self> {
        let mut s = Self::new(device, weights, cfg, layout, max_tokens)?;
        s.gguf = Some(gguf);
        Ok(s)
    }

    /// Build a per-call layer scratch view referencing the persistent
    /// device buffers and the 1-element host position slot.
    pub(crate) fn layer_scratch_view(&mut self) -> crate::scratch::LayerDecodeScratch<'_> {
        crate::scratch::LayerDecodeScratch {
            x_q8_1: self.layer_scratch.x_q8_1,
            mmvq_f32: self.layer_scratch.mmvq_f32,
            q_f16: self.layer_scratch.q_f16,
            k_f16: self.layer_scratch.k_f16,
            v_f16: self.layer_scratch.v_f16,
            attn_out_f16: self.layer_scratch.attn_out_f16,
            post_attn_norm_f16: self.layer_scratch.post_attn_norm_f16,
            attn_residual_f16: self.layer_scratch.attn_residual_f16,
            ffn_norm_f16: self.layer_scratch.ffn_norm_f16,
            gate_f32: self.layer_scratch.gate_f32,
            up_f32: self.layer_scratch.up_f32,
            activated_f16: self.layer_scratch.activated_f16,
            activated_q8_1: self.layer_scratch.activated_q8_1,
            down_f32: self.layer_scratch.down_f32,
            post_ffw_norm_f16: self.layer_scratch.post_ffw_norm_f16,
            positions: self.layer_scratch.positions,
            positions_host: &mut self.positions_host,
            v_ones_f16: self.layer_scratch.v_ones_f16,
            splitk_partials_m: self.layer_scratch.splitk_partials_m,
            splitk_partials_s: self.layer_scratch.splitk_partials_s,
            splitk_partials_o: self.layer_scratch.splitk_partials_o,
        }
    }

    pub(crate) fn outer_residual(&self) -> DevicePtr {
        self.outer_scratch.x_residual_f16
    }

    pub(crate) fn outer_next(&self) -> DevicePtr {
        self.outer_scratch.x_next_f16
    }

    pub(crate) fn swap_residual(&mut self) {
        std::mem::swap(
            &mut self.outer_scratch.x_residual_f16,
            &mut self.outer_scratch.x_next_f16,
        );
    }

    pub(crate) fn output_head_scratch(&mut self) -> &mut OutputHeadScratch {
        &mut self.output_head
    }

    /// MoE composer scratch view. Returns `None` on dense variants.
    /// Lifetime is `'self` — the underlying buffers live on the
    /// session and are freed by `dispose`.
    pub(crate) fn moe_scratch_view(&self) -> Option<Gemma4MoeScratch> {
        let m = self.moe_scratch.as_ref()?;
        Some(Gemma4MoeScratch {
            router_input_f16: m.router_input_f16,
            cur_mlp_f16: m.cur_mlp_f16,
            cur_moe_f16: m.cur_moe_f16,
            cur_combined_f16: m.cur_combined_f16,
            zero_hidden_f16: m.zero_hidden_f16,
            moe_scratch: MoeExpertsDecodeScratch {
                x_q8_1: m.x_q8_1,
                router_logits: m.router_logits,
                expert_ids: m.expert_ids,
                expert_weights: m.expert_weights,
                gate_out_f32: m.gate_out_f32,
                up_out_f32: m.up_out_f32,
                activated_f16: m.activated_f16,
                activated_q8_1: m.activated_q8_1,
                down_f32: m.down_f32,
                down_f16: m.down_f16,
            },
        })
    }

    /// Free every device allocation. Caller passes the device handle
    /// used at construction.
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;

        let kvs = std::mem::take(&mut self.kv_caches);
        for kv in kvs.into_iter().flatten() {
            // SAFETY: KvCache.dispose contract — see runtime/kv_cache.rs.
            kv.dispose(device)
                .map_err(|e| anyhow::anyhow!("kv dispose: {e}"))?;
        }
        self.raw_alloc
            .dispose(device)
            .map_err(|e| anyhow::anyhow!("raw_alloc dispose: {e}"))?;
        self.weights.dispose(device).context("weights dispose")?;
        Ok(())
    }
}

impl Drop for Gemma4Session {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                "Gemma4Session dropped without dispose(); device {} resources leaked",
                self.device_id
            );
        }
    }
}
