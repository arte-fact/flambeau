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
use flambeau_blocks::MoeExpertsDecodeScratch;

/// Persistent device scratch for one decode step. Allocated once at
/// session-init; the per-call [`crate::scratch::LayerDecodeScratch`]
/// view is reconstructed each forward call from these pointers.
struct LayerScratchPtrs {
    x_q8_1: DevicePtr,
    x_q8_1_bytes: usize,
    mmvq_f32: DevicePtr,
    mmvq_f32_bytes: usize,
    q_f16: DevicePtr,
    q_f16_bytes: usize,
    k_f16: DevicePtr,
    k_f16_bytes: usize,
    v_f16: DevicePtr,
    v_f16_bytes: usize,
    attn_out_f16: DevicePtr,
    attn_out_bytes: usize,
    post_attn_norm_f16: DevicePtr,
    post_attn_norm_bytes: usize,
    attn_residual_f16: DevicePtr,
    attn_residual_bytes: usize,
    ffn_norm_f16: DevicePtr,
    ffn_norm_bytes: usize,
    gate_f32: DevicePtr,
    gate_f32_bytes: usize,
    up_f32: DevicePtr,
    up_f32_bytes: usize,
    activated_f16: DevicePtr,
    activated_f16_bytes: usize,
    activated_q8_1: DevicePtr,
    activated_q8_1_bytes: usize,
    down_f32: DevicePtr,
    down_f32_bytes: usize,
    post_ffw_norm_f16: DevicePtr,
    post_ffw_norm_bytes: usize,
    positions: DevicePtr,
    positions_bytes: usize,
    v_ones_f16: DevicePtr,
    v_ones_f16_bytes: usize,
}

struct OuterScratchPtrs {
    /// F16 [hidden] — residual stream (input of next layer, output of last).
    x_residual_f16: DevicePtr,
    x_residual_bytes: usize,
    /// F16 [hidden] — destination of the current layer (swapped with residual each step).
    x_next_f16: DevicePtr,
    x_next_bytes: usize,
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
    /// Total byte count of every device alloc above — used for the
    /// dispose pass.
    total_bytes: usize,
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
    inp_per_layer_table_bytes: usize,
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
        // Q8_1 block = `d + s + 32 i8` = 36 bytes (matches kernels-hip layout).
        let q8_1_blocks = hidden.max(ff_len).div_ceil(32);
        let q8_1_bytes_per_block = 36;
        let x_q8_1_bytes = q8_1_blocks * q8_1_bytes_per_block;
        let activated_q8_1_bytes = ff_len.div_ceil(32) * q8_1_bytes_per_block;

        let alloc = |bytes: usize| -> Result<DevicePtr> {
            let p = device.alloc(bytes).map_err(|e| anyhow::anyhow!("alloc {bytes}: {e}"))?;
            Ok(p)
        };

        let layer_scratch = LayerScratchPtrs {
            x_q8_1: alloc(x_q8_1_bytes)?,
            x_q8_1_bytes,
            mmvq_f32: alloc(mmvq_max * 4)?,
            mmvq_f32_bytes: mmvq_max * 4,
            q_f16: alloc(q_width_max * 2)?,
            q_f16_bytes: q_width_max * 2,
            k_f16: alloc(kv_width_max * 2)?,
            k_f16_bytes: kv_width_max * 2,
            v_f16: alloc(kv_width_max * 2)?,
            v_f16_bytes: kv_width_max * 2,
            attn_out_f16: alloc(q_width_max.max(hidden) * 2)?,
            attn_out_bytes: q_width_max.max(hidden) * 2,
            post_attn_norm_f16: alloc(hidden * 2)?,
            post_attn_norm_bytes: hidden * 2,
            attn_residual_f16: alloc(hidden * 2)?,
            attn_residual_bytes: hidden * 2,
            ffn_norm_f16: alloc(hidden * 2)?,
            ffn_norm_bytes: hidden * 2,
            gate_f32: alloc(ff_len * 4)?,
            gate_f32_bytes: ff_len * 4,
            up_f32: alloc(ff_len * 4)?,
            up_f32_bytes: ff_len * 4,
            activated_f16: alloc(ff_len * 2)?,
            activated_f16_bytes: ff_len * 2,
            activated_q8_1: alloc(activated_q8_1_bytes)?,
            activated_q8_1_bytes,
            down_f32: alloc(hidden * 4)?,
            down_f32_bytes: hidden * 4,
            post_ffw_norm_f16: alloc(hidden * 2)?,
            post_ffw_norm_bytes: hidden * 2,
            positions: alloc(4)?,
            positions_bytes: 4,
            v_ones_f16: alloc(head_dim_max * 2)?,
            v_ones_f16_bytes: head_dim_max * 2,
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
            x_residual_f16: alloc(hidden * 2)?,
            x_residual_bytes: hidden * 2,
            x_next_f16: alloc(hidden * 2)?,
            x_next_bytes: hidden * 2,
        };

        // MoE composer scratch (26B-A4B only). `zero_hidden_f16` is
        // uploaded once with zeros and fed read-only as the `residual=0`
        // input to `MoeExperts::forward_decode`.
        let moe_scratch = if let Some(moe_dims) = cfg.moe {
            let top_k = moe_dims.num_experts_per_tok;
            let n_experts = moe_dims.num_experts;
            let intermediate = moe_dims.moe_intermediate_size;

            let router_input_f16 = alloc(hidden * 2)?;
            let cur_mlp_f16 = alloc(hidden * 2)?;
            let cur_moe_f16 = alloc(hidden * 2)?;
            let cur_combined_f16 = alloc(hidden * 2)?;
            let zero_hidden_f16 = alloc(hidden * 2)?;

            // x_q8_1: Q8_1 [hidden / 32] (36-byte blocks).
            let x_q8_1_bytes_moe = hidden.div_ceil(32) * 36;
            let x_q8_1 = alloc(x_q8_1_bytes_moe)?;

            let router_logits = alloc(n_experts * 4)?;
            let expert_ids = alloc(top_k * 4)?;
            let expert_weights = alloc(top_k * 4)?;

            // Indexed gate/up/down work over [top_k, intermediate] /
            // [top_k, hidden] slabs (block-2D launches treat each top_k
            // slot as its own effective token; the indexed-MMVQ kernel
            // reads expert_ids[k] to pick the slab).
            let gate_out_f32 = alloc(top_k * intermediate * 4)?;
            let up_out_f32 = alloc(top_k * intermediate * 4)?;
            let activated_f16 = alloc(top_k * intermediate * 2)?;
            let activated_q8_1 =
                alloc(top_k * intermediate.div_ceil(32) * 36)?;
            let down_f32 = alloc(top_k * hidden * 4)?;
            let down_f16 = alloc(top_k * hidden * 2)?;

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

            let total_bytes = hidden * 2 * 5
                + x_q8_1_bytes_moe
                + n_experts * 4
                + top_k * 4 * 2
                + top_k * intermediate * 4 * 2
                + top_k * intermediate * 2
                + top_k * intermediate.div_ceil(32) * 36
                + top_k * hidden * 4
                + top_k * hidden * 2;

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
                total_bytes,
            })
        } else {
            None
        };

        let output_head = OutputHeadScratch {
            x_norm_f16: alloc(hidden * 2)?,
            x_q8_1: alloc(x_q8_1_bytes)?,
            logits_f32: alloc(vocab * 4)?,
        };

        // Per-layer-embd table buffer: F32 [n_layer × pe]. Allocated
        // only when the variant has the side-channel; otherwise None.
        let (inp_per_layer_table_buf, inp_per_layer_table_bytes) =
            if let Some(ple) = cfg.per_layer_embed {
                let bytes = cfg.num_layers * ple.n_embd_per_layer * 4;
                (Some(alloc(bytes)?), bytes)
            } else {
                (None, 0)
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
            inp_per_layer_table_bytes,
            gguf: None,
            device_id: device.id(),
            max_tokens,
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
        std::mem::swap(
            &mut self.outer_scratch.x_residual_bytes,
            &mut self.outer_scratch.x_next_bytes,
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

        // KV caches.
        let kvs = std::mem::take(&mut self.kv_caches);
        for kv in kvs.into_iter().flatten() {
            // SAFETY: KvCache.dispose contract — see runtime/kv_cache.rs.
            kv.dispose(device)
                .map_err(|e| anyhow::anyhow!("kv dispose: {e}"))?;
        }

        // Layer scratch.
        let lf = &self.layer_scratch;
        let deallocs: &[(DevicePtr, usize)] = &[
            (lf.x_q8_1, lf.x_q8_1_bytes),
            (lf.mmvq_f32, lf.mmvq_f32_bytes),
            (lf.q_f16, lf.q_f16_bytes),
            (lf.k_f16, lf.k_f16_bytes),
            (lf.v_f16, lf.v_f16_bytes),
            (lf.attn_out_f16, lf.attn_out_bytes),
            (lf.post_attn_norm_f16, lf.post_attn_norm_bytes),
            (lf.attn_residual_f16, lf.attn_residual_bytes),
            (lf.ffn_norm_f16, lf.ffn_norm_bytes),
            (lf.gate_f32, lf.gate_f32_bytes),
            (lf.up_f32, lf.up_f32_bytes),
            (lf.activated_f16, lf.activated_f16_bytes),
            (lf.activated_q8_1, lf.activated_q8_1_bytes),
            (lf.down_f32, lf.down_f32_bytes),
            (lf.post_ffw_norm_f16, lf.post_ffw_norm_bytes),
            (lf.positions, lf.positions_bytes),
            (lf.v_ones_f16, lf.v_ones_f16_bytes),
            (self.outer_scratch.x_residual_f16, self.outer_scratch.x_residual_bytes),
            (self.outer_scratch.x_next_f16, self.outer_scratch.x_next_bytes),
            (self.output_head.x_norm_f16, self.cfg.hidden_size * 2),
            (self.output_head.x_q8_1, lf.x_q8_1_bytes),
            (self.output_head.logits_f32, self.cfg.vocab_size * 4),
        ];
        for &(ptr, bytes) in deallocs {
            // SAFETY: every ptr came from `device.alloc(bytes)` in `new`.
            unsafe {
                let _ = device.dealloc(ptr, bytes);
            }
        }

        if let Some(ptr) = self.inp_per_layer_table_buf.take() {
            // SAFETY: ptr came from `alloc(inp_per_layer_table_bytes)` in `new`.
            unsafe {
                let _ = device.dealloc(ptr, self.inp_per_layer_table_bytes);
            }
        }

        if let Some(m) = self.moe_scratch.take() {
            let moe_dims = self
                .cfg
                .moe
                .expect("moe_scratch alive but cfg.moe is None");
            let hidden = self.cfg.hidden_size;
            let top_k = moe_dims.num_experts_per_tok;
            let n_experts = moe_dims.num_experts;
            let intermediate = moe_dims.moe_intermediate_size;
            let x_q8_1_bytes_moe = hidden.div_ceil(32) * 36;
            let activated_q8_1_bytes = top_k * intermediate.div_ceil(32) * 36;
            let moe_deallocs: &[(DevicePtr, usize)] = &[
                (m.router_input_f16, hidden * 2),
                (m.cur_mlp_f16, hidden * 2),
                (m.cur_moe_f16, hidden * 2),
                (m.cur_combined_f16, hidden * 2),
                (m.zero_hidden_f16, hidden * 2),
                (m.x_q8_1, x_q8_1_bytes_moe),
                (m.router_logits, n_experts * 4),
                (m.expert_ids, top_k * 4),
                (m.expert_weights, top_k * 4),
                (m.gate_out_f32, top_k * intermediate * 4),
                (m.up_out_f32, top_k * intermediate * 4),
                (m.activated_f16, top_k * intermediate * 2),
                (m.activated_q8_1, activated_q8_1_bytes),
                (m.down_f32, top_k * hidden * 4),
                (m.down_f16, top_k * hidden * 2),
            ];
            for &(ptr, bytes) in moe_deallocs {
                // SAFETY: every ptr came from `device.alloc(bytes)` in `new`.
                unsafe {
                    let _ = device.dealloc(ptr, bytes);
                }
            }
            let _ = m.total_bytes;
        }

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
