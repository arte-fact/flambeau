//! Per-layer descriptor table.
//!
//! `LayerSpec` collapses per-layer config (SWA vs full, has-own-KV,
//! head dims, rope base, MoE/dense FFN flavour) into a single struct
//! the forward composer matches on. Built once at load time from
//! [`crate::config::Gemma4Config`].
//!
//! For shared-KV tail layers (`has_kv == false`), the KV source layer
//! is **not** resolved at this stage — that pairing is a forward-time
//! concern handled by the KV-cache slot allocation (see S6). The
//! `kv_share_src` field is reserved here for that resolver to fill in.

use crate::config::{Gemma4Config, Gemma4Variant};

/// Which FFN flavour fires on this layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnKind {
    /// Single gate/up/down triple (E2B, E4B, 31B, plus the
    /// shared-MLP branch within every MoE layer of 26B-A4B).
    Dense,
    /// Routed experts. On 26B-A4B every layer is MoE — the shared
    /// MLP runs in parallel with the routed branch and the two sum.
    Moe,
}

/// Per-layer specification. Drives the forward composer's match.
#[derive(Debug, Clone, Copy)]
pub struct LayerSpec {
    pub index: usize,
    /// SWA mask radius (`config.sliding_window`) when `is_swa`, else 0.
    pub window: u32,
    pub is_swa: bool,
    /// `true` iff this layer owns its KV cache. `false` for the
    /// shared-KV tail on E4B.
    pub has_kv: bool,
    /// Layer index providing K/V when `has_kv == false`. Filled in by
    /// the cache allocator post-load; `None` at construction time.
    pub kv_share_src: Option<usize>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Rotated-dim count for RoPE (gemma4 uses head_dim_swa=128 with
    /// rope_dim_swa=128 on SWA, head_dim=256 with rope_dim=256 on full).
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub ffn_kind: FfnKind,
}

#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub variant: Gemma4Variant,
    pub layers: Vec<LayerSpec>,
    /// `true` iff the model carries a per-layer side-channel embedding
    /// (E2B + E4B). Drives whether the loader expects
    /// `per_layer_token_embd`, `per_layer_model_proj`,
    /// `per_layer_proj_norm` globally and `inp_gate` / `proj` /
    /// `post_norm` per layer.
    pub has_per_layer_embed: bool,
    /// Final-logit softcap value (`config.final_logit_softcap`). The
    /// forward path applies `tanh(logits / cap) * cap` when `cap > 0`.
    pub final_logit_softcap: f32,
}

impl ModelLayout {
    pub fn from_config(cfg: &Gemma4Config) -> Self {
        let ffn_kind_global = if cfg.variant.is_moe() {
            FfnKind::Moe
        } else {
            FfnKind::Dense
        };
        let layers = (0..cfg.num_layers)
            .map(|il| {
                let is_swa = cfg.is_swa(il);
                LayerSpec {
                    index: il,
                    window: if is_swa { cfg.sliding_window as u32 } else { 0 },
                    is_swa,
                    has_kv: cfg.has_kv(il),
                    kv_share_src: None,
                    n_heads: cfg.num_heads,
                    n_kv_heads: cfg.n_kv_heads(il),
                    head_dim: cfg.head_dim_for_layer(il),
                    rope_dim: cfg.rope_dim_for_layer(il),
                    rope_freq_base: cfg.rope_freq_base_for_layer(il),
                    ffn_kind: ffn_kind_global,
                }
            })
            .collect();
        Self {
            variant: cfg.variant,
            layers,
            has_per_layer_embed: cfg.per_layer_embed.is_some(),
            final_logit_softcap: cfg.final_logit_softcap,
        }
    }

    /// Resolve the KV-source layer for every shared-KV tail layer.
    /// Convention: a tail layer at index `i` reads from the most
    /// recent earlier layer of the **same SWA type** that owns its
    /// KV. This mirrors llama.cpp's `build_attn_inp_kv_iswa` — each
    /// layer attends through the input slot matched on `is_swa(il)`
    /// among the layers that allocated cache.
    ///
    /// Call after construction to fill in `kv_share_src` for the tail
    /// layers. Layers with `has_kv == true` are left untouched.
    /// Returns the number of tail layers resolved.
    pub fn resolve_kv_sharing(&mut self) -> usize {
        let mut last_full_with_kv: Option<usize> = None;
        let mut last_swa_with_kv: Option<usize> = None;
        let mut resolved = 0;
        let n = self.layers.len();
        for i in 0..n {
            let spec = self.layers[i];
            if spec.has_kv {
                if spec.is_swa {
                    last_swa_with_kv = Some(i);
                } else {
                    last_full_with_kv = Some(i);
                }
            } else {
                let src = if spec.is_swa {
                    last_swa_with_kv
                } else {
                    last_full_with_kv
                };
                self.layers[i].kv_share_src = src;
                if src.is_some() {
                    resolved += 1;
                }
            }
        }
        resolved
    }

    pub fn num_full_attn_layers(&self) -> usize {
        self.layers.iter().filter(|l| !l.is_swa).count()
    }

    pub fn num_swa_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.is_swa).count()
    }

    pub fn num_shared_kv_layers(&self) -> usize {
        self.layers.iter().filter(|l| !l.has_kv).count()
    }
}
