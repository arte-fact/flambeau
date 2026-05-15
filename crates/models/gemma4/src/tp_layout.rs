//! Gemma 4 tensor-name → [`WeightLayout`] table.
//!
//! Maps every weight name produced by `names::*` for Gemma 4
//! (E4B + 31B dense, 26B-A4B MoE) to its TP shard layout. Mirrors
//! `qwen3-moe::tp_layout::Qwen35DenseTpLayout` for the gemma4 family.
//!
//! ## Sharding scheme (Megatron-LM, decode-time TP)
//!
//! | Tensor | Shape | Layout | AR? |
//! |----------------------------|--------------------------------|-------------------------|-----|
//! | `attn_q.weight` | `[nQ·D, hidden]` | `ColParallel{dim=0}` | no |
//! | `attn_k.weight` | `[nKV·D, hidden]` | `ColParallel{dim=0}` | no |
//! | `attn_v.weight` | `[nKV·D, hidden]` | `ColParallel{dim=0}` | no |
//! | `attn_output.weight` | `[hidden, nQ·D]` | `RowParallel{dim=1}` | yes |
//! | `attn_q_norm.weight` | `[head_dim]` | `Replicated` | no |
//! | `attn_k_norm.weight` | `[head_dim]` | `Replicated` | no |
//! | `attn_norm.weight` | `[hidden]` | `Replicated` | no |
//! | `post_attention_norm.weight` | `[hidden]` | `Replicated` | no |
//! | `ffn_norm.weight` | `[hidden]` | `Replicated` | no |
//! | `ffn_gate.weight` | `[ff_len, hidden]` | `ColParallel{dim=0}` | no |
//! | `ffn_up.weight` | `[ff_len, hidden]` | `ColParallel{dim=0}` | no |
//! | `ffn_down.weight` | `[hidden, ff_len]` | `RowParallel{dim=1}` | yes |
//! | `post_ffw_norm.weight` | `[hidden]` | `Replicated` | no |
//! | `token_embd.weight` | `[vocab, hidden]` | `Replicated` | n/a |
//! | `output_norm.weight` | `[hidden]` | `Replicated` | no |
//! | `output.weight` (when present) | `[vocab, hidden]` | `Replicated` | n/a |
//!
//! ## MoE-specific (26B-A4B)
//!
//! Gemma 4 26B-A4B has a hybrid FFN: each MoE layer runs the dense
//! shared MLP (above ffn_gate/ffn_up/ffn_down) **in parallel** with a
//! routed-experts branch and sums the two post-norm. The routed-experts
//! tensors:
//!
//! | Tensor | Shape | Layout | AR? |
//! |----------------------------|--------------------------------|-------------------------|-----|
//! | `ffn_gate_inp.weight` | `[n_experts, hidden]` | `Replicated` | no |
//! | `ffn_gate_inp.scale` | `[hidden]` | `Replicated` | no |
//! | `ffn_gate_up_exps.weight` | `[n_experts, 2·n_ff_exp, hidden]` | **split + ColParallel{dim=1}** | no |
//! | `ffn_down_exps.weight` | `[n_experts, hidden, n_ff_exp]` | `RowParallel{dim=2}` | yes |
//! | `pre_ffw_norm_2.weight` | `[hidden]` | `Replicated` | no |
//! | `post_ffw_norm_1.weight` | `[hidden]` | `Replicated` | no |
//! | `post_ffw_norm_2.weight` | `[hidden]` | `Replicated` | no |
//!
//! ### Fused gate+up handling
//!
//! The 26B-A4B GGUF stores `ffn_gate_up_exps` as a single fused tensor
//! along axis 1 (`[n_experts, 2·n_ff_exp, hidden]`); per
//! `weights_hip.rs::split_fused_gate_up`, the upload pipeline splits
//! this host-side into separate per-expert gate + up slabs. For TP, the
//! split is performed *before* the ColParallel{dim=1} slice so each
//! rank receives `[n_experts, n_ff_exp_local, hidden]` for both gate
//! and up. The per-rank fused tensor sits at `[n_experts,
//! 2·n_ff_exp_local, hidden]`; the policy here reports the *logical*
//! split tensors (`ffn_gate_exps` + `ffn_up_exps`) for downstream
//! uploaders that consume the layout.
//!
//! ## Two-AR composition for MoE layers
//!
//! Each MoE layer produces TWO row-parallel partials (shared MLP via
//! `ffn_down`, MoE branch via `ffn_down_exps`) with DIFFERENT
//! per-branch RMSNorm weights (`post_ffw_norm_1` for shared MLP,
//! `post_ffw_norm_2` for MoE). Per-branch norms require full-hidden
//! values, so each partial needs its own AllReduce before its
//! post-norm. Net: 1 AR for attention + 2 ARs for FFN = 3 AR per MoE
//! layer (vs 2 AR for dense layers). [`LayerComposerTp`] in
//! `flambeau_blocks` only models the dense 6-phase shape; the MoE path
//! has its own composer entry point.
//!
//! ## Per-layer-embd (E4B)
//!
//! The E4B per-layer-embd side-channel (`per_layer_token_embd.weight`,
//! `[vocab, n_layers · pe_dim]`) is **Replicated** — small enough that
//! per-rank duplication is cheap, and the per-rank application is a
//! purely local elementwise add.
//!
//! ## Divisibility preconditions
//!
//! - `num_heads % world == 0` (cleanly partitions head ownership).
//! - `num_kv_heads % world == 0` per layer (or the per-layer KV is
//!   replicated; gemma4 doesn't currently fall back).
//! - `feed_forward_length % world == 0` (dense FFN intermediate).
//! - `moe.moe_intermediate_size % world == 0` (MoE expert intermediate).
//! - Quantised tensors must keep the per-rank slice on a 32-elem ggml
//!   block boundary; for ColParallel that's automatic (the inner
//!   `hidden` axis isn't sharded), and for RowParallel{dim=1 or 2} the
//!   sharded axis is the inner dim — requires `(sliced / world) %
//!   block_size == 0`.

#![cfg(feature = "hip")]

use flambeau_runtime::WeightLayout;

use crate::config::Gemma4Config;

/// Per-tensor TP layout selector for the gemma4 family.
/// Construct via [`Gemma4TpLayout::new`] with the model config + mesh
/// size; the constructor validates divisibility and returns a reusable
/// selector. `for_tensor(name)` returns the layout for any tensor name
/// produced by `names::CommonNames` / `names::AttnNames` /
/// `names::DenseFfnNames` / `names::MoeFfnNames`.
#[derive(Debug, Clone, Copy)]
pub struct Gemma4TpLayout {
    world: u32,
    /// `true` when the model carries MoE FFN layers (26B-A4B).
    /// Causes `for_tensor` to recognise the MoE-specific tensor names.
    has_moe: bool,
}

impl Gemma4TpLayout {
    /// Build the selector for `cfg` on a `world`-rank TP mesh.
    /// # Errors
    /// [`TpLayoutError::WorldZero`] — mesh size of 0 is meaningless.
    /// [`TpLayoutError::IndivisibleNumHeads`] — `num_heads % world != 0`.
    /// [`TpLayoutError::IndivisibleNumKvHeads`] — any layer's
    ///   `n_kv_heads % world != 0`.
    /// [`TpLayoutError::IndivisibleIntermediate`] —
    ///   `feed_forward_length % world != 0`.
    /// [`TpLayoutError::IndivisibleMoeIntermediate`] —
    ///   `moe.moe_intermediate_size % world != 0` (only checked if
    ///   `cfg.moe.is_some()`).
    pub fn new(cfg: &Gemma4Config, world: u32) -> Result<Self, TpLayoutError> {
        if world == 0 {
            return Err(TpLayoutError::WorldZero);
        }
        if (cfg.num_heads as u32) % world != 0 {
            return Err(TpLayoutError::IndivisibleNumHeads {
                num_heads: cfg.num_heads,
                world,
            });
        }
        for (il, &kv) in cfg.num_kv_heads.iter().enumerate() {
            if (kv as u32) % world != 0 {
                return Err(TpLayoutError::IndivisibleNumKvHeads {
                    layer: il,
                    num_kv_heads: kv,
                    world,
                });
            }
        }
        if (cfg.feed_forward_length as u32) % world != 0 {
            return Err(TpLayoutError::IndivisibleIntermediate {
                intermediate: cfg.feed_forward_length,
                world,
            });
        }
        if let Some(moe) = &cfg.moe {
            if (moe.moe_intermediate_size as u32) % world != 0 {
                return Err(TpLayoutError::IndivisibleMoeIntermediate {
                    moe_intermediate: moe.moe_intermediate_size,
                    world,
                });
            }
        }
        Ok(Self {
            world,
            has_moe: cfg.moe.is_some(),
        })
    }

    /// Mesh size this layout was built for.
    pub fn world(&self) -> u32 {
        self.world
    }

    /// `true` when the model carries MoE FFN layers.
    pub fn has_moe(&self) -> bool {
        self.has_moe
    }

    /// Cheap pre-check used in tests / cert generation.
    pub fn validate(cfg: &Gemma4Config, world: u32) -> Result<(), TpLayoutError> {
        Self::new(cfg, world).map(|_| ())
    }

    /// Layout for a specific weight tensor name. Recognises every
    /// gemma4 tensor produced by `names::*`. Unknown names return
    /// `None`; callers should treat that as an error.
    pub fn for_tensor(&self, tensor_name: &str) -> Option<WeightLayout> {
        // Globals (no `blk.<L>.` prefix).
        match tensor_name {
            "token_embd.weight" | "output_norm.weight" | "output.weight" => {
                return Some(WeightLayout::Replicated);
            }
            "per_layer_token_embd.weight" => {
                // E4B side-channel; per-layer slices are small.
                return Some(WeightLayout::Replicated);
            }
            _ => {}
        }

        let suffix = strip_blk_prefix(tensor_name)?;
        let layout = match suffix {
            // Attention.
            "attn_norm.weight" => WeightLayout::Replicated,
            "post_attention_norm.weight" => WeightLayout::Replicated,
            "attn_q.weight" | "attn_k.weight" | "attn_v.weight" => {
                WeightLayout::col_parallel(self.world, 0)
            }
            "attn_q_norm.weight" | "attn_k_norm.weight" => WeightLayout::Replicated,
            "attn_output.weight" => WeightLayout::row_parallel(self.world, 1),

            // Dense FFN (also the shared-MLP branch on MoE layers).
            "ffn_norm.weight" => WeightLayout::Replicated,
            "post_ffw_norm.weight" => WeightLayout::Replicated,
            "ffn_gate.weight" | "ffn_up.weight" => WeightLayout::col_parallel(self.world, 0),
            "ffn_down.weight" => WeightLayout::row_parallel(self.world, 1),

            // MoE-specific (26B-A4B). Only recognised when has_moe.
            "ffn_gate_inp.weight" | "ffn_gate_inp.scale" if self.has_moe => {
                WeightLayout::Replicated
            }
            // The fused gate+up tensor is split host-side; the per-
            // expert slice is then ColParallel along the intermediate
            // axis. Reporting ColParallel{dim=1} here documents the
            // POST-SPLIT layout. Upload code must split then slice.
            "ffn_gate_up_exps.weight" if self.has_moe => WeightLayout::col_parallel(self.world, 1),
            "ffn_gate_exps.weight" | "ffn_up_exps.weight" if self.has_moe => {
                WeightLayout::col_parallel(self.world, 1)
            }
            "ffn_down_exps.weight" if self.has_moe => WeightLayout::row_parallel(self.world, 2),
            "ffn_down_exps.scale" if self.has_moe => WeightLayout::Replicated,
            "pre_ffw_norm_2.weight" | "post_ffw_norm_1.weight" | "post_ffw_norm_2.weight"
                if self.has_moe =>
            {
                WeightLayout::Replicated
            }

            _ => return None,
        };
        Some(layout)
    }
}

/// Errors produced by [`Gemma4TpLayout::new`].
#[derive(Debug, thiserror::Error)]
pub enum TpLayoutError {
    #[error("TP mesh size must be >= 1 (got 0)")]
    WorldZero,
    #[error("num_heads={num_heads} not divisible by world={world}")]
    IndivisibleNumHeads { num_heads: usize, world: u32 },
    #[error("layer {layer}: num_kv_heads={num_kv_heads} not divisible by world={world}")]
    IndivisibleNumKvHeads {
        layer: usize,
        num_kv_heads: usize,
        world: u32,
    },
    #[error("feed_forward_length={intermediate} not divisible by world={world}")]
    IndivisibleIntermediate { intermediate: usize, world: u32 },
    #[error("moe_intermediate_size={moe_intermediate} not divisible by world={world}")]
    IndivisibleMoeIntermediate {
        moe_intermediate: usize,
        world: u32,
    },
}

fn strip_blk_prefix(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let layer_part = &rest[..dot];
    if layer_part.is_empty() || !layer_part.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&rest[dot + 1..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Gemma4Variant;
    use flambeau_runtime::WeightLayout;

    fn cfg_dense() -> Gemma4Config {
        Gemma4Config {
            arch: "gemma4".to_string(),
            variant: Gemma4Variant::Dense31B,
            hidden_size: 5376,
            vocab_size: 256000,
            num_layers: 2,
            num_heads: 32,
            num_kv_heads: vec![16, 16],
            head_dim: 256,
            context_length: 8192,
            rms_norm_eps: 1e-6,
            feed_forward_length: 21504,
            rope_freq_base: 10000.0,
            rope_dim: 256,
            swa: crate::swa_policy::SwaAlternationPolicy {
                swa_layers: vec![false, false],
                sliding_window: 1024,
                head_dim_swa: 256,
                rope_dim_swa: 256,
                rope_freq_base_swa: 10000.0,
            },
            shared_kv_layers: 0,
            moe: None,
            per_layer_embed: None,
            final_logit_softcap: 30.0,
            tied_lm_head: true,
        }
    }

    #[test]
    fn dense_for_tensor() {
        let cfg = cfg_dense();
        let l = Gemma4TpLayout::new(&cfg, 2).unwrap();
        assert!(matches!(
            l.for_tensor("token_embd.weight"),
            Some(WeightLayout::Replicated)
        ));
        assert!(matches!(
            l.for_tensor("blk.0.attn_q.weight"),
            Some(WeightLayout::ColParallel { world: 2, dim: 0 })
        ));
        assert!(matches!(
            l.for_tensor("blk.0.attn_output.weight"),
            Some(WeightLayout::RowParallel { world: 2, dim: 1 })
        ));
        assert!(matches!(
            l.for_tensor("blk.0.ffn_down.weight"),
            Some(WeightLayout::RowParallel { world: 2, dim: 1 })
        ));
        assert!(matches!(
            l.for_tensor("blk.0.post_attention_norm.weight"),
            Some(WeightLayout::Replicated)
        ));
        // MoE tensors not recognised when has_moe == false.
        assert!(l.for_tensor("blk.0.ffn_gate_up_exps.weight").is_none());
    }

    #[test]
    fn divisibility_errors() {
        let mut cfg = cfg_dense();
        cfg.num_heads = 33;
        assert!(matches!(
            Gemma4TpLayout::new(&cfg, 2),
            Err(TpLayoutError::IndivisibleNumHeads { .. })
        ));
        let mut cfg = cfg_dense();
        cfg.feed_forward_length = 21505;
        assert!(matches!(
            Gemma4TpLayout::new(&cfg, 2),
            Err(TpLayoutError::IndivisibleIntermediate { .. })
        ));
    }

    #[test]
    fn world_zero() {
        let cfg = cfg_dense();
        assert!(matches!(
            Gemma4TpLayout::new(&cfg, 0),
            Err(TpLayoutError::WorldZero)
        ));
    }

    #[test]
    fn moe_for_tensor() {
        use crate::config::MoeDims;
        let mut cfg = cfg_dense();
        cfg.variant = Gemma4Variant::Moe26BA4B;
        cfg.moe = Some(MoeDims {
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 1024,
        });
        let l = Gemma4TpLayout::new(&cfg, 2).unwrap();
        assert!(l.has_moe());
        assert!(matches!(
            l.for_tensor("blk.0.ffn_gate_inp.weight"),
            Some(WeightLayout::Replicated)
        ));
        assert!(matches!(
            l.for_tensor("blk.0.ffn_gate_up_exps.weight"),
            Some(WeightLayout::ColParallel { world: 2, dim: 1 })
        ));
        assert!(matches!(
            l.for_tensor("blk.0.ffn_down_exps.weight"),
            Some(WeightLayout::RowParallel { world: 2, dim: 2 })
        ));
        assert!(matches!(
            l.for_tensor("blk.0.post_ffw_norm_2.weight"),
            Some(WeightLayout::Replicated)
        ));
        // Shared-MLP tensors still recognised.
        assert!(matches!(
            l.for_tensor("blk.0.ffn_down.weight"),
            Some(WeightLayout::RowParallel { world: 2, dim: 1 })
        ));
    }
}
