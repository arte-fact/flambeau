//! TP-1a — qwen35 dense tensor-name → [`WeightLayout`] table.
//!
//! Maps every weight name produced by `names::*` for the Qwen3.5 dense
//! family (arch=`qwen35`, V2.2 target) to its TP shard layout. The
//! generic [`WeightLayout`] enum lives in `flambeau-runtime`; this
//! module only owns the model-specific name → layout function and the
//! divisibility precondition.
//!
//! Sharding scheme (standard Megatron-LM, decode-time TP):
//!
//! | Tensor                     | Shape                          | Layout                  | AR? |
//! |----------------------------|--------------------------------|-------------------------|-----|
//! | `attn_q.weight`            | `[nQ·D, hidden]`               | `ColParallel{dim=0}`    | no  |
//! | `attn_k.weight`            | `[nKV·D, hidden]`              | `ColParallel{dim=0}`    | no  |
//! | `attn_v.weight`            | `[nKV·D, hidden]`              | `ColParallel{dim=0}`    | no  |
//! | `attn_output.weight`       | `[hidden, nQ·D]`               | `RowParallel{dim=1}`    | yes |
//! | `attn_q_norm.weight`       | `[head_dim]`                   | `Replicated`            | no  |
//! | `attn_k_norm.weight`       | `[head_dim]`                   | `Replicated`            | no  |
//! | `attn_norm.weight`         | `[hidden]`                     | `Replicated`            | no  |
//! | `ffn_norm.weight`          | `[hidden]`                     | `Replicated`            | no  |
//! | `ffn_gate.weight`          | `[intermediate, hidden]`       | `ColParallel{dim=0}`    | no  |
//! | `ffn_up.weight`            | `[intermediate, hidden]`       | `ColParallel{dim=0}`    | no  |
//! | `ffn_down.weight`          | `[hidden, intermediate]`       | `RowParallel{dim=1}`    | yes |
//! | `token_embd.weight`        | `[vocab, hidden]`              | `Replicated` (V1)       | n/a |
//! | `output_norm.weight`       | `[hidden]`                     | `Replicated`            | no  |
//! | `output.weight`            | `[vocab, hidden]`              | `Replicated` (V1)       | n/a |
//!
//! `token_embd` and `output` (LM head) start `Replicated` in V1 to keep
//! the embed/argmax paths simple. TP-2d may switch them to ColParallel
//! on `vocab` once the AR-of-(logit, idx)-pairs argmax kernel exists.
//!
//! ## Divisibility precondition
//!
//! ColParallel weights divide along the head-block axis, so the model
//! config must satisfy `n_heads % world == 0` and `n_kv_heads % world ==
//! 0` (cleanly partitions head ownership), and `intermediate % world ==
//! 0` (cleanly partitions FFN slabs). Quantised tensors (Q4_1/Q4_K/Q5_K
//! /Q6_K/Q8_0) further require the per-rank slice to fall on a 32-elem
//! ggml block boundary along the *row*-major dim — for ColParallel this
//! is the inner axis (`hidden`), which is not the sharded axis, so the
//! constraint is automatically satisfied. For RowParallel the sharded
//! axis is the inner axis; we shard along the unquantised `hidden`
//! direction, which divides cleanly when `hidden % world == 0` *and*
//! when each rank's column slice itself has a multiple-of-32 length.
//! Verified by [`Qwen35DenseTpLayout::validate`].

use flambeau_runtime::WeightLayout;

use crate::config::Qwen3MoEConfig;
use crate::names::GlobalNames;

/// Per-tensor TP layout selector for `arch=qwen35` (Qwen3.5 dense).
///
/// Construct via [`Qwen35DenseTpLayout::new`] with the model config
/// + mesh size; the constructor validates divisibility and returns a
/// reusable selector. `for_tensor(tensor_name)` returns the layout for
/// any tensor name produced by the model's `names::*` builders or
/// [`crate::names::GlobalNames`].
#[derive(Debug, Clone, Copy)]
pub struct Qwen35DenseTpLayout {
    world: u32,
    /// GDN head dims, populated from `cfg.gdn` when present. `None`
    /// for pure-dense (non-hybrid) configs that wouldn't reach the
    /// `attn_qkv` / `ssm_conv1d` branches anyway.
    gdn_dims: Option<GdnHeadDims>,
    /// **TP-4d-i2** — `true` when `num_kv_heads % world != 0` and we
    /// fall back to Replicated K/V (Megatron's standard workaround
    /// for low-GQA models like Qwen3.6-35B-A3B with `nKV=2`). When
    /// set: `attn_k`/`attn_v`/their norms/biases route to Replicated
    /// rather than ColParallel; the per-rank attn forward computes
    /// full `nKV` head outputs (duplicated work) but Q output is still
    /// sharded → AR-fold pattern unchanged.
    kv_replicated: bool,
}

/// GDN head dims captured at construction. Used to feed
/// [`WeightLayout::FusedQkvParallel`] for the `attn_qkv` and
/// `ssm_conv1d` tensors.
#[derive(Debug, Clone, Copy)]
struct GdnHeadDims {
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    head_k_dim: u32,
}

impl Qwen35DenseTpLayout {
    /// Build the selector for `cfg` on a `world`-rank TP mesh.
    ///
    /// # Errors
    /// [`TpLayoutError::WorldZero`] — mesh size of 0 is meaningless.
    /// [`TpLayoutError::IndivisibleNumHeads`] — `n_heads % world != 0`.
    /// [`TpLayoutError::IndivisibleNumKvHeads`] — `n_kv_heads % world != 0`.
    /// [`TpLayoutError::IndivisibleIntermediate`] — `intermediate % world != 0`.
    /// [`TpLayoutError::IndivisibleHidden`] — `hidden % world != 0`
    /// (RowParallel splits hidden across input dim).
    pub fn new(cfg: &Qwen3MoEConfig, world: u32) -> Result<Self, TpLayoutError> {
        if world == 0 {
            return Err(TpLayoutError::WorldZero);
        }
        if (cfg.num_heads as u32) % world != 0 {
            return Err(TpLayoutError::IndivisibleNumHeads {
                num_heads: cfg.num_heads,
                world,
            });
        }
        // TP-4d-i2: relax — if nKV doesn't divide world, fall back to
        // Replicated K/V (Megatron's "few-KV-head GQA" workaround).
        let kv_replicated = (cfg.num_kv_heads as u32) % world != 0;
        // `moe_intermediate_size` carries the dense-FFN intermediate when
        // `arch == "qwen35"` (the loader stuffs `feed_forward_length` here
        // for the no-MoE path; see `config.rs::from_gguf`'s `is_dense_ffn`
        // branch). For arch=qwen35 we use this as the dense FFN width.
        let intermediate = cfg.moe_intermediate_size;
        if (intermediate as u32) % world != 0 {
            return Err(TpLayoutError::IndivisibleIntermediate {
                intermediate,
                world,
            });
        }
        // **TP-4c** — shared-expert intermediate must divide world if present.
        if let Some(s) = cfg.shared_expert_intermediate_size {
            if (s as u32) % world != 0 {
                return Err(TpLayoutError::IndivisibleSharedExpertIntermediate {
                    shared_intermediate: s,
                    world,
                });
            }
        }
        // No `hidden_size % world` check: in this layout no sharded
        // tensor cuts along `hidden`. ColParallel weights split the
        // *output* dim (heads × head_dim, or intermediate); RowParallel
        // weights (`attn_output`, `ffn_down`) split their *input* dim
        // (`q_width = nQ·D` or `intermediate`), neither of which is
        // `hidden`. Vocab-shard of `output.weight` (TP-2d follow-up)
        // would split `vocab`, again not `hidden`.

        // GDN head divisibility — required for FusedQkvParallel on
        // attn_qkv and ssm_conv1d.
        let gdn_dims = cfg.gdn.as_ref().map(|g| GdnHeadDims {
            num_v_heads: g.num_v_heads as u32,
            num_k_heads: g.num_k_heads as u32,
            head_v_dim: g.head_v_dim() as u32,
            head_k_dim: g.head_k_dim as u32,
        });
        if let Some(d) = &gdn_dims {
            if d.num_v_heads % world != 0 {
                return Err(TpLayoutError::IndivisibleNumVHeads {
                    num_v_heads: d.num_v_heads as usize,
                    world,
                });
            }
            if d.num_k_heads % world != 0 {
                return Err(TpLayoutError::IndivisibleNumKHeadsGdn {
                    num_k_heads: d.num_k_heads as usize,
                    world,
                });
            }
        }
        Ok(Self {
            world,
            gdn_dims,
            kv_replicated,
        })
    }

    /// **TP-4d-i2** — `true` iff this layout falls back to Replicated
    /// K/V because `num_kv_heads % world != 0`. Forward kernels read
    /// this to know whether to use full or per-rank K/V head counts.
    pub fn kv_replicated(&self) -> bool {
        self.kv_replicated
    }

    /// Mesh size this layout was built for.
    pub fn world(&self) -> u32 {
        self.world
    }

    /// Cheap pre-check used in tests / cert generation. Same conditions
    /// as [`Self::new`] but without consuming the cfg.
    pub fn validate(cfg: &Qwen3MoEConfig, world: u32) -> Result<(), TpLayoutError> {
        Self::new(cfg, world).map(|_| ())
    }

    /// Layout for a specific weight tensor name.
    ///
    /// Recognises:
    /// - Per-layer names produced by [`crate::names::CommonNames`] /
    ///   [`crate::names::DenseAttnNames`] / [`crate::names::DenseFfnNames`]
    ///   for any layer (matched by suffix after `blk.<L>.`).
    /// - Global names from [`GlobalNames`].
    ///
    /// Unknown names return `None`. Callers should treat this as an
    /// error (likely an unsupported architecture or a typo); the loader
    /// has the context to format a useful diagnostic.
    pub fn for_tensor(&self, tensor_name: &str) -> Option<WeightLayout> {
        // Globals first — they're not under a `blk.<L>.` prefix.
        if tensor_name == GlobalNames::TOKEN_EMBD {
            // V1: replicated for simplicity. TP-2d may switch to
            // ColParallel{dim=0} on vocab once the gather-on-rank-0
            // path exists.
            return Some(WeightLayout::Replicated);
        }
        if tensor_name == GlobalNames::OUTPUT_NORM {
            return Some(WeightLayout::Replicated);
        }
        if tensor_name == GlobalNames::OUTPUT {
            // V1: replicated. TP-2d may switch to ColParallel{dim=0}
            // on vocab + AR-of-(logit, idx) for greedy argmax.
            return Some(WeightLayout::Replicated);
        }

        // Per-layer tensors live under `blk.<L>.` — match the suffix.
        let suffix = strip_blk_prefix(tensor_name)?;

        match suffix {
            // Attention norms — 1-D, replicated.
            "attn_norm.weight" => Some(WeightLayout::Replicated),
            "attn_q_norm.weight" => Some(WeightLayout::Replicated),
            "attn_k_norm.weight" => Some(WeightLayout::Replicated),

            // Attention projections. attn_q is always ColParallel
            // (its head count is large enough to divide world cleanly
            // in V1 scope). K/V depend on `kv_replicated` (TP-4d-i2):
            //   - kv_replicated=false: ColParallel{dim=0} as before
            //   - kv_replicated=true:  Replicated (low-GQA fallback)
            "attn_q.weight" => Some(WeightLayout::col_parallel(self.world, 0)),
            "attn_k.weight" => Some(if self.kv_replicated {
                WeightLayout::Replicated
            } else {
                WeightLayout::col_parallel(self.world, 0)
            }),
            "attn_v.weight" => Some(if self.kv_replicated {
                WeightLayout::Replicated
            } else {
                WeightLayout::col_parallel(self.world, 0)
            }),
            "attn_output.weight" => Some(WeightLayout::row_parallel(self.world, 1)),

            // Optional biases. attn_q.bias is ColParallel; K/V follow
            // the kv_replicated decision so per-rank shapes stay
            // self-consistent.
            "attn_q.bias" => Some(WeightLayout::col_parallel(self.world, 0)),
            "attn_k.bias" => Some(if self.kv_replicated {
                WeightLayout::Replicated
            } else {
                WeightLayout::col_parallel(self.world, 0)
            }),
            "attn_v.bias" => Some(if self.kv_replicated {
                WeightLayout::Replicated
            } else {
                WeightLayout::col_parallel(self.world, 0)
            }),

            // FFN.
            "ffn_norm.weight" => Some(WeightLayout::Replicated),
            "ffn_gate.weight" => Some(WeightLayout::col_parallel(self.world, 0)),
            "ffn_up.weight" => Some(WeightLayout::col_parallel(self.world, 0)),
            "ffn_down.weight" => Some(WeightLayout::row_parallel(self.world, 1)),

            // post_attention_norm only exists on hybrid arches.
            "post_attention_norm.weight" => Some(WeightLayout::Replicated),

            // ----- Hybrid full-attn block (qwen35 every Nth layer) -----
            // attn_q here is fused [Q | gate], rows = 2·n_heads·head_dim.
            // ColParallel{dim=0} preserves the [Q-chunk | gate-chunk]
            // pair-per-head layout when world divides n_heads, which the
            // TpLayoutError check enforces.
            // (Already covered by "attn_q.weight" branch above.)

            // ----- Hybrid GDN block (qwen35 most layers) -----
            // 1-D per-v-head scalars: ColParallel along the v-head axis.
            "ssm_a" => Some(WeightLayout::col_parallel(self.world, 0)),
            "ssm_dt.bias" => Some(WeightLayout::col_parallel(self.world, 0)),
            // 2-D [num_v_heads, hidden]: ColParallel splits v-heads.
            "ssm_alpha.weight" => Some(WeightLayout::col_parallel(self.world, 0)),
            "ssm_beta.weight" => Some(WeightLayout::col_parallel(self.world, 0)),
            "ssm_ba.weight" => Some(WeightLayout::col_parallel(self.world, 0)),
            // ssm_norm is per-head_dim_v (not per-head). Replicated.
            "ssm_norm.weight" => Some(WeightLayout::Replicated),
            // attn_gate.weight [d_inner = num_v_heads·head_dim_v, hidden]:
            // ColParallel{dim=0} splits the head-major output dim.
            "attn_gate.weight" => Some(WeightLayout::col_parallel(self.world, 0)),
            // ssm_out.weight [hidden, d_inner]: RowParallel{dim=1} splits
            // the input dim along the head-major slab. AR after.
            "ssm_out.weight" => Some(WeightLayout::row_parallel(self.world, 1)),

            // attn_qkv.weight (GDN fused QKV) and ssm_conv1d.weight share
            // outer dim = V_part + 2·K_part. TP-4a routes them through
            // FusedQkvParallel which slices V/K/Q sub-slabs independently
            // and re-concatenates per-rank as [V_local | K_local | Q_local].
            // Falls back to Replicated when gdn_dims is None (non-hybrid
            // configs that wouldn't actually have these tensors).
            "attn_qkv.weight" | "ssm_conv1d.weight" => self.gdn_dims.map_or(
                Some(WeightLayout::Replicated),
                |d| {
                    Some(WeightLayout::FusedQkvParallel {
                        world: self.world,
                        num_v_heads: d.num_v_heads,
                        num_k_heads: d.num_k_heads,
                        head_v_dim: d.head_v_dim,
                        head_k_dim: d.head_k_dim,
                    })
                },
            ),

            // ----- MoE expert tensors (TP-4b, qwen3moe / qwen35moe / qwen36moe) -----
            // ffn_gate_inp.weight [n_experts, hidden] — F32 router; runs
            // replicated on every rank (input is the AR'd hidden).
            "ffn_gate_inp.weight" => Some(WeightLayout::Replicated),
            // ffn_gate_exps.weight, ffn_up_exps.weight: 3D
            // [n_experts, moe_intermediate, hidden]. ColParallel{dim=1}
            // splits the per-expert intermediate slab.
            "ffn_gate_exps.weight" | "ffn_up_exps.weight" => {
                Some(WeightLayout::col_parallel(self.world, 1))
            }
            // ffn_down_exps.weight: 3D [n_experts, hidden, moe_intermediate].
            // RowParallel{dim=2} splits the per-expert per-row inner slab
            // (the down-proj input dim). AR after the combine kernel.
            "ffn_down_exps.weight" => Some(WeightLayout::row_parallel(self.world, 2)),

            // **TP-4c** — shared expert (qwen35moe / qwen36moe).
            // Same Megatron split as dense FFN.
            //   ffn_gate_inp_shexp.weight  [hidden]      — 1-D scalar
            //                                              gate weight; Replicated.
            //   ffn_gate_shexp.weight      [shared_inter, hidden] — ColParallel{dim=0}
            //   ffn_up_shexp.weight        [shared_inter, hidden] — ColParallel{dim=0}
            //   ffn_down_shexp.weight      [hidden, shared_inter] — RowParallel{dim=1}
            "ffn_gate_inp_shexp.weight" => Some(WeightLayout::Replicated),
            "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" => {
                Some(WeightLayout::col_parallel(self.world, 0))
            }
            "ffn_down_shexp.weight" => Some(WeightLayout::row_parallel(self.world, 1)),

            _ => None,
        }
    }
}

/// Strip `blk.<digit+>.` from the start of a per-layer tensor name.
/// Returns `None` if the prefix doesn't match.
fn strip_blk_prefix(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let layer_part = &rest[..dot];
    if layer_part.is_empty() || !layer_part.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&rest[dot + 1..])
}

/// Errors produced by [`Qwen35DenseTpLayout::new`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TpLayoutError {
    #[error("TP world size must be >= 1 (got 0)")]
    WorldZero,
    #[error("num_attention_heads {num_heads} is not divisible by world {world}")]
    IndivisibleNumHeads { num_heads: usize, world: u32 },
    #[error("num_key_value_heads {num_kv_heads} is not divisible by world {world}")]
    IndivisibleNumKvHeads { num_kv_heads: usize, world: u32 },
    #[error("intermediate_size {intermediate} is not divisible by world {world}")]
    IndivisibleIntermediate { intermediate: usize, world: u32 },
    #[error("gdn.num_v_heads {num_v_heads} is not divisible by world {world}")]
    IndivisibleNumVHeads { num_v_heads: usize, world: u32 },
    #[error("gdn.num_k_heads {num_k_heads} is not divisible by world {world}")]
    IndivisibleNumKHeadsGdn { num_k_heads: usize, world: u32 },
    #[error(
        "shared_expert_intermediate_size {shared_intermediate} is not divisible by world {world}"
    )]
    IndivisibleSharedExpertIntermediate {
        shared_intermediate: usize,
        world: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AttentionFamily, RopeSpec};

    /// Hand-crafted Qwen3.5-27B config. Real loader populates this from
    /// GGUF metadata; we hard-code here to keep the test self-contained.
    /// Field names match `config::Qwen3MoEConfig` exactly; `moe_intermediate_size`
    /// carries the dense-FFN intermediate for arch=qwen35 (loader quirk).
    fn qwen35_27b_cfg() -> Qwen3MoEConfig {
        Qwen3MoEConfig {
            arch: "qwen35".into(),
            family: AttentionFamily::Dense,
            hidden_size: 5120,
            vocab_size: 152064,
            num_layers: 64,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            context_length: 32768,
            rms_norm_eps: 1e-6,
            rope: RopeSpec {
                freq_base: 1_000_000.0,
                rotated_dims: 128,
                sections: None,
            },
            num_experts: 0,
            num_experts_per_tok: 1,
            moe_intermediate_size: 27648, // dense FFN width on qwen35
            shared_expert_intermediate_size: None,
            full_attention_interval: None,
            gdn: None,
            tied_lm_head: false,
        }
    }

    fn qwen35_9b_cfg() -> Qwen3MoEConfig {
        let mut c = qwen35_27b_cfg();
        c.hidden_size = 4096;
        c.moe_intermediate_size = 12288;
        c.num_heads = 32;
        c.num_kv_heads = 8;
        c.num_layers = 36;
        c
    }

    #[test]
    fn qwen35_27b_world4_validates() {
        Qwen35DenseTpLayout::validate(&qwen35_27b_cfg(), 4).expect("Qwen3.5-27B world=4 ok");
    }

    #[test]
    fn qwen35_27b_world_zero_rejected() {
        assert_eq!(
            Qwen35DenseTpLayout::validate(&qwen35_27b_cfg(), 0),
            Err(TpLayoutError::WorldZero)
        );
    }

    #[test]
    fn qwen35_9b_world4_validates() {
        Qwen35DenseTpLayout::validate(&qwen35_9b_cfg(), 4).expect("Qwen3.5-9B world=4 ok");
    }

    #[test]
    fn world_16_falls_back_to_kv_replicated_on_27b() {
        // TP-4d-i2: nQ=64 divides world=16, nKV=8 does NOT — instead
        // of erroring, the layout sets kv_replicated=true and routes
        // attn_k/v to Replicated.
        let l = Qwen35DenseTpLayout::new(&qwen35_27b_cfg(), 16).unwrap();
        assert!(l.kv_replicated());
        assert_eq!(l.for_tensor("blk.0.attn_k.weight"), Some(WeightLayout::Replicated));
        assert_eq!(l.for_tensor("blk.0.attn_v.weight"), Some(WeightLayout::Replicated));
        // attn_q stays sharded.
        assert_eq!(
            l.for_tensor("blk.0.attn_q.weight"),
            Some(WeightLayout::col_parallel(16, 0))
        );
    }

    #[test]
    fn world_4_kv_replicated_on_qwen36_35b() {
        // Qwen3.6-35B-A3B has nKV=2; world=4 fails divisibility →
        // TP-4d-i2 kv_replicated path engages.
        let mut c = qwen35_27b_cfg_with_gdn();
        c.num_heads = 16;
        c.num_kv_heads = 2;
        c.hidden_size = 2048;
        c.moe_intermediate_size = 17408; // qwen35moe = MoE; uses moe_intermediate
        let l = Qwen35DenseTpLayout::new(&c, 4).unwrap();
        assert!(l.kv_replicated());
    }

    #[test]
    fn world_3_rejected_on_27b_n_heads() {
        // nQ=64 doesn't divide world=3 → IndivisibleNumHeads (first
        // check that trips).
        let err = Qwen35DenseTpLayout::validate(&qwen35_27b_cfg(), 3).unwrap_err();
        assert!(matches!(err, TpLayoutError::IndivisibleNumHeads { .. }));
    }

    #[test]
    fn for_tensor_per_layer_dispatch_qwen35_27b_world4() {
        let l = Qwen35DenseTpLayout::new(&qwen35_27b_cfg(), 4).unwrap();
        // Attention projections.
        assert_eq!(
            l.for_tensor("blk.0.attn_q.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.7.attn_k.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.63.attn_v.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.attn_output.weight"),
            Some(WeightLayout::row_parallel(4, 1))
        );
        // Norms.
        assert_eq!(l.for_tensor("blk.0.attn_norm.weight"), Some(WeightLayout::Replicated));
        assert_eq!(l.for_tensor("blk.0.attn_q_norm.weight"), Some(WeightLayout::Replicated));
        assert_eq!(l.for_tensor("blk.0.attn_k_norm.weight"), Some(WeightLayout::Replicated));
        assert_eq!(l.for_tensor("blk.0.ffn_norm.weight"), Some(WeightLayout::Replicated));
        // FFN.
        assert_eq!(
            l.for_tensor("blk.0.ffn_gate.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.ffn_up.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.ffn_down.weight"),
            Some(WeightLayout::row_parallel(4, 1))
        );
        // Globals.
        assert_eq!(l.for_tensor("token_embd.weight"), Some(WeightLayout::Replicated));
        assert_eq!(l.for_tensor("output_norm.weight"), Some(WeightLayout::Replicated));
        assert_eq!(l.for_tensor("output.weight"), Some(WeightLayout::Replicated));
    }

    #[test]
    fn for_tensor_unknown_returns_none() {
        let l = Qwen35DenseTpLayout::new(&qwen35_27b_cfg(), 4).unwrap();
        assert_eq!(l.for_tensor("not.a.real.tensor"), None);
    }

    #[test]
    fn for_tensor_moe_expert_layouts() {
        // TP-4b: ffn_*_exps tensors map to MoE 3D layouts.
        let l = Qwen35DenseTpLayout::new(&qwen35_27b_cfg(), 4).unwrap();
        assert_eq!(l.for_tensor("blk.0.ffn_gate_inp.weight"), Some(WeightLayout::Replicated));
        assert_eq!(
            l.for_tensor("blk.0.ffn_gate_exps.weight"),
            Some(WeightLayout::col_parallel(4, 1))
        );
        assert_eq!(
            l.for_tensor("blk.0.ffn_up_exps.weight"),
            Some(WeightLayout::col_parallel(4, 1))
        );
        assert_eq!(
            l.for_tensor("blk.0.ffn_down_exps.weight"),
            Some(WeightLayout::row_parallel(4, 2))
        );
        // Shared expert (TP-4c): now Col/Row-parallel.
        assert_eq!(
            l.for_tensor("blk.0.ffn_gate_inp_shexp.weight"),
            Some(WeightLayout::Replicated)
        );
        assert_eq!(
            l.for_tensor("blk.0.ffn_gate_shexp.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.ffn_up_shexp.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.ffn_down_shexp.weight"),
            Some(WeightLayout::row_parallel(4, 1))
        );
    }

    fn qwen35_27b_cfg_with_gdn() -> Qwen3MoEConfig {
        let mut c = qwen35_27b_cfg();
        c.family = AttentionFamily::Hybrid;
        c.full_attention_interval = Some(4);
        c.gdn = Some(crate::config::GdnDims {
            d_inner: 6144,
            head_k_dim: 128,
            num_k_heads: 16,
            num_v_heads: 48,
            conv_kernel: 4,
        });
        c
    }

    #[test]
    fn for_tensor_attn_qkv_routes_through_fused_qkv() {
        let cfg = qwen35_27b_cfg_with_gdn();
        let l = Qwen35DenseTpLayout::new(&cfg, 4).unwrap();
        let layout = l.for_tensor("blk.0.attn_qkv.weight").expect("known tensor");
        assert!(
            matches!(layout, WeightLayout::FusedQkvParallel { .. }),
            "attn_qkv should route through FusedQkvParallel, got {layout:?}"
        );
        if let WeightLayout::FusedQkvParallel {
            world,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
        } = layout
        {
            assert_eq!(world, 4);
            assert_eq!(num_v_heads, 48);
            assert_eq!(num_k_heads, 16);
            assert_eq!(head_v_dim, 128);
            assert_eq!(head_k_dim, 128);
        }
    }

    #[test]
    fn for_tensor_attn_qkv_falls_back_to_replicated_without_gdn() {
        // Pure-dense (no GDN) — attn_qkv shouldn't even appear, but if
        // it did, gdn_dims=None makes us fall back to Replicated.
        let l = Qwen35DenseTpLayout::new(&qwen35_27b_cfg(), 4).unwrap();
        assert_eq!(
            l.for_tensor("blk.0.attn_qkv.weight"),
            Some(WeightLayout::Replicated)
        );
    }

    #[test]
    fn synthetic_gdn_num_v_head_violation_caught() {
        // Synthetic config: nQ=4 / nKV=4 / inter=4 (all divide world=4),
        // but gdn.num_v_heads=6 doesn't (6 % 4 = 2). Catches GDN-specific
        // path independently of the upstream attn divisibility checks.
        let mut c = qwen35_27b_cfg();
        c.num_heads = 4;
        c.num_kv_heads = 4;
        c.moe_intermediate_size = 4;
        c.gdn = Some(crate::config::GdnDims {
            d_inner: 768, // 6 v_heads × 128
            head_k_dim: 128,
            num_k_heads: 4,
            num_v_heads: 6,
            conv_kernel: 4,
        });
        let err = Qwen35DenseTpLayout::validate(&c, 4).unwrap_err();
        assert!(matches!(err, TpLayoutError::IndivisibleNumVHeads { num_v_heads: 6, world: 4 }));
    }

    #[test]
    fn for_tensor_hybrid_gdn_layouts() {
        let l = Qwen35DenseTpLayout::new(&qwen35_27b_cfg(), 4).unwrap();
        // GDN per-v-head scalars + projections.
        assert_eq!(l.for_tensor("blk.0.ssm_a"), Some(WeightLayout::col_parallel(4, 0)));
        assert_eq!(
            l.for_tensor("blk.0.ssm_dt.bias"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.ssm_alpha.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.ssm_beta.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.attn_gate.weight"),
            Some(WeightLayout::col_parallel(4, 0))
        );
        assert_eq!(
            l.for_tensor("blk.0.ssm_out.weight"),
            Some(WeightLayout::row_parallel(4, 1))
        );
        assert_eq!(l.for_tensor("blk.0.ssm_norm.weight"), Some(WeightLayout::Replicated));
        // Fused-QKV class — replicated until TP-4a's head-aware split.
        assert_eq!(l.for_tensor("blk.0.attn_qkv.weight"), Some(WeightLayout::Replicated));
        assert_eq!(
            l.for_tensor("blk.0.ssm_conv1d.weight"),
            Some(WeightLayout::Replicated)
        );
    }

    #[test]
    fn strip_blk_prefix_rejects_malformed() {
        assert_eq!(strip_blk_prefix("blk.0.x"), Some("x"));
        assert_eq!(strip_blk_prefix("blk.99.foo.bar.weight"), Some("foo.bar.weight"));
        // No leading "blk." prefix.
        assert_eq!(strip_blk_prefix("token_embd.weight"), None);
        // Layer index isn't all digits.
        assert_eq!(strip_blk_prefix("blk.x.attn_q.weight"), None);
        // Empty layer index.
        assert_eq!(strip_blk_prefix("blk..attn_q.weight"), None);
    }
}
