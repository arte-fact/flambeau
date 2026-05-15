//! Sliding-window-attention alternation policy.
//!
//! Gemma 4 mixes full-attention and SWA layers in a per-layer pattern
//! (e.g. the 26B-A4B `SSSSSF` repeat) where SWA layers have a smaller
//! `head_dim`, different RoPE config, and a non-zero window radius.
//! All of this is keyed on per-layer index, so we bundle it in one
//! `SwaAlternationPolicy` substruct on [`crate::Gemma4Config`].
//!
//! The non-SWA defaults (full-attention `head_dim`, `rope_dim`,
//! `rope_freq_base`) live on `Gemma4Config` directly — the policy
//! only carries the SWA-specific overrides.

/// Sliding-window-attention alternation policy. `swa_layers[il]`
/// decides whether layer `il` uses the SWA values
/// (`head_dim_swa` / `rope_dim_swa` / `rope_freq_base_swa` /
/// `sliding_window`) or the full-attention defaults from
/// [`crate::Gemma4Config`].
#[derive(Debug, Clone, PartialEq)]
pub struct SwaAlternationPolicy {
    /// `swa_layers[il] == true` ⇒ layer `il` is SWA. Read from
    /// `gemma4.attention.sliding_window_pattern`.
    pub swa_layers: Vec<bool>,
    /// SWA mask radius (`gemma4.attention.sliding_window`). Only
    /// meaningful when at least one entry of `swa_layers` is `true`.
    pub sliding_window: usize,
    /// `head_dim` for SWA layers (`gemma4.attention.key_length_swa`
    /// in spec; 256 on most gemma4 variants).
    pub head_dim_swa: usize,
    /// Rotated-dim count for RoPE on SWA layers
    /// (`gemma4.rope.dimension_count_swa`).
    pub rope_dim_swa: usize,
    /// RoPE frequency base for SWA layers (`gemma4.rope.freq_base_swa`).
    pub rope_freq_base_swa: f32,
}

impl SwaAlternationPolicy {
    /// `true` iff layer `il` is sliding-window-attention.
    pub fn is_swa(&self, il: usize) -> bool {
        self.swa_layers.get(il).copied().unwrap_or(false)
    }

    /// Per-layer head_dim — `head_dim_swa` for SWA layers,
    /// `full_head_dim` otherwise.
    pub fn head_dim_for(&self, il: usize, full_head_dim: usize) -> usize {
        if self.is_swa(il) {
            self.head_dim_swa
        } else {
            full_head_dim
        }
    }

    /// Per-layer RoPE rotated-dim count.
    pub fn rope_dim_for(&self, il: usize, full_rope_dim: usize) -> usize {
        if self.is_swa(il) {
            self.rope_dim_swa
        } else {
            full_rope_dim
        }
    }

    /// Per-layer RoPE frequency base.
    pub fn rope_freq_base_for(&self, il: usize, full_rope_freq_base: f32) -> f32 {
        if self.is_swa(il) {
            self.rope_freq_base_swa
        } else {
            full_rope_freq_base
        }
    }

    /// Per-layer mask window radius — `sliding_window` for SWA
    /// layers, `0` for full-attention layers (no window).
    pub fn window_for(&self, il: usize) -> u32 {
        if self.is_swa(il) {
            self.sliding_window as u32
        } else {
            0
        }
    }

    /// Count of full-attention layers (used by callers that allocate
    /// per-policy arrays sized by full-attn layer count).
    pub fn num_full_attn_layers(&self) -> usize {
        self.swa_layers.iter().filter(|b| !**b).count()
    }
}
