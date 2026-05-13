//! `flambeau_runtime::Model` implementor — declares which GGUF
//! `general.architecture` strings this crate handles.

use flambeau_runtime::Model;

/// Gemma 4 runtime entry. Covers the four sizes documented in
/// `certs/research/gemma4_recon.md`:
/// - `26B-A4B` (30 layers, MoE, n_experts=128 used=8).
/// - `E2B` (35 layers, edge dense, has per-layer side-channel embedding).
/// - `E4B` (42 layers, edge dense, has per-layer side-channel embedding and
///   a shared-KV tail of `attention.shared_kv_layers` layers).
/// - `31B` (60 layers, dense, no side-channel, no shared-KV).
///
/// All four share `general.architecture = "gemma4"`. Variant detection is
/// by `block_count` (see [`crate::config::Gemma4Variant`]).
pub struct Gemma4ModelArch;

impl Model for Gemma4ModelArch {
    fn supported_archs(&self) -> &[&'static str] {
        &["gemma4"]
    }

    fn description(&self) -> &'static str {
        "Gemma 4 family (E2B / E4B edge + 26B-A4B MoE + 31B dense, iSWA + softcap)"
    }
}
