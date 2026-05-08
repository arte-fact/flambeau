//! `flambeau_runtime::Model` implementor — declares which GGUF
//! `general.architecture` strings this crate handles.

use flambeau_runtime::Model;

/// Qwen3 MoE-family runtime entry. Covers Qwen3.5 / Qwen3.6 dense
/// (full-attn + dense-FFN) and MoE (full-attn + indexed-MoE-FFN), plus
/// Qwen3-Coder-Next (GDN hybrid + MoE).
pub struct Qwen3MoEModelArch;

impl Model for Qwen3MoEModelArch {
    fn supported_archs(&self) -> &[&'static str] {
        // Keep aligned with `Qwen3MoEConfig::from_gguf` — every arch
        // string the config parser accepts must be registered here so
        // `Registry::validate` matches GGUFs the model can actually
        // load.
        &["qwen35moe", "qwen3moe", "qwen3next"]
    }

    fn description(&self) -> &'static str {
        "Qwen3 family (Qwen3.5/3.6 dense + MoE, Qwen3-Coder-Next GDN hybrid)"
    }
}
