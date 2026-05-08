//! Minimal reader for `dispatch/<backend>/<arch>.toml`.
//! `bench cert-check` loads every dispatch row's `cert` path and asserts the
//! file exists + has `pass: true` + matches the row's `impl_id` / `dtype`.
//! Any failure fails the check; CI can wire this up as a build-time gate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ShapePredicate {
    Literal(toml::Value),
}

#[derive(Debug, Clone, Deserialize)]
pub struct DispatchRow {
    pub dtype: String,
    #[serde(default)]
    pub dtype_q: Option<String>,
    #[serde(default)]
    pub shape: HashMap<String, toml::Value>,
    pub r#impl: String,
    pub cert: String,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct DispatchTable {
    /// Decode-path MMVQ (matrix × vector).
    #[serde(default)]
    pub qmatmul: Vec<DispatchRow>,
    /// Prefill-path MMQ (matrix × matrix).
    #[serde(default)]
    pub qmatmul_mmq: Vec<DispatchRow>,
    /// 1.b — F16-weight × Q8_1 / BF16-weight × BF16 mmvq
    /// rows. The wrappers short-circuit the dispatch table for these
    /// dtypes (no shape selection); rows live here purely to gate
    /// cert-check.
    #[serde(default)]
    pub qmatmul_mmvq: Vec<DispatchRow>,
    /// fused decode path — RMSNorm, SwiGLU, RoPE, softmax, attention.
    #[serde(default)]
    pub rmsnorm: Vec<DispatchRow>,
    #[serde(default)]
    pub swiglu: Vec<DispatchRow>,
    #[serde(default)]
    pub rope: Vec<DispatchRow>,
    #[serde(default)]
    pub rope_neox_partial: Vec<DispatchRow>,
    #[serde(default)]
    pub l2_norm: Vec<DispatchRow>,
    #[serde(default)]
    pub add_f16: Vec<DispatchRow>,
    #[serde(default)]
    pub cast_f32_f16: Vec<DispatchRow>,
    #[serde(default)]
    pub cast_f16_f32: Vec<DispatchRow>,
    /// BF16 cast surfaces.
    #[serde(default)]
    pub cast_f32_bf16: Vec<DispatchRow>,
    #[serde(default)]
    pub cast_bf16_f32: Vec<DispatchRow>,
    #[serde(default)]
    pub cast_f16_bf16: Vec<DispatchRow>,
    #[serde(default)]
    pub cast_bf16_f16: Vec<DispatchRow>,
    #[serde(default)]
    pub causal_conv1d: Vec<DispatchRow>,
    #[serde(default)]
    pub dense_gemv_f32_f16: Vec<DispatchRow>,
    #[serde(default)]
    pub gdn_alpha_beta: Vec<DispatchRow>,
    #[serde(default)]
    pub gdn_state_step: Vec<DispatchRow>,
    #[serde(default)]
    pub quantize_f16_q8_1: Vec<DispatchRow>,
    #[serde(default)]
    pub rmsnorm_f32: Vec<DispatchRow>,
    #[serde(default)]
    pub scale_f32: Vec<DispatchRow>,
    #[serde(default)]
    pub shared_expert_scale: Vec<DispatchRow>,
    #[serde(default)]
    pub silu_f32: Vec<DispatchRow>,
    #[serde(default)]
    pub swiglu_f32: Vec<DispatchRow>,
    #[serde(default)]
    pub split_q_gate: Vec<DispatchRow>,
    #[serde(default)]
    pub softmax: Vec<DispatchRow>,
    #[serde(default)]
    pub attention_decode: Vec<DispatchRow>,
    #[serde(default)]
    pub attention_prefill: Vec<DispatchRow>,
    /// MoE kernels.
    #[serde(default)]
    pub topk: Vec<DispatchRow>,
    #[serde(default)]
    pub indexed_moe_mmvq: Vec<DispatchRow>,
    #[serde(default)]
    pub moe_combine: Vec<DispatchRow>,
    #[serde(default)]
    pub indexed_moe_mmvq_gate_up: Vec<DispatchRow>,
    #[serde(default)]
    pub indexed_moe_mmq: Vec<DispatchRow>,
    /// Named groups of qmatmul rows (e.g. `.example`) — kept so the file can
    /// hold commented-out experimental rows without breaking cert-check.
    /// `cert-check` validates all top-level op lists above; rows under a
    /// group prefix are informational.
    #[serde(default)]
    pub qmatmul_example: Vec<DispatchRow>,
}

impl DispatchTable {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading dispatch table {}", path.display()))?;
        let table: Self = toml::from_str(&text)
            .with_context(|| format!("parsing dispatch table {}", path.display()))?;
        Ok(table)
    }
}

/// Run cert-check over `dispatch_path`. For every active row (not under an
/// `example` group), assert the cert file exists, parses, has `pass: true`,
/// and matches the row's `impl` + dtype.
pub fn cert_check(repo_root: &Path, dispatch_path: &Path) -> Result<CertCheckReport> {
    let table = DispatchTable::load(dispatch_path)?;
    let mut rows_checked = 0usize;
    let mut failures = Vec::new();

    for row in &table.qmatmul {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.qmatmul_mmq {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.qmatmul_mmvq {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.rmsnorm {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.swiglu {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.rope {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.rope_neox_partial {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.l2_norm {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.add_f16 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.cast_f32_f16 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.cast_f16_f32 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.cast_f32_bf16 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.cast_bf16_f32 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.cast_f16_bf16 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.cast_bf16_f16 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.causal_conv1d {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.dense_gemv_f32_f16 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.gdn_alpha_beta {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.gdn_state_step {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.quantize_f16_q8_1 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.rmsnorm_f32 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.scale_f32 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.shared_expert_scale {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.silu_f32 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.swiglu_f32 {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.split_q_gate {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.softmax {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.attention_decode {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.attention_prefill {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.topk {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.indexed_moe_mmvq {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.moe_combine {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.indexed_moe_mmvq_gate_up {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }
    for row in &table.indexed_moe_mmq {
        rows_checked += 1;
        if let Err(e) = check_one(repo_root, row) {
            failures.push((row.r#impl.clone(), e));
        }
    }

    Ok(CertCheckReport {
        rows_checked,
        failures,
    })
}

fn check_one(repo_root: &Path, row: &DispatchRow) -> Result<()> {
    let cert_path: PathBuf = repo_root.join(&row.cert);
    if !cert_path.exists() {
        return Err(anyhow!(
            "cert file missing: {} (dispatch row impl={})",
            cert_path.display(),
            row.r#impl
        ));
    }
    let cert = crate::cert::Cert::read_from_disk(&cert_path)
        .with_context(|| format!("reading cert {}", cert_path.display()))?;

    if cert.impl_id != row.r#impl {
        return Err(anyhow!(
            "cert impl_id {} != dispatch row impl {}",
            cert.impl_id,
            row.r#impl
        ));
    }
    if cert.dtype_weight != row.dtype {
        return Err(anyhow!(
            "cert dtype_weight {} != dispatch row dtype {}",
            cert.dtype_weight,
            row.dtype
        ));
    }
    if let Some(dq) = &row.dtype_q {
        if &cert.dtype_activation != dq {
            return Err(anyhow!(
                "cert dtype_activation {} != dispatch row dtype_q {}",
                cert.dtype_activation,
                dq
            ));
        }
    }
    if !cert.pass {
        return Err(anyhow!("cert pass=false for {}", row.r#impl));
    }
    Ok(())
}

#[derive(Debug)]
pub struct CertCheckReport {
    pub rows_checked: usize,
    pub failures: Vec<(String, anyhow::Error)>,
}

impl CertCheckReport {
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}
