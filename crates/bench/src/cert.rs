//! Correctness-cert JSON schema.
//! A cert is a signed contract that `impl_id` produces results within the
//! declared tolerance on a fixed shape grid. Every dispatch row in
//! `dispatch/<backend>/<arch>.toml` must point to a cert file that exists on
//! disk and has `pass: true`.
//! The JSON is stable — adding fields is fine, removing or re-typing
//! requires a version bump. `bench cert-check` is the build-time gate that
//! catches dispatch rows whose cert is missing or stale.

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

/// Top-level cert document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cert {
    pub schema_version: u32,
    /// Matches `KernelImpl::ID` — no leading `qmatmul_`, caller-chosen.
    pub impl_id: String,
    pub backend: String,
    pub arch: String,
    pub op: String,
    /// "Q4_K", "Q8_0", etc. Operand dtype (weights); activation dtype is in
    /// `dtype_activation`.
    pub dtype_weight: String,
    pub dtype_activation: String,
    /// Tolerance formula as plain text so future-you / PR reviewers see it
    /// without re-reading this crate.
    pub tolerance_formula: String,
    /// Shape × result rows.
    pub results: Vec<ShapeResult>,
    /// True iff every result's `max_rel_err <= tol`.
    pub pass: bool,
    /// ISO-8601-ish UTC stamp.
    pub emitted_at: String,
    /// Identifier for the rig that produced this cert (hostname + GPU arch).
    pub rig: String,
    /// First-level PMC metrics, if captured. Empty for (rocprofv3 wiring
    /// lands later).
    #[serde(default)]
    pub pmc: Option<PmcSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeResult {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub seed: u64,
    pub max_rel_err: f32,
    pub tolerance: f32,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmcSnapshot {
    pub vgpr_count: Option<u32>,
    pub sgpr_count: Option<u32>,
    /// `waves_per_simd` reported by rocprofv3 — derived from VGPR budget.
    pub waves_per_simd: Option<u32>,
    pub mem_busy_pct: Option<f32>,
    pub valu_busy_pct: Option<f32>,
}

impl Cert {
    pub fn cert_path(backend: &str, arch: &str, impl_id: &str) -> std::path::PathBuf {
        std::path::Path::new("certs")
            .join(backend)
            .join(arch)
            .join(format!("{impl_id}.json"))
    }

    pub fn write_to_disk(&self, repo_root: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
        let rel = Self::cert_path(&self.backend, &self.arch, &self.impl_id);
        let abs = repo_root.join(&rel);
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&abs, json)?;
        Ok(rel)
    }

    pub fn read_from_disk(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let cert: Self = serde_json::from_str(&data)?;
        Ok(cert)
    }
}

/// ISO-8601-ish UTC stamp (`YYYY-MM-DDTHH:MM:SSZ`). Avoids pulling in `chrono`.
pub fn now_utc_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_unix_utc(secs)
}

fn format_unix_utc(mut secs: u64) -> String {
    // Plain Gregorian calendar conversion — good enough for cert timestamps.
    let seconds_in_minute = 60;
    let seconds_in_hour = 3600;
    let seconds_in_day = 86400;
    let days = secs / seconds_in_day;
    secs %= seconds_in_day;
    let hours = (secs / seconds_in_hour) as u32;
    secs %= seconds_in_hour;
    let minutes = (secs / seconds_in_minute) as u32;
    let seconds = (secs % seconds_in_minute) as u32;

    // Days since 1970-01-01 → Y-M-D.
    let (year, month, day) = days_to_ymd(days as i64);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn days_to_ymd(mut days: i64) -> (i32, u32, u32) {
    // Derived from Howard Hinnant's date algorithms.
    days += 719468;
    let era = days.div_euclid(146097);
    let doe = days.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}
