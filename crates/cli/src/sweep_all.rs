//! `flambeau sweep --all-dispatch`: re-sweeps every active row of an arch's
//! dispatch table into a separate directory, then cert-checks that
//! directory against the table.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use flambeau_bench::cert::Cert;
use flambeau_bench::dispatch::{cert_check, DispatchTable};

/// `(arch, impl_id, --op, --dtype)`: the sweep invocation that writes each
/// dispatched impl's cert. The tests below keep it in step with
/// `dispatch/*/*.toml`.
const DISPATCH_SWEEPS: &[(&str, &str, &str, &str)] = &[
    ("gfx906", "add_f16", "add_f16", "all"),
    ("gfx906", "attention_decode_f16_gfx906", "attention_decode", "all"),
    ("gfx906", "attention_decode_f16_splitk_gfx906", "attention_decode_splitk", "all"),
    ("gfx906", "attention_decode_q8_kv_gfx906", "attention_decode_q8_kv_gfx906", "all"),
    ("gfx906", "attention_prefill_f16_gfx906", "attention_prefill", "all"),
    ("gfx906", "cast_f16_f32", "cast_f16_f32", "all"),
    ("gfx906", "cast_f32_f16", "cast_f32_f16", "all"),
    ("gfx906", "causal_conv1d_f32_gfx906", "causal_conv1d", "all"),
    ("gfx906", "dense_gemv_f32_f16_gfx906", "dense_gemv_f32_f16_gfx906", "all"),
    ("gfx906", "gdn_alpha_beta_f32_gfx906", "gdn_alpha_beta", "all"),
    ("gfx906", "gdn_state_step_alphabeta_f32_s128_gfx906", "gdn_state_step_alphabeta", "all"),
    ("gfx906", "gdn_state_step_f32_s128_gfx906", "gdn_state_step", "all"),
    ("gfx906", "indexed_moe_mmvq_q4_0_gfx906", "indexed_moe_mmvq_q4_0", "all"),
    ("gfx906", "indexed_moe_mmvq_q4_k_gate_up_r4", "qmatmul", "Q4_K_gate_up_r4_moe_mmvq"),
    ("gfx906", "indexed_moe_mmvq_q4_k_r2", "qmatmul", "Q4_K_r2_moe_mmvq"),
    ("gfx906", "indexed_moe_mmvq_q4_k_r4", "qmatmul", "Q4_K_r4_moe_mmvq"),
    ("gfx906", "indexed_moe_mmvq_q6_k_gfx906", "indexed_moe_mmvq_q6_k", "all"),
    ("gfx906", "indexed_moe_mmvq_q8_0_gfx906", "indexed_moe_mmvq_q8_0", "all"),
    ("gfx906", "l2_norm_f32", "l2_norm_f32", "all"),
    ("gfx906", "mmvq_q4_0_gfx906", "mmvq_q4_0", "all"),
    ("gfx906", "mmvq_q5_0_gfx906", "mmvq_q5_0", "all"),
    ("gfx906", "mmvq_q5_1_gfx906", "mmvq_q5_1", "all"),
    ("gfx906", "moe_combine_f16_gfx906", "moe_combine", "all"),
    ("gfx906", "qmatmul_iq1_m_mmq_wave64_gfx906", "qmatmul_mmq", "IQ1_M_wave64"),
    ("gfx906", "qmatmul_iq1_m_mmvq_dp4a", "qmatmul", "IQ1_M_dp4a"),
    ("gfx906", "qmatmul_iq1_s_mmq_wave64_gfx906", "qmatmul_mmq", "IQ1_S_wave64"),
    ("gfx906", "qmatmul_iq1_s_mmvq_dp4a", "qmatmul", "IQ1_S_dp4a"),
    ("gfx906", "qmatmul_iq2_s_mmq_wave64_gfx906", "qmatmul_mmq", "IQ2_S_wave64"),
    ("gfx906", "qmatmul_iq2_s_mmvq_dp4a", "qmatmul", "IQ2_S_dp4a"),
    ("gfx906", "qmatmul_iq2_xs_mmq_wave64_gfx906", "qmatmul_mmq", "IQ2_XS_wave64"),
    ("gfx906", "qmatmul_iq2_xs_mmvq_dp4a", "qmatmul", "IQ2_XS_dp4a"),
    ("gfx906", "qmatmul_iq2_xxs_mmq_wave64_gfx906", "qmatmul_mmq", "IQ2_XXS_wave64"),
    ("gfx906", "qmatmul_iq2_xxs_mmvq_dp4a", "qmatmul", "IQ2_XXS_dp4a"),
    ("gfx906", "qmatmul_iq3_s_mmq_wave64_gfx906", "qmatmul_mmq", "IQ3_S_wave64"),
    ("gfx906", "qmatmul_iq3_s_mmvq_dp4a", "qmatmul", "IQ3_S_dp4a"),
    ("gfx906", "qmatmul_iq3_xxs_mmq_wave64_gfx906", "qmatmul_mmq", "IQ3_XXS_wave64"),
    ("gfx906", "qmatmul_iq3_xxs_mmvq_dp4a", "qmatmul", "IQ3_XXS_dp4a"),
    ("gfx906", "qmatmul_iq4_nl_mmq_lds_tiled_gfx906", "qmatmul_mmq", "IQ4_NL_lds_tiled"),
    ("gfx906", "qmatmul_iq4_nl_mmq_lds_tiled_x8_gfx906", "qmatmul_mmq", "IQ4_NL_lds_tiled_x8"),
    ("gfx906", "qmatmul_iq4_nl_mmvq_dp4a", "qmatmul", "IQ4_NL_dp4a"),
    ("gfx906", "qmatmul_iq4_xs_mmq_lds_tiled_gfx906", "qmatmul_mmq", "IQ4_XS_lds_tiled"),
    ("gfx906", "qmatmul_iq4_xs_mmq_lds_tiled_x8_gfx906", "qmatmul_mmq", "IQ4_XS_lds_tiled_x8"),
    ("gfx906", "qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4", "qmatmul", "IQ4_XS_dp4a_t128_vdr4"),
    ("gfx906", "qmatmul_q2_K_mmq_wave64_gfx906", "qmatmul_mmq", "Q2_K_wave64"),
    ("gfx906", "qmatmul_q2_K_mmvq_r2_dp4a_gfx906", "qmatmul", "Q2_K_r2_dp4a"),
    ("gfx906", "qmatmul_q3_K_mmq_lds_tiled_gfx906", "qmatmul_mmq", "Q3_K_lds_tiled"),
    ("gfx906", "qmatmul_q3_K_mmq_lds_tiled_x8_gfx906", "qmatmul_mmq", "Q3_K_lds_tiled_x8"),
    ("gfx906", "qmatmul_q3_K_mmvq_r2_dp4a_gfx906", "qmatmul", "Q3_K_r2_dp4a"),
    ("gfx906", "qmatmul_q4_0_mmq_4warp_lds_gfx906", "qmatmul_mmq", "Q4_0_4warp"),
    ("gfx906", "qmatmul_q4_0_mmq_wave64_gfx906", "qmatmul_mmq", "Q4_0_wave64"),
    ("gfx906", "qmatmul_q4_1_mmq_4warp_lds_gfx906", "qmatmul_mmq", "Q4_1_4warp"),
    ("gfx906", "qmatmul_q4_1_mmq_wave64_gfx906", "qmatmul_mmq", "Q4_1_wave64"),
    ("gfx906", "qmatmul_q4_1_mmvq_t128_gfx906", "qmatmul", "Q4_1_t128"),
    ("gfx906", "qmatmul_q4_K_mmq_lds_tiled_y64_gfx906", "qmatmul_mmq", "Q4_K_lds_tiled_y64"),
    ("gfx906", "qmatmul_q4_K_mmq_lds_tiled_y64_x8_gfx906", "qmatmul_mmq", "Q4_K_lds_tiled_y64_x8"),
    ("gfx906", "qmatmul_q4_k_mmvq_dp4a_t128_vdr2", "qmatmul", "Q4_K_dp4a_t128_vdr2"),
    ("gfx906", "qmatmul_q5_0_mmq_wave64_gfx906", "qmatmul_mmq", "Q5_0_wave64"),
    ("gfx906", "qmatmul_q5_1_mmq_wave64_gfx906", "qmatmul_mmq", "Q5_1_wave64"),
    ("gfx906", "qmatmul_q5_K_mmq_lds_tiled_gfx906", "qmatmul_mmq", "Q5_K_lds_tiled"),
    ("gfx906", "qmatmul_q5_K_mmq_lds_tiled_x8_gfx906", "qmatmul_mmq", "Q5_K_lds_tiled_x8"),
    ("gfx906", "qmatmul_q5_k_mmvq_dp4a_t128_vdr2", "qmatmul", "Q5_K_dp4a_t128_vdr2"),
    ("gfx906", "qmatmul_q6_K_mmq_lds_tiled_gfx906", "qmatmul_mmq", "Q6_K_lds_tiled"),
    ("gfx906", "qmatmul_q6_K_mmq_lds_tiled_x8_gfx906", "qmatmul_mmq", "Q6_K_lds_tiled_x8"),
    ("gfx906", "qmatmul_q6_k_mmvq_dp4a", "qmatmul", "Q6_K_dp4a"),
    ("gfx906", "qmatmul_q8_0_mmq_lds_tiled_gfx906", "qmatmul_mmq", "Q8_0_lds_tiled"),
    ("gfx906", "qmatmul_q8_0_mmq_oracle_gfx906", "qmatmul_mmq", "Q8_0_oracle"),
    ("gfx906", "qmatmul_q8_0_mmvq_single_row_gfx906", "qmatmul", "Q8_0"),
    ("gfx906", "qmatmul_q8_K_mmq_wave64_gfx906", "qmatmul_mmq", "Q8_K_wave64"),
    ("gfx906", "qmatmul_q8_k_mmvq_single_row", "qmatmul", "Q8_K"),
    ("gfx906", "qmatmul_f16_mmq", "qmatmul", "F16_mmq"),
    ("gfx906", "qmatmul_f16_mmq_tile", "qmatmul", "F16_mmq_tile"),
    ("gfx906", "qmatmul_f16_mmvq", "qmatmul", "F16_mmvq"),
    ("gfx906", "quantize_f16_q8_1_gfx906", "quantize_f16_q8_1_gfx906", "all"),
    ("gfx906", "rmsnorm_f16_gfx906", "rmsnorm", "all"),
    ("gfx906", "rmsnorm_f32", "rmsnorm_f32", "all"),
    ("gfx906", "rmsnorm_q8_1_fused_gfx906", "rmsnorm_q8_1", "all"),
    ("gfx906", "rope_f16_gfx906", "rope", "all"),
    ("gfx906", "rope_neox_partial_f16_gfx906", "rope_neox_partial", "all"),
    ("gfx906", "scale_f32", "scale_f32", "all"),
    ("gfx906", "shared_expert_scale_f32_gfx906", "shared_expert_scale", "all"),
    ("gfx906", "silu_f32_gfx906", "silu_f32_gfx906", "all"),
    ("gfx906", "softmax_masked_f16_gfx906", "softmax", "all"),
    ("gfx906", "split_q_gate_f16", "split_q_gate_f16", "all"),
    ("gfx906", "swiglu_f16_gfx906", "swiglu", "all"),
    ("gfx906", "swiglu_f32_gfx906", "swiglu_f32_gfx906", "all"),
    ("gfx906", "topk_f32_gfx906", "topk", "all"),
    ("sm_86", "qmatmul_iq1_m_mmvq_dp4a", "qmatmul", "IQ1_M_dp4a"),
    ("sm_86", "qmatmul_iq1_s_mmvq_dp4a", "qmatmul", "IQ1_S_dp4a"),
    ("sm_86", "qmatmul_iq2_s_mmvq_dp4a", "qmatmul", "IQ2_S_dp4a"),
    ("sm_86", "qmatmul_iq2_xs_mmvq_dp4a", "qmatmul", "IQ2_XS_dp4a"),
    ("sm_86", "qmatmul_iq2_xxs_mmvq_dp4a", "qmatmul", "IQ2_XXS_dp4a"),
    ("sm_86", "qmatmul_iq3_s_mmvq_dp4a_t128_vdr2", "qmatmul", "IQ3_S_dp4a_t128_vdr2"),
    ("sm_86", "qmatmul_iq3_xxs_mmvq_dp4a", "qmatmul", "IQ3_XXS_dp4a"),
    ("sm_86", "qmatmul_iq4_nl_mmvq_dp4a_t128_vdr2", "qmatmul", "IQ4_NL_dp4a_t128_vdr2"),
    ("sm_86", "qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4", "qmatmul", "IQ4_XS_dp4a_t128_vdr4"),
    ("sm_86", "qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4_batched", "qmatmul", "IQ4_XS_vdr_batched"),
    ("sm_86", "qmatmul_q2_k_mmvq_dp4a", "qmatmul", "Q2_K"),
    ("sm_86", "qmatmul_q3_k_mmvq_dp4a_t128_vdr2", "qmatmul", "Q3_K_dp4a_t128_vdr2"),
    ("sm_86", "qmatmul_q4_0_mmvq_dp4a", "qmatmul", "Q4_0"),
    ("sm_86", "qmatmul_q4_1_mmvq_dp4a", "qmatmul", "Q4_1"),
    ("sm_86", "qmatmul_q4_k_mmvq_dp4a_t128_vdr2", "qmatmul", "Q4_K_dp4a_t128_vdr2"),
    ("sm_86", "qmatmul_q5_0_mmvq_dp4a", "qmatmul", "Q5_0"),
    ("sm_86", "qmatmul_q5_1_mmvq_dp4a", "qmatmul", "Q5_1"),
    ("sm_86", "qmatmul_q5_k_mmvq_dp4a_t128_vdr2", "qmatmul", "Q5_K_dp4a_t128_vdr2"),
    ("sm_86", "qmatmul_q5_k_mmvq_dp4a_t128_vdr2_batched", "qmatmul", "Q5_K_vdr_batched"),
    ("sm_86", "qmatmul_q6_k_mmvq_dp4a", "qmatmul", "Q6_K"),
    ("sm_86", "qmatmul_q8_0_mmvq_dp4a", "qmatmul", "Q8_0_dp4a"),
    ("sm_86", "qmatmul_q8_k_mmvq_single_row", "qmatmul", "Q8_K"),
];

fn sweep_for(arch: &str, impl_id: &str) -> Option<(&'static str, &'static str)> {
    DISPATCH_SWEEPS
        .iter()
        .find(|(a, i, _, _)| *a == arch && *i == impl_id)
        .map(|&(_, _, op, dtype)| (op, dtype))
}

fn backend_of(arch: &str) -> Result<&'static str> {
    match arch {
        "gfx906" => Ok("hip"),
        "sm_86" => Ok("cuda"),
        other => anyhow::bail!("--all-dispatch: no dispatch table for arch {other:?}"),
    }
}

/// Runs one sweep invocation in a child `flambeau` process. A kernel or
/// launch fault (SIGSEGV inside the HIP runtime, GPU memory abort) kills
/// the process it happens in; isolating each invocation keeps the remaining
/// sweeps and the final verdict table alive.
fn sweep_in_child(arch: &str, op: &str, dtype: &str, out: &Path) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("locating own executable: {e}"))?;
    let status = std::process::Command::new(exe)
        .args(["sweep", "--arch", arch, "--op", op, "--dtype", dtype, "--out"])
        .arg(out)
        .status()
        .map_err(|e| format!("spawning sweep: {e}"))?;
    if status.success() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return Err(format!("crashed (signal {sig})"));
        }
    }
    Err(format!("exited with {status}"))
}

pub(crate) fn run(arch: &str, out: &Path, dispatch: Option<&Path>) -> Result<()> {
    let backend = backend_of(arch)?;
    let dispatch_path = match dispatch {
        Some(p) => p.to_path_buf(),
        None => crate::repo_root().join(format!("dispatch/{backend}/{arch}.toml")),
    };
    let table = DispatchTable::load(&dispatch_path)?;

    let mut certs: BTreeMap<&str, &str> = BTreeMap::new();
    for row in table.active_rows() {
        certs.entry(row.r#impl.as_str()).or_insert(row.cert.as_str());
    }

    // Clear every target first: a cert left by an earlier run must not
    // stand in for one this run failed to write, and one invocation can
    // write several targets.
    for cert in certs.values() {
        let path = out.join(cert);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => anyhow::bail!("removing stale {}: {e}", path.display()),
        }
    }
    std::fs::create_dir_all(out)?;

    let mut runs: BTreeMap<(&str, &str), Result<(), String>> = BTreeMap::new();
    for impl_id in certs.keys() {
        let Some((op, dtype)) = sweep_for(arch, impl_id) else {
            continue;
        };
        if runs.contains_key(&(op, dtype)) {
            continue;
        }
        println!("=== sweep --arch {arch} --op {op} --dtype {dtype}");
        let outcome = sweep_in_child(arch, op, dtype, out);
        if let Err(e) = &outcome {
            println!("    error: {e}");
        }
        runs.insert((op, dtype), outcome);
    }

    let report = cert_check(out, &dispatch_path)?;
    let mut check_failures: BTreeMap<&str, String> = BTreeMap::new();
    for (impl_id, e) in &report.failures {
        check_failures.entry(impl_id.as_str()).or_insert_with(|| format!("{e:#}"));
    }

    println!();
    println!("all-dispatch {backend}/{arch} -> {}", out.display());
    let mut bad = 0usize;
    for (impl_id, cert) in &certs {
        let verdict = match (sweep_for(arch, impl_id), check_failures.get(impl_id)) {
            (None, _) => {
                bad += 1;
                "NO SWEEP  no sweep invocation writes this cert".to_string()
            }
            (Some((op, dtype)), Some(why)) => {
                bad += 1;
                match &runs[&(op, dtype)] {
                    Err(e) => format!("FAIL      --op {op} --dtype {dtype}: {e}"),
                    Ok(()) => format!("FAIL      --op {op} --dtype {dtype}: {why}"),
                }
            }
            (Some(_), None) => {
                let shapes = Cert::read_from_disk(&out.join(cert))
                    .map(|c| c.results.len())
                    .unwrap_or(0);
                format!("PASS      shapes={shapes}")
            }
        };
        println!("  {impl_id:<48} {verdict}");
    }
    println!(
        "all-dispatch {backend}/{arch}: {} impls, {} pass, {bad} failed or without sweep",
        certs.len(),
        certs.len() - bad
    );
    if bad > 0 {
        anyhow::bail!("all-dispatch {backend}/{arch}: {bad} impl(s) failed or have no sweep");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn dispatched(backend: &str, arch: &str) -> BTreeSet<String> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(format!("dispatch/{backend}/{arch}.toml"));
        DispatchTable::load(&path)
            .unwrap()
            .active_rows()
            .map(|r| r.r#impl.clone())
            .collect()
    }

    #[test]
    fn every_dispatch_row_has_a_sweep() {
        for (backend, arch) in [("hip", "gfx906"), ("cuda", "sm_86")] {
            let missing: Vec<String> = dispatched(backend, arch)
                .into_iter()
                .filter(|i| sweep_for(arch, i).is_none())
                .collect();
            assert!(missing.is_empty(), "{arch}: rows without a sweep entry: {missing:?}");
        }
    }

    #[test]
    fn every_sweep_entry_is_dispatched() {
        for (backend, arch) in [("hip", "gfx906"), ("cuda", "sm_86")] {
            let rows = dispatched(backend, arch);
            let stale: Vec<&str> = DISPATCH_SWEEPS
                .iter()
                .filter(|&&(a, i, _, _)| a == arch && !rows.contains(i))
                .map(|&(_, i, _, _)| i)
                .collect();
            assert!(stale.is_empty(), "{arch}: entries for impls not in the dispatch table: {stale:?}");
        }
    }

    #[test]
    fn one_entry_per_impl() {
        let mut seen = BTreeSet::new();
        for (arch, impl_id, _, _) in DISPATCH_SWEEPS {
            assert!(seen.insert((arch, impl_id)), "duplicate entry {arch}/{impl_id}");
        }
    }
}
