//! Runtime PMC collection via rocprofv3 (ROCm 7.1.1 matched stack).
//!
//! `rocprofv3` launches the target binary with HSA-level interception,
//! records per-kernel counter values, writes a CSV like
//! `probe_counter_collection.csv`:
//!
//!   Correlation_Id,Dispatch_Id,Agent_Id,...,Kernel_Name,...,VGPR_Count,...,SGPR_Count,Counter_Name,Counter_Value,Start,End
//!   3,3,"Agent 1",...,"flambeau_mmvq_q8_0_q8_1",256,512,0,20,0,32,"MemUnitBusy",5.483540,...
//!
//! We parse this CSV, filter to our target kernel, and aggregate per counter
//! (one line per counter per dispatch). Every PMC snapshot captures both
//! static PMC (VGPR/SGPR are columns in the CSV itself — no separate
//! `hipFuncGetAttribute` call needed) and runtime counters.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

use crate::cert::PmcSnapshot;

const DEFAULT_COUNTERS: &[&str] = &["VALUBusy", "MemUnitBusy", "MemUnitStalled"];

/// One row of rocprofv3's counter-collection CSV.
#[derive(Debug, Clone)]
struct CounterRow {
    kernel_name: String,
    vgpr_count: u32,
    sgpr_count: u32,
    counter_name: String,
    counter_value: f64,
}

/// Run `cmd` (and its args) under rocprofv3, collect `counters`, return the
/// merged `PmcSnapshot` for the first invocation of `kernel_name`.
///
/// `rocprofv3_bin` should come from `$ROCPROFV3` (set by `.env`). The
/// target binary must already link against the matched ROCm 7.1.1 runtime
/// (our build.rs rpaths `$ROCM_PATH/{core-7.13/lib, lib}`).
pub fn capture_runtime_pmc(
    rocprofv3_bin: &Path,
    workdir: &Path,
    cmd: &Path,
    args: &[String],
    kernel_name: &str,
    counters: &[&str],
) -> Result<PmcSnapshot> {
    let outdir = workdir.join("rocprof_out");
    let _ = std::fs::remove_dir_all(&outdir);
    std::fs::create_dir_all(&outdir)?;

    // rocprofv3 expects the `pmc:` syntax in the input file (one `pmc:` line
    // per counter group, counters separated by spaces on the same line).
    let pmc_file = workdir.join("pmc.txt");
    std::fs::write(
        &pmc_file,
        format!("pmc: {}\n", counters.join(" ")),
    )?;

    let mut c = Command::new(rocprofv3_bin);
    c.args(["-i", pmc_file.to_str().unwrap()])
        .args(["-d", outdir.to_str().unwrap()])
        .args(["-o", "probe"])
        .args(["-f", "csv"])
        .arg("--");
    c.arg(cmd);
    for a in args {
        c.arg(a);
    }
    let output = c.output().context("spawn rocprofv3")?;
    if !output.status.success() {
        bail!(
            "rocprofv3 exited with status {}; stderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // The counter CSV lives under `outdir/pmc_<N>/probe_counter_collection.csv`
    // — rocprofv3 opens a numbered subdir per counter group. For one `pmc:`
    // line we expect exactly `pmc_1`.
    let csv = find_counter_csv(&outdir)?;
    let rows = parse_counter_csv(&csv)?;
    aggregate_pmc(&rows, kernel_name)
}

fn find_counter_csv(outdir: &Path) -> Result<PathBuf> {
    for sub in std::fs::read_dir(outdir)? {
        let sub = sub?;
        if sub.file_type()?.is_dir() {
            for f in std::fs::read_dir(sub.path())? {
                let f = f?;
                if f.file_name()
                    .to_string_lossy()
                    .ends_with("counter_collection.csv")
                {
                    return Ok(f.path());
                }
            }
        }
    }
    Err(anyhow!(
        "rocprofv3 produced no counter_collection.csv under {}",
        outdir.display()
    ))
}

fn parse_counter_csv(path: &Path) -> Result<Vec<CounterRow>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| anyhow!("empty rocprof CSV {}", path.display()))?;
    let cols: Vec<&str> = header.split(',').map(|c| c.trim_matches('"')).collect();
    let idx = |name: &str| -> Result<usize> {
        cols.iter()
            .position(|c| *c == name)
            .ok_or_else(|| anyhow!("rocprof CSV missing column `{name}` (got: {cols:?})"))
    };
    let i_kernel = idx("Kernel_Name")?;
    let i_vgpr = idx("VGPR_Count")?;
    let i_sgpr = idx("SGPR_Count")?;
    let i_counter_name = idx("Counter_Name")?;
    let i_counter_value = idx("Counter_Value")?;

    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = split_csv(line);
        let n = fields.len();
        // Bound-check so a malformed line yields a clear error instead of a panic.
        let max_idx = *[i_kernel, i_vgpr, i_sgpr, i_counter_name, i_counter_value]
            .iter()
            .max()
            .unwrap();
        if n <= max_idx {
            continue;
        }
        let kernel = fields[i_kernel].trim_matches('"').to_string();
        let vgpr: u32 = fields[i_vgpr].parse().unwrap_or(0);
        let sgpr: u32 = fields[i_sgpr].parse().unwrap_or(0);
        let cname = fields[i_counter_name].trim_matches('"').to_string();
        let cval: f64 = fields[i_counter_value].parse().unwrap_or(0.0);
        rows.push(CounterRow {
            kernel_name: kernel,
            vgpr_count: vgpr,
            sgpr_count: sgpr,
            counter_name: cname,
            counter_value: cval,
        });
    }
    Ok(rows)
}

fn split_csv(line: &str) -> Vec<&str> {
    // rocprofv3 CSV has a simple schema: all string fields are `"..."`-quoted
    // and contain no commas. A naive split is enough for our read path. We
    // still walk the string to tolerate the occasional stray field.
    let mut out = Vec::new();
    let mut in_quote = false;
    let bytes = line.as_bytes();
    let mut start = 0usize;
    for i in 0..bytes.len() {
        match bytes[i] {
            b'"' => in_quote = !in_quote,
            b',' if !in_quote => {
                out.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&line[start..]);
    out
}

fn aggregate_pmc(rows: &[CounterRow], kernel_name: &str) -> Result<PmcSnapshot> {
    // Filter rows belonging to this kernel. rocprofv3 may log multiple
    // dispatches if the kernel is called repeatedly (e.g. an A/B harness);
    // we average counter values across dispatches.
    let filtered: Vec<&CounterRow> = rows.iter().filter(|r| r.kernel_name == kernel_name).collect();
    if filtered.is_empty() {
        let seen: Vec<&str> = rows
            .iter()
            .map(|r| r.kernel_name.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        bail!(
            "kernel {kernel_name:?} never dispatched under rocprofv3 (saw: {seen:?})"
        );
    }

    let vgpr = filtered[0].vgpr_count;
    let sgpr = filtered[0].sgpr_count;

    let mean = |counter: &str| -> Option<f32> {
        let vals: Vec<f64> = filtered
            .iter()
            .filter(|r| r.counter_name == counter)
            .map(|r| r.counter_value)
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some((vals.iter().sum::<f64>() / vals.len() as f64) as f32)
        }
    };

    let mem_busy = mean("MemUnitBusy");
    let valu_busy = mean("VALUBusy");
    // gfx906 VGPR occupancy ceiling: min(10, 256/VGPR). Same derivation as
    // `FuncAttributes::gfx906_waves_per_simd`.
    let waves_per_simd = if vgpr == 0 { 10 } else { (256 / vgpr).min(10) };

    Ok(PmcSnapshot {
        vgpr_count: Some(vgpr),
        sgpr_count: Some(sgpr),
        waves_per_simd: Some(waves_per_simd),
        mem_busy_pct: mem_busy,
        valu_busy_pct: valu_busy,
    })
}

/// Convenience wrapper: read `$ROCPROFV3` env var and call
/// [`capture_runtime_pmc`] with the V1.3 default counter set.
pub fn capture_runtime_pmc_default(
    workdir: &Path,
    cmd: &Path,
    args: &[String],
    kernel_name: &str,
) -> Result<PmcSnapshot> {
    // Prefer the pinned binary from `.env` (ROCPROFV3), otherwise fall back
    // to `PATH` lookup. No hardcoded absolute fallbacks — ROCm's install
    // prefix varies per rig and committing one breaks CI/portability.
    let bin = std::env::var("ROCPROFV3")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("rocprofv3"));
    capture_runtime_pmc(&bin, workdir, cmd, args, kernel_name, DEFAULT_COUNTERS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rocprofv3_csv() {
        let csv = r#""Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id","Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"
1,1,"Agent 1",1,100,101,2048,17,"flambeau_mmvq_q8_0_q8_1",256,512,0,20,0,32,"MemUnitBusy",5.4,1,2
1,1,"Agent 1",1,100,101,2048,17,"flambeau_mmvq_q8_0_q8_1",256,512,0,20,0,32,"VALUBusy",0.4,1,2
1,1,"Agent 1",1,100,101,2048,17,"flambeau_mmvq_q8_0_q8_1",256,512,0,20,0,32,"MemUnitStalled",0.01,1,2
"#;
        let tmp = std::env::temp_dir().join(format!(
            "flambeau-pmc-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&tmp, csv).unwrap();
        let rows = parse_counter_csv(&tmp).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].kernel_name, "flambeau_mmvq_q8_0_q8_1");
        assert_eq!(rows[0].vgpr_count, 20);
        assert_eq!(rows[0].sgpr_count, 32);

        let pmc = aggregate_pmc(&rows, "flambeau_mmvq_q8_0_q8_1").unwrap();
        assert_eq!(pmc.vgpr_count, Some(20));
        assert_eq!(pmc.sgpr_count, Some(32));
        assert_eq!(pmc.waves_per_simd, Some(10));
        assert!((pmc.mem_busy_pct.unwrap() - 5.4).abs() < 1e-3);
        assert!((pmc.valu_busy_pct.unwrap() - 0.4).abs() < 1e-3);
        let _ = std::fs::remove_file(&tmp);
    }
}
