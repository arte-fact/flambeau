//! `flambeau` CLI — subcommand dispatch.
//!
//! V1.0 stub: subcommands parse but print a "not yet implemented" message until
//! their target step lands. See `doc/ROADMAP-V1-QWEN36-GFX906.md` for what each
//! subcommand requires.

use anyhow::Result;
use clap::{Parser, Subcommand};
use flambeau_quant::gguf::{GgufFile, Value};

#[derive(Parser)]
#[command(name = "flambeau", version, about = "Max-perf inference server for modern LLMs on HIP + CUDA.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump GGUF tensor list, dtype audit, metadata (V1.1).
    InspectGguf {
        path: String,
    },
    /// Dump HIP `.hsaco` kernel symbols + VGPR budgets (V1.3+).
    InspectHsaco {
        path: String,
    },
    /// Single-prompt inference; prints completion to stdout (V1.7).
    Infer {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Comma-separated device list, e.g. `hip:0,1,2,3`.
        #[arg(long, default_value = "hip:0")]
        devices: String,
    },
    /// OpenAI-compatible HTTP server (V1.8).
    Serve {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "hip:0")]
        devices: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// T-track warmup-tuner (V1.x side-track).
    Tune {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "hip:0")]
        devices: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// M-track MCP server (V1.x side-track). Dev-only.
    Mcp {
        #[arg(long, default_value_t = 9090)]
        port: u16,
    },
    /// Correctness-sweep harness; emits certs (V1.3+).
    Sweep {
        #[arg(long)]
        arch: String,
        #[arg(long)]
        op: Option<String>,
        /// Weight dtype to sweep ("Q8_0", "Q4_K", "Q5_K", "Q6_K", "all").
        #[arg(long, default_value = "all")]
        dtype: String,
    },
    /// Validate that every dispatch row has a matching green cert (V1.3+).
    CertCheck {
        #[arg(long, default_value = "gfx906")]
        arch: String,
        #[arg(long, default_value = "hip")]
        backend: String,
    },
    /// One-shot MMVQ/MMQ kernel launch — used by the rocprofv3 PMC wrapper.
    /// Exits immediately after the launch + sync, so rocprofv3 sees a clean
    /// per-kernel dispatch trace. Not intended for human use.
    PmcProbe {
        /// Kernel stem: `mmvq_q8_0` / `mmq_q8_0_4warp` / etc.
        #[arg(long)]
        kernel: String,
        #[arg(long, default_value_t = 4)]
        m: usize,
        #[arg(long, default_value_t = 2048)]
        k: usize,
        #[arg(long, default_value_t = 2048)]
        n: usize,
    },
    /// Run the full sweep + rocprofv3 PMC capture for each shape and attach
    /// the counters to the committed cert JSON. Requires `.env` with
    /// `ROCPROFV3=.../rocprofv3` pointing at the matched 7.1.1 binary.
    PmcRefresh {
        #[arg(long, default_value = "gfx906")]
        arch: String,
    },
    /// Perf-regression matrix (V1.4+).
    Matrix {
        #[arg(long)]
        models: Vec<String>,
        #[arg(long, default_value_t = 512)]
        prompt_len: usize,
        #[arg(long, default_value_t = 64)]
        tg_len: usize,
    },
}

fn main() -> Result<()> {
    // Load `.env` from the workspace root so `flambeau sweep` picks up
    // ROCM_PATH / ROCPROFV3 / LD_LIBRARY_PATH without shell sourcing.
    // Missing `.env` is not an error — running outside the repo is fine.
    load_env_file(".env");

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::InspectGguf { path } => inspect_gguf(&path)?,
        Cmd::InspectHsaco { path } => todo!("V1.3+: implement inspect-hsaco for {path}"),
        Cmd::Infer { model, .. } => todo!("V1.7: implement infer for {model}"),
        Cmd::Serve { model, devices, port } => serve_cmd(&model, &devices, port)?,
        Cmd::Tune { model, .. } => todo!("T-track: implement tune for {model}"),
        Cmd::Mcp { port } => todo!("M-track: implement mcp on :{port}"),
        Cmd::Sweep { arch, op, dtype } => sweep(&arch, op.as_deref(), &dtype)?,
        Cmd::CertCheck { arch, backend } => cert_check(&backend, &arch)?,
        Cmd::PmcProbe { kernel, m, k, n } => pmc_probe(&kernel, m, k, n)?,
        Cmd::PmcRefresh { arch } => pmc_refresh(&arch)?,
        Cmd::Matrix { .. } => todo!("V1.4+: implement matrix"),
    }
    Ok(())
}

#[cfg(not(feature = "hip_serve"))]
fn serve_cmd(_model: &str, _devices: &str, _port: u16) -> Result<()> {
    anyhow::bail!(
        "`flambeau serve` requires building with --features hip_serve (needs ROCm + HIP devices)"
    );
}

#[cfg(feature = "hip_serve")]
fn serve_cmd(model: &str, devices: &str, port: u16) -> Result<()> {
    use std::net::SocketAddr;
    use std::path::PathBuf;

    // Parse `hip:0,1,2,3` or `0,1,2,3` → Vec<i32>.
    let dev_str = devices.strip_prefix("hip:").unwrap_or(devices);
    let device_ids: Vec<i32> = dev_str
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<i32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("--devices parse error: {e}"))?;
    if device_ids.is_empty() {
        anyhow::bail!("--devices must list at least one device ID");
    }

    let bind_addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    let cfg = flambeau_server::ServeConfig {
        gguf_path: PathBuf::from(model),
        device_ids,
        bind_addr,
        model_id: PathBuf::from(model)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "flambeau".to_string()),
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(flambeau_server::serve(cfg))
}

fn inspect_gguf(path: &str) -> Result<()> {
    let file = GgufFile::open(path)?;
    println!("path: {}", file.path.display());
    println!("version: {:?}", file.version);
    println!("tensor_data_offset: {}", file.tensor_data_offset);
    println!("metadata ({} entries):", file.metadata.len());
    let mut kv: Vec<(&String, &Value)> = file.metadata.iter().collect();
    kv.sort_by(|a, b| a.0.cmp(b.0));
    for (k, v) in kv {
        println!("  {k} = {}", format_value(v));
    }
    println!("tensors ({}):", file.tensors.len());
    for name in &file.tensor_order {
        let info = &file.tensors[name];
        let dims: Vec<String> = info.dims.iter().map(|d| d.to_string()).collect();
        println!(
            "  {name:<56} {:<5} [{}] off={} size={}",
            info.dtype.name(),
            dims.join(","),
            info.rel_offset,
            info.size_in_bytes()
        );
    }
    Ok(())
}

/// Read `KEY=VALUE` pairs from `path` and set them on `std::env` unless
/// they're already defined. Lines starting with `#` and blank lines are
/// ignored. Quotes around values are stripped. No substitution — this is a
/// minimum-viable loader, not a dotenv replacement.
fn load_env_file(path: &str) {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').trim_matches('\'');
        if std::env::var_os(k).is_none() {
            // SAFETY: set_var is unsafe-marked in recent Rust nightlies for
            // thread-safety reasons; we call it before any thread is spawned
            // (first thing in main), so it's safe here.
            unsafe { std::env::set_var(k, v) };
        }
    }
}

fn repo_root() -> std::path::PathBuf {
    // Workspace root is the parent of `crates/*`. We anchor via CARGO_MANIFEST_DIR
    // of the CLI crate at build time; at runtime we just use the current dir
    // since `flambeau sweep` is expected to run from the workspace root.
    std::env::current_dir().expect("cwd")
}

fn sweep(arch: &str, op: Option<&str>, dtype: &str) -> Result<()> {
    let op = op.unwrap_or("qmatmul");
    if arch != "gfx906" {
        anyhow::bail!("only --arch gfx906 is implemented in V1.3+ (got {arch:?})");
    }

    #[cfg(feature = "hip_sweep")]
    {
        match op {
            "qmatmul" => {
                use flambeau_bench::sweep_mmvq::{run_sweep, Dtype, SweepSpec};
                let dtypes: Vec<Dtype> = match dtype {
                    "all" => vec![Dtype::Q8_0, Dtype::Q4K, Dtype::Q5K, Dtype::Q6K],
                    "all-multirow" => vec![Dtype::Q4KR2, Dtype::Q5KR2, Dtype::Q6KR4],
                    "Q8_0" => vec![Dtype::Q8_0],
                    "Q4_K" => vec![Dtype::Q4K],
                    "Q5_K" => vec![Dtype::Q5K],
                    "Q6_K" => vec![Dtype::Q6K],
                    "Q4_K_r2" => vec![Dtype::Q4KR2],
                    "Q5_K_r2" => vec![Dtype::Q5KR2],
                    "Q6_K_r4" => vec![Dtype::Q6KR4],
                    "Q6_K_dp4a" => vec![Dtype::Q6KDP4A],
                    "Q4_1" => vec![Dtype::Q4_1],
                    other => anyhow::bail!("unknown dtype {other}"),
                };
                let root = repo_root();
                for d in dtypes {
                    let spec = SweepSpec::v1_3_default(d);
                    let cert = run_sweep(&spec, &root)?;
                    println!(
                        "sweep {} ({}): pass={} shapes={} rig={}",
                        op,
                        d.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                }
                Ok(())
            }
            "qmatmul_mmq" => {
                use flambeau_bench::sweep_mmq::{run_sweep, Dtype as MmqDtype, SweepSpec};
                let (dtypes, spec_builder): (Vec<MmqDtype>, fn(MmqDtype) -> SweepSpec) = match dtype {
                    "Q8_0" | "Q8_0_oracle" => (vec![MmqDtype::Q8_0Oracle], SweepSpec::v1_4_oracle),
                    "Q8_0_4warp" => (vec![MmqDtype::Q8_04Warp], SweepSpec::v1_4_prefill),
                    "Q8_0_wave64" => (vec![MmqDtype::Q8_0Wave64], SweepSpec::v1_4_prefill),
                    "Q8_0_wave64_tile16" => {
                        (vec![MmqDtype::Q8_0Wave64Tile16], SweepSpec::v1_4_prefill)
                    }
                    "Q4_1" | "Q4_1_4warp" => (vec![MmqDtype::Q4_14Warp], SweepSpec::v1_4_prefill),
                    "Q4_1_wave64" => (vec![MmqDtype::Q4_1Wave64], SweepSpec::v1_4_prefill),
                    "Q4_K" | "Q4_K_4warp" => (vec![MmqDtype::Q4K4Warp], SweepSpec::v1_4_prefill),
                    "Q4_K_wave64" => (vec![MmqDtype::Q4KWave64], SweepSpec::v1_4_prefill),
                    "Q4_K_turbo" => (vec![MmqDtype::Q4KTurbo], SweepSpec::v1_4_prefill),
                    "Q5_K" | "Q5_K_wave64" => (vec![MmqDtype::Q5KWave64], SweepSpec::v1_4_prefill),
                    "Q6_K" | "Q6_K_4warp" => (vec![MmqDtype::Q6K4Warp], SweepSpec::v1_4_prefill),
                    "Q6_K_wave64" => (vec![MmqDtype::Q6KWave64], SweepSpec::v1_4_prefill),
                    "all" => (
                        vec![
                            MmqDtype::Q8_0Oracle,
                            MmqDtype::Q8_04Warp,
                            MmqDtype::Q8_0Wave64,
                            MmqDtype::Q8_0Wave64Tile16,
                            MmqDtype::Q4_14Warp,
                            MmqDtype::Q4_1Wave64,
                            MmqDtype::Q4K4Warp,
                            MmqDtype::Q4KWave64,
                            MmqDtype::Q5KWave64,
                            MmqDtype::Q6K4Warp,
                            MmqDtype::Q6KWave64,
                        ],
                        // Oracle uses its small grid, 4warp uses the prefill grid.
                        |d| match d {
                            MmqDtype::Q8_0Oracle => SweepSpec::v1_4_oracle(d),
                            MmqDtype::Q8_04Warp
                            | MmqDtype::Q8_0Wave64
                            | MmqDtype::Q8_0Wave64Tile16
                            | MmqDtype::Q4_14Warp
                            | MmqDtype::Q4_1Wave64
                            | MmqDtype::Q4KTurbo
                            | MmqDtype::Q4K4Warp
                            | MmqDtype::Q4KWave64
                            | MmqDtype::Q5KWave64
                            | MmqDtype::Q6K4Warp
                            | MmqDtype::Q6KWave64 => SweepSpec::v1_4_prefill(d),
                        },
                    ),
                    other => anyhow::bail!("unknown MMQ dtype {other}"),
                };
                let root = repo_root();
                for d in dtypes {
                    let spec = spec_builder(d);
                    let cert = run_sweep(&spec, &root)?;
                    println!(
                        "sweep {} ({} {}): pass={} shapes={} rig={}",
                        op,
                        d.name(),
                        d.impl_id_short(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                }
                Ok(())
            }
            "rmsnorm" => {
                use flambeau_bench::sweep_rmsnorm::{run_sweep, SweepSpec};
                let _ = dtype;
                let spec = SweepSpec::v1_6_rmsnorm();
                let cert = run_sweep(&spec, &repo_root())?;
                println!(
                    "sweep rmsnorm: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "swiglu" => {
                use flambeau_bench::sweep_swiglu::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep swiglu: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "rmsnorm_q8_1" => {
                use flambeau_bench::sweep_rmsnorm_q8_1::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep rmsnorm_q8_1: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "quantize_q8_1_mmq" => {
                use flambeau_bench::sweep_quantize_q8_1_mmq::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep quantize_q8_1_mmq: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "attention_prefill_flash_tile" => {
                use flambeau_bench::sweep_attention_prefill::run_sweep_flash_tile;
                let _ = dtype;
                let cert = run_sweep_flash_tile(&repo_root())?;
                println!(
                    "sweep attention_prefill_flash_tile: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "rope" => {
                use flambeau_bench::sweep_rope::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep rope: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "rope_neox_partial" => {
                use flambeau_bench::sweep_rope_neox::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep rope_neox_partial: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "l2_norm" => {
                use flambeau_bench::sweep_l2_norm::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep l2_norm: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "causal_conv1d" => {
                use flambeau_bench::sweep_causal_conv1d::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep causal_conv1d: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "gdn_state_step" => {
                use flambeau_bench::sweep_gdn_step::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep gdn_state_step: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "cast_f32_f16" => {
                use flambeau_bench::sweep_cast::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep cast_f32_f16: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "cast_f16_f32" => {
                use flambeau_bench::sweep_f32_pointwise::run_cast_f16_f32_sweep;
                let _ = dtype;
                let cert = run_cast_f16_f32_sweep(&repo_root())?;
                println!(
                    "sweep cast_f16_f32: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "silu_f32" => {
                use flambeau_bench::sweep_f32_pointwise::run_silu_sweep;
                let _ = dtype;
                let cert = run_silu_sweep(&repo_root())?;
                println!(
                    "sweep silu_f32: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "swiglu_f32" => {
                use flambeau_bench::sweep_f32_pointwise::run_swiglu_sweep;
                let _ = dtype;
                let cert = run_swiglu_sweep(&repo_root())?;
                println!(
                    "sweep swiglu_f32: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "scale_f32" => {
                use flambeau_bench::sweep_f32_pointwise::run_scale_sweep;
                let _ = dtype;
                let cert = run_scale_sweep(&repo_root())?;
                println!(
                    "sweep scale_f32: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "rmsnorm_f32" => {
                use flambeau_bench::sweep_f32_pointwise::run_rmsnorm_f32_sweep;
                let _ = dtype;
                let cert = run_rmsnorm_f32_sweep(&repo_root())?;
                println!(
                    "sweep rmsnorm_f32: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "gdn_alpha_beta" => {
                use flambeau_bench::sweep_f32_pointwise::run_gdn_alpha_beta_sweep;
                let _ = dtype;
                let cert = run_gdn_alpha_beta_sweep(&repo_root())?;
                println!(
                    "sweep gdn_alpha_beta: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "quantize_f16_q8_1" => {
                use flambeau_bench::sweep_f32_pointwise::run_quantize_f16_q8_1_sweep;
                let _ = dtype;
                let cert = run_quantize_f16_q8_1_sweep(&repo_root())?;
                println!(
                    "sweep quantize_f16_q8_1: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "dense_gemv_f32_f16" => {
                use flambeau_bench::sweep_f32_pointwise::run_dense_gemv_sweep;
                let _ = dtype;
                let cert = run_dense_gemv_sweep(&repo_root())?;
                println!(
                    "sweep dense_gemv_f32_f16: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "add_f16" => {
                use flambeau_bench::sweep_f32_pointwise::run_add_f16_sweep;
                let _ = dtype;
                let cert = run_add_f16_sweep(&repo_root())?;
                println!(
                    "sweep add_f16: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "peer_copy_via_host" => {
                use flambeau_bench::sweep_peer_copy::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep peer_copy_via_host: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "shared_expert_scale" => {
                use flambeau_bench::sweep_shared_expert::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep shared_expert_scale: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "split_q_gate" => {
                use flambeau_bench::sweep_split_q_gate::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep split_q_gate: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "softmax" => {
                use flambeau_bench::sweep_softmax::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep softmax: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "attention_decode" => {
                use flambeau_bench::sweep_attention::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep attention_decode: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "attention_prefill" => {
                use flambeau_bench::sweep_attention_prefill::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep attention_prefill: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "attention_decode_q8_kv" => {
                use flambeau_bench::sweep_attention_q8_kv::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep attention_decode_q8_kv: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "attention_decode_splitk" => {
                use flambeau_bench::sweep_attention_splitk::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep attention_decode_splitk: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "mmvq_f16" => {
                use flambeau_bench::sweep_mmvq_f16::run_sweep;
                let _ = dtype;
                let cert = run_sweep(&repo_root())?;
                println!(
                    "sweep mmvq_f16: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "mmq_f16" => {
                use flambeau_bench::sweep_mmvq_f16::run_mmq_sweep;
                let _ = dtype;
                let cert = run_mmq_sweep(&repo_root())?;
                println!(
                    "sweep mmq_f16: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "mmvq_q4_0" => {
                use flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q4_0_sweep;
                let _ = dtype;
                let cert = run_mmvq_q4_0_sweep(&repo_root())?;
                println!("sweep mmvq_q4_0: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig);
                Ok(())
            }
            "mmvq_q5_0" => {
                use flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q5_0_sweep;
                let _ = dtype;
                let cert = run_mmvq_q5_0_sweep(&repo_root())?;
                println!("sweep mmvq_q5_0: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig);
                Ok(())
            }
            "mmvq_q5_1" => {
                use flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q5_1_sweep;
                let _ = dtype;
                let cert = run_mmvq_q5_1_sweep(&repo_root())?;
                println!("sweep mmvq_q5_1: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig);
                Ok(())
            }
            "indexed_moe_mmvq_q4_0" => {
                use flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q4_0_sweep;
                let _ = dtype;
                let cert = run_indexed_moe_mmvq_q4_0_sweep(&repo_root())?;
                println!("sweep indexed_moe_mmvq_q4_0: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig);
                Ok(())
            }
            "topk" => {
                use flambeau_bench::sweep_moe::run_topk_sweep;
                let _ = dtype;
                let cert = run_topk_sweep(&repo_root())?;
                println!(
                    "sweep topk: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "indexed_moe_mmvq" => {
                use flambeau_bench::sweep_moe::run_indexed_moe_mmvq_sweep;
                let _ = dtype;
                let cert = run_indexed_moe_mmvq_sweep(&repo_root())?;
                println!(
                    "sweep indexed_moe_mmvq: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "moe_combine" => {
                use flambeau_bench::sweep_moe::run_moe_combine_sweep;
                let _ = dtype;
                let cert = run_moe_combine_sweep(&repo_root())?;
                println!(
                    "sweep moe_combine: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "indexed_moe_mmvq_gate_up" => {
                use flambeau_bench::sweep_moe::run_gate_up_sweep;
                let _ = dtype;
                let cert = run_gate_up_sweep(&repo_root())?;
                println!(
                    "sweep indexed_moe_mmvq_gate_up: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "indexed_moe_mmvq_r2" => {
                use flambeau_bench::sweep_moe::run_indexed_moe_mmvq_r2_sweep;
                let _ = dtype;
                let cert = run_indexed_moe_mmvq_r2_sweep(&repo_root())?;
                println!(
                    "sweep indexed_moe_mmvq_r2: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "indexed_moe_mmvq_q6_k" => {
                use flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q6_k_sweep;
                let _ = dtype;
                let cert = run_indexed_moe_mmvq_q6_k_sweep(&repo_root())?;
                println!(
                    "sweep indexed_moe_mmvq_q6_k: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "indexed_moe_mmvq_q8_0" => {
                use flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q8_0_sweep;
                let _ = dtype;
                let cert = run_indexed_moe_mmvq_q8_0_sweep(&repo_root())?;
                println!(
                    "sweep indexed_moe_mmvq_q8_0: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            "indexed_moe_mmq" => {
                use flambeau_bench::sweep_moe::run_indexed_moe_mmq_sweep;
                let _ = dtype;
                let cert = run_indexed_moe_mmq_sweep(&repo_root())?;
                println!(
                    "sweep indexed_moe_mmq: pass={} shapes={} rig={}",
                    cert.pass, cert.results.len(), cert.rig
                );
                Ok(())
            }
            other => anyhow::bail!("unknown --op {other} (qmatmul | qmatmul_mmq | rmsnorm)"),
        }
    }
    #[cfg(not(feature = "hip_sweep"))]
    {
        let _ = (dtype, op);
        anyhow::bail!("rebuild with --features hip_sweep to enable sweep");
    }
}

fn pmc_probe(kernel: &str, m: usize, k: usize, n: usize) -> Result<()> {
    #[cfg(feature = "hip_sweep")]
    {
        let entry = flambeau_bench::pmc_probe::run_one(kernel, m, k, n)?;
        println!("probe ok: entry={entry} m={m} k={k} n={n}");
        Ok(())
    }
    #[cfg(not(feature = "hip_sweep"))]
    {
        let _ = (kernel, m, k, n);
        anyhow::bail!("rebuild with --features hip_sweep");
    }
}

fn pmc_refresh(arch: &str) -> Result<()> {
    #[cfg(feature = "hip_sweep")]
    {
        use flambeau_bench::cert::Cert;
        use flambeau_bench::pmc::capture_runtime_pmc_default;

        if arch != "gfx906" {
            anyhow::bail!("only gfx906 supported today");
        }

        // Probe binary is the CLI itself in `pmc-probe` mode — keeps build
        // deps tight (no extra bin target).
        let this_exe = std::env::current_exe()?;
        let workdir = std::env::temp_dir().join(format!(
            "flambeau-pmc-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&workdir)?;

        // The set of (kernel_stem, representative shape) we refresh per cert.
        // Keep shapes small — rocprofv3 is slow (seconds per dispatch group).
        let targets: &[(&str, &str, usize, usize, usize)] = &[
            // (impl_id, kernel_stem, m, k, n)
            ("qmatmul_q8_0_mmvq_single_row_gfx906", "mmvq_q8_0", 1, 2048, 64),
            ("qmatmul_q4_K_mmvq_single_row_gfx906", "mmvq_q4_k", 1, 2048, 64),
            ("qmatmul_q5_K_mmvq_single_row_gfx906", "mmvq_q5_k", 1, 2048, 64),
            ("qmatmul_q6_K_mmvq_single_row_gfx906", "mmvq_q6_k", 1, 2048, 64),
            ("qmatmul_q4_K_mmvq_nw1_r2_gfx906", "mmvq_q4_k_r2", 1, 2048, 64),
            ("qmatmul_q5_K_mmvq_nw1_r2_gfx906", "mmvq_q5_k_r2", 1, 2048, 64),
            ("qmatmul_q6_K_mmvq_nw1_r4_gfx906", "mmvq_q6_k_r4", 1, 2048, 64),
            ("qmatmul_q8_0_mmq_oracle_gfx906", "mmq_q8_0_oracle", 8, 2048, 64),
            ("qmatmul_q8_0_mmq_4warp_lds_gfx906", "mmq_q8_0_4warp", 128, 2048, 64),
            ("qmatmul_q8_0_mmq_wave64_gfx906", "mmq_q8_0_wave64", 128, 2048, 4096),
            ("qmatmul_q4_1_mmq_4warp_lds_gfx906", "mmq_q4_1_4warp_lds", 128, 2048, 4096),
            ("qmatmul_q4_K_mmq_wave64_gfx906", "mmq_q4_K_wave64", 128, 2048, 4096),
            ("qmatmul_q5_K_mmq_wave64_gfx906", "mmq_q5_K_wave64", 128, 2048, 4096),
            ("qmatmul_q6_K_mmq_wave64_gfx906", "mmq_q6_K_wave64", 128, 2048, 4096),
        ];

        for &(impl_id, stem, m, k, n) in targets {
            let cert_path = repo_root()
                .join("certs/hip/gfx906")
                .join(format!("{impl_id}.json"));
            if !cert_path.exists() {
                println!("  skip {impl_id}: cert not found");
                continue;
            }
            let mut cert = Cert::read_from_disk(&cert_path)?;

            // Launch the probe via self-exec so the profiler sees a clean
            // one-kernel process lifecycle. No other sweep noise.
            let args = vec![
                "pmc-probe".to_string(),
                "--kernel".to_string(),
                stem.to_string(),
                "--m".to_string(),
                m.to_string(),
                "--k".to_string(),
                k.to_string(),
                "--n".to_string(),
                n.to_string(),
            ];
            // Entry name is the symbol rocprofv3 logs in Kernel_Name.
            let entry = kernel_entry(stem)?;
            let pmc = match capture_runtime_pmc_default(&workdir, &this_exe, &args, entry) {
                Ok(p) => p,
                Err(e) => {
                    println!("  fail {impl_id}: {e}");
                    continue;
                }
            };
            cert.pmc = Some(pmc.clone());
            let root = repo_root();
            cert.write_to_disk(&root)?;
            println!(
                "  {impl_id}: vgpr={} sgpr={:?} waves/SIMD={} mem_busy={:?}% valu_busy={:?}%",
                pmc.vgpr_count.unwrap_or(0),
                pmc.sgpr_count,
                pmc.waves_per_simd.unwrap_or(0),
                pmc.mem_busy_pct,
                pmc.valu_busy_pct,
            );
        }
        Ok(())
    }
    #[cfg(not(feature = "hip_sweep"))]
    {
        let _ = arch;
        anyhow::bail!("rebuild with --features hip_sweep");
    }
}

#[cfg(feature = "hip_sweep")]
fn kernel_entry(stem: &str) -> Result<&'static str> {
    Ok(match stem {
        "mmvq_q8_0" => "flambeau_mmvq_q8_0_q8_1",
        "mmvq_q4_k" => "flambeau_mmvq_q4_k_q8_1",
        "mmvq_q5_k" => "flambeau_mmvq_q5_k_q8_1",
        "mmvq_q6_k" => "flambeau_mmvq_q6_k_q8_1",
        "mmvq_q4_k_r2" => "flambeau_mmvq_q4_k_r2_q8_1",
        "mmvq_q5_k_r2" => "flambeau_mmvq_q5_k_r2_q8_1",
        "mmvq_q6_k_r4" => "flambeau_mmvq_q6_k_r4_q8_1",
        "mmq_q8_0_oracle" => "flambeau_mmq_q8_0_oracle_q8_1",
        "mmq_q8_0_4warp" => "flambeau_mmq_q8_0_4warp_q8_1",
        "mmq_q8_0_wave64" => "flambeau_mmq_q8_0_wave64_q8_1",
        "mmq_q4_1_4warp_lds" => "flambeau_mmq_q4_1_4warp_lds_q8_1",
        "mmq_q4_K_wave64" => "flambeau_mmq_q4_K_wave64_q8_1",
        "mmq_q5_K_wave64" => "flambeau_mmq_q5_K_wave64_q8_1",
        "mmq_q6_K_wave64" => "flambeau_mmq_q6_K_wave64_q8_1",
        other => anyhow::bail!("unknown kernel stem {other}"),
    })
}

fn cert_check(backend: &str, arch: &str) -> Result<()> {
    use flambeau_bench::dispatch::cert_check as do_check;
    let dispatch_path = repo_root().join(format!("dispatch/{backend}/{arch}.toml"));
    let report = do_check(&repo_root(), &dispatch_path)?;
    println!(
        "cert-check {backend}/{arch}: {} rows, {} failures",
        report.rows_checked,
        report.failures.len()
    );
    for (impl_id, err) in &report.failures {
        println!("  FAIL {impl_id}: {err}");
    }
    if !report.ok() {
        anyhow::bail!("cert-check failed — {} row(s) missing or stale", report.failures.len());
    }
    Ok(())
}

fn format_value(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
        Value::String(s) if s.len() <= 160 => format!("{s:?}"),
        Value::String(s) => format!("{:?}…({} bytes)", &s[..160.min(s.len())], s.len()),
        Value::Array(a) if a.len() <= 8 => {
            let parts: Vec<String> = a.iter().map(format_value).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Array(a) => {
            let head: Vec<String> = a.iter().take(4).map(format_value).collect();
            format!("[{}, …({} elems)]", head.join(", "), a.len())
        }
    }
}
