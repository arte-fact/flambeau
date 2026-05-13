//! `flambeau` CLI — subcommand dispatch.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use flambeau_quant::gguf::{GgufFile, Value};

/// Parse a "loose" boolean from CLI / env. Accepts the conventional
/// truthy/falsy strings (`1`, `0`, `true`, `false`, `yes`, `no`,
/// `on`, `off`) case-insensitively. Pre-S5 the env-var booleans
/// triggered on any-value-set; clap's strict `bool` parser broke
/// that contract for operators with existing `FLAMBEAU_X=1` scripts.
fn parse_bool_loose(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        other => Err(format!(
            "invalid boolean `{other}` (accepts 1/0, true/false, yes/no, on/off)"
        )),
    }
}

#[derive(Parser)]
#[command(name = "flambeau", version, about = "Max-perf inference server for modern LLMs on HIP + CUDA.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump GGUF tensor list, dtype audit, metadata ().
    InspectGguf {
        path: String,
    },
    /// Dump the GGUF-embedded Jinja chat template to a `.jinja` file.
    /// Used by `certs/chat_template/qwen35moe_tools/regenerate.sh` and as
    /// a general utility for anyone wanting to feed the model's actual
    /// chat template to an external Jinja renderer (e.g. llama.cpp's
    /// `test-chat-template`). Non-destructive — prints to stdout when
    /// `--out` is omitted.
    ExtractChatTemplate {
        /// Path to the GGUF file.
        #[arg(long)]
        path: String,
        /// Optional output file. Default: stdout.
        #[arg(long)]
        out: Option<String>,
    },
    /// Dump HIP `.hsaco` kernel symbols + VGPR budgets (+).
    InspectHsaco {
        path: String,
    },
    /// Single-prompt inference; prints completion to stdout ().
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
    /// OpenAI-compatible HTTP server () + optional MCP upstream
    /// client (7 — ROADMAP-V2 §M2.1).
    Serve {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "hip:0")]
        devices: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// / mesh topology:
        /// - `pp` — pipeline-parallel (V1 default, LayerAssignment-based).
        /// - `tp` — tensor-parallel (Megatron-style per-tensor sharding
        /// with BAR1 P2P AllReduce).
        /// - `pp+tp` (alias `hybrid`) — manual PP-of-TP. Requires both
        /// `--pp-size` and `--tp-size`; `pp_size * tp_size` must equal
        /// `--devices` count. Devices are interpreted in stage-major
        /// order. flambeau does not autodetect the right topology —
        /// pick one with the bracket-bench harness.
        #[arg(long = "mesh-mode", default_value = "pp")]
        mesh_mode: String,
        /// TP world size when `--mesh-mode tp`, or per-stage
        /// TP size when `--mesh-mode pp+tp`. Must equal `--devices` count
        /// (tp) or `--devices count / --pp-size` (pp+tp). Ignored for
        /// `--mesh-mode pp` (which uses `--devices` count as the PP rank
        /// count).
        #[arg(long = "tp-size", default_value_t = 0)]
        tp_size: u32,
        /// number of pipeline stages when
        /// `--mesh-mode pp+tp`. Must divide `num_layers` and satisfy
        /// `pp_size * tp_size == --devices count`. Ignored for `pp`/`tp`.
        #[arg(long = "pp-size", default_value_t = 0)]
        pp_size: u32,
        /// **#230 P2.11a** — optional path to a `qwen3` arch embedding
        /// GGUF (e.g. `Qwen3-Embedding-0.6B-Q8_0.gguf`). Loaded
        /// alongside the chat model on a single device; powers the
        /// `/v1/embeddings` endpoint (#231). Omit to disable embeddings.
        #[arg(long = "embedding-model")]
        embedding_model: Option<String>,
        /// **#230 P2.11a** — HIP device for the embedding model
        /// (numeric, e.g. `0`). Defaults to the first device in
        /// `--devices`. The embedding model shares VRAM with whatever
        /// chat-model rank lives on the same device; a 600 M / 4 B
        /// embedding fits alongside a 27 B chat shard on 16 GB MI50.
        #[arg(long = "embedding-device")]
        embedding_device: Option<i32>,
        /// Number of concurrent inflight decode slots (1-32).
        #[arg(long = "inflight-slots", env = "FLAMBEAU_INFLIGHT_SLOTS", default_value_t = 4)]
        inflight_slots: usize,
        /// Prefill chunk size (tokens per forward step).
        #[arg(long = "prefill-ubatch", env = "FLAMBEAU_PREFILL_UBATCH", default_value_t = 512)]
        prefill_ubatch: usize,
        /// Admission-control queue depth beyond `inflight_slots` before
        /// returning 503 + Retry-After. 0 disables (legacy).
        #[arg(long = "max-queue-depth", env = "FLAMBEAU_MAX_QUEUE_DEPTH", default_value_t = 16)]
        max_queue_depth: usize,
        /// Clamp the model's `context_length` to this many tokens.
        /// Useful for preventing per-slot KV-cache OOM on consumer VRAM.
        /// Only shrinks; explicit increases are ignored.
        #[arg(long = "ctx-cap", env = "FLAMBEAU_CTX_CAP")]
        ctx_cap: Option<usize>,
        /// Disable the on-device GPU sampler (default: enabled).
        #[arg(
            long = "no-gpu-sampler",
            env = "FLAMBEAU_NO_GPU_SAMPLER",
            value_parser = parse_bool_loose,
            num_args = 0..=1,
            default_value = "false",
            default_missing_value = "true",
        )]
        no_gpu_sampler: bool,
        /// Disable the batched-decode scheduler (default: enabled).
        #[arg(
            long = "no-batched-decode",
            env = "FLAMBEAU_NO_BATCHED_DECODE",
            value_parser = parse_bool_loose,
            num_args = 0..=1,
            default_value = "false",
            default_missing_value = "true",
        )]
        no_batched_decode: bool,
        /// Enable prompt prefix cache (chat workloads).
        #[arg(
            long = "prefix-cache",
            env = "FLAMBEAU_PREFIX_CACHE",
            value_parser = parse_bool_loose,
            num_args = 0..=1,
            default_value = "false",
            default_missing_value = "true",
        )]
        prefix_cache: bool,
        /// Prefix-cache LRU size in GB. Only used when --prefix-cache is set.
        #[arg(long = "prefix-cache-max-gb", env = "FLAMBEAU_PREFIX_CACHE_MAX_GB", default_value_t = 2.0)]
        prefix_cache_max_gb: f64,
        /// KV cache layout. `f16` (default, canonical) or `q8` (~2× HBM
        /// saving on decode; quality cert required per model).
        #[arg(long = "kv", env = "FLAMBEAU_KV", default_value = "f16")]
        kv: String,
        /// Default system prompt prepended to chat-template requests
        /// when none is provided in the request.
        #[arg(long = "default-system", env = "FLAMBEAU_DEFAULT_SYSTEM")]
        default_system: Option<String>,
        /// /v1/embeddings endpoint per-prompt token cap.
        #[arg(long = "embedding-max-tokens", env = "FLAMBEAU_EMBEDDING_MAX_TOKENS", default_value_t = 8192)]
        embedding_max_tokens: usize,
    },
    /// Correctness-sweep harness; emits certs (+).
    Sweep {
        #[arg(long)]
        arch: String,
        #[arg(long)]
        op: Option<String>,
        /// Weight dtype to sweep ("Q8_0", "Q4_K", "Q5_K", "Q6_K", "all").
        #[arg(long, default_value = "all")]
        dtype: String,
    },
    /// Validate that every dispatch row has a matching green cert (+).
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
    /// Perf-regression matrix (+).
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
        Cmd::ExtractChatTemplate { path, out } => extract_chat_template(&path, out.as_deref())?,
        Cmd::InspectHsaco { path } => todo!("implement inspect-hsaco for {path}"),
        Cmd::Infer { model, .. } => todo!("implement infer for {model}"),
        Cmd::Serve {
            model,
            devices,
            port,
            mesh_mode,
            tp_size,
            pp_size,
            embedding_model,
            embedding_device,
            inflight_slots,
            prefill_ubatch,
            max_queue_depth,
            ctx_cap,
            no_gpu_sampler,
            no_batched_decode,
            prefix_cache,
            prefix_cache_max_gb,
            kv,
            default_system,
            embedding_max_tokens,
        } => serve_cmd(ServeArgs {
            model,
            devices,
            port,
            mesh_mode,
            tp_size,
            pp_size,
            embedding_model,
            embedding_device,
            inflight_slots,
            prefill_ubatch,
            max_queue_depth,
            ctx_cap,
            gpu_sampler: !no_gpu_sampler,
            batched_decode: !no_batched_decode,
            prefix_cache,
            prefix_cache_max_gb,
            kv,
            default_system,
            embedding_max_tokens,
        })?,
        Cmd::Sweep { arch, op, dtype } => sweep(&arch, op.as_deref(), &dtype)?,
        Cmd::CertCheck { arch, backend } => cert_check(&backend, &arch)?,
        Cmd::PmcProbe { kernel, m, k, n } => pmc_probe(&kernel, m, k, n)?,
        Cmd::PmcRefresh { arch } => pmc_refresh(&arch)?,
        Cmd::Matrix { .. } => todo!("implement matrix"),
    }
    Ok(())
}

struct ServeArgs {
    model: String,
    devices: String,
    port: u16,
    mesh_mode: String,
    tp_size: u32,
    pp_size: u32,
    embedding_model: Option<String>,
    embedding_device: Option<i32>,
    inflight_slots: usize,
    prefill_ubatch: usize,
    max_queue_depth: usize,
    ctx_cap: Option<usize>,
    gpu_sampler: bool,
    batched_decode: bool,
    prefix_cache: bool,
    prefix_cache_max_gb: f64,
    kv: String,
    default_system: Option<String>,
    embedding_max_tokens: usize,
}

#[cfg(not(feature = "hip_serve"))]
fn serve_cmd(_args: ServeArgs) -> Result<()> {
    anyhow::bail!(
        "`flambeau serve` requires building with --features hip_serve (needs ROCm + HIP devices)"
    );
}

#[cfg(feature = "hip_serve")]
fn serve_cmd(args: ServeArgs) -> Result<()> {
    use std::net::SocketAddr;
    use std::path::PathBuf;

    let ServeArgs {
        model,
        devices,
        port,
        mesh_mode,
        tp_size,
        pp_size,
        embedding_model,
        embedding_device,
        inflight_slots,
        prefill_ubatch,
        max_queue_depth,
        ctx_cap,
        gpu_sampler,
        batched_decode,
        prefix_cache,
        prefix_cache_max_gb,
        kv,
        default_system,
        embedding_max_tokens,
    } = args;

    // Parse `hip:0,1,2,3` or `0,1,2,3` → Vec<i32>.
    let dev_str = devices.strip_prefix("hip:").unwrap_or(&devices);
    let device_ids: Vec<i32> = dev_str
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<i32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("--devices parse error: {e}"))?;
    if device_ids.is_empty() {
        anyhow::bail!("--devices must list at least one device ID");
    }

    let mesh_mode_parsed = match mesh_mode.as_str() {
        "pp" => flambeau_server::MeshMode::Pp,
        "tp" => {
            let tp_size_resolved = if tp_size == 0 {
                device_ids.len() as u32
            } else {
                tp_size
            };
            if (tp_size_resolved as usize) != device_ids.len() {
                anyhow::bail!(
                    "--mesh-mode tp: --tp-size {tp_size_resolved} must equal --devices count {}",
                    device_ids.len()
                );
            }
            flambeau_server::MeshMode::Tp { world: tp_size_resolved }
        }
        "pp+tp" | "hybrid" => {
            // both axes are explicit — operator-driven, no
            // autodetect. Either both unset → bail with usage; otherwise
            // require pp_size * tp_size == |devices|.
            if pp_size == 0 || tp_size == 0 {
                anyhow::bail!(
                    "--mesh-mode pp+tp requires both --pp-size and --tp-size \
                     (got pp_size={pp_size}, tp_size={tp_size})"
                );
            }
            let want = (pp_size as usize) * (tp_size as usize);
            if want != device_ids.len() {
                anyhow::bail!(
                    "--mesh-mode pp+tp: pp_size={pp_size} * tp_size={tp_size} = {want} \
                     must equal --devices count {}",
                    device_ids.len()
                );
            }
            flambeau_server::MeshMode::Hybrid { pp_size, tp_size }
        }
        other => anyhow::bail!(
            "--mesh-mode must be `pp`, `tp`, or `pp+tp` (got `{other}`)"
        ),
    };

    let bind_addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    // **#230** — resolve embedding device. Default to the first chat
    // device; reject explicit IDs that aren't already in `--devices`
    // (the cluster's HipDevice handles cover only those).
    let resolved_embedding_device = if embedding_model.is_some() {
        let chosen = embedding_device.unwrap_or(device_ids[0]);
        if !device_ids.contains(&chosen) {
            anyhow::bail!(
                "--embedding-device {chosen} must be one of --devices ({:?}); \
                 the embedding model reuses the chat-cluster's HipDevice handle",
                device_ids
            );
        }
        Some(chosen)
    } else {
        None
    };
    let cfg = flambeau_server::ServeConfig {
        gguf_path: PathBuf::from(&model),
        device_ids,
        bind_addr,
        model_id: PathBuf::from(&model)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "flambeau".to_string()),
        mesh_mode: mesh_mode_parsed,
        embedding_gguf_path: embedding_model.map(PathBuf::from),
        embedding_device_id: resolved_embedding_device,
        inflight_slots,
        prefill_ubatch,
        max_queue_depth,
        ctx_cap,
        gpu_sampler,
        batched_decode,
        prefix_cache,
        prefix_cache_max_gb,
        kv,
        default_system,
        embedding_max_tokens,
    };

    // Each binary populates its own registry; register every model
    // crate this CLI links. Future binaries (sweeps, custom servers)
    // can build different registries.
    let mut registry = flambeau_runtime::Registry::new();
    registry.register(std::sync::Arc::new(flambeau_qwen3_moe::Qwen3MoEModelArch));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(flambeau_server::serve(cfg, registry))
}

/// Dump the GGUF's embedded `tokenizer.chat_template` Jinja source.
/// Writes to `out` (truncating) or stdout when `out` is `None`.
fn extract_chat_template(path: &str, out: Option<&str>) -> Result<()> {
    use std::io::Write;
    let file = GgufFile::open(path)?;
    let tpl = file
        .metadata_str("tokenizer.chat_template")
        .ok_or_else(|| anyhow::anyhow!("tokenizer.chat_template missing from {path}"))?;
    match out {
        Some(p) => {
            std::fs::write(p, tpl)?;
            eprintln!("wrote {} bytes of chat template to {p}", tpl.len());
        }
        None => {
            std::io::stdout().write_all(tpl.as_bytes())?;
        }
    }
    Ok(())
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
        let dims: Vec<String> = info.dims.iter().map(std::string::ToString::to_string).collect();
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

/// Registry of sweep ops whose entry point is `fn(&Path) -> Result<Cert>`
/// (i.e. no dtype-dispatch logic — that lives in `qmatmul` / `qmatmul_mmq`
/// / `rmsnorm` which stay hand-rolled). Adding a new single-shape sweep is
/// one row here; typos become compile errors.
#[cfg(feature = "hip_sweep")]
type SweepFn =
    fn(&std::path::Path) -> anyhow::Result<flambeau_bench::cert::Cert>;

#[cfg(feature = "hip_sweep")]
const SIMPLE_SWEEPS: &[(&str, SweepFn)] = &[
    ("rmsnorm_q8_1", flambeau_bench::sweep_rmsnorm_q8_1::run_sweep),
    ("swiglu", flambeau_bench::sweep_swiglu::run_sweep),
    ("quantize_q8_1_mmq", flambeau_bench::sweep_quantize_q8_1_mmq::run_sweep),
    ("attention_prefill_flash_tile", flambeau_bench::sweep_attention_prefill::run_sweep_flash_tile),
    ("rope", flambeau_bench::sweep_rope::run_sweep),
    ("rope_neox_partial", flambeau_bench::sweep_rope_neox::run_sweep),
    ("l2_norm", flambeau_bench::sweep_l2_norm::run_sweep),
    ("causal_conv1d", flambeau_bench::sweep_causal_conv1d::run_sweep),
    ("gdn_state_step", flambeau_bench::sweep_gdn_step::run_sweep),
    ("gdn_state_step_alphabeta", flambeau_bench::sweep_gdn_step_alphabeta::run_sweep),
    ("cast_f32_f16", flambeau_bench::sweep_cast::run_sweep),
    ("cast_f16_f32", flambeau_bench::sweep_f32_pointwise::run_cast_f16_f32_sweep),
    ("silu_f32", flambeau_bench::sweep_f32_pointwise::run_silu_sweep),
    ("swiglu_f32", flambeau_bench::sweep_f32_pointwise::run_swiglu_sweep),
    ("scale_f32", flambeau_bench::sweep_f32_pointwise::run_scale_sweep),
    ("rmsnorm_f32", flambeau_bench::sweep_f32_pointwise::run_rmsnorm_f32_sweep),
    ("gdn_alpha_beta", flambeau_bench::sweep_f32_pointwise::run_gdn_alpha_beta_sweep),
    ("quantize_f16_q8_1", flambeau_bench::sweep_f32_pointwise::run_quantize_f16_q8_1_sweep),
    ("dense_gemv_f32_f16", flambeau_bench::sweep_f32_pointwise::run_dense_gemv_sweep),
    ("add_f16", flambeau_bench::sweep_f32_pointwise::run_add_f16_sweep),
    ("peer_copy_via_host", flambeau_bench::sweep_peer_copy::run_sweep),
    ("shared_expert_scale", flambeau_bench::sweep_shared_expert::run_sweep),
    ("split_q_gate", flambeau_bench::sweep_split_q_gate::run_sweep),
    ("softmax", flambeau_bench::sweep_softmax::run_sweep),
    ("attention_decode", flambeau_bench::sweep_attention::run_sweep),
    ("attention_prefill", flambeau_bench::sweep_attention_prefill::run_sweep),
    ("attention_decode_q8_kv", flambeau_bench::sweep_attention_q8_kv::run_sweep),
    ("attention_decode_splitk", flambeau_bench::sweep_attention_splitk::run_sweep),
    ("mmvq_f16", flambeau_bench::sweep_mmvq_f16::run_sweep),
    ("mmq_f16", flambeau_bench::sweep_mmvq_f16::run_mmq_sweep),
    ("mmq_f16_tile", flambeau_bench::sweep_mmvq_f16::run_mmq_tile_sweep),
    ("mmvq_q4_0", flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q4_0_sweep),
    ("mmvq_q4_0_warpcoop64", flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q4_0_warpcoop64_sweep),
    ("mmvq_q5_0", flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q5_0_sweep),
    ("mmvq_q5_1", flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q5_1_sweep),
    ("indexed_moe_mmvq_q4_0", flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q4_0_sweep),
    ("indexed_moe_mmvq_q4_1", flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q4_1_sweep),
    ("indexed_moe_mmvq_q5_0", flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q5_0_sweep),
    ("indexed_moe_mmvq_q5_1", flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q5_1_sweep),
    ("topk", flambeau_bench::sweep_moe::run_topk_sweep),
    ("indexed_moe_mmvq", flambeau_bench::sweep_moe::run_indexed_moe_mmvq_sweep),
    ("moe_combine", flambeau_bench::sweep_moe::run_moe_combine_sweep),
    ("indexed_moe_mmvq_gate_up", flambeau_bench::sweep_moe::run_gate_up_sweep),
    ("indexed_moe_mmvq_r2", flambeau_bench::sweep_moe::run_indexed_moe_mmvq_r2_sweep),
    ("indexed_moe_mmvq_q2_k", flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q2_k_sweep),
    ("indexed_moe_mmvq_q3_k", flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q3_k_sweep),
    ("indexed_moe_mmvq_q5_k", flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q5_k_sweep),
    ("indexed_moe_mmvq_q6_k", flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q6_k_sweep),
    ("indexed_moe_mmvq_q8_0", flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q8_0_sweep),
    ("indexed_moe_mmq", flambeau_bench::sweep_moe::run_indexed_moe_mmq_sweep),
    ("indexed_moe_mmq_q8_0_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q8_0_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q8_0_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q8_0_down_tile8_sweep),
    ("indexed_moe_mmq_q4_0_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_0_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q4_0_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_0_down_tile8_sweep),
    ("indexed_moe_mmq_q4_1_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_1_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q4_1_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_1_down_tile8_sweep),
    ("indexed_moe_mmq_q5_0_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_0_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q5_0_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_0_down_tile8_sweep),
    ("indexed_moe_mmq_q5_1_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_1_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q5_1_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_1_down_tile8_sweep),
    ("indexed_moe_mmq_q4_k_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_k_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q4_k_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_k_down_tile8_sweep),
    ("indexed_moe_mmq_q5_k_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_k_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q5_k_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_k_down_tile8_sweep),
    ("indexed_moe_mmq_q6_k_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q6_k_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q6_k_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q6_k_down_tile8_sweep),
    ("indexed_moe_mmq_q2_k_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q2_k_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q2_k_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q2_k_down_tile8_sweep),
    ("indexed_moe_mmq_q3_k_gate_up_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q3_k_gate_up_tile8_sweep),
    ("indexed_moe_mmq_q3_k_down_tile8", flambeau_bench::sweep_moe::run_indexed_moe_mmq_q3_k_down_tile8_sweep),
];

#[cfg(feature = "hip_sweep")]
fn find_simple_sweep(op: &str) -> Option<SweepFn> {
    SIMPLE_SWEEPS.iter().find(|(k, _)| *k == op).map(|(_, f)| *f)
}

fn sweep(arch: &str, op: Option<&str>, dtype: &str) -> Result<()> {
    let op = op.unwrap_or("qmatmul");
    if arch != "gfx906" {
        anyhow::bail!("only --arch gfx906 is implemented (got {arch:?})");
    }

    #[cfg(feature = "hip_sweep")]
    {
        match op {
            "qmatmul" => {
                use flambeau_bench::sweep_mmvq::{run_sweep, Dtype, SweepSpec};
                let dtypes: Vec<Dtype> = match dtype {
                    "all" => vec![Dtype::Q8_0, Dtype::Q2K, Dtype::Q3K, Dtype::Q4K, Dtype::Q5K, Dtype::Q6K, Dtype::Q8K],
                    "Q2_K" => vec![Dtype::Q2K],
                    "Q2_K_r2" => vec![Dtype::Q2KR2],
                    "Q3_K" => vec![Dtype::Q3K],
                    "Q3_K_r2" => vec![Dtype::Q3KR2],
                    "Q8_K" => vec![Dtype::Q8K],
                    "all-multirow" => vec![Dtype::Q2KR2, Dtype::Q3KR2, Dtype::Q4KR2, Dtype::Q5KR2, Dtype::Q6KR4],
                    "Q8_0" => vec![Dtype::Q8_0],
                    "Q4_K" => vec![Dtype::Q4K],
                    "Q5_K" => vec![Dtype::Q5K],
                    "Q6_K" => vec![Dtype::Q6K],
                    "Q4_K_r2" => vec![Dtype::Q4KR2],
                    "Q5_K_r2" => vec![Dtype::Q5KR2],
                    "Q6_K_r4" => vec![Dtype::Q6KR4],
                    "Q6_K_dp4a" => vec![Dtype::Q6KDP4A],
                    "Q4_1" => vec![Dtype::Q4_1, Dtype::Q4_1R2, Dtype::Q4_1R2DP4A, Dtype::Q4_1T128],
                    "Q4_1_r2" => vec![Dtype::Q4_1R2],
                    "Q4_1_r2_dp4a" => vec![Dtype::Q4_1R2DP4A],
                    "Q4_1_t128" => vec![Dtype::Q4_1T128],
                    "Q8_0_t128" => vec![Dtype::Q8_0T128],
                    "Q8_0_t128_vdr2" => vec![Dtype::Q8_0T128VDR2],
                    "IQ4_NL" => vec![Dtype::Iq4Nl, Dtype::Iq4NlR2],
                    "IQ4_NL_r2" => vec![Dtype::Iq4NlR2],
                    "IQ4_XS" => vec![Dtype::Iq4Xs, Dtype::Iq4XsR2],
                    "IQ4_XS_r2" => vec![Dtype::Iq4XsR2],
                    "IQ3_XXS" => vec![Dtype::Iq3Xxs, Dtype::Iq3XxsR2],
                    "IQ3_XXS_r2" => vec![Dtype::Iq3XxsR2],
                    "IQ3_S" => vec![Dtype::Iq3S, Dtype::Iq3SR2],
                    "IQ3_S_r2" => vec![Dtype::Iq3SR2],
                    "IQ2_XXS" => vec![Dtype::Iq2Xxs, Dtype::Iq2XxsR2],
                    "IQ2_XXS_r2" => vec![Dtype::Iq2XxsR2],
                    "IQ2_XS" => vec![Dtype::Iq2Xs, Dtype::Iq2XsR2],
                    "IQ2_XS_r2" => vec![Dtype::Iq2XsR2],
                    "IQ2_S" => vec![Dtype::Iq2S, Dtype::Iq2SR2],
                    "IQ2_S_r2" => vec![Dtype::Iq2SR2],
                    "IQ1_S" => vec![Dtype::Iq1S, Dtype::Iq1SR2],
                    "IQ1_S_r2" => vec![Dtype::Iq1SR2],
                    "IQ1_M" => vec![Dtype::Iq1M, Dtype::Iq1MR2],
                    "IQ1_M_r2" => vec![Dtype::Iq1MR2],
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
                    "Q4_0" | "Q4_0_wave64" => (vec![MmqDtype::Q4_0Wave64], SweepSpec::v1_4_prefill),
                    "Q4_0_4warp" => (vec![MmqDtype::Q4_04Warp], SweepSpec::v1_4_prefill),
                    "Q5_0" | "Q5_0_wave64" => (vec![MmqDtype::Q5_0Wave64], SweepSpec::v1_4_prefill),
                    "Q5_1" | "Q5_1_wave64" => (vec![MmqDtype::Q5_1Wave64], SweepSpec::v1_4_prefill),
                    "Q4_K" | "Q4_K_4warp" => (vec![MmqDtype::Q4K4Warp], SweepSpec::v1_4_prefill),
                    "Q4_K_wave64" => (vec![MmqDtype::Q4KWave64], SweepSpec::v1_4_prefill),
                    "Q4_K_turbo" => (vec![MmqDtype::Q4KTurbo], SweepSpec::v1_4_prefill),
                    "Q5_K" | "Q5_K_wave64" => (vec![MmqDtype::Q5KWave64], SweepSpec::v1_4_prefill),
                    "Q6_K" | "Q6_K_4warp" => (vec![MmqDtype::Q6K4Warp], SweepSpec::v1_4_prefill),
                    "Q6_K_wave64" => (vec![MmqDtype::Q6KWave64], SweepSpec::v1_4_prefill),
                    "Q8_K" | "Q8_K_wave64" => (vec![MmqDtype::Q8KWave64], SweepSpec::v1_4_prefill),
                    "Q2_K" | "Q2_K_wave64" => (vec![MmqDtype::Q2KWave64], SweepSpec::v1_4_prefill),
                    "Q3_K" | "Q3_K_wave64" => (vec![MmqDtype::Q3KWave64], SweepSpec::v1_4_prefill),
                    "IQ4_XS" | "IQ4_XS_wave64" => (vec![MmqDtype::Iq4XsWave64], SweepSpec::v1_4_prefill),
                    "IQ3_S" | "IQ3_S_wave64" => (vec![MmqDtype::Iq3SWave64], SweepSpec::v1_4_prefill),
                    "IQ4_NL" | "IQ4_NL_wave64" => (vec![MmqDtype::Iq4NlWave64], SweepSpec::v1_4_prefill),
                    "IQ3_XXS" | "IQ3_XXS_wave64" => (vec![MmqDtype::Iq3XxsWave64], SweepSpec::v1_4_prefill),
                    "IQ2_XXS" | "IQ2_XXS_wave64" => (vec![MmqDtype::Iq2XxsWave64], SweepSpec::v1_4_prefill),
                    "IQ2_XS" | "IQ2_XS_wave64" => (vec![MmqDtype::Iq2XsWave64], SweepSpec::v1_4_prefill),
                    "IQ2_S" | "IQ2_S_wave64" => (vec![MmqDtype::Iq2SWave64], SweepSpec::v1_4_prefill),
                    "IQ1_S" | "IQ1_S_wave64" => (vec![MmqDtype::Iq1SWave64], SweepSpec::v1_4_prefill),
                    "IQ1_M" | "IQ1_M_wave64" => (vec![MmqDtype::Iq1MWave64], SweepSpec::v1_4_prefill),
                    "all" => (
                        vec![
                            MmqDtype::Q8_0Oracle,
                            MmqDtype::Q8_04Warp,
                            MmqDtype::Q8_0Wave64,
                            MmqDtype::Q8_0Wave64Tile16,
                            MmqDtype::Q4_14Warp,
                            MmqDtype::Q4_1Wave64,
                            MmqDtype::Q4_0Wave64,
                            MmqDtype::Q4_04Warp,
                            MmqDtype::Q5_0Wave64,
                            MmqDtype::Q5_1Wave64,
                            MmqDtype::Q4K4Warp,
                            MmqDtype::Q4KWave64,
                            MmqDtype::Q5KWave64,
                            MmqDtype::Q6K4Warp,
                            MmqDtype::Q6KWave64,
                            MmqDtype::Q8KWave64,
                            MmqDtype::Q2KWave64,
                            MmqDtype::Q3KWave64,
                        ],
                        // Oracle uses its small grid, 4warp uses the prefill grid.
                        |d| match d {
                            MmqDtype::Q8_0Oracle => SweepSpec::v1_4_oracle(d),
                            MmqDtype::Q8_04Warp
                            | MmqDtype::Q8_0Wave64
                            | MmqDtype::Q8_0Wave64Tile16
                            | MmqDtype::Q4_14Warp
                            | MmqDtype::Q4_1Wave64
                            | MmqDtype::Q4_0Wave64
                            | MmqDtype::Q4_04Warp
                            | MmqDtype::Q5_0Wave64
                            | MmqDtype::Q5_1Wave64
                            | MmqDtype::Q4KTurbo
                            | MmqDtype::Q4K4Warp
                            | MmqDtype::Q4KWave64
                            | MmqDtype::Q5KWave64
                            | MmqDtype::Q6K4Warp
                            | MmqDtype::Q6KWave64
                            | MmqDtype::Q8KWave64
                            | MmqDtype::Q2KWave64
                            | MmqDtype::Q3KWave64
                            | MmqDtype::Iq4XsWave64
                            | MmqDtype::Iq3SWave64
                            | MmqDtype::Iq4NlWave64
                            | MmqDtype::Iq3XxsWave64
                            | MmqDtype::Iq2XxsWave64
                            | MmqDtype::Iq2XsWave64
                            | MmqDtype::Iq2SWave64
                            | MmqDtype::Iq1SWave64
                            | MmqDtype::Iq1MWave64 => SweepSpec::v1_4_prefill(d),
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
            other => {
                if let Some(run) = find_simple_sweep(other) {
                    let _ = dtype;
                    let cert = run(&repo_root())?;
                    println!(
                        "sweep {other}: pass={} shapes={} rig={}",
                        cert.pass, cert.results.len(), cert.rig
                    );
                    Ok(())
                } else {
                    anyhow::bail!(
                        "unknown --op {other} (qmatmul | qmatmul_mmq | rmsnorm | one of {:?})",
                        SIMPLE_SWEEPS.iter().map(|(k, _)| *k).collect::<Vec<_>>()
                    )
                }
            }
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
