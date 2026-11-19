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
#[command(
    name = "flambeau",
    version,
    about = "Max-perf inference server for modern LLMs on HIP + CUDA."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump GGUF tensor list, dtype audit, metadata ().
    InspectGguf { path: String },
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
    InspectHsaco { path: String },
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
        /// Topology — `pp` (pipeline-parallel, default) or `tp`
        /// (tensor-parallel). Single-device collapses both.
        #[arg(long, default_value = "pp")]
        mesh_mode: String,
    },
    /// OpenAI-compatible HTTP server () + optional MCP upstream
    /// client.
    Serve {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "hip:0")]
        devices: String,
        /// GPU backend: `hip` (default, multi-topology) or `cuda`
        /// (single-device only, RTX).
        #[arg(long, default_value = "hip")]
        backend: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// / mesh topology:
        /// - `pp` — pipeline-parallel (V1 default, LayerAssignment-based).
        /// - `tp` — tensor-parallel (Megatron-style per-tensor sharding
        ///   with BAR1 P2P AllReduce).
        /// - `pp+tp` (alias `hybrid`) — manual PP-of-TP. Requires both
        ///   `--pp-size` and `--tp-size`; `pp_size * tp_size` must equal
        ///   `--devices` count. Devices are interpreted in stage-major
        ///   order. flambeau does not autodetect the right topology —
        ///   pick one with the bracket-bench harness.
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
        /// Override the per-rank layer split. Comma-separated integers,
        /// one per PP rank. Sum must equal the model's num_layers.
        /// Example: `--layer-split 12,16,16,20` for PP4 to put more
        /// layers on the last (lighter) rank in a hybrid GDN+attn
        /// arch like Qwen3.6. Default: uniform split.
        #[arg(long = "layer-split")]
        layer_split: Option<String>,
        /// Optional path to a `qwen3` arch embedding
        /// GGUF (e.g. `Qwen3-Embedding-0.6B-Q8_0.gguf`). Loaded
        /// alongside the chat model on a single device; powers the
        /// `/v1/embeddings` endpoint. Omit to disable embeddings.
        #[arg(long = "embedding-model")]
        embedding_model: Option<String>,
        /// HIP device for the embedding model
        /// (numeric, e.g. `0`). Defaults to the first device in
        /// `--devices`. The embedding model shares VRAM with whatever
        /// chat-model rank lives on the same device; a 600 M / 4 B
        /// embedding fits alongside a 27 B chat shard on 16 GB MI50.
        #[arg(long = "embedding-device")]
        embedding_device: Option<i32>,
        /// Number of concurrent inflight decode slots (1-32).
        #[arg(
            long = "inflight-slots",
            env = "FLAMBEAU_INFLIGHT_SLOTS",
            default_value_t = 4
        )]
        inflight_slots: usize,
        /// Prefill chunk size (tokens per forward step).
        #[arg(
            long = "prefill-ubatch",
            env = "FLAMBEAU_PREFILL_UBATCH",
            default_value_t = 512
        )]
        prefill_ubatch: usize,
        /// Max MTP (multi-token prediction) draft tokens verified per step
        /// using the GGUF's embedded nextn head. 0 disables MTP.
        #[arg(long = "mtp", env = "FLAMBEAU_MTP", default_value_t = 0)]
        mtp: usize,
        /// Per-chunk prefill budget used by the K4c mixed-batch
        /// scheduler. Smaller chunks improve short-request TTFT
        /// under load (Sarathi-Serve trade) at the cost of slightly
        /// higher long-request prefill latency.
        #[arg(
            long = "prefill-chunk-tokens",
            default_value_t = 512
        )]
        prefill_chunk_tokens: usize,
        /// Activate PagedAttention with `N` pages per layer (page size
        /// fixed at 16 tokens). Default off (contiguous per-slot KV
        /// slab). Set `N` to the per-layer page budget; pair with a
        /// high `--inflight-slots` to pack more concurrent requests
        /// into the same VRAM. `N=1` clamps to the
        /// `max_slots * max_pages_per_slot` floor (correctness only,
        /// no VRAM win).
        #[arg(long = "paged-kv")]
        paged_kv: Option<usize>,
        /// Admission-control queue depth beyond `inflight_slots` before
        /// returning 503 + Retry-After. 0 disables (legacy).
        #[arg(
            long = "max-queue-depth",
            env = "FLAMBEAU_MAX_QUEUE_DEPTH",
            default_value_t = 16
        )]
        max_queue_depth: usize,
        /// Decode-batching coalescence window (microseconds). The
        /// leader sleeps this long before draining the pending-decode
        /// queue so concurrent requests join the same batched forward.
        #[arg(
            long = "decode-batch-window-us",
            env = "FLAMBEAU_DECODE_BATCH_WINDOW_US",
            default_value_t = 1500
        )]
        decode_batch_window_us: u64,
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
        #[arg(
            long = "prefix-cache-max-gb",
            env = "FLAMBEAU_PREFIX_CACHE_MAX_GB",
            default_value_t = 2.0
        )]
        prefix_cache_max_gb: f64,
        /// KV cache layout. `auto` (default: per-arch — gemma uses `q8`,
        /// others `f16`) or force `f16` / `q8` (`q8` ≈ 2× HBM saving on
        /// decode; correctness cert required per model).
        #[arg(long = "kv", env = "FLAMBEAU_KV", default_value = "auto")]
        kv: String,
        /// Default system prompt prepended to chat-template requests
        /// when none is provided in the request.
        #[arg(long = "default-system", env = "FLAMBEAU_DEFAULT_SYSTEM")]
        default_system: Option<String>,
        /// /v1/embeddings endpoint per-prompt token cap.
        #[arg(
            long = "embedding-max-tokens",
            env = "FLAMBEAU_EMBEDDING_MAX_TOKENS",
            default_value_t = 8192
        )]
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
    /// Time certified kernel variants per shape and emit autotuned dispatch
    /// rows to `dispatch/<backend>/<arch>.toml`. gfx906 ranks all compiled
    /// candidates (a check on the curated table); sm_86 fills the CUDA table.
    Autotune {
        #[arg(long)]
        arch: String,
        #[arg(long, default_value = "qmatmul")]
        op: String,
        #[arg(long, default_value = "Q8_0")]
        dtype: String,
        /// Write the dispatch table. Without it, autotune is a dry run that
        /// only prints the ranking.
        #[arg(long)]
        emit: bool,
        /// CI ratchet — re-measure and fail (non-zero exit) if the committed
        /// dispatch table disagrees (a certified candidate now wins by >3%, or
        /// an incumbent lost its cert). Does not write. Mutually exclusive with
        /// `--emit`.
        #[arg(long)]
        check: bool,
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
        Cmd::Infer {
            model,
            prompt,
            max_tokens,
            devices,
            mesh_mode,
        } => infer_main(&model, &prompt, max_tokens, &devices, &mesh_mode)?,
        #[cfg(not(any(feature = "hip_serve", feature = "cuda_serve")))]
        Cmd::Serve { .. } => anyhow::bail!(
            "`flambeau serve` requires building with --features hip_serve and/or cuda_serve"
        ),
        #[cfg(any(feature = "hip_serve", feature = "cuda_serve"))]
        Cmd::Serve {
            model,
            devices,
            backend,
            port,
            mesh_mode,
            tp_size,
            pp_size,
            layer_split,
            embedding_model,
            embedding_device,
            inflight_slots,
            prefill_ubatch,
            mtp,
            prefill_chunk_tokens,
            paged_kv,
            max_queue_depth,
            decode_batch_window_us,
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
            backend,
            port,
            mesh_mode,
            tp_size,
            pp_size,
            layer_split,
            embedding_model,
            embedding_device,
            inflight_slots,
            prefill_ubatch,
            mtp,
            prefill_chunk_tokens,
            paged_kv,
            max_queue_depth,
            decode_batch_window_us,
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
        Cmd::Autotune { arch, op, dtype, emit, check } => autotune(&arch, &op, &dtype, emit, check)?,
        Cmd::CertCheck { arch, backend } => cert_check(&backend, &arch)?,
        Cmd::PmcProbe { kernel, m, k, n } => pmc_probe(&kernel, m, k, n)?,
        Cmd::PmcRefresh { arch } => pmc_refresh(&arch)?,
        Cmd::Matrix { .. } => todo!("implement matrix"),
    }
    Ok(())
}

#[cfg(any(feature = "hip_serve", feature = "cuda_serve"))]
struct ServeArgs {
    model: String,
    devices: String,
    backend: String,
    port: u16,
    mesh_mode: String,
    tp_size: u32,
    pp_size: u32,
    layer_split: Option<String>,
    embedding_model: Option<String>,
    embedding_device: Option<i32>,
    inflight_slots: usize,
    prefill_ubatch: usize,
    mtp: usize,
    prefill_chunk_tokens: usize,
    paged_kv: Option<usize>,
    max_queue_depth: usize,
    decode_batch_window_us: u64,
    ctx_cap: Option<usize>,
    gpu_sampler: bool,
    batched_decode: bool,
    prefix_cache: bool,
    prefix_cache_max_gb: f64,
    kv: String,
    default_system: Option<String>,
    embedding_max_tokens: usize,
}

fn infer_main(
    _model: &str,
    _prompt: &str,
    _max_tokens: usize,
    _devices: &str,
    _mesh_mode: &str,
) -> Result<()> {
    anyhow::bail!(
        "`flambeau infer` was removed in #221 — the legacy ModelDriver path is gone. \
         Use `flambeau serve` + an OpenAI-compatible client for inference."
    );
}

#[cfg(any(feature = "hip_serve", feature = "cuda_serve"))]
fn serve_cmd(args: ServeArgs) -> Result<()> {
    use std::net::SocketAddr;
    use std::path::PathBuf;

    let ServeArgs {
        model,
        devices,
        backend,
        port,
        mesh_mode,
        tp_size,
        pp_size,
        layer_split,
        embedding_model,
        embedding_device,
        inflight_slots,
        prefill_ubatch,
        mtp,
        prefill_chunk_tokens,
        paged_kv,
        max_queue_depth,
        decode_batch_window_us,
        ctx_cap,
        gpu_sampler,
        batched_decode,
        prefix_cache,
        prefix_cache_max_gb,
        kv,
        default_system,
        embedding_max_tokens,
    } = args;

    let backend = match backend.as_str() {
        "hip" => flambeau_server::ServeBackend::Hip,
        "cuda" => flambeau_server::ServeBackend::Cuda,
        other => anyhow::bail!("--backend must be `hip` or `cuda` (got `{other}`)"),
    };

    // Parse `hip:0,1,2,3` / `cuda:0` / `0,1,2,3` → Vec<i32>.
    let dev_str = devices
        .strip_prefix("hip:")
        .or_else(|| devices.strip_prefix("cuda:"))
        .unwrap_or(&devices);
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
            flambeau_server::MeshMode::Tp {
                world: tp_size_resolved,
            }
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
        other => anyhow::bail!("--mesh-mode must be `pp`, `tp`, or `pp+tp` (got `{other}`)"),
    };

    let bind_addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    // Resolve embedding device. Default to the first chat
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
        backend,
        bind_addr,
        model_id: PathBuf::from(&model)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "flambeau".to_string()),
        mesh_mode: mesh_mode_parsed,
        layer_split: match layer_split.as_deref() {
            None => None,
            Some(s) => Some(
                s.split(',')
                    .filter(|p| !p.is_empty())
                    .map(|p| p.trim().parse::<usize>())
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| anyhow::anyhow!("--layer-split parse error: {e}"))?,
            ),
        },
        embedding_gguf_path: embedding_model.map(PathBuf::from),
        embedding_device_id: resolved_embedding_device,
        inflight_slots,
        prefill_ubatch,
        mtp,
        prefill_chunk_tokens,
        paged_kv_pages: paged_kv,
        max_queue_depth,
        decode_batch_window_us,
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
    struct Qwen3FamilyArch;
    impl flambeau_runtime::Model for Qwen3FamilyArch {
        fn supported_archs(&self) -> &[&'static str] {
            &["qwen35", "qwen35moe", "qwen3moe", "qwen3next"]
        }
        fn description(&self) -> &'static str {
            "Qwen3 family (Qwen3.5/3.6 dense + MoE, Qwen3-Coder-Next GDN hybrid)"
        }
    }
    struct Gemma4FamilyArch;
    impl flambeau_runtime::Model for Gemma4FamilyArch {
        fn supported_archs(&self) -> &[&'static str] {
            &["gemma4"]
        }
        fn description(&self) -> &'static str {
            "Gemma 4 family (E2B / E4B edge + 26B-A4B MoE + 31B dense, iSWA + softcap)"
        }
    }
    registry.register(std::sync::Arc::new(Qwen3FamilyArch));
    registry.register(std::sync::Arc::new(Gemma4FamilyArch));

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
        let dims: Vec<String> = info
            .dims
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
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
type SweepFn = fn(&std::path::Path) -> anyhow::Result<flambeau_bench::cert::Cert>;

#[cfg(feature = "hip_sweep")]
const SIMPLE_SWEEPS: &[(&str, SweepFn)] = &[
    (
        "rmsnorm_q8_1",
        flambeau_bench::sweep_rmsnorm_q8_1::run_sweep,
    ),
    ("swiglu", flambeau_bench::sweep_swiglu::run_sweep),
    (
        "quantize_q8_1_mmq",
        flambeau_bench::sweep_quantize_q8_1_mmq::run_sweep,
    ),
    (
        "attention_prefill_flash_tile",
        flambeau_bench::sweep_attention_prefill::run_sweep_flash_tile,
    ),
    ("rope", flambeau_bench::sweep_rope::run_sweep),
    (
        "rope_neox_partial",
        flambeau_bench::sweep_rope_neox::run_sweep,
    ),
    ("l2_norm", flambeau_bench::sweep_l2_norm::run_sweep),
    (
        "causal_conv1d",
        flambeau_bench::sweep_causal_conv1d::run_sweep,
    ),
    ("gdn_state_step", flambeau_bench::sweep_gdn_step::run_sweep),
    (
        "gdn_state_step_alphabeta",
        flambeau_bench::sweep_gdn_step_alphabeta::run_sweep,
    ),
    (
        "gdn_alpha_beta",
        flambeau_bench::sweep_f32_pointwise::run_gdn_alpha_beta_sweep,
    ),
    (
        "peer_copy_via_host",
        flambeau_bench::sweep_peer_copy::run_sweep,
    ),
    (
        "shared_expert_scale",
        flambeau_bench::sweep_shared_expert::run_sweep,
    ),
    (
        "split_q_gate",
        flambeau_bench::sweep_split_q_gate::run_sweep,
    ),
    ("softmax", flambeau_bench::sweep_softmax::run_sweep),
    (
        "attention_decode",
        flambeau_bench::sweep_attention::run_sweep,
    ),
    (
        "attention_prefill",
        flambeau_bench::sweep_attention_prefill::run_sweep,
    ),
    (
        "attention_decode_q8_kv",
        flambeau_bench::sweep_attention_q8_kv::run_sweep,
    ),
    (
        "attention_prefill_q8_kv",
        flambeau_bench::sweep_attention_prefill_q8_kv::run_sweep,
    ),
    (
        "attention_decode_splitk",
        flambeau_bench::sweep_attention_splitk::run_sweep,
    ),
    (
        // The bare `attention_decode_q8_kv_splitk` op resolves to the
        // cross-backend sweep first; this one emits the gfx906 cert.
        "attention_decode_q8_kv_splitk_gfx906",
        flambeau_bench::sweep_attention_q8_kv_splitk::run_sweep,
    ),
    (
        "mmvq_q4_0",
        flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q4_0_sweep,
    ),
    (
        "mmvq_q4_0_warpcoop64",
        flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q4_0_warpcoop64_sweep,
    ),
    (
        "mmvq_q5_0",
        flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q5_0_sweep,
    ),
    (
        "mmvq_q5_1",
        flambeau_bench::sweep_q4_0_q5_0::run_mmvq_q5_1_sweep,
    ),
    (
        "indexed_moe_mmvq_q4_0",
        flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q4_0_sweep,
    ),
    (
        "indexed_moe_mmvq_q4_1",
        flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q4_1_sweep,
    ),
    (
        "indexed_moe_mmvq_q5_0",
        flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q5_0_sweep,
    ),
    (
        "indexed_moe_mmvq_q5_1",
        flambeau_bench::sweep_q4_0_q5_0::run_indexed_moe_mmvq_q5_1_sweep,
    ),
    ("topk", flambeau_bench::sweep_moe::run_topk_sweep),
    (
        "indexed_moe_mmvq",
        flambeau_bench::sweep_moe::run_indexed_moe_mmvq_sweep,
    ),
    (
        "moe_combine",
        flambeau_bench::sweep_moe::run_moe_combine_sweep,
    ),
    (
        "indexed_moe_mmvq_gate_up",
        flambeau_bench::sweep_moe::run_gate_up_sweep,
    ),
    (
        "indexed_moe_mmvq_r2",
        flambeau_bench::sweep_moe::run_indexed_moe_mmvq_r2_sweep,
    ),
    (
        "indexed_moe_mmvq_q2_k",
        flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q2_k_sweep,
    ),
    (
        "indexed_moe_mmvq_q3_k",
        flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q3_k_sweep,
    ),
    (
        "indexed_moe_mmvq_q5_k",
        flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q5_k_sweep,
    ),
    (
        "indexed_moe_mmvq_q6_k",
        flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q6_k_sweep,
    ),
    (
        "indexed_moe_mmvq_q8_0",
        flambeau_bench::sweep_moe::run_indexed_moe_mmvq_q8_0_sweep,
    ),
    (
        "indexed_moe_mmq",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_sweep,
    ),
    (
        "indexed_moe_mmq_q8_0_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q8_0_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q8_0_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q8_0_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q4_0_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_0_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q4_0_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_0_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q4_1_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_1_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q4_1_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_1_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q5_0_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_0_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q5_0_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_0_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q5_1_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_1_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q5_1_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_1_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q4_k_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_k_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q4_k_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q4_k_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q5_k_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_k_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q5_k_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q5_k_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q6_k_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q6_k_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q6_k_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q6_k_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q2_k_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q2_k_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q2_k_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q2_k_down_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q3_k_gate_up_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q3_k_gate_up_tile8_sweep,
    ),
    (
        "indexed_moe_mmq_q3_k_down_tile8",
        flambeau_bench::sweep_moe::run_indexed_moe_mmq_q3_k_down_tile8_sweep,
    ),
];

#[cfg(feature = "hip_sweep")]
fn find_simple_sweep(op: &str) -> Option<SweepFn> {
    SIMPLE_SWEEPS
        .iter()
        .find(|(k, _)| *k == op)
        .map(|(_, f)| *f)
}

fn sweep(arch: &str, op: Option<&str>, dtype: &str) -> Result<()> {
    let op = op.unwrap_or("qmatmul");

    #[cfg(feature = "cuda_sweep")]
    if arch == "sm_86" {
        return sweep_cuda(op, dtype);
    }

    if arch != "gfx906" {
        anyhow::bail!(
            "only --arch gfx906 (and sm_86 with --features cuda_sweep) is implemented (got {arch:?})"
        );
    }

    #[cfg(feature = "hip_sweep")]
    {
        match op {
            "qmatmul" => {
                use flambeau_bench::sweep_mmvq::{run_sweep, Dtype, SweepSpec};
                if dtype == "Q4_0_kv_f16dst" {
                    let cert = flambeau_bench::sweep_mmvq::run_kv_f16dst_sweep::<
                        flambeau_bench::backend::Hip,
                    >(&repo_root(), arch)?;
                    println!(
                        "sweep mmvq_q4_0_kv_f16dst: pass={} shapes={} rig={}",
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(gu) = gate_up_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_gate_up_sweep;
                    let root = repo_root();
                    let cert = run_gate_up_sweep::<flambeau_bench::backend::Hip>(gu, &root, arch)?;
                    println!(
                        "sweep qmatmul_gate_up ({}): pass={} shapes={} rig={}",
                        gu.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(bd) = batched_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_batched_sweep;
                    let root = repo_root();
                    let cert = run_batched_sweep::<flambeau_bench::backend::Hip>(bd, &root, arch)?;
                    println!(
                        "sweep qmatmul_batched ({}): pass={} shapes={} rig={}",
                        bd.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(gb) = gate_up_batched_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_gate_up_batched_sweep;
                    let root = repo_root();
                    let cert = run_gate_up_batched_sweep::<flambeau_bench::backend::Hip>(gb, &root, arch)?;
                    println!(
                        "sweep qmatmul_gate_up_batched ({}): pass={} shapes={} rig={}",
                        gb.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(od) = mmq_oracle_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_mmq_oracle_sweep;
                    let root = repo_root();
                    let cert = run_mmq_oracle_sweep::<flambeau_bench::backend::Hip>(od, &root, arch)?;
                    println!(
                        "sweep qmatmul_mmq_oracle ({}): pass={} shapes={} rig={}",
                        od.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(gd) = mmq_gemm_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_mmq_gemm_sweep;
                    let root = repo_root();
                    let cert = run_mmq_gemm_sweep::<flambeau_bench::backend::Hip>(gd, &root, arch)?;
                    println!(
                        "sweep qmatmul_mmq_gemm ({}): pass={} shapes={} rig={}",
                        gd.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(wd) = mmq_4warp_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_mmq_4warp_sweep;
                    let root = repo_root();
                    let cert = run_mmq_4warp_sweep::<flambeau_bench::backend::Hip>(wd, &root, arch)?;
                    println!(
                        "sweep qmatmul_mmq_4warp ({}): pass={} shapes={} rig={}",
                        wd.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(ld) = mmq_4warp_lds_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_mmq_4warp_lds_sweep;
                    let root = repo_root();
                    let cert =
                        run_mmq_4warp_lds_sweep::<flambeau_bench::backend::Hip>(ld, &root, arch)?;
                    println!(
                        "sweep qmatmul_mmq_4warp_lds ({}): pass={} shapes={} rig={}",
                        ld.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(md) = moe_mmvq_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_moe_mmvq_sweep;
                    let root = repo_root();
                    let cert = run_moe_mmvq_sweep::<flambeau_bench::backend::Hip>(md, &root, arch)?;
                    println!(
                        "sweep indexed_moe_mmvq ({}): pass={} shapes={} rig={}",
                        md.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(gu) = moe_gate_up_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_moe_gate_up_sweep;
                    let root = repo_root();
                    let cert = run_moe_gate_up_sweep::<flambeau_bench::backend::Hip>(gu, &root, arch)?;
                    println!(
                        "sweep indexed_moe_mmvq_gate_up ({}): pass={} shapes={} rig={}",
                        gu.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(ld) = moe_mmq_lds_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_moe_mmq_lds_sweep;
                    let root = repo_root();
                    let cert = run_moe_mmq_lds_sweep::<flambeau_bench::backend::Hip>(ld, &root, arch)?;
                    println!(
                        "sweep indexed_moe_mmq_lds_tiled ({}): pass={} shapes={} rig={}",
                        ld.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(t8) = moe_mmq_tile8_dtype(dtype) {
                    use flambeau_bench::sweep_mmvq::run_moe_mmq_tile8_sweep;
                    let root = repo_root();
                    let cert = run_moe_mmq_tile8_sweep::<flambeau_bench::backend::Hip>(t8, &root, arch)?;
                    println!(
                        "sweep indexed_moe_mmq_tile8 ({}): pass={} shapes={} rig={}",
                        t8.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(fk) = f16_sweep_kind(dtype) {
                    use flambeau_bench::sweep_mmvq_f16::run_f16_sweep;
                    let root = repo_root();
                    let cert = run_f16_sweep::<flambeau_bench::backend::Hip>(fk, &root, arch)?;
                    println!(
                        "sweep qmatmul_f16 ({}): pass={} shapes={} rig={}",
                        fk.name(),
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                let dtypes: Vec<Dtype> = match dtype {
                    "all" => vec![
                        Dtype::Q8_0,
                        Dtype::Q2K,
                        Dtype::Q3K,
                        Dtype::Q4K,
                        Dtype::Q5K,
                        Dtype::Q6K,
                        Dtype::Q8K,
                    ],
                    "Q2_K" => vec![Dtype::Q2K, Dtype::Q2KR2, Dtype::Q2KDp4a, Dtype::Q2KR2Dp4a],
                    "Q2_K_r2" => vec![Dtype::Q2KR2],
                    "Q2_K_dp4a" => vec![Dtype::Q2KDp4a],
                    "Q2_K_r2_dp4a" => vec![Dtype::Q2KR2Dp4a],
                    "Q3_K" => vec![Dtype::Q3K, Dtype::Q3KR2, Dtype::Q3KDp4a, Dtype::Q3KR2Dp4a],
                    "Q3_K_r2" => vec![Dtype::Q3KR2],
                    "Q3_K_dp4a" => vec![Dtype::Q3KDp4a],
                    "Q3_K_r2_dp4a" => vec![Dtype::Q3KR2Dp4a],
                    "Q8_K" => vec![Dtype::Q8K],
                    "all-multirow" => vec![
                        Dtype::Q2KR2,
                        Dtype::Q3KR2,
                        Dtype::Q4KR2,
                        Dtype::Q5KR2,
                        Dtype::Q6KR4,
                    ],
                    "Q8_0" => vec![Dtype::Q8_0],
                    "Q8_0_dp4a" => vec![Dtype::Q8_0Dp4a],
                    "Q8_0_r2" => vec![Dtype::Q8_0R2],
                    "Q8_0_r4" => vec![Dtype::Q8_0R4],
                    "Q4_K" => vec![Dtype::Q4K, Dtype::Q4KR2, Dtype::Q4KR2Dp4a],
                    "Q4_K_r2_dp4a" => vec![Dtype::Q4KR2Dp4a],
                    "Q5_K" => vec![Dtype::Q5K, Dtype::Q5KR2, Dtype::Q5KDp4a, Dtype::Q5KR2Dp4a],
                    "Q6_K" => vec![Dtype::Q6K],
                    "Q4_K_r2" => vec![Dtype::Q4KR2],
                    "Q5_K_r2" => vec![Dtype::Q5KR2],
                    "Q5_K_dp4a" => vec![Dtype::Q5KDp4a],
                    "Q5_K_r2_dp4a" => vec![Dtype::Q5KR2Dp4a],
                    "Q6_K_r4" => vec![Dtype::Q6KR4],
                    "Q6_K_dp4a" => vec![Dtype::Q6KDP4A],
                    "Q4_1" => vec![
                        Dtype::Q4_1,
                        Dtype::Q4_1R2,
                        Dtype::Q4_1R2DP4A,
                        Dtype::Q4_1T128,
                    ],
                    "Q4_1_r2" => vec![Dtype::Q4_1R2],
                    "Q4_1_r2_dp4a" => vec![Dtype::Q4_1R2DP4A],
                    "Q4_1_t128" => vec![Dtype::Q4_1T128],
                    "Q8_0_t128" => vec![Dtype::Q8_0T128],
                    "Q8_0_t128_vdr2" => vec![Dtype::Q8_0T128VDR2],
                    "IQ4_NL" => vec![Dtype::Iq4Nl, Dtype::Iq4NlR2, Dtype::Iq4NlDp4a],
                    "IQ4_NL_r2" => vec![Dtype::Iq4NlR2],
                    "IQ4_NL_dp4a" => vec![Dtype::Iq4NlDp4a],
                    "IQ4_XS" => vec![Dtype::Iq4Xs, Dtype::Iq4XsR2, Dtype::Iq4XsDp4a],
                    "IQ4_XS_r2" => vec![Dtype::Iq4XsR2],
                    "IQ4_XS_dp4a" => vec![Dtype::Iq4XsDp4a],
                    "IQ3_XXS" => vec![Dtype::Iq3Xxs, Dtype::Iq3XxsR2, Dtype::Iq3XxsDp4a],
                    "IQ3_XXS_r2" => vec![Dtype::Iq3XxsR2],
                    "IQ3_XXS_dp4a" => vec![Dtype::Iq3XxsDp4a],
                    "IQ3_S" => vec![Dtype::Iq3S, Dtype::Iq3SR2, Dtype::Iq3SDp4a],
                    "IQ3_S_r2" => vec![Dtype::Iq3SR2],
                    "IQ3_S_dp4a" => vec![Dtype::Iq3SDp4a],
                    "IQ2_XXS" => vec![Dtype::Iq2Xxs, Dtype::Iq2XxsR2, Dtype::Iq2XxsDp4a],
                    "IQ2_XXS_r2" => vec![Dtype::Iq2XxsR2],
                    "IQ2_XXS_dp4a" => vec![Dtype::Iq2XxsDp4a],
                    "IQ2_XS" => vec![Dtype::Iq2Xs, Dtype::Iq2XsR2, Dtype::Iq2XsDp4a],
                    "IQ2_XS_r2" => vec![Dtype::Iq2XsR2],
                    "IQ2_XS_dp4a" => vec![Dtype::Iq2XsDp4a],
                    "IQ2_S" => vec![Dtype::Iq2S, Dtype::Iq2SR2, Dtype::Iq2SDp4a],
                    "IQ2_S_r2" => vec![Dtype::Iq2SR2],
                    "IQ2_S_dp4a" => vec![Dtype::Iq2SDp4a],
                    "IQ1_S" => vec![Dtype::Iq1S, Dtype::Iq1SR2, Dtype::Iq1SDp4a],
                    "IQ1_S_r2" => vec![Dtype::Iq1SR2],
                    "IQ1_S_dp4a" => vec![Dtype::Iq1SDp4a],
                    "IQ1_M" => vec![Dtype::Iq1M, Dtype::Iq1MR2, Dtype::Iq1MDp4a],
                    "IQ1_M_r2" => vec![Dtype::Iq1MR2],
                    "IQ1_M_dp4a" => vec![Dtype::Iq1MDp4a],
                    other => anyhow::bail!("unknown dtype {other}"),
                };
                let root = repo_root();
                for d in dtypes {
                    let spec = SweepSpec::v1_3_default(d);
                    let cert = run_sweep::<flambeau_bench::backend::Hip>(&spec, &root, arch)?;
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
                let (dtypes, spec_builder): (Vec<MmqDtype>, fn(MmqDtype) -> SweepSpec) = match dtype
                {
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
                    "Q4_K_lds_tiled" => (vec![MmqDtype::Q4KLdsTiled], SweepSpec::v1_4_prefill),
                    "Q5_K_lds_tiled" => (vec![MmqDtype::Q5KLdsTiled], SweepSpec::v1_4_prefill),
                    "Q5_K_lds_tiled_x8" => (vec![MmqDtype::Q5KLdsTiledX8], SweepSpec::v1_4_prefill),
                    "Q6_K_lds_tiled" => (vec![MmqDtype::Q6KLdsTiled], SweepSpec::v1_4_prefill),
                    "Q6_K_lds_tiled_x8" => (vec![MmqDtype::Q6KLdsTiledX8], SweepSpec::v1_4_prefill),
                    "IQ4_XS_lds_tiled" => (vec![MmqDtype::Iq4XsLdsTiled], SweepSpec::v1_4_prefill),
                    "IQ4_XS_lds_tiled_x8" => (vec![MmqDtype::Iq4XsLdsTiledX8], SweepSpec::v1_4_prefill),
                    "Q4_K_lds_tiled_y64" => (vec![MmqDtype::Q4KLdsTiledY64], SweepSpec::v1_4_prefill),
                    "Q4_K_lds_tiled_y64_x8" => (vec![MmqDtype::Q4KLdsTiledY64X8], SweepSpec::v1_4_prefill),
                    "Q3_K_lds_tiled" => (vec![MmqDtype::Q3KLdsTiled], SweepSpec::v1_4_prefill),
                    "Q3_K_lds_tiled_x8" => (vec![MmqDtype::Q3KLdsTiledX8], SweepSpec::v1_4_prefill),
                    "IQ4_NL_lds_tiled" => (vec![MmqDtype::Iq4NlLdsTiled], SweepSpec::v1_4_prefill),
                    "IQ4_NL_lds_tiled_x8" => (vec![MmqDtype::Iq4NlLdsTiledX8], SweepSpec::v1_4_prefill),
                    "Q8_0_lds_tiled" => (vec![MmqDtype::Q8_0LdsTiled], SweepSpec::v1_4_prefill),
                    "Q5_K" | "Q5_K_wave64" => (vec![MmqDtype::Q5KWave64], SweepSpec::v1_4_prefill),
                    "Q6_K" | "Q6_K_4warp" => (vec![MmqDtype::Q6K4Warp], SweepSpec::v1_4_prefill),
                    "Q6_K_wave64" => (vec![MmqDtype::Q6KWave64], SweepSpec::v1_4_prefill),
                    "Q8_K" | "Q8_K_wave64" => (vec![MmqDtype::Q8KWave64], SweepSpec::v1_4_prefill),
                    "Q2_K" | "Q2_K_wave64" => (vec![MmqDtype::Q2KWave64], SweepSpec::v1_4_prefill),
                    "Q3_K" | "Q3_K_wave64" => (vec![MmqDtype::Q3KWave64], SweepSpec::v1_4_prefill),
                    "IQ4_XS" | "IQ4_XS_wave64" => {
                        (vec![MmqDtype::Iq4XsWave64], SweepSpec::v1_4_prefill)
                    }
                    "IQ3_S" | "IQ3_S_wave64" => {
                        (vec![MmqDtype::Iq3SWave64], SweepSpec::v1_4_prefill)
                    }
                    "IQ4_NL" | "IQ4_NL_wave64" => {
                        (vec![MmqDtype::Iq4NlWave64], SweepSpec::v1_4_prefill)
                    }
                    "IQ3_XXS" | "IQ3_XXS_wave64" => {
                        (vec![MmqDtype::Iq3XxsWave64], SweepSpec::v1_4_prefill)
                    }
                    "IQ2_XXS" | "IQ2_XXS_wave64" => {
                        (vec![MmqDtype::Iq2XxsWave64], SweepSpec::v1_4_prefill)
                    }
                    "IQ2_XS" | "IQ2_XS_wave64" => {
                        (vec![MmqDtype::Iq2XsWave64], SweepSpec::v1_4_prefill)
                    }
                    "IQ2_S" | "IQ2_S_wave64" => {
                        (vec![MmqDtype::Iq2SWave64], SweepSpec::v1_4_prefill)
                    }
                    "IQ1_S" | "IQ1_S_wave64" => {
                        (vec![MmqDtype::Iq1SWave64], SweepSpec::v1_4_prefill)
                    }
                    "IQ1_M" | "IQ1_M_wave64" => {
                        (vec![MmqDtype::Iq1MWave64], SweepSpec::v1_4_prefill)
                    }
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
                            | MmqDtype::Q4KLdsTiled
                            | MmqDtype::Q5KLdsTiled
                            | MmqDtype::Q5KLdsTiledX8
                            | MmqDtype::Q6KLdsTiled
                            | MmqDtype::Q6KLdsTiledX8
                            | MmqDtype::Iq4XsLdsTiled
                            | MmqDtype::Iq4XsLdsTiledX8
                            | MmqDtype::Q4KLdsTiledY64
                            | MmqDtype::Q4KLdsTiledY64X8
                            | MmqDtype::Q3KLdsTiled
                            | MmqDtype::Q3KLdsTiledX8
                            | MmqDtype::Iq4NlLdsTiled
                            | MmqDtype::Iq4NlLdsTiledX8
                            | MmqDtype::Q8_0LdsTiled
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
                    cert.pass,
                    cert.results.len(),
                    cert.rig
                );
                Ok(())
            }
            other => {
                if let Some(res) = flambeau_bench::sweep_pointwise_cb::run_pointwise_sweep::<
                    flambeau_bench::backend::Hip,
                >(other, &repo_root(), arch)
                {
                    let cert = res?;
                    println!(
                        "sweep {other}: pass={} shapes={} rig={}",
                        cert.pass,
                        cert.results.len(),
                        cert.rig
                    );
                    return Ok(());
                }
                if let Some(run) = find_simple_sweep(other) {
                    let _ = dtype;
                    let cert = run(&repo_root())?;
                    println!(
                        "sweep {other}: pass={} shapes={} rig={}",
                        cert.pass,
                        cert.results.len(),
                        cert.rig
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

fn autotune(arch: &str, op: &str, dtype: &str, emit: bool, check: bool) -> Result<()> {
    if op == "qmatmul_mmq" {
        return autotune_mmq(arch, dtype);
    }
    if op != "qmatmul" {
        anyhow::bail!("autotune supports --op qmatmul (mmvq decode) or qmatmul_mmq (prefill)");
    }
    if emit && check {
        anyhow::bail!("autotune: --emit and --check are mutually exclusive");
    }
    #[cfg(any(feature = "hip_sweep", feature = "cuda_sweep"))]
    {
        use flambeau_bench::sweep_mmvq::{autotune as run, AutotuneMode, Dtype};
        let mode = if check {
            AutotuneMode::Check
        } else if emit {
            AutotuneMode::Emit
        } else {
            AutotuneMode::DryRun
        };
        // Config-variant fan-out is arch-aware: CUDA only has the templated
        // variants, so the candidate set for sm_86 is the ported subset.
        let cuda = arch == "sm_86";
        let candidates: Vec<Dtype> = match dtype {
            "Q8_0" if cuda => vec![
                Dtype::Q8_0Dp4a,
                Dtype::Q8_0T128,
                Dtype::Q8_0T128VDR2,
                Dtype::Q8_0R2,
                Dtype::Q8_0R4,
            ],
            "Q8_0" => vec![
                Dtype::Q8_0,
                Dtype::Q8_0Dp4a,
                Dtype::Q8_0T128,
                Dtype::Q8_0T128VDR2,
                Dtype::Q8_0R2,
                Dtype::Q8_0R4,
            ],
            // Q4_0 m=1 decode thread-count tune on sm_86: the 256-thread
            // base vs the templated 128-thread and t128+VDR2 candidates. HIP keeps
            // the single productionised base kernel (its t128 hand kernel is
            // gfx906-only and dispatched through MMQ, not this row).
            "Q4_0" if cuda => vec![Dtype::Q4_0, Dtype::Q4_0T128, Dtype::Q4_0T128VDR2],
            // The legacy quants are each a single productionised kernel (no
            // shape-dependent variant choice) — a degenerate one-candidate
            // rank that just emits the dispatch row, the same on either backend.
            "Q4_0" => vec![Dtype::Q4_0],
            "Q4_1" => vec![Dtype::Q4_1Dp4a],
            "Q5_0" => vec![Dtype::Q5_0],
            "Q5_1" => vec![Dtype::Q5_1],
            "Q2_K" => vec![Dtype::Q2KDp4a],
            "Q3_K" => vec![Dtype::Q3KDp4a],
            "Q4_K" => vec![Dtype::Q4KR2Dp4a],
            "Q5_K" => vec![Dtype::Q5KDp4a],
            "Q6_K" => vec![Dtype::Q6KDP4A],
            "Q8_K" => vec![Dtype::Q8K],
            "IQ4_NL" => vec![Dtype::Iq4NlDp4a],
            "IQ4_XS" => vec![Dtype::Iq4XsDp4a],
            "IQ2_XXS" => vec![Dtype::Iq2XxsDp4a],
            "IQ3_XXS" => vec![Dtype::Iq3XxsDp4a],
            "IQ3_S" => vec![Dtype::Iq3SDp4a],
            "IQ2_XS" => vec![Dtype::Iq2XsDp4a],
            "IQ2_S" => vec![Dtype::Iq2SDp4a],
            "IQ1_S" => vec![Dtype::Iq1SDp4a],
            "IQ1_M" => vec![Dtype::Iq1MDp4a],
            other => anyhow::bail!(
                "autotune: only --dtype Q8_0 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K / IQ4_NL / IQ4_XS / IQ2_XXS / IQ3_XXS / IQ3_S / IQ2_XS / IQ2_S / IQ1_S / IQ1_M are wired so far (got {other:?})"
            ),
        };
        let m_grid = [1usize, 8, 16];
        let k_grid = [2048usize, 4096];
        let root = repo_root();
        match arch {
            #[cfg(feature = "hip_sweep")]
            "gfx906" => {
                run::<flambeau_bench::backend::Hip>(arch, &candidates, &m_grid, &k_grid, &root, mode)
            }
            #[cfg(feature = "cuda_sweep")]
            "sm_86" => {
                run::<flambeau_bench::backend::Cuda>(arch, &candidates, &m_grid, &k_grid, &root, mode)
            }
            other => anyhow::bail!(
                "autotune: unsupported --arch {other:?} (need gfx906 / sm_86 with the matching feature)"
            ),
        }
    }
    #[cfg(not(any(feature = "hip_sweep", feature = "cuda_sweep")))]
    {
        let _ = (arch, dtype, emit, check);
        anyhow::bail!("autotune requires --features hip_sweep and/or cuda_sweep")
    }
}

fn autotune_mmq(arch: &str, dtype: &str) -> Result<()> {
    #[cfg(any(feature = "hip_sweep", feature = "cuda_sweep"))]
    {
        use flambeau_bench::sweep_mmvq::autotune_mmq as run;
        // M = n_tokens grid spanning the prefill dispatch bands (32..127 wave64,
        // >=128 4warp_lds), at a representative projection (n_out, k).
        let m_grid = [64usize, 128, 512];
        let (n_out, k) = (4096usize, 4096);
        match arch {
            #[cfg(feature = "hip_sweep")]
            "gfx906" => run::<flambeau_bench::backend::Hip>(arch, dtype, &m_grid, n_out, k),
            #[cfg(feature = "cuda_sweep")]
            "sm_86" => run::<flambeau_bench::backend::Cuda>(arch, dtype, &m_grid, n_out, k),
            other => anyhow::bail!("autotune --op qmatmul_mmq: unsupported --arch {other:?}"),
        }
    }
    #[cfg(not(any(feature = "hip_sweep", feature = "cuda_sweep")))]
    {
        let _ = (arch, dtype);
        anyhow::bail!("autotune requires --features hip_sweep and/or cuda_sweep")
    }
}

/// Map a `--dtype` string to a fused gate+up sweep target, if it names one.
#[cfg(any(feature = "hip_sweep", feature = "cuda_sweep"))]
fn gate_up_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::GateUpDtype> {
    use flambeau_bench::sweep_mmvq::GateUpDtype;
    match dtype {
        "Q8_0_gate_up" => Some(GateUpDtype::Q8_0),
        "Q4_0_gate_up" => Some(GateUpDtype::Q4_0),
        "Q4_0_gate_up_t128" => Some(GateUpDtype::Q4_0T128),
        "Q4_1_gate_up" => Some(GateUpDtype::Q4_1),
        "Q5_K_gate_up" => Some(GateUpDtype::Q5K),
        _ => None,
    }
}

/// Map a `--dtype` string to a batched (N-slot) sweep target, if it names one.
#[cfg(any(feature = "hip_sweep", feature = "cuda_sweep"))]
fn batched_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::BatchedDtype> {
    use flambeau_bench::sweep_mmvq::BatchedDtype;
    match dtype {
        "Q8_0_batched" => Some(BatchedDtype::Q8_0),
        "Q4_0_batched" => Some(BatchedDtype::Q4_0),
        "Q4_1_batched" => Some(BatchedDtype::Q4_1),
        "Q4_K_batched" => Some(BatchedDtype::Q4K),
        "Q6_K_batched" => Some(BatchedDtype::Q6K),
        "Q5_K_batched" => Some(BatchedDtype::Q5K),
        "Q8_0_row_tile_batched" => Some(BatchedDtype::Q8_0RowTile),
        "Q4_0_row_tile_batched" => Some(BatchedDtype::Q4_0RowTile),
        "Q5_K_row_tile_batched" => Some(BatchedDtype::Q5KRowTile),
        "Q4_K_dp4a_row_tile_batched" => Some(BatchedDtype::Q4KDp4aRowTile),
        "Q5_K_dp4a_row_tile_batched" => Some(BatchedDtype::Q5KDp4aRowTile),
        "IQ4_XS_dp4a_row_tile_batched" => Some(BatchedDtype::Iq4XsDp4aRowTile),
        "IQ3_S_dp4a_row_tile_batched" => Some(BatchedDtype::Iq3SDp4aRowTile),
        "IQ3_XXS_dp4a_row_tile_batched" => Some(BatchedDtype::Iq3XxsDp4aRowTile),
        "IQ2_S_dp4a_row_tile_batched" => Some(BatchedDtype::Iq2SDp4aRowTile),
        "Q6_K_dp4a_row_tile_batched" => Some(BatchedDtype::Q6KDp4aRowTile),
        _ => None,
    }
}

/// Map a `--dtype` string to a fused gate+up batched sweep target.
#[cfg(any(feature = "hip_sweep", feature = "cuda_sweep"))]
fn gate_up_batched_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::GateUpBatchedDtype> {
    use flambeau_bench::sweep_mmvq::GateUpBatchedDtype;
    match dtype {
        "Q4_0_gate_up_row_tile_batched" => Some(GateUpBatchedDtype::Q4_0RowTile),
        _ => None,
    }
}

/// Map a `--dtype` string to an MMQ correctness-oracle sweep target.
#[cfg(any(feature = "hip_sweep", feature = "cuda_sweep"))]
fn mmq_oracle_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::MmqOracleDtype> {
    use flambeau_bench::sweep_mmvq::MmqOracleDtype;
    match dtype {
        "Q8_0_mmq_oracle" => Some(MmqOracleDtype::Q8_0),
        _ => None,
    }
}

/// Map a `--dtype` string to a tiled MMQ GEMM sweep target.
#[cfg(any(feature = "hip_sweep", feature = "cuda_sweep"))]
fn mmq_gemm_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::MmqGemmDtype> {
    use flambeau_bench::sweep_mmvq::MmqGemmDtype;
    match dtype {
        "Q8_0_mmq_wave64" => Some(MmqGemmDtype::Q8_0Wave64),
        "Q4_0_mmq_wave64" => Some(MmqGemmDtype::Q4_0Wave64),
        "Q4_1_mmq_wave64" => Some(MmqGemmDtype::Q4_1Wave64),
        "Q5_0_mmq_wave64" => Some(MmqGemmDtype::Q5_0Wave64),
        "Q5_1_mmq_wave64" => Some(MmqGemmDtype::Q5_1Wave64),
        "Q6_K_mmq_wave64" => Some(MmqGemmDtype::Q6KWave64),
        "Q4_K_mmq_wave64" => Some(MmqGemmDtype::Q4KWave64),
        "Q5_K_mmq_wave64" => Some(MmqGemmDtype::Q5KWave64),
        "Q2_K_mmq_wave64" => Some(MmqGemmDtype::Q2KWave64),
        "Q3_K_mmq_wave64" => Some(MmqGemmDtype::Q3KWave64),
        "Q8_K_mmq_wave64" => Some(MmqGemmDtype::Q8KWave64),
        "IQ4_NL_mmq_wave64" => Some(MmqGemmDtype::Iq4NlWave64),
        "IQ4_XS_mmq_wave64" => Some(MmqGemmDtype::Iq4XsWave64),
        "IQ2_XXS_mmq_wave64" => Some(MmqGemmDtype::Iq2XxsWave64),
        "IQ2_XS_mmq_wave64" => Some(MmqGemmDtype::Iq2XsWave64),
        "IQ2_S_mmq_wave64" => Some(MmqGemmDtype::Iq2SWave64),
        "IQ3_XXS_mmq_wave64" => Some(MmqGemmDtype::Iq3XxsWave64),
        "IQ3_S_mmq_wave64" => Some(MmqGemmDtype::Iq3SWave64),
        "IQ1_S_mmq_wave64" => Some(MmqGemmDtype::Iq1SWave64),
        "IQ1_M_mmq_wave64" => Some(MmqGemmDtype::Iq1MWave64),
        "Q8_0_mmq_wave64_tile16" => Some(MmqGemmDtype::Q8_0Wave64Tile16),
        _ => None,
    }
}

fn mmq_4warp_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::Mmq4WarpDtype> {
    use flambeau_bench::sweep_mmvq::Mmq4WarpDtype;
    match dtype {
        "Q8_0_mmq_4warp" => Some(Mmq4WarpDtype::Q8_0),
        "Q4_K_mmq_4warp" => Some(Mmq4WarpDtype::Q4K),
        "Q6_K_mmq_4warp" => Some(Mmq4WarpDtype::Q6K),
        _ => None,
    }
}

fn mmq_4warp_lds_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::Mmq4WarpLdsDtype> {
    use flambeau_bench::sweep_mmvq::Mmq4WarpLdsDtype;
    match dtype {
        "Q4_0_mmq_4warp_lds" => Some(Mmq4WarpLdsDtype::Q4_0),
        "Q4_1_mmq_4warp_lds" => Some(Mmq4WarpLdsDtype::Q4_1),
        "Q4_K_mmq_lds_tiled" => Some(Mmq4WarpLdsDtype::Q4K),
        _ => None,
    }
}

fn moe_mmvq_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::MoeMmvqDtype> {
    use flambeau_bench::sweep_mmvq::MoeMmvqDtype;
    match dtype {
        "Q8_0_moe_mmvq" => Some(MoeMmvqDtype::Q8_0),
        "Q4_0_moe_mmvq" => Some(MoeMmvqDtype::Q4_0),
        "Q4_1_moe_mmvq" => Some(MoeMmvqDtype::Q4_1),
        "Q5_0_moe_mmvq" => Some(MoeMmvqDtype::Q5_0),
        "Q5_1_moe_mmvq" => Some(MoeMmvqDtype::Q5_1),
        "Q4_K_r2_moe_mmvq" => Some(MoeMmvqDtype::Q4KR2),
        "Q4_K_r4_moe_mmvq" => Some(MoeMmvqDtype::Q4KR4),
        "Q5_K_r2_moe_mmvq" => Some(MoeMmvqDtype::Q5KR2),
        "Q6_K_r2_moe_mmvq" => Some(MoeMmvqDtype::Q6KR2),
        "Q2_K_r2_moe_mmvq" => Some(MoeMmvqDtype::Q2KR2),
        "Q3_K_r2_moe_mmvq" => Some(MoeMmvqDtype::Q3KR2),
        "IQ4_NL_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq4NlR2),
        "IQ4_XS_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq4XsR2),
        "IQ2_XXS_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq2XxsR2),
        "IQ2_XS_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq2XsR2),
        "IQ2_S_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq2SR2),
        "IQ3_XXS_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq3XxsR2),
        "IQ3_S_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq3SR2),
        "IQ1_S_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq1SR2),
        "IQ1_M_r2_moe_mmvq" => Some(MoeMmvqDtype::Iq1MR2),
        _ => None,
    }
}

fn moe_mmq_lds_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::MoeMmqLdsDtype> {
    use flambeau_bench::sweep_mmvq::MoeMmqLdsDtype;
    match dtype {
        "Q4_K_gate_up_lds_tiled_moe_mmq" => Some(MoeMmqLdsDtype::Q4KGateUp),
        "Q4_K_down_lds_tiled_moe_mmq" => Some(MoeMmqLdsDtype::Q4KDown),
        _ => None,
    }
}

fn moe_mmq_tile8_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::MoeMmqTile8Dtype> {
    use flambeau_bench::sweep_mmvq::MoeMmqTile8Dtype;
    match dtype {
        "Q8_0_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q8_0GateUp),
        "Q8_0_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q8_0Down),
        "Q4_0_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q4_0GateUp),
        "Q4_0_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q4_0Down),
        "Q4_1_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q4_1GateUp),
        "Q4_1_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q4_1Down),
        "Q5_0_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q5_0GateUp),
        "Q5_0_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q5_0Down),
        "Q5_1_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q5_1GateUp),
        "Q5_1_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q5_1Down),
        "Q4_K_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q4KGateUp),
        "Q4_K_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q4KDown),
        "Q5_K_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q5KGateUp),
        "Q5_K_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q5KDown),
        "Q6_K_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q6KGateUp),
        "Q6_K_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q6KDown),
        "Q2_K_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q2KGateUp),
        "Q2_K_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q2KDown),
        "Q3_K_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q3KGateUp),
        "Q3_K_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Q3KDown),
        "IQ4_NL_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq4NlGateUp),
        "IQ4_NL_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq4NlDown),
        "IQ4_XS_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq4XsGateUp),
        "IQ4_XS_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq4XsDown),
        "IQ2_XXS_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq2XxsGateUp),
        "IQ2_XXS_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq2XxsDown),
        "IQ2_XS_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq2XsGateUp),
        "IQ2_XS_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq2XsDown),
        "IQ2_S_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq2SGateUp),
        "IQ2_S_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq2SDown),
        "IQ3_XXS_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq3XxsGateUp),
        "IQ3_XXS_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq3XxsDown),
        "IQ3_S_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq3SGateUp),
        "IQ3_S_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq3SDown),
        "IQ1_S_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq1SGateUp),
        "IQ1_S_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq1SDown),
        "IQ1_M_gate_up_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq1MGateUp),
        "IQ1_M_down_tile8_moe_mmq" => Some(MoeMmqTile8Dtype::Iq1MDown),
        _ => None,
    }
}

fn moe_gate_up_dtype(dtype: &str) -> Option<flambeau_bench::sweep_mmvq::MoeGateUpDtype> {
    use flambeau_bench::sweep_mmvq::MoeGateUpDtype;
    match dtype {
        "Q8_0_gate_up_moe_mmvq" => Some(MoeGateUpDtype::Q8_0),
        "Q4_0_gate_up_moe_mmvq" => Some(MoeGateUpDtype::Q4_0),
        "Q4_K_gate_up_moe_mmvq" => Some(MoeGateUpDtype::Q4K),
        "Q4_K_gate_up_r4_moe_mmvq" => Some(MoeGateUpDtype::Q4KR4),
        "Q4_K_gate_up_r4_sorted_moe_mmvq" => Some(MoeGateUpDtype::Q4KR4Sorted),
        _ => None,
    }
}

fn f16_sweep_kind(dtype: &str) -> Option<flambeau_bench::sweep_mmvq_f16::F16SweepKind> {
    use flambeau_bench::sweep_mmvq_f16::F16SweepKind;
    match dtype {
        "F16_mmvq" => Some(F16SweepKind::Mmvq),
        "F16_mmq" => Some(F16SweepKind::Mmq),
        "F16_mmq_tile" => Some(F16SweepKind::MmqTile),
        _ => None,
    }
}

#[cfg(feature = "cuda_sweep")]
fn sweep_cuda(op: &str, dtype: &str) -> Result<()> {
    use flambeau_bench::backend::Cuda;
    use flambeau_bench::sweep_mmvq::{run_sweep, Dtype, SweepSpec};

    if let Some(res) =
        flambeau_bench::sweep_pointwise_cb::run_pointwise_sweep::<Cuda>(op, &repo_root(), "sm_86")
    {
        let cert = res?;
        println!(
            "sweep {op}: pass={} shapes={} rig={}",
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if op == "attention_decode_batched_splitk" {
        let cert = flambeau_bench::sweep_attention_batched_splitk::run_sweep::<Cuda>(
            &repo_root(),
            "sm_86",
        )?;
        println!(
            "sweep attention_decode_batched_splitk: pass={} shapes={} rig={}",
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if op == "attention_decode_f16_splitk_dpos" {
        let cert = flambeau_bench::sweep_attention_splitk_dpos::run_sweep::<Cuda>(
            &repo_root(),
            "sm_86",
        )?;
        println!(
            "sweep attention_decode_f16_splitk_dpos: pass={} shapes={} rig={}",
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if op == "kv_append_v_unit_norm_f16_dpos" {
        let cert = flambeau_bench::sweep_kv_append_v_unit_norm_dpos::run_sweep::<Cuda>(
            &repo_root(),
            "sm_86",
        )?;
        println!(
            "sweep kv_append_v_unit_norm_f16_dpos: pass={} shapes={} rig={}",
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if op == "embed_gather_f16_dpos" {
        let cert = flambeau_bench::sweep_embed_gather_dpos::run_sweep::<Cuda>(
            &repo_root(),
            "sm_86",
        )?;
        println!(
            "sweep embed_gather_f16_dpos: pass={} shapes={} rig={}",
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if op == "attention_decode_q8_kv_splitk_dpos" {
        let cert = flambeau_bench::sweep_attention_q8_kv_splitk_dpos::run_sweep::<Cuda>(
            &repo_root(),
            "sm_86",
        )?;
        println!(
            "sweep attention_decode_q8_kv_splitk_dpos: pass={} shapes={} rig={}",
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if op == "quantize_f16_q8_0_dpos" {
        let cert = flambeau_bench::sweep_quantize_f16_q8_0_dpos::run_sweep::<Cuda>(
            &repo_root(),
            "sm_86",
        )?;
        println!(
            "sweep quantize_f16_q8_0_dpos: pass={} shapes={} rig={}",
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if op != "qmatmul" {
        anyhow::bail!(
            "--arch sm_86: only --op qmatmul (mmvq) has CUDA kernels; \
             other ops are gfx906-only (Stream F port in progress, see P3-STREAM-F-PORT.md)"
        );
    }
    if dtype == "Q4_0_kv_f16dst" {
        let cert = flambeau_bench::sweep_mmvq::run_kv_f16dst_sweep::<Cuda>(&repo_root(), "sm_86")?;
        println!(
            "sweep mmvq_q4_0_kv_f16dst: pass={} shapes={} rig={}",
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(gu) = gate_up_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_gate_up_sweep;
        let cert = run_gate_up_sweep::<Cuda>(gu, &repo_root(), "sm_86")?;
        println!(
            "sweep qmatmul_gate_up ({}): pass={} shapes={} rig={}",
            gu.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(bd) = batched_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_batched_sweep;
        let cert = run_batched_sweep::<Cuda>(bd, &repo_root(), "sm_86")?;
        println!(
            "sweep qmatmul_batched ({}): pass={} shapes={} rig={}",
            bd.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(gb) = gate_up_batched_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_gate_up_batched_sweep;
        let cert = run_gate_up_batched_sweep::<Cuda>(gb, &repo_root(), "sm_86")?;
        println!(
            "sweep qmatmul_gate_up_batched ({}): pass={} shapes={} rig={}",
            gb.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(od) = mmq_oracle_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_mmq_oracle_sweep;
        let cert = run_mmq_oracle_sweep::<Cuda>(od, &repo_root(), "sm_86")?;
        println!(
            "sweep qmatmul_mmq_oracle ({}): pass={} shapes={} rig={}",
            od.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(gd) = mmq_gemm_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_mmq_gemm_sweep;
        let cert = run_mmq_gemm_sweep::<Cuda>(gd, &repo_root(), "sm_86")?;
        println!(
            "sweep qmatmul_mmq_gemm ({}): pass={} shapes={} rig={}",
            gd.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(wd) = mmq_4warp_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_mmq_4warp_sweep;
        let cert = run_mmq_4warp_sweep::<Cuda>(wd, &repo_root(), "sm_86")?;
        println!(
            "sweep qmatmul_mmq_4warp ({}): pass={} shapes={} rig={}",
            wd.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(md) = moe_mmvq_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_moe_mmvq_sweep;
        let cert = run_moe_mmvq_sweep::<Cuda>(md, &repo_root(), "sm_86")?;
        println!(
            "sweep indexed_moe_mmvq ({}): pass={} shapes={} rig={}",
            md.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(gu) = moe_gate_up_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_moe_gate_up_sweep;
        let cert = run_moe_gate_up_sweep::<Cuda>(gu, &repo_root(), "sm_86")?;
        println!(
            "sweep indexed_moe_mmvq_gate_up ({}): pass={} shapes={} rig={}",
            gu.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(t8) = moe_mmq_tile8_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_moe_mmq_tile8_sweep;
        let cert = run_moe_mmq_tile8_sweep::<Cuda>(t8, &repo_root(), "sm_86")?;
        println!(
            "sweep indexed_moe_mmq_tile8 ({}): pass={} shapes={} rig={}",
            t8.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(ld) = moe_mmq_lds_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_moe_mmq_lds_sweep;
        let cert = run_moe_mmq_lds_sweep::<Cuda>(ld, &repo_root(), "sm_86")?;
        println!(
            "sweep indexed_moe_mmq_lds_tiled ({}): pass={} shapes={} rig={}",
            ld.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(ld) = mmq_4warp_lds_dtype(dtype) {
        use flambeau_bench::sweep_mmvq::run_mmq_4warp_lds_sweep;
        let cert = run_mmq_4warp_lds_sweep::<Cuda>(ld, &repo_root(), "sm_86")?;
        println!(
            "sweep qmatmul_mmq_4warp_lds ({}): pass={} shapes={} rig={}",
            ld.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    if let Some(fk) = f16_sweep_kind(dtype) {
        use flambeau_bench::sweep_mmvq_f16::run_f16_sweep;
        let cert = run_f16_sweep::<Cuda>(fk, &repo_root(), "sm_86")?;
        println!(
            "sweep qmatmul_f16 ({}): pass={} shapes={} rig={}",
            fk.name(),
            cert.pass,
            cert.results.len(),
            cert.rig
        );
        return Ok(());
    }
    let dtypes: Vec<Dtype> = match dtype {
        "Q8_0" | "Q8_0_dp4a" => vec![Dtype::Q8_0Dp4a],
        "Q8_0_t128" => vec![Dtype::Q8_0T128],
        "Q8_0_t128_vdr2" => vec![Dtype::Q8_0T128VDR2],
        "Q8_0_r2" => vec![Dtype::Q8_0R2],
        "Q8_0_r4" => vec![Dtype::Q8_0R4],
        "Q4_0" => vec![Dtype::Q4_0, Dtype::Q4_0T128, Dtype::Q4_0T128VDR2],
        "Q4_0_t128" => vec![Dtype::Q4_0T128],
        "Q4_0_t128_vdr2" => vec![Dtype::Q4_0T128VDR2],
        "Q4_1" => vec![Dtype::Q4_1Dp4a],
        "Q5_0" => vec![Dtype::Q5_0],
        "Q5_1" => vec![Dtype::Q5_1],
        "Q2_K" => vec![Dtype::Q2KDp4a],
        "Q3_K" => vec![Dtype::Q3KDp4a],
        "Q4_K" => vec![Dtype::Q4KR2Dp4a],
        "Q5_K" => vec![Dtype::Q5KDp4a],
        "Q6_K" => vec![Dtype::Q6KDP4A],
        "Q8_K" => vec![Dtype::Q8K],
        "IQ4_NL" => vec![Dtype::Iq4Nl, Dtype::Iq4NlR2, Dtype::Iq4NlDp4a],
        "IQ4_XS" => vec![Dtype::Iq4Xs, Dtype::Iq4XsR2, Dtype::Iq4XsDp4a],
        "IQ2_XXS" => vec![Dtype::Iq2Xxs, Dtype::Iq2XxsR2, Dtype::Iq2XxsDp4a],
        "IQ3_XXS" => vec![Dtype::Iq3Xxs, Dtype::Iq3XxsR2, Dtype::Iq3XxsDp4a],
        "IQ3_S" => vec![Dtype::Iq3S, Dtype::Iq3SR2, Dtype::Iq3SDp4a],
        "IQ2_XS" => vec![Dtype::Iq2Xs, Dtype::Iq2XsR2, Dtype::Iq2XsDp4a],
        "IQ2_S" => vec![Dtype::Iq2S, Dtype::Iq2SR2, Dtype::Iq2SDp4a],
        "IQ1_S" => vec![Dtype::Iq1S, Dtype::Iq1SR2, Dtype::Iq1SDp4a],
        "IQ1_M" => vec![Dtype::Iq1M, Dtype::Iq1MR2, Dtype::Iq1MDp4a],
        "all" => vec![
            Dtype::Q8_0Dp4a,
            Dtype::Q4_0,
            Dtype::Q4_1Dp4a,
            Dtype::Q5_0,
            Dtype::Q5_1,
            Dtype::Q2KDp4a,
            Dtype::Q3KDp4a,
            Dtype::Q4KR2Dp4a,
            Dtype::Q5KDp4a,
            Dtype::Q6KDP4A,
            Dtype::Q8K,
            Dtype::Iq4NlDp4a,
            Dtype::Iq4XsDp4a,
            Dtype::Iq2XxsDp4a,
            Dtype::Iq3XxsDp4a,
            Dtype::Iq3SDp4a,
            Dtype::Iq2XsDp4a,
            Dtype::Iq2SDp4a,
            Dtype::Iq1SDp4a,
            Dtype::Iq1MDp4a,
        ],
        other => {
            anyhow::bail!("--arch sm_86: only Q8_0 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K / IQ4_NL / IQ4_XS / IQ2_XXS / IQ3_XXS / IQ3_S / IQ2_XS / IQ2_S / IQ1_S / IQ1_M mmvq dp4a are ported to CUDA so far (got {other:?})")
        }
    };
    let root = repo_root();
    for d in dtypes {
        let spec = SweepSpec::v1_3_default(d);
        let cert = run_sweep::<Cuda>(&spec, &root, "sm_86")?;
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
            (
                "qmatmul_q8_0_mmvq_single_row_gfx906",
                "mmvq_q8_0",
                1,
                2048,
                64,
            ),
            (
                "qmatmul_q4_K_mmvq_single_row_gfx906",
                "mmvq_q4_k",
                1,
                2048,
                64,
            ),
            (
                "qmatmul_q5_K_mmvq_single_row_gfx906",
                "mmvq_q5_k",
                1,
                2048,
                64,
            ),
            (
                "qmatmul_q6_K_mmvq_single_row_gfx906",
                "mmvq_q6_k",
                1,
                2048,
                64,
            ),
            (
                "qmatmul_q4_K_mmvq_nw1_r2_gfx906",
                "mmvq_q4_k_r2",
                1,
                2048,
                64,
            ),
            (
                "qmatmul_q4_k_mmvq_dp4a",
                "mmvq_q4_k_r2_dp4a",
                1,
                2048,
                64,
            ),
            (
                "qmatmul_q5_K_mmvq_nw1_r2_gfx906",
                "mmvq_q5_k_r2",
                1,
                2048,
                64,
            ),
            (
                "qmatmul_q5_k_mmvq_dp4a",
                "mmvq_q5_k_dp4a",
                1,
                2048,
                256,
            ),
            (
                "qmatmul_q6_K_mmvq_nw1_r4_gfx906",
                "mmvq_q6_k_r4",
                1,
                2048,
                64,
            ),
            (
                "qmatmul_q8_0_mmq_oracle_gfx906",
                "mmq_q8_0_oracle",
                8,
                2048,
                64,
            ),
            (
                "qmatmul_q8_0_mmq_4warp_lds_gfx906",
                "mmq_q8_0_4warp",
                128,
                2048,
                64,
            ),
            (
                "qmatmul_q8_0_mmq_wave64_gfx906",
                "mmq_q8_0_wave64",
                128,
                2048,
                4096,
            ),
            (
                "qmatmul_q4_1_mmq_4warp_lds_gfx906",
                "mmq_q4_1_4warp_lds",
                128,
                2048,
                4096,
            ),
            (
                "qmatmul_q4_K_mmq_wave64_gfx906",
                "mmq_q4_K_wave64",
                128,
                2048,
                4096,
            ),
            (
                "qmatmul_q5_K_mmq_wave64_gfx906",
                "mmq_q5_K_wave64",
                128,
                2048,
                4096,
            ),
            (
                "qmatmul_q6_K_mmq_wave64_gfx906",
                "mmq_q6_K_wave64",
                128,
                2048,
                4096,
            ),
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
        anyhow::bail!(
            "cert-check failed — {} row(s) missing or stale",
            report.failures.len()
        );
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
