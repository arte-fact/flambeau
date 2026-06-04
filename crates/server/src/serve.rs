//! `flambeau serve` entry point — loads the model + tokenizer, starts axum.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_runtime::Registry;
use tokio::sync::Mutex;
use tracing::info;

/// mesh topology selector. PP-V1 default; TP engages the
/// Qwen3MoETpModel loader + the BarP2pAllReduce-based forward path.
/// `Hybrid` adds a manual PP-of-TP composition where
/// `pp_size` contiguous layer stages each own a `tp_size`-rank TP
/// subgroup. Selection is operator-driven; flambeau does not autodetect
/// the right topology for a given rig (the bracket-bench harness in
/// produces the data, the operator picks the winner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum MeshMode {
    /// Pipeline parallelism — V1 default. `LayerAssignment` distributes
    /// whole layers across ranks; one `peer_copy_via_host` per stage
    /// transition.
    #[default]
    Pp,
    /// Tensor parallelism — every rank holds every layer (sliced).
    /// `world` ranks; intra-layer Megatron splits + BAR1 P2P AllReduce.
    Tp { world: u32 },
    /// Hybrid PP-of-TP — `pp_size` contiguous layer stages, each owning
    /// a `tp_size`-rank TP subgroup. Total ranks = `pp_size * tp_size`.
    /// Devices are interpreted in stage-major order: `--devices d0,d1,...`
    /// with `tp_size = 2`, `pp_size = 2` means stage 0 = {d0, d1},
    /// stage 1 = {d2, d3}. Forward wiring lands in f; this
    /// variant currently boots up to the loader and bails.
    Hybrid { pp_size: u32, tp_size: u32 },
}

/// Runtime config for `flambeau serve`.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub gguf_path: PathBuf,
    pub device_ids: Vec<i32>,
    pub bind_addr: SocketAddr,
    pub model_id: String,
    /// mesh topology. Defaults to `Pp` for V1 callers that
    /// don't set the field explicitly (constructors use struct-update
    /// syntax with `..Default::default()`).
    pub mesh_mode: MeshMode,
    /// **#230 P2.11a** — optional path to a `qwen3` arch embedding
    /// GGUF loaded alongside the chat model. `None` disables the
    /// embedding subsystem; `/v1/embeddings` (#231) returns 503 when
    /// unset.
    pub embedding_gguf_path: Option<PathBuf>,
    /// **#230 P2.11a** — HIP device id for the embedding model. Must
    /// be one of `device_ids`; the embedding model reuses the
    /// chat-cluster's `HipDevice` handle for the matching rank.
    pub embedding_device_id: Option<i32>,
    /// Number of concurrent inflight decode slots (1-32). VRAM scales
    /// linearly with this value (each slot owns its own KV cache).
    pub inflight_slots: usize,
    /// Prefill chunk size in tokens. Default 512 is the production sweet
    /// spot across pp/tp/hybrid topologies; tune for short-prompt TTFT.
    pub prefill_ubatch: usize,
    /// Per-chunk prefill budget used by the K4c mixed-batch scheduler.
    /// Smaller chunks improve short-request TTFT under load
    /// (Sarathi-Serve trade) at the cost of slightly higher long-request
    /// prefill latency. Default 512.
    pub prefill_chunk_tokens: usize,
    /// PagedAttention activation. `None` (default) uses the legacy
    /// contiguous per-slot KV slab. `Some(N)` for `N > 1` builds the
    /// page pool with `N` pages per layer — pair with a high
    /// `inflight_slots` to pack concurrent requests into the same
    /// VRAM. `Some(1)` clamps to `max_slots * max_pages_per_slot`
    /// (correctness only, no VRAM win). Activation also unlocks the
    /// paged decode + prefill attention kernels in `standard_attn`.
    pub paged_kv_pages: Option<usize>,
    /// #232 admission-control queue depth beyond the inflight pool. 0
    /// disables (legacy unbounded queue). Default 16.
    pub max_queue_depth: usize,
    /// Decode-batching coalescence window (microseconds). The leader
    /// thread sleeps this long before draining `batched_pending` so
    /// concurrent decode requests join the same batched forward.
    /// Default 1500 µs; raise for higher concurrency at the cost of
    /// per-step latency.
    pub decode_batch_window_us: u64,
    /// Clamp the model's `context_length`. `None` keeps the GGUF's
    /// architectural max; many GGUFs ship 262 144 which OOMs the per-rank
    /// KV cache on 16 GB MI50. Only shrinks.
    pub ctx_cap: Option<usize>,
    /// On-device GPU sampler (top-k + softmax + penalties on the head
    /// rank, single DtoH per token). Drops sampler cost from ~12 ms to
    /// ~0 ms on chat workloads.
    pub gpu_sampler: bool,
    /// Batched-decode scheduler. Coalesces concurrent decode steps via
    /// the inflight-slot leader. Required for N>1 throughput.
    pub batched_decode: bool,
    /// #229 prompt prefix cache. Caches prompt-prefix KV across requests.
    pub prefix_cache: bool,
    /// Prefix-cache LRU size in GB. Tune to free VRAM minus model + KV.
    pub prefix_cache_max_gb: f64,
    /// KV cache layout: `"f16"` or `"q8"`.
    pub kv: String,
    /// Route TP/Hybrid AllReduce through the host-bounce coordinator
    /// instead of BAR1 P2P, making greedy (temp=0) output bit-
    /// reproducible. The BAR1 aperture read is non-coherent on gfx906
    /// (see `doc/DETERMINISM_INVESTIGATION.md`); host-bounce trades
    /// decode throughput for determinism. Off by default.
    pub deterministic: bool,
    /// Default system prompt prepended to chat-template requests when
    /// none is provided.
    pub default_system: Option<String>,
    /// /v1/embeddings per-prompt token cap.
    pub embedding_max_tokens: usize,
    /// Override the per-rank layer split. `None` falls back to the
    /// default uniform `num_layers / n_groups` split. When `Some`, the
    /// vector length must equal the PP rank count (== `device_ids.len()`
    /// for `Pp`, == `pp_size` for `Hybrid`), and the sum must equal the
    /// model's `num_layers`.
    ///
    /// Useful for hybrid arches like Qwen3.6 where the recurrent (GDN)
    /// layers are much lighter than the full-attn layers. With a
    /// uniform split, early ranks (holding heavier layers) become the
    /// pipeline bottleneck; rocprofv3 trace on PP4 Qwen3.6-27B showed
    /// rank 0 doing 1.8× the kernel work of rank 3.
    pub layer_split: Option<Vec<usize>>,
}


/// Blocking serve loop — loads the model, starts the HTTP server, runs
/// until terminated. Caller owns the tokio runtime.
pub async fn serve(cfg: ServeConfig, registry: Registry) -> Result<()> {
    info!(forward_stack = "v2", "flambeau serve: loading model");
    info!(?cfg, "flambeau serve config");

    let gguf = GgufFile::open(&cfg.gguf_path)
        .with_context(|| format!("open GGUF at {}", cfg.gguf_path.display()))?;

    let gguf_arch = gguf.metadata_str("general.architecture").unwrap_or("");
    let model_arch = registry
        .validate(gguf_arch)
        .with_context(|| format!("flambeau serve: unsupported GGUF arch `{gguf_arch}`"))?;
    info!(
        arch = gguf_arch,
        handler = model_arch.description(),
        "GGUF arch validated against registry"
    );

    let boot = crate::serve_common::BootMetadata::from_gguf(&gguf, &cfg)?;
    serve_inner_v2(cfg, gguf, boot).await
}

/// Arch dispatch is intentionally a `match` on the arch string in
/// `create_v2_driver`: one branch per supported arch crate. Adding a
/// new v2 arch = one new branch + a `flambeau-<arch>-v2` workspace
/// dep, with no churn at the routes/sampler/parser layer.
pub(crate) async fn serve_inner_v2(
    cfg: ServeConfig,
    gguf: GgufFile,
    boot: crate::serve_common::BootMetadata,
) -> Result<()> {
    let gguf_arch_owned = gguf
        .metadata_str("general.architecture")
        .unwrap_or("")
        .to_string();
    let gguf_arch: &str = &gguf_arch_owned;

    let kv_layout = match cfg.kv.as_str() {
        "f16" => flambeau_forward::KvLayout::F16Contig,
        "q8" => flambeau_forward::KvLayout::Q8Contig,
        other => anyhow::bail!(
            "--kv {other}: unknown KV cache layout (supported: f16, q8)"
        ),
    };

    let n_available = device_count().unwrap_or(0);
    for d in &cfg.device_ids {
        if *d < 0 || *d >= n_available {
            bail!("device {d} not available (have {n_available} HIP devices)");
        }
    }

    let model_cfg = parse_v2_server_model_cfg(gguf_arch, &cfg.gguf_path)?;
    let topology = topology_from_mesh(
        cfg.mesh_mode,
        &cfg.device_ids,
        model_cfg.num_layers,
        cfg.layer_split.as_deref(),
        model_cfg.kv_share_pp_boundary,
    )?;
    let topology_label: &'static str = match cfg.mesh_mode {
        MeshMode::Pp => "pp",
        MeshMode::Tp { .. } => "tp",
        MeshMode::Hybrid { .. } => "pp+tp",
    };

    let inflight_slots = cfg.inflight_slots.clamp(1, 32);
    let prefill_ubatch_raw = cfg.prefill_ubatch.max(128);
    // Mixed-batch engagement (K prefill rows + N decode rows in one
    // forward) is on by default for every arch whose Model::
    // supports_mixed_batch returns true. The ScratchPool's
    // `max_prefill_tokens` must accommodate K + N or the residual /
    // QKV / output buffers overrun, so bump `prefill_ubatch` to
    // `chunk + inflight_slots` whenever the raw value is below that.
    // Cost on archs that never engage is trivial (a few extra rows
    // of scratch capacity).
    let chunk_tokens = cfg.prefill_chunk_tokens.max(1);
    let prefill_ubatch = {
        let needed = chunk_tokens + inflight_slots;
        if prefill_ubatch_raw < needed {
            info!(
                from = prefill_ubatch_raw,
                to = needed,
                chunk_tokens,
                inflight_slots,
                "bumping prefill_ubatch to accommodate mixed-batch K+N forward"
            );
            needed
        } else {
            prefill_ubatch_raw
        }
    };
    let max_queue_depth = cfg.max_queue_depth;
    info!(
        arch = gguf_arch,
        topology = topology_label,
        inflight_slots,
        prefill_ubatch,
        max_queue_depth,
        "v2: building shared Session with max_slots=N"
    );

    drop(gguf);
    let gguf_path = cfg.gguf_path.clone();
    let chat_stops = crate::v2_handle::chat_stops_for(gguf_arch);
    let bos_id = if crate::v2_handle::wants_bos_prepend(gguf_arch) {
        boot.tokenizer.bos_id
    } else {
        None
    };

    let shared_gguf = GgufFile::open(&gguf_path)
        .with_context(|| "v2: re-open GGUF for shared Session".to_string())?;
    let shared_session: Box<dyn crate::v2_handle::V2BatchableSession> = create_v2_shared_session(
        gguf_arch,
        shared_gguf,
        topology.clone(),
        flambeau_forward::LaunchParams {
            ctx_cap: cfg.ctx_cap,
            prefill_ubatch,
            max_slots: inflight_slots,
            paged_kv_pages: cfg.paged_kv_pages,
            kv_layout,
            deterministic_ar: cfg.deterministic,
        },
    )
    .with_context(|| format!("v2 shared session ({gguf_arch})"))?;
    let shared: crate::v2_handle::SharedV2Session = Arc::new(Mutex::new(shared_session));

    let mut inflight_pool: Vec<Mutex<Box<dyn crate::Session>>> = Vec::with_capacity(inflight_slots);
    for slot_idx in 0..inflight_slots {
        let conv: Box<dyn crate::Session> = Box::new(crate::v2_handle::V2Conv {
            shared: Arc::clone(&shared),
            slot_id: slot_idx,
            bos_id,
            chat_stops,
        });
        inflight_pool.push(Mutex::new(conv));
    }

    let cluster: Arc<HipCluster> =
        Arc::new(HipCluster::new(&cfg.device_ids).context("v2: HipCluster::new (state side)")?);
    let topology_tag =
        crate::serve_common::topology_tag_from_mesh(cfg.mesh_mode, cfg.device_ids.len());
    let prefix_cache = crate::serve_common::build_prefix_cache(&cfg, topology_tag.mesh_kind);

    // Embedding endpoint disabled — legacy qwen3-moe `EmbeddingModel`
    // was removed in #221; v2 reimplementation is a follow-up.
    let embedding_rank: Option<usize> = None;
    let embedding: Option<crate::serve_common::EmbeddingHandle> = None;

    let model = std::sync::Arc::new(crate::v2_handle::V2Model {
        gguf_arch: gguf_arch_to_static(gguf_arch),
        topology: topology_label,
        chat_stops,
        shared: Arc::clone(&shared),
        vocab: model_cfg.vocab_size,
    }) as crate::model_handle::LoadedModel;

    let state = crate::serve_common::build_server_state(crate::serve_common::ServerStateInputs {
        model_id: cfg.model_id.clone(),
        model_cfg,
        model,
        cluster,
        inflight_pool,
        embedding,
        embedding_rank,
        gpu_sampler: false,
        batched_decode: true,
        max_queue_depth,
        prefill_ubatch,
        prefill_chunk_tokens: chunk_tokens,
        topology_tag,
        prefix_cache,
        boot,
        decode_batch_window_us: cfg.decode_batch_window_us,
    });

    crate::serve_common::run_axum(state, cfg.bind_addr, "v2").await
}

fn topology_from_mesh(
    mesh: MeshMode,
    device_ids: &[i32],
    num_layers: usize,
    override_split: Option<&[usize]>,
    kv_share_boundary: Option<usize>,
) -> Result<flambeau_forward::Topology> {
    use flambeau_forward::Topology;
    // `flambeau_forward`'s orchestrator falls back to (start=0, end=0)
    // empty per-rank layer ranges when `layer_split: None`; an
    // explicit even split is required for PP / Hybrid to actually
    // execute layers. SingleDevice avoids the issue entirely at N=1.
    let resolve_split = |n_groups: usize| -> Result<Vec<usize>> {
        if let Some(s) = override_split {
            if s.len() != n_groups {
                bail!(
                    "--layer-split has {} entries but topology has {n_groups} PP ranks",
                    s.len()
                );
            }
            let sum: usize = s.iter().sum();
            if sum != num_layers {
                bail!(
                    "--layer-split sums to {sum} but model has {num_layers} layers"
                );
            }
            return Ok(s.to_vec());
        }
        if let Some(b) = kv_share_boundary {
            if n_groups == 1 {
                return Ok(vec![num_layers]);
            }
            if b == 0 || b >= num_layers {
                bail!(
                    "kv_share_pp_boundary={b} cannot be honored \
                     (num_layers={num_layers}, n_groups={n_groups})"
                );
            }
            // Last rank owns [b..num_layers]; remaining ranks split
            // [0..b) evenly. Bail if the prefix can't be split (would
            // leave an empty intermediate rank).
            let prefix = b;
            let prefix_groups = n_groups - 1;
            if prefix < prefix_groups {
                bail!(
                    "kv_share boundary {b} too small to split across \
                     {prefix_groups} non-tail ranks (need >= {prefix_groups})"
                );
            }
            let base = prefix / prefix_groups;
            let rem = prefix % prefix_groups;
            let mut split: Vec<usize> = (0..prefix_groups)
                .map(|i| base + usize::from(i < rem))
                .collect();
            split.push(num_layers - prefix);
            return Ok(split);
        }
        let base = num_layers / n_groups;
        let rem = num_layers % n_groups;
        Ok((0..n_groups).map(|i| base + usize::from(i < rem)).collect())
    };
    match mesh {
        MeshMode::Pp if device_ids.len() == 1 => Ok(Topology::SingleDevice {
            device: device_ids[0],
        }),
        MeshMode::Pp => Ok(Topology::Pp {
            devices: device_ids.to_vec(),
            layer_split: Some(resolve_split(device_ids.len())?),
        }),
        MeshMode::Tp { world } => {
            if device_ids.len() as u32 != world {
                bail!(
                    "--mesh-mode tp: --tp-size {world} but {} devices supplied",
                    device_ids.len()
                );
            }
            Ok(Topology::Tp {
                devices: device_ids.to_vec(),
            })
        }
        MeshMode::Hybrid { pp_size, tp_size } => {
            let pp = pp_size as usize;
            let tp = tp_size as usize;
            if pp * tp != device_ids.len() {
                bail!(
                    "--mesh-mode pp+tp: {pp}*{tp} != {} devices",
                    device_ids.len()
                );
            }
            let stages: Vec<Vec<i32>> = (0..pp)
                .map(|s| device_ids[s * tp..(s + 1) * tp].to_vec())
                .collect();
            Ok(Topology::Hybrid {
                stages,
                layer_split: Some(resolve_split(pp)?),
            })
        }
    }
}

fn create_v2_shared_session(
    gguf_arch: &str,
    file: GgufFile,
    topology: flambeau_forward::Topology,
    params: flambeau_forward::LaunchParams,
) -> Result<Box<dyn crate::v2_handle::V2BatchableSession>> {
    use flambeau_forward::Session;
    match gguf_arch {
        "qwen35" => {
            let s = Session::<flambeau_qwen35_v2::Qwen35V2>::new(file, topology, params)?;
            Ok(Box::new(s))
        }
        "qwen35moe" => {
            let s = Session::<flambeau_qwen35moe_v2::Qwen35MoeV2>::new(file, topology, params)?;
            Ok(Box::new(s))
        }
        "gemma3" | "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b" => {
            let s = Session::<flambeau_gemma4_v2::Gemma4V2>::new(file, topology, params)?;
            Ok(Box::new(s))
        }
        other => bail!("v2 serve: unsupported GGUF arch `{other}`"),
    }
}

fn gguf_arch_to_static(arch: &str) -> &'static str {
    match arch {
        "qwen35" => "qwen35",
        "qwen35moe" => "qwen35moe",
        "gemma3" => "gemma3",
        "gemma4" => "gemma4",
        "gemma4-26b-a4b" => "gemma4-26b-a4b",
        "gemma4-31b" => "gemma4-31b",
        "gemma4-9b" => "gemma4-9b",
        "gemma4-2b" => "gemma4-2b",
        _ => "v2",
    }
}

fn parse_v2_server_model_cfg(
    gguf_arch: &str,
    gguf_path: &std::path::Path,
) -> Result<crate::model_cfg::ServerModelCfg> {
    let f = GgufFile::open(gguf_path).context("v2 cfg: re-open GGUF for metadata")?;
    // GGUF metadata keys are namespaced by `general.architecture`.
    let prefix = gguf_arch;
    let key = |suffix: &str| format!("{prefix}.{suffix}");
    let num_layers = f
        .metadata_u32(&key("block_count"))
        .ok_or_else(|| anyhow::anyhow!("v2 cfg: missing `{}.block_count`", prefix))?
        as usize;
    let context_length = f
        .metadata_u32(&key("context_length"))
        .ok_or_else(|| anyhow::anyhow!("v2 cfg: missing `{}.context_length`", prefix))?
        as usize;
    let vocab_size = f
        .info("token_embd.weight")
        .ok()
        .and_then(|ti| ti.dims.first().copied())
        .map(|v| v as usize)
        .ok_or_else(|| anyhow::anyhow!("v2 cfg: token_embd.weight missing"))?;
    let kv_share_pp_boundary = match gguf_arch {
        #[cfg(feature = "hip")]
        "gemma4" => {
            let cfg = flambeau_gemma4_v2::Gemma4V2Config::from_gguf(&f)
                .context("v2 cfg: gemma4 config parse for kv_share boundary")?;
            cfg.pp_kv_share_boundary()
        }
        _ => None,
    };
    Ok(crate::model_cfg::ServerModelCfg {
        arch: gguf_arch.to_string(),
        vocab_size,
        context_length,
        num_layers,
        kv_share_pp_boundary,
    })
}
