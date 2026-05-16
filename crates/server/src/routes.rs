//! HTTP route handlers. Requires `hip` feature (loads real model).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use axum::response::sse::{Event, KeepAlive, Sse};
use flambeau_backend_hip::HipCluster;
use crate::model_cfg::ServerModelCfg;
use flambeau_quant::{ChatTemplate, GgufTokenizer};
use flambeau_runtime::json_grammar::JsonState;
use flambeau_runtime::Sampler;
use serde_json::json;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

use crate::api::*;
use crate::gpu_sampler::{self, GpuSamplerScratch};
use crate::model::{
    capture_kv_from_inflight, prefill_logits, restore_kv_into_inflight, snapshot_bytes,
    LoadedModel,
};
use crate::prefix_cache::{PrefixCache, PrefixKeys, TopologyTag};

#[cfg(feature = "dev_trace")]
fn dev_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}
#[cfg(not(feature = "dev_trace"))]
#[inline(always)]
fn dev_flag(_name: &str) -> bool {
    false
}

#[cfg(feature = "dev_trace")]
fn dev_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
#[cfg(not(feature = "dev_trace"))]
#[inline(always)]
fn dev_usize(_name: &str, default: usize) -> usize {
    default
}

/// Outcome of `ServerState::prefix_cache_try_restore` — drives the
/// caller's prefill branch.
#[derive(Debug)]
pub enum PrefixCacheRestore {
    /// No useful entry; caller does fresh full prefill.
    Miss,
    /// Full-prompt match. KV+GDN restored into inflight; caller skips
    /// `prefill_logits` and samples the first decode token from `logits`.
    FullHit { logits: std::sync::Arc<Vec<f32>> },
    /// Chunk-aligned prefix match. KV+GDN restored at boundary
    /// `n_matched`; caller does a partial prefill of `prompt_ids[n_matched..]`
    /// with `start_position = n_matched` to advance to prompt end.
    PrefixHit { n_matched: usize },
}
use crate::state::SamplingParams;


pub use crate::model::Qwen3MoeServerExtras;

/// Server-wide shared state — built once at startup.
pub struct ServerState {
    pub model_id: String,
    pub cfg: ServerModelCfg,
    /// PP or TP loaded model. Handlers dispatch via the
    /// `crate::model::{prefill_logits, decode_logits}` helpers; they
    /// don't need to inspect this variant directly.
    pub model: LoadedModel,
    /// `Arc` so the TP variant's `BarP2pAllReduce` can
    /// hold a peer reference to the same cluster the handlers borrow
    /// from.
    pub cluster: Arc<HipCluster>,
    pub tokenizer: GgufTokenizer,
    pub chat_template: ChatTemplate,
    /// **P2.9b-i1 (multi-slot pool)** — N pre-allocated per-request
    /// sessions sized to `FLAMBEAU_INFLIGHT_SLOTS` (default 1). A
    /// request acquires any free slot via `acquire_inflight_blocking()`
    /// (try-lock round-robin, then block on slot 0 if all busy).
    /// Holding the guard means "this request owns the slot"; releasing
    /// it returns the slot to the pool. The element type is the
    /// model-agnostic `Session` trait so future model crates (e.g.
    /// gemma4) can plug in without churn at the pool / handler layer.
    /// Concrete qwen3-moe forward-pass dispatch still drills down via
    /// `as_pp_mut()` / `as_tp_mut()` / `as_hybrid_mut()` trait
    /// accessors.
    pub inflight_pool: Vec<Mutex<Box<dyn crate::Session>>>,
    /// **P2.9b-i2-B (scheduler)** — request-lifetime claim flag for
    /// each slot. Distinct from `inflight_pool`'s mutex: the mutex
    /// guards short-term *exclusive access* to the `Inflight`; this
    /// `AtomicBool` records the *long-term ownership* of the slot
    /// across one whole HTTP request. Handlers `claim_slot_blocking`
    /// at request entry, perform prefill while holding the slot's
    /// mutex briefly, then *release the mutex* during the decode loop
    /// so the scheduler-leader can `blocking_lock` the slot
    /// alongside other slots' mutexes for batched dispatch. The
    /// claim is released only at request exit. Sized parallel to
    /// `inflight_pool`. Active when `FLAMBEAU_BATCHED_DECODE=1`;
    /// otherwise the legacy decode path holds the mutex for the full
    /// request lifetime and this field is ignored.
    pub slot_in_use: Vec<std::sync::atomic::AtomicBool>,
    /// **P2.9b-i2-B** — pending-decode queue. Each handler in the
    /// scheduler-aware decode loop pushes a `PendingDecode` carrying
    /// `(slot_idx, token, position, response_tx)`. The leader (the
    /// handler that wins `batched_dispatcher`) drains the queue,
    /// locks each referenced slot, runs `forward_decode_batched_pp`
    /// across the batch, and sends per-slot logits back via the
    /// `response_tx` channels. Non-leader handlers just wait on
    /// their `rx`.
    pub batched_pending: std::sync::Mutex<Vec<PendingDecode>>,
    /// **P2.9b-i2-B** — single-leader gate. The handler that
    /// `try_lock`s this becomes the dispatch leader for the next
    /// batched call. Held only during dispatch (lock ⇒ drain queue
    /// ⇒ blocking_lock the relevant slots ⇒ batched forward ⇒
    /// distribute responses ⇒ unlock).
    pub batched_dispatcher: std::sync::Mutex<()>,
    /// Qwen3-moe-specific shared workspaces. `Some` on the qwen3-moe
    /// boot path; `None` otherwise. Reached via
    /// `state.qwen3_moe.as_ref().expect(...)` from dispatch branches
    /// that are already arch-gated upstream (`model.as_pp/_tp/_hybrid`
    /// returning `Some`, or the scheduler path).
    pub qwen3_moe: Option<Qwen3MoeServerExtras>,
    /// **#229 P2.10c** — process-local prompt prefix cache. Always
    /// constructed; methods short-circuit when `state.prefix_cache.enabled()`
    /// is false (default OFF; flip via `FLAMBEAU_PREFIX_CACHE=1`).
    /// Stores host-RAM KV snapshots keyed by chained chunk hashes;
    /// LRU-evicts under `FLAMBEAU_PREFIX_CACHE_MAX_GB` (default 2 GB).
    /// PP and TP supported; Hybrid bails on capture/restore in V1.
    pub prefix_cache: Arc<PrefixCache>,
    /// **#229** — chunk size used by every cache entry in this server's
    /// lifetime (snapshot of `FLAMBEAU_PREFILL_UBATCH` at boot). Cache
    /// rejects lookups with a different chunk size. Default 512.
    pub prefix_cache_chunk_tokens: usize,
    /// **#229** — topology fingerprint stored on every cache entry.
    /// Defensive guard against cross-topology pollution.
    pub topology_tag: TopologyTag,
    /// **#230 / #231** — optional embedding model for the
    /// `/v1/embeddings` endpoint. `None` when the server was started
    /// without `--embedding-model`; the endpoint returns 503 in that
    /// case. Wrapped in `tokio::sync::Mutex` because forward state
    /// (per-layer scratch + KV) is shared and concurrent requests
    /// must serialise — embedding inference is fast enough on a
    /// 0.6B model that V1 doesn't bother with multi-slot pooling.
    pub embedding_model:
        Option<Arc<tokio::sync::Mutex<Box<dyn crate::embedding::EmbeddingHandle>>>>,
    /// **#231 quality fix** — embedding model's own tokenizer.
    /// Qwen3-Embedding ships a vocab (151669) that diverges from
    /// chat-side tokenizers (151424 on Qwen3.5-9B); reusing the chat
    /// tokenizer feeds wrong token ids into the embedding model's
    /// `token_embd`. `None` mirrors `embedding_model = None`.
    pub embedding_tokenizer: Option<Arc<flambeau_quant::GgufTokenizer>>,
    /// **#231** — HIP rank index for the embedding device, derived at
    /// boot from `--embedding-device`. The endpoint uses
    /// `cluster.device(rank)` to get a `&HipDevice` for the forward
    /// call. `None` mirrors `embedding_model = None`.
    pub embedding_rank: Option<usize>,
    /// **#232 P2.12** — admission-control counter. Incremented at
    /// request entry, decremented at response. When this exceeds
    /// `inflight_slots + max_queue_depth`, new requests get 503 +
    /// `Retry-After: 2` instead of queueing on the slot mutex.
    /// Without this, a 100-request flood pile-ups against the slot
    /// pool's blocking_lock, holding tokio runtime threads + per-
    /// request memory until the GPU drains, eventually OOMing the
    /// host. The counter is a cheap atomic; `AdmissionGuard` makes
    /// the decrement RAII-safe across early returns.
    pub in_flight: std::sync::atomic::AtomicUsize,
    /// **#232** — max queued requests beyond the inflight slot pool.
    /// `0` disables the check (legacy behaviour). Default 16 means
    /// `inflight_slots + 16` admitted requests at any time; further
    /// requests get 503. Sized so the queue empties in a few seconds
    /// even on long generations: at 16 slots × ~5s avg request, the
    /// queue takes ~80s to drain, which is the upper bound a
    /// well-behaved client should retry over.
    pub max_queue_depth: usize,
    /// Prefill chunk size (`--prefill-ubatch`). Read by the chunked
    /// prefill driver and the lazy TP-prefill scratch allocator.
    pub prefill_ubatch: usize,
    /// `--gpu-sampler` (default true). When true the head rank's
    /// top-K + softmax (+ penalties) run on device.
    pub gpu_sampler: bool,
    /// `--batched-decode` (default true). When true, concurrent decode
    /// requests aggregate via the scheduler leader.
    pub batched_decode: bool,
    /// Per-iteration agent-loop telemetry. Ring buffer; surfaced
    /// read-only at `GET /v1/agent/stats`.
    pub agent_stats: crate::agent_stats::AgentStatsRing,
    /// L3 — tool-call format detected at boot from the GGUF chat
    /// template. `general.architecture=qwen35moe` alone is not enough
    /// to decide: the Unsloth UD Qwen3.6 GGUFs ship a Coder-XML
    /// template even though the arch tag says `qwen35moe`. Honoured
    /// when a request omits `tool_call_format` or sets it to "auto".
    pub tool_call_format_default: crate::tool_call_parser::ToolCallFormat,
    /// **#235 P3.15** — `true` when the chat template carries the
    /// `enable_thinking` Jinja variable (Qwen3.6 reasoning mode).
    /// Surfaced as the `"thinking"` capability in `/v1/models` so
    /// clients know whether to expose the request-side
    /// `enable_thinking` flag (#233). Detected once at boot by
    /// substring-matching the raw template source.
    pub supports_thinking: bool,
    /// **#235 P3.15** — human-readable quantization label derived
    /// from the GGUF `general.file_type` integer (`"Q4_0"`,
    /// `"Q4_K_M"`, `"F16"`, etc). `None` when the GGUF lacks the
    /// field. Surfaced in `/v1/models`.
    pub quantization: Option<String>,
    /// Sampling defaults read from the GGUF (`general.sampling.*`).
    /// Filled into omitted request fields by
    /// [`crate::state::SamplingParams::from_parts`]. Values that the
    /// GGUF doesn't carry stay `None`; the OpenAI-shaped fallback in
    /// `from_parts` then takes effect.
    pub model_defaults: crate::state::ModelDefaults,
    /// **P0.5** — fallback system prompt injected when the request
    /// carries no `role: "system"` message. Sourced at boot from
    /// `FLAMBEAU_DEFAULT_SYSTEM` (env), or `None` if unset. Some clients
    /// (Aider, plain `curl`, the embedded UI) routinely send only a
    /// `user` message, and Qwen3.6 then degrades into terse one-line
    /// replies because the chat template's neutral default doesn't
    /// frame the assistant role. Operator-controlled — never derived
    /// from request fields.
    pub default_system: Option<String>,
}

pub type SharedState = Arc<ServerState>;

/// **P2.9b-i2-B** — one queued decode request awaiting batched dispatch.
/// Pushed by the scheduler-aware decode loop (`decode_via_scheduler`)
/// and drained by the leader (the first handler to acquire
/// `batched_dispatcher`).
pub struct PendingDecode {
    pub slot_idx: usize,
    pub token_id: u32,
    pub position: usize,
    pub response: std::sync::mpsc::Sender<anyhow::Result<Vec<f32>>>,
}

/// **#232 P2.12** — RAII guard for admission control. Holding one
/// of these means the request is counted against
/// `ServerState.in_flight`; dropping it decrements the counter on
/// every exit path (success, error, panic-unwind).
pub struct AdmissionGuard {
    /// `None` when admission control is disabled
    /// (`max_queue_depth = 0`); the guard then becomes a no-op.
    state: Option<Arc<ServerState>>,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            state
                .in_flight
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        }
    }
}

/// **#236 P0.1b** — strip the chat-template-emitted assistant
/// terminator from a rendered prompt so the model continues the
/// assistant prefill content rather than seeing a closed turn.
/// Qwen-family templates emit `<|im_end|>` followed by a newline at
/// the end of every closed turn; `add_generation_prompt=false` keeps
/// that terminator on the in-progress assistant turn. For prefill /
/// continue-the-message semantics we want to delete it so the model's
/// next-token distribution is conditioned on the partial assistant
/// content, not on "another turn finished, what's next".
/// Conservative: trim trailing whitespace, then a single `<|im_end|>`
/// substring, then more trailing whitespace. Does nothing if the
/// terminator isn't found (works as a no-op for templates that
/// already rendered without one).
fn strip_trailing_assistant_terminator(prompt: &str) -> String {
    let trimmed = prompt.trim_end();
    if let Some(stripped) = trimmed.strip_suffix("<|im_end|>") {
        stripped.trim_end().to_string()
    } else {
        prompt.to_string()
    }
}

impl ServerState {
    /// **#324** — lock the shared TP prefill scratch, lazy-initialising
    /// on first call. Caller MUST already hold `prefill_serialiser` to
    /// avoid concurrent init races and concurrent kernel writes (the
    /// scratch buffers are not safe for parallel use).
    /// Sized for `FLAMBEAU_PREFILL_UBATCH` (default 512). Returns the
    /// locked option as a guard so the caller can borrow `&mut` for
    /// the duration of `prefill_logits`. PP-only models don't call
    /// this; Hybrid currently doesn't use it either (V2 follow-up).
    pub fn lock_tp_prefill_scratch(
        &self,
    ) -> anyhow::Result<
        std::sync::MutexGuard<
            '_,
            Option<flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp>,
        >,
    > {
        let qwen3_moe = self
            .qwen3_moe
            .as_ref()
            .context("lock_tp_prefill_scratch on non-qwen3-moe boot")?;
        let mut guard = qwen3_moe.tp_prefill_scratch.lock().unwrap();
        if guard.is_none() {
            let prefill_ubatch = self.prefill_ubatch;
            let cfg = &self
                .model
                .as_tp()
                .context("lock_tp_prefill_scratch on non-TP model")?
                .model
                .config;
            let scratch = flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp::new(
                cfg,
                &self.cluster,
                prefill_ubatch,
            )
            .context("lazy-init shared TP prefill scratch")?;
            tracing::info!(
                target: "server.prefill",
                prefill_ubatch,
                "lazy-init shared TP prefill scratch (#324)"
            );
            *guard = Some(scratch);
        }
        Ok(guard)
    }

    /// **#229 P2.10c** — look up the prefix cache for the given
    /// prompt. On a hit, restore the cached KV+GDN state into
    /// `inflight` and report whether it covers the FULL prompt (with
    /// cached logits, prefill skipped) or just a chunk-aligned PREFIX
    /// (caller must run a partial tail prefill).
    /// Returns `Ok(Miss)` (caller does fresh full prefill) when:
    /// - `FLAMBEAU_PREFIX_CACHE` is unset (default OFF).
    /// - Prompt is shorter than one chunk (< chunk_tokens).
    /// - Topology is `Hybrid` (per-stage per-rank shape unsupported).
    /// - No matching chain in the cache.
    /// - Matched entry lacks a KV snapshot.
    /// - Restore fails (logged + downgraded to miss).
    pub fn prefix_cache_try_restore(
        &self,
        inflight: &mut dyn crate::Session,
        prompt_ids: &[u32],
    ) -> anyhow::Result<PrefixCacheRestore> {
        tracing::debug!(
            target: "server.prefix_cache",
            prompt_tokens = prompt_ids.len(),
            enabled = self.prefix_cache.enabled(),
            hybrid = self.model.as_hybrid().is_some(),
            "prefix_cache_try_restore called"
        );
        if !self.prefix_cache.enabled() {
            return Ok(PrefixCacheRestore::Miss);
        }
        // Phase 12.7 — prefix cache stays qwen3-moe-typed in V1.
        // Sessions from other archs (gemma4) silently skip without the
        // capture/restore warn-log churn.
        if inflight.as_pp().is_none()
            && inflight.as_tp().is_none()
            && inflight.as_hybrid().is_none()
        {
            return Ok(PrefixCacheRestore::Miss);
        }
        let chunk_tokens = self.prefix_cache_chunk_tokens;
        if prompt_ids.len() < chunk_tokens {
            return Ok(PrefixCacheRestore::Miss);
        }
        let keys = PrefixKeys::from_prompt(prompt_ids, chunk_tokens);
        if keys.n_chunks() == 0 {
            return Ok(PrefixCacheRestore::Miss);
        }
        let topology = self.topology_tag;
        let info = self
            .prefix_cache
            .longest_match(&keys, topology, |_terminal| {});
        let info = match info {
            Some(m) => m,
            None => return Ok(PrefixCacheRestore::Miss),
        };
        let snapshot = match self.prefix_cache.snapshot_for(info.terminal) {
            Some(arc) => arc,
            None => return Ok(PrefixCacheRestore::Miss),
        };
        let is_full = info.n_tokens == prompt_ids.len();
        // For full-prompt match we need cached logits to skip prefill.
        // If the entry was inserted at a chunk boundary (intermediate)
        // there are no logits — degrade to partial-prefill at that
        // boundary (still saves work).
        let logits = if is_full {
            self.prefix_cache.logits_for(info.terminal)
        } else {
            None
        };
        let restore_start = std::time::Instant::now();
        if let Err(e) = restore_kv_into_inflight(
            inflight,
            &self.cluster,
            snapshot.as_ref(),
            &self.model,
        ) {
            tracing::warn!(
                target: "server.prefix_cache",
                error = %e,
                "restore failed — falling through to fresh prefill"
            );
            return Ok(PrefixCacheRestore::Miss);
        }
        let restore_ms = restore_start.elapsed().as_secs_f64() * 1000.0;
        match logits {
            Some(arc) => {
                tracing::info!(
                    target: "server.prefix_cache",
                    event = "full_hit",
                    n_matched_tokens = info.n_tokens,
                    n_matched_chunks = info.n_chunks,
                    prompt_tokens = prompt_ids.len(),
                    restore_ms,
                    "prefix-cache full hit — KV restored, prefill skipped"
                );
                Ok(PrefixCacheRestore::FullHit { logits: arc })
            }
            None => {
                tracing::info!(
                    target: "server.prefix_cache",
                    event = "prefix_hit",
                    n_matched_tokens = info.n_tokens,
                    n_matched_chunks = info.n_chunks,
                    prompt_tokens = prompt_ids.len(),
                    restore_ms,
                    "prefix-cache prefix hit — KV restored, partial prefill of tail"
                );
                Ok(PrefixCacheRestore::PrefixHit {
                    n_matched: info.n_tokens,
                })
            }
        }
    }

    /// **#229 insert an intermediate (chunk-boundary) cache
    /// entry produced during a fresh prefill's per-chunk loop. No
    /// logits are stored (callers can't sample mid-prefill); future
    /// requests that hit this entry restore at the boundary and
    /// proceed with a partial-tail prefill via `prefill_logits`.
    pub fn prefix_cache_insert_intermediate(
        &self,
        prompt_ids: &[u32],
        n_tokens_completed: usize,
        snap: Vec<crate::prefix_cache::RankSnapshot>,
    ) {
        if !self.prefix_cache.enabled() {
            return;
        }
        let chunk_tokens = self.prefix_cache_chunk_tokens;
        if n_tokens_completed == 0 || n_tokens_completed % chunk_tokens != 0 {
            return;
        }
        let n_full_chunks = n_tokens_completed / chunk_tokens;
        if n_full_chunks == 0 {
            return;
        }
        let keys = PrefixKeys::from_prompt(prompt_ids, chunk_tokens);
        if n_full_chunks > keys.chunk_keys.len() {
            return;
        }
        let chain = keys.chunk_keys[..n_full_chunks].to_vec();
        let bytes = snapshot_bytes(&snap);
        let snap_arc = std::sync::Arc::new(snap);
        self.prefix_cache.insert_with_kv(
            chain,
            self.topology_tag,
            chunk_tokens,
            n_tokens_completed,
            snap_arc,
            None,
            bytes,
        );
        tracing::info!(
            target: "server.prefix_cache",
            event = "insert_intermediate",
            n_chunks = n_full_chunks,
            n_tokens = n_tokens_completed,
            kv_bytes = bytes,
            used_bytes = self.prefix_cache.used_bytes(),
            entries = self.prefix_cache.len(),
            "prefix-cache write — intermediate boundary entry"
        );
    }

    /// **#229 P2.10c** — best-effort capture of the post-prefill KV
    /// state into the prefix cache. Inserts under the chunk-key chain
    /// for the full prompt, replacing any prior entry with the same
    /// chain (idempotent). Eligibility filter:
    /// - `FLAMBEAU_PREFIX_CACHE` must be set.
    /// - Topology must be PP or TP (Hybrid bails).
    /// - Prompt must have at least one new complete chunk past
    /// `n_already_matched`.
    /// - Prompt must be at least 50 tokens (cache-hit savings won't
    /// justify the host-RAM cost on tiny prompts).
    /// Errors are logged and swallowed — capture is opportunistic; a
    /// failed snapshot must not break the caller's request.
    /// **#229 V1** — capture the post-prefill KV+GDN state plus the
    /// last-position logits row into the prefix cache. V1 only inserts
    /// full-prompt entries — partial-chunk-boundary captures need
    /// GDN-at-position snapshotting (V2). Keyed by the full chain
    /// (chunk-keys including partial tail) so future identical
    /// prompts hit and can skip prefill entirely.
    /// `last_logits` is the prefill's last-position F32 vocab row,
    /// the same one the caller is about to feed into the first-token
    /// sampler. Cloned into the cache entry; ~600 KB on Qwen3.6.
    /// Eligibility:
    /// - `FLAMBEAU_PREFIX_CACHE` set.
    /// - Topology PP or TP (Hybrid bails).
    /// - Prompt ≥ 50 tokens AND at least one full chunk in the chain.
    pub fn prefix_cache_try_capture_full(
        &self,
        inflight: &dyn crate::Session,
        prompt_ids: &[u32],
        last_logits: &[f32],
    ) {
        tracing::debug!(
            target: "server.prefix_cache",
            prompt_tokens = prompt_ids.len(),
            logits_len = last_logits.len(),
            enabled = self.prefix_cache.enabled(),
            hybrid = self.model.as_hybrid().is_some(),
            "prefix_cache_try_capture_full called"
        );
        if !self.prefix_cache.enabled() {
            return;
        }
        // Phase 12.7 — see `prefix_cache_try_restore` early-out doc.
        if inflight.as_pp().is_none()
            && inflight.as_tp().is_none()
            && inflight.as_hybrid().is_none()
        {
            return;
        }
        const MIN_PROMPT_TOKENS: usize = 50;
        if prompt_ids.len() < MIN_PROMPT_TOKENS {
            tracing::debug!(target: "server.prefix_cache", "skip: prompt < MIN_PROMPT_TOKENS");
            return;
        }
        let chunk_tokens = self.prefix_cache_chunk_tokens;
        if prompt_ids.len() < chunk_tokens {
            tracing::debug!(target: "server.prefix_cache", "skip: prompt < chunk_tokens");
            return;
        }
        let keys = PrefixKeys::from_prompt(prompt_ids, chunk_tokens);
        if keys.n_chunks() == 0 || last_logits.is_empty() {
            tracing::debug!(target: "server.prefix_cache", "skip: 0 chunks or empty logits");
            return;
        }
        let snap = match capture_kv_from_inflight(inflight, &self.cluster, &self.model) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    target: "server.prefix_cache",
                    error = %e,
                    "capture failed — entry not inserted"
                );
                return;
            }
        };
        // **#229 V1** — capture the FULL post-prefill state (no
        // truncation). Pair it with the last-position logits so the
        // hit path skips prefill entirely. GDN is captured as the
        // post-prompt-end state; future identical-prompt hit gets
        // bit-equivalent state restored.
        let kv_bytes = snapshot_bytes(&snap);
        let logits_bytes = last_logits.len() * std::mem::size_of::<f32>();
        let total_bytes = kv_bytes + logits_bytes;
        let snap_arc = std::sync::Arc::new(snap);
        let logits_arc = std::sync::Arc::new(last_logits.to_vec());
        self.prefix_cache.insert_with_kv(
            keys.chunk_keys.clone(),
            self.topology_tag,
            chunk_tokens,
            prompt_ids.len(),
            snap_arc,
            Some(logits_arc),
            total_bytes,
        );
        tracing::info!(
            target: "server.prefix_cache",
            event = "insert",
            n_chunks = keys.n_chunks(),
            n_tokens = prompt_ids.len(),
            kv_bytes,
            logits_bytes,
            used_bytes = self.prefix_cache.used_bytes(),
            budget_bytes = self.prefix_cache.vram_budget_bytes,
            entries = self.prefix_cache.len(),
            "prefix-cache write — full-prompt entry inserted"
        );
    }

    /// **#232 P2.12** — try to admit a new request. Bumps `in_flight`
    /// if under cap (`inflight_slots + max_queue_depth`); returns
    /// `None` when the cap is hit (caller should respond with 503).
    /// `max_queue_depth = 0` disables admission control.
    pub fn try_admit(self: &Arc<Self>) -> Option<AdmissionGuard> {
        if self.max_queue_depth == 0 {
            // Disabled — pass through without counting.
            return Some(AdmissionGuard {
                state: None,
            });
        }
        let cap = self.inflight_pool.len() + self.max_queue_depth;
        let prev = self
            .in_flight
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if prev >= cap {
            // Roll back the bump and reject.
            self.in_flight
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            tracing::warn!(
                target: "server.admission",
                in_flight = prev,
                cap,
                "queue full — rejecting with 503"
            );
            return None;
        }
        Some(AdmissionGuard {
            state: Some(Arc::clone(self)),
        })
    }

    /// **P2.9b-i1** — acquire an idle inflight slot, blocking until one
    /// is available. Iterates the pool with `try_lock` first; if every
    /// slot is busy, blocks on slot 0 (head-of-line, but bounded by
    /// the longest in-flight decode). The returned guard ties the slot
    /// to the request scope — dropping it returns the slot to the pool.
    /// Returns `(slot_idx, guard)` so callers can log which slot served
    /// the request.
    pub fn acquire_inflight_blocking(
        &self,
    ) -> (usize, tokio::sync::MutexGuard<'_, Box<dyn crate::Session>>) {
        for (idx, slot) in self.inflight_pool.iter().enumerate() {
            if let Ok(g) = slot.try_lock() {
                return (idx, g);
            }
        }
        // All busy — fall back to slot 0. tokio fairness guarantees
        // FIFO on contended `lock()`/`blocking_lock()` so this approximates
        // a single-queue head-of-line.
        let g = self.inflight_pool[0].blocking_lock();
        (0, g)
    }

    /// **P2.9b-i2-B** — claim a slot for the lifetime of a request via
    /// `slot_in_use[idx]` CAS. Distinct from `acquire_inflight_blocking`,
    /// which holds the slot's mutex; the claim records *long-term
    /// ownership* of the slot across the full request, while the
    /// mutex is acquired only briefly within prefill / scheduler
    /// dispatch / sampler-sandwich. Spins with a short backoff if no
    /// slot is free.
    pub fn claim_slot_blocking(&self) -> usize {
        use std::sync::atomic::Ordering::{Acquire, Relaxed};
        loop {
            for (idx, taken) in self.slot_in_use.iter().enumerate() {
                if taken
                    .compare_exchange(false, true, Acquire, Relaxed)
                    .is_ok()
                {
                    return idx;
                }
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }

    /// **P2.9b-i2-B** — release a request-lifetime slot claim. Pair
    /// with [`claim_slot_blocking`].
    pub fn release_slot(&self, idx: usize) {
        self.slot_in_use[idx].store(false, std::sync::atomic::Ordering::Release);
    }

    /// **P2.9b-i2-B** — push one decode request into the batched
    /// queue and wait for the leader to dispatch.
    /// Caller must NOT be holding `inflight_pool[slot_idx]`'s mutex —
    /// the leader needs to `blocking_lock` it during dispatch.
    /// If we win `batched_dispatcher`, we become the leader: brief
    /// 200 µs sleep to allow other handlers to push, then drain the
    /// queue, lock each referenced slot's `Inflight`, run
    /// `forward_decode_batched_pp` across the batch, and fire each
    /// pending entry's response sender. The leader then awaits its
    /// own response on the same channel as the others.
    /// Errors propagate through the response channel; PP-only
    /// (TP/Hybrid still go through legacy `decode_logits`).
    pub fn decode_via_scheduler(
        &self,
        slot_idx: usize,
        token: u32,
        position: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let mut buf: Vec<f32> = Vec::with_capacity(self.cfg.vocab_size);
        self.decode_via_scheduler_into(slot_idx, token, position, &mut buf)?;
        Ok(buf)
    }

    /// **Cycle 3** — sibling of [`decode_via_scheduler`] that fills a
    /// caller-provided buffer instead of allocating a fresh `Vec` per
    /// step. Used by the scheduler-aware handler's decode loop to
    /// recycle one ~600 KB logits buffer across all decode steps
    /// (vocab=151424 F32 = 591 KB on Qwen3.6).
    pub fn decode_via_scheduler_into(
        &self,
        slot_idx: usize,
        token: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        // **Cycle 2 optimisation (single-user fast path)** — when no
        // other slot is active, aggregation is structurally impossible.
        // Fall through to the legacy `decode_logits` path which uses
        // decode-flavoured kernels (fused rmsnorm+quant_q8_1, per-step
        // mmvq) and skips the scheduler's queue + mpsc round-trip.
        // The batched code path (`forward_decode_batched_*`) replays
        // prefill-flavoured kernels which add a few extra launches per
        // layer (separate rmsnorm + 2 quant variants); at N=1 those
        // launches are pure overhead vs the fused decode form.
        let n_others_active = self
            .slot_in_use
            .iter()
            .enumerate()
            .filter(|(idx, taken)| {
                *idx != slot_idx
                    && taken.load(std::sync::atomic::Ordering::Relaxed)
            })
            .count();
        let trace = dev_flag("FLAMBEAU_TRACE_BATCH");
        macro_rules! tr {
            ($($arg:tt)*) => {
                if trace {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() % 1_000_000)
                        .unwrap_or(0);
                    eprintln!("[TRACE-BATCH t={:06}ms slot={} pos={}] {}", now, slot_idx, position, format_args!($($arg)*));
                }
            };
        }
        tr!("entry n_others_active={}", n_others_active);
        if n_others_active == 0 {
            // #16 race-safe fast-path: hold prefill_serialiser across
            // decode_logits so a concurrent prefill on a sibling slot
            // can't race shared HipCluster scratch mid-decode. Cheap
            // (microseconds) on the fast-path; the lock is uncontended
            // when n_others_active==0. Scheduler-path is qwen3-moe-only
            // (gated on `as_pp/as_tp/as_hybrid` upstream), so the
            // qwen3_moe Some is guaranteed here.
            let _prefill_lock = self
                .qwen3_moe
                .as_ref()
                .expect("scheduler path: qwen3_moe extras present")
                .prefill_serialiser
                .lock()
                .unwrap();
            tr!("FAST_PATH lock_inflight start");
            let mut guard = self.inflight_pool[slot_idx].blocking_lock();
            tr!("FAST_PATH lock_inflight done; decode start");
            self.dispatch_decode_one(&mut **guard, token, position, logits_out)
                .context("decode_via_scheduler single-user fast path")?;
            tr!("FAST_PATH decode done; return");
            return Ok(());
        }

        let (tx, rx) = std::sync::mpsc::channel::<anyhow::Result<Vec<f32>>>();
        {
            let mut q = self
                .batched_pending
                .lock()
                .expect("batched_pending mutex poisoned");
            q.push(PendingDecode {
                slot_idx,
                token_id: token,
                position,
                response: tx,
            });
            tr!("PUSH pending q_len_now={}", q.len());
        }

        // Try to become the dispatch leader for this round. Stored in an
        // Option so we can deterministically drop it *while holding*
        // `batched_pending` to close the post-dispatch race (see #276 fix).
        let mut dispatch_lock_opt: Option<_> =
            self.batched_dispatcher.try_lock().ok();
        tr!("LEADER try_lock={}", dispatch_lock_opt.is_some());
        if dispatch_lock_opt.is_some() {
            // **Cycle 1 optimisation (single-user fast path)** — only
            // sleep on the batching window when there's actually an
            // active OTHER slot that could plausibly join the batch.
            let n_others_active = self
                .slot_in_use
                .iter()
                .enumerate()
                .filter(|(idx, taken)| {
                    *idx != slot_idx
                        && taken.load(std::sync::atomic::Ordering::Relaxed)
                })
                .count();
            if n_others_active > 0 {
                std::thread::sleep(std::time::Duration::from_micros(1500));
            }

            // **#276 fix** — drain-dispatch-loop with atomic empty-drop.
            // Without this restructure, late-arriving pending entries
            // could be stranded by a two-step race:
            // 1. Leader L drains queue (perhaps empty). Releases
            // `batched_pending`.
            // 2. Thread T2 takes `batched_pending`, pushes its entry,
            // releases.
            // 3. T2 try_locks `batched_dispatcher` — STILL HELD by L
            // (which is mid-dispatch or mid-cleanup). Returns Err.
            // T2 falls through to `rx.recv()`.
            // 4. L releases `batched_dispatcher`, returns from
            // `decode_via_scheduler_into`. L's request finishes
            // (max_tokens / stop) without re-entering the scheduler.
            // 5. T2's pending entry never drained → forever blocked on
            // `rx.recv()`.
            // Fix: when L sees an empty queue, it must release
            // `batched_dispatcher` *while still holding `batched_pending`*.
            // After that, any T2 push observes a clean dispatch lock and
            // its `try_lock` succeeds, making T2 the next leader.
            loop {
                let mut q = self
                    .batched_pending
                    .lock()
                    .expect("batched_pending mutex poisoned");
                if q.is_empty() {
                    // Drop dispatch_lock *while holding `q`* (atomic
                    // wrt future pushers). After q drops, any T2 push
                    // will see dispatch_lock free.
                    drop(dispatch_lock_opt.take());
                    tr!("DRAIN drained=0 (release-leader-while-holding-q)");
                    drop(q);
                    break;
                }
                let pending: Vec<PendingDecode> = std::mem::take(&mut *q);
                drop(q);
                tr!("DRAIN drained={}", pending.len());
                tracing::info!(
                    target: "server.scheduler",
                    pending = pending.len(),
                    "scheduler dispatch"
                );
                let ids: Vec<usize> = pending.iter().map(|p| p.slot_idx).collect();
                tr!("DISPATCH start slots={:?}", ids);
                if let Err(e) = self.dispatch_batched_pending(&pending) {
                    for p in &pending {
                        let _ = p.response.send(Err(anyhow!(
                            "batched dispatch failed: {e}"
                        )));
                    }
                }
                tr!("DISPATCH done slots={:?}", ids);
            }
            // dispatch_lock_opt is None here; explicit release happened
            // inside the loop while holding `batched_pending`.
        }

        // Wait for our response (whether we were leader or not).
        tr!("WAIT recv start");
        let recv_buf = rx
            .recv()
            .map_err(|e| anyhow!("decode_via_scheduler recv: {e}"))??;
        tr!("WAIT recv done");
        *logits_out = recv_buf;
        Ok(())
    }

    /// **P2.9b-i2-B** — leader's dispatch step. Locks each referenced
    /// slot's `Inflight`, builds the `BatchSlot` list, runs
    /// `forward_decode_batched_pp`, and sends per-slot logits via
    /// the response senders.
    /// Returns Err on dispatch failure; caller fans the error to all
    /// pending senders.
    fn dispatch_batched_pending(
        &self,
        pending: &[PendingDecode],
    ) -> anyhow::Result<()> {
        use flambeau_qwen3_moe::forward::BatchSlot;
        let trace = dev_flag("FLAMBEAU_TRACE_BATCH");
        macro_rules! tr_d {
            ($($arg:tt)*) => {
                if trace {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() % 1_000_000)
                        .unwrap_or(0);
                    eprintln!("[TRACE-BATCH t={:06}ms DISPATCH] {}", now, format_args!($($arg)*));
                }
            };
        }
        // Acquire each referenced slot's mutex. blocking_lock here is
        // safe — the request handlers have *released* the mutex
        // before pushing pending (their long-term claim is
        // `slot_in_use`, not the mutex).
        let mut guards: Vec<tokio::sync::MutexGuard<'_, Box<dyn crate::Session>>> =
            Vec::with_capacity(pending.len());
        for p in pending {
            tr_d!("locking inflight slot={}", p.slot_idx);
            guards.push(self.inflight_pool[p.slot_idx].blocking_lock());
            tr_d!("locked  inflight slot={}", p.slot_idx);
        }

        let n = pending.len();
        let slots: Vec<BatchSlot> = pending
            .iter()
            .enumerate()
            .map(|(s, p)| BatchSlot {
                idx: s,
                token_id: p.token_id,
                position: p.position,
            })
            .collect();

        let vocab = self.cfg.vocab_size;
        let mut logits_owned: Vec<Vec<f32>> = (0..n)
            .map(|_| Vec::with_capacity(vocab))
            .collect();

        // Deref each MutexGuard<Box<dyn Session>> to a
        // `&mut dyn Session` and hand the distinct-by-index slice to
        // the shared batched dispatcher. Scope the reborrow so the
        // mutable borrow of `guards` ends before the explicit `drop`.
        {
            let mut inflights: Vec<&mut dyn crate::Session> = Vec::with_capacity(n);
            for g in guards.iter_mut() {
                let inflight: &mut dyn crate::Session = &mut ***g;
                inflights.push(inflight);
            }
            let mut logits_refs: Vec<&mut Vec<f32>> =
                logits_owned.iter_mut().collect();
            self.forward_decode_batched_with_inflights(
                inflights.as_mut_slice(),
                &slots,
                logits_refs.as_mut_slice(),
            )?;
        }

        for (s, p) in pending.iter().enumerate() {
            let logits = std::mem::take(&mut logits_owned[s]);
            tr_d!("send response slot={} logits_len={}", p.slot_idx, logits.len());
            let _ = p.response.send(Ok(logits));
        }
        tr_d!("dispatch_done dropping guards");
        drop(guards);
        Ok(())
    }

    /// Shared batched-decode dispatcher. Drives `slots.len()` concurrent
    /// decode steps through whichever topology is active, writing each
    /// slot's logits row into `logits_refs[slot.idx]`. The mutexes
    /// protecting each `Inflight` must be held by the caller for the
    /// duration of this call; pass distinct `&mut Inflight` refs
    /// (deref'd from the held `MutexGuard`s).
    /// Used by both `dispatch_batched_pending` (N≥1, scheduler-aware)
    /// and `dispatch_decode_one` (N=1, legacy single-decode path).
    fn forward_decode_batched_with_inflights(
        &self,
        inflights: &mut [&mut dyn crate::Session],
        slots: &[flambeau_qwen3_moe::forward::BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> anyhow::Result<()> {
        if inflights.is_empty() {
            bail!("forward_decode_batched_with_inflights: empty inflight slice");
        }
        self.model
            .forward_decode_batched(self, inflights, slots, logits_refs)
    }

    /// Phase 12.5 — single-slot decode through the batched path. Replaces
    /// the legacy `crate::model::decode_logits` free function. Caller
    /// must hold the slot's mutex (passing the live `&mut Inflight`
    /// borrowed from the guard).
    pub fn dispatch_decode_one(
        &self,
        inflight: &mut dyn crate::Session,
        token: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        use flambeau_qwen3_moe::forward::BatchSlot;
        let slots = [BatchSlot {
            idx: 0,
            token_id: token,
            position,
        }];
        let vocab = self.cfg.vocab_size;
        if logits_out.capacity() < vocab {
            logits_out.reserve(vocab - logits_out.capacity());
        }
        logits_out.clear();
        let mut inflights_arr: [&mut dyn crate::Session; 1] = [inflight];
        let mut logits_refs: [&mut Vec<f32>; 1] = [logits_out];
        self.forward_decode_batched_with_inflights(
            &mut inflights_arr,
            &slots,
            &mut logits_refs,
        )
    }
}

pub mod health;
pub use health::{agent_stats, health, models};

pub mod tokenize;
pub use tokenize::{detokenize, tokenize};

pub mod embeddings;
pub use embeddings::embeddings;

pub mod chat;
pub use chat::chat_completions;


pub mod completions;
pub use completions::completions;

/// **P1.7** — build one `ChatLogProbContent` entry for a single decoded
/// step. Calls [`flambeau_runtime::sampling::build_distribution`] to
/// reproduce the same penalty + temperature + top-k/top-p/min-p
/// transforms the sampler applied, then extracts the chosen token's
/// log-probability and the top-`top_n` alternatives.
/// Returns `None` if the chosen token is outside the post-filter
/// distribution (defensive — shouldn't happen because the sampler
/// drew from the same distribution). Logprobs below -100 are clamped
/// to -100, matching OpenAI's surface.
fn build_logprob_entry(
    tokenizer: &flambeau_quant::GgufTokenizer,
    logits: &[f32],
    sampling: &flambeau_runtime::Sampling,
    history: &[u32],
    chosen: u32,
    top_n: usize,
) -> Option<ChatLogProbContent> {
    use flambeau_runtime::sampling::build_distribution;
    let dist = build_distribution(logits, sampling, history);
    if dist.is_empty() {
        return None;
    }
    // Locate the chosen token's prob; if missing (filtered out), the
    // sampler couldn't have picked it, so skip the entry.
    let chosen_prob = dist
        .iter()
        .find(|(id, _)| *id == chosen)
        .map(|(_, p)| *p)?;
    let chosen_logprob = log_clamped(chosen_prob);

    // Top alternatives are already in the front of `dist` if it was
    // sorted (when filters were active). When filters are off,
    // `build_distribution` returns an unsorted full-vocab list — sort
    // a partial copy. Skip the chosen token from the alternatives;
    // OpenAI-spec lists only OTHER candidates here.
    let mut alts: Vec<(u32, f32)> = dist
        .iter()
        .filter(|(id, _)| *id != chosen)
        .copied()
        .collect();
    if alts.len() > top_n {
        alts.select_nth_unstable_by(top_n, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        alts.truncate(top_n);
    }
    alts.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let chosen_text = tokenizer.decode(&[chosen]).unwrap_or_default();
    let chosen_bytes = chosen_text.as_bytes().to_vec();

    let top_logprobs: Vec<TopLogProb> = alts
        .into_iter()
        .map(|(id, p)| {
            let t = tokenizer.decode(&[id]).unwrap_or_default();
            let bytes = t.as_bytes().to_vec();
            TopLogProb {
                token: t,
                logprob: log_clamped(p),
                bytes,
            }
        })
        .collect();

    Some(ChatLogProbContent {
        token: chosen_text,
        logprob: chosen_logprob,
        bytes: chosen_bytes,
        top_logprobs,
    })
}

/// log(p) clamped to OpenAI's [-100, 0] surface. Zero-prob → -100.
fn log_clamped(p: f32) -> f32 {
    if p <= 0.0 {
        -100.0
    } else {
        p.ln().max(-100.0)
    }
}


pub mod infill;
pub use infill::infill;

pub mod messages;
pub use messages::messages_anthropic;

/// Build an SSE stream from the blocking completion pipeline.
/// Emits OpenAI-compatible `chat.completion.chunk` frames: role
/// announcement, zero-or-more content / tool-call deltas, and a final
/// frame with `finish_reason` + `[DONE]` sentinel.
/// T3.1: tool-call deltas are emitted as the parser transitions — we
/// never buffer a tool call body to end-of-stream and then dump it as
/// raw text (the specific anti-pattern that produces the "raw XML at
/// end of stream" complaints against llama.cpp).
fn stream_completion_sse(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    tool_call_format: Option<String>,
    parallel_tool_calls: bool,
    relax_stop_mask: bool,
    include_usage: bool,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    use crate::tool_call_parser::{dispatcher, ParserEvent};

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let id = request_id("chatcmpl");
    let created = now_unix();
    let model = state.model_id.clone();

    // First chunk: role announcement, matches OpenAI's streaming shape.
    let role_frame = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant" },
            "finish_reason": null,
        }],
    });
    let _ = tx.try_send(Ok(Event::default().data(role_frame.to_string())));

    let state_clone = state.clone();
    let id_clone = id.clone();
    let model_clone = model.clone();
    let tx_clone = tx.clone();
    tokio::task::spawn_blocking(move || {
        // Build the parser once per request. Default format is
        // boot-detected from the chat template (L3); the request may
        // override explicitly.
        let parser_result = dispatcher(
            tool_call_format.as_deref(),
            state_clone.tool_call_format_default,
        );
        let mut parser = match parser_result {
            Ok(p) => p,
            Err(e) => {
                // Bad tool_call_format — surface as a streamed error.
                let err = json!({ "error": { "message": e.to_string() } });
                let _ = tx_clone
                    .blocking_send(Ok(Event::default().data(err.to_string())));
                let _ = tx_clone
                    .blocking_send(Ok(Event::default().data("[DONE]")));
                return;
            }
        };

        // Shared helpers for emitting the different chunk shapes. Each
        // returns `true` on successful send, `false` if the receiver
        // dropped — the outer pipeline aborts early in that case.
        let emit_chunk = |delta_value: serde_json::Value| -> bool {
            let frame = json!({
                "id": id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_clone,
                "choices": [{
                    "index": 0,
                    "delta": delta_value,
                    "finish_reason": null,
                }],
            });
            tx_clone
                .blocking_send(Ok(Event::default().data(frame.to_string())))
                .is_ok()
        };

        // T3.1 / T3.2: state threaded through the per-token text feed.
        let mut has_tool_calls = false;
        // T3.2: when parallel_tool_calls=false, set on the first
        // ToolCallClose so the next emit_delta call returns false and
        // the decode loop hard-terminates.
        let mut abort_after_close = false;

        // Drain a batch of parser events into SSE chunks. Returns
        // false when the receiver dropped OR a !parallel close fired.
        let emit_events = |events: Vec<ParserEvent>,
                           has_tool_calls: &mut bool,
                           abort_after_close: &mut bool|
         -> bool {
            for e in events {
                let sent = match e {
                    ParserEvent::TextDelta(s) => emit_chunk(json!({ "content": s })),
                    ParserEvent::ThinkDelta(_) => true, // discard; V3 will surface
                    ParserEvent::ToolCallOpen { index, name } => {
                        *has_tool_calls = true;
                        emit_chunk(json!({
                            "tool_calls": [{
                                "index": index,
                                "id": format!("call_{index}"),
                                "type": "function",
                                "function": { "name": name },
                            }]
                        }))
                    }
                    ParserEvent::ToolCallArgumentsDelta { index, arguments } => {
                        *has_tool_calls = true;
                        emit_chunk(json!({
                            "tool_calls": [{
                                "index": index,
                                "function": { "arguments": arguments },
                            }]
                        }))
                    }
                    ParserEvent::ToolCallClose { .. } => {
                        if !parallel_tool_calls {
                            *abort_after_close = true;
                        }
                        true
                    }
                };
                if !sent {
                    return false;
                }
            }
            true
        };

        // Token-level text feed: the decoder emits one text delta per
        // sampled token (or coalesced multi-byte run). We push into the
        // parser and stream out events as they come. Never buffer to
        // end-of-stream.
        let mut emit_delta = |text: &str| -> bool {
            let events = parser.push(text);
            let sent = emit_events(events, &mut has_tool_calls, &mut abort_after_close);
            sent && !abort_after_close
        };

        let res = run_completion_blocking_streaming(
            state_clone.clone(),
            prompt,
            params,
            relax_stop_mask,
            &mut emit_delta,
        );

        // Flush the parser — dumps any buffered mid-tag content and
        // closes out pending think text.
        let tail = parser.finish();
        let _ = emit_events(tail, &mut has_tool_calls, &mut abort_after_close);

        // finish_reason flips to "tool_calls" when the turn produced
        // any, per OpenAI contract.
        let finish_reason = if has_tool_calls {
            "tool_calls"
        } else {
            match &res {
                Ok((r, _, _)) => r.as_str(),
                Err(_) => "error",
            }
        };
        // L3 — emit `usage` on the final chunk so the chat UI can
        // compute prefill / token-generation rates. Token counts come
        // from run_completion_blocking_streaming's tuple return.
        let (prompt_tokens, completion_tokens) = match &res {
            Ok((_, p, c)) => (*p, *c),
            Err(_) => (0u32, 0u32),
        };
        // Finish chunk. When `include_usage=true`, the canonical OpenAI
        // shape carries `usage: null` here and ships the real usage on
        // the next (choices=[]) chunk; that's the form expected by the
        // openai-python SDK and LangChain. When false, embed `usage`
        // inline so existing clients that only watch the finish chunk
        // still get token counts.
        let done_frame = if include_usage {
            json!({
                "id": id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_clone,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
                "usage": serde_json::Value::Null,
            })
        } else {
            json!({
                "id": id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_clone,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens.saturating_add(completion_tokens),
                },
            })
        };
        let _ = tx_clone.blocking_send(Ok(Event::default().data(done_frame.to_string())));

        // P0.3 — canonical separate usage chunk. Emitted strictly after
        // the finish chunk and strictly before [DONE], with choices=[]
        // so clients that match on `choices[0].finish_reason` don't
        // double-fire.
        if include_usage {
            let usage_frame = json!({
                "id": id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_clone,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens.saturating_add(completion_tokens),
                },
            });
            let _ = tx_clone.blocking_send(Ok(Event::default().data(usage_frame.to_string())));
        }

        if let Err(e) = res {
            tracing::error!(
                target: "server.completion.error",
                err = format!("{e:#}"),
                "streaming chat error (full chain)"
            );
            let err = json!({ "error": { "message": format!("{e:#}") } });
            let _ = tx_clone.blocking_send(Ok(Event::default().data(err.to_string())));
        }
        let _ = tx_clone.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

/// Shared engine: text prompt → generated text + token counts + finish reason.
/// Runs inside `spawn_blocking` because HIP kernels + mutex hold are sync.
/// `relax_stop_mask`: when `true`, disables the first-N-token stop-token
/// suppression and the post-N stop-bias — used on turns where the client
/// supplied `tools[]` (T4.1). Tool-call responses are legitimately short
/// (a JSON blob fits in ~20 tokens); injecting the stop mask forces the
/// model to pad before emitting `</tool_call>`.
/// Engine return shape: text, prompt_tokens, completion_tokens,
/// finish_reason, optional per-token logprobs (P1.7), optional
/// reasoning_content (#233 — populated when the request set
/// `enable_thinking=true` and the model emitted a `<think>...</think>`
/// block; the leading reasoning is split off and returned here while
/// `text` keeps only the post-think answer).
type CompletionOutput = (
    String,
    u32,
    u32,
    String,
    Option<Vec<ChatLogProbContent>>,
    Option<String>,
);

async fn run_completion(
    state: SharedState,
    prompt: &str,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    let prompt = prompt.to_owned();
    tokio::task::spawn_blocking(move || {
        let ids = state
            .tokenizer
            .encode(&prompt)
            .context("tokenize prompt")?;
        run_completion_blocking_ids(state, ids, params, relax_stop_mask)
    })
    .await
    .map_err(|e| anyhow!("spawn_blocking join failed: {e}"))?
}

/// **P1.6b** — token-id entry point used by the FIM `/infill` route.
/// Skips the tokenizer string round-trip so callers that build prompts
/// directly out of pre-tokenized fragments (e.g., FIM specials wrapping
/// user prefix/suffix) don't depend on the tokenizer's special-token
/// auto-registration heuristic.
async fn run_completion_ids(
    state: SharedState,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    tokio::task::spawn_blocking(move || {
        run_completion_blocking_ids(state, prompt_ids, params, relax_stop_mask)
    })
    .await
    .map_err(|e| anyhow!("spawn_blocking join failed: {e}"))?
}

/// **P2.9b-i2-B-wire** — gate predicate for the scheduler-aware
/// chat handler. The scheduler path is a focused subset of the full
/// `run_completion_blocking_ids` flow: greedy, PP topology, no
/// spec-decode, no GPU sampler, no JSON mode, no logprobs, no tools.
/// All other configurations fall through to the legacy handler.
fn scheduler_can_engage(state: &ServerState, params: &SamplingParams) -> bool {
    if !state.batched_decode {
        return false;
    }
    // PP, TP, and Hybrid all supported. JSON / logprobs paths still go
    // through the legacy handler — those features carry extra device-
    // side state (JSON DFA, top-K logprobs grab) that isn't yet plumbed
    // through the scheduler-aware handler.
    let topo_ok = state.model.as_pp().is_some()
        || state.model.as_tp().is_some()
        || state.model.as_hybrid().is_some();
    if !topo_ok {
        return false;
    }
    !params.json_mode && params.collect_logprobs.is_none()
}

/// **P2.9b-i2-B-wire (cleanup 2026-05-03)** — chat handler that uses
/// the scheduler. Engaged when [`scheduler_can_engage`] returns true.
/// Releases the slot's mutex during the decode loop so the scheduler-
/// leader can `blocking_lock` other slots' mutexes for batched dispatch.
/// Supports the full host-sampler feature set: greedy or temp/top_k/
/// top_p/min_p sampling, repetition / presence / frequency penalties.
/// Does NOT (yet) support: spec-decode (MTP), JSON-grammar masking,
/// logprobs, GPU sampler. Those still go through the legacy
/// `run_completion_blocking_ids`.
fn run_completion_scheduler_pp_blocking(
    state: SharedState,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    let request_start = Instant::now();
    let slot_idx = state.claim_slot_blocking();
    // Wrap the body in a closure so any error path runs `release_slot`.
    let result = (|| -> Result<CompletionOutput> {
        if prompt_ids.is_empty() {
            bail!("prompt tokenized to 0 tokens");
        }
        let prompt_tokens = prompt_ids.len() as u32;

        tracing::info!(
            target: "server.completion.start",
            prompt_tokens,
            max_tokens = params.max_tokens,
            slot_idx,
            scheduler = true,
            "completion request accepted (scheduler path)"
        );

        let cluster: &flambeau_backend_hip::HipCluster = &state.cluster;
        let model = &state.model;
        let stop_ids = &state.tokenizer.stop_ids;
        let always_stop_ids = &state.tokenizer.always_stop_ids;
        let is_stop = |t: u32| stop_ids.contains(&t);
        let vocab = state.cfg.vocab_size;
        let sampling = &params.sampling;

        // Sampler shared across first-token + decode loop. Mirrors the
        // legacy handler's setup.
        let mut sampler = Sampler::from_seed(params.seed);
        sampler.reserve(vocab);

        // ---------- Stage 1: brief mutex hold for prefill + first-token sample.
        let first_next = {
            let mut guard = state.inflight_pool[slot_idx].blocking_lock();
            guard
                .reset_for_next_request()
                .context("reset inflight for new request")?;
            let mut logits_buf: Vec<f32> = Vec::with_capacity(vocab);
            // **#321** — TP/Hybrid prefill alloc serialiser. See field
            // doc on ServerState::prefill_serialiser.
            let _prefill_lock = if model.as_tp().is_some() || model.as_hybrid().is_some() {
                Some(state.qwen3_moe.as_ref().expect("TP/Hybrid serialiser requires qwen3-moe boot").prefill_serialiser.lock().unwrap())
            } else {
                None
            };
            // **#324** — for TP, hand the shared pre-allocated scratch
            // through so prefill_logits skips the per-call alloc.
            let mut tp_scratch_g = if model.as_tp().is_some() {
                Some(state.lock_tp_prefill_scratch()?)
            } else {
                None
            };
            let tp_pool: Option<
                &mut flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp,
            > = tp_scratch_g.as_mut().and_then(|g| g.as_mut());
            // **#229 prefix-cache lookup. Three outcomes:
            // FullHit: KV+GDN restored, logits cached → skip prefill.
            // PrefixHit: KV+GDN restored at chunk boundary → partial
            // prefill of the tail starting at n_matched.
            // Miss: full fresh prefill, capture both intermediate
            // (chunk-boundary) and final (full prompt) entries.
            let restore =
                state.prefix_cache_try_restore(&mut **guard, &prompt_ids)?;
            match restore {
                PrefixCacheRestore::FullHit { logits } => {
                    logits_buf.clear();
                    logits_buf.extend_from_slice(logits.as_ref());
                }
                PrefixCacheRestore::PrefixHit { n_matched } => {
                    crate::model::prefill_logits(
                        model,
                        cluster,
                        &mut **guard,
                        &prompt_ids[n_matched..],
                        n_matched,
                        &mut logits_buf,
                        tp_pool,
                        None,
                        state.prefill_ubatch,
                    )
                    .context("scheduler-path tail prefill (after prefix-hit)")?;
                    // After tail prefill we have full state — capture
                    // the FULL entry (with logits) for future requests.
                    state.prefix_cache_try_capture_full(
                        &**guard,
                        &prompt_ids,
                        &logits_buf,
                    );
                }
                PrefixCacheRestore::Miss => {
                    let mut boundary_cb = |snap, n_tok| {
                        state.prefix_cache_insert_intermediate(
                            &prompt_ids,
                            n_tok,
                            snap,
                        );
                        Ok(())
                    };
                    let cb_opt: Option<crate::model::BoundaryCallback<'_>> =
                        if state.prefix_cache.enabled() {
                            Some(&mut boundary_cb)
                        } else {
                            None
                        };
                    crate::model::prefill_logits(
                        model,
                        cluster,
                        &mut **guard,
                        &prompt_ids,
                        0,
                        &mut logits_buf,
                        tp_pool,
                        cb_opt,
                        state.prefill_ubatch,
                    )
                    .context("scheduler-path prefill")?;
                    state.prefix_cache_try_capture_full(
                        &**guard,
                        &prompt_ids,
                        &logits_buf,
                    );
                }
            }
            // First-token stop mask: NEG_INFINITY all stop ids so the
            // model is forced to emit a content token first.
            if !relax_stop_mask {
                for &sid in stop_ids {
                    if (sid as usize) < logits_buf.len() {
                        logits_buf[sid as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            sampler.sample(&logits_buf, sampling, &[])
        };

        let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
        generated.push(first_next);
        if is_stop(first_next) {
            return finalise(
                &state,
                prompt_tokens,
                generated,
                "stop",
                &params.stop_strings,
                None,
                params.enable_thinking,
            );
        }

        let mut last_token = first_next;
        let mut finish_reason = "length";
        const MIN_RESPONSE_TOKENS: usize = 24;
        const STOP_BIAS: f32 = 3.0;
        let user_stop_max = params
            .stop_strings
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);

        // ---------- Stage 2: decode loop via scheduler. NO mutex held.
        // **Cycle 3** — reuse one logits buffer across the whole decode
        // loop instead of allocating a fresh ~600 KB Vec per step.
        let mut logits: Vec<f32> = Vec::with_capacity(vocab);
        for step in 1..params.max_tokens as usize {
            let force_mask = step < MIN_RESPONSE_TOKENS && !relax_stop_mask;
            state
                .decode_via_scheduler_into(
                    slot_idx,
                    last_token,
                    prompt_ids.len() + step,
                    &mut logits,
                )
                .context("scheduler-path decode step")?;
            if !relax_stop_mask {
                for &sid in stop_ids {
                    if (sid as usize) < logits.len() {
                        if always_stop_ids.contains(&sid) {
                            logits[sid as usize] = f32::NEG_INFINITY;
                        } else if force_mask {
                            logits[sid as usize] = f32::NEG_INFINITY;
                        } else {
                            logits[sid as usize] -= STOP_BIAS;
                        }
                    }
                }
            }
            let next = sampler.sample(&logits, sampling, &generated);
            if is_stop(next) {
                finish_reason = "stop";
                break;
            }
            generated.push(next);
            last_token = next;
            if generated.len() >= params.max_tokens as usize {
                break;
            }
            // Stop-string detection on the recent decoded window
            // (mirrors the legacy handler).
            let need_user_check = !params.stop_strings.is_empty();
            if (need_user_check || !relax_stop_mask)
                && (step % 4 == 0 || step >= MIN_RESPONSE_TOKENS)
            {
                let n = generated.len();
                let token_window = 16.max((user_stop_max / 2).min(64));
                let from = n.saturating_sub(token_window);
                if let Ok(tail) = state.tokenizer.decode(&generated[from..]) {
                    let marker_hit = !relax_stop_mask
                        && !params.enable_thinking
                        && (tail.contains("</think>")
                            || tail.contains("<end_thought>")
                            || tail.contains("<end_think>")
                            || tail.contains("</thought>"));
                    let arch_hit = state
                        .model
                        .chat_stop_markers()
                        .iter()
                        .any(|m| tail.contains(*m));
                    let user_hit = params
                        .stop_strings
                        .iter()
                        .any(|s| tail.contains(s.as_str()));
                    if marker_hit || arch_hit || user_hit {
                        finish_reason = "stop";
                        break;
                    }
                }
            }
        }

        tracing::info!(
            target: "server.completion.finish",
            prompt_tokens,
            completion_tokens = generated.len() as u32,
            finish_reason,
            total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
            scheduler = true,
            "completion request finished (scheduler path)"
        );

        finalise(
            &state,
            prompt_tokens,
            generated,
            finish_reason,
            &params.stop_strings,
            None,
            params.enable_thinking,
        )
    })();
    state.release_slot(slot_idx);
    result
}

fn run_completion_blocking_ids(
    state: SharedState,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    // **P2.9b-i2-B-wire** — opt into the scheduler path when
    // conditions allow. Greedy + PP + no MTP + no JSON + no logprobs.
    if scheduler_can_engage(&state, &params) {
        return run_completion_scheduler_pp_blocking(state, prompt_ids, params, relax_stop_mask);
    }

    let request_start = Instant::now();

    // **P2.9b-i1 (multi-slot pool)** — acquire an idle inflight slot.
    // With N=1 (default) this is identical to P2.9a; with N>1 distinct
    // requests can hold separate slots and run their forwards through
    // the same GPU stream concurrently (kernel-serialised; true batched
    // throughput is P2.9b-i2).
    let (slot_idx, mut inflight_guard) = state.acquire_inflight_blocking();

    if prompt_ids.is_empty() {
        bail!("prompt tokenized to 0 tokens");
    }
    let prompt_tokens = prompt_ids.len() as u32;

    tracing::info!(
        target: "server.completion.start",
        prompt_tokens,
        max_tokens = params.max_tokens,
        slot_idx,
        "completion request accepted after queue wait"
    );

    let cluster: &HipCluster = &state.cluster;
    let model = &state.model;

    // Reset KV state on the pooled inflight before this request's
    // prefill — clears full-attn `current_tokens` and zeros GDN
    // recurrent state without freeing scratch buffers.
    inflight_guard
        .reset_for_next_request()
        .context("reset inflight for new request")?;
    // Shadow with a reborrow so existing `&mut inflight` / `&inflight`
    // call-site syntax works unchanged.
    let inflight: &mut dyn crate::Session = &mut **inflight_guard;

    // Sampler holds vocab-sized scratch reused across all decode steps
    // (C2 in RUST-PERF-CORRECTIONS.md). Reserve upfront to avoid the
    // first-token grow.
    let mut sampler = Sampler::from_seed(params.seed);
    sampler.reserve(state.cfg.vocab_size);
    let sampling = &params.sampling;
    let stop_ids = &state.tokenizer.stop_ids;
    let always_stop_ids = &state.tokenizer.always_stop_ids;
    let vocab = state.cfg.vocab_size;
    // Always allocate the logit buffer — we need it for the first-token
    // EOS mask regardless of sampling mode.
    let mut logits_buf: Vec<f32> = Vec::with_capacity(vocab);

    // **Sampler-D3 / D4 / Hybrid** — GPU-side top-K sampler. Opts in
    // via `FLAMBEAU_GPU_SAMPLER=1` for any non-greedy TP or Hybrid
    // request; the penalty-active path goes through
    // `run_gpu_topk_with_penalties` (D4 — applies repetition /
    // presence / frequency on device before topk) instead of
    // `run_gpu_topk` (D3 — bare topk). Resolves the head device
    // differently per topology:
    // - TP: global cluster's `decode.head_rank` device
    // - Hybrid: head stage's sub_cluster's head TP-rank device
    // (head_stage = pp_size - 1; head_rank within stage
    // defaults to 0 per ShardedForwardOneTokenScratchHybrid).
    // Phase 12.5 — temporarily disabled. The keep-logits-on-device
    // optimisation reads `pp/tp/hybrid_session.decode.per_rank[head].
    // output_head.logits_f32`, which was populated by the legacy
    // `forward_one_token_*_keep_logits_on_device` kernels. With decode
    // collapsed onto `forward_decode_batched_*`, those buffers go stale
    // (the batched output head writes to the *batched* scratch's
    // `output_head.logits_f32` instead). Re-wiring `resolve_head_logits`
    // to read the batched scratch's buffer is its own follow-up.
    let use_gpu_sampler = false;
    let _ = state.gpu_sampler;
    let mut gpu_scratch: Option<GpuSamplerScratch> = if use_gpu_sampler {
        // Resolve the head device for whichever topology is active.
        let head_device = if let (Some(p), Some(_)) = (model.as_pp(), inflight.as_pp()) {
            // PP head rank is the last shard.
            let head_rank = p.model.shards.len().saturating_sub(1);
            if head_rank >= cluster.ranks() {
                bail!(
                    "GPU sampler: PP head_rank={head_rank} >= cluster ranks {}",
                    cluster.ranks()
                );
            }
            cluster.device(head_rank)
        } else if let (Some(_), Some(s)) = (model.as_tp(), inflight.as_tp()) {
            let head_rank = s.decode.head_rank.0 as usize;
            if head_rank >= cluster.ranks() {
                bail!(
                    "GPU sampler: TP head_rank={head_rank} >= cluster ranks {}",
                    cluster.ranks()
                );
            }
            cluster.device(head_rank)
        } else if let (Some(hm), Some(s)) = (model.as_hybrid(), inflight.as_hybrid()) {
            let decode = &s.decode;
            let head_stage = decode.head_stage as usize;
            let stage_model = hm.model.stages.get(head_stage).ok_or_else(|| {
                anyhow!("GPU sampler: hybrid head_stage {head_stage} out of range")
            })?;
            let stage_scratch = decode
                .per_stage
                .get(head_stage)
                .ok_or_else(|| anyhow!("GPU sampler: hybrid decode missing head_stage"))?;
            let head_rank = stage_scratch.head_rank.0 as usize;
            if head_rank >= stage_model.sub_cluster.ranks() {
                bail!(
                    "GPU sampler: hybrid head_rank={head_rank} >= stage sub-cluster ranks {}",
                    stage_model.sub_cluster.ranks()
                );
            }
            stage_model.sub_cluster.device(head_rank)
        } else {
            bail!("GPU sampler: unsupported (model, inflight) combination");
        };
        Some(
            // K=2048 matches Sampler-A's `effective_top_k` default
            // for `top_p`/`min_p` callers without explicit `top_k`.
            // Smaller K caused the GPU sampler to bias multinomial
            // toward EOS at natural-endpoint positions.
            GpuSamplerScratch::new(head_device, 2048)
                .context("alloc GpuSamplerScratch")?,
        )
    } else {
        None
    };

    // Prefill. Always download logits so we can mask stop tokens on the
    // first generated token — Qwen3.6 sometimes argmaxes `<|im_end|>` as
    // the first response token on multi-turn prompts, producing an empty
    // reply. Suppress it until at least one content token is emitted.
    let prefill_start = Instant::now();
    // **#321** — TP/Hybrid prefill alloc serialiser. See field
    // doc on ServerState::prefill_serialiser.
    let _prefill_lock = if model.as_tp().is_some() || model.as_hybrid().is_some() {
        Some(state.qwen3_moe.as_ref().expect("TP/Hybrid serialiser requires qwen3-moe boot").prefill_serialiser.lock().unwrap())
    } else {
        None
    };
    // **#324** — for TP, hand the shared pre-allocated scratch through.
    let mut tp_scratch_g = if model.as_tp().is_some() {
        Some(state.lock_tp_prefill_scratch()?)
    } else {
        None
    };
    let tp_pool: Option<
        &mut flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp,
    > = tp_scratch_g.as_mut().and_then(|g| g.as_mut());
    // **#229 prefix-cache restore (legacy path). Three
    // outcomes per `prefix_cache_try_restore`. Bypass when logprobs
    // active.
    let cache_eligible = params.collect_logprobs.is_none();
    let restore = if cache_eligible {
        state.prefix_cache_try_restore(&mut *inflight, &prompt_ids)?
    } else {
        PrefixCacheRestore::Miss
    };
    match restore {
        PrefixCacheRestore::FullHit { logits } => {
            logits_buf.clear();
            logits_buf.extend_from_slice(logits.as_ref());
        }
        PrefixCacheRestore::PrefixHit { n_matched } => {
            prefill_logits(
                model,
                cluster,
                &mut *inflight,
                &prompt_ids[n_matched..],
                n_matched,
                &mut logits_buf,
                tp_pool,
                None,
                state.prefill_ubatch,
            )
            .context("legacy-path tail prefill (after prefix-hit)")?;
            state.prefix_cache_try_capture_full(&*inflight, &prompt_ids, &logits_buf);
        }
        PrefixCacheRestore::Miss => {
            let mut boundary_cb = |snap, n_tok| {
                state.prefix_cache_insert_intermediate(&prompt_ids, n_tok, snap);
                Ok(())
            };
            let cb_opt: Option<crate::model::BoundaryCallback<'_>> =
                if cache_eligible && state.prefix_cache.enabled() {
                    Some(&mut boundary_cb)
                } else {
                    None
                };
            prefill_logits(
                model,
                cluster,
                &mut *inflight,
                &prompt_ids,
                0,
                &mut logits_buf,
                tp_pool,
                cb_opt,
                state.prefill_ubatch,
            )
            .context("prefill logits")?;
            if cache_eligible {
                state.prefix_cache_try_capture_full(&*inflight, &prompt_ids, &logits_buf);
            }
        }
    }
    drop(tp_scratch_g);
    drop(_prefill_lock);
    for &sid in stop_ids {
        if (sid as usize) < logits_buf.len() {
            logits_buf[sid as usize] = f32::NEG_INFINITY;
        }
    }
    // First token: ALWAYS goes through the host path. Prefill writes
    // logits to its own local scratch (disposed at end of prefill); the
    // decode scratch's `output_head.logits_f32` is uninitialised at
    // this point. The GPU sampler path activates from the SECOND token
    // onwards once the decode forward populates the right buffer.
    // (One-shot first token doesn't matter perf-wise; D3 Phase B's
    // bigger win is the per-step decode DtoH-skip.)
    // Phase 12.5 — `inv_temp` was the GPU-sampler entry, retired with
    // the keep-on-device branch. Re-introduce when `run_gpu_topk` re-
    // wires onto the batched output buffer.
    let _inv_temp = if sampling.temperature > 0.0 {
        1.0 / sampling.temperature
    } else {
        1.0
    };
    // **#236 P0.1b** — JSON state primed BEFORE first_next. When the
    // request is in json_mode AND the chat handler detected an
    // assistant prefill (last message had `role: "assistant"`),
    // `params.json_prime_bytes` carries those bytes; the JSON state
    // machine advances through them before the first decoded token,
    // so the mask honours the model's actual continuation position.
    // For non-prefill json_mode requests the prime is empty and the
    // state starts at `JsonState::new()`. Created here (rather than
    // post-first_next as before) so the mask can apply to the very
    // first sampled token.
    let mut json_state: Option<JsonState> = if params.json_mode {
        let mut js = JsonState::new();
        if !params.json_prime_bytes.is_empty() {
            let _ = js.feed_slice(&params.json_prime_bytes);
        }
        Some(js)
    } else {
        None
    };
    // **#236 P0.1b** — mask first-token logits when json_mode is
    // active. Without this the very first sampled token can violate
    // the grammar (e.g. emit `Hello` before `{`), and the existing
    // post-sample feed_slice would silently advance into a doomed
    // state. Top-K=2048 candidate cap keeps this cheap (~1–2 ms) on
    // a 151 k Qwen vocab.
    if let Some(js) = json_state.as_ref() {
        gpu_sampler::apply_json_mask_to_logits(
            js,
            &state.tokenizer,
            &mut logits_buf,
            /*max_candidates=*/ 2048,
        );
    }
    let first_next = sampler.sample(&logits_buf, sampling, &[]);
    let is_greedy = sampling.is_greedy();
    tracing::info!(
        target: "server.completion.first_token",
        prompt_tokens,
        ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
        greedy = is_greedy,
        topology = model.topology(),
        "first token produced (time-to-first-token)"
    );

    // **P1.7** — accumulate logprobs when the request opted in AND
    // the path supports it (host sampler only; GPU sampler / spec-
    // decode silently disable). `top_logprobs` may be 0 → just the
    // chosen token's logprob.
    let mut logprobs_acc: Option<Vec<ChatLogProbContent>> =
        if params.collect_logprobs.is_some() && !use_gpu_sampler {
            Some(Vec::with_capacity(params.max_tokens as usize))
        } else {
            if params.collect_logprobs.is_some() {
                tracing::warn!(
                    target: "server.logprobs",
                    "logprobs requested but path is GPU sampler — returning null logprobs"
                );
            }
            None
        };
    if let Some(lp) = logprobs_acc.as_mut() {
        if let Some(entry) = build_logprob_entry(
            &state.tokenizer,
            &logits_buf,
            sampling,
            &[],
            first_next,
            params.collect_logprobs.unwrap_or(0) as usize,
        ) {
            lp.push(entry);
        }
    }

    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
    let is_stop = |t: u32| stop_ids.contains(&t);

    // **P0.1** — advance JSON state by first_next's bytes (the
    // pre-sample mask above already filtered candidates that would
    // invalidate the running JSON, so this should always succeed
    // when the mask was active).
    if let Some(js) = json_state.as_mut() {
        if let Ok(text) = state.tokenizer.decode(&[first_next]) {
            let _ = js.feed_slice(text.as_bytes());
        }
    }

    generated.push(first_next);
    // First-token stop mask means `is_stop(first_next)` cannot fire here,
    // but we keep the check as a defensive guard for future logit-mask
    // changes.
    if is_stop(first_next) {
        // Slot stays pooled; mutex releases on function return.
        return finalise(&state, prompt_tokens, generated, "stop", &params.stop_strings, logprobs_acc, params.enable_thinking);
    }

    let mut finish_reason = "length";
    let mut last_token = first_next;
    // Mask stop tokens for the first MIN_RESPONSE_TOKENS steps. Qwen3.6
    // on multi-turn prompts otherwise emits `<|im_end|>` after 0-1 content
    // tokens, producing unusable one-word replies. MIN is small enough
    // that short on-topic answers ("Yes.", "42.") are still possible.
    // T4.1: tool-call turns disable this entirely. When the model is
    // asked to emit a `<tool_call>…</tool_call>` body it may legitimately
    // take only ~10 tokens; forcing 24 content tokens before allowing
    // stop injects noise between the body and the `<|im_end|>` and
    // breaks downstream parsing. `relax_stop_mask` flips both knobs to
    // no-ops — trust the model on turns where `tools[]` is present.
    // **Sampler-G (2026-04-30)** — was 24, lowered to 8. With Qwen3.6-27B
    // at temp=0.7+top_p=0.8 the 24-token floor forced the model to keep
    // generating 13+ tokens past natural endpoints like
    // "Hello! How can I help you today?" (~10 tokens), at which point it
    // wandered into reasoning-marker leaks (`</think>`, `<end_thought>`),
    // hallucinated chat formats (`<|user|>\n<|assistant|>`), or duplicate-
    // the-response loops. 8 lets short greetings stop naturally; the
    // first-token NEG_INFINITY mask still prevents immediate-EOS on
    // multi-turn prompts.
    const MIN_RESPONSE_TOKENS: usize = 24;
    // Nats subtracted from every stop-token logit beyond MIN_RESPONSE_TOKENS.
    // was 0.5 (was 3.0 before that). Even 0.5 is enough to push
    // EOS below the next-best continuation when the model wants to stop at
    // the end of a paragraph; under top_k=20 sampling the next-best is
    // typically "regenerate the paragraph" → whole-block repetition loops
    // (observed live on Coder-Next acknowledgement responses). The
    // first-token NEG_INFINITY mask + MIN_RESPONSE_TOKENS=24 hard mask are
    // sufficient on their own to prevent immediate-EOS failure modes;
    // beyond that, do not bias the model's natural stopping decision.
    // **Sampler-G follow-up (2026-04-30)** — non-zero STOP_BIAS subtracts
    // from every stop-token logit beyond MIN_RESPONSE_TOKENS. Was 0.0;
    // raised to 3.0 because Qwen3.6-27B at default sampling
    // (temp=1.0 / top_p=0.95 / top_k=20 — what Open-WebUI sends when
    // the user hasn't set an override) picks `<|im_end|>` at the
    // first natural sentence break, e.g. "I will include the full
    // implementation below." → STOP, instead of actually delivering
    // the implementation. 1.5 nats was insufficient at temp=1.0;
    // 3.0 is enough to keep EOS below the next-best continuation at
    // most natural endpoints. The prior concern about `0.5
    // → repeat loops` was driven by Coder-Next-80B specifically;
    // Qwen3.6 doesn't show that failure at this bias on chat tests.
    const STOP_BIAS: f32 = 3.0;
    for step in 1..params.max_tokens as usize {
        let force_mask = step < MIN_RESPONSE_TOKENS && !relax_stop_mask;
        // Phase 12.5 — decode goes through the host-path DtoH always.
        // The GPU sampler keep-on-device branch was retired with the
        // decode/batched collapse (see `use_gpu_sampler` initialiser).
        let next = {
            state
                .dispatch_decode_one(
                    &mut *inflight,
                    last_token,
                    prompt_ids.len() + step,
                    &mut logits_buf,
                )
                .context("decode step logits")?;
            if !relax_stop_mask {
                for &sid in stop_ids {
                    if (sid as usize) < logits_buf.len() {
                        // Sampler-G — `<think>` / `</think>` always
                        // get NEG_INFINITY, even outside the early
                        // window: the chat template ran with
                        // `enable_thinking=false` so the model
                        // should never emit them.
                        if always_stop_ids.contains(&sid) {
                            logits_buf[sid as usize] = f32::NEG_INFINITY;
                        } else if force_mask {
                            logits_buf[sid as usize] = f32::NEG_INFINITY;
                        } else {
                            logits_buf[sid as usize] -= STOP_BIAS;
                        }
                    }
                }
            }
            // **#236 P0.1b** — host-path JSON mask. Mirrors the
            // GPU-path mask above: set logit of any top-K candidate
            // that would invalidate the running JSON to NEG_INFINITY
            // before the sampler picks. Only active when the request
            // set `response_format=json_object`. Top-K=2048 cap keeps
            // the per-token cost in the low-ms range.
            if let Some(js) = json_state.as_ref() {
                gpu_sampler::apply_json_mask_to_logits(
                    js,
                    &state.tokenizer,
                    &mut logits_buf,
                    /*max_candidates=*/ 2048,
                );
            }
            // Pass `generated` as history so penalties can fire on
            // repeats / frequent tokens. T4.b.2 — without this,
            // Qwen3.5/3.6 agent loops degrade to long-CoT drift.
            let next = sampler.sample(&logits_buf, sampling, &generated);
            // P1.7 — collect per-token logprobs (host path only).
            if let Some(lp) = logprobs_acc.as_mut() {
                if let Some(entry) = build_logprob_entry(
                    &state.tokenizer,
                    &logits_buf,
                    sampling,
                    &generated,
                    next,
                    params.collect_logprobs.unwrap_or(0) as usize,
                ) {
                    lp.push(entry);
                }
            }
            next
        };
        // P0.1 — advance JSON state with the chosen token's bytes.
        if let Some(js) = json_state.as_mut() {
            if let Ok(text) = state.tokenizer.decode(&[next]) {
                let _ = js.feed_slice(text.as_bytes());
            }
        }
        generated.push(next);
        last_token = next;
        if is_stop(next) {
            finish_reason = "stop";
            break;
        }
        // **Sampler-G** — string-level stop on reasoning markers.
        // The model can route around the single-token `</think>`
        // mask by emitting the multi-token text form. Detokenize
        // the recent tail and stop if a leak is present. Final
        // response cleanup happens in `finalise`.
        // **P0.2** — same mechanism extended to the per-request
        // `stop` strings. Tail window grows with the longest
        // user stop so multi-token caller stops are catchable.
        let user_stop_max = params
            .stop_strings
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        let need_user_check = !params.stop_strings.is_empty();
        if (need_user_check || !relax_stop_mask)
            && (step % 4 == 0 || step >= MIN_RESPONSE_TOKENS)
        {
            let n = generated.len();
            // 16 tokens covers ≥48 chars typical; widen if a user
            // stop string is longer than ~32 chars.
            let token_window = 16.max((user_stop_max / 2).min(64));
            let from = n.saturating_sub(token_window);
            if let Ok(tail) = state.tokenizer.decode(&generated[from..]) {
                let marker_hit = !relax_stop_mask
                    && !params.enable_thinking
                    && (tail.contains("</think>")
                        || tail.contains("<end_thought>")
                        || tail.contains("<end_think>")
                        || tail.contains("</thought>"));
                let arch_hit = state
                    .model
                    .chat_stop_markers()
                    .iter()
                    .any(|m| tail.contains(*m));
                let user_hit = params
                    .stop_strings
                    .iter()
                    .any(|s| tail.contains(s.as_str()));
                if marker_hit || arch_hit || user_hit {
                    finish_reason = "stop";
                    break;
                }
            }
        }
    }

    // Dispose GPU sampler scratch (if allocated) before tearing down
    // the inflight session. The dispose device must match the device
    // the scratch was allocated on (recorded at construction time);
    // resolve it the same way the constructor did, depending on
    // topology.
    if let Some(scratch) = gpu_scratch.take() {
        let head_device = if let (Some(p), Some(_)) = (model.as_pp(), inflight.as_pp()) {
            cluster.device(p.model.shards.len().saturating_sub(1))
        } else if let (Some(_), Some(s)) = (model.as_tp(), inflight.as_tp()) {
            cluster.device(s.decode.head_rank.0 as usize)
        } else if let (Some(hm), Some(s)) = (model.as_hybrid(), inflight.as_hybrid()) {
            let decode = &s.decode;
            let head_stage = decode.head_stage as usize;
            let stage_model = hm.model.stages.get(head_stage).ok_or_else(|| {
                anyhow!("dispose GpuSamplerScratch: hybrid head_stage out of range")
            })?;
            let stage_scratch = decode.per_stage.get(head_stage).ok_or_else(|| {
                anyhow!("dispose GpuSamplerScratch: hybrid decode missing head_stage")
            })?;
            stage_model
                .sub_cluster
                .device(stage_scratch.head_rank.0 as usize)
        } else {
            bail!("dispose GpuSamplerScratch: unsupported topology");
        };
        scratch
            .dispose(head_device)
            .context("dispose GpuSamplerScratch")?;
    }

    // **P2.9a (slot pool)** — no dispose. The pooled inflight stays
    // allocated; releasing the mutex returns the slot to the pool
    // for the next request, which will reset_for_next_request on
    // claim. Removing per-request dispose saves the ~ms cost of
    // session/scratch teardown + re-alloc.
    let _ = inflight; // silence unused-warn after removing dispose

    tracing::info!(
        target: "server.completion.finish",
        prompt_tokens,
        completion_tokens = generated.len() as u32,
        finish_reason,
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "completion request finished"
    );

    let result = finalise(
        &state,
        prompt_tokens,
        generated,
        finish_reason,
        &params.stop_strings,
        logprobs_acc,
        params.enable_thinking,
    )?;
    // Sampler-G debug — emit the completed response text (head + tail
    // preview) so we can correlate request shape with what the model
    // actually produced. Truncated to 240 chars on each end so the
    // log line stays readable.
    let resp_preview = preview_text(&result.0, 240);
    tracing::info!(
        target: "server.resp",
        completion_tokens = result.2,
        finish_reason = %result.3,
        resp = %resp_preview,
        "chat_completions response"
    );
    Ok(result)
}

/// Truncate `text` to a head/tail preview suitable for log lines.
/// Replaces newlines with `\n` for single-line readability.
fn preview_text(text: &str, head: usize) -> String {
    let n_chars = text.chars().count();
    let escape = |s: &str| s.replace('\n', "\\n");
    if n_chars <= head * 2 + 20 {
        return escape(text);
    }
    let head_str: String = text.chars().take(head).collect();
    let tail_str: String = text.chars().skip(n_chars - head).collect();
    format!("{} … <{}c omitted> … {}", escape(&head_str), n_chars - head * 2, escape(&tail_str))
}

/// Streaming variant: pushes text deltas through `emit` as each token is
/// produced. Returns the finish reason on success.
/// Streaming variant. Returns `(finish_reason, prompt_tokens,
/// completion_tokens)` so the SSE producer can emit a final usage
/// chunk (L3 — pp/tg indicators in the chat UI).
fn run_completion_blocking_streaming(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    relax_stop_mask: bool,
    emit: &mut dyn FnMut(&str) -> bool,
) -> Result<(String, u32, u32)> {
    let request_start = Instant::now();

    // **P2.9b-i1 (multi-slot pool)** — acquire an idle inflight slot.
    // Same lifecycle as run_completion_blocking_ids.
    let (slot_idx, mut inflight_guard) = state.acquire_inflight_blocking();

    let prompt_ids = state.tokenizer.encode(&prompt).context("tokenize prompt")?;
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to 0 tokens");
    }
    let prompt_tokens = prompt_ids.len() as u32;

    tracing::info!(
        target: "server.completion.start",
        prompt_tokens,
        max_tokens = params.max_tokens,
        stream = true,
        slot_idx,
        "streaming completion accepted"
    );

    let cluster: &HipCluster = &state.cluster;
    let model = &state.model;
    let stop_ids = &state.tokenizer.stop_ids;
    let always_stop_ids = &state.tokenizer.always_stop_ids;
    let is_stop = |t: u32| stop_ids.contains(&t);

    inflight_guard
        .reset_for_next_request()
        .context("reset inflight for new streaming request")?;
    let inflight: &mut dyn crate::Session = &mut **inflight_guard;

    let mut sampler = Sampler::from_seed(params.seed);
    sampler.reserve(state.cfg.vocab_size);
    let sampling = &params.sampling;
    let is_greedy = sampling.is_greedy();
    let vocab = state.cfg.vocab_size;
    let mut logits_buf: Vec<f32> = Vec::with_capacity(vocab);

    // First token: always download logits and mask stop ids. See the
    // non-streaming path for the rationale (multi-turn Qwen3.6 argmaxes
    // `<|im_end|>` immediately otherwise).
    let prefill_start = Instant::now();
    // **#321** — TP/Hybrid prefill alloc serialiser. See field
    // doc on ServerState::prefill_serialiser.
    let _prefill_lock = if model.as_tp().is_some() || model.as_hybrid().is_some() {
        Some(state.qwen3_moe.as_ref().expect("TP/Hybrid serialiser requires qwen3-moe boot").prefill_serialiser.lock().unwrap())
    } else {
        None
    };
    // **#324** — for TP, hand the shared pre-allocated scratch through.
    let mut tp_scratch_g = if model.as_tp().is_some() {
        Some(state.lock_tp_prefill_scratch()?)
    } else {
        None
    };
    let tp_pool: Option<
        &mut flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp,
    > = tp_scratch_g.as_mut().and_then(|g| g.as_mut());
    // **#229 prefix-cache restore (streaming path).
    let restore_stream =
        state.prefix_cache_try_restore(&mut *inflight, &prompt_ids)?;
    match restore_stream {
        PrefixCacheRestore::FullHit { logits } => {
            logits_buf.clear();
            logits_buf.extend_from_slice(logits.as_ref());
        }
        PrefixCacheRestore::PrefixHit { n_matched } => {
            prefill_logits(
                model,
                cluster,
                &mut *inflight,
                &prompt_ids[n_matched..],
                n_matched,
                &mut logits_buf,
                tp_pool,
                None,
                state.prefill_ubatch,
            )
            .context("streaming-path tail prefill (after prefix-hit)")?;
            state.prefix_cache_try_capture_full(&*inflight, &prompt_ids, &logits_buf);
        }
        PrefixCacheRestore::Miss => {
            let mut boundary_cb = |snap, n_tok| {
                state.prefix_cache_insert_intermediate(&prompt_ids, n_tok, snap);
                Ok(())
            };
            let cb_opt: Option<crate::model::BoundaryCallback<'_>> =
                if state.prefix_cache.enabled() {
                    Some(&mut boundary_cb)
                } else {
                    None
                };
            prefill_logits(
                model,
                cluster,
                &mut *inflight,
                &prompt_ids,
                0,
                &mut logits_buf,
                tp_pool,
                cb_opt,
                state.prefill_ubatch,
            )
            .context("prefill logits")?;
            state.prefix_cache_try_capture_full(&*inflight, &prompt_ids, &logits_buf);
        }
    }
    drop(tp_scratch_g);
    drop(_prefill_lock);
    for &sid in stop_ids {
        if (sid as usize) < logits_buf.len() {
            logits_buf[sid as usize] = f32::NEG_INFINITY;
        }
    }
    let first_next = sampler.sample(&logits_buf, sampling, &[]);
    tracing::info!(
        target: "server.completion.first_token",
        prompt_tokens,
        ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
        greedy = is_greedy,
        topology = model.topology(),
        "first token produced (time-to-first-token)"
    );

    // Rolling decode: re-decode the full generated list each step and emit
    // the UTF-8 suffix beyond what was already emitted. Handles multi-byte
    // tokens without surfacing partial codepoints to the client.
    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
    let mut emitted_text = String::new();
    // incremental detokenize state. `decode_cursor` is the
    // index of the first token NOT yet decoded into clean emitted bytes.
    // `pending_emitted_in_segment` tracks how many bytes of the current
    // open segment (`generated[decode_cursor..]`) we've already streamed
    // to the client, so a re-decode after a multi-byte boundary doesn't
    // re-emit safe bytes. When the segment finishes cleanly (no
    // trailing U+FFFD), `decode_cursor` advances and the segment resets.
    // Reduces per-step decode cost from O(generated.len()) to ~O(1) —
    // typical "open segment" is 1-3 tokens, only growing when a multi-
    // byte glyph straddles a BPE-token boundary.
    let mut decode_cursor: usize = 0;
    let mut pending_emitted_in_segment: usize = 0;

    let mut push_and_emit = |tok: u32,
                             generated: &mut Vec<u32>,
                             emitted_text: &mut String|
     -> Result<bool> {
        generated.push(tok);
        let stop_hit = is_stop(tok);
        // Decode only the still-open segment, not the full sequence.
        // `end` excludes the trailing stop token so its raw text never leaks.
        let end = if stop_hit { generated.len() - 1 } else { generated.len() };
        if end <= decode_cursor {
            return Ok(!stop_hit);
        }
        let raw = state
            .tokenizer
            .decode(&generated[decode_cursor..end])
            .context("decode")?;
        // Trim trailing U+FFFD: the HF BPE+ByteLevel decoder substitutes
        // it when the byte tail is an incomplete UTF-8 codepoint
        // (multi-byte glyph straddling two BPE tokens). The trimmed
        // bytes will resolve once the next token's bytes arrive.
        let safe = raw.trim_end_matches('\u{FFFD}');

        // Emit anything new within the current open segment.
        if safe.len() > pending_emitted_in_segment {
            let delta = &safe[pending_emitted_in_segment..];
            if !emit(delta) {
                return Ok(false);
            }
            emitted_text.push_str(delta);
            pending_emitted_in_segment = safe.len();
        }

        // If nothing was trimmed, the segment closed cleanly: advance
        // the cursor and reset segment-local emit accounting.
        if safe.len() == raw.len() {
            decode_cursor = end;
            pending_emitted_in_segment = 0;
        }
        Ok(!stop_hit)
    };

    let alive = push_and_emit(first_next, &mut generated, &mut emitted_text)?;
    if !alive {
        // Slot stays pooled; mutex releases on function return.
        return Ok(("stop".into(), prompt_tokens, generated.len() as u32));
    }

    let mut finish_reason: &str = "length";
    let mut last_token = first_next;
    // See non-streaming path for the MIN_RESPONSE_TOKENS rationale.
    // Sampler-G — lowered from 24 to 8 to let short greetings stop
    // at natural endpoints instead of wandering into leak territory.
    const MIN_RESPONSE_TOKENS: usize = 24;
    // Nats subtracted from every stop-token logit beyond MIN_RESPONSE_TOKENS.
    // was 0.5 (was 3.0 before that). Even 0.5 is enough to push
    // EOS below the next-best continuation when the model wants to stop at
    // the end of a paragraph; under top_k=20 sampling the next-best is
    // typically "regenerate the paragraph" → whole-block repetition loops
    // (observed live on Coder-Next acknowledgement responses). The
    // first-token NEG_INFINITY mask + MIN_RESPONSE_TOKENS=24 hard mask are
    // sufficient on their own to prevent immediate-EOS failure modes;
    // beyond that, do not bias the model's natural stopping decision.
    // **Sampler-G follow-up (2026-04-30)** — non-zero STOP_BIAS subtracts
    // from every stop-token logit beyond MIN_RESPONSE_TOKENS. Was 0.0;
    // raised to 3.0 because Qwen3.6-27B at default sampling
    // (temp=1.0 / top_p=0.95 / top_k=20 — what Open-WebUI sends when
    // the user hasn't set an override) picks `<|im_end|>` at the
    // first natural sentence break, e.g. "I will include the full
    // implementation below." → STOP, instead of actually delivering
    // the implementation. 1.5 nats was insufficient at temp=1.0;
    // 3.0 is enough to keep EOS below the next-best continuation at
    // most natural endpoints. The prior concern about `0.5
    // → repeat loops` was driven by Coder-Next-80B specifically;
    // Qwen3.6 doesn't show that failure at this bias on chat tests.
    const STOP_BIAS: f32 = 3.0;
    // env-gated TP-decode profiling. When FLAMBEAU_PROFILE_DECODE
    // is set, enable HipEvent section recording for `n` warm-up-skipped decode
    // steps, then flush + dump aggregate per-section ms to stderr. Skips the
    // first 8 steps (cold-cache effects, allocator warmup).
    let profile_decode_n: usize = dev_usize("FLAMBEAU_PROFILE_DECODE", 0);
    let profile_skip: usize = 8;

    // **Lever B (host profile)** — per-section host wall accumulator,
    // gated by FLAMBEAU_HOST_PROFILE=1. Skips the first 8 decode
    // steps to dodge cold-cache effects, accumulates ms-per-section
    // over the rest, dumps a one-line summary at end-of-loop.
    let host_profile_on = dev_flag("FLAMBEAU_HOST_PROFILE");
    let mut hp_n: usize = 0;
    let mut hp_decode_us: u128 = 0;
    let mut hp_mask_us: u128 = 0;
    let mut hp_sample_us: u128 = 0;
    let mut hp_emit_us: u128 = 0;
    let mut hp_stopstr_us: u128 = 0;
    let mut hp_step_us: u128 = 0;

    for step in 1..params.max_tokens as usize {
        if profile_decode_n > 0 {
            if step == profile_skip + 1 {
                flambeau_backend_hip::profile::enable();
                tracing::info!(target: "server.profile", "enabled decode profiling");
            } else if step == profile_skip + profile_decode_n + 1 {
                match flambeau_backend_hip::profile::flush() {
                    Ok(stats) => {
                        let total: f32 = stats.iter().map(|s| s.total_ms).sum();
                        let mut msg = String::from(
                            "\n=== TP decode profile ===\nsection                    total_ms     count    mean_ms  ms/token\n",
                        );
                        for s in &stats {
                            msg.push_str(&format!(
                                "{:<24}  {:>10.2}  {:>8}  {:>10.4}  {:>9.3}\n",
                                s.name,
                                s.total_ms,
                                s.count,
                                s.mean_ms,
                                s.total_ms / profile_decode_n as f32
                            ));
                        }
                        msg.push_str(&format!(
                            "{:<24}  {:>10.2}  {:>8}  {:>10}  {:>9.3}\n",
                            "TOTAL_RECORDED",
                            total,
                            "-",
                            "-",
                            total / profile_decode_n as f32
                        ));
                        eprintln!("{}", msg);
                    }
                    Err(e) => tracing::error!(target: "server.profile", "flush: {e}"),
                }
            }
        }
        // T4.1: same relax-stop-mask behaviour as the non-streaming path.
        let force_mask = step < MIN_RESPONSE_TOKENS && !relax_stop_mask;
        let hp_step_t0 = if host_profile_on { Some(Instant::now()) } else { None };
        state
            .dispatch_decode_one(
                &mut *inflight,
                last_token,
                prompt_ids.len() + step,
                &mut logits_buf,
            )
            .context("decode step logits")?;
        let hp_after_decode = if host_profile_on { Some(Instant::now()) } else { None };
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    // Sampler-G — always_stop_ids (`<think>`/`</think>`)
                    // get NEG_INFINITY in every step, even outside the
                    // early-window mask. The model should never emit
                    // them when `enable_thinking=false`.
                    if always_stop_ids.contains(&sid) {
                        logits_buf[sid as usize] = f32::NEG_INFINITY;
                    } else if force_mask {
                        logits_buf[sid as usize] = f32::NEG_INFINITY;
                    } else {
                        logits_buf[sid as usize] -= STOP_BIAS;
                    }
                }
            }
        }
        let hp_after_mask = if host_profile_on { Some(Instant::now()) } else { None };
        let next = sampler.sample(&logits_buf, sampling, &generated);
        let hp_after_sample = if host_profile_on { Some(Instant::now()) } else { None };
        let alive = push_and_emit(next, &mut generated, &mut emitted_text)?;
        let hp_after_emit = if host_profile_on { Some(Instant::now()) } else { None };
        last_token = next;
        if !alive {
            finish_reason = "stop";
            break;
        }
        // **Sampler-G** string-level stop. The model can route around
        // single-token `</think>` masks by emitting the multi-token
        // text form (`</`, `think`, `>` or `<end`, `_thought`, `>`).
        // Detect on the running emitted text and cut cleanly. Cheap:
        // string contains check on the trailing window only. Use a
        // char-safe slice — `emitted_text.len() - 64` can land inside
        // a multi-byte UTF-8 codepoint (e.g. `’` at byte 805..808),
        // which would panic. Walk back to the nearest char boundary.
        // P0.2 — user-supplied stop sequences also use this tail-window
        // detector. Window size grows with the longest user stop so
        // multi-codepoint caller stops are catchable.
        let user_stop_max = params
            .stop_strings
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        if !relax_stop_mask || !params.stop_strings.is_empty() {
            let window_bytes = 64.max(user_stop_max + 16);
            let tail_window = if emitted_text.len() > window_bytes {
                let mut start = emitted_text.len() - window_bytes;
                while start < emitted_text.len() && !emitted_text.is_char_boundary(start) {
                    start += 1;
                }
                &emitted_text[start..]
            } else {
                emitted_text.as_str()
            };
            let marker_hit = !relax_stop_mask
                && !params.enable_thinking
                && (tail_window.contains("</think>")
                    || tail_window.contains("<end_thought>")
                    || tail_window.contains("<end_think>")
                    || tail_window.contains("</thought>"));
            let user_hit = params
                .stop_strings
                .iter()
                .any(|s| tail_window.contains(s.as_str()));
            if marker_hit || user_hit {
                finish_reason = "stop";
                tracing::info!(
                    target: "server.completion.string_stop",
                    user_stop = user_hit,
                    "string-level stop on reasoning-marker or user-stop leak"
                );
                break;
            }
        }
        // **Lever B (host profile)** — accumulate per-section wall.
        // Skip first 8 steps (cold cache / allocator warmup).
        if host_profile_on && step > profile_skip {
            let t_step_end = Instant::now();
            if let (Some(t0), Some(td), Some(tm), Some(ts), Some(te)) =
                (hp_step_t0, hp_after_decode, hp_after_mask, hp_after_sample, hp_after_emit)
            {
                hp_n += 1;
                hp_decode_us += (td - t0).as_micros();
                hp_mask_us += (tm - td).as_micros();
                hp_sample_us += (ts - tm).as_micros();
                hp_emit_us += (te - ts).as_micros();
                hp_stopstr_us += (t_step_end - te).as_micros();
                hp_step_us += (t_step_end - t0).as_micros();
            }
        }
    }

    if host_profile_on && hp_n > 0 {
        let f = hp_n as f64;
        eprintln!(
            "\n=== HOST decode profile (n={hp_n}, post-warmup) ===\n  decode_logits  : {:>7.3} ms/tok\n  stop-mask      : {:>7.3} ms/tok\n  sampler.sample : {:>7.3} ms/tok\n  push_and_emit  : {:>7.3} ms/tok\n  stopstr_check  : {:>7.3} ms/tok\n  TOTAL_per_step : {:>7.3} ms/tok\n",
            hp_decode_us as f64 / f / 1000.0,
            hp_mask_us as f64 / f / 1000.0,
            hp_sample_us as f64 / f / 1000.0,
            hp_emit_us as f64 / f / 1000.0,
            hp_stopstr_us as f64 / f / 1000.0,
            hp_step_us as f64 / f / 1000.0,
        );
    }

    // Slot stays pooled; mutex releases on function return.

    tracing::info!(
        target: "server.completion.finish",
        prompt_tokens,
        completion_tokens = generated.len() as u32,
        finish_reason,
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "streaming completion finished"
    );

    // Sampler-G debug — preview of what we actually streamed.
    let resp_preview = preview_text(&emitted_text, 240);
    tracing::info!(
        target: "server.resp",
        completion_tokens = generated.len() as u32,
        finish_reason = %finish_reason,
        resp = %resp_preview,
        "streaming chat_completions response"
    );

    Ok((finish_reason.to_owned(), prompt_tokens, generated.len() as u32))
}

fn finalise(
    state: &ServerState,
    prompt_tokens: u32,
    mut generated: Vec<u32>,
    reason: &str,
    stop_strings: &[String],
    mut logprobs: Option<Vec<ChatLogProbContent>>,
    enable_thinking: bool,
) -> Result<CompletionOutput> {
    // Strip ALL stop tokens (eos, <|im_end|>, etc.) from decoded text so the
    // client sees clean content. Raw count preserved for `usage` honesty.
    // C6: `retain` mutates in place instead of allocating a second Vec.
    let stop_ids = &state.tokenizer.stop_ids;
    let completion_tokens = generated.len() as u32;
    // P1.7 — when logprobs were collected, drop entries that line up
    // with stop-token positions so the per-token list mirrors the
    // text-content `retain` filter.
    if let Some(lp) = logprobs.as_mut() {
        if lp.len() == generated.len() {
            let mut idx = 0;
            generated.retain(|t| {
                let keep = !stop_ids.contains(t);
                if !keep {
                    if idx < lp.len() {
                        lp.remove(idx);
                    }
                } else {
                    idx += 1;
                }
                keep
            });
        } else {
            // Length mismatch — keep the original retain semantics for
            // text and clear logprobs to avoid a misleading partial
            // mapping (should never happen in practice).
            generated.retain(|t| !stop_ids.contains(t));
            lp.clear();
        }
    } else {
        generated.retain(|t| !stop_ids.contains(t));
    }
    let mut text = state.tokenizer.decode(&generated).context("decode")?;
    // **#233** — split out the reasoning block when thinking was
    // enabled. The model emits `<think>{cot}</think>{answer}`; in
    // legacy (`enable_thinking=false`) mode, the chat template
    // suppresses the block and any leaked marker is treated as a
    // stop. In thinking mode, capture the cot and let the answer
    // through. The opening `<think>` is the very first token Qwen3.6
    // emits in this mode; we still defensively look for it before
    // splitting on `</think>`.
    let reasoning_content: Option<String> = if enable_thinking {
        if let Some(end_idx) = text.find("</think>") {
            let cot_raw = &text[..end_idx];
            let cot = cot_raw
                .trim_start_matches("<think>")
                .trim()
                .to_string();
            let answer_start = end_idx + "</think>".len();
            let answer = text[answer_start..].trim_start().to_string();
            text = answer;
            if cot.is_empty() {
                None
            } else {
                Some(cot)
            }
        } else {
            // Model didn't close the block — the entire generation is
            // still chain-of-thought (typically because we hit the
            // max_tokens cap before the answer started). Surface the
            // raw text as reasoning_content and leave content empty
            // so clients distinguish "no answer yet" from "empty
            // answer".
            let cot = text.trim_start_matches("<think>").trim().to_string();
            text = String::new();
            if cot.is_empty() {
                None
            } else {
                Some(cot)
            }
        }
    } else {
        // **Sampler-G** — truncate at any leaked reasoning marker.
        // The string-level stop in the decode loop catches these
        // mid-flight, but the marker itself is already in `text`;
        // cut before its first occurrence so the client sees a
        // clean response.
        for marker in ["</think>", "<end_thought>", "<end_think>", "</thought>"] {
            if let Some(idx) = text.find(marker) {
                text.truncate(idx);
            }
        }
        // Arch-specific chat-template fragment truncation. Each model
        // exposes its set via `Model::chat_stop_markers` (empty for
        // qwen3-moe; populated for gemma4 — see `gemma4_handle.rs`).
        // We've already string-stopped on these in the decode loop, but
        // the marker itself can land in `text` if it slipped into the
        // tail window or split a chunk boundary.
        for marker in state.model.chat_stop_markers() {
            if let Some(idx) = text.find(*marker) {
                text.truncate(idx);
            }
        }
        None
    };
    // **P0.2** — same treatment for caller-supplied stop sequences.
    // OpenAI semantics: the stop sequence itself is not part of the
    // returned content. Truncate at the earliest match across all
    // user stops.
    let mut earliest = text.len();
    for s in stop_strings {
        if let Some(idx) = text.find(s.as_str()) {
            if idx < earliest {
                earliest = idx;
            }
        }
    }
    text.truncate(earliest);
    // Trim trailing whitespace introduced by the now-removed marker.
    let trimmed_len = text.trim_end().len();
    text.truncate(trimmed_len);
    Ok((
        text,
        prompt_tokens,
        completion_tokens,
        reason.to_owned(),
        logprobs,
        reasoning_content,
    ))
}

pub mod errors;
pub use errors::{now_unix, queue_full_response, request_id, ApiError};


