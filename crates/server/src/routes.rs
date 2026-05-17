//! HTTP route handlers. Requires `hip` feature (loads real model).

use std::sync::Arc;

use anyhow::{anyhow, bail, Context};
use flambeau_backend_hip::HipCluster;
use crate::model_cfg::ServerModelCfg;
use flambeau_quant::{ChatTemplate, GgufTokenizer};
use tokio::sync::Mutex;

use crate::qwen3moe_handle::{
    capture_kv_from_inflight, restore_kv_into_inflight, snapshot_bytes, LoadedModel,
    Qwen3MoeModelExt, Qwen3MoeSessionExt,
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

pub use crate::qwen3moe_handle::Qwen3MoeServerExtras;

/// Server-wide shared state — built once at startup.
pub struct ServerState {
    pub model_id: String,
    pub cfg: ServerModelCfg,
    /// PP or TP loaded model. Handlers dispatch via the
    /// `crate::qwen3moe_handle::{prefill_logits, decode_logits}` helpers; they
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

impl crate::model_handle::SessionContext for ServerState {
    fn cluster(&self) -> &flambeau_backend_hip::HipCluster {
        &self.cluster
    }
    fn max_inflight_slots(&self) -> usize {
        self.inflight_pool.len()
    }
    fn extras(&self) -> Option<&dyn std::any::Any> {
        self.qwen3_moe.as_ref().map(|e| e as &dyn std::any::Any)
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
            // when n_others_active==0. The lock lives in `qwen3_moe`
            // extras (qwen3-moe TP/Hybrid need it); gemma4 boots
            // without those extras, so we just skip the lock there.
            let _prefill_lock = self
                .qwen3_moe
                .as_ref()
                .map(|x| x.prefill_serialiser.lock().unwrap());
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
        use crate::model_handle::BatchSlot;
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
        slots: &[crate::model_handle::BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> anyhow::Result<()> {
        if inflights.is_empty() {
            bail!("forward_decode_batched_with_inflights: empty inflight slice");
        }
        self.model
            .forward_decode_batched(self, inflights, slots, logits_refs)
    }

    /// Phase 12.5 — single-slot decode through the batched path. Replaces
    /// the legacy `crate::qwen3moe_handle::decode_logits` free function. Caller
    /// must hold the slot's mutex (passing the live `&mut Inflight`
    /// borrowed from the guard).
    pub fn dispatch_decode_one(
        &self,
        inflight: &mut dyn crate::Session,
        token: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        use crate::model_handle::BatchSlot;
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

pub mod finalise;


pub mod infill;
pub use infill::infill;

pub mod messages;
pub use messages::messages_anthropic;

pub mod decode_loop;
pub(crate) use decode_loop::{
    run_completion, run_completion_blocking_streaming, run_completion_ids, stream_completion_sse,
};

pub mod errors;
pub use errors::{now_unix, queue_full_response, request_id, ApiError};


