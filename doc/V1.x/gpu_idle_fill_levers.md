# GPU-idle-fill levers — research synthesis

**Context**: the v2 bench cert
(`certs/perf/topology_compare/qwen36_27b_topo_x_mode_pp1024_2026_05_04.md`)
shows **GPU% mean 25 % at PP=4 / 44 % at TP=2 / 46 % at PP2TP2** at 4
concurrent users. Half or more of the silicon is idling. The user
asked: what's the production state-of-the-art for filling that gap?

This doc synthesizes findings from vLLM v1 RFC, Sarathi-Serve OSDI'24,
and DeepSeek/SGLang/TensorRT-LLM blogs into two flambeau-actionable
levers, plus a third "out of scope" pointer.

## Why GPU% mean is low: prefill ⊥ decode resource profiles

Prefill and decode have **inverted bottlenecks**:

|              | prefill                  | decode                          |
|--------------|--------------------------|---------------------------------|
| input shape  | (1, L) where L is large  | (1, 1) per step                  |
| matmul shape | wide (large M, K)        | thin (M=1)                       |
| bottleneck   | tensor cores / FLOPs     | HBM bandwidth (KV cache reads)   |
| roofline     | compute-bound            | bandwidth-bound                  |

When colocated on the same GPU, decode steps queued behind a long
prefill stall for the prefill duration. For a 27B model on MI50, our
v2 numbers: ~3 s prefill blocks ~26 t/s decode = 78 missed tokens.
That's the "TPOT spike" production traces show.

For PP topologies it's even worse: at single-slot the per-stage GPU
sits idle 75 % of the time at PP=4 because each stage processes one
microbatch sequentially.

## Lever 1: Sarathi-Serve mixed prefill+decode batching

### Pattern

(Agrawal et al., OSDI'24. Default-on in vLLM 2024+, TensorRT-LLM
opt-in, SGLang variant.)

1. **Chunked prefill**: split a request's prefill into K-token chunks
   (typical K = 256–1024). Each chunk is a valid forward pass over a
   contiguous prompt window; outputs assemble the same KV cache as
   one-shot prefill.
2. **Decode-maximal batching**: each iteration processes 1 prefill
   chunk + N decodes from already-prefilled slots in the **same forward
   pass**. The attention kernel handles a mixed batch with per-sequence
   variable-length context (`flash-attention varlen`).
3. **Per-iteration token budget**: 1 prefill chunk's K tokens + N
   decodes' 1 token each. Total per step ≈ K + N ≈ 520 tokens at
   K=512, N=8.

### Why this fills GPU idle

- Prefill chunk drives tensor-core utilization (compute-bound).
- Decodes drive HBM bandwidth (loading per-slot KV caches).
- Both subsystems saturate simultaneously — different parts of the
  roofline filled in the same step.

vLLM's blog claim: "naive static batching leaves 60 % of your GPU idle
on average." Mixed batching closes most of that gap **without**
changing kernels.

### Tradeoffs (chunk-size dial)

| chunk size | TTFT  | TPOT  | overhead |
|-----------:|------:|------:|---------:|
| small (256) | ↑ (more iters per prefill) | ↓ (decode barely perturbed) | + (kernel launch × n_chunks) |
| large (1024) | ↓ (~unchunked baseline) | ↑ (compute-heavy step delays decode) | small |

Sarathi reports K=512–1024 sweet spot for mid-sized models. Each chunk
also re-attends to all prior chunks (causal mask) → small attn
overhead vs one-shot, ~few % at flash-attn varlen.

### Mapping to flambeau

What we **already have**:
- Chunked prefill helper (Phase B1/B3/B4): the prefill driver can
  process arbitrary token chunks via `forward_prefill_*_logits` with
  `start_position` arg.
- Per-slot KV caches (P2.9b-i1).
- Per-layer batched-decode driver (`forward_decode_batched_hybrid`,
  N slots through each layer at n_tokens=N).

What we **don't have** (this is the lever):
- A scheduler that runs **mixed prefill-chunk + decode** in one forward
  pass.
- A "varlen attention" path that handles per-slot heterogeneous context
  lengths in a single batched-attn kernel call.

### Implementation skeleton

New scheduler iteration:
```rust
// At each iteration:
let mut tokens_in_batch: Vec<TokenSlot> = vec![];
let token_budget = 1024;  // configurable

// 1. Pick one pending prefill, slice off a chunk to fit budget.
if let Some(prefill_req) = scheduler.next_prefill() {
    let chunk_len = min(prefill_req.remaining_prefill(), token_budget);
    tokens_in_batch.push(PrefillChunk { req, chunk_len });
}

// 2. Fill remaining budget with decodes from already-prefilled slots.
let remaining = token_budget - chunk_len;
for slot in scheduler.ready_decodes().take(remaining) {
    tokens_in_batch.push(DecodeStep { slot });
}

// 3. Single forward pass over the mixed batch.
forward_mixed(&model, &mut tokens_in_batch)?;
```

The "varlen attention" piece: each slot in the batch has its own
context length. The decode tokens see their own KV history; the prefill
chunk sees the prefill history accumulated so far. flambeau's batched
attention kernel (#266b) handles n_slots × (head, head_dim) but with
**identical query length per slot (=1)** — extending it to varlen
queries is the kernel work.

Estimated effort: 3–5 sessions for chunk-aware scheduler + varlen attn
+ mixed prefill/decode forward driver.

### Projected win

vLLM blog reports 3–5× over naive static batching on H100 with
chunked-prefill + paged-attn + continuous batching combined. For
flambeau at 4×MI50 / 4 concurrent realistic-traffic mix, projected
**1.5–2.5× over current batched-decode** wall throughput. **This is
the highest-leverage single change** to push GPU% mean from 44 % up
toward 70–80 %.

## Lever 2: True multi-microbatch in-flight pipeline parallelism

### Pattern

(vLLM v1 RFC #11945, Option 2 — shipped May 2025. Underlying technique
predates LLMs: Megatron-LM 1F1B, GPipe.)

At PP=k stages, keep k microbatches in flight at any moment. At time t:
- Stage 0 processes microbatch (t)
- Stage 1 processes microbatch (t-1)
- Stage 2 processes microbatch (t-2)
- Stage 3 processes microbatch (t-3)

All k stages busy simultaneously — closes the per-stage idle gap that
naive sequential PP creates.

### Engine architecture (vLLM Option 2)

Async two-stage loop:
- `schedule(microbatch)` — picks the next microbatch's request set.
- `submit(microbatch)` — non-blocking dispatch to executor for stage 0.
- (driver thread receives stage outputs as they complete)
- `finish(microbatch)` — fires when oldest microbatch exits stage k-1.
- `update(microbatch_output)` — scheduler ingests result, updates KV +
  request state.

The async loop is event-driven on three triggers: new request,
existing request becomes schedulable, oldest microbatch finished. This
lets the scheduler keep submitting new microbatches without blocking
on stage k-1's output.

### Mapping to flambeau

What we **have** (`forward_decode_pipelined_hybrid`, #290/#294/#295):
- Per-slot iteration through stage 0 → peer_copy → stage 1 → peer_copy
  → ... → output_head.
- Async peer_copy with bridge events.
- The N=4 case dispatches all 4 slots through every stage sequentially
  on host before re-iterating.

What's **missing**:
- The host loop is **slot-major**, not **stage-major-with-overlap**.
  At PP=4, slot 0 walks all 4 stages sequentially before slot 1 starts
  stage 0. Pipelining benefit is "stage 1 work for slot 0 overlaps
  stage 0 work for slot 1" — limited fill.
- **True 1F1B-style scheduler**: at every "tick", advance every stage
  by one slot. Slot 3 finishes (head + DtoH) at tick t while slot 4
  starts stage 0. All ranks always have GPU work.

Live measurement (#295 cert): PP=4/N=4 pipelining delivers **1.0×**
(break-even) — confirms the missing piece. Per the math, full 1F1B at
PP=4/N=4 should be **2.3×**; we deliver 1.0× because the host
serialises slot iteration.

### Implementation shape

The vLLM Option 2 design would map to flambeau as:
1. Replace the slot-major outer loop in
   `forward_decode_pipelined_hybrid` with a **stage-major timestep
   loop** like the V2.25.h 1F1B prefill in `pp.rs:1480-1700`.
2. Per timestep t, for each rank r in 0..n_stages: dispatch slot
   `(t - r)` (if in range) on rank r.
3. After timestep N + n_stages - 1, all slots have completed.

For our 27B/PP=4/N=4 case: 7 timesteps total (4 + 4 - 1). Every
timestep exercises all 4 ranks once filled (timesteps 3..6 are "steady
state").

Estimated effort: 1–2 sessions (the V2.25.h prefill template gives
most of the structure).

### Projected win

PP=4/N=4 actual ceiling is 2.3× per the design table. Current 1.0×.
Closing this delta would lift the v2 cert from 1.34× (PP=4 batched) to
~3.0× — **alone clears the 3× gate on PP=4**.

## Lever 3 (out of scope for this rig): prefill-decode disaggregation

(Splitwise NSDI'24, DistServe OSDI'24, vLLM "P/D disagg" mode.)

Run prefill on one GPU pool, decode on another. KV cache transfers
between pools at handoff. Eliminates interference entirely.

Requires: ≥ 8 GPUs, fast interconnect (xGMI / NVLink) for the cache
transfer. On our 4×MI50 PCIe-only rig: not applicable.

## Recommendation: Lever 1 first

| | Lever 1 (Sarathi mixed) | Lever 2 (true 1F1B PP) |
|---|---|---|
| **Projected win** | 1.5–2.5× (full GPU% saturation) | 2.3× ceiling at PP=4/N=4 |
| **Topology applicability** | All (especially TP / hybrid) | PP only (we already have hybrid PP=2 baseline) |
| **Engineering** | 3–5 sessions: scheduler + varlen attn + mixed forward | 1–2 sessions: stage-major loop refactor |
| **Risk** | Touch attention kernel — correctness work | Mostly host-side loop refactor |
| **Hits cert gate?** | Maybe (combined with current pipelining: ~3×) | Yes, alone on PP=4 (1.34 × 2.3 ≈ 3.1×) |

**Lever 2 is the smaller engineering bet** with directly-projected
clearance of the 3× gate on PP=4. Recommended first.

**Lever 1 is the bigger lift** but applies across all topologies and
gives the dramatic GPU% utilisation win the v2 cert pointed at. Should
follow Lever 2 if we want pp2tp2 / tp2 to also clear 3× (currently
0.92–0.95×).

## References

- **Sarathi-Serve OSDI'24**: Agrawal et al., "Taming Throughput-
  Latency Tradeoff in LLM Inference with Sarathi-Serve". USENIX paper:
  https://www.usenix.org/system/files/osdi24-agrawal.pdf. Original
  arXiv: https://arxiv.org/abs/2308.16369.
- **vLLM Pipeline-Parallelism RFC**: GitHub issue #11945, two-stage
  async engine loop design. Implemented in vLLM v1 (#12996, May 2025).
  https://github.com/vllm-project/vllm/issues/11945.
- **vLLM chunked-prefill default-on**: Documented at
  https://docs.vllm.ai/en/v0.8.5/performance/optimization.html.
- **General Compute blog (2026)**: chunked prefill production
  experience. https://www.generalcompute.com/blog/chunked-prefill-
  overlapping-compute-and-communication.
- **Sequence-level 1F1B (NAACL'25)**: Seq1F1B for long-sequence
  pipeline parallelism. https://arxiv.org/abs/2406.03488.
