# How other frameworks handle MoE + TP — research note (2026-04-27)

Context: B5 diagnosed flambeau's TP parity drift on Qwen3.6-35B-A3B as
fundamental to its **intra-expert tensor-parallel** sharding scheme
(`ColParallel{dim=1}` for `ffn_gate/up_exps`, `RowParallel{dim=2}` for
`ffn_down_exps`) combined with MoE topology chaos. This note surveys
the strategies vLLM, DeepSpeed-MoE, Megatron-LM, and recent academic
work use to avoid the same problem.

## Two sharding modes for MoE + TP

The literature consistently distinguishes two ways to combine MoE and
multi-GPU partitioning:

| Mode | Definition | Per-expert compute lives on… | Communication |
|---|---|---|---|
| **Sharded experts** (a.k.a. "TP without EP" in vLLM, "expert tensor parallel" in DeepSpeed) | All experts present on every rank, but each expert's weight tensor is partitioned across ranks along the intermediate axis. | **Multiple ranks** — each rank computes a partial dot product over its slab of the inter dim; cross-rank AllReduce sums the partials. | AllReduce after the down-projection. |
| **Split experts** (a.k.a. "expert parallelism", "EP", `--enable-expert-parallel` in vLLM) | Each rank holds a **different subset of complete experts**. Weights for any one expert live entirely on one rank. | **One rank** — for any given (token, expert) pair, the gate/up/down GEMMs all run on the same GPU. | AllToAll routes tokens to the rank that holds their selected expert. |

**flambeau's current MoE TP is the first mode**:
`crates/models/qwen3-moe/src/tp_layout.rs` declares
`ColParallel{dim=1}`/`RowParallel{dim=2}` for the expert tensors, and
`forward_moe_ffn_decode_tp` follows up with an AllReduce on the
combined per-rank partials. That mode is what introduces the
F32 reduce-order divergence at the bottom of the GDN attention
hand-off, which then forks the layer-1 router top-k.

## What each framework actually ships

### vLLM

The definitive reference is the AMD ROCm blog "The vLLM MoE Playbook"
(2025-11-24) which dissects vLLM's MoE distribution code path.

* **Without `--enable-expert-parallel`** vLLM's MoE layers run in
  **sharded-experts** mode: every GPU owns all routed experts but
  with weights partitioned via
  `vllm/model_executor/layers/fused_moe/config.py::flatten_tp_across_dp`.
  The post-MoE communication is AllReduce. This is the same shape
  flambeau ships.
* **With `--enable-expert-parallel`** vLLM switches to
  **split-experts** mode via `determine_expert_map`: the 256 experts of
  DeepSeek-R1 distribute as 32-experts/GPU across an 8-GPU node, and
  the communication degenerates to AllReduce when DP=1 or AllToAll
  when DP>1. The flag is a no-op when `TP_SIZE × DP_SIZE == 1`.
* The vLLM team's empirical finding: for ultra-sparse models
  (Llama-4-Maverick at 0.78 % activation density) `EP=0` (sharded
  experts) **outperforms** EP=1 by 7-12 % because the AllToAll
  overhead exceeds the memory-bandwidth win of distributing experts.
  For sparse-but-not-ultra-sparse (Qwen3-235B at 6.25 %, DeepSeek-R1
  at 3.13 %) `EP=1` wins, especially at high concurrency where
  KV-cache partitioning under DP+EP unlocks 8× larger batches.
* Crucially, **sharded-experts mode does not give bit-exact results
  across TP sizes** — it has the same F32 reduction-order issue
  flambeau hit. vLLM uses it anyway because raw throughput matters
  more than determinism in their default deployment.

Source: <https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html>

### DeepSpeed-MoE

DeepSpeed's MoE tutorial at
<https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/>
describes the canonical split: "data-parallelism and tensor-slicing
for the **non-expert** parameters and expert-parallelism and
expert-slicing for the **expert** parameters."

The `MoE` API has an `enable_expert_tensor_parallelism` flag
(<https://deepspeed.readthedocs.io/en/latest/moe.html>) that defaults
to **False** — i.e., DeepSpeed-MoE's *default* is split-experts
across the EP group, *with* tensor-parallel applied only to the
non-expert layers. Tensor-slicing the experts themselves is opt-in
and rarely used in inference recipes.

So DeepSpeed-MoE chose split-experts as the primary MoE distribution
strategy, with intra-expert TP available as an optional add-on. The
3D approach (DeepSpeed-TED, arXiv:2303.06318) generalises this to
"data + tensor + expert" hybrid for very large training jobs.

### Megatron-LM / Megatron-DeepSpeed

Megatron's MoE branch follows the same default: experts are
distributed across an `expert_parallel_size`-sized group via
all-to-all, while `tensor_parallel_size` shards only the non-expert
projections. The two groups can overlap, producing the TP × EP
hybrid that vLLM also exposes.

(Megatron's documentation is sparser on the public web; the implementation
lives in `megatron/core/transformer/moe/` and follows the same shape
DeepSpeed-MoE ships.)

### Academic state-of-the-art (2025)

* **Tree-Based Invariant Kernels** (Zhang et al., arXiv:2511.17826,
  Nov 2025) addresses the *exact* problem flambeau hit: "identical
  inputs can yield different outputs when system configurations
  (e.g., tensor parallel (TP) size, batch size) vary, even under
  greedy decoding. This arises from the non-associativity of
  floating-point arithmetic and inconsistent reduction orders across
  GPUs." Their fix is a co-designed matmul + AllReduce primitive
  that uses a unified hierarchical binary-tree reduction order so
  that intra-GPU and inter-GPU sums combine into the same global
  order regardless of how many ranks are participating. They report
  zero probability divergence and bit-exact reproducibility across
  TP sizes after integration into vLLM and FSDP. The paper is
  motivated by RL pipelines where the trainer (TP=1, FSDP) and the
  rollout engine (multi-GPU TP) need to produce bit-identical logits.
* The companion library **BitExact**
  (<https://github.com/aaravkohli1/BitExact>) packages similar
  batch-invariant + TP-invariant kernels for PyTorch.

The TBIK paper confirms, with proof and a concrete kernel design,
that **flambeau's drift is not a bug to be fixed in an afternoon** —
it's a known open problem in the literature, and the published
solution (TBIK) requires re-engineering both the GEMM kernel and
the AllReduce primitive, then aligning their reduction orders.

## Implications for flambeau

The literature gives three robust paths, in increasing engineering
cost and decreasing pragmatism:

1. **Quality-gate, accept the drift** (matches vLLM's default
   behaviour for sharded-experts). Validate that perplexity and
   chat smoke pass within an acceptable delta vs PP reference; ship
   `--mesh-mode tp` for MoE arches with that gate. This is
   essentially what every production-grade vLLM deployment does
   without `--enable-expert-parallel`. **Pragmatic V1 path**, since
   the post-AR-attn drift we measured is small (well within the
   activation noise envelope of Q4_K weights).

2. **Switch to split-experts for MoE** (matches DeepSpeed-MoE's
   default and vLLM's `--enable-expert-parallel` mode). The MoE
   expert tensors get a new layout `ExpertParallel{dim=0}` (each
   rank owns a contiguous slice of the n_experts axis). The router
   runs replicated; each rank dispatches activations to the rank
   that holds its expert via host-bounce all-to-all (PCIe topology
   constraint — same constraint that picked PP-primary in V1).
   Co-fixes #52 (Coder-30B K-quant alignment wall) since per-expert
   slabs are unsharded along inner dim. **Right long-term move**;
   roughly one focused session of work.

3. **TP-invariant kernels** (TBIK-style). Re-engineer the
   AllReduce + per-rank GEMM to use a unified binary-tree reduction
   order. Ships bit-exact across TP sizes but requires kernel-level
   work on both the per-rank MMVQ kernels and the
   `p2p_allreduce_residual_*` family. Heaviest lift; the right
   move only if RL-style determinism becomes a project goal.

For V1 the recommendation in `b5_tp_drift_bisect_findings.md`
holds: option (1) (perplexity quality-gate) ships now, option (2)
(split-experts) lands in V2 alongside the broader MoE refactor for
qwen3moe / Qwen3-Coder-30B. Option (3) is V3+ and only if a
deterministic-inference use case lands on the roadmap.

## Cross-reference

* `certs/parity/b5_tp_drift_bisect_findings.md` — flambeau-specific
  bisect data and per-layer router divergence table.
* `doc/ROADMAP-V1-QWEN36-GFX906.md` — V1 plan; needs an addendum
  noting the MoE-TP quality-gate decision.
* Task #55 — V1 ship-decision pending: quality-gate vs reject vs
  defer.
* Task #52 (TP-7-arch, Coder-30B alignment wall) — same MoE refactor
  resolves both this and B5 if option (2) is chosen.

## Sources cited

* AMD ROCm Blogs — "The vLLM MoE Playbook" (2025-11-24).
  <https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html>
* vLLM docs — "Expert Parallel Deployment".
  <https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/>
* DeepSpeed — "Getting Started with DeepSpeed-MoE for Inferencing".
  <https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/>
* DeepSpeed — `MoE` API reference (`enable_expert_tensor_parallelism`).
  <https://deepspeed.readthedocs.io/en/latest/moe.html>
* Singh et al., DeepSpeed-TED, arXiv:2303.06318.
  <https://arxiv.org/abs/2303.06318>
* Zhang et al., "Deterministic Inference across Tensor Parallel
  Sizes That Eliminates Training-Inference Mismatch",
  arXiv:2511.17826 (Nov 2025).
  <https://arxiv.org/abs/2511.17826>
* HuggingFace transformers — `Qwen3NextGatedDeltaNet` reference (for
  ruling out the on-disk QKV layout misinterpretation hypothesis;
  qwen3next is a separate arch from qwen35moe with explicit
  per-k-group interleaving).
  <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py>
* llama.cpp PR #16095 — "Model: Qwen3 Next" (merged 2025-11-28),
  first native qwen3next support in GGUF.
  <https://github.com/ggml-org/llama.cpp/pull/16095>
