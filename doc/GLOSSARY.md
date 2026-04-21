# Flambeau — Glossary for Beginners

A plain-English guide to the jargon that appears across Flambeau's docs (`README.md`, `CLAUDE.md`, `doc/ARCHITECTURE.md`, `doc/ROADMAP-V1-QWEN36-GFX906.md`, `doc/candle-prior-art.md`) and code comments.

Flambeau is a max-performance inference server for modern large language models running on AMD (HIP) and NVIDIA (CUDA) GPUs, exposed over an OpenAI-compatible HTTP API. Every piece of jargon below shows up somewhere on the path from "load a quantized model file" to "a chat client streams tokens back".

## How to read this

- **New to the project?** Start at §1 (GPU basics), then §2 (LLM basics), then §3 (quantization). Sections §4 onward assume those three.
- **Hit an unfamiliar word while reading another doc?** Jump to the alphabetical index in §9; each term points back to the explanation.
- **Want to understand a specific design decision?** Sections §5 (kernel engineering) and §6 (Flambeau architecture) connect terms to the *why* behind choices in the codebase.

Each entry follows the same shape: **what it is** (plain definition) → **why Flambeau cares** (what problem it solves, or what tradeoff it embodies) → **where you see it** (a file, a kernel name, a dispatch row, or a doc section). Entries are intentionally short — this is a reference, not a textbook. Cross-references like *(see §N)* link to the authoritative entry when a term is used under multiple headings.

No prior knowledge of GPU programming or LLM internals is assumed. Where we simplify, we flag it.

---

## §1. GPU & Hardware Fundamentals

Modern LLM inference is a GPU workload. Understanding the vocabulary for *what a GPU is made of* and *how we talk about its performance* is the prerequisite for reading any kernel-related section of the project.

### Architectures (the specific chips we target)

**gfx906 (AMD Radeon VII / Instinct MI50).** The V1 primary target. A 2018-era "GCN5/Vega20" data-center GPU with 1 TB/s HBM2 memory and 13.4 TFLOPS of F32 compute. It uses **wave64** execution (see below). All correctness certificates and default dispatch tables are tuned against this chip first. Used because several are available in the dev rig and because the silicon ceiling is well-characterized.

**gfx908 / gfx90a / gfx942 (AMD Instinct MI100 / MI210–250 / MI300).** Newer CDNA data-center AMD chips. Portability targets — Flambeau must compile and eventually run here, but they are not V1 test hardware. Adding one of these means adding a dispatch-table row and a cert, not writing new kernels.

**gfx1031 (AMD Radeon RX 6750 XT, "RDNA2 Navi 22").** The **portability canary**. A consumer GPU that runs **wave32** instead of wave64, has no matrix cores, and a differently-structured LDS. If any kernel abstraction silently assumes wave64, gfx1031 is where it breaks. Exercised on real hardware at every Phase-2+ gate.

**gfx1100 (AMD Radeon RX 7900 series, "RDNA3").** Another consumer target: wave32 + WMMA matrix instructions. Second-class but kept compilable.

**sm_80 / sm_86 / sm_89 / sm_90 (NVIDIA Ampere A100 / RTX 3090 / Ada / Hopper).** CUDA compute capability codes. V1 development uses **sm_86 (RTX 3090)**. sm_80 is the stated baseline. Newer caps (sm_89 Ada, sm_90 Hopper) are forward-compatible via `__CUDA_ARCH__` guards for features like TMA and `wgmma`.

### Inside a GPU

**Wavefront (AMD) / Warp (NVIDIA).** A group of threads on a GPU that execute in lock-step. AMD waves are 64 threads (gfx906, CDNA) or 32 threads (RDNA: gfx1031, gfx1100). NVIDIA warps are always 32 threads. Two words for the same concept; "wave" and "warp" are used interchangeably in the kernel code.

**Wave64 vs wave32.** AMD's two wave sizes. GCN/CDNA chips (gfx906, gfx908, gfx90a, gfx942) are wave64; RDNA chips (gfx1031, gfx1100) are wave32. This one difference changes how lane-wise intrinsics (DPP, cross-lane shuffles) must be written, which is why `kernels-hip/arch_primitives/<arch>.cuh` exists (see §6).

**SIMD (Single Instruction, Multiple Data).** The execution unit inside a compute unit. Each SIMD can hold multiple active waves and time-slice between them; how many it can hold is **occupancy**.

**Occupancy / waves per SIMD (or "waves per EU").** How many waves a SIMD can run concurrently. Higher occupancy hides memory latency because while one wave is waiting for memory another can do arithmetic. Occupancy is limited by register pressure (VGPR) and shared memory use. On gfx906, crossing the 2→10 waves/SIMD threshold is more valuable than small unrolling wins.

**VGPR (Vector General-Purpose Register).** A vector register on an AMD GPU. Each SIMD has a fixed pool; a kernel that uses *n* VGPRs per thread limits how many waves can occupy the SIMD. VGPR count is **first-order** for performance on gfx906 — it is the primary thing `CLAUDE.md` asks you to profile before proposing any perf lever. Inspectable via `flambeau inspect-hsaco <path>`.

**LDS (Local Data Share).** AMD's name for GPU shared memory: a small, fast, per-compute-unit scratchpad. Used for cooperative algorithms where multiple threads in a wave or workgroup need to share data (e.g. the 4-warp LDS-tiled MMQ). NVIDIA calls the same thing "shared memory" or "smem".

**HBM (High-Bandwidth Memory).** The main GPU DRAM. gfx906 has 1 TB/s HBM2. Decode on MoE models is typically **HBM-bound** — you cannot generate tokens faster than you can read the weights from HBM, so the perf target is stated as an HBM-utilization percentage (see §7).

**L2 cache.** A cache between HBM and the compute units. "L2 prefetch" (see §5) refers to issuing explicit loads to warm the L2 before a kernel needs them, hiding latency.

### Communication between GPU lanes

**DPP (Data-Parallel Primitives).** AMD cross-lane instructions that let one lane of a wave see another lane's data without going through shared memory. Used to implement fast tree reductions inside a wave (sum, max, etc.). Syntax and lane masks differ between gfx906 (wave64) and gfx1031 (wave32) — always go through `arch_primitives/<arch>.cuh`.

**Cross-lane reduce / half-warp reduce / quarter-warp reduce.** A sum (or max) across 16, 32, or 64 lanes of a wave, built on DPP. Candle's multi-row MMVQ kernels use half-warp and quarter-warp DPP reductions to produce 2 or 4 output rows per wave in one pass. Key primitive: `gfx906_half_warp_reduce_sum_dpp`.

**dp4a.** A 4-byte dot-product-with-accumulate instruction (`__builtin_amdgcn_sdot4` on AMD). Takes four int8 × int8 products and adds them to an int32 accumulator in one instruction. Wins when the int→float and activation-quantization overhead is amortized over many output rows (MMVQ on K-quants), loses on decode attention where there's only one query row. **dp4a is not a universal speedup.**

### Programming models and libraries

**HIP (Heterogeneous-compute Interface for Portability).** AMD's CUDA-like GPU programming model. Kernel source is `.cu` files compiled with `hipcc`. Flambeau's HIP kernels live in `crates/kernels-hip/src/*.cu`.

**CUDA.** NVIDIA's GPU programming model. In Flambeau, lives in `crates/kernels-cuda/` (V2+ — V1 is HIP only). The two share algorithmic-core headers via `crates/kernels-shared/`.

**rocBLAS.** AMD's BLAS library (matrix multiplication, etc.). Used as a correctness fallback and for dense F32 paths. Its Rust handle has a subtle drop-order quirk (`Mutex<Option<RocBlas>>` must drop *before* the stream) captured in `CLAUDE.md` as the "rocBLAS handle drop-order" rule.

**RCCL (ROCm Collective Communications Library).** AMD's multi-GPU collective primitives: `all_reduce`, `all_gather`, `all_to_all`, `broadcast`. Used to move data directly between GPUs over PCIe P2P without a host bounce. V1.2 lands the bindings; TP and EP rely on it (see §2 and §6).

**NCCL.** NVIDIA's equivalent of RCCL. Relevant for future CUDA multi-GPU work; V1 does not ship it.

### Kernel artefacts and profiling

**hsaco.** A compiled HIP kernel object (analogous to a `.o`). `flambeau inspect-hsaco <path>` prints the symbol table and VGPR budget of every kernel in the binary. Use it to verify a build change did not balloon register usage past the occupancy threshold.

**PTX.** NVIDIA's parallel-thread-execution intermediate language — CUDA's equivalent of looking at what was actually emitted. `flambeau inspect-ptx <path>` is the CUDA counterpart of `inspect-hsaco`.

**rocprofv3.** AMD's GPU profiler. Emits **PMC** (Performance Monitoring Counter) data per kernel launch. The authoritative way to attribute wall-clock time to specific kernels on AMD. `CLAUDE.md` requires a rocprofv3 pass before any perf claim.

**Nsight.** NVIDIA's equivalent profiler. Same role as rocprofv3 for CUDA kernels.

**PMC (Performance Monitoring Counter).** A hardware counter that records one specific aspect of GPU execution over a kernel's run. Flambeau stores a PMC snapshot alongside every correctness cert — the "PMC" part of the cert is what lets you tell, later, *why* a kernel is fast.

**MemBusy.** PMC counter: fraction of wall-clock time the HBM controller was servicing a request. On gfx906, decode target is ≥ 70% MemBusy; if we're below, the kernel is leaving bandwidth on the table.

**MemStall.** PMC counter: fraction of wall-clock time compute was stalled waiting on memory. High MemStall with low MemBusy is **latency-bound** (request pipeline is the bottleneck, not bandwidth).

**VALUBusy.** PMC counter: fraction of wall-clock time the vector ALU was doing useful work. High VALUBusy = **compute-bound**.

**Bandwidth-bound vs compute-bound vs latency-bound.** Three canonical bottleneck classes:
- *Bandwidth-bound* — memory throughput is the ceiling. Sign: MemBusy ≥ 60%, MemStall ≈ 1%. Fix is structural (tile differently, prefetch, reduce bytes read).
- *Compute-bound* — arithmetic throughput is the ceiling. Sign: VALUBusy ≥ 60%, MemBusy low. Fix is more arithmetic intensity or better instructions (dp4a, WMMA).
- *Latency-bound* — neither memory nor compute is saturated, but waves are stalled in memory-pipeline dependencies. Fix: more occupancy, or hide the latency with prefetch.

Classifying the bottleneck correctly is what makes a perf lever land; `CLAUDE.md` requires the PMC snapshot to match the proposed fix.

### Asynchrony

**Stream.** A GPU execution queue. Kernels launched on the same stream run in order; kernels on different streams can overlap. In Flambeau, every kernel launch takes a `&Stream` argument — this is the **"explicit async"** rule in `CLAUDE.md`.

**hipDeviceSynchronize / cudaDeviceSynchronize.** A host-side blocking call that waits for all GPU work to finish. Flambeau permits this only at session boundaries and inside the `sweep` harness; *never* inside a model forward pass. Hidden syncs have historically masked whole classes of performance bugs.

---

## §2. LLM & Model Architecture

You do not need to train a transformer to work on Flambeau, but you do need to know what an attention head is, why decode is different from prefill, and what an MoE router does. This section covers the minimum.

### Basics

**Token.** The unit a language model sees. A token is usually a subword (a few characters, sometimes a whole short word). Tokenizers (see §8) convert a text string to a sequence of token ids and back.

**Prompt.** The sequence of input tokens given to the model before it starts generating.

**Autoregressive generation.** The model generates one token at a time; each new token depends on every previous one. Total cost per-sequence therefore grows — but the **KV cache** (below) keeps it linear per token after the prompt is processed.

**Prefill vs decode.** Two completely different workloads against the same model:
- *Prefill* — Run the entire prompt through the model in one batched pass. All prompt tokens are present at once. Compute-heavy; perf is reported as "pp512" (prefill 512 tokens).
- *Decode* — Generate output tokens one by one. Each decode step sees a single new token as input plus the full KV cache. Bandwidth-heavy; perf is reported as "tg64" (target-generate 64 tokens).

Kernel selection differs between prefill and decode (see MMVQ vs MMQ in §5).

**Embedding.** The lookup table that turns a token id into a vector of size *hidden_dim*. The first layer of a transformer.

**LM head.** The final projection back from *hidden_dim* to *vocab_size*, producing one **logit** per token in the vocabulary. Often the same weights as the embedding, transposed (tied embeddings).

**Logits.** The raw, unnormalized scores the model emits for the next token. Token-parity checks (see §6) compare logits between Flambeau and llama.cpp using cosine similarity ≥ 0.9999.

### Attention

**Attention.** The mechanism that lets each token look at every earlier token. For token *i*, compute query/key/value vectors (Q, K, V), score Q against all K, softmax, and take a weighted sum of V. The quadratic-in-sequence-length core of a transformer.

**Causal mask.** During training (and prefill), we want token *i* to only see tokens ≤ *i*. The mask sets "future" attention scores to −∞ so they become 0 after softmax.

**Softmax.** Exp-normalize a vector so it sums to 1. Used to turn raw attention scores into weights. See `masked_softmax_scale_fused` — Flambeau's fused kernel that does scale + causal mask + softmax in one launch.

**MHA / MQA / GQA.** Three variants of multi-head attention:
- *MHA* (multi-head) — one K and one V head per Q head.
- *MQA* (multi-query) — one K/V head total, all Q heads share it.
- **GQA (grouped-query)** — compromise: *num_kv_heads* << *num_q_heads* and Q heads within a group share K/V. Qwen3.6 uses "GQA 32/4" (32 query heads, 4 KV heads).

**head_dim.** Dimension of a single attention head's Q/K/V vector (e.g. 128 for Qwen3.6). Const-generic in Flambeau so the compiler specializes kernels per head_dim.

**RoPE (Rotary Position Embedding).** A way to encode *which* position a token is at, by rotating pairs of entries in Q and K by an angle proportional to position. Reversibly injects position into attention with no extra learned parameters. The `rope_theta` value comes from the GGUF metadata.

**KV cache.** The tensor of all past K and V vectors from every layer of the model, kept across decode steps so each new token does not have to re-compute attention over the whole history. Size grows with sequence length × num_layers × num_kv_heads × head_dim × dtype_bytes. For Qwen3.6, KV cache quickly becomes larger than the weights — which is why Flambeau makes its storage layout a first-class, type-level choice (see §6).

### Normalization and activation

**RMSNorm (Root Mean Square Normalization).** A simple, cheap alternative to LayerNorm: divide the hidden state by its root-mean-square and multiply by a learned scale. No mean-subtraction, no learned bias. Standard in modern LLMs. Kernel: `rmsnorm_q8_fused` (fuses norm + Q8 quant in one launch).

**SiLU (Sigmoid Linear Unit).** The activation `x * sigmoid(x)`. Smooth, monotonic. Also called "Swish".

**SwiGLU.** The feed-forward activation `silu(gate(x)) * up(x)`. Two linear projections, one of them gated by SiLU. Standard in Llama/Qwen/Mistral/Gemma. Flambeau's `SwiGLU` kernel fuses the gate and up-projection outputs into a single launch.

**Fused activation.** General pattern: combine consecutive small ops (norm + quantize, gate + up, softmax + scale + mask) into a single kernel to avoid round-tripping through HBM between them. Only a win when the unfused path was bandwidth-bound — fusing a compute-bound kernel usually loses because register pressure rises.

### Feedforward / experts

**Dense MLP / FFN (feed-forward network).** The per-token two-layer MLP that follows each attention block. In SwiGLU models this is `down(silu(gate(x)) * up(x))`.

**MoE (Mixture of Experts).** Instead of one dense MLP, have *N* "expert" MLPs and, for each token, route it to the top-*k* experts. Only those experts compute for that token. Flambeau's V1 target Qwen3.6-35B-A3B has 128 experts, top-8 per token — effective active parameters drop from 30B to ~3B per token, which is why "A3B" is in the name.

**Router.** The small linear + softmax + top-k that picks which experts each token goes to. Emits expert ids and weights. Kernel: `forward_router_decode` (dense GEMV → top-k).

**Top-k / topk.** Select the *k* highest values from a vector. Done with a bitonic sort on-chip in Flambeau's `TopK` kernel.

**Shared expert.** A *non-gated* expert that runs for every token, in parallel with the routed experts. Qwen3.6 has one. It's a small dense MLP that catches "always-useful" behavior no router would need to select.

**IndexedMoE.** Flambeau's MMVQ/MMQ kernel variants that take an explicit `(token_id, expert_id)` index so many experts can share a single GEMM launch. See `indexed_moe_mmvq_q4_k_*`.

**MoE combine.** After each expert computes its per-token output, combine them (weighted sum by the router's top-k weights) and optionally add the residual. Kernel: `moe_combine_f16`.

### Hybrid architectures

**Full-attention layer.** A "standard" transformer layer with causal multi-head attention + MLP or MoE.

**GDN (Gated Delta Net).** An *alternative* to attention used in some Qwen3-Next layers. Replaces the quadratic-in-sequence attention with a recurrent, linear-in-sequence gated update over a fixed-size hidden state (see `gdn_state_step_f32_s128` — state size 128). Trades expressiveness for constant-per-token compute.

**Hybrid (full-attention + GDN).** The Qwen3.6-35B-A3B / `qwen35moe` arch — some layers are full attention, others are GDN. Flambeau handles this by branching in the per-layer forward between `forward_full_attn_decode` and `forward_gdn_decode`.

### Target model families (V1 scope)

- **Mistral / Devstral** — dense, standard transformer.
- **Gemma-4** — dense with gated attention + sliding-window context.
- **Qwen3.5 dense** — dense SwiGLU transformer.
- **Qwen3.6 MoE (Qwen3.6-31B-A3B)** — V1 primary target; `qwen35moe` arch, hybrid GDN + full-attention with MoE + shared expert, Q4_K_S GGUF (~20 GB). 128 experts, top-8 routing, GQA 32/4, head_dim 128, partial NeoX RoPE, per-token sigmoid gate.
- **Qwen3-Coder MoE** — code-specialized MoE, same family.
- **Qwen3-Next** — GDN + MoE hybrid.

---

## §3. Quantization & GGUF

"Quantization" is how a 30-billion-parameter model fits in 18 GB of VRAM. It replaces 32-bit floats with low-bit integers plus per-group scales. This section is the densest for newcomers — but once the pattern "packed ints + a scale per block" clicks, every Q-name makes sense.

### Floating-point passthrough

**F32.** IEEE 32-bit float. The "reference" dtype; CPU dequant produces F32 that Flambeau compares kernel output against.

**F16.** IEEE 16-bit float ("half"). Default activation and KV-cache dtype.

**BF16.** Brain-float 16-bit: same exponent range as F32 but 7-bit mantissa. Training-friendly; inferred through as-is when GGUF files store weights in BF16.

### GGUF

**GGUF (GGML Unified Format).** The file format Flambeau loads. Originated with llama.cpp. One file contains: tensor data (possibly quantized), per-tensor metadata (shape, dtype), model-level metadata (architecture, rope_theta, etc.), the tokenizer vocabulary, and the chat template. Flambeau parses GGUF v3 in `crates/quant/src/gguf.rs`. Inspect a file with `flambeau inspect-gguf`.

**Tensor metadata.** Per-tensor: name, shape, dtype, byte offset. Flambeau uses this to map GGUF tensor names to model weight slots.

**Chat template.** A `minijinja` template inside the GGUF metadata that turns `{role, content}` messages into the exact prompt string the model expects. Applied once, at the server layer, so there is no per-model hard-coded prompting in kernels.

### Legacy block quants (Q*_0 / Q*_1)

These are the older, simpler GGUF quant layouts. Each stores a small block of values (usually 32) with one scale (and optionally one offset). Slower to dequant than K-quants but ubiquitous.

**Q4_0.** 32 weights per block. 4 bits per weight + one F16 scale. 5 bits of effective storage per weight.

**Q4_1.** 32 weights per block. 4 bits per weight + F16 scale + F16 min-value (offset). 6 bits per weight.

**Q5_0 / Q5_1.** Same as Q4_0 / Q4_1 but 5 bits per weight.

**Q8_0.** 32 weights per block. 8 bits per weight + one F16 scale. 9 bits per weight. Near-lossless; used heavily for activation quantization.

### K-quants (Q*_K)

K-quants use a **super-block** of 256 weights. A single F32 (or F16) block-scale parameterizes 16 sub-blocks; each sub-block has its own quantized mini-scale. Better bit-packing per weight than legacy quants because the scales themselves are compressed.

**Q2_K.** ~2.6 bits/weight. Smallest. Used for very aggressive compression.

**Q3_K.** ~3.4 bits/weight.

**Q4_K.** ~4.5 bits/weight. Qwen3.6's dominant weight dtype in `Q4_K_M` mix.

**Q5_K.** ~5.5 bits/weight.

**Q6_K.** ~6.6 bits/weight. Used for "sensitive" tensors (output head, attention projections) in the `Q4_K_M` mix.

**Q8_K.** Roughly equivalent to F16 quality, used for internal activations during dot products with K-quants.

**Q4_K_M (the "M" mix).** A llama.cpp-style mixed quantization: most weights are Q4_K, sensitive tensors (e.g. `attn_output`, `ffn_down`, `output.weight`) get Q6_K. The "M" stands for "Medium" — there are also `_S` (Small, all Q4_K) and `_L` (Large) mixes. Qwen3.6 ships as `Q4_K_M` in GGUF.

### Activation quantization

**Q8_1.** Not a GGUF weight format — an **activation** quantization. Packs 32 F32 activations into 32 int8s plus an F32 scale and F32 sum-of-values, in a specific llama.cpp-compatible layout. MMVQ and MMQ take Q8_1 activations (weights are Q*_K, activations get on-the-fly quantized to Q8_1 before the dot product). `quantize_f16_q8_1` is the fused kernel that does this.

### Turbo-quant KV cache

**Turbo-quant (Q4 / Q5 KV cache layouts).** A packed, lower-bit KV-cache scheme ported from `llamacpp-turbo`. Higher compression than Q8 KV cache, measurable quality cost. Gated behind a **quality cert** (not just correctness) — see §6. V1 defers turbo-quant to V2; V1 ships only F16 and Q8 KV.

### The CPU dequant reference

**CPU dequant reference.** For every supported quant dtype, `crates/quant/src/reference.rs` contains a pure-CPU implementation that dequantizes a block to F32. This is the oracle every kernel is checked against: the `sweep` harness runs the GPU kernel, runs the CPU reference, and compares. "Bit-for-bit vs llama.cpp" is the bar for the CPU reference itself.

---

## §4. Kernel Engineering Vocabulary

Terms you will see in roadmap commit descriptions and in kernel source comments. Most of this is not specific to Flambeau — it is the broad vocabulary of GPU kernel engineering.

### Basics

**Kernel.** A GPU function. One `.cu` source typically exports several kernels. Flambeau's HIP kernels are in `crates/kernels-hip/src/*.cu`.

**Launch.** The act of submitting a kernel to a stream with a given grid and block size. Has non-trivial per-launch overhead (a few microseconds); fusing consecutive small kernels avoids this overhead.

**Block / threadblock.** A group of threads (typically 64, 128, 256) that can share LDS and synchronize. One or more waves per block.

**Grid.** The collection of all blocks for a single kernel launch.

### Matmul vocabulary

**GEMV (General Matrix-Vector multiply).** *M × K* weight matrix times a *K*-vector → *M*-vector. The decode-path shape.

**GEMM (General Matrix-Matrix multiply).** *M × K* weight matrix times a *K × N* activation matrix → *M × N*. The prefill-path shape.

**MMVQ (Matrix × Matrix-Vector, Quantized).** The decode-path kernel. Input: quantized weights (Q4_K, Q6_K, etc.) + Q8_1 activation vector (M=1). Output: F16 or F32 vector. On gfx906, implemented as one workgroup per output-row-group using DPP cross-lane reductions.

**MMQ (Matrix × Matrix, Quantized).** The prefill-path kernel. Same weight dtypes as MMVQ but the activation is a *batch of tokens* (M ≥ 128). Uses 4-warp LDS-tiled cooperative structure (below).

### Matmul implementation patterns

**Tile / tiling.** A fixed-size rectangular chunk of a matrix. A "4-warp LDS tile" is a tile whose load is cooperatively performed by 4 warps using shared LDS.

**4-warp LDS-tiled (MMQ).** The canonical prefill kernel shape, ported from `llamacpp-turbo`. 4 warps (256 threads) share one LDS tile of activations; each warp computes a row-block of output; kernel reuses each activation 4× after paying one LDS-load. Dominant perf lever for prefill on gfx906.

**stream-K (fix-up).** A load-balancing scheme for GEMMs whose M/N dimensions don't divide evenly by the tile size. Instead of letting one "tail" workgroup do extra work, stream-K splits it across all workgroups and fixes up the final partial results.

**L2 prefetch.** Explicit loads that warm the L2 cache before the main compute loop needs the data. Hides a fraction of the HBM→L2 latency.

**Multi-row MMVQ (nw1 / r2 / r4).** MMVQ kernel variants that produce 2 or 4 output rows per wave, using DPP half-warp or quarter-warp reductions. "nw1" = one wave per workgroup; "r2" = 2 rows per wave; "r4" = 4 rows per wave. Candle P29 showed r2 is best for Q4_K / Q5_K, r4 is best for Q6_K. The variant suffix appears in impl_ids like `qmatmul_q4_K_mmvq_nw1_r2_gfx906`.

### Reductions

**Reduce.** Collapse a vector down to a single value (sum, max, etc.) in parallel.

**Tree reduce.** The textbook parallel reduction: log(n) rounds, each round halves active threads. Uses LDS for inter-wave reduces.

**DPP reduce.** A fully intra-wave tree reduce using DPP instructions (no LDS traffic). Fastest on AMD for reductions that fit within a wave.

**Half-warp reduce / quarter-warp reduce.** Reduces over 32 or 16 lanes within a wave, leaving multiple independent partial sums per wave — the enabling primitive for multi-row MMVQ.

### Naming convention

**Variant naming.** `{op}_{dtype}_{backend}_{shape_tag}_{variant}`. Example: `qmatmul_q4_K_mmvq_nw1_r2_gfx906` = "QMatMul on Q4_K weights, MMVQ kernel, one-wave-per-workgroup, 2 rows per wave, gfx906 arch". Grep-able, no "`_v2f_tile32_repacked`" suffix drift (a cautionary tale from candle).

### Memory discipline

**alloc vs alloc_zeros.** Allocate (uninitialized) vs allocate and fill with zeros. `alloc_zeros` costs a separate GPU write of every byte before the real kernel runs. `CLAUDE.md` forbids `alloc_zeros` in hot paths — use `alloc` and have the kernel write every output lane. Saves a measurable wall-clock fraction on large model forward passes.

### Fusion

**Fused kernel.** Multiple consecutive ops combined into a single launch, eliminating the intermediate HBM round-trip. Wins when the unfused path was bandwidth-bound; risks losing when the fused kernel blows the VGPR budget and halves occupancy. Always measure; *fusion is not automatic*.

**Concrete fusions in Flambeau.** `rmsnorm_q8_fused` (norm + Q8_1 quantize), `swiglu_f32` (silu(gate) × up), `masked_softmax_scale_fused` (scale + mask + softmax), `gdn_state_step_f32_s128` (fused GDN recurrence), `quantize_f16_q8_1` (F16 → Q8_1 in one launch — landed in V1.7.3-g).

### Attention kernel families

**flash-attention-v2-style prefill.** Tile over Q × K^T and V in a way that keeps the softmax numerator and denominator in registers, never materializing the full attention matrix. The dominant prefill kernel on both CUDA and HIP.

**GQA decode `gqa_decode_mv_fast_d{128,256}`.** Gemma-4 / Qwen-family fused F16 KV decode attention. One kernel, F32 accumulator, specialized per head_dim (128 or 256).

**Q8-KV GQA decode.** Decomposed decode with a Q8 KV cache: quantize-K, compute Q·K_q8, softmax, multiply V_q8, in a chain of ~9 small kernels. Wins over fused F16 when the attention is not already compute-light.

### Half-measures

**#[cfg(unverified)].** A kernel that compiles but is unreachable from the dispatcher. This is the *only* place a broken-but-kept kernel is allowed to live. Alternatives (env-flag resurrection, silent deletion) are forbidden by `CLAUDE.md`.

---

## §5. Flambeau Architecture

The project's own conceptual vocabulary. This is the vocabulary that makes the *rules* in `CLAUDE.md` readable.

### The type-driven stack

**Op (trait).** A typed *contract* for one operation (MatMul, QMatMul, Attention, RMSNorm, RoPE, Softmax, SiLU, etc.). Has typed inputs, a `contract()` method returning the output shape, output dtype, and the minimum tolerance the op promises. Defined in `crates/core/src/op.rs`. **One contract per op, many impls** — never a parallel "fast path" op trait.

**KernelImpl (trait).** A concrete kernel that implements an Op for a specific Device. Has a `const ID` (the impl_id), an `applies()` predicate, a `launch()` method, and a static `cert()`. Registered at build time.

**impl_id.** The unique string id of a kernel. Example: `qmatmul_q4_K_mmvq_nw1_r2_gfx906`. Ties a dispatch-table row to a cert file; if they disagree, build fails.

**OpContract.** The typed promise a dispatcher inspects: `output_shape`, `output_dtype`, `min_tolerance`. Any impl selected for an Op must satisfy its contract.

### Dispatch and certification

**Dispatch table (`dispatch/<backend>/<arch>.toml`).** A TOML file that maps `(backend, dtype_tuple, shape_predicate) → impl_id` per op. The single source of truth for kernel selection. No `if env::var(...)` anywhere in the code — variant selection is a *reviewed artifact*.

Example from `dispatch/hip/gfx906.toml`:

```toml
[[qmatmul]]
dtype = "Q4_K"
shape = { m = ">=128", k = "any", n = "any" }
impl  = "mul_mat_q4_K_turbo_dense_4warp_x8"
cert  = "certs/hip/mul_mat_q4_K_turbo_dense_4warp_x8.json"
```

The dispatcher picks the *most-specific-matching* row; two rows matching the same shape is a build-time error.

**Cert / Correctness certificate.** A JSON file at `certs/<backend>/<arch>/<impl_id>.json` that records: the kernel id, the reference it was tested against, per-shape tolerance bounds, and a PMC snapshot (VGPR count, occupancy, MemBusy, VALUBusy) from a representative shape. Every impl_id referenced in any dispatch table *must* have a matching cert — `cert-check` enforces this at build time.

**Sweep (harness).** The correctness grid that produces and refreshes certs. `cargo run -p bench -- sweep --arch gfx906 --op qmatmul` runs the GPU kernel across a shape grid, compares against the CPU dequant reference (see §3), and emits a fresh cert. No perf claim is valid before its kernel's sweep is green.

**Quality cert.** An *additional* cert for lossy layouts (turbo-quant KV cache). Measures delta-perplexity on a fixed eval (wikitext-2) and chat smoke across 3 multi-turn conversations. Stored under `certs/quality/<model>/<layout>.json`. A turbo-quant layout without a quality cert never appears in a dispatch row.

### KV cache (typed, not flagged)

**KvCache<Layout>.** A generic KV cache where the layout is a **type parameter**, not a runtime enum. Compiler statically rejects mixing incompatible layouts.

**F16Contig / F16Transposed.** Baseline layouts. Contig = row-major; Transposed = K is transposed for better GEMV perf during decode.

**Q8Contig / Q8Transposed.** Candle's Q8 KV cache shape. Per-group scale + zero, `QK_K=32`. ~2× memory saving vs F16, near-F16 quality. Ships in V1.

**TurboQ4Contig / TurboQ5Contig.** Packed Q4/Q5 KV layouts from llamacpp-turbo. More aggressive compression, quality-cert-gated. V2.

### Multi-GPU

**Mesh<N>.** The multi-GPU abstraction, where `N` is the number of GPUs. `Mesh<1>` is a degenerate single-GPU instance; `Mesh<2>` and `Mesh<4>` are the V1 physical targets. **Every downstream component is mesh-generic from the first commit** — single-GPU code paths never branch on `N == 1`.

**Collectives.** Multi-GPU primitives implemented as ops: `AllReduce`, `AllGather`, `AllToAll`, `Broadcast`. Backends: RCCL (first-class) + host-bounce CPU reference (for the cert harness). **In V1 these are architectural trait surface but not on the decode hot path** — see PP below.

**PP (Pipeline Parallelism) — V1 primary topology.** Shard *layers* across ranks: each rank owns roughly `num_layers / N` **contiguous** layers with its MoE experts held locally (no cross-rank routing). At each stage boundary, one F16 hidden-state vector (~4 KB at `hidden=2048`) is handed off per token per hop. Hand-off is **host-bounced** via a pinned-memory ping-pong (DtoH → HtoD, 4 MiB chunks, 2 buffers per rank, ~6.75 GB/s measured on MI50 PCIe 3.0 x16 — plenty for 4 KB hand-offs). Candle's `hip_backend/cluster.rs` is the pattern. V1 perf gate: **PP scaling ≥ 3.0× on Mesh<4>** vs Mesh<1>.

**PP is V1's choice because the rig is PCIe-only.** No xGMI, no NVLink peer links. TP>2 and wide-EP with router all-to-all are bandwidth-bound on PCIe (40–50% comm overhead measured in the vLLM / llama.cpp / DeepEP literature; llama.cpp's `split_mode=row` is 3–8× slower than `layer` on non-P2P topologies and has long-context miscompiles). PP hand-offs are cheap (~10–50 µs per hop) and fit the topology. RCCL bindings stay for future use.

**TP (Tensor Parallelism).** Shard the feature dimension of weight tensors across ranks; each rank computes a partial output; `all_reduce` gathers. **Architectural trait surface in V1, not the V1 hot path** (PP is). TP>2 revisited in V2+ once an xGMI / NVLink rig is available.

**EP (Expert Parallelism).** Shard MoE experts across ranks; each rank holds `num_experts / N` experts; tokens routed via `all_to_all`. **Scoped to V2+** — wide-EP router all-to-all on PCIe is dominated by comm overhead. V1 keeps all of a stage's experts local to that stage via PP.

### Kernel factoring

**Crate layout (condensed).**
- `core/` — Device-independent traits (Tensor, DType, Shape, Op, DispatchTable, Registry).
- `quant/` — GGUF, block-quant layouts, CPU dequant reference.
- `kernels-shared/` — Backend-neutral algorithmic-core `.cuh` (MMQ tile structure, block-quant unpack, softmax math). **Zero intrinsics.**
- `kernels-hip/` — HIP `.cu` + `build.rs` that calls `hipcc`.
- `kernels-cuda/` — CUDA `.cu` + `build.rs` (V2).
- `backend-hip/` — HipDevice, HipStream, kernel-impl registrations.
- `ops/` — High-level fused ops (Attention, MoE, FFN, GatedDeltaNet) composed from core traits. Mesh-generic.
- `models/<family>/` — Model composition only: weight-name map, block list, `forward_one_token`, `forward_prefill`. **Zero new kernels.**
- `runtime/` — KvCache, Session, Mesh<N>, scheduler.
- `bench/` — Sweep + matrix + cert-diff harness.
- `autotune/` — T-track warmup-tuner.
- `server/` — OpenAI-compatible HTTP.
- `mcp-server/` — M-track dev MCP server.
- `cli/` — The `flambeau` binary.

**arch_primitives.** The per-arch header directory `kernels-hip/arch_primitives/{gfx906,gfx908,gfx90a,gfx942,gfx1031,gfx1100}.cuh`. Each header exposes **the same signatures** (e.g. `half_warp_reduce_sum_dpp`) with arch-specific bodies. Wave-width constants live here — never hard-code `WAVE_SIZE = 64` anywhere else.

### Development side-tracks

**T-track (warmup-tuner).** A runtime microbench that picks, at session start, between *already certed* variants to specialize the dispatch table for *this rig's* exact shapes. Writes a gitignored `dispatch/<backend>/<arch>.local.toml` that layers over the committed table. Never runtime codegen — only runtime selection. Implemented in `crates/autotune`. CLI: `flambeau tune`.

**M-track (MCP server).** A dev-only MCP server exposing sweep/matrix/profile/A-B/inspect/cert-diff tools to a Claude session. Lives in `crates/mcp-server`. Never part of production. CLI: `flambeau mcp --port 9090`.

### Non-negotiable rules (paraphrase)

These appear verbatim in `CLAUDE.md`; a paraphrased list here so the glossary is self-contained:

1. No env-flag variant selection; dispatch lives in `dispatch/<backend>/<arch>.toml`.
2. No kernel ships without a cert.
3. One Op trait, many impls. Never a parallel "fast-path" trait.
4. Models import only from `ops/`, never from `backend-*` or `kernels-*`.
5. `kernels-shared/` is backend-neutral; intrinsics live in `kernels-hip/` or `kernels-cuda/`.
6. KV cache layout is a type, not a runtime flag.
7. Explicit async: every kernel launch takes `&Stream`.
8. No `alloc_zeros` in hot paths.
9. Variant naming is `{op}_{dtype}_{backend}_{shape_tag}_{variant}`.
10. Un-certed kernels live behind `#[cfg(unverified)]`, never behind env flags.

---

## §6. Measurement, Benchmarking, CLI

Commands and metrics that appear in logs and PR descriptions.

### CLI subcommands

`cargo run -p cli -- <subcommand>` or (once built) `flambeau <subcommand>`.

- **`infer`** — Single-prompt inference. `flambeau infer --model <id> --prompt "Hello" --devices hip:0,1,2,3`.
- **`serve`** — OpenAI-compatible HTTP server. `flambeau serve --model <id> --port 8080`.
- **`inspect-gguf <path>`** — Tensor listing, dtype audit for a GGUF file.
- **`inspect-hsaco <path>`** — Kernel symbols + VGPR budgets for a HIP binary.
- **`inspect-ptx <path>`** — Kernel symbols + register usage for a CUDA binary.
- **`tune`** — T-track warmup-tuner session. `--dry-run` reports proposed overrides without writing.
- **`mcp`** — M-track MCP server.

### Bench subcommands

- **`bench sweep`** — Correctness grid for an op on an arch. Produces / refreshes certs.
- **`bench matrix`** — End-to-end perf matrix. One row per `(model, dtype, kv_layout, prompt_len, tg_len)`; reports pp / tg tokens/s; regressions flagged against the last snapshot.
- **`bench quality`** — Delta-perplexity + chat smoke for turbo-quant layouts. Updates `certs/quality/`.

### Perf metrics

**tokens/s (tok/s).** Token throughput. Two standard configurations:
- **pp512** — Prefill 512 tokens (first-token latency at prompt length 512).
- **tg64** — Target-generate 64 tokens (decode throughput after prefill).

Both are wall-clock end-to-end. Kernel-level attribution is done separately with rocprofv3 / Nsight.

**V1 success targets (1× MI50, Qwen3.6).**
- Decode tg64 ≥ 60 tok/s on 1× MI50.
- Prefill pp512 ≥ 600 tok/s on 1× MI50.
- Decode HBM utilization (MemBusy) ≥ 70%.
- Decode tg64 on 4× MI50 with **PP** ≥ 3.0× the 1× number. (Decode roofline on 4× PP: ~5 GB active weights per stage / 1 TB/s HBM ≈ 1.6 ms/token/stage × 4 stages pipelined ≈ ~150 tok/s ceiling; the 60 tok/s gate leaves room for PP bubbles + per-hop PCIe latency.)

### Silicon ceilings (gfx906)

- **HBM:** 1 TB/s.
- **F32 compute:** 13.4 TFLOPS.
- **Int8 compute:** 26.5 TOPS.

The **perf bar is silicon, not llama.cpp**. If Flambeau hits the HBM ceiling and llama.cpp is faster, the bug is in llama.cpp's reporting, not in Flambeau; we file a diagnosis, we do not ship a regression to "catch up".

### llama.cpp's role

**Correctness oracle.** Logit cosine similarity ≥ 0.9999 vs llama.cpp on the same GGUF is a hard gate for any model.

**Lower-bound reference.** llama.cpp's tokens/s is the floor, not the ceiling. Flambeau targets gfx906 silicon ceilings, which are often above what llama.cpp extracts.

### The pre-claim checklist

Before any "faster" or "correct" claim, `CLAUDE.md` requires:

- [ ] Rebuilt the crate that owns the changed code (`cargo clean -p kernels-hip` if kernels touched — `.hsaco` caching has bitten us).
- [ ] `bench sweep --impl <id>` green.
- [ ] PMC snapshot matches the expected bottleneck (bandwidth / compute / latency).
- [ ] Wall-clock end-to-end at pp ≥ 512, tg ≥ 64 on ≥ 1 target model per backend.
- [ ] Diff vs previous cert snapshot documented in the PR body.
- [ ] Null results filed honestly (no "slight regression" rewording).

---

## §7. Server & OpenAI API

The thin outer layer that turns Flambeau's inference engine into something chat clients can talk to.

**OpenAI-compatible API.** Flambeau speaks the OpenAI REST dialect, so any client that works against the OpenAI API or against llama.cpp's `server` binary works against `flambeau serve` with just a host/port change.

**Endpoints (V1 surface).**
- `GET /v1/models` — Lists loaded models (GGUF path, arch, quant, KV layout, max context).
- `POST /v1/chat/completions` — Chat-template-aware completions. Supports `stream: true` via **SSE**.
- `POST /v1/completions` — Raw-prompt completions (legacy clients).
- `GET /health` — Liveness.

**Deferred to V2.**
- `GET /metrics` — Prometheus counters (tok/s, queue depth, KV utilization).
- `/v1/embeddings`, function calling, tool use, structured outputs, logprobs — shipped per concrete client need.

**SSE (Server-Sent Events).** The HTTP streaming protocol OpenAI uses. Server pushes `data: {...}` JSON chunks as tokens are produced; client reads them as a stream.

**Chat template.** A `minijinja` template, extracted from `tokenizer.chat_template` in the GGUF metadata, that turns `{role, content}` messages into the model's expected prompt string. Applied in `server/src/chat_template.rs` — **no per-model hardcoded prompts**.

**Sampler.** The CPU-side last step: given the model's final logits, pick the next token id. V1 supports **temperature** and **top-p** only. More samplers (top-k, min-p, repetition penalty) are V2.

**Compatibility bar.** `bench/server_smoke.sh` asserts round-trips from `curl` (both non-streaming and SSE) and at least one real OpenAI-compat client (Aider or Continue).

---

## §8. Rust / Build

A small section for readers new to Rust conventions that appear in the codebase.

**Crate.** A Rust package. Flambeau has ~15 interdependent crates (see §6 crate layout).

**Workspace.** A top-level Cargo project that owns multiple crates with shared dependencies. The root `Cargo.toml` lists all members.

**`cargo build --release`.** Build with optimizations on. Required for any perf measurement. Debug builds are misleading for anything kernel-adjacent.

**`cargo clean -p <crate>`.** Force-rebuild a specific crate. Required when editing `.cu` kernels — the `.hsaco` build cache has bitten us before.

**`cargo test -p <crate>`.** Run tests inside one crate.

---

## §9. Alphabetical Index

A quick lookup of every term in this document. Terms are listed with the section they're defined in; skim §1 through §8 after locating a term.

| Term | Where |
|------|-------|
| 4-warp LDS-tiled (MMQ) | §4 Matmul patterns |
| alloc / alloc_zeros | §4 Memory discipline |
| arch_primitives | §5 Kernel factoring |
| Attention (prefill/decode) | §2 |
| Autoregressive generation | §2 |
| Bandwidth-bound | §1 PMC |
| BF16 | §3 |
| Block / threadblock | §4 Basics |
| cargo build --release | §8 |
| Causal mask | §2 Attention |
| Cert (correctness certificate) | §5 Dispatch and certification |
| cfg(unverified) | §4 Half-measures |
| Chat template | §7 |
| Collectives (AllReduce / AllGather / AllToAll / Broadcast) | §5 Multi-GPU |
| Compute-bound | §1 PMC |
| CPU dequant reference | §3 |
| Crate / workspace | §8 |
| Cross-lane reduce (half-warp / quarter-warp) | §1 Communication |
| CUDA | §1 Programming models |
| Decode | §2 Basics |
| Dense MLP / FFN | §2 Feedforward |
| Dispatch table | §5 Dispatch and certification |
| DPP | §1 Communication |
| dp4a | §1 Communication |
| Embedding | §2 Basics |
| EP (Expert Parallelism) | §5 Multi-GPU |
| F16 | §3 |
| F32 | §3 |
| F16Contig / F16Transposed | §5 KV cache |
| flash-attention-v2-style prefill | §4 Attention kernels |
| Fused kernel | §4 Fusion |
| GDN (Gated Delta Net) | §2 Hybrid |
| gfx906 / gfx908 / gfx90a / gfx942 / gfx1031 / gfx1100 | §1 Architectures |
| GEMM / GEMV | §4 Matmul vocabulary |
| GGUF | §3 |
| GQA decode (mv_fast_d128 / d256) | §4 Attention kernels |
| GQA (grouped-query attention) | §2 Attention |
| Grid | §4 Basics |
| HBM | §1 Inside a GPU |
| head_dim | §2 Attention |
| HIP | §1 Programming models |
| hipDeviceSynchronize | §1 Asynchrony |
| hsaco | §1 Kernel artefacts |
| Hybrid (full-attn + GDN) | §2 Hybrid |
| impl_id | §5 The type-driven stack |
| IndexedMoE | §2 Feedforward |
| Kernel | §4 Basics |
| KernelImpl (trait) | §5 The type-driven stack |
| KV cache | §2 Attention |
| KvCache<Layout> | §5 KV cache |
| L2 prefetch | §4 Matmul patterns |
| Latency-bound | §1 PMC |
| Launch | §4 Basics |
| LDS | §1 Inside a GPU |
| LM head | §2 Basics |
| llama.cpp (as oracle) | §6 |
| Logits | §2 Basics |
| Mesh<N> | §5 Multi-GPU |
| MHA / MQA / GQA | §2 Attention |
| MemBusy / MemStall | §1 PMC |
| MMQ | §4 Matmul vocabulary |
| MMVQ | §4 Matmul vocabulary |
| MoE (Mixture of Experts) | §2 Feedforward |
| MoE combine | §2 Feedforward |
| Multi-row MMVQ (nw1 / r2 / r4) | §4 Matmul patterns |
| M-track (MCP server) | §5 Development side-tracks |
| NCCL | §1 Programming models |
| Nsight | §1 Kernel artefacts |
| Occupancy | §1 Inside a GPU |
| Op (trait) | §5 The type-driven stack |
| OpContract | §5 The type-driven stack |
| PMC (Performance Monitoring Counter) | §1 Kernel artefacts |
| pp512 / tg64 | §6 Perf metrics |
| Prefill | §2 Basics |
| Prompt | §2 Basics |
| PTX | §1 Kernel artefacts |
| PP (Pipeline Parallelism) | §5 Multi-GPU |
| Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 | §3 Legacy block quants |
| Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K | §3 K-quants |
| Q4_K_M | §3 K-quants |
| Q8_1 | §3 Activation quantization |
| Quality cert | §5 Dispatch and certification |
| RCCL | §1 Programming models |
| Reduce / tree reduce / DPP reduce | §4 Reductions |
| RMSNorm | §2 Normalization |
| rocBLAS | §1 Programming models |
| rocprofv3 | §1 Kernel artefacts |
| RoPE | §2 Attention |
| Router | §2 Feedforward |
| Sampler | §7 |
| Shared expert | §2 Feedforward |
| SIMD | §1 Inside a GPU |
| SiLU | §2 Normalization |
| sm_80 / sm_86 / sm_89 / sm_90 | §1 Architectures |
| Softmax | §2 Attention |
| SSE (Server-Sent Events) | §7 |
| Stream | §1 Asynchrony |
| stream-K | §4 Matmul patterns |
| Sweep (harness) | §5 Dispatch and certification |
| SwiGLU | §2 Normalization |
| Tile | §4 Matmul patterns |
| Token | §2 Basics |
| Top-k / topk | §2 Feedforward |
| TP (Tensor Parallelism) | §5 Multi-GPU |
| T-track (warmup-tuner) | §5 Development side-tracks |
| Turbo-quant KV | §3 |
| Tensor metadata | §3 GGUF |
| Variant naming | §4 Naming convention |
| VALUBusy | §1 PMC |
| VGPR | §1 Inside a GPU |
| Wave64 / Wave32 | §1 Inside a GPU |
| Wavefront / Warp | §1 Inside a GPU |

---

## §10. V1 Timeline — What Each Version Means

Flambeau's work is organized into V1.0 → V1.8. Roadmap commits and memory entries constantly reference these version numbers; the summary here lets you place a commit like "V1.7.3-e forward_one_token" in the overall story.

The authoritative version of this timeline is `doc/ROADMAP-V1-QWEN36-GFX906.md` — this is a condensed recap.

- **V1.0 — Workspace + mesh skeleton.** Cargo workspace, all crates compile, `Mesh<N>` + collective traits with a CPU-reference impl, empty dispatch table. End: `flambeau --help` prints subcommands; `cargo test -p runtime` exercises `Mesh<1/2/4>` against reference collectives.
- **V1.1 — GGUF loader + CPU dequant reference.** `crates/quant` parses GGUF v3, CPU dequant for every Qwen3.6 dtype. `inspect-gguf` round-trips bit-for-bit vs llama.cpp. Sharded-on-load weight loader.
- **V1.2 — HIP device + RCCL collectives.** HipDevice, HipStream, alloc, H↔D copy. Real RCCL bindings. All four collectives certed on 2× and 4× MI50.
- **V1.3 — First-class MMVQ (Q4_K / Q5_K / Q6_K / Q8_0).** Multi-row DPP reduce from candle P29. Four MMVQ certs green with PMC snapshots.
- **V1.4 — First-class MMQ prefill (Q4_K / Q6_K / Q8_0, 4-warp LDS-tiled).** Ported from llamacpp-turbo. Stream-K fixup, L2 prefetch. Prefill pp512 ≥ 600 tok/s on 1× MI50.
- **V1.5 — MoE kernels.** Bitonic top-8, IndexedMoE MMVQ r2, IndexedMoE MMQ (4-warp turbo-dense pattern adapted for MoE), MoE combine. Isolated MoE-layer forward matches CPU reference.
- **V1.6 — Fused decode path + attention + KV cache families.** RMSNorm+Q8_1 fused, SwiGLU, RoPE, masked-softmax, flash-attn-v2 prefill, fused F16 and Q8 KV decode. `KvCache<F16Contig>` + `KvCache<Q8Contig>` both instantiable; Q8 KV quality cert (delta-ppl ≤ 0.5%).
- **V1.7 — Qwen3.6 model + PP composition + token parity.** Mesh-generic ops, `crates/models/qwen3_moe`, per-layer sharded loader, token-parity cert (cosine ≥ 0.9999 vs llama.cpp on Mesh<1/2/4>), PP scaling ≥ 3.0× gate on Mesh<4>. (TP/EP surface exists but is not on the V1 decode hot path — see §5 PP entry for why.)
- **V1.7.2 / V1.7.3 (sub-increments).** The hybrid-surface build-out inside V1.7: weight upload (V1.7.3-a), per-layer decode forwards for full-attention / GDN / MoE / shared expert / router (b–d), end-to-end `forward_one_token` (e), `forward_prefill` (f), two on-device fusion kernels that replaced host round-trips (g), and two bug fixes (h, i). See the memory entries under `project_v1_7_*` for detail.
- **V1.8 — Tokenizer, sampler, chat template, OpenAI server.** `tokenizers` crate, `minijinja` chat template, CPU temperature+top-p sampler, `axum` server, `bench/server_smoke.sh`. End: tag `v0.1.0-qwen36-gfx906`.

Two side-tracks run alongside from V1.2 onward:

- **T-track (T1–T5, warmup-tuner)** — shape recorder → variant enumerator → micro-bench runner → session integration → auto-promote rule. Output: `dispatch/<backend>/<arch>.local.toml` machine-local overrides, always layered over a committed table.
- **M-track (M1–M5, MCP server)** — `flambeau_sweep`, `flambeau_matrix`, `flambeau_profile`, `flambeau_dispatch_ab`, `flambeau_inspect`, `flambeau_cert_diff`, `flambeau_tune_dry`. Output: every MCP finding round-trips into a committable artifact (cert, matrix snapshot, dispatch row).

---

## §11. Where to Go Next

After this glossary, the project docs in recommended reading order:

1. **`README.md`** — Two-screen entry point. Scope, repo layout, first-run commands.
2. **`CLAUDE.md`** — Architectural rules + measurement discipline. The load-bearing rules file for anyone touching code. You now have the vocabulary to read it end-to-end.
3. **`doc/ARCHITECTURE.md`** — Framework design: crates, traits, dispatch, KV cache families, phasing. Start with §"Architectural principles" and §"Crate layout".
4. **`doc/ROADMAP-V1-QWEN36-GFX906.md`** — Current execution plan, V1.0 → V1.8. Each step names its entry crate and its port target.
5. **`doc/candle-prior-art.md`** — Where in `/artefact/candle/`, `/artefact/llamacpp-turbo/`, and `/artefact/llama.cpp/` to find each kernel we port. Essential when you start porting.

Further reading, outside this repo:

- **`/artefact/candle/`** — The source framework. Read for kernel prior art (especially `candle-hip-kernels/src/`), *not* for architectural patterns.
- **`/artefact/llama.cpp/`** — Correctness oracle and baseline reference. `ggml-cuda/mmq*.cu` is the 4-warp LDS-tiled reference that V1.4 ports.
- **`/artefact/llamacpp-turbo/`** — Additional reference for turbo kernel shape and turbo-quant KV layouts.

When you encounter an unfamiliar term in any of the above, come back here. This glossary is the index.
