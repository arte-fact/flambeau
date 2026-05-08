# MTP side-by-side — llama.cpp PR #22673 vs flambeau MTP-5e on 4× MI50 PCIe

**Date:** 2026-05-07
**llama.cpp:** am17an `mtp-clean` @ `5d5f1b4` (PR #22673, draft)
**flambeau:** `main` @ `d7c3cc6`
**Rig:** 4× MI50 (gfx906) / PCIe 3.0 x16 / ROCm 7.1.1
**Model:** Qwen3.6-27B (dense, hybrid GDN+full-attn), `general.architecture = qwen35`

## Headline

**am17an MTP at K=1 is +13% net-positive on the same PCIe 4×MI50 rig where flambeau MTP-5e went −18%.** The MTP-5e claim that "PCIe-PP is structurally hostile to spec-decode" is **partially refuted**: the topology is hostile at K≥2 (matches our diagnosis), but K=1 wins on the right verify primitive. flambeau lost K=1 because its L=2 paired-verify primitive scaled 1.92× wall over L=1 — am17an's separate-MTP-context design exploits the base model's batched ubatch=K+1 decode path, which on this rig is essentially free at K=1.

## Matrix

| Stack | Config | tg t/s (median of 3) | ms/tok | Accept | vs baseline |
|---|---|---:|---:|---:|---:|
| llama.cpp #22673 | none (baseline) | 20.60 | 48.55 | — | 1.00× |
| llama.cpp #22673 | MTP K=1 | **23.26** | **43.00** | 93.3% | **1.13×** ✅ |
| llama.cpp #22673 | MTP K=2 | 20.34 | 49.17 | 84.1% | 0.99× ≈ |
| llama.cpp #22673 | MTP K=3 | 17.77 | 56.26 | 65.0% | 0.86× ❌ |
| flambeau (MTP-5e) | none (baseline) | 19.87 | 50.33 | — | 1.00× |
| flambeau (MTP-5e) | MTP K=1 paired-L=2 | 16.86 | 59.33 | 87.5% | 0.85× ❌ |

Both stacks: same 4× MI50 / PCIe 3.0 x16, `--parallel 1`, greedy temp=0, 60-token decode, 5-token prompt, `-sm layer -ts 1,1,1,1` (llama.cpp) ≈ Mesh<4> contiguous-layer PP (flambeau).

**Quant disclosure:** llama.cpp uses Q4_K_M (`froggeric/Qwen3.6-27B-MTP-GGUF`, MTP `nextn` tensors baked into base GGUF at `blk.64.nextn.*`); flambeau uses Q4_0 base + `Qwen3.6-27B-mtp.gguf` Q8_0 sidecar. Q4_K_M is ~7% larger than Q4_0; baseline-vs-baseline confirms within-noise (48.55 vs 50.33 ms/tok ≈ 3.5% delta, expected from quant + decode-path differences).

## Acceptance comparison vs am17an's RTX 3090 numbers

| K | am17an RTX 3090 | this rig (MI50) | accept ≈ same? |
|---|---:|---:|---|
| 2 | ~83% | 84.1% | yes (+1.1pp) |
| 3 | ~72% | 65.0% | lower (−7pp) |

Acceptance rate is **content-dependent, not topology-dependent** — confirms MTP-INV-5 ceiling finding. Our 5-token prompt ("The capital of France is") drives a ~60-token informative-prose continuation; am17an's reported numbers are likely averaged over more code-like prompts (where MTP head's training distribution skews higher).

## Why am17an wins K=1 on the same rig flambeau lost

Per-macro cost decomposition:

**flambeau MTP-5e (paired-L=2 verify on legacy single-token decode kernel)**
- Base verify L=2: 95.9 ms/macro (1.92× the L=1 wall of 50 ms — *not* 1.5× as projected)
- Per macro tokens accepted at 87.5%: 1.875
- Effective ms/tok: 95.9 / 1.875 ≈ 56.5 → measured 59.3 ✗ NET-NEG

**llama.cpp #22673 (separate MTP context, base verify via ubatch=K+1)**
- MTP draft (separate context, server stat `dur(g)` for K=1): ~6.8 ms/macro (408 ms / 60 macros)
- Base verify ubatch=2: ~36 ms/macro (the dominant cost; 30 macros × 36 ≈ 1080 of 2580 ms eval)
- Per macro tokens accepted at 93.3%: 1.93
- Effective ms/tok: ~43 ms ✓ NET-POS

**Two design deltas explain the 33% wall gap at K=1:**

1. **Verify shape.** flambeau's `forward_one_token_pp_logits` at L=2 is *not* the same kernel as ubatch=1 with twice the work — it's two sequential single-token forwards with the same per-stage PCIe peer-copy/AR sync overhead per token. am17an's verify goes through the base's *normal* batched decode path at ubatch=K+1, which on a weight-HBM-bound regime is nearly free for small K (the activation HBM grows by K×4 KB, weight HBM is unchanged). flambeau memory `feedback_qmatmul_small_m_no_amortize` + `feedback_mmvq_batched_activation_hbm` describe this exact regime in our own kernels.
2. **Draft cost.** flambeau snapshots GDN state per-rank in-process every macro (~2 ms) and reuses the trunk's compute path. am17an pays a fixed ~7 ms/macro for the separate-context draft but doesn't touch the trunk's GDN state. At K=1 these are similar; at K≥2 the separate-context overhead amortizes better.

## Why both stacks lose at K≥2

llama.cpp K=2 / K=3 on this rig go flat / negative, while am17an reports K=2/K=3 wins on RTX 3090 (1.4× / 2.0×). This **does** validate the MTP-5e PCIe diagnosis at K≥2:

- K=1 → ubatch=2 verify, +4 KB hidden per stage hop → trivial
- K=2 → ubatch=3 verify, +8 KB → still trivial in transfer, but verify wall + PCIe sync per layer × ranks scales meaningfully
- K=3 → ubatch=4 verify, +12 KB transfer + ~33% more compute per layer + same PCIe sync count → loses to baseline savings

On a no-PCIe-peer rig (single device, NVLink, xGMI), the ubatch=K+1 decode wall is dominated by weight HBM and stays roughly constant for small K — explaining am17an's RTX 3090 2.0× at K=3.

## Lever for flambeau

**Re-run MTP-5e through the P2.9b batched-decode primitive** (`forward_decode_batched_pp`). MTP-5e was 2026-04-29; P2.9b shipped 2026-04-30 — MTP-5e's L=2 paired primitive predates batched-decode. Wiring MTP verify through ubatch=2 batched-decode should reproduce the +13% K=1 win we measured here for am17an. Filed as task; no code in this cert.

Out-of-scope here: K≥2 will likely still lose on PCIe-PP per the diagnosis above, regardless of which verify primitive we use.

## Reproducer

llama.cpp side:
```
cd /artefact/llama.cpp-mtp  # am17an mtp-clean @ 5d5f1b4
PATH=/opt/rocm-host/bin:$PATH cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_HIP_COMPILER=/opt/rocm-host/bin/amdclang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm-host
cmake --build build -j 8

# Per-config (K ∈ {none, 1, 2, 3}):
HIP_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=/opt/rocm-host/lib \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-host/lib/rocblas/library \
./build/bin/llama-server \
  -m /artefact/models/Qwen3.6-27B-Q4_K_M-mtp.gguf \
  -ngl 99 -sm layer -ts 1,1,1,1 --parallel 1 --port 8088 --no-warmup -c 4096 \
  [--spec-type mtp --spec-draft-n-max K]

# Then 3× POST /completion {prompt:"The capital of France is", n_predict:60, temperature:0, seed:N, cache_prompt:false}
# Median tg/s + accept rate per server log line "draft acceptance rate = …".
```

flambeau side: `mtp_5e_perf_ab.md` reproducer, unchanged.

## What this cert does NOT establish

- **am17an's PR is draft, not merged.** Numbers may shift before merge.
- **Sample size is small** (3 runs/config, single prompt). Acceptance variance on a different prompt can be ±10pp per MTP-INV-5.
- **No quant-equalised comparison.** A flambeau-quant Q4_K_M base + same-MTP head would let us compare wall directly without the 7% bpw delta.
- **No K=1 measurement on llama.cpp via flambeau-style paired-L=2.** The two impls verify differently; the comparison is impl-vs-impl, not just primitive-vs-primitive.

## Pointer

- llama.cpp PR: https://github.com/ggml-org/llama.cpp/pull/22673
- am17an branch: `mtp-clean` @ `5d5f1b4` in fork `am17an/llama.cpp`
- llama.cpp clone: `/artefact/llama.cpp-mtp` (separate from the user's `/artefact/llama.cpp` master clone)
- GGUF: `/artefact/models/Qwen3.6-27B-Q4_K_M-mtp.gguf` (17 GB, from `froggeric/Qwen3.6-27B-MTP-GGUF`)
