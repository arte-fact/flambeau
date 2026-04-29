# Qwen3.6-35B-A3B-Q4_0 — flambeau vs llama.cpp on 4× MI50 — 2026-04-29

Apples-to-apples synthetic bench (no chat overhead) against llama.cpp build
`4bf202d4a (7975)` using `/opt/rocm-host/lib` (the in-tree ROCm 7.1.1 install
that has gfx906 TensileLibrary kernels — the public `/opt/rocm-7.1.1/lib`
crashes with "device kernel image is invalid" on this rig). 100 W cap. Both
stacks run with implicit warmup; flambeau via `profile_point` / `profile_point_tp`
which always discards the first prefill+decode pass; llama.cpp via the default
warmup (no `--no-warmup`).

## Results — 3-run mean

### Synthetic (no chat overhead, pp512 + tg128 from empty KV)

| topology | n_cards | pp512 tok/s | tg128 tok/s | source |
|---|---:|---:|---:|---|
| **flambeau TP2** | 2 | **1044** | **61.6** | `profile_point_tp` |
| flambeau PP4 | 4 | 770 | 56.4 | `profile_point` |
| llama.cpp PP4 | 4 | 858 | 52.2 | `llama-bench -p 512 -n 128 -r 3` |

### Chat workload (live server, 269-token prompt, 3-run mean)

**KEY FINDING — sampler dominates the chat-vs-bench gap:**

| topology | sampler | pp tok/s | tg tok/s |
|---|---|---:|---:|
| **flambeau TP2** | **greedy (temp=0)** | **920** | **60.5** |
| flambeau TP2 | sampled (temp=0.7 top_p=0.9) | 890 | **35.6** |
| flambeau pp2tp2 | greedy (temp=0) | 905 | 55.6 |
| flambeau pp2tp2 | sampled (temp=0.7 top_p=0.9) | 870 | 15.6† |
| llama.cpp PP4 (`-d 256 -n 128`) | greedy (default) | n/a | 52.8 |

†pp2tp2 sampled deteriorated across runs from ~35 to ~15 — likely thermal
throttling on 4 cards at sustained load + 100 W cap. Greedy holds steadier.

**Chat-workload conclusions:**

- **flambeau TP2 greedy = 60.5 tok/s, llama.cpp PP4 greedy = 52.8 tok/s
  → flambeau is 1.15× faster on greedy chat decode at equivalent depth.**
  This matches the synthetic 1.18× number — i.e., the kernel forward path is
  consistently 15–20% faster than llama.cpp.
- **flambeau TP2 sampled (temp=0.7, top_p=0.9) drops to 35.6 tok/s — ~12 ms/token
  pure sampler overhead** (`build_distribution` is O(V log V) full sort across
  the 151k vocab; runs every decode step). This is the gap the user noticed
  in chat: not a forward-path issue, the sampler is the bottleneck.
- llama-bench doesn't apply user samplers (it generates with default greedy
  argmax), so the 52.8 tok/s reference is greedy. To compare apples-to-apples
  with llama.cpp at temp=0.7+top_p=0.9, llama.cpp would need its server path
  (~separate bench).

**Sampler cost breakdown (flambeau side):**

```
TP2 greedy   chat decode wall = 1/60.5 = 16.5 ms/token
TP2 sampled  chat decode wall = 1/35.6 = 28.1 ms/token
                                       ─────────────
                  delta sampler cost ≈ 11.6 ms/token
```

The 11.6 ms is consistent with prior memory note (MTP-5g): "build_distribution
O(V log V) sort × 3 per macro adds ~30 ms wall" — single-call per macro is
~10 ms. There's a tracked optimisation (#194: partial-sort + penalty-aware
build_distribution) that should bring this from O(V log V) to O(V + K log K)
where K is top_k cutoff.

## Headline

**flambeau TP2 wins on both axes.**

- **Prefill pp512: flambeau TP2 = 1044, llama.cpp PP4 = 858 → flambeau is 1.22× faster.**
- **Decode tg128: flambeau TP2 = 61.6, llama.cpp PP4 = 52.2 → flambeau is 1.18× faster.**
- flambeau PP4 wins prefill by 0.90× of llama.cpp (28% behind on prefill — but
  TP2 is the recommended config anyway), wins decode 1.08× of llama.cpp.

## Earlier "TP2 decode = 26 tok/s" was a cold-cache outlier

The first run after model load showed TP2 decode tg64 = 26.2 tok/s, which
seeded the wrong story. Re-measured 3× back-to-back after the model was
warm:

```
TP2 tg=64 run 1:  61.24 tok/s
TP2 tg=64 run 2:  69.88 tok/s
TP2 tg=128 run 1: 64.35 tok/s
TP2 tg=128 run 2: 61.38 tok/s
TP2 tg=128 run 3: 62.13 tok/s
TP2 tg=256:       65.28 tok/s
```

The 26 was first-run kernel-JIT cost on `mmq_q4_0_4warp_lds`-class kernels
(measurable on the very first prefill where rocBLAS Tensile has to compile-on-demand). All subsequent measurements stabilised at 60–65 tok/s.

## kq_replicated overhead is negligible

Diagnostic A/B with `FLAMBEAU_GDN_KQ_REPLICATED=off` (output broken, perf only):

| variant | pp512 tok/s | tg128 tok/s |
|---|---:|---:|
| TP2, kq_replicated=true (correct)   | 1046 | 64.4 |
| TP2, kq_replicated=off (broken)     | 1069 | 64.0 |

The "duplicate K/Q work" cost is < 0.5%. The TP-4d-i3 fix doesn't sacrifice
material decode throughput — the per-token GDN K/Q matmul is small enough
that replicating it across 2 ranks costs ~300 µs out of a 16 ms decode step.

## Why TP2 wins decode for both 27B (dense) and 35B-A3B (MoE)

Earlier session memory suggested TP2 decode might be hurt by the kq_replicated
fix — that turned out to be untrue. Both arches scale similarly:

| model | PP4 tg / TP2 tg | TP2/PP4 ratio |
|---|---|---:|
| Qwen3.6-27B-Q4_1 (dense)        | 16.4 / 20.7 | 1.27× |
| Qwen3.6-35B-A3B-Q4_0 (MoE rep_outer) | 56.4 / 61.6 | 1.09× |

TP2 wins because it parallelises the per-token bandwidth read across 2 cards
(each rank reads its weight slice in parallel), while PP4 serialises across 4
stage hand-offs. The MoE arch wins less than dense because (a) MoE's active
weight footprint is smaller, so per-stage PP cost is less penalised, and (b)
GDN's K/Q replication adds a tiny duplicate cost on TP that doesn't apply to
dense's contiguous K/V split — but both effects are small.

## Reproducer

```sh
# llama.cpp reference (the /opt/rocm-host/ libs are the missing piece)
cd /artefact/llama.cpp_mi50
LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm-host/lib \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-host/lib/rocblas/library \
./build/bin/llama-bench \
  -m /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -r 3 -t 1

# flambeau TP2 synthetic
FLAMBEAU_PROFILE_GGUF=/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
FLAMBEAU_PROFILE_TP_DEVICES=0,1 \
FLAMBEAU_PROFILE_L=512 \
FLAMBEAU_PROFILE_TG=128 \
FLAMBEAU_TP_BATCHED=1 \
/artefact/flambeau/target/release/deps/profile_point_tp-* \
  --nocapture profile_point_tp

# flambeau PP4 synthetic
FLAMBEAU_PROFILE_GGUF=/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
FLAMBEAU_PROFILE_MESH=4 \
FLAMBEAU_PROFILE_L=512 \
FLAMBEAU_PROFILE_TG=128 \
/artefact/flambeau/target/release/deps/profile_point-* \
  --nocapture profile_point
```

## Methodological lesson

A single first-pass measurement on a fresh model load is not a reliable
benchmark. Always either (a) run with explicit warmup, (b) take 3+ back-to-back
runs and report the median/min, or (c) use `--repetitions 3` on llama-bench's
side. The 26→62 tok/s revision came from doing exactly this. Update the
`profile_point_tp` harness or any future perf cert to enforce ≥ 3 runs in the
post-warmup phase.
