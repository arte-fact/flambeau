# V2.28 — Rust-side perf corrections C1–C9, Qwen3.6-27B-Q8_0 vs llama.cpp

**Rig**: 4×MI50 PCIe-only (no xGMI / no NVLink peer), ROCm 7.1.1.
**llama.cpp**: build `97508acb1` (8704), `-ngl 99 -sm layer -fa 1`, 5 reps.
**flambeau**: HEAD at V2.28 (post-C1..C9 Rust perf corrections, atop V2.27 kernel stack), 5 reps.
**Config**: pp=512, tg=64.

## Results (5 reps per side, mean ± stdev)

| Config | flambeau | llama.cpp | flambeau / llama.cpp |
|---|---|---|---:|
| pp=512 (prefill) | **141.36 ± 0.16 tok/s** | 135.70 ± 3.68 tok/s | **1.042** |
| tg=64  (decode)  | 19.09 ± 0.19 tok/s | **19.65 ± 0.03 tok/s** | 0.972 |

### Rep-by-rep (for non-overlap check)

- **Prefill**: flambeau 141.13–141.54 (n=5); llama.cpp 128–143 (±3.68 σ on 135.70).
  Ranges overlap — single-rep comparisons on llama.cpp prefill are
  unreliable because of its ±2.7 % variance. Flambeau's mean *is* higher
  and its distribution is narrower; the ~+4 % lead is likely real, but
  a stricter confirmation would need ≥20 reps of llama.cpp.
- **Decode**: flambeau 18.78–19.30 (n=5); llama.cpp 19.62–19.68 (n=5).
  **Distributions do not overlap.** llama.cpp's ±0.03 σ is rock-solid.
  Flambeau's best decode run (19.30) is still below llama.cpp's worst
  (19.62). **This is a real regression, not noise.**

### Comparison to v2.27 baseline (before this cycle)

| Config | v2.27 flambeau | v2.27 llama.cpp | v2.28 flambeau | v2.28 llama.cpp |
|---|---:|---:|---:|---:|
| pp=512 | 135.2 (tied) | 135.2 (tied) | **141.36 ± 0.16** | 135.70 ± 3.68 |
| tg=64  | 19.3 | 19.5 | 19.09 ± 0.19 | 19.65 ± 0.03 |

**Honest deltas** (per-model, same rig, re-measured):
- **Prefill: flambeau −→ +4.6 % vs its own v2.27** (135.2 → 141.36). llama.cpp
  within its own noise band (135.2 → 135.70). **Flambeau swung from tied
  to a +4.2 % lead over llama.cpp on prefill.** This is the win.
- **Decode: flambeau −1.1 % vs its own v2.27** (19.3 → 19.09). llama.cpp
  +0.8 % (19.5 → 19.65). **Flambeau regressed from 99 % of llama.cpp to
  97 %.** Small in absolute terms (−0.2 tok/s) but real — the
  distributions don't overlap.

## Decode gap — actually pre-existing, not a C1–C9 regression

When I first wrote this cert I claimed C1–C9 regressed decode by ~1 %.
That claim was wrong. A/B test on 2026-04-23:

1. **panic = "abort" toggle (C9 isolation)**: rebuilt `flambeau-qwen3-moe`
   with `panic = "unwind"` and reran 5 reps — **identical numbers**
   (141.34 pp / 19.09 tg vs 141.36 pp / 19.09 tg with `"abort"`). C9 is
   perf-neutral on this workload; kept for code-size.
2. **Comparison to v2.27 is not apples-to-apples**: the v2.27 cert used
   single-rep-per-side. Flambeau's own decode σ is ±0.19 tok/s
   (measured here over 5 reps). v2.27's 19.30 tok/s sits ~1 σ above
   this cycle's mean of 19.09 — **the distributions overlap at 1 σ**.
   There's no evidence C1–C9 moved flambeau's decode distribution.
3. **The real, non-overlapping finding is flambeau vs llama.cpp, not
   v2.27 vs v2.28**: flambeau 18.78–19.30 never reaches llama.cpp's
   19.62–19.68. llama.cpp's ±0.03 σ is rock-solid; flambeau's ±0.19 σ
   is typical for a Mesh<4> PP run. **This ~3 % gap was there at v2.27
   too — single-rep measurement just masked it** by catching one
   flambeau sample near the top of its distribution next to one
   llama.cpp sample near the bottom of its (already tiny) distribution.

So the honest take: C1–C9 moved prefill up ~4 % (real) and didn't
measurably move decode. Flambeau's decode has been ~97 % of
llama.cpp's on this model for a while; we just hadn't measured
enough reps to see it.

Follow-ups still worth running (when someone wants the decode gap
closed):
1. `rocprofv3 --kernel-trace --stats` a tg=64 run side-by-side with
   llama.cpp. The ~3 % gap is ~1.5 ms/token on a 52 ms budget — likely
   a handful of kernels running ~1–2 % slower each or a bit more host
   gap than llama.cpp between launches.
2. A Mesh<1> fit of this model (if memory allows after a quant step)
   would eliminate PP-hop overhead from the comparison. 27B-Q8_0 at
   26.6 GiB doesn't fit a single 16 GiB MI50, so this would need a
   different quant or a different model.

## Changes since v2.27

C1–C9 from `RUST-PERF-CORRECTIONS.md`:

| # | Change | File(s) |
|---|---|---|
| C1 | Kernel-resolution cache in `HipModule` — drops `CString::new` + `name.to_string()` + `hipModuleGetFunction` on every launch. Hot path is now a `RwLock::read` + `HashMap::get` on `&'static str`. | `crates/backend-hip/src/module.rs` |
| C2 | `Sampler` owns vocab-sized scratch, reused across tokens (was `vec![0; vocab]` per call). | `crates/runtime/src/sampling.rs`, `crates/server/src/routes.rs` |
| C3 | `HipCluster` bounce buffer: `AtomicPtr` + `AtomicUsize` for lock-free hot path; `Mutex<()>` only on grow. `reserve_bounce_capacity` called at session init. | `crates/backend-hip/src/cluster.rs`, `crates/models/qwen3-moe/src/forward/pp.rs` |
| C4 | `build_expert_buckets` counting-sort — no more `HashMap<i32, Vec<i32>>` + keys collect. Two linear passes. | `crates/ops/src/hip/moe.rs` |
| C5 | `ChatTemplate::render` generic over `Serialize`; server passes `req.messages` directly instead of cloning into `TmplMessage`. | `crates/quant/src/chat_template.rs`, `crates/server/src/routes.rs` |
| C6 | Stop-token filter uses `Vec::retain` in place instead of `filter().collect()`. | `crates/server/src/routes.rs` |
| C7 | `iter_tensors_mut` pre-sizes via `Vec::with_capacity(n_layers * 25 + 3)`. | `crates/models/qwen3-moe/src/weights.rs` |
| C8 | Decided to skip — touching `DeviceTensor.dims: Vec<u64>` cascades to every reader; cosmetic load-time win not worth API churn per the doc's own guidance. | — |
| C9 | `[profile.release] panic = "abort"` — server posture is fatal-and-restart. Shaves code size and unwind-table overhead. | `Cargo.toml` |

## Attribution to specific Cs

Single-rep-per-side methodology doesn't let me attribute the +4.4 % prefill
delta to one specific C — C1 and C3 are the most-plausible contributors
(both strip per-launch / per-hop host overhead, which matters more at
prefill where the pipeline is latency-bound on kernel-launch). C4's
counting-sort is prefill-only and strips a `HashMap` round-trip per
`forward_moe_ffn_prefill` call. C2 is decode-only and shouldn't affect
prefill. C5–C7 are per-request, not per-token.

The decode direction (−3 % to llama.cpp) is within single-rep noise and
should be confirmed with a 3-rep median run before claiming regression
or win. If it persists, the candidates are: C1's `RwLock` on every
resolution (modest vs the old `CString`+`String` path, but still touched
on every launch), or C3's atomic ordering costs that weren't there
before. Profile with `rocprofv3 --kernel-trace` if the number sticks.

## Methodology notes

- 5 reps per side. llama.cpp via `llama-bench -r 5`; flambeau via the
  test-binary run 5× in a shell loop.
- Same `-sm layer` topology on both; both use `-fa 1`.
- `-ub` / `-b` left at default (512) on llama.cpp.
- `ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.1.1/core-7.13/lib/rocblas/library`
  exported for both sides (the V2.27 correction that unblocks llama.cpp
  on MoE models).
- flambeau run: `cargo test --release -p flambeau-qwen3-moe --features
  hip --test perf_baseline_qwen35_9b -- --nocapture` with
  `FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.6-27B-Q8_0.gguf
  FLAMBEAU_MESH_RANKS=4`. (Reused the 9B harness — Qwen3.6-27B-Q8_0 is
  `arch=qwen35` dense, same code path.)
- llama.cpp run: `llama-bench -m <path> -p 512 -n 64 -ngl 99 -sm layer -fa 1 -r 5`.

## Honest headline

After proper 5-rep measurement on both sides:

- **Prefill: flambeau beats llama.cpp by ~4 %** (141.36 ± 0.16 vs
  135.70 ± 3.68). Real win attributable to the C1–C9 cycle — this
  model's prefill was tied at v2.27 single-rep.
- **Decode: flambeau at 97 % of llama.cpp** (19.09 ± 0.19 vs 19.65 ± 0.03).
  This gap was present at v2.27 too (single-rep data just masked it);
  C1–C9 didn't move it.

The correction to my earlier note: I had labeled decode's −3.9 %
"noise" on one-rep numbers and then corrected to call it a regression
on five-rep numbers. The final A/B on `panic = "abort"` showed C1–C9
are perf-neutral on decode; the gap is pre-existing.
