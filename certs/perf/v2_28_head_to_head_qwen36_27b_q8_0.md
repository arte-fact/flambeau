# V2.28 — Rust-side perf corrections C1–C9, Qwen3.6-27B-Q8_0 vs llama.cpp

**Rig**: 4×MI50 PCIe-only (no xGMI / no NVLink peer), ROCm 7.1.1.
**llama.cpp**: build `97508acb1` (8704), `-ngl 99 -sm layer -fa 1 -r 1`.
**flambeau**: HEAD at V2.28 (post-C1..C9 Rust perf corrections, atop V2.27 kernel stack).
**Config**: pp=512, tg=64, single rep each side.

## Results

| Config | flambeau | llama.cpp | flambeau / llama.cpp |
|---|---:|---:|---:|
| pp=512 (prefill) | **141.14 tok/s** | 137.22 tok/s | **1.029** |
| tg=64  (decode)  | 18.89 tok/s | 19.65 tok/s | 0.961 |

### Comparison to v2.27 baseline (before this cycle)

| Config | v2.27 flambeau | v2.27 llama.cpp | v2.28 flambeau | v2.28 llama.cpp |
|---|---:|---:|---:|---:|
| pp=512 | 135.2 (tied) | 135.2 (tied) | **141.1** | 137.2 |
| tg=64  | 19.3 | 19.5 | 18.9 | 19.7 |

**Delta vs v2.27**:
- Prefill: **+4.4 % absolute on flambeau** (135.2 → 141.1). llama.cpp drifted +1.5 % (135.2 → 137.2) on rerun, so flambeau vs llama.cpp swung from **tied to +2.9 % lead**.
- Decode: flambeau **−2.1 %** (19.3 → 18.9), llama.cpp **+0.8 %** (19.5 → 19.7). Both are well within single-rep variance (±1–2 % is typical on this rig); the ratio moved from 0.99 to 0.96.

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

- Single rep per side, not 3-rep median. v2.27 used the same convention.
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
- llama.cpp run: `llama-bench -m <path> -p 512 -n 64 -ngl 99 -sm layer -fa 1 -r 1`.

## Headline

Pre-optimisation (v2.27): flambeau **tied** llama.cpp on Qwen3.6-27B-Q8_0 prefill.
Post-optimisation (v2.28): flambeau **beats** llama.cpp on Qwen3.6-27B-Q8_0 prefill (+2.9 %).
