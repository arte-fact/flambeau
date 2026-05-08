//! Kernel ABI-contract constants — values that must stay in lock-step between
//! Rust-side launcher code and the matching `#define` in a `.cu` file.
//!
//! These are not "tuning knobs" the dispatch table picks from — those live in
//! `dispatch/<backend>/<arch>.toml`. Constants here are **upper bounds baked
//! into a kernel's compile-time shape** (LDS allocation, fixed-size on-chip
//! arrays) that Rust has to know about to assert at launch time.
//!
//! Each constant documents:
//! - **What kernel(s) depend on it.** Where the matching `#define` lives.
//! - **What breaks if it's raised.** Typically LDS / VGPR pressure → occupancy
//!   cliff, or silent OOB on undersized allocations.
//! - **What breaks if it's lowered.** The runtime assert at the launcher
//!   starts firing on previously-valid shapes; certs added for the larger
//!   limit become non-exercisable.
//!
//! Bumping a constant here requires bumping the matching `#define` in the
//! same PR + running the correctness sweep against the new ceiling.

/// Compile-time upper bound on `n_experts` for the `topk_f32` /
/// `flambeau_topk_softmax_f32` kernel. Matches `#define TOPK_MAX_EXPERTS` in
/// `crates/kernels-hip/src/kernels/topk_softmax.cu`.
///
/// The kernel allocates an LDS-resident scratch of `f32` + `i32` pairs sized
/// to this constant for the cross-warp top-k merge. Exceeding it silently
/// drops the tail and (pre-V1.7.4.a) triggered OOB LDS writes at the reduce
/// step.
///
/// - **Raise**: bump `#define` in the `.cu` and add a cert shape for the new
///   expert count. Each doubling grows LDS by ~2 KiB (8 B/pair) which, on
///   gfx906, can force a waves-per-SIMD step down — re-run `rocprofv3`.
/// - **Lower**: run `cargo run -p bench -- sweep --impl topk_softmax_f32` to
///   confirm no live model exceeds the new ceiling.
///
/// Historical incident: V1.7.4.a parity regression on Qwen3.6-35B-A3B was
/// caused by this being hardcoded at 128 (pre-V1.7.4.a), silently discarding
/// experts 128..255. Do not reduce below **256** without auditing every
/// supported model's expert count.
pub const TOPK_MAX_EXPERTS: usize = 512;

/// Compile-time upper bound on `n_experts` for the `moe_sort_by_expert`
/// kernel family. Matches `#define MOE_SORT_MAX_EXPERTS` (or equivalent)
/// in `crates/kernels-hip/src/kernels/moe_sort.cu`.
///
/// The histogram + prefix-sum kernels allocate `counts[MAX]`, `offsets[MAX+1]`,
/// and `cursors[MAX]` in LDS. The `512` ceiling gives headroom past any known
/// model (Qwen3.6: 256 experts, Qwen3-Coder: 128) at ~6 KiB LDS usage across
/// the three arrays.
///
/// - **Raise**: bump `#define`, re-verify LDS fits in the target arch's budget
///   (gfx906 max 64 KiB / CU, gfx1031 = 64 KiB as well).
/// - **Lower**: confirm every supported model's expert count fits.
pub const MOE_SORT_MAX_EXPERTS: usize = 512;
