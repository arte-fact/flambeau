# V2.2 dense-FFN integration — design note

**Status:** scaffold shipped (`qwen35` arch recognised, config parses,
`Qwen3MoEConfig::is_dense_ffn()` accessor added). Loader + forward
changes below are the next-session work.

## Target model

- Qwen3.5-9B-Q4_1 (arch=`qwen35`, 5.5 GiB, fits 1× MI50)
- Same hybrid GDN + full-attn structure as Qwen3.6-35B (`qwen35moe`), but
  **dense FFN** per layer instead of MoE + shared expert.

## Per-layer tensor inventory (from `inspect-gguf`)

Dense (qwen35) layer:
- `blk.N.ffn_gate.weight`  [inter, hidden]   e.g. [12288, 4096]
- `blk.N.ffn_up.weight`    [inter, hidden]   e.g. [12288, 4096]
- `blk.N.ffn_down.weight`  [hidden, inter]   e.g. [4096, 12288]

MoE (qwen35moe/36moe) layer has the above _exps + _shexp equivalents.

## Simplest-viable refactor strategy

Reuse the existing **shared-expert** code path for qwen35's dense FFN.
The shared-expert forward (`forward_shared_expert_decode`) is already a
dense FFN over Q8_0 weights — the exact shape we need, just with
different dtypes (Q4_1 in the Qwen3.5-9B GGUF; verify Q4_1 MMVQ certs
exist or port a DP4A Q4_1 variant).

Structural approach: **map qwen35's dense FFN tensors into the shared-
expert slots** at load time, leave MoE slots empty, and branch in
forward when `cfg.is_dense_ffn()`:

### 1. `names.rs`
Add dense tensor names alongside MoE-expert names:
```rust
pub struct ModelTensorNames {
    ...existing MoE / shexp names...
    // Populated only for arch=qwen35:
    pub ffn_gate_dense: String,  // format!("{prefix}.ffn_gate.weight")
    pub ffn_up_dense:   String,
    pub ffn_down_dense: String,
}
```

### 2. `layout.rs`
When resolving layer tensors: if arch=qwen35, resolve `ffn_gate/up/down`
from the dense names. Put them in the `shared: Option<..>` slot of
`FfnLayout` (reuse the struct). Leave `ffn_gate_exps/up_exps/down_exps`
as None/absent.

Two routes:
- **Option A** (minimal-churn): add `dense: Option<DenseFfnLayout>` to
  `FfnLayout`, mutually exclusive with MoE fields.
- **Option B** (cleaner): make `FfnLayout` an enum `Moe{...} | Dense{...}`.

Pick A to keep the diff tight.

### 3. `weights.rs`
Mirror the layout change. `FfnWeights` gets `dense: Option<DenseFfnWeights>`
mutually exclusive with the _exps triple. Loader reads one branch or
the other.

### 4. `sharded.rs`
Weight upload: for qwen35 layers, upload only the dense triple to the
layer's owning rank. Skip the MoE per-expert slice logic entirely.

### 5. `forward.rs`
In `forward_one_token_pp`'s per-layer body, after attn + post_attention_norm:
```rust
if cfg.is_dense_ffn() {
    // One dense FFN; use shared-expert helper directly (or inline).
    forward_dense_ffn_decode(
        ops, stream,
        &layer.ffn.dense.as_ref().unwrap(),
        scratch, x_norm, x_out,
    )?;
} else {
    // Existing MoE path: router → topk → indexed_moe gate+up + down
    //                   + shared-expert → combine.
    forward_routed_moe_decode(...);
    forward_shared_expert_decode(...);  // adds to x_out
}
```

`forward_dense_ffn_decode` is essentially `forward_shared_expert_decode`
without the sigmoid-gate post-scale (qwen35 has no `ffn_gate_inp_shexp`
per-token gate). Likely a 30-line helper that mmvq(ffn_gate) + mmvq(ffn_up)
+ swiglu + mmvq(ffn_down) → x_out.

### 6. Dtype coverage
Qwen3.5-9B-Q4_1.gguf uses Q4_1 for main MMVQ weights and Q5_K for
ssm_out. Confirm our existing MMVQ dispatch covers these:
- Q4_1: check `crates/ops/src/hip/qmatmul.rs` — likely needs a Q4_1
  MMVQ kernel (don't think we have one yet; we only shipped Q4_K, Q5_K,
  Q6_K, Q8_0 DP4A). Either port Q4_1 MMVQ (V2.2.a) or run with a
  dequantise-then-F16-matmul fallback (slow).
- Q5_K ssm_out: already certed + dispatched.

## Parity test (V2.2.b)
`tests/parity_qwen35_vs_llama_cpp.rs` — 8-token greedy decode from
`[9419]`, compare against `llama-cli -m Qwen3.5-9B-Q4_1.gguf -p "Hello"
-n 8 --temp 0`. Must be bit-exact like the Qwen3.6 parity cert.

## Bench Mesh<1>/<2>/<4>
Once loader + forward + parity land, run `perf_baseline_qwen3_moe.rs`
on Qwen3.5-9B across Mesh<1>/<2>/<4>. This is the first true test of
the V1 roadmap's 1×-MI50 gates (≥60 tok/s tg64). Snapshot JSON goes to
`certs/perf/qwen35_9b_q4_1_mesh{1,2,4}.json`.

## Effort estimate
- Session 1: names.rs, layout.rs, weights.rs, sharded.rs. No forward
  changes yet; builds green, qwen35 loads but forward errors.
- Session 2: Q4_1 MMVQ kernel + cert + dispatch row (if needed).
- Session 3: forward_dense_ffn_decode + wire into forward_one_token_pp,
  parity test, perf baseline snapshot.

Total: **3 sessions** for a bit-exact Qwen3.5-9B running Mesh<1>/<2>/<4>
with perf snapshot. Unlocks V2.4 (speculative decoding using Qwen3.5-9B
as draft for Qwen3.6-35B).
