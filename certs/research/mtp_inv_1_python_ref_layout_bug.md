# MTP-INV-1 — Python reference q_proj split was buggy; flambeau forward is correct

**Status:** structural divergence resolved. Parity test now passes
(`max_abs=0.14, mean_abs=0.026`, tol 0.5/0.1). The bug was in the
**reference**, not in flambeau.

## What happened

The MTP parity test (`mtp_step_parity.rs`) had been failing by
`max_abs=10, mean_abs=1.4` against the Python reference for several
sessions, dismissed as "stale reference data". The MTP-4-C BF16
investigation closed with a null result, suggesting the 56.2 %
acceptance ceiling wasn't a precision issue. To investigate further
the user requested an activation bisect vs vLLM/sglang.

Before reaching for vLLM, we instrumented the parity test
(`mtp_step_bisect.rs`) to download every intermediate MTP-step
buffer (`fc_in`, `h0`, `attn_pre_gate`, `attn_post_gate`, `h1`,
`h2`, `h_final`) and compare each stage to the Python reference.

The bisect immediately localised the divergence to **`attn_post_gate`**
(the `sigmoid(gate) * attn_out` step):

| stage | max_abs (orig) | max_abs (after fix) |
|-------|---------------:|--------------------:|
| fc_in           | 1.9e-3 | 1.9e-3 |
| h0              | 3.3e-2 | 3.3e-2 |
| attn_pre_gate   | 6.2e-2 | 6.2e-2 |
| **attn_post_gate** | **7.6e+0** | **6.6e-2** |
| h1              | 3.2e+1 | 7.2e-2 |
| h2              | 3.0e+1 | 1.1e-1 |
| h_final         | 1.0e+1 | 1.4e-1 |

`attn_pre_gate` matched everywhere — single-token attention with
softmax([Q·K])=1 makes the output = V regardless of Q, so this
diagnostic is blind to Q-split bugs. `attn_post_gate` is where the
gate values are first consumed, and the divergence at element 0
implied gate≈+1.02 (ref) vs gate≈-2.57 (us). The two implementations
were splitting `q_full` differently.

## Root cause

`mtp.layers.0.self_attn.q_proj.weight` projects to `2 * num_q_heads
* head_dim = 12288` outputs. Two plausible layouts:

- **Layout A (concatenated halves):** `[Qx24, gatex24]` —
  `gate = q_full[6144:12288]`
- **Layout B (per-head interleaved):** `[Q_h0, gate_h0, Q_h1,
  gate_h1, ...]` —
  `gate[h, d] = q_full[h*512 + 256 + d]`

flambeau's `split_q_gate_f16` kernel (and `split_q_gate_bf16` from
MTP-4-C-5) implements Layout B. The Python reference at
`tools/mtp_reference.py:191-192` implemented Layout A. The two agree
only at head 0 (`q_full[0:256]` is head-0 Q under both), explaining
why attn_pre_gate matched (single-token attn doesn't depend on Q at
all in this harness).

llama.cpp is the canonical authority. From
`/artefact/llama.cpp/src/models/qwen35moe.cpp:132-156`:

```cpp
ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
    ggml_element_size(Qcur_full) * n_embd_head * 2,
    ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);

ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
    ggml_element_size(Qcur_full) * n_embd_head * 2,
    ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
    ggml_element_size(Qcur_full) * n_embd_head);  // offset = head_dim
```

Both views use stride `head_dim*2` per head with offsets 0 (Q) and
`head_dim` (gate). That's **Layout B — flambeau is correct.**

V1.7.4.b had previously verified flambeau's split layout bit-exact
vs llama.cpp on the BASE Qwen3.5/3.6 model's full-attn layers; the
MTP head reuses the same `q_proj` shape convention.

## Fix

`tools/mtp_reference.py` updated to:
```python
q_full_v = q_full.view(NUM_Q_HEADS, 2 * HEAD_DIM)
q_flat = q_full_v[:, :HEAD_DIM].reshape(-1)
gate_flat = q_full_v[:, HEAD_DIM:].reshape(-1)
```

Reference vectors regenerated. Parity test now passes with all
stages in the F16/Q8 noise band.

## What this implies for acceptance

flambeau's MTP forward is **structurally correct**. The 56.2 %
greedy acceptance is the **genuine model behaviour** on this prompt
+ harness, not a flambeau bug.

Remaining hypotheses for the gap to vLLM/sglang's reported >75 %
greedy acceptance:

1. **KV priming over prefill.** vLLM's `spec_info.hidden_states`
   flow runs MTP forward over every prefill token, populating
   the MTP KV cache with the prompt prefix. Our acceptance harness
   only accumulates MTP's K/V starting at decode step 0 — at the
   first decode step the MTP attention sees a 1-slot KV with no
   prompt context. Closing this could plausibly add the missing
   ~20 pp.
2. **Position alignment.** Our harness uses `position+1` for the
   MTP step (the position of the predicted token). Need to verify
   vLLM uses the same convention.
3. **Sampling/verify policy.** Greedy compare vs the base model's
   greedy can systematically penalise tied / near-tied
   distributions; vLLM may verify against the *full* sampled
   distribution.

Filed as MTP-INV-2 (KV prefill priming) — the most likely single
lever to clear 75 %.

## Code state

- `tools/mtp_reference.py` — Python q_proj split fixed (Layout B)
  and F16-dtype loader added (the converter switched to F16 linears
  in MTP-4 but the reference loader still expected Q8_0).
- `tests/data/mtp_ref/expected_*.bin` — regenerated.
- `crates/models/qwen3-moe/tests/mtp_step_bisect.rs` — new
  per-stage diagnostic test; useful for any future MTP forward
  changes.
- `MtpForwardScratch` fields exposed `pub` so integration tests
  can download intermediate buffers.
- `tools/mtp_reference.py:91` — F16 dtype handler added (was
  rejecting non-Q8_0 linears since session 4).

## Sources

- [`llama.cpp/src/models/qwen35moe.cpp:132-156`] — Qcur / gate ggml views
- [`tools/mtp_reference.py`] — Python reference, post-fix
- [`mtp_step_bisect.rs`] — flambeau stage-by-stage bisect harness
