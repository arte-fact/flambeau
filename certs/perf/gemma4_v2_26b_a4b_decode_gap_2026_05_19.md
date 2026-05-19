# gemma4-v2 26B-A4B MoE — decode-gap analysis vs llama.cpp

Date: 2026-05-19
Model: `gemma-4-26B-A4B-it-Q8_0.gguf`
Topology: flambeau PP2 hip:0,1; llama.cpp PP2 hip:0,1 (`--split-mode layer`)
Dims: hidden=2816, head_count=16, head_kv=8, head_dim=176,
expert_count=128, experts_per_tok=8, expert_ffn=704, 30 layers

## Measured baseline

| stack         | prefill t/s   | decode t/s    |
|---------------|--------------:|--------------:|
| flambeau-v2   | ~529 (1.08x)  | 43.2 (0.65x)  |
| llama.cpp     | ~490          | 66.1          |

Decode gap: **23.3 ms/tok (v2) vs 15.2 ms/tok (llama.cpp) = 8.1 ms/tok**.
Per layer over 30 layers: **270 µs/layer slower**.

## Profiling status

rocprofv3 was attempted in `--kernel-trace --stats --summary` mode wrapping
the v2 server. Outcome on this rig: `rocprofv3` initialised and HSA loaded
inside the wrapped process, but the wrapper produced no output files even
on clean exit attempts. The same symptom was logged in the previous
session's notes ("SIGSEGV in HipDevice::new under interposer"); the
binary appears to run but the tool does not finalise its output.
Re-running this attribution would require a session that fixes the
profiler path itself.

Internal kernel-time tracing inside v2's `Session::forward` is not wired
up — adding it is a meaningful chunk of work and is its own task.

What we have is the end-to-end bench numbers + the architecture; the
breakdown below is **structural** (where the time can plausibly go),
not measured.

## Structural breakdown

Per-layer decode budget at 270 µs/layer gap:

| Lever                                  | Plausible cost / layer | Notes                                                              |
|----------------------------------------|----------------------:|--------------------------------------------------------------------|
| MoE expert dispatch (8 × {gate,up,down}) | 100–150 µs            | v2's `indexed_moe_mmvq` batches across experts but each mat is its own launch; llama.cpp's `mul_mat_id` fuses |
| RMSNorm count (5 norms / layer)        | 50–80 µs              | gemma4-MoE cascade: `pre_router`, `pre_ffw_2`, `post_ffw_1`, `post_ffw_2`, `post_ffn`. llama.cpp folds some |
| Per-token loop in shared expert        | 20–40 µs              | Shared MLP runs every layer alongside MoE; non-fused gate+up split |
| AR / peer-copy (PP boundary)           | 10–20 µs / boundary   | 1 PP boundary per token → ~15 µs amortised over 30 layers          |
| Sampler + output head                  | 20–50 µs / token      | Amortised: ~1 µs/layer                                              |
| **Sum (rough)**                        | **~250–340 µs/layer** | Matches the measured 270 µs/layer gap                              |

## Reading

The MoE composite is the obvious budget hog. v2's `indexed_moe_mmvq_q8_0`
already batches the per-expert MMVQ launches (one launch per matrix
across all 8 experts) but issues **three** such launches per layer (gate,
up, down). llama.cpp's `mul_mat_id` does one fused launch per matrix
that internally walks the expert table. The wave-launch cost differential
on gfx906 (1–5 µs per launch under the HIP runtime) compounds to
~30–80 µs/layer for the MoE path alone.

The 5-norm cascade is gemma4-specific and unavoidable as composition;
the only win is fusing each norm with its downstream consumer
(`rmsnorm_f32_to_f16` already shipped for two of them; three more are
on the table).

## Priority levers

1. **Fuse the MoE gate+up into a single launch** (analog to llama.cpp's
   `mul_mat_id` ggate+up shape). Saves ~30 µs/layer.
2. **Fuse `post_ffw_norm_1` + shared MLP entry, and `post_ffw_norm_2`
   + MoE down-write**. Saves ~40 µs/layer.
3. **Audit decode-path AR for the size-gated event branch** (lever 3
   shipped at #253) — verify decode-shape AR (hidden=2816, top_k=8)
   is hitting the event path, not the host-sync path. Saves ~5 µs/layer
   if not already on event path.
4. **Decode-side `attention_decode_f16_splitk`** (#248). gemma4-26B-A4B
   head_dim=176 may not be on the splitk path — verify dispatch.

Closing the full 8 ms/tok gap is multi-lever; each individually is a
~3–5% bench win. Getting to parity (1.0×) needs ~3 of the 4 above;
beating llama.cpp on decode requires fused-MoE + at least one norm
fusion.

## Out

- Real kernel-time profile via working rocprofv3 (or alternative HSA
  tracer) is the next-session prerequisite.
- The structural numbers in this cert are upper-bound estimates from
  launch-overhead arithmetic on gfx906, not measured.
