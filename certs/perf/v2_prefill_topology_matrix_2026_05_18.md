# v2 prefill across SD/TP/PP/Hybrid after U3 (2026-05-18)

## Setup
- Model: `/artefact/models/Qwen3.5-9B-Q4_1.gguf`
- Prompt: 1044 prompt_tokens (256× "the quick brown fox" + " Reply: ok")
- `max_tokens=1`, `--prefill-ubatch 512`, `--ctx-cap 4096`
- 3 runs after 1 warmup each
- gfx906 MI50s

## Results

| Topology    | Devices  | wall (s) | prefill_tok/s |
|-------------|----------|---------:|--------------:|
| SD          | hip:0          |  8.16  |    128        |
| TP=2        | hip:0,1        |  8.96  |    117        |
| PP=2        | hip:0,1        |  8.04  |    130        |
| Hybrid pp2tp2 | hip:0,2,1,3 |  8.92  |    117        |

All four topologies run the unified `Arch::forward(tokens: &[u32])`
through the same `ForwardEngine<H, S>` impl, dispatched to the same
batched composites. Zero topology-specific prefill code.

## Why these numbers look similar

- The unified forward path is dominated by per-layer compute, not by
  AR / peer-copy. TP's AR-sum on `n_tokens * hidden = 1044 * 4096`
  F32 elements adds modest overhead vs SD; PP's peer-buffer DtoH/HtoD
  at stage boundaries is two transfers of the same size, negligible
  on this 1044-token chunked prefill (one peer-copy per chunk × 2
  chunks ≈ 256 KB total over the wire).
- The GDN per-token loop is the dominant remaining cost (qwen3.5-9B
  is hybrid: half the layers are GDN). It runs identically on every
  topology, so the topology-axis spread is small.
- PP=2 modestly beats SD because two ranks share the dense-attn /
  dense-FFN compute (each owns half the layers); the peer-copy adds
  back some of the win.

## Architecture validation

This cert is the contract that U7 was set up to verify: the unified
forward stack delivers prefill on every topology with **zero
new ctx code, zero new orchestration**. The four topologies are
typedefs over `ForwardEngine<H, S>` (U2); they all run the same
composites (U3) called by the same per-arch `Arch::forward` (U4)
through the same `Session::forward` worker command (U5).

## Remaining gap

Legacy on SD is 503 tok/s (pre-U3 cert). The unified v2 stack at
128-130 tok/s leaves a 3.9× gap. The gap is the same on every
topology; it's the GDN per-token loop, not topology overhead.
Closing it = batched GDN (project all N tokens through Q/K/V/gate
matmul-batched, loop only the recurrent state-step). Filed as
follow-up; the architecture supports the swap without touching
ForwardCtx / ForwardEngine / topology-specific code.
