# doc/archive — superseded plans & phase trackers

These are **completed or superseded** planning documents, kept for historical
rationale. They are **not** live trackers. The single live roadmap is
[`../ROADMAP.md`](../ROADMAP.md).

Cross-path references inside these files are historical and may point at their
pre-archive locations.

| Doc | Nature | Status |
| --- | --- | --- |
| `PORTABILITY-ROADMAP.md` | The prior top-level roadmap (P0–P6 portability program) | Superseded by `../ROADMAP.md`; P0–P5 done, P6 folded into Track C |
| `HANDOFF.md` | Chronological program log | Superseded by `../ROADMAP.md` + memory ledger |
| `P3-STREAM-F-PORT.md` | Stream-F CUDA port phase | ✅ complete |
| `P4-CUDA-MULTIGPU.md` | CUDA multi-GPU runtime phase | 🔄 single-GPU parts done; multi-GPU hardware-gated (Track C) |
| `P5-CUDA-DECODE-PARITY.md` | End-to-end CUDA decode parity phase | ✅ complete (gemma-4-12B Q4_K token-for-token vs llama.cpp) |
| `CLIPPY_STRUCT_REFACTOR_PLAN.md` | Argument-count refactor (P2–P8) | ✅ complete |
| `BATCHED_DECODE_PLAN.md` | Batched decode | ✅ shipped |
| `MIXED_BATCH_V2_PLAN.md` | Mixed-batch V2 forward path | ✅ shipped (current decode path) |
| `PREFIX_CACHE_V2_PLAN.md` | Prefix cache V2 | ✅ shipped |
| `GEMMA4_PERF_PLAN.md` | Gemma-4 perf | Superseded by the opt-loop ledger (memory) |
| `QUANT_PERF_PLAN.md` | Quant perf sweep | ✅ done |
| `TENSOR-PARALLEL-PLAN.md` | TP topology | ✅ shipped |
| `TOOL_CALLING_FIX_PLAN.md` | Tool-calling fixes | ✅ done (protocol kept in `../TOOL_CALLING_TEST_PROTOCOL.md`) |
| `A2.4_CASCADE_PROGRESS.md` | Cascade progress snapshot | historical |
| `v2_plan_2026_05_17.md`, `v2_forward_unification.md` | V2 forward unification | ✅ shipped |
| `arch_state_2026_05_17.md`, `stage_inventory_2026_05_17.md`, `topology_refactor_2026_05_17.md` | May-2026 point-in-time snapshots | historical |
