//! flambeau-runtime — the glue between models and physical devices.
//!
//! V1.0 skeleton: `Mesh<N>` trait surface + reference (host-bounce) collectives.
//! V1.2: real RCCL collectives land in `flambeau-backend-hip`, registered against
//! the runtime's collective ops.
//! V1.6: typed `KvCache<L>` for `F16Contig`, `F16Transposed`, `Q8Contig`, `Q8Transposed`.
//! V1.7: chat template + tokenizer glue + CPU-side sampler.
