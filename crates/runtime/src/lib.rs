//! flambeau-runtime — the glue between models and physical devices.
//!
//! V1.2: `Mesh<N>` trait surface + collective ops (`AllReduce`, `AllGather`,
//! `AllToAll`, `Broadcast`) with a CPU host-bounce reference impl. Real RCCL
//! impls land in `flambeau-backend-hip` and register against these traits.
//!
//! V1.6: typed `KvCache<L>` for `F16Contig`, `F16Transposed`, `Q8Contig`,
//! `Q8Transposed`. V1.7: chat template + tokenizer glue + CPU-side sampler.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod collective;
pub mod kv_cache;
pub mod mesh;
pub mod sampling;
pub mod tp_layout;

pub use collective::{
    AllGather, AllReduce, AllToAll, Broadcast, CollectiveError, CollectiveResult, RefMesh,
    RefRankHandle,
};
pub use kv_cache::{
    CacheLayout, F16Contig, KvCache, KvCacheError, KvCacheResult, Q8Contig, Q8_0_BLOCK_BYTES,
};
pub use mesh::{CollectiveCfg, CollectiveDType, LayerAssignment, Mesh, RankId, ReduceOp};
pub use sampling::{sample, Rng, Sampler, Sampling};
pub use tp_layout::{LayoutError, WeightLayout};
