//! Kernel launch configuration, shared by every backend.

/// Launch configuration — grid + block dimensions + dynamic shared memory.
#[derive(Debug, Clone, Copy)]
pub struct LaunchCfg {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_bytes: u32,
}

impl LaunchCfg {
    /// One-dimensional launch: `grid_x` blocks of `block_x` threads, no dynamic
    /// shared memory.
    #[must_use]
    pub fn one_d(grid_x: u32, block_x: u32) -> Self {
        Self {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_bytes: 0,
        }
    }
}
