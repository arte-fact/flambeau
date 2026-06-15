//! Per-iteration agent-loop telemetry, exposed at
//! `GET /v1/agent/stats`.
//! Read-only debug surface. Each chat completion contributes zero or
//! more `IterStat` entries (one per agent-loop iteration); they land
//! in an in-memory ring keyed by session id. Bounded capacity, no
//! persistence.
//! Intent: catch Qwen3.6 MoE-routing oscillation in long agent
//! sessions — the community-reported failure mode where the model
//! loops through tool calls, gets stuck on the same tool, or
//! latency-drifts across iterations. Surfacing raw per-iteration
//! metrics lets the operator notice degradation from outside, without
//! having to re-instrument the kernels.

use std::collections::VecDeque;
use std::sync::Mutex;

/// One agent-loop iteration stat.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IterStat {
    /// `chatcmpl-...` id — groups all iterations from one HTTP request.
    pub session_id: String,
    /// 0-based iteration index within the session.
    pub iteration: u32,
    /// Tool names the model emitted this iteration (if any).
    pub tool_names: Vec<String>,
    /// Reserved for future use (always 0 on the single-pass path).
    pub remote_count: u32,
    /// Wall-clock ms for this iteration.
    pub latency_ms: u64,
    /// Tokens consumed this iteration (prompt + completion).
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Finish reason at this iteration (`"stop" | "tool_calls" | "length"`).
    pub finish_reason: String,
}

/// Ring buffer of iteration stats. Oldest entries are dropped when the
/// buffer fills. `capacity` is a soft cap — kept small to avoid
/// unbounded memory on long-running servers.
#[derive(Debug)]
pub struct AgentStatsRing {
    inner: Mutex<VecDeque<IterStat>>,
    capacity: usize,
}

impl AgentStatsRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    pub fn push(&self, stat: IterStat) {
        let mut q = self.inner.lock().expect("agent stats lock");
        if q.len() >= self.capacity {
            q.pop_front();
        }
        q.push_back(stat);
    }

    /// Snapshot the current buffer into a Vec. Used by the
    /// `/v1/agent/stats` handler.
    pub fn snapshot(&self) -> Vec<IterStat> {
        self.inner
            .lock()
            .expect("agent stats lock")
            .iter()
            .cloned()
            .collect()
    }
}

impl Default for AgentStatsRing {
    fn default() -> Self {
        // 256 iterations ≈ 25 recent chat completions at ~10 iters each.
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_evicts_oldest_at_capacity() {
        let ring = AgentStatsRing::new(3);
        for i in 0..5 {
            ring.push(IterStat {
                session_id: format!("s{i}"),
                iteration: 0,
                tool_names: vec![],
                remote_count: 0,
                latency_ms: i,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".into(),
            });
        }
        let snap = ring.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].session_id, "s2");
        assert_eq!(snap[2].session_id, "s4");
    }
}
