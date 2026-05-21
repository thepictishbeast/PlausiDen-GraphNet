//! ForwardTrace — per-operation activation capture during a forward pass.
//!
//! Phase 1's introspection primitive. Used by:
//! - The Jupyter notebook viz (Phase 3) to populate live heatmaps
//! - The audit pipeline (Phase 10) to compute per-op contribution scores
//! - The debugger (plan §21.1) for step / backstep traversal
//!
//! Full channel-based subscription with multiple receivers is Phase 6
//! (continuous-execution mode). Phase 1 just captures intermediate state
//! into a returned struct.

use plausiden_hdc::Hypervector;
use serde::{Deserialize, Serialize};

/// One observation: which operation produced what intermediate output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationOutput {
    /// Operation tag (`"identity"`, `"dense"`, ...).
    pub tag: String,
    /// Index of this operation within the Stack.
    pub index: usize,
    /// The operation's output hypervector.
    pub output: Hypervector,
}

/// Per-op activation capture from one forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardTrace {
    /// The input that was fed to the Stack.
    pub input: Hypervector,
    /// Per-operation outputs, in stack order.
    pub per_op: Vec<OperationOutput>,
    /// The final bundled output (what `Stack::forward` returns).
    pub bundled: Hypervector,
}

impl ForwardTrace {
    /// Returns the operation tag at the given index, or `None`.
    #[must_use]
    pub fn tag_at(&self, index: usize) -> Option<&str> {
        self.per_op.get(index).map(|o| o.tag.as_str())
    }

    /// Number of operations captured in this trace.
    #[must_use]
    pub fn len(&self) -> usize {
        self.per_op.len()
    }

    /// Returns true if the trace captured zero operations
    /// (Stack was empty — but then `forward` would have errored).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.per_op.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::op::Operation;
    use crate::stack::Stack;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn trace_captures_per_op_outputs() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense { key: hv(1) });
        let input = hv(2);
        let trace = s.forward_with_trace(&input).expect("ok");
        assert_eq!(trace.input, input);
        assert_eq!(trace.per_op.len(), 2);
        assert_eq!(trace.per_op[0].tag, "identity");
        assert_eq!(trace.per_op[0].index, 0);
        assert_eq!(trace.per_op[0].output, input); // identity preserves
        assert_eq!(trace.per_op[1].tag, "dense");
        let _ = trace.per_op[1].index;
        assert_eq!(trace.per_op[1].index, 1);
        assert_eq!(trace.bundled.dim(), 1_000);
    }

    #[test]
    fn trace_bundled_matches_plain_forward() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let input = hv(1);
        let plain = s.forward(&input).expect("ok");
        let trace = s.forward_with_trace(&input).expect("ok");
        assert_eq!(plain, trace.bundled);
    }

    #[test]
    fn tag_at_returns_op_tag() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let trace = s.forward_with_trace(&hv(1)).expect("ok");
        assert_eq!(trace.tag_at(0), Some("identity"));
        assert_eq!(trace.tag_at(99), None);
    }

    #[test]
    fn len_and_is_empty_match_per_op() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let trace = s.forward_with_trace(&hv(1)).expect("ok");
        assert_eq!(trace.len(), 1);
        assert!(!trace.is_empty());
    }
}
