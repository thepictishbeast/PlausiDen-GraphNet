//! A Stack — heterogeneous bundle of operations on a shared hypervector.
//!
//! The Stack primitive is the architectural unit. It contains N operations
//! (Phase 1 supports 3 modes), applies them all in parallel to the same
//! input, then bundles the results into a single output hypervector.
//!
//! This is **not** Mixture of Experts. The operations share the input, the
//! output is a single bundled hypervector (not selected from competing
//! candidates), and the operations may be architecturally heterogeneous
//! (one Dense + one HrrBind + one Identity, for example).
//!
//! Phase 1 surface:
//!
//! - [`Stack::new`] — empty Stack at dimensionality `D`
//! - [`Stack::with_operation`] — builder appending an op
//! - [`Stack::forward`] — apply all ops + bundle
//! - [`Stack::dim`] / [`Stack::len`] / [`Stack::is_empty`] — introspection
//!
//! Phase 2+ adds tau (entropy gradient), gating, snapshot/restore,
//! intervention API.

use plausiden_hdc::{bundle, Hypervector};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::op::{Operation, OperationError};

/// Errors during Stack execution.
#[derive(Debug, Error)]
pub enum StackError {
    /// A nested operation failed.
    #[error("op: {0}")]
    Op(#[from] OperationError),

    /// HDC primitive failure.
    #[error("hdc: {0}")]
    Hdc(#[from] plausiden_hdc::HdcError),

    /// `forward` was called on an empty Stack.
    #[error("forward on empty Stack (need at least one operation)")]
    Empty,

    /// An operation produced an output of incompatible dimensionality.
    #[error("op `{op}` produced dim {got}, expected {expected}")]
    DimMismatch {
        /// Operation tag.
        op: &'static str,
        /// Actual output dimensionality.
        got: usize,
        /// Expected dimensionality (the Stack's `dim`).
        expected: usize,
    },
}

/// A Stack — N operations sharing a hypervector input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stack {
    dim: usize,
    operations: Vec<Operation>,
}

impl Stack {
    /// Create an empty Stack at dimensionality `dim`.
    ///
    /// BUG ASSUMPTION: `dim` must match the dimensionality of any keys that
    /// will be added via [`Operation::Dense`] or [`Operation::HrrBind`].
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            operations: Vec::new(),
        }
    }

    /// Builder: append `op` to this Stack and return.
    #[must_use]
    pub fn with_operation(mut self, op: Operation) -> Self {
        self.operations.push(op);
        self
    }

    /// Add `op` in place (returns `&mut Self` for chaining).
    pub fn add_operation(&mut self, op: Operation) -> &mut Self {
        self.operations.push(op);
        self
    }

    /// Returns the Stack's dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of operations in the Stack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns true if no operations have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Read-only access to operations (for inspection by viz / debugging).
    #[must_use]
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Apply all operations to `input` in parallel, then bundle the outputs.
    ///
    /// Per the Stack semantics: heterogeneous compute, single bundled output.
    ///
    /// BUG ASSUMPTION: returns `Err(Empty)` if no operations have been added.
    /// All operations must produce outputs of equal dimensionality to the
    /// Stack's `dim`; mismatches surface as `Err(DimMismatch)`.
    pub fn forward(&self, input: &Hypervector) -> Result<Hypervector, StackError> {
        if self.operations.is_empty() {
            return Err(StackError::Empty);
        }
        if input.dim() != self.dim {
            return Err(StackError::DimMismatch {
                op: "stack.input",
                got: input.dim(),
                expected: self.dim,
            });
        }

        let outs: Result<Vec<Hypervector>, OperationError> =
            self.operations.iter().map(|op| op.apply(input)).collect();
        let outs = outs?;

        for (op, out) in self.operations.iter().zip(&outs) {
            if out.dim() != self.dim {
                return Err(StackError::DimMismatch {
                    op: op.tag(),
                    got: out.dim(),
                    expected: self.dim,
                });
            }
        }

        let refs: Vec<&Hypervector> = outs.iter().collect();
        Ok(bundle(&refs)?)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use plausiden_hdc::cos_sim;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn empty_stack_errors() {
        let s = Stack::new(1_000);
        assert!(s.is_empty());
        let v = hv(1);
        let err = s.forward(&v).expect_err("empty stack should error");
        assert!(matches!(err, StackError::Empty));
    }

    #[test]
    fn identity_stack_is_identity() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let v = hv(1);
        let out = s.forward(&v).expect("ok");
        assert_eq!(v, out);
    }

    #[test]
    fn two_identity_ops_still_identity_via_majority() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let v = hv(1);
        let out = s.forward(&v).expect("ok");
        // bundle of identical vectors returns the input (every bit sums positive).
        assert_eq!(v, out);
    }

    #[test]
    fn dense_then_identity_partial_recovery() {
        // Stack of (Dense, Identity): output = bundle(v ⊗ k, v).
        // Result should retain partial similarity to v (cos_sim > 0) but be
        // less similar than pure identity.
        let v = hv(1);
        let k = hv(2);
        let s = Stack::new(1_000)
            .with_operation(Operation::Dense { key: k })
            .with_operation(Operation::Identity);
        let out = s.forward(&v).expect("ok");
        let sim = cos_sim(&out, &v).expect("ok");
        assert!(sim > 0.3, "should retain identity contribution: {sim}");
        assert!(sim < 0.95, "should be diluted by dense binding: {sim}");
    }

    #[test]
    fn input_dim_mismatch_errors() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let wrong = Hypervector::random_seeded(500, 1);
        let err = s.forward(&wrong).expect_err("mismatch should error");
        assert!(matches!(err, StackError::DimMismatch { .. }));
    }

    #[test]
    fn add_operation_mutator() {
        let mut s = Stack::new(1_000);
        s.add_operation(Operation::Identity)
            .add_operation(Operation::Identity);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn len_and_is_empty_track_operations() {
        let mut s = Stack::new(1_000);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        s.add_operation(Operation::Identity);
        assert!(!s.is_empty());
        assert_eq!(s.len(), 1);
    }
}
