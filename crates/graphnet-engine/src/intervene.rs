//! Intervention API — live edits to a Stack with full undo support.
//!
//! Every architectural mutation a user makes through GraphNet's GUI or REPL
//! routes through this module. Each intervention records enough information
//! (in the returned [`UndoToken`]) to reverse it exactly, supporting the
//! time-travel debugging vision (plan §21.1).
//!
//! Phase 1 surface — operation-level interventions on a single Stack:
//!
//! - [`Intervention::AddOperation`] — append a new operation (or insert at
//!   a specific index)
//! - [`Intervention::RemoveOperation`] — remove the operation at index
//! - [`Intervention::ReplaceOperation`] — swap the operation at index for a
//!   new one
//!
//! Phase 7+ adds weight-level interventions (HV scalar edits) and Stack-of-
//! Stacks-level interventions (spawn / merge / delete substructures).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::op::Operation;
use crate::stack::Stack;

/// Errors during intervention application or undo.
#[derive(Debug, Error)]
pub enum InterventionError {
    /// The intervention targeted an index out of range for the current Stack.
    #[error("index {index} out of range (Stack has {len} ops)")]
    IndexOutOfRange {
        /// The bad index.
        index: usize,
        /// Stack length at the time of intervention.
        len: usize,
    },

    /// The undo token doesn't match the Stack's current state (e.g. it was
    /// generated against a different Stack or after subsequent interventions
    /// have been applied non-LIFO).
    #[error(
        "undo token mismatch: token for op `{token_op}` but Stack op at index is `{actual_op}`"
    )]
    UndoMismatch {
        /// The op the token expected to remove.
        token_op: String,
        /// The op currently at that position.
        actual_op: String,
    },
}

/// One live edit a user (or meta-controller) can apply to a Stack.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Intervention {
    /// Append `op` to the Stack (default), or insert at `at` if provided.
    AddOperation {
        /// The operation to add.
        op: Operation,
        /// Optional insertion index; `None` = push to end.
        at: Option<usize>,
    },
    /// Remove the operation at the given index.
    RemoveOperation {
        /// Index of the op to remove.
        index: usize,
    },
    /// Replace the operation at `index` with `op`.
    ReplaceOperation {
        /// Index whose operation is replaced.
        index: usize,
        /// The new operation.
        op: Operation,
    },
}

/// Information returned by [`apply_intervention`] sufficient to undo it.
///
/// Tokens are LIFO-correct: applying interventions A → B → C and then
/// undoing in order C → B → A always succeeds. Non-LIFO undos may fail
/// with [`InterventionError::UndoMismatch`] if intermediate state moved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoToken {
    /// Unique id of the intervention this token undoes.
    pub id: Uuid,
    /// When the intervention was applied.
    pub timestamp: DateTime<Utc>,
    /// The original intervention (kept for audit logging).
    pub intervention: Intervention,
    /// Information required to reverse the intervention.
    inverse: InverseAction,
}

/// The concrete reverse action for an applied intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum InverseAction {
    /// Original `AddOperation`'s inverse is to remove from that index.
    RemoveAt(usize),
    /// Original `RemoveOperation`'s inverse is to insert the removed op back.
    InsertAt { index: usize, op: Operation },
    /// Original `ReplaceOperation`'s inverse is to swap the old op back.
    ReplaceWith { index: usize, op: Operation },
}

/// Apply an intervention to `stack` and return an undo token.
pub fn apply_intervention(
    stack: &mut Stack,
    intervention: Intervention,
) -> Result<UndoToken, InterventionError> {
    let id = Uuid::new_v4();
    let timestamp = Utc::now();
    let inverse = match &intervention {
        Intervention::AddOperation { op, at } => {
            let pos = at.unwrap_or(stack.len());
            if pos > stack.len() {
                return Err(InterventionError::IndexOutOfRange {
                    index: pos,
                    len: stack.len(),
                });
            }
            stack.insert_operation(pos, op.clone());
            InverseAction::RemoveAt(pos)
        }
        Intervention::RemoveOperation { index } => {
            if *index >= stack.len() {
                return Err(InterventionError::IndexOutOfRange {
                    index: *index,
                    len: stack.len(),
                });
            }
            let removed = stack.remove_operation(*index);
            InverseAction::InsertAt {
                index: *index,
                op: removed,
            }
        }
        Intervention::ReplaceOperation { index, op } => {
            if *index >= stack.len() {
                return Err(InterventionError::IndexOutOfRange {
                    index: *index,
                    len: stack.len(),
                });
            }
            let old = stack.replace_operation(*index, op.clone());
            InverseAction::ReplaceWith {
                index: *index,
                op: old,
            }
        }
    };
    Ok(UndoToken {
        id,
        timestamp,
        intervention,
        inverse,
    })
}

/// Undo an intervention previously applied via [`apply_intervention`].
pub fn undo(stack: &mut Stack, token: UndoToken) -> Result<(), InterventionError> {
    match token.inverse {
        InverseAction::RemoveAt(index) => {
            if index >= stack.len() {
                return Err(InterventionError::IndexOutOfRange {
                    index,
                    len: stack.len(),
                });
            }
            stack.remove_operation(index);
        }
        InverseAction::InsertAt { index, op } => {
            if index > stack.len() {
                return Err(InterventionError::IndexOutOfRange {
                    index,
                    len: stack.len(),
                });
            }
            stack.insert_operation(index, op);
        }
        InverseAction::ReplaceWith { index, op } => {
            if index >= stack.len() {
                return Err(InterventionError::IndexOutOfRange {
                    index,
                    len: stack.len(),
                });
            }
            stack.replace_operation(index, op);
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use plausiden_hdc::Hypervector;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn add_then_undo_restores_empty_stack() {
        let mut s = Stack::new(1_000);
        let token = apply_intervention(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Identity,
                at: None,
            },
        )
        .expect("ok");
        assert_eq!(s.len(), 1);
        undo(&mut s, token).expect("undo ok");
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn remove_then_undo_restores_op() {
        let mut s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense { key: hv(1) });
        let token =
            apply_intervention(&mut s, Intervention::RemoveOperation { index: 0 }).expect("ok");
        assert_eq!(s.len(), 1);
        // After remove, the Dense op is now at index 0.
        assert_eq!(s.operations()[0].tag(), "dense");
        undo(&mut s, token).expect("undo ok");
        assert_eq!(s.len(), 2);
        assert_eq!(s.operations()[0].tag(), "identity");
    }

    #[test]
    fn replace_then_undo_restores_original_op() {
        let mut s = Stack::new(1_000).with_operation(Operation::Identity);
        let token = apply_intervention(
            &mut s,
            Intervention::ReplaceOperation {
                index: 0,
                op: Operation::Dense { key: hv(1) },
            },
        )
        .expect("ok");
        assert_eq!(s.operations()[0].tag(), "dense");
        undo(&mut s, token).expect("undo ok");
        assert_eq!(s.operations()[0].tag(), "identity");
    }

    #[test]
    fn add_at_specific_index() {
        let mut s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        apply_intervention(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Dense { key: hv(1) },
                at: Some(1),
            },
        )
        .expect("ok");
        assert_eq!(s.len(), 3);
        assert_eq!(s.operations()[0].tag(), "identity");
        assert_eq!(s.operations()[1].tag(), "dense");
        assert_eq!(s.operations()[2].tag(), "identity");
    }

    #[test]
    fn out_of_range_remove_errors() {
        let mut s = Stack::new(1_000);
        let err = apply_intervention(&mut s, Intervention::RemoveOperation { index: 0 })
            .expect_err("should err");
        assert!(matches!(
            err,
            InterventionError::IndexOutOfRange { index: 0, len: 0 }
        ));
    }

    #[test]
    fn lifo_undo_chain() {
        let mut s = Stack::new(1_000);
        let t1 = apply_intervention(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Identity,
                at: None,
            },
        )
        .expect("ok");
        let t2 = apply_intervention(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Dense { key: hv(1) },
                at: None,
            },
        )
        .expect("ok");
        let t3 = apply_intervention(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Identity,
                at: None,
            },
        )
        .expect("ok");
        assert_eq!(s.len(), 3);
        undo(&mut s, t3).expect("ok");
        undo(&mut s, t2).expect("ok");
        undo(&mut s, t1).expect("ok");
        assert_eq!(s.len(), 0);
    }
}
