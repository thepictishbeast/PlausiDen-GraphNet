//! Intervention history with undo/redo — Phase 7.
//!
//! Wraps the per-call [`crate::apply_intervention`] / [`crate::undo`] pair with
//! a two-stack history model (one undo stack, one redo stack) so the GUI can
//! offer Ctrl/Cmd+Z and Ctrl/Cmd+Shift+Z without the caller having to track
//! tokens by hand.
//!
//! Semantics:
//!
//! - `apply` pushes the undo token onto the undo stack AND clears the redo
//!   stack (the standard "branch divergence" rule — applying a new edit makes
//!   the future you came from unreachable).
//! - `undo` pops the undo stack, reverses the action, AND pushes the original
//!   intervention onto the redo stack (so `redo` can replay it).
//! - `redo` pops the redo stack, re-applies the intervention, AND pushes the
//!   new undo token onto the undo stack.
//!
//! Bounded by `capacity`: when the undo stack exceeds capacity, the oldest
//! entry is dropped silently (matches IDE undo-buffer semantics).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::intervene::{apply_intervention, undo, Intervention, InterventionError, UndoToken};
use crate::stack::Stack;

/// Errors raised by the history wrapper.
#[derive(Debug, Error)]
pub enum HistoryError {
    /// Underlying intervention failed (bad index, etc.).
    #[error("intervene: {0}")]
    Intervene(#[from] InterventionError),
}

/// Default maximum number of undo entries to retain.
pub const DEFAULT_HISTORY_CAPACITY: usize = 256;

/// Two-stack undo/redo history for `Stack` interventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionHistory {
    capacity: usize,
    undo_stack: Vec<UndoToken>,
    redo_stack: Vec<Intervention>,
}

impl Default for InterventionHistory {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_HISTORY_CAPACITY)
    }
}

impl InterventionHistory {
    /// Construct an empty history with the given capacity (≥ 1).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    /// Apply an intervention through the history.
    ///
    /// On success: pushes the undo token onto the undo stack, clears the
    /// redo stack (branch divergence), trims to capacity.
    pub fn apply(
        &mut self,
        stack: &mut Stack,
        intervention: Intervention,
    ) -> Result<(), HistoryError> {
        let token = apply_intervention(stack, intervention)?;
        self.undo_stack.push(token);
        self.redo_stack.clear();
        if self.undo_stack.len() > self.capacity {
            // Drop oldest (front).
            self.undo_stack.remove(0);
        }
        Ok(())
    }

    /// Undo the most recent intervention.
    ///
    /// Returns `Ok(true)` if an intervention was undone; `Ok(false)` if the
    /// undo stack was empty. The undone intervention is pushed onto the
    /// redo stack so [`redo`](Self::redo) can replay it.
    pub fn undo(&mut self, stack: &mut Stack) -> Result<bool, HistoryError> {
        let Some(token) = self.undo_stack.pop() else {
            return Ok(false);
        };
        let intervention = token.intervention.clone();
        undo(stack, token)?;
        self.redo_stack.push(intervention);
        Ok(true)
    }

    /// Redo the most recently undone intervention.
    ///
    /// Returns `Ok(true)` if an intervention was re-applied; `Ok(false)` if
    /// the redo stack was empty.
    pub fn redo(&mut self, stack: &mut Stack) -> Result<bool, HistoryError> {
        let Some(intervention) = self.redo_stack.pop() else {
            return Ok(false);
        };
        let token = apply_intervention(stack, intervention)?;
        self.undo_stack.push(token);
        Ok(true)
    }

    /// True if at least one intervention can be undone.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// True if at least one intervention can be redone.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Number of undo entries currently held.
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of redo entries currently held.
    #[must_use]
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }

    /// Configured history capacity (max undo entries before trimming).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all history; both undo and redo stacks empty afterwards.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::op::Operation;
    use plausiden_hdc::Hypervector;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn empty_history_cannot_undo_or_redo() {
        let mut h = InterventionHistory::default();
        let mut s = Stack::new(1_000);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert!(!h.undo(&mut s).expect("ok"));
        assert!(!h.redo(&mut s).expect("ok"));
    }

    #[test]
    fn apply_then_undo_then_redo_round_trip() {
        let mut h = InterventionHistory::default();
        let mut s = Stack::new(1_000);

        h.apply(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Identity,
                at: None,
            },
        )
        .expect("ok");
        assert_eq!(s.len(), 1);
        assert!(h.can_undo());
        assert!(!h.can_redo());

        assert!(h.undo(&mut s).expect("ok"));
        assert_eq!(s.len(), 0);
        assert!(!h.can_undo());
        assert!(h.can_redo());

        assert!(h.redo(&mut s).expect("ok"));
        assert_eq!(s.len(), 1);
        assert!(h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn apply_clears_redo_stack() {
        let mut h = InterventionHistory::default();
        let mut s = Stack::new(1_000);

        h.apply(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Identity,
                at: None,
            },
        )
        .expect("ok");
        h.undo(&mut s).expect("ok");
        assert!(h.can_redo());

        // Applying a NEW intervention should clear redo (branch divergence).
        h.apply(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Dense { key: hv(1) },
                at: None,
            },
        )
        .expect("ok");
        assert!(!h.can_redo());
    }

    #[test]
    fn capacity_trims_oldest() {
        let mut h = InterventionHistory::with_capacity(3);
        let mut s = Stack::new(1_000);

        for _ in 0..5 {
            h.apply(
                &mut s,
                Intervention::AddOperation {
                    op: Operation::Identity,
                    at: None,
                },
            )
            .expect("ok");
        }
        // 5 applies + capacity 3 → only 3 undos available.
        assert_eq!(h.undo_depth(), 3);
        assert_eq!(h.capacity(), 3);
    }

    #[test]
    fn multi_step_undo_redo_chain() {
        let mut h = InterventionHistory::default();
        let mut s = Stack::new(1_000);

        for _ in 0..4 {
            h.apply(
                &mut s,
                Intervention::AddOperation {
                    op: Operation::Identity,
                    at: None,
                },
            )
            .expect("ok");
        }
        assert_eq!(s.len(), 4);

        // Undo all 4.
        for _ in 0..4 {
            assert!(h.undo(&mut s).expect("ok"));
        }
        assert_eq!(s.len(), 0);

        // Redo all 4.
        for _ in 0..4 {
            assert!(h.redo(&mut s).expect("ok"));
        }
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn clear_empties_both_stacks() {
        let mut h = InterventionHistory::default();
        let mut s = Stack::new(1_000);
        h.apply(
            &mut s,
            Intervention::AddOperation {
                op: Operation::Identity,
                at: None,
            },
        )
        .expect("ok");
        h.undo(&mut s).expect("ok");
        assert!(h.can_undo() || h.can_redo());

        h.clear();
        assert!(!h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn default_capacity_is_reasonable() {
        let h = InterventionHistory::default();
        assert_eq!(h.capacity(), DEFAULT_HISTORY_CAPACITY);
    }

    #[test]
    fn with_capacity_zero_clamps_to_one() {
        let h = InterventionHistory::with_capacity(0);
        assert_eq!(h.capacity(), 1);
    }
}
