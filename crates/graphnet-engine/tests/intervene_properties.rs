//! Property tests for the intervention API + history round-trip — Phase 10.
//!
//! Per AVP-2 doctrine §10 Tier 6 (meta-validation): assert the algebraic
//! invariants that intervention semantics must obey, across many inputs.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use graphnet_engine::{
    apply_intervention, undo, Intervention, InterventionHistory, Operation, Stack,
};
use plausiden_hdc::Hypervector;
use proptest::prelude::*;

const D: usize = 500;

fn arb_op() -> impl Strategy<Value = Operation> {
    prop_oneof![
        Just(Operation::Identity),
        any::<u64>().prop_map(|s| Operation::Dense {
            key: Hypervector::random_seeded(D, s),
        }),
        any::<u64>().prop_map(|s| Operation::HrrBind {
            key: Hypervector::random_seeded(D, s),
        }),
    ]
}

fn arb_initial_stack(max_ops: usize) -> impl Strategy<Value = Stack> {
    proptest::collection::vec(arb_op(), 0..=max_ops).prop_map(|ops| {
        let mut s = Stack::new(D);
        for op in ops {
            s.add_operation(op);
        }
        s
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn apply_then_undo_restores_op_count(
        initial in arb_initial_stack(8),
        new_op in arb_op(),
    ) {
        let mut s = initial.clone();
        let before = s.len();
        let token = apply_intervention(
            &mut s,
            Intervention::AddOperation { op: new_op, at: None },
        ).expect("apply ok");
        prop_assert_eq!(s.len(), before + 1);
        undo(&mut s, token).expect("undo ok");
        prop_assert_eq!(s.len(), before);
    }

    #[test]
    fn replace_then_undo_restores_tag(
        ops in proptest::collection::vec(arb_op(), 1..=8),
        new_op in arb_op(),
    ) {
        let mut s = Stack::new(D);
        for op in &ops {
            s.add_operation(op.clone());
        }
        let target_index = 0;
        let original_tag = s.operations()[target_index].tag();
        let token = apply_intervention(
            &mut s,
            Intervention::ReplaceOperation { index: target_index, op: new_op },
        ).expect("replace ok");
        // After replace, tag may differ.
        undo(&mut s, token).expect("undo ok");
        prop_assert_eq!(s.operations()[target_index].tag(), original_tag);
    }

    #[test]
    fn history_undo_then_redo_is_no_op_on_state(
        initial in arb_initial_stack(6),
        edits in proptest::collection::vec(arb_op(), 1..=4),
    ) {
        let mut s = initial.clone();
        let mut h = InterventionHistory::default();
        for op in edits {
            h.apply(
                &mut s,
                Intervention::AddOperation { op, at: None },
            ).expect("apply ok");
        }
        let after_apply = s.len();
        // Undo all.
        while h.can_undo() {
            h.undo(&mut s).expect("undo ok");
        }
        prop_assert_eq!(s.len(), initial.len());
        // Redo all.
        while h.can_redo() {
            h.redo(&mut s).expect("redo ok");
        }
        prop_assert_eq!(s.len(), after_apply);
    }

    #[test]
    fn lifo_undo_chain_restores_initial_state(
        initial in arb_initial_stack(4),
        edits in proptest::collection::vec(arb_op(), 1..=6),
    ) {
        let mut s = initial.clone();
        let initial_len = s.len();
        let mut tokens = Vec::new();
        for op in edits {
            let token = apply_intervention(
                &mut s,
                Intervention::AddOperation { op, at: None },
            ).expect("apply ok");
            tokens.push(token);
        }
        // Undo in reverse (LIFO).
        for token in tokens.into_iter().rev() {
            undo(&mut s, token).expect("undo ok");
        }
        prop_assert_eq!(s.len(), initial_len);
    }
}
