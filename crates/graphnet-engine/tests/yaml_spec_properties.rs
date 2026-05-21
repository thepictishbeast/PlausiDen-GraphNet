//! Property tests for YAML spec round-trip — Phase 10.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use graphnet_engine::{stack_from_yaml, stack_to_yaml, Operation, Stack};
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

fn arb_stack(max_ops: usize) -> impl Strategy<Value = Stack> {
    proptest::collection::vec(arb_op(), 0..=max_ops).prop_map(|ops| {
        let mut s = Stack::new(D);
        for op in ops {
            s.add_operation(op);
        }
        s
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn yaml_round_trip_preserves_op_count_and_tags(s in arb_stack(6)) {
        let yaml = stack_to_yaml(&s).expect("encode ok");
        let restored = stack_from_yaml(&yaml).expect("decode ok");
        prop_assert_eq!(restored.dim(), s.dim());
        prop_assert_eq!(restored.len(), s.len());
        for (orig_op, rest_op) in s.operations().iter().zip(restored.operations()) {
            prop_assert_eq!(orig_op.tag(), rest_op.tag());
        }
    }

    #[test]
    fn yaml_round_trip_preserves_forward_output(
        s in arb_stack(4),
        seed in any::<u64>(),
    ) {
        if s.is_empty() {
            return Ok(());
        }
        let input = Hypervector::random_seeded(D, seed);
        let original = s.forward(&input).expect("forward ok");
        let yaml = stack_to_yaml(&s).expect("encode ok");
        let restored = stack_from_yaml(&yaml).expect("decode ok");
        let restored_out = restored.forward(&input).expect("forward ok");
        prop_assert_eq!(original, restored_out);
    }
}
