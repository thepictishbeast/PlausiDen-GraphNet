//! Benchmarks for Stack forward at the canonical D=10,000.

#![allow(missing_docs, clippy::expect_used, clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use graphnet_engine::{Operation, Stack};
use plausiden_hdc::Hypervector;

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_forward");
    for &dim in &[1_000usize, 10_000] {
        let v = Hypervector::random_seeded(dim, 1);
        let k = Hypervector::random_seeded(dim, 2);

        let s_identity = Stack::new(dim).with_operation(Operation::Identity);
        group.bench_with_input(BenchmarkId::new("identity_only", dim), &dim, |bch, _| {
            bch.iter(|| s_identity.forward(&v).expect("ok"));
        });

        let s_dense = Stack::new(dim).with_operation(Operation::Dense { key: k.clone() });
        group.bench_with_input(BenchmarkId::new("dense_only", dim), &dim, |bch, _| {
            bch.iter(|| s_dense.forward(&v).expect("ok"));
        });

        let s_mixed = Stack::new(dim)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense { key: k.clone() });
        group.bench_with_input(
            BenchmarkId::new("identity_plus_dense", dim),
            &dim,
            |bch, _| {
                bch.iter(|| s_mixed.forward(&v).expect("ok"));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
