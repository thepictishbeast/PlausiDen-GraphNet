//! Benchmarks for HDC operations (Phase 0 stub).
//!
//! Replaced with real operation benches in Phase 1 once `Model::forward` lands.

#![allow(missing_docs)] // criterion_group!/criterion_main! emit undocumented fns

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder_bench(c: &mut Criterion) {
    c.bench_function("graphnet_engine::banner", |b| {
        b.iter(graphnet_engine::banner);
    });
}

criterion_group!(benches, placeholder_bench);
criterion_main!(benches);
