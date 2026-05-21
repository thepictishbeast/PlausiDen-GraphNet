"""End-to-end tests for the native PyO3 bindings.

These tests are skipped when the native extension isn't built (e.g. fresh
clones before `maturin develop`). CI installs maturin + builds the wheel
before pytest, so the gate is exercised there.
"""

from __future__ import annotations

import graphnet
import pytest

pytestmark = pytest.mark.skipif(
    not graphnet.native_available(),
    reason="native extension not installed; run `maturin develop` first",
)


def test_hypervector_random_seeded_deterministic() -> None:
    a = graphnet.Hypervector.random(1_000, 42)
    b = graphnet.Hypervector.random(1_000, 42)
    assert a == b
    assert a.dim() == 1_000


def test_hypervector_different_seed_different_vector() -> None:
    a = graphnet.Hypervector.random(1_000, 42)
    b = graphnet.Hypervector.random(1_000, 43)
    assert a != b


def test_bind_is_self_inverse() -> None:
    a = graphnet.Hypervector.random(1_000, 1)
    k = graphnet.Hypervector.random(1_000, 2)
    bound = graphnet.bind(a, k)
    recovered = graphnet.unbind(bound, k)
    assert recovered == a


def test_bundle_is_commutative() -> None:
    a = graphnet.Hypervector.random(1_000, 1)
    b = graphnet.Hypervector.random(1_000, 2)
    assert graphnet.bundle([a, b]) == graphnet.bundle([b, a])


def test_cos_sim_in_range() -> None:
    a = graphnet.Hypervector.random(1_000, 1)
    b = graphnet.Hypervector.random(1_000, 2)
    s = graphnet.cos_sim(a, b)
    assert -1.0 <= s <= 1.0


def test_hamming_in_range() -> None:
    a = graphnet.Hypervector.random(1_000, 1)
    b = graphnet.Hypervector.random(1_000, 2)
    h = graphnet.hamming(a, b)
    assert 0.0 <= h <= 1.0


def test_stack_construct_and_add_operation() -> None:
    s = graphnet.Stack(1_000)
    assert len(s) == 0
    assert s.is_empty()
    s.add_operation(graphnet.Operation.identity())
    assert len(s) == 1
    assert not s.is_empty()
    assert s.op_tags() == ["identity"]


def test_stack_identity_forward_preserves_input() -> None:
    s = graphnet.Stack(1_000)
    s.add_operation(graphnet.Operation.identity())
    v = graphnet.Hypervector.random(1_000, 1)
    out = s.forward(v)
    assert out == v


def test_stack_dense_then_undense_via_double_bind() -> None:
    v = graphnet.Hypervector.random(1_000, 1)
    k = graphnet.Hypervector.random(1_000, 2)
    s1 = graphnet.Stack(1_000)
    s1.add_operation(graphnet.Operation.dense(k))
    once = s1.forward(v)
    s2 = graphnet.Stack(1_000)
    s2.add_operation(graphnet.Operation.dense(k))
    twice = s2.forward(once)
    assert twice == v


def test_forward_with_trace_captures_per_op() -> None:
    s = graphnet.Stack(1_000)
    s.add_operation(graphnet.Operation.identity())
    s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(1_000, 99)))
    v = graphnet.Hypervector.random(1_000, 1)
    trace = s.forward_with_trace(v)
    assert trace.input == v
    assert len(trace.per_op) == 2
    assert trace.per_op[0].tag == "identity"
    assert trace.per_op[0].index == 0
    assert trace.per_op[1].tag == "dense"
    assert trace.per_op[1].index == 1


def test_snapshot_restore_round_trip_preserves_forward() -> None:
    s = graphnet.Stack(1_000)
    s.add_operation(graphnet.Operation.identity())
    s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(1_000, 7)))
    bytes_ = graphnet.snapshot(s)
    assert len(bytes_) > 0
    restored = graphnet.restore(bytes_)
    assert restored.dim() == 1_000
    assert len(restored) == 2
    v = graphnet.Hypervector.random(1_000, 5)
    assert s.forward(v) == restored.forward(v)


def test_intervention_add_then_undo() -> None:
    s = graphnet.Stack(1_000)
    token = s.apply_intervention("add", op=graphnet.Operation.identity())
    assert len(s) == 1
    s.undo_intervention(token)
    assert len(s) == 0


def test_intervention_replace_then_undo() -> None:
    s = graphnet.Stack(1_000)
    s.add_operation(graphnet.Operation.identity())
    token = s.apply_intervention(
        "replace",
        op=graphnet.Operation.dense(graphnet.Hypervector.random(1_000, 1)),
        index=0,
    )
    assert s.op_tags() == ["dense"]
    s.undo_intervention(token)
    assert s.op_tags() == ["identity"]


def test_banner_mentions_phase() -> None:
    b = graphnet.banner()
    assert "graphnet-engine" in b
    assert "Phase" in b
