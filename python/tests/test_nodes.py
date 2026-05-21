"""Tests for graphnet.nodes — HDC operation graph (Phase 16)."""

from __future__ import annotations

import graphnet
import pytest
from graphnet.nodes import HdcNode, NodeGraph, NodeGraphError, NodeKind, available_kinds


def test_node_kind_catalog_covers_plan_section_16() -> None:
    # Plan §16 lists 15+ kinds. We have at least the 16 enumerated.
    assert len(list(NodeKind)) >= 15


def test_available_kinds_reports_wired_state() -> None:
    pairs = available_kinds()
    # At least RANDOM / BIND / UNBIND / BUNDLE / COS_SIM / HAMMING are wired.
    wired = [name for name, ok in pairs if ok]
    assert "random" in wired
    assert "bind" in wired
    assert "unbind" in wired
    assert "bundle" in wired
    assert "cos_sim" in wired
    assert "hamming" in wired


def test_empty_graph_executes_to_empty_results() -> None:
    g = NodeGraph()
    assert len(g) == 0
    results = g.execute() if graphnet.native_available() else {}
    assert results == {}


def test_add_returns_increasing_ids() -> None:
    g = NodeGraph()
    a = g.add(NodeKind.RANDOM, params={"dim": 100, "seed": 1})
    b = g.add(NodeKind.RANDOM, params={"dim": 100, "seed": 2})
    assert a == 0
    assert b == 1
    assert len(g) == 2


def test_topological_order_simple_chain() -> None:
    g = NodeGraph()
    a = g.add(NodeKind.RANDOM, params={"dim": 100, "seed": 1})
    b = g.add(NodeKind.RANDOM, params={"dim": 100, "seed": 2})
    c = g.add(NodeKind.BIND, inputs=[a, b])
    order = g.topological_order()
    # a and b come before c.
    assert order.index(c) > order.index(a)
    assert order.index(c) > order.index(b)


def test_topological_order_dangling_input_errors() -> None:
    g = NodeGraph()
    g.add(NodeKind.BIND, inputs=[42, 43])  # references missing IDs
    with pytest.raises(NodeGraphError, match="missing input"):
        g.topological_order()


def test_get_missing_node_errors() -> None:
    g = NodeGraph()
    with pytest.raises(NodeGraphError, match="no node with id"):
        g.get(99)


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_execute_simple_bind_graph() -> None:
    g = NodeGraph()
    a = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 1})
    b = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 2})
    c = g.add(NodeKind.BIND, inputs=[a, b])
    results = g.execute()
    assert results[a].dim() == 1_000
    assert results[b].dim() == 1_000
    assert results[c].dim() == 1_000


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_execute_bind_unbind_round_trip() -> None:
    g = NodeGraph()
    v = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 7})
    k = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 8})
    bound = g.add(NodeKind.BIND, inputs=[v, k])
    recovered = g.add(NodeKind.UNBIND, inputs=[bound, k])
    results = g.execute()
    assert results[recovered] == results[v]


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_execute_cos_sim_self_is_one() -> None:
    g = NodeGraph()
    v = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 1})
    sim = g.add(NodeKind.COS_SIM, inputs=[v, v])
    results = g.execute()
    assert abs(results[sim] - 1.0) < 1e-9


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_execute_bundle_three_random() -> None:
    g = NodeGraph()
    a = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 1})
    b = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 2})
    c = g.add(NodeKind.RANDOM, params={"dim": 1_000, "seed": 3})
    bundled = g.add(NodeKind.BUNDLE, inputs=[a, b, c])
    results = g.execute()
    assert results[bundled].dim() == 1_000


def test_bind_with_wrong_input_count_errors() -> None:
    g = NodeGraph()
    a = g.add(NodeKind.RANDOM, params={"dim": 100, "seed": 1})
    g.add(NodeKind.BIND, inputs=[a])  # only 1 input
    if graphnet.native_available():
        with pytest.raises(NodeGraphError, match="expected 2"):
            g.execute()


def test_unwired_kind_errors_clearly() -> None:
    g = NodeGraph()
    a = g.add(NodeKind.RANDOM, params={"dim": 100, "seed": 1})
    g.add(NodeKind.HRR_BIND, inputs=[a, a])
    if graphnet.native_available():
        with pytest.raises(NodeGraphError, match="not yet wired"):
            g.execute()


def test_hdcnode_repr_includes_kind() -> None:
    n = HdcNode(node_id=1, kind=NodeKind.BIND, inputs=[2, 3], params={})
    s = repr(n)
    assert "bind" in s
    assert "id=1" in s
