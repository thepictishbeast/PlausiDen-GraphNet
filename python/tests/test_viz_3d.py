"""Tests for graphnet.viz 3D rotatable visualisations (Phase 4).

Skip when plotly/scikit-learn/native ext aren't installed. The basic shape
of returned plotly.Figure objects is checked; rendering fidelity is the
visual-regression suite's job (Phase 10).
"""

from __future__ import annotations

import graphnet
import pytest
from graphnet.viz import available_backends


def _have(*names: str) -> bool:
    backends = available_backends()
    for name in names:
        if name in backends and not backends[name]:
            return False
        if name not in backends:
            try:
                __import__(name)
            except ImportError:
                return False
    return True


@pytest.mark.skipif(
    not graphnet.native_available()
    or not _have("plotly")
    or not _have("sklearn"),
    reason="needs native graphnet + plotly + scikit-learn",
)
def test_hypervector_3d_scatter_pca_returns_figure() -> None:
    from graphnet.viz import hypervector_3d_scatter

    vs = [graphnet.Hypervector.random(1000, i) for i in range(5)]
    fig = hypervector_3d_scatter(vs, method="pca", labels=[f"v{i}" for i in range(5)])
    assert fig.data, "figure should have at least one trace"
    assert fig.data[0].type == "scatter3d"


@pytest.mark.skipif(
    not graphnet.native_available() or not _have("plotly"),
    reason="needs native graphnet + plotly",
)
def test_stack_graph_3d_returns_figure() -> None:
    from graphnet.viz import stack_graph_3d

    s = graphnet.Stack(1000)
    s.add_operation(graphnet.Operation.identity())
    s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(1000, 99)))
    fig = stack_graph_3d(s)
    assert len(fig.data) >= 2  # edges + nodes
    # Node trace should be the second one.
    nodes = fig.data[1]
    assert nodes.type == "scatter3d"
    # input + ops + bundle + output = 1 + 2 + 1 + 1 = 5 nodes
    assert len(nodes.text) == 5


@pytest.mark.skipif(
    not graphnet.native_available()
    or not _have("plotly")
    or not _have("sklearn"),
    reason="needs native graphnet + plotly + scikit-learn",
)
def test_forward_trace_3d_returns_figure() -> None:
    from graphnet.viz import forward_trace_3d

    s = graphnet.Stack(1000)
    s.add_operation(graphnet.Operation.identity())
    s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(1000, 99)))
    trace = s.forward_with_trace(graphnet.Hypervector.random(1000, 1))
    fig = forward_trace_3d(trace)
    assert fig.data
    # input + 2 ops + bundled = 4 points
    nodes = fig.data[0]
    assert len(nodes.text) == 4


@pytest.mark.skipif(
    not graphnet.native_available()
    or not _have("plotly")
    or not _have("sklearn"),
    reason="needs native graphnet + plotly + scikit-learn",
)
def test_hypervector_3d_scatter_requires_min_2_vectors() -> None:
    from graphnet.viz import hypervector_3d_scatter

    v = graphnet.Hypervector.random(1000, 1)
    with pytest.raises(ValueError, match="at least 2"):
        hypervector_3d_scatter([v])


@pytest.mark.skipif(
    not graphnet.native_available()
    or not _have("plotly")
    or not _have("sklearn"),
    reason="needs native graphnet + plotly + scikit-learn",
)
def test_hypervector_3d_scatter_rejects_unknown_method() -> None:
    from graphnet.viz import hypervector_3d_scatter

    vs = [graphnet.Hypervector.random(1000, i) for i in range(3)]
    with pytest.raises(ValueError, match="unknown method"):
        hypervector_3d_scatter(vs, method="bogus")
