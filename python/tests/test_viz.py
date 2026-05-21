"""Tests for graphnet.viz — Phase 3 2D visualisation layer.

Most tests skip when matplotlib / graphviz / native ext aren't installed.
The widget HTML tests don't need any heavy deps and run everywhere.
"""

from __future__ import annotations

import graphnet
import pytest
from graphnet.viz import (
    available_backends,
    forward_trace_repr_html,
    stack_repr_html,
)


def test_available_backends_returns_dict() -> None:
    backends = available_backends()
    assert "matplotlib" in backends
    assert "plotly" in backends
    assert "networkx" in backends
    assert "graphviz" in backends
    assert "numpy" in backends


class _FakeStack:
    """Stand-in Stack object for HTML repr tests when native ext isn't available."""

    def __init__(self, dim: int, tags: list[str]) -> None:
        self._dim = dim
        self._tags = tags

    def dim(self) -> int:
        return self._dim

    def op_tags(self) -> list[str]:
        return list(self._tags)

    def __len__(self) -> int:
        return len(self._tags)


class _FakeHV:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def dim(self) -> int:
        return self._dim


class _FakeOpOutput:
    def __init__(self, tag: str, index: int, dim: int) -> None:
        self.tag = tag
        self.index = index
        self.output = _FakeHV(dim)


class _FakeTrace:
    def __init__(self) -> None:
        self.input = _FakeHV(1000)
        self.per_op = [_FakeOpOutput("identity", 0, 1000), _FakeOpOutput("dense", 1, 1000)]
        self.bundled = _FakeHV(1000)


def test_stack_repr_html_with_fake_stack() -> None:
    s = _FakeStack(1000, ["identity", "dense"])
    html = stack_repr_html(s)
    assert "Stack" in html
    assert "dim=1000" in html
    assert "identity" in html
    assert "dense" in html


def test_stack_repr_html_empty_stack() -> None:
    s = _FakeStack(1000, [])
    html = stack_repr_html(s)
    assert "no operations yet" in html


def test_forward_trace_repr_html_with_fake_trace() -> None:
    t = _FakeTrace()
    html = forward_trace_repr_html(t)
    assert "ForwardTrace" in html
    assert "ops=2" in html
    assert "input" in html
    assert "bundled" in html
    assert "identity" in html
    assert "dense" in html


@pytest.mark.skipif(
    not graphnet.native_available() or not available_backends().get("matplotlib"),
    reason="needs native graphnet + matplotlib",
)
def test_hypervector_heatmap_returns_figure() -> None:
    from graphnet.viz import hypervector_heatmap

    v = graphnet.Hypervector.random(1000, 1)
    fig = hypervector_heatmap(v)
    assert hasattr(fig, "savefig")  # matplotlib Figure


@pytest.mark.skipif(
    not graphnet.native_available() or not available_backends().get("matplotlib"),
    reason="needs native graphnet + matplotlib",
)
def test_forward_trace_heatmap_returns_figure() -> None:
    from graphnet.viz import forward_trace_heatmap

    s = graphnet.Stack(1000)
    s.add_operation(graphnet.Operation.identity())
    s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(1000, 99)))
    trace = s.forward_with_trace(graphnet.Hypervector.random(1000, 1))
    fig = forward_trace_heatmap(trace)
    assert hasattr(fig, "savefig")


@pytest.mark.skipif(
    not graphnet.native_available() or not available_backends().get("graphviz"),
    reason="needs native graphnet + graphviz",
)
def test_stack_graph_returns_digraph() -> None:
    from graphnet.viz import stack_graph

    s = graphnet.Stack(1000)
    s.add_operation(graphnet.Operation.identity())
    s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(1000, 99)))
    dot = stack_graph(s)
    src = dot.source
    assert "Stack(dim=1000)" in src
    assert "identity" in src
    assert "dense" in src
    assert "Bundle" in src


@pytest.mark.skipif(
    not graphnet.native_available() or not available_backends().get("matplotlib"),
    reason="needs native graphnet + matplotlib",
)
def test_similarity_matrix_returns_figure() -> None:
    from graphnet.viz import similarity_matrix

    vs = [graphnet.Hypervector.random(1000, i) for i in range(3)]
    fig = similarity_matrix(vs, labels=["a", "b", "c"])
    assert hasattr(fig, "savefig")
