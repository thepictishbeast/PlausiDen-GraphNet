"""Tests for graphnet.adapters — Phase 5 multi-architecture adapter layer."""

from __future__ import annotations

import graphnet
import pytest
from graphnet.adapters import (
    AdapterError,
    ArchSummary,
    ExternalAdapter,
    ModelAdapter,
    StackAdapter,
    available_adapters,
)


def test_available_adapters_includes_native() -> None:
    avail = available_adapters()
    assert avail["stack"] is True
    assert avail["external"] is True
    assert "transformers" in avail
    assert "mamba" in avail
    assert "torch" in avail


def test_external_adapter_runs_closure() -> None:
    seen = []

    def fn(x: int) -> int:
        seen.append(x)
        return x * 2

    a = ExternalAdapter(
        family="doubler",
        input_dim=None,
        output_dim=None,
        forward_fn=fn,
    )
    assert a.forward(21) == 42
    assert seen == [21]


def test_external_adapter_satisfies_protocol() -> None:
    a = ExternalAdapter(
        family="id",
        input_dim=10,
        output_dim=10,
        forward_fn=lambda x: x,
    )
    assert isinstance(a, ModelAdapter)


def test_external_adapter_arch_summary() -> None:
    a = ExternalAdapter(
        family="my-lfi",
        input_dim=10_000,
        output_dim=10_000,
        forward_fn=lambda x: x,
        substructures=42,
        notes=["trained 2026"],
    )
    s = a.arch_summary()
    assert isinstance(s, ArchSummary)
    assert s.family == "my-lfi"
    assert s.input_dim == 10_000
    assert s.output_dim == 10_000
    assert s.substructures == 42
    assert s.notes == ["trained 2026"]


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_stack_adapter_wraps_native_stack() -> None:
    s = graphnet.Stack(1_000)
    s.add_operation(graphnet.Operation.identity())
    a = StackAdapter(s)
    assert isinstance(a, ModelAdapter)
    assert a.family == "stack"

    v = graphnet.Hypervector.random(1_000, 1)
    out = a.forward(v)
    assert out == v


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_stack_adapter_arch_summary_reflects_ops() -> None:
    s = graphnet.Stack(1_000)
    s.add_operation(graphnet.Operation.identity())
    s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(1_000, 99)))
    a = StackAdapter(s)
    summary = a.arch_summary()
    assert summary.family == "stack"
    assert summary.input_dim == 1_000
    assert summary.output_dim == 1_000
    assert summary.substructures == 2
    assert "identity" in summary.notes[0]
    assert "dense" in summary.notes[0]


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_stack_adapter_rejects_non_stack() -> None:
    with pytest.raises(AdapterError, match=r"expected graphnet\.Stack"):
        StackAdapter("not a stack")


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_stack_adapter_forward_with_trace_exposed() -> None:
    s = graphnet.Stack(1_000)
    s.add_operation(graphnet.Operation.identity())
    a = StackAdapter(s)
    trace = a.forward_with_trace(graphnet.Hypervector.random(1_000, 1))
    assert len(trace.per_op) == 1


def test_transformers_adapter_raises_when_framework_missing() -> None:
    # Even if transformers IS installed, .from_pretrained with a bogus name
    # should still error — but the import-error path is the one we test here.
    if available_adapters().get("transformers"):
        pytest.skip("transformers IS installed; can't test missing-framework path")

    from graphnet.adapters.transformers import TransformerAdapter

    with pytest.raises(AdapterError, match="transformers not installed"):
        TransformerAdapter.from_pretrained("gpt2")


def test_mamba_adapter_raises_when_framework_missing() -> None:
    if available_adapters().get("mamba"):
        pytest.skip("mamba_ssm IS installed; can't test missing-framework path")

    from graphnet.adapters.mamba import MambaAdapter

    with pytest.raises(AdapterError, match="mamba_ssm not installed"):
        MambaAdapter.from_pretrained("state-spaces/mamba-130m")


def test_torch_adapter_raises_when_framework_missing() -> None:
    if available_adapters().get("torch"):
        pytest.skip("torch IS installed; can't test missing-framework path")

    from graphnet.adapters.torch import PytorchAdapter

    with pytest.raises(AdapterError, match="torch not installed"):
        PytorchAdapter("not a module")  # type: ignore[arg-type]
