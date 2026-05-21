"""Tests for Phase 11 rolling features — gallery + benchmarks + share."""

from __future__ import annotations

import graphnet
import pytest
from graphnet import benchmarks, gallery, share


def test_gallery_list_templates_nonempty() -> None:
    items = gallery.list_templates()
    assert len(items) >= 5
    names = {name for name, _ in items}
    assert "stack-tiny" in names
    assert "stack-standard" in names
    assert "hdc-from-scratch" in names


def test_gallery_describe_known_template() -> None:
    desc = gallery.describe("stack-tiny")
    assert "Identity" in desc or "baseline" in desc


def test_gallery_describe_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown template"):
        gallery.describe("nope-stack-99")


def test_gallery_load_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown template"):
        gallery.load("nope-stack-99")


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_gallery_load_stack_tiny() -> None:
    s = gallery.load("stack-tiny")
    assert s.dim() == 1_000
    assert len(s) == 1


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_gallery_load_stack_standard_runs_forward() -> None:
    s = gallery.load("stack-standard")
    assert s.dim() == 10_000
    assert len(s) == 3
    v = graphnet.Hypervector.random(10_000, 42)
    out = s.forward(v)
    assert out.dim() == 10_000


def test_benchmarks_list_returns_at_least_two() -> None:
    items = benchmarks.list_benchmarks()
    assert len(items) >= 2
    names = {name for name, _ in items}
    assert "associative-recall" in names
    assert "needle-in-haystack" in names


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_associative_recall_perfect_on_identity_model() -> None:
    # Identity model returns memory unchanged → unbind should give back v_i
    # exactly, so cleanup should be perfect.
    model = gallery.load("hdc-from-scratch")
    model.add_operation(graphnet.Operation.identity())
    result = benchmarks.associative_recall(
        model, n_pairs=5, n_trials=20, dim=10_000, seed=1
    )
    assert result.name == "associative-recall"
    assert result.n_trials == 20
    # Identity model + small N + large D should hit near-perfect recall.
    assert result.accuracy >= 0.9, f"got {result.accuracy}"


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_needle_in_haystack_runs() -> None:
    model = gallery.load("hdc-from-scratch")
    model.add_operation(graphnet.Operation.identity())
    result = benchmarks.needle_in_haystack(
        model, haystack_size=10, n_trials=5, dim=10_000, seed=1
    )
    assert result.name == "needle-in-haystack"
    assert result.n_trials == 5
    assert 0.0 <= result.accuracy <= 1.0


def test_share_channels_listed() -> None:
    channels = share.list_channels()
    assert any(name == "file" for name, _ in channels)
    assert any(name == "gist" for name, _ in channels)


@pytest.mark.skipif(
    not graphnet.native_available(),
    reason="needs native graphnet",
)
def test_share_to_file_writes_yaml(tmp_path: object) -> None:
    s = gallery.load("stack-tiny")
    path = str(tmp_path) + "/arch.yaml"  # type: ignore[operator]
    result = share.to_file(s, path)
    assert result.channel == "file"
    assert result.bytes_uploaded > 0
    from pathlib import Path

    written = Path(path).read_text(encoding="utf-8")
    assert "version" in written
    assert "stack" in written
