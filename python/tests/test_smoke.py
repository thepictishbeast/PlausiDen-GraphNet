"""Smoke tests for the Python facade (Phase 0).

These run with or without the native extension installed. Phase 2 expands
this when PyO3 bindings ship.
"""

from __future__ import annotations

import graphnet


def test_version_string_present() -> None:
    v = graphnet.version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_dunder_version_matches_module() -> None:
    assert graphnet.__version__ == "0.1.0"


def test_banner_callable_when_native_available() -> None:
    try:
        b = graphnet.banner()
    except NotImplementedError:
        return  # native extension not installed in this environment
    assert isinstance(b, str)
    assert "graphnet" in b.lower()
