"""Architecture template library — Phase 11 / plan §21.3.

Curated starter Stacks so a new user can ``gn.gallery.load("stack-tiny")``
and have something to explore in seconds, without designing an architecture
from scratch.

Templates always return a fresh Stack each call — they're factories, not
shared singletons.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _stack_tiny() -> Any:
    """Single Identity op at D=1_000; the absolute baseline."""
    import graphnet

    if graphnet.Stack is None:
        raise RuntimeError("native graphnet not installed")
    s = graphnet.Stack(1_000)
    s.add_operation(graphnet.Operation.identity())
    return s


def _stack_standard() -> Any:
    """3-op heterogeneous Stack at D=10_000: Identity + Dense + HrrBind.

    The default starter for non-trivial experimentation.
    """
    import graphnet

    if graphnet.Stack is None:
        raise RuntimeError("native graphnet not installed")
    d = 10_000
    s = graphnet.Stack(d)
    s.add_operation(graphnet.Operation.identity())
    s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(d, 1)))
    s.add_operation(graphnet.Operation.hrr_bind(graphnet.Hypervector.random(d, 2)))
    return s


def _stack_dense_only() -> Any:
    """4 Dense ops at D=10_000 with distinct keys — for spectral diversity tests."""
    import graphnet

    if graphnet.Stack is None:
        raise RuntimeError("native graphnet not installed")
    d = 10_000
    s = graphnet.Stack(d)
    for seed in range(1, 5):
        s.add_operation(graphnet.Operation.dense(graphnet.Hypervector.random(d, seed)))
    return s


def _stack_fft_heavy() -> Any:
    """3 HrrBind ops at D=1_024 (FFT-friendly power-of-2 dim)."""
    import graphnet

    if graphnet.Stack is None:
        raise RuntimeError("native graphnet not installed")
    d = 1_024
    s = graphnet.Stack(d)
    for seed in range(10, 13):
        s.add_operation(graphnet.Operation.hrr_bind(graphnet.Hypervector.random(d, seed)))
    return s


def _hdc_from_scratch() -> Any:
    """Empty Stack at D=10_000 — for hand-building from scratch in the REPL."""
    import graphnet

    if graphnet.Stack is None:
        raise RuntimeError("native graphnet not installed")
    return graphnet.Stack(10_000)


_TEMPLATES: dict[str, tuple[str, Callable[[], Any]]] = {
    "stack-tiny": ("Single Identity op at D=1,000 — baseline", _stack_tiny),
    "stack-standard": (
        "3-op heterogeneous Stack at D=10,000 (Identity + Dense + HrrBind)",
        _stack_standard,
    ),
    "stack-dense-only": ("4 Dense ops at D=10,000 with distinct keys", _stack_dense_only),
    "stack-fft-heavy": ("3 HrrBind ops at D=1,024 (power-of-2 FFT-friendly)", _stack_fft_heavy),
    "hdc-from-scratch": ("Empty Stack at D=10,000 — hand-build in REPL", _hdc_from_scratch),
}


def list_templates() -> list[tuple[str, str]]:
    """Return (name, description) pairs for every available template."""
    return [(name, desc) for name, (desc, _) in _TEMPLATES.items()]


def load(name: str) -> Any:
    """Load a starter Stack by template name."""
    if name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES))
        raise KeyError(f"unknown template `{name}`; available: {available}")
    _, factory = _TEMPLATES[name]
    return factory()


def describe(name: str) -> str:
    """Return the human-readable description for a template."""
    if name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES))
        raise KeyError(f"unknown template `{name}`; available: {available}")
    desc, _ = _TEMPLATES[name]
    return desc
