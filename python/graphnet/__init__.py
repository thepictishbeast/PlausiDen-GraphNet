"""GraphNet — live REPL + graphical environment for neural-network work.

This Python facade re-exports the PyO3-built native extension as the
`graphnet` namespace. The native module ships from `crates/graphnet-bindings`
via maturin; until the native module is installed the facade raises
NotImplementedError for native calls but still exposes `__version__`.

See ``docs/PLAN.md`` (project root) for the full design plan.
"""

from __future__ import annotations

__version__ = "0.1.0"

try:
    from graphnet._graphnet_native import (  # type: ignore[import-not-found]
        ForwardTrace,
        Hypervector,
        Operation,
        OperationOutput,
        Stack,
        bind,
        bundle,
        cos_sim,
        hamming,
        negate,
        permute,
        restore,
        snapshot,
        stack_from_yaml,
        stack_to_yaml,
        unbind,
    )
    from graphnet._graphnet_native import (
        banner as _native_banner,
    )
    from graphnet._graphnet_native import (
        version as _native_version,
    )

    _NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when native lib is missing
    _NATIVE_AVAILABLE = False
    Hypervector = None  # type: ignore[assignment, misc]
    Operation = None  # type: ignore[assignment, misc]
    Stack = None  # type: ignore[assignment, misc]
    ForwardTrace = None  # type: ignore[assignment, misc]
    OperationOutput = None  # type: ignore[assignment, misc]
    bind = None  # type: ignore[assignment]
    unbind = None  # type: ignore[assignment]
    bundle = None  # type: ignore[assignment]
    cos_sim = None  # type: ignore[assignment]
    hamming = None  # type: ignore[assignment]
    permute = None  # type: ignore[assignment]
    negate = None  # type: ignore[assignment]
    snapshot = None  # type: ignore[assignment]
    restore = None  # type: ignore[assignment]
    stack_to_yaml = None  # type: ignore[assignment]
    stack_from_yaml = None  # type: ignore[assignment]


def banner() -> str:
    """Return the engine banner string."""
    if not _NATIVE_AVAILABLE:
        raise NotImplementedError(
            "graphnet native extension not installed; "
            "build with `maturin develop` or `pip install -e .`"
        )
    return _native_banner()


def version() -> str:
    """Return the engine version string."""
    if not _NATIVE_AVAILABLE:
        return __version__
    return _native_version()


def native_available() -> bool:
    """Return True if the PyO3-built native extension is importable."""
    return _NATIVE_AVAILABLE


# Phase 11 rolling features: gallery + benchmarks + share are pure-Python
# and safe to import unconditionally (their entry points raise
# NotImplementedError when the native extension is needed).
from graphnet import advisor, benchmarks, gallery, maths, nodes, share  # noqa: E402

__all__ = [
    "ForwardTrace",
    "Hypervector",
    "Operation",
    "OperationOutput",
    "Stack",
    "__version__",
    "advisor",
    "banner",
    "benchmarks",
    "bind",
    "bundle",
    "cos_sim",
    "gallery",
    "hamming",
    "maths",
    "native_available",
    "negate",
    "nodes",
    "permute",
    "restore",
    "share",
    "snapshot",
    "stack_from_yaml",
    "stack_to_yaml",
    "unbind",
    "version",
]
