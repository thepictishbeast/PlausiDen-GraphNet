"""GraphNet — live REPL + graphical environment for neural-network work.

This Python facade re-exports the PyO3-built native extension as the
`graphnet` namespace. The native module ships from `crates/graphnet-bindings`
via maturin; until Phase 2 lands the native module may not be installed in
which case stubs raise NotImplementedError.

See ``docs/PLAN.md`` (project root) for the full design plan.
"""

from __future__ import annotations

__version__ = "0.1.0"

try:
    from graphnet._graphnet_native import (  # type: ignore[import-not-found]
        banner as _native_banner,
        version as _native_version,
    )

    _NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when native lib is missing
    _NATIVE_AVAILABLE = False


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


__all__ = ["__version__", "banner", "version"]
