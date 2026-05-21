"""StackAdapter — wraps a native PlausiDen Stack as a ModelAdapter.

Always available; requires no external framework.
"""

from __future__ import annotations

from typing import Any

from graphnet.adapters.base import AdapterError, ArchSummary


class StackAdapter:
    """Adapter around a native :class:`graphnet.Stack`."""

    family: str = "stack"

    def __init__(self, stack: Any) -> None:
        try:
            import graphnet as _gn
        except ImportError as e:  # pragma: no cover
            raise AdapterError("graphnet not importable") from e
        if _gn.Stack is None:
            raise AdapterError(
                "graphnet native extension not installed; "
                "run `pip install -e .` to build it"
            )
        if not isinstance(stack, _gn.Stack):
            raise AdapterError(f"expected graphnet.Stack, got {type(stack).__name__}")
        self._stack = stack

    def forward(self, input_: Any) -> Any:
        """Forward via the Stack; input must be a graphnet.Hypervector."""
        return self._stack.forward(input_)

    def forward_with_trace(self, input_: Any) -> Any:
        """Forward + capture per-op activations."""
        return self._stack.forward_with_trace(input_)

    def arch_summary(self) -> ArchSummary:
        return ArchSummary(
            family=self.family,
            input_dim=self._stack.dim(),
            output_dim=self._stack.dim(),
            substructures=len(self._stack),
            notes=[f"ops: [{', '.join(self._stack.op_tags())}]"],
        )

    @property
    def underlying(self) -> Any:
        """Direct access to the wrapped Stack for advanced use cases."""
        return self._stack

    def __repr__(self) -> str:
        return f"StackAdapter({self._stack!r})"
