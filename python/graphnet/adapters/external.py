"""ExternalAdapter — bring-your-own-model via closure.

Mirror of graphnet-engine's ExternalModel Rust struct, but at the Python
level so users can wrap arbitrary Python forward functions (an HTTP-backed
inference server, their own LFI Python wrapper, a not-yet-supported
framework, anything callable that maps input → output) without writing
Rust.

For HDC-substrate models that produce :class:`graphnet.Hypervector`, this
slots cleanly into the rest of GraphNet's viz / intervention / snapshot
pipeline. For arbitrary input/output types, GraphNet treats the model as
opaque (forward works, but Stack-specific introspection doesn't apply).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graphnet.adapters.base import ArchSummary


class ExternalAdapter:
    """Wrap any callable as a ModelAdapter.

    Example:
        my_ai = ExternalAdapter(
            family="my-custom-lfi",
            input_dim=10_000,
            output_dim=10_000,
            forward_fn=lambda v: v,  # identity stub
            notes=["trained 2026-05-17"],
        )
    """

    def __init__(
        self,
        *,
        family: str,
        input_dim: int | None,
        output_dim: int | None,
        forward_fn: Callable[[Any], Any],
        substructures: int = 0,
        notes: list[str] | None = None,
    ) -> None:
        self._family = family
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._forward_fn = forward_fn
        self._substructures = substructures
        self._notes = list(notes or [])

    def forward(self, input_: Any) -> Any:
        return self._forward_fn(input_)

    def arch_summary(self) -> ArchSummary:
        return ArchSummary(
            family=self._family,
            input_dim=self._input_dim,
            output_dim=self._output_dim,
            substructures=self._substructures,
            notes=list(self._notes),
        )

    @property
    def family(self) -> str:
        return self._family

    def __repr__(self) -> str:
        return (
            f"ExternalAdapter(family={self._family!r}, "
            f"input_dim={self._input_dim}, output_dim={self._output_dim})"
        )
