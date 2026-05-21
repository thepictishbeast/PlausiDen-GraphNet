"""Base ModelAdapter protocol — uniform interface every adapter satisfies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class AdapterError(RuntimeError):
    """Raised when an adapter can't be loaded or invoked."""


@dataclass(frozen=True)
class ArchSummary:
    """Lightweight description of a model's architecture for viz + audit."""

    family: str
    input_dim: int | None  # None for variable-length text models
    output_dim: int | None
    substructures: int
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class ModelAdapter(Protocol):
    """Every GraphNet adapter satisfies this protocol.

    Implementations may operate on hypervectors (Stack, LFI, custom HDC),
    on tokens (transformers), on text (decoder LLMs), or on tensors
    (PyTorch nn.Module). The ``forward`` input/output types are adapter-
    specific — the protocol doesn't constrain them.

    What IS uniform:
    - :meth:`forward` runs one inference pass
    - :meth:`arch_summary` returns introspection metadata for the viz layer
    - :meth:`family` returns a short identifier for the adapter kind
    """

    def forward(self, input_: Any) -> Any:
        """Run one forward pass on the model."""

    def arch_summary(self) -> ArchSummary:
        """Return architectural metadata for the visualisation layer."""

    @property
    def family(self) -> str:
        """Short identifier of the adapter family (e.g. ``"transformer"``)."""


def available_adapters() -> dict[str, bool]:
    """Report which adapter families are usable in this environment.

    An adapter family is "available" if its underlying framework can be
    imported. Native adapters (``stack``, ``external``) are always available
    when graphnet is installed.
    """
    result: dict[str, bool] = {"stack": True, "external": True}
    for name, module in [
        ("transformers", "transformers"),
        ("mamba", "mamba_ssm"),
        ("torch", "torch"),
    ]:
        try:
            __import__(module)
            result[name] = True
        except ImportError:
            result[name] = False
    return result
