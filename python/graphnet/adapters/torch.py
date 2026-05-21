"""Generic PyTorch nn.Module adapter — Phase 5.

Wraps any ``torch.nn.Module`` as a ModelAdapter. For models without
HuggingFace-style introspection metadata, the arch summary is minimal —
just the module's class name and parameter count.
"""

from __future__ import annotations

from typing import Any

from graphnet.adapters.base import AdapterError, ArchSummary


class PytorchAdapter:
    """Wrap any ``torch.nn.Module`` as a ModelAdapter."""

    family: str = "pytorch"

    def __init__(
        self,
        module: Any,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        notes: list[str] | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as e:
            raise AdapterError("torch not installed; `pip install torch`") from e

        if not isinstance(module, torch.nn.Module):
            raise AdapterError(
                f"expected torch.nn.Module, got {type(module).__name__}"
            )

        self._module = module
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._notes = list(notes or [])
        self._param_count = sum(p.numel() for p in module.parameters())
        self._submodule_count = sum(1 for _ in module.modules()) - 1

    def forward(self, input_: Any) -> Any:
        return self._module(input_)

    def arch_summary(self) -> ArchSummary:
        notes = list(self._notes)
        notes.append(f"class={type(self._module).__name__}")
        notes.append(f"params={self._param_count}")
        return ArchSummary(
            family=self.family,
            input_dim=self._input_dim,
            output_dim=self._output_dim,
            substructures=self._submodule_count,
            notes=notes,
        )

    @property
    def underlying(self) -> Any:
        return self._module

    def __repr__(self) -> str:
        return (
            f"PytorchAdapter({type(self._module).__name__}, "
            f"params={self._param_count})"
        )
