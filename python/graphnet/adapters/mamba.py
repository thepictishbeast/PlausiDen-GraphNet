"""Mamba state-space model adapter — Phase 5.

Lazy-imports ``mamba_ssm``; raises AdapterError on load attempts if the
framework isn't installed.
"""

from __future__ import annotations

from typing import Any

from graphnet.adapters.base import AdapterError, ArchSummary


class MambaAdapter:
    """Wrap a ``mamba_ssm`` model as a ModelAdapter."""

    family: str = "mamba"

    def __init__(self, model: Any, *, notes: list[str] | None = None) -> None:
        self._model = model
        self._notes = list(notes or [])

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        notes: list[str] | None = None,
    ) -> MambaAdapter:
        """Load a Mamba model by name from HuggingFace hub via mamba_ssm."""
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        except ImportError as e:
            raise AdapterError(
                "mamba_ssm not installed; "
                "`pip install mamba-ssm` (requires CUDA + PyTorch)"
            ) from e
        try:
            model = MambaLMHeadModel.from_pretrained(name)
        except Exception as e:  # pragma: no cover
            raise AdapterError(f"failed to load Mamba {name!r}: {e}") from e
        return cls(model, notes=notes or [f"from_pretrained({name!r})"])

    def forward(self, input_: Any) -> Any:
        """Run a forward pass; input is a tensor of token ids."""
        return self._model(input_)

    def arch_summary(self) -> ArchSummary:
        config = getattr(self._model, "config", None)
        d_model = getattr(config, "d_model", None) if config else None
        n_layer = getattr(config, "n_layer", None) if config else None
        notes = list(self._notes)
        if config is not None:
            for attr in ("vocab_size", "ssm_cfg"):
                value = getattr(config, attr, None)
                if value is not None:
                    notes.append(f"{attr}={value}")
        return ArchSummary(
            family=self.family,
            input_dim=d_model,
            output_dim=d_model,
            substructures=n_layer or 0,
            notes=notes,
        )

    @property
    def underlying(self) -> Any:
        return self._model

    def __repr__(self) -> str:
        return f"MambaAdapter({type(self._model).__name__})"
