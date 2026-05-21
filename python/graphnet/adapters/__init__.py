"""Multi-architecture adapters for GraphNet — Phase 5.

GraphNet treats every model behind a uniform :class:`ModelAdapter` protocol.
v1 ships adapters for:

- :mod:`graphnet.adapters.stack` — PlausiDen Stack (native, always available)
- :mod:`graphnet.adapters.transformers` — HuggingFace `transformers` models
- :mod:`graphnet.adapters.mamba` — `mamba_ssm` state-space models
- :mod:`graphnet.adapters.torch` — generic PyTorch `nn.Module`
- :mod:`graphnet.adapters.external` — bring-your-own-model via closure

All heavy framework imports are lazy; a fresh install without
transformers / mamba_ssm / torch can still ``import graphnet.adapters``.
:func:`available_adapters` reports which adapters are usable.
"""

from __future__ import annotations

from graphnet.adapters.base import (
    AdapterError,
    ArchSummary,
    ModelAdapter,
    available_adapters,
)
from graphnet.adapters.external import ExternalAdapter
from graphnet.adapters.stack import StackAdapter

__all__ = [
    "AdapterError",
    "ArchSummary",
    "ExternalAdapter",
    "ModelAdapter",
    "StackAdapter",
    "available_adapters",
]
