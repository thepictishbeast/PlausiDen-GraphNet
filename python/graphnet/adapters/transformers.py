"""HuggingFace transformers adapter — Phase 5.

Lazy-imports ``transformers``; raises AdapterError on load attempts if
the framework isn't installed.
"""

from __future__ import annotations

from typing import Any

from graphnet.adapters.base import AdapterError, ArchSummary


class TransformerAdapter:
    """Wrap any HuggingFace ``transformers`` model as a ModelAdapter.

    Forward accepts a token-ids tensor (or a string + tokenizer is wired
    automatically when constructed with ``tokenizer_name``). Output is the
    model's native output type (logits / generated tokens / hidden states).
    """

    family: str = "transformer"

    def __init__(
        self,
        model: Any,
        *,
        tokenizer: Any | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._notes = list(notes or [])

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        revision: str | None = None,
        notes: list[str] | None = None,
    ) -> TransformerAdapter:
        """Load a HuggingFace model + tokenizer by name (raises AdapterError
        if ``transformers`` isn't installed).
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise AdapterError(
                "transformers not installed; "
                "`pip install transformers` or `pip install plausiden-graphnet[adapters]`"
            ) from e
        try:
            model = AutoModel.from_pretrained(name, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        except Exception as e:  # pragma: no cover - network / model errors
            raise AdapterError(f"failed to load {name!r}: {e}") from e
        return cls(model, tokenizer=tokenizer, notes=notes or [f"from_pretrained({name!r})"])

    def forward(self, input_: Any) -> Any:
        """Run a forward pass.

        If ``input_`` is a string, tokenize it first (requires tokenizer).
        Otherwise pass through unchanged (caller has prepared tensors).
        """
        if isinstance(input_, str):
            if self._tokenizer is None:
                raise AdapterError("string input requires a tokenizer; pass tokens instead")
            inputs = self._tokenizer(input_, return_tensors="pt")
            return self._model(**inputs)
        return self._model(input_)

    def arch_summary(self) -> ArchSummary:
        # HuggingFace models expose .config with attributes that vary by family.
        # Pull what we can; the introspection layer can dig deeper if needed.
        config = getattr(self._model, "config", None)
        hidden = getattr(config, "hidden_size", None) if config else None
        n_layers = getattr(config, "num_hidden_layers", None) if config else None
        notes = list(self._notes)
        if config is not None:
            for attr in ("model_type", "vocab_size", "max_position_embeddings"):
                value = getattr(config, attr, None)
                if value is not None:
                    notes.append(f"{attr}={value}")
        return ArchSummary(
            family=self.family,
            input_dim=hidden,
            output_dim=hidden,
            substructures=n_layers or 0,
            notes=notes,
        )

    @property
    def underlying(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def __repr__(self) -> str:
        cls_name = type(self._model).__name__
        return f"TransformerAdapter({cls_name})"
