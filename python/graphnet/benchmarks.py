"""Benchmark library — Phase 11 / plan §21.4.

Synthetic tasks suitable for evaluating any GraphNet-loadable model
end-to-end without external dataset wrangling. Phase 11 ships the two
HDC-native benchmarks (associative recall + needle-in-haystack); larger
benchmarks (RULER, GLUE, ARC) require token-level adapters and land
later.

Every benchmark is a callable returning a :class:`BenchmarkResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkResult:
    """The result of running a benchmark."""

    name: str
    """Benchmark identifier (e.g. ``"associative-recall"``)."""
    accuracy: float
    """0.0-1.0 — fraction of trials the model got correct."""
    n_trials: int
    """How many trials were averaged."""
    notes: list[str]
    """Free-form annotations: hyperparameters, seeds, runtime, etc."""


def associative_recall(
    model: Any,
    *,
    n_pairs: int = 10,
    n_trials: int = 100,
    seed: int = 0,
    dim: int | None = None,
) -> BenchmarkResult:
    """Associative recall: memorize (key → value) pairs, recall value given key.

    Procedure:
    1. Generate ``n_pairs`` random (key, value) hypervector pairs.
    2. Bundle them all into one memory: ``M = bundle(k_1 ⊗ v_1, ..., k_n ⊗ v_n)``.
    3. For each trial, pick a random pair (k_i, v_i), query: ``v̂ = M ⊗ k_i``.
    4. Score: cleanup ``v̂`` against the codebook; success if argmax is v_i.

    This is the canonical HDC capacity test; for D=10_000 and n_pairs<=10
    a healthy Stack should achieve ~1.0 accuracy.

    Requires the native graphnet extension.
    """
    import graphnet

    if graphnet.Hypervector is None:
        raise RuntimeError("native graphnet not installed")

    d = dim if dim is not None else model.dim() if hasattr(model, "dim") else 10_000
    keys = [graphnet.Hypervector.random(d, seed * 1000 + i * 2) for i in range(n_pairs)]
    values = [graphnet.Hypervector.random(d, seed * 1000 + i * 2 + 1) for i in range(n_pairs)]

    # Build memory: M = bundle( bind(k_i, v_i) for i )
    bindings = [graphnet.bind(k, v) for k, v in zip(keys, values, strict=False)]
    memory = graphnet.bundle(bindings)

    # Run model forward on the memory to get the "stored" representation
    # (a passthrough model returns memory unchanged).
    stored = model.forward(memory)

    correct = 0
    for trial in range(n_trials):
        i = (seed + trial) % n_pairs
        # Query: v̂ = stored ⊗ k_i
        query = graphnet.unbind(stored, keys[i])
        # Cleanup: find the codebook value most similar to query.
        sims = [graphnet.cos_sim(query, v) for v in values]
        argmax = sims.index(max(sims))
        if argmax == i:
            correct += 1

    return BenchmarkResult(
        name="associative-recall",
        accuracy=correct / n_trials,
        n_trials=n_trials,
        notes=[f"n_pairs={n_pairs}", f"dim={d}", f"seed={seed}"],
    )


def needle_in_haystack(
    model: Any,
    *,
    haystack_size: int = 50,
    n_trials: int = 50,
    seed: int = 0,
    dim: int | None = None,
) -> BenchmarkResult:
    """Needle-in-haystack: long-context retrieval baseline.

    Procedure:
    1. Generate ``haystack_size`` random distractor hypervectors.
    2. Inject a "needle" hypervector at a random position (bundled into the
       haystack with a unique position-key).
    3. After model forward, recover the needle by querying with the
       position-key.
    4. Success if the recovered vector is most similar to the original needle.

    HDC-native version of the transformer-research benchmark; tests the
    Stack's ability to preserve fine-grained binding information through
    its operations.
    """
    import graphnet

    if graphnet.Hypervector is None:
        raise RuntimeError("native graphnet not installed")

    d = dim if dim is not None else model.dim() if hasattr(model, "dim") else 10_000

    correct = 0
    for trial in range(n_trials):
        base = seed * 10_000 + trial * 200
        # Build haystack + needle
        distractors = [graphnet.Hypervector.random(d, base + i) for i in range(haystack_size)]
        needle = graphnet.Hypervector.random(d, base + haystack_size + 1)
        position_keys = [
            graphnet.Hypervector.random(d, base + haystack_size + 2 + i)
            for i in range(haystack_size + 1)
        ]
        needle_pos = trial % (haystack_size + 1)

        all_items = [*distractors[:needle_pos], needle, *distractors[needle_pos:]]
        bindings = [
            graphnet.bind(k, item)
            for k, item in zip(position_keys, all_items, strict=False)
        ]
        memory = graphnet.bundle(bindings)

        retrieved = model.forward(memory)
        query = graphnet.unbind(retrieved, position_keys[needle_pos])

        sim_to_needle = graphnet.cos_sim(query, needle)
        max_distractor_sim = max(graphnet.cos_sim(query, d_) for d_ in distractors)
        if sim_to_needle > max_distractor_sim:
            correct += 1

    return BenchmarkResult(
        name="needle-in-haystack",
        accuracy=correct / n_trials,
        n_trials=n_trials,
        notes=[f"haystack_size={haystack_size}", f"dim={d}", f"seed={seed}"],
    )


def list_benchmarks() -> list[tuple[str, str]]:
    """List available benchmarks with one-line descriptions."""
    return [
        (
            "associative-recall",
            "memorize N (key, value) pairs, recall value given key (HDC capacity test)",
        ),
        (
            "needle-in-haystack",
            "find one injected hypervector among N distractors (long-context retrieval)",
        ),
    ]
