"""2D visualisation for GraphNet — matplotlib + plotly + graphviz.

Phase 3 surface:

- ``hypervector_heatmap(v, ...)``      — matplotlib heatmap of a Hypervector
- ``forward_trace_heatmap(trace)``    — stacked heatmap of per-op outputs
- ``stack_graph(stack, ...)``          — graphviz/networkx 2D architecture graph
- ``similarity_matrix(vectors, ...)``  — pairwise cos_sim matrix viz

All functions return Matplotlib / Plotly objects so callers can ``.show()``
them in Jupyter or save them to disk. None of these functions render on
their own — that's the caller's responsibility.

Heavy deps (matplotlib / plotly / networkx / graphviz) are imported lazily
so a clean Python install without them can still ``import graphnet.viz``
and discover what isn't available via :func:`available_backends`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def available_backends() -> dict[str, bool]:
    """Return a map of viz backend → whether it's importable."""
    backends: dict[str, bool] = {}
    for name in ("matplotlib", "plotly", "networkx", "graphviz", "numpy"):
        try:
            __import__(name)
            backends[name] = True
        except ImportError:
            backends[name] = False
    return backends


def hypervector_heatmap(
    v: Any,
    *,
    width: int = 100,
    cmap: str = "RdBu",
    title: str | None = None,
) -> Any:
    """Render a Hypervector as a 2D matplotlib heatmap.

    Reshapes the bipolar vector into a roughly-square grid (``width`` columns)
    for visual inspection. Useful for spotting structure / clusters.

    Returns a matplotlib Figure; caller invokes ``.show()`` or ``.savefig()``.

    Raises ``ImportError`` if matplotlib isn't installed.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.asarray(v.as_list(), dtype=np.int8)
    pad = (-len(data)) % width
    if pad:
        data = np.concatenate([data, np.zeros(pad, dtype=np.int8)])
    grid = data.reshape(-1, width)

    fig, ax = plt.subplots(figsize=(min(12, width * 0.1), min(12, grid.shape[0] * 0.1)))
    im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xlabel(f"col (width={width})")
    ax.set_ylabel("row")
    ax.set_title(title or f"Hypervector (dim={v.dim()})")
    fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], label="bipolar")
    fig.tight_layout()
    return fig


def forward_trace_heatmap(
    trace: Any,
    *,
    width: int = 100,
    cmap: str = "RdBu",
) -> Any:
    """Render a ForwardTrace as a vertical stack of per-op heatmaps.

    Shows the input row + every per-op output + the final bundled output,
    so the user can scan the cascade visually.

    Raises ``ImportError`` if matplotlib isn't installed.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    layers = [("input", trace.input)] + [
        (f"{o.tag}[{o.index}]", o.output) for o in trace.per_op
    ] + [("bundled", trace.bundled)]

    n = len(layers)
    fig, axes = plt.subplots(n, 1, figsize=(min(14, width * 0.12), 1.5 * n), squeeze=False)
    for ax, (label, vec) in zip(axes[:, 0], layers, strict=False):
        data = np.asarray(vec.as_list(), dtype=np.int8)
        pad = (-len(data)) % width
        if pad:
            data = np.concatenate([data, np.zeros(pad, dtype=np.int8)])
        grid = data.reshape(-1, width)
        ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1, 0].set_xlabel(f"width={width}")
    fig.suptitle(f"ForwardTrace ({len(trace.per_op)} ops)")
    fig.tight_layout()
    return fig


def stack_graph(
    stack: Any,
    *,
    rankdir: str = "TB",
    fmt: str = "svg",
) -> Any:
    """Render a Stack as a 2D graphviz architecture diagram.

    Returns a ``graphviz.Digraph`` whose ``.render()`` produces SVG/PNG/PDF.
    In Jupyter, displaying it renders inline.

    Raises ``ImportError`` if graphviz isn't installed.
    """
    import graphviz

    dot = graphviz.Digraph(
        comment=f"Stack(dim={stack.dim()})",
        graph_attr={"rankdir": rankdir, "bgcolor": "transparent"},
        node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica"},
        edge_attr={"color": "#555555"},
        format=fmt,
    )
    dot.node("input", f"Input\\n(D={stack.dim()})", fillcolor="#a8d5e2")
    dot.node("bundle", "Bundle\\n(majority)", fillcolor="#f9c74f")
    dot.node("output", f"Output\\n(D={stack.dim()})", fillcolor="#a8d5e2")

    tags = stack.op_tags()
    palette = {
        "identity": "#cfe2f3",
        "dense": "#d9ead3",
        "hrr_bind": "#f4cccc",
    }
    for i, tag in enumerate(tags):
        node_id = f"op{i}"
        colour = palette.get(tag, "#eeeeee")
        dot.node(node_id, f"{tag}\\n[{i}]", fillcolor=colour)
        dot.edge("input", node_id)
        dot.edge(node_id, "bundle")
    dot.edge("bundle", "output")
    return dot


def similarity_matrix(
    vectors: list[Any],
    *,
    labels: list[str] | None = None,
    title: str | None = None,
) -> Any:
    """Render a pairwise cosine-similarity matrix as a heatmap.

    Requires the native graphnet extension (uses :func:`graphnet.cos_sim`).
    Raises ``ImportError`` if matplotlib isn't installed.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from graphnet import cos_sim

    if cos_sim is None:
        raise RuntimeError("native graphnet not available; cannot compute cos_sim")

    n = len(vectors)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cos_sim(vectors[i], vectors[j])

    labels = labels or [f"v{i}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(0.5 * n + 2, 0.5 * n + 2))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title or "Pairwise cosine similarity")
    fig.colorbar(im, ax=ax, label="cos_sim")
    fig.tight_layout()
    return fig
