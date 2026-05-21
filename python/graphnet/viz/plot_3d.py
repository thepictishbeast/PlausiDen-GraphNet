"""3D rotatable interactive visualisation — Phase 4.

Owner requirement: "have graphical representations of the AI that's editable
and rotatable and where you can select inputs and outputs."

Phase 4 ships plotly 3D-based views; pure-WebGL k3d-jupyter backend lands
later for >10k-node Stacks.

Functions return ``plotly.graph_objects.Figure`` objects so callers can
``.show()`` (auto-renders inline in Jupyter with full orbit/pan/zoom/click)
or ``.write_html(path)`` to save.

All functions lazily import plotly + numpy + scikit-learn so a clean install
without them can still ``import graphnet.viz``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def hypervector_3d_scatter(
    vectors: list[Any],
    *,
    labels: list[str] | None = None,
    title: str | None = None,
    method: str = "pca",
) -> Any:
    """Project hypervectors to 3D via PCA / t-SNE / UMAP and scatter-plot.

    For high-dimensional bipolar hypervectors, PCA gives a deterministic
    projection that preserves global structure; t-SNE/UMAP give more
    cluster-faithful but stochastic projections.

    Returns a plotly Figure; render with ``.show()``.

    Requires plotly + scikit-learn + numpy. ``method="umap"`` additionally
    requires umap-learn.
    """
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    n = len(vectors)
    if n < 2:
        raise ValueError(f"need at least 2 vectors for 3D projection, got {n}")
    if any(v.dim() != vectors[0].dim() for v in vectors[1:]):
        raise ValueError("all hypervectors must share the same dim")

    matrix = np.array([v.as_list() for v in vectors], dtype=np.float64)
    method_lower = method.lower()

    if method_lower == "pca":
        coords = PCA(n_components=3).fit_transform(matrix)
        method_label = "PCA"
    elif method_lower == "tsne":
        from sklearn.manifold import TSNE

        perp = min(30, max(2, n - 1))
        coords = TSNE(n_components=3, perplexity=perp).fit_transform(matrix)
        method_label = "t-SNE"
    elif method_lower == "umap":
        import umap

        coords = umap.UMAP(n_components=3, n_neighbors=min(15, n - 1)).fit_transform(matrix)
        method_label = "UMAP"
    else:
        raise ValueError(f"unknown method: {method!r} (use 'pca', 'tsne', or 'umap')")

    labels = labels or [f"v{i}" for i in range(n)]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"size": 6, "color": list(range(n)), "colorscale": "Viridis"},
                hovertext=labels,
            )
        ]
    )
    fig.update_layout(
        title=title or f"{method_label} projection of {n} hypervectors",
        scene={
            "xaxis_title": f"{method_label}-1",
            "yaxis_title": f"{method_label}-2",
            "zaxis_title": f"{method_label}-3",
        },
        height=600,
    )
    return fig


def stack_graph_3d(
    stack: Any,
    *,
    title: str | None = None,
) -> Any:
    """Render a Stack as a 3D plotly architecture diagram.

    Operations are placed on a horizontal ring around the central bundle
    node; input + output are positioned above and below the bundle. The
    plot is fully interactive in Jupyter (orbit / pan / zoom / hover).

    Click-to-select node detail panels will land in a follow-up tick
    (currently the click target only logs the node id).
    """
    import math

    import plotly.graph_objects as go

    tags = stack.op_tags()
    n_ops = len(tags)

    palette = {
        "identity": "#7286D3",
        "dense": "#3F7D58",
        "hrr_bind": "#BC1823",
    }

    # Positions: input at y=+2, ops on a horizontal ring at y=0, bundle at
    # y=-1, output at y=-2.
    nodes_x = [0.0]
    nodes_y = [2.0]
    nodes_z = [0.0]
    node_labels = ["Input"]
    node_colours = ["#a8d5e2"]

    radius = max(1.0, n_ops * 0.35)
    for i, tag in enumerate(tags):
        angle = (2 * math.pi * i) / max(1, n_ops)
        nodes_x.append(radius * math.cos(angle))
        nodes_z.append(radius * math.sin(angle))
        nodes_y.append(0.0)
        node_labels.append(f"{tag}[{i}]")
        node_colours.append(palette.get(tag, "#cccccc"))

    nodes_x.append(0.0)
    nodes_y.append(-1.0)
    nodes_z.append(0.0)
    node_labels.append("Bundle")
    node_colours.append("#f9c74f")

    nodes_x.append(0.0)
    nodes_y.append(-2.0)
    nodes_z.append(0.0)
    node_labels.append("Output")
    node_colours.append("#a8d5e2")

    # Edges: input → each op, each op → bundle, bundle → output.
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    input_idx = 0
    bundle_idx = 1 + n_ops
    output_idx = 2 + n_ops
    for op_idx in range(1, 1 + n_ops):
        # input -> op
        edge_x.extend([nodes_x[input_idx], nodes_x[op_idx], None])
        edge_y.extend([nodes_y[input_idx], nodes_y[op_idx], None])
        edge_z.extend([nodes_z[input_idx], nodes_z[op_idx], None])
        # op -> bundle
        edge_x.extend([nodes_x[op_idx], nodes_x[bundle_idx], None])
        edge_y.extend([nodes_y[op_idx], nodes_y[bundle_idx], None])
        edge_z.extend([nodes_z[op_idx], nodes_z[bundle_idx], None])
    # bundle -> output
    edge_x.extend([nodes_x[bundle_idx], nodes_x[output_idx], None])
    edge_y.extend([nodes_y[bundle_idx], nodes_y[output_idx], None])
    edge_z.extend([nodes_z[bundle_idx], nodes_z[output_idx], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={"color": "#888888", "width": 2},
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=nodes_x,
            y=nodes_y,
            z=nodes_z,
            mode="markers+text",
            marker={"size": 14, "color": node_colours, "line": {"width": 1, "color": "#333"}},
            text=node_labels,
            textposition="top center",
            hovertext=node_labels,
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title or f"Stack(dim={stack.dim()}, ops={n_ops}) — 3D architecture",
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        height=600,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def forward_trace_3d(
    trace: Any,
    *,
    title: str | None = None,
    method: str = "pca",
) -> Any:
    """Visualise a ForwardTrace as a 3D scatter showing the input + each
    per-op output + the bundled output, all projected into the same 3D
    space via PCA so the cascade trajectory is visible.
    """
    layers = [("input", trace.input)] + [
        (f"{o.tag}[{o.index}]", o.output) for o in trace.per_op
    ] + [("bundled", trace.bundled)]
    labels, vectors = zip(*layers, strict=False)
    return hypervector_3d_scatter(
        list(vectors),
        labels=list(labels),
        title=title or "ForwardTrace 3D trajectory",
        method=method,
    )
