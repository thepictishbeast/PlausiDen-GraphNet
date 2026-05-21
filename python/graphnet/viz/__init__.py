"""GraphNet visualisation layer.

Phase 3 (2D + Jupyter HTML) lives here. Phase 4 (3D rotatable) lands in
``plot_3d``. Phase 17 (Maths Panel) lands in ``maths``.

Public surface:

- :func:`plot_2d.hypervector_heatmap` — Hypervector → matplotlib heatmap
- :func:`plot_2d.forward_trace_heatmap` — ForwardTrace → stacked heatmap
- :func:`plot_2d.stack_graph` — Stack → graphviz architecture diagram
- :func:`plot_2d.similarity_matrix` — pairwise cos_sim heatmap
- :func:`widgets.stack_repr_html` — HTML summary of a Stack
- :func:`widgets.forward_trace_repr_html` — HTML summary of a ForwardTrace
- :func:`widgets.register_jupyter_reprs` — wire HTML reprs into IPython

On import, attempts to wire IPython HTML reprs automatically (no-op outside
IPython).
"""

from __future__ import annotations

from graphnet.viz import plot_2d, widgets
from graphnet.viz.plot_2d import (
    available_backends,
    forward_trace_heatmap,
    hypervector_heatmap,
    similarity_matrix,
    stack_graph,
)
from graphnet.viz.widgets import (
    forward_trace_repr_html,
    register_jupyter_reprs,
    stack_repr_html,
)

# Auto-wire IPython HTML reprs at import time when in a notebook.
register_jupyter_reprs()

__all__ = [
    "available_backends",
    "forward_trace_heatmap",
    "forward_trace_repr_html",
    "hypervector_heatmap",
    "plot_2d",
    "register_jupyter_reprs",
    "similarity_matrix",
    "stack_graph",
    "stack_repr_html",
    "widgets",
]
