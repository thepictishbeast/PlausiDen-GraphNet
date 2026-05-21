"""HTML / Jupyter-widget helpers for GraphNet objects.

Provides :func:`stack_repr_html` and :func:`forward_trace_repr_html` that
render compact rich-HTML summaries of a Stack / ForwardTrace, suitable for
``_repr_html_`` integration in Jupyter / IPython.

These functions don't require any heavy deps; they emit plain HTML strings.
Pair them with the visualisations in :mod:`graphnet.viz.plot_2d` for richer
inline displays.
"""

from __future__ import annotations

from typing import Any


def _badge(text: str, colour: str) -> str:
    return (
        f'<span style="display:inline-block; padding:2px 8px; '
        f'border-radius:10px; background:{colour}; color:#fff; '
        f'font-family:Helvetica,Arial,sans-serif; font-size:11px; '
        f'margin-right:4px">{text}</span>'
    )


def _op_palette(tag: str) -> str:
    return {
        "identity": "#7286D3",
        "dense": "#3F7D58",
        "hrr_bind": "#BC1823",
    }.get(tag, "#666")


def stack_repr_html(stack: Any) -> str:
    """Render a Stack as a compact HTML summary.

    Inline badges per operation; counts + dim header. Safe to embed in
    Jupyter notebooks without any extra dependencies.
    """
    tags = stack.op_tags()
    badges = "".join(_badge(f"{i}: {tag}", _op_palette(tag)) for i, tag in enumerate(tags))
    if not badges:
        badges = '<span style="color:#999; font-style:italic">no operations yet</span>'

    return (
        f'<div style="font-family:Helvetica,Arial,sans-serif; '
        f'border:1px solid #ddd; border-radius:6px; padding:10px; '
        f'background:#fafafa; max-width:600px">'
        f'<div style="font-weight:600; margin-bottom:6px">'
        f'Stack <span style="color:#888">'
        f'(dim={stack.dim()}, ops={len(stack)})</span>'
        f"</div>"
        f'<div>{badges}</div>'
        f"</div>"
    )


def forward_trace_repr_html(trace: Any) -> str:
    """Render a ForwardTrace as a compact HTML summary table."""
    rows = ['<tr><th align="left">step</th><th align="left">tag</th><th align="left">dim</th></tr>']
    rows.append(
        f'<tr><td>input</td><td><em>—</em></td><td>{trace.input.dim()}</td></tr>'
    )
    for o in trace.per_op:
        rows.append(
            f'<tr><td>{o.index}</td><td>{_badge(o.tag, _op_palette(o.tag))}</td>'
            f'<td>{o.output.dim()}</td></tr>'
        )
    rows.append(
        f'<tr><td><b>bundled</b></td><td><em>—</em></td>'
        f'<td>{trace.bundled.dim()}</td></tr>'
    )
    table = (
        f'<table style="border-collapse:collapse; font-family:Helvetica,Arial,sans-serif; '
        f'font-size:12px; margin-top:6px">{"".join(rows)}</table>'
    )
    return (
        f'<div style="border:1px solid #ddd; border-radius:6px; padding:10px; '
        f'background:#fafafa; max-width:600px">'
        f'<div style="font-weight:600; font-family:Helvetica,Arial,sans-serif">'
        f'ForwardTrace <span style="color:#888">(ops={len(trace.per_op)})</span></div>'
        f"{table}"
        f"</div>"
    )


def register_jupyter_reprs() -> None:
    """Install ``_repr_html_`` on the native PyO3 classes for Jupyter.

    PyO3 classes can't be subclassed in Python, but IPython's display system
    checks for an ``_repr_html_`` attribute on the instance OR a registered
    formatter. We use the formatter approach.

    Called once at import-time by :mod:`graphnet.viz`; safe to call again
    (no-op if already registered).
    """
    try:
        from IPython import get_ipython
        from IPython.core.formatters import HTMLFormatter

        from graphnet import ForwardTrace, Stack

        ip = get_ipython()
        if ip is None:
            return
        fmt: HTMLFormatter = ip.display_formatter.formatters["text/html"]
        if Stack is not None:
            fmt.for_type(Stack, stack_repr_html)
        if ForwardTrace is not None:
            fmt.for_type(ForwardTrace, forward_trace_repr_html)
    except Exception:  # pragma: no cover - IPython optional
        return
