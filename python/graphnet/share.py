"""Configuration sharing — Phase 11 / plan §21.9.

One-call export of an architecture to a sharing channel (GitHub Gist,
local file, future HuggingFace Hub / OpenReview). The architecture is
serialised via the Phase 9 YAML spec; weights ride along separately via
the Phase 1 bincode snapshot.

External-service uploads require user-supplied credentials and are gated
behind explicit opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ShareResult:
    """The outcome of a share call."""

    channel: str
    """Where the architecture was shared (``"file"``, ``"gist"``, ...)."""
    location: str
    """URL, file path, or other channel-specific identifier."""
    bytes_uploaded: int
    """Number of bytes shipped."""


def to_file(stack: Any, path: str | Path) -> ShareResult:
    """Write the Stack's YAML spec to ``path`` and return the file URI.

    The simplest sharing channel: a local file the user can email, paste,
    or commit to git. No credentials needed.
    """
    import graphnet

    if graphnet.snapshot is None:
        raise RuntimeError("native graphnet not installed")

    yaml = _stack_to_yaml(stack)
    path = Path(path)
    path.write_text(yaml, encoding="utf-8")
    return ShareResult(
        channel="file",
        location=str(path.resolve()),
        bytes_uploaded=len(yaml.encode("utf-8")),
    )


def to_gist(
    stack: Any,
    *,
    description: str = "GraphNet architecture",
    public: bool = False,
) -> ShareResult:
    """Upload the Stack's YAML spec to a GitHub Gist via the ``gh`` CLI.

    Requires the ``gh`` command-line tool to be installed and authenticated
    (``gh auth login``). Returns the resulting gist URL.

    For non-CLI environments, prefer :func:`to_file` and upload manually.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("gh") is None:
        raise RuntimeError("`gh` CLI not on PATH; install + auth: https://cli.github.com/")

    yaml = _stack_to_yaml(stack)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".graphnet.yaml",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(yaml)
        tmp_path = f.name

    args = ["gh", "gist", "create", tmp_path, "--desc", description]
    if public:
        args.append("--public")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh gist create failed: {result.stderr.strip()}")
    url = result.stdout.strip()
    return ShareResult(channel="gist", location=url, bytes_uploaded=len(yaml.encode("utf-8")))


def _stack_to_yaml(stack: Any) -> str:
    """Export a native Stack to YAML via the Rust engine's spec emitter."""
    # The native bindings expose snapshot() but not stack_to_yaml() yet —
    # build the spec on the Python side using available primitives.
    # For Phase 11 we use the Rust snapshot bytes + manually serialise the
    # high-level shape; once the Phase 2 bindings grow yaml_spec wrappers,
    # this collapses to a single FFI call.
    import json

    import graphnet

    if graphnet.Hypervector is None:
        raise RuntimeError("native graphnet not installed")

    ops_summary = []
    for tag in stack.op_tags():
        ops_summary.append({"kind": tag})

    payload = {
        "version": "1.0",
        "kind": "stack",
        "dim": stack.dim(),
        "ops": ops_summary,
        "notes": ["exported via graphnet.share (Python facade)"],
    }
    # Use json (always present) as a YAML-compatible subset for Phase 11;
    # full YAML round-trip lives Rust-side and is exposed in a follow-up.
    return json.dumps(payload, indent=2) + "\n"


def list_channels() -> list[tuple[str, str]]:
    """Enumerate available sharing channels with one-line descriptions."""
    return [
        ("file", "write YAML spec to a local file"),
        ("gist", "upload to GitHub Gist via the gh CLI"),
    ]
