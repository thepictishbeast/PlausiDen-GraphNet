"""Pytest configuration for GraphNet tests.

Prepends `python/` to sys.path so tests can import `graphnet` without first
installing the package via maturin. When the package IS installed (e.g. in CI
after `maturin develop`), this is a no-op because the installed location
shadows the source tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_SRC = _PROJECT_ROOT / "python"

if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))
