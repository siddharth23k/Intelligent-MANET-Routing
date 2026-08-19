"""Single place that fixes up sys.path for every entry point in this repo.

The project is a collection of scripts rather than an installed package, so each
entry point needs the same import roots. Doing it in one module keeps the paths
consistent and means a new script cannot accidentally import a stale copy of a
module from a different directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_IMPORT_ROOTS = [
    ROOT,
    ROOT / "config",
    ROOT / "methods" / "common",
    ROOT / "methods" / "ours",
    ROOT / "methods" / "baseline",
    ROOT / "methods" / "eval",
]


def setup_paths() -> Path:
    """Prepend every project import root to sys.path. Returns the repo root."""
    for p in reversed(_IMPORT_ROOTS):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    return ROOT
