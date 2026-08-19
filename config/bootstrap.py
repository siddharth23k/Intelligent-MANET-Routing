"""Import path setup shared by every entry point.

The project is a set of scripts, not an installed package, so each entry point
needs the same roots on sys.path. Doing it in one module keeps them consistent.
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
    """Prepend the project import roots to sys.path and return the repo root."""
    for path in reversed(_IMPORT_ROOTS):
        entry = str(path)
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    return ROOT
