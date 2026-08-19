import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "config"))

from bootstrap import setup_paths  # noqa: E402

setup_paths()
