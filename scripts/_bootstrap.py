"""Minimal path setup so ``scripts/*.py`` can ``import src.*`` when run
directly (``python scripts/01_build_features.py``)."""
from __future__ import annotations

import sys
from pathlib import Path


def setup() -> Path:
    here = Path(__file__).resolve().parent
    repo = here.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo
