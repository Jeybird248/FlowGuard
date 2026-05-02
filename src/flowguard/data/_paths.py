"""Resolve dataset directories under ``$FLOWGUARD_DATA_DIR``."""
from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    root = os.environ.get("FLOWGUARD_DATA_DIR")
    if not root:
        raise RuntimeError(
            "FLOWGUARD_DATA_DIR is not set; source env.sh or export it explicitly."
        )
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_dir(name: str) -> Path:
    p = data_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p
