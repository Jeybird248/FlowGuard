"""IO helpers for jsonl traces and metrics.json artifacts."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Write a metrics.json with a stable schema.

    The ``timestamp`` and ``schema_version`` fields are added if missing so
    aggregate_results.py has a fixed set of header columns.
    """
    payload = dict(metrics)
    payload.setdefault("schema_version", 1)
    payload.setdefault("timestamp", int(time.time()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def features_path(features_root: str | Path,
                  model_key: str, split: str, seed: int,
                  decoding_step: int = 1) -> Path:
    """Canonical layout for cached FlowVectors.

    {features_root}/{model_key}/k{decoding_step}/{split}_seed{seed}.jsonl
    """
    return Path(features_root) / model_key / f"k{decoding_step}" / f"{split}_seed{seed}.jsonl"
