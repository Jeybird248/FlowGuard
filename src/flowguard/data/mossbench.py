"""MOSSBench loader.

MOSSBench tests over-refusal on benign-but-sensitive inputs (Li et al. 2024).
Expected layout under ``$FLOWGUARD_DATA_DIR/mossbench/``:
    metadata.json    # list of {id, image, query, label, category}
    images/<filename>
"""
from __future__ import annotations

import json
from typing import Iterator

from PIL import Image

from flowguard.data.base import DatasetSplit, Sample, register
from flowguard.data._paths import dataset_dir


def _iter_mossbench() -> Iterator[Sample]:
    root = dataset_dir("mossbench")
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"MOSSBench metadata.json not found at {meta_path}. "
            "Download from https://github.com/AIcell/MOSSBench"
        )
    items = json.loads(meta_path.read_text())
    img_dir = root / "images"
    for it in items:
        img_path = img_dir / it["image"]
        if not img_path.exists():
            continue
        yield Sample(
            sample_id=str(it["id"]),
            image=Image.open(img_path).convert("RGB"),
            query=it["query"],
            label="benign",
            meta={"category": it.get("category", "")},
        )


@register("mossbench")
def mossbench() -> DatasetSplit:
    return DatasetSplit(
        name="mossbench",
        iterator=_iter_mossbench,
        expected_size=300,
        label_distribution="benign:1.0",
    )
