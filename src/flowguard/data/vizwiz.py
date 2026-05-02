"""VizWiz-VQA loader (out-of-distribution benign benchmark).

Layout under ``$FLOWGUARD_DATA_DIR/vizwiz/``:
    annotations/val.json
    images/val/<image_filename>
"""
from __future__ import annotations

import json
from typing import Iterator

from PIL import Image

from flowguard.data.base import DatasetSplit, Sample, register
from flowguard.data._paths import dataset_dir


def _iter_vizwiz() -> Iterator[Sample]:
    root = dataset_dir("vizwiz")
    ann_path = root / "annotations" / "val.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"VizWiz val.json not found at {ann_path}")
    items = json.loads(ann_path.read_text())
    img_dir = root / "images" / "val"
    for i, it in enumerate(items):
        img_path = img_dir / it["image"]
        if not img_path.exists():
            continue
        yield Sample(
            sample_id=str(i),
            image=Image.open(img_path).convert("RGB"),
            query=it["question"],
            label="benign",
            meta={
                "answers": [a["answer"] for a in it.get("answers", [])],
                "answerable": it.get("answerable", 1),
            },
        )


@register("vizwiz_val")
def vizwiz_val() -> DatasetSplit:
    return DatasetSplit(
        name="vizwiz_val",
        iterator=_iter_vizwiz,
        expected_size=4319,
        label_distribution="benign:1.0",
    )
