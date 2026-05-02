"""VLSafe loader (Zong et al. 2024)."""
from __future__ import annotations

import json
from typing import Iterator

from PIL import Image

from flowguard.data.base import DatasetSplit, Sample, register
from flowguard.data._paths import dataset_dir


def _iter_vlsafe() -> Iterator[Sample]:
    root = dataset_dir("vlsafe")
    meta = root / "harmful.json"
    if not meta.exists():
        raise FileNotFoundError(f"VLSafe harmful.json not found at {meta}")
    items = json.loads(meta.read_text())
    img_dir = root / "images"
    for i, it in enumerate(items):
        img_path = img_dir / it["image"]
        if not img_path.exists():
            continue
        yield Sample(
            sample_id=str(i),
            image=Image.open(img_path).convert("RGB"),
            query=it["query"],
            label="unsafe",
            meta={"category": it.get("category", "")},
        )


@register("vlsafe")
def vlsafe() -> DatasetSplit:
    return DatasetSplit(
        name="vlsafe",
        iterator=_iter_vlsafe,
        expected_size=None,
        label_distribution="unsafe:1.0",
    )
