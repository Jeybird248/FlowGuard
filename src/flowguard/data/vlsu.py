"""VLSU loader (Palaskar et al. 2025).

VLSU isolates *compositional cross-modal threats* where harmful semantics
emerge only after fusion. The release contains ``safe`` and ``unsafe`` subsets,
both of which we expose so utility evaluation reuses the same loader.

Layout: ``$FLOWGUARD_DATA_DIR/vlsu/{safe,unsafe}/{metadata.json,images/}``.
"""
from __future__ import annotations

import json
from typing import Iterator

from PIL import Image

from flowguard.data.base import DatasetSplit, Sample, register
from flowguard.data._paths import dataset_dir


def _iter_vlsu(subset: str) -> Iterator[Sample]:
    root = dataset_dir("vlsu") / subset
    meta = root / "metadata.json"
    if not meta.exists():
        raise FileNotFoundError(f"VLSU subset metadata not found: {meta}")
    items = json.loads(meta.read_text())
    img_dir = root / "images"
    for it in items:
        img_path = img_dir / it["image"]
        if not img_path.exists():
            continue
        yield Sample(
            sample_id=f"vlsu_{subset}_{it['id']}",
            image=Image.open(img_path).convert("RGB"),
            query=it["query"],
            label="unsafe" if subset == "unsafe" else "safe",
            meta={
                "category": it.get("category", ""),
                "modality_split": it.get("modality_split", ""),
                "subset": subset,
            },
        )


@register("vlsu_unsafe")
def vlsu_unsafe() -> DatasetSplit:
    return DatasetSplit(
        name="vlsu_unsafe",
        iterator=lambda: _iter_vlsu("unsafe"),
        expected_size=None,
        label_distribution="unsafe:1.0",
    )


@register("vlsu_safe")
def vlsu_safe() -> DatasetSplit:
    return DatasetSplit(
        name="vlsu_safe",
        iterator=lambda: _iter_vlsu("safe"),
        expected_size=None,
        label_distribution="safe:1.0",
    )
