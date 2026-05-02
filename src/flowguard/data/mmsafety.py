"""MM-SafetyBench loader (Liu et al. 2023).

The MM-SafetyBench release is organized by *harm category* (e.g.
``01-Illegal_Activity``, ``02-HateSpeech``, ...). Each category contains
``imgs/<n>.png`` and ``questions/<n>.json`` (with ``Question`` / ``Rephrased Question``
fields). FlowGuard treats every such row as a single unsafe direct query
(plus optional attack-method overlays applied later).

Layout under ``$FLOWGUARD_DATA_DIR/mmsafetybench/``:
    01-Illegal_Activity/
        imgs/0.png ...
        questions/0.json ...
    02-HateSpeech/...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from PIL import Image

from flowguard.data.base import DatasetSplit, Sample, register
from flowguard.data._paths import dataset_dir


CATEGORIES = [
    "01-Illegal_Activity",
    "02-HateSpeech",
    "03-Malware_Generation",
    "04-Physical_Harm",
    "05-EconomicHarm",
    "06-Fraud",
    "07-Sex",
    "08-Political_Lobbying",
    "09-Privacy_Violence",
    "10-Legal_Opinion",
    "11-Financial_Advice",
    "12-Health_Consultation",
    "13-Gov_Decision",
]


def _iter_mmsb(category: str | None = None) -> Iterator[Sample]:
    root = dataset_dir("mmsafetybench")
    if not root.exists():
        raise FileNotFoundError(f"MM-SafetyBench root not found: {root}")
    cats = [category] if category else CATEGORIES
    for cat in cats:
        cat_dir = root / cat
        if not cat_dir.exists():
            continue
        q_dir = cat_dir / "questions"
        i_dir = cat_dir / "imgs"
        for qfile in sorted(q_dir.glob("*.json")):
            qid = qfile.stem
            ipath = i_dir / f"{qid}.png"
            if not ipath.exists():
                ipath = i_dir / f"{qid}.jpg"
            if not ipath.exists():
                continue
            data = json.loads(qfile.read_text())
            query = data.get("Rephrased Question") or data.get("Question") or ""
            yield Sample(
                sample_id=f"{cat}/{qid}",
                image=Image.open(ipath).convert("RGB"),
                query=query,
                label="unsafe",
                meta={"category": cat, "raw_question": data.get("Question", "")},
            )


@register("mmsb")
def mmsb() -> DatasetSplit:
    return DatasetSplit(
        name="mmsb",
        iterator=_iter_mmsb,
        expected_size=None,
        label_distribution="unsafe:1.0",
    )


for _cat in CATEGORIES:
    def _make(cat=_cat):
        return DatasetSplit(
            name=f"mmsb_{cat}",
            iterator=lambda c=cat: _iter_mmsb(c),
            expected_size=None,
            label_distribution="unsafe:1.0",
        )
    register(f"mmsb_{_cat}")(_make)
