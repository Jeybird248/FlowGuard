"""VQAv2 loader.

Used both as the *benign training corpus* for the Isolation Forest (10k random
val samples) and as a *utility benchmark* (held-out val accuracy).

Expected layout under ``$FLOWGUARD_DATA_DIR/vqav2/``:
    questions/v2_OpenEnded_mscoco_val2014_questions.json
    annotations/v2_mscoco_val2014_annotations.json
    images/val2014/COCO_val2014_<img_id:012d>.jpg
"""
from __future__ import annotations

import json
import random
from typing import Iterator

from PIL import Image

from flowguard.data.base import DatasetSplit, Sample, register
from flowguard.data._paths import dataset_dir


def _iter_vqav2(seed: int = 248, n: int | None = None) -> Iterator[Sample]:
    root = dataset_dir("vqav2")
    qpath = root / "questions" / "v2_OpenEnded_mscoco_val2014_questions.json"
    apath = root / "annotations" / "v2_mscoco_val2014_annotations.json"
    if not qpath.exists() or not apath.exists():
        raise FileNotFoundError(
            f"VQAv2 not found under {root}. Download from https://visualqa.org/download.html"
        )
    questions = json.loads(qpath.read_text())["questions"]
    annotations = {a["question_id"]: a for a in json.loads(apath.read_text())["annotations"]}

    rng = random.Random(seed)
    rng.shuffle(questions)
    if n is not None:
        questions = questions[:n]

    img_dir = root / "images" / "val2014"
    for q in questions:
        img_path = img_dir / f"COCO_val2014_{q['image_id']:012d}.jpg"
        if not img_path.exists():
            continue
        ann = annotations.get(q["question_id"], {})
        ans_set = [a["answer"] for a in ann.get("answers", [])]
        yield Sample(
            sample_id=str(q["question_id"]),
            image=Image.open(img_path).convert("RGB"),
            query=q["question"],
            label="benign",
            meta={
                "image_id": q["image_id"],
                "answer_set": ans_set,
                "majority_answer": max(set(ans_set), key=ans_set.count) if ans_set else "",
            },
        )


@register("vqav2_train")
def vqav2_train(seed: int = 248, n: int = 10000) -> DatasetSplit:
    """Benign training corpus (10k random val samples)."""
    return DatasetSplit(
        name="vqav2_train",
        iterator=lambda: _iter_vqav2(seed=seed, n=n),
        expected_size=n,
        label_distribution="benign:1.0",
    )


@register("vqav2_val")
def vqav2_val(seed: int = 1337, n: int = 5000) -> DatasetSplit:
    """Held-out utility split (disjoint from training via the seed)."""
    return DatasetSplit(
        name="vqav2_val",
        iterator=lambda: _iter_vqav2(seed=seed, n=n),
        expected_size=n,
        label_distribution="benign:1.0",
    )
