"""Tests for jsonl IO and the canonical features path layout."""
from __future__ import annotations

import json

from flowguard.utils.io import dump_metrics, features_path, load_jsonl, save_jsonl


def test_features_path_layout(tmp_path):
    p = features_path(tmp_path, "llava-1.5-7b", "vqav2_train", 248, decoding_step=1)
    assert p.parent.name == "k1"
    assert "llava-1.5-7b" in p.parts
    assert p.name == "vqav2_train_seed248.jsonl"


def test_jsonl_roundtrip(tmp_path):
    rows = [{"i": 1, "x": 0.5}, {"i": 2, "x": 1.5}]
    p = tmp_path / "out.jsonl"
    n = save_jsonl(p, rows)
    assert n == 2
    loaded = list(load_jsonl(p))
    assert loaded == rows


def test_dump_metrics_adds_schema_version(tmp_path):
    p = tmp_path / "metrics.json"
    dump_metrics(p, {"asr": 0.1})
    payload = json.loads(p.read_text())
    assert payload["schema_version"] == 1
    assert "timestamp" in payload
    assert payload["asr"] == 0.1
