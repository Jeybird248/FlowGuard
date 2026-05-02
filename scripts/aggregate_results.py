#!/usr/bin/env python3
"""Walk a FlowGuard results directory and produce a flat summary CSV.

Each row corresponds to one ``metrics.json`` from an attack-eval or utility-eval
job. Mirrors the InferenceBench script of the same name.

Usage:
    python scripts/aggregate_results.py "${FLOWGUARD_RESULTS_DIR}"
    python scripts/aggregate_results.py "${FLOWGUARD_RESULTS_DIR}" --csv main.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


_SCHEMA = [
    "model", "benchmark", "seed",
    "n", "asr", "fpr_benign", "auroc",
    "elapsed_s", "metrics_path",
]


def _row(p: Path) -> dict[str, Any] | None:
    try:
        m = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    out = {k: m.get(k) for k in _SCHEMA}
    out["metrics_path"] = str(p)
    return out


def aggregate(root: Path, group: str | None = None) -> list[dict[str, Any]]:
    base = root / group if group else root
    if not base.is_dir():
        print(f"Directory not found: {base}", file=sys.stderr)
        return []
    rows = []
    for p in sorted(base.rglob("metrics.json")):
        r = _row(p)
        if r is not None and (r.get("asr") is not None or r.get("auroc") is not None):
            rows.append(r)
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No results found.")
        return
    cols = [
        ("model", 18), ("benchmark", 14), ("seed", 6),
        ("asr", 8), ("fpr_benign", 10), ("auroc", 8), ("n", 6),
    ]
    head = "  ".join(c.ljust(w) for c, w in cols)
    print(head)
    print("-" * len(head))
    for r in rows:
        parts = []
        for c, w in cols:
            v = r.get(c)
            if v is None:
                s = ""
            elif isinstance(v, float):
                s = f"{v:.4f}"
            else:
                s = str(v)
            parts.append(s.ljust(w))
        print("  ".join(parts))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        print("No rows.", file=sys.stderr)
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--group", default=None, help="Subdirectory to scan")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = aggregate(Path(args.results_dir), args.group)
    print_table(rows)
    if args.csv:
        write_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
