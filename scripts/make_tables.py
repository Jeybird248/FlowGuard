#!/usr/bin/env python3
"""Pivot the aggregate CSV into a per-benchmark mean ± std table.

Reads the CSV produced by ``aggregate_results.py`` and emits a small LaTeX
table with one row per benchmark and three columns (ASR, FPR, AUROC),
aggregated across seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(mean: float, std: float, pct: bool = True) -> str:
    if np.isnan(mean):
        return "--"
    scale = 100.0 if pct else 1.0
    return f"{mean*scale:.1f}\\pm{std*scale:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--out", default="results/tables/flowguard_per_benchmark.tex")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["model"] == args.model]

    rows = []
    for bench, sub in df.groupby("benchmark"):
        rows.append({
            "benchmark": bench,
            "asr_mean": sub["asr"].mean(),        "asr_std": sub["asr"].std(ddof=0),
            "fpr_mean": sub["fpr_benign"].mean(), "fpr_std": sub["fpr_benign"].std(ddof=0),
            "au_mean":  sub["auroc"].mean(),      "au_std":  sub["auroc"].std(ddof=0),
        })
    rows.sort(key=lambda r: r["benchmark"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Benchmark & ASR (\\%) & FPR (\\%) & AUROC \\\\",
        "\\midrule",
    ]
    for r in rows:
        bench = r["benchmark"].replace("_", "\\_")
        lines.append(
            f"{bench} & "
            f"${_fmt(r['asr_mean'], r['asr_std'])}$ & "
            f"${_fmt(r['fpr_mean'], r['fpr_std'])}$ & "
            f"${_fmt(r['au_mean'], r['au_std'], pct=False)}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
