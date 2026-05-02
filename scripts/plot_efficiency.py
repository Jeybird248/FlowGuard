#!/usr/bin/env python3
"""Latency vs. AUROC scatter, one point per benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Aggregated CSV from aggregate_results.py")
    ap.add_argument("--out", default="results/figures/latency_auroc.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["auroc"]).copy()
    by = df.groupby("benchmark")
    summary = by.agg({
        "asr": "mean",
        "fpr_benign": "mean",
        "auroc": "mean",
        "elapsed_s": "median",
        "n": "median",
    }).reset_index()
    summary["latency_per_sample"] = summary["elapsed_s"] / summary["n"].clip(lower=1)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for _, row in summary.iterrows():
        ax.scatter(row["latency_per_sample"], row["auroc"], s=80)
        ax.annotate(row["benchmark"], (row["latency_per_sample"], row["auroc"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Latency per sample (s)")
    ax.set_ylabel("AUROC")
    ax.grid(True, linestyle=":", alpha=0.6)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
