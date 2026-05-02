#!/usr/bin/env python3
"""t-SNE comparison: raw first-token logits vs. FlowVectors (Figure tsne_panels)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from flowguard.flowvectors import stack_flowvectors
from flowguard.utils.io import load_jsonl


def _load(jsonl: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(load_jsonl(jsonl))
    X = stack_flowvectors(rows)
    y = np.array([1 if r.get("label") == "unsafe" else 0 for r in rows], dtype=np.int8)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", required=True, help="JSONL of benign FlowVectors")
    ap.add_argument("--adversarial", required=True, help="JSONL of adversarial FlowVectors")
    ap.add_argument("--raw-benign", default=None, help="Optional JSONL with raw_embedding field")
    ap.add_argument("--raw-adv", default=None, help="Optional JSONL with raw_embedding field")
    ap.add_argument("--out", default="results/figures/tsne.pdf")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xb, yb = _load(Path(args.benign))
    Xa, ya = _load(Path(args.adversarial))
    X = np.concatenate([Xb, Xa])
    y = np.concatenate([yb, np.ones_like(ya)])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    if args.raw_benign and args.raw_adv:
        raw_b = np.array([r["raw_embedding"] for r in load_jsonl(Path(args.raw_benign))])
        raw_a = np.array([r["raw_embedding"] for r in load_jsonl(Path(args.raw_adv))])
        Xr = np.concatenate([raw_b, raw_a])
        yr = np.concatenate([np.zeros(len(raw_b)), np.ones(len(raw_a))])
        emb_raw = TSNE(n_components=2, random_state=args.seed, init="pca",
                       perplexity=30).fit_transform(Xr)
        axes[0].scatter(emb_raw[yr == 0, 0], emb_raw[yr == 0, 1], s=6, alpha=0.5, label="benign")
        axes[0].scatter(emb_raw[yr == 1, 0], emb_raw[yr == 1, 1], s=6, alpha=0.5, label="adversarial")
        axes[0].set_title("Raw first-step logits")
    else:
        axes[0].text(0.5, 0.5, "no raw embeddings", ha="center", va="center")

    emb = TSNE(n_components=2, random_state=args.seed, init="pca",
               perplexity=30).fit_transform(X)
    axes[1].scatter(emb[y == 0, 0], emb[y == 0, 1], s=6, alpha=0.5, label="benign")
    axes[1].scatter(emb[y == 1, 0], emb[y == 1, 1], s=6, alpha=0.5, label="adversarial")
    axes[1].set_title("FlowVectors (4D)")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
