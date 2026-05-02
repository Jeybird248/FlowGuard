"""Fit the per-model FlowGuard Isolation Forest on benign FlowVectors."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from flowguard.detector import DetectorConfig, FlowGuardDetector
from flowguard.flowvectors import stack_flowvectors
from flowguard.utils.io import dump_metrics, ensure_dir, features_path, load_jsonl
from flowguard.utils.seeding import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=248)
    parser.add_argument("--contamination", default="auto",
                        help="'auto' or a float in (0, 0.5)")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--features-split", default="vqav2_train",
                        help="JSONL split to load benign FlowVectors from")
    parser.add_argument("--decoding-steps", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    features_root = Path(args.features_dir) if args.features_dir else _features_root()
    in_path = features_path(features_root, args.model, args.features_split, args.seed, args.decoding_steps)
    if not in_path.exists():
        print(f"[train] feature file missing: {in_path}", file=sys.stderr)
        return 2

    contam: str | float = args.contamination
    try:
        contam = float(args.contamination)
    except ValueError:
        pass

    rows = list(load_jsonl(in_path))
    benign = [r for r in rows if r.get("label") == "benign"]
    if not benign:
        print(f"[train] no benign rows in {in_path}", file=sys.stderr)
        return 2
    X = stack_flowvectors(benign)
    print(f"[train] loaded {X.shape[0]} benign FlowVectors from {in_path}", file=sys.stderr)

    cfg = DetectorConfig(
        contamination=contam,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
    det = FlowGuardDetector(cfg)
    t0 = time.perf_counter()
    det.fit(X)
    elapsed = time.perf_counter() - t0

    out = Path(args.out) if args.out else features_root / args.model / f"detector_seed{args.seed}.pkl"
    ensure_dir(out.parent)
    det.save(out)
    benign_mean_path = out.parent / f"benign_mean_seed{args.seed}.npy"
    np.save(benign_mean_path, X.mean(axis=0))
    dump_metrics(
        out.with_suffix(".metrics.json"),
        {
            "model": args.model,
            "seed": args.seed,
            "n_train": int(X.shape[0]),
            "contamination": cfg.contamination,
            "n_estimators": cfg.n_estimators,
            "fit_seconds": elapsed,
            "detector_path": str(out),
            "benign_mean_path": str(benign_mean_path),
            "summary": det.summary(),
        },
    )
    print(f"[train] saved {out}", file=sys.stderr)
    return 0


def _features_root() -> Path:
    import os
    root = os.environ.get("FLOWGUARD_FEATURES_DIR")
    if not root:
        raise RuntimeError("FLOWGUARD_FEATURES_DIR not set; source env.sh")
    return Path(root)


if __name__ == "__main__":
    raise SystemExit(main())
