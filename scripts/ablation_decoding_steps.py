#!/usr/bin/env python3
"""Re-train and evaluate FlowGuard with k = 1, 3, 5, 10 decoding steps.

Generates the data behind Table tab:temporal_efficiency. Expects the per-k
feature jsonl files to already exist under
``$FLOWGUARD_FEATURES_DIR/<model>/k<k>/`` (produced by
``submit_extract_features.sh --decoding-steps k``).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from flowguard.detector import DetectorConfig, FlowGuardDetector
from flowguard.eval.metrics import auroc
from flowguard.flowvectors import stack_flowvectors
from flowguard.utils.io import load_jsonl


def _features(model: str, split: str, seed: int, k: int) -> Path:
    root = Path(os.environ["FLOWGUARD_FEATURES_DIR"])
    return root / model / f"k{k}" / f"{split}_seed{seed}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--benign-split", default="vqav2_train")
    ap.add_argument("--adv-split", default="mmsb",
                    help="Adversarial split to score against (e.g. mmsb, vlsafe, vlsu_unsafe)")
    ap.add_argument("--seed", type=int, default=248)
    ap.add_argument("--out", default="results/ablations/decoding_steps.json")
    args = ap.parse_args()

    out: dict[str, dict] = {}
    for k in [int(x) for x in args.ks.split(",")]:
        benign_p = _features(args.model, args.benign_split, args.seed, k)
        adv_p = _features(args.model, args.adv_split, args.seed, k)
        if not benign_p.exists() or not adv_p.exists():
            print(f"[ablation] missing k={k}: benign={benign_p.exists()} adv={adv_p.exists()}")
            continue
        Xb = stack_flowvectors(list(load_jsonl(benign_p)))
        Xa = stack_flowvectors(list(load_jsonl(adv_p)))
        det = FlowGuardDetector(DetectorConfig(random_state=args.seed)).fit(Xb)
        sb = det.score(Xb); sa = det.score(Xa)
        au = auroc(sa, sb)
        out[str(k)] = {"auroc": au, "n_benign": int(Xb.shape[0]), "n_adv": int(Xa.shape[0])}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
