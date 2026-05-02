"""Evaluate FlowGuard on a benchmark split.

For each sample we (a) probe the model to get the three predictive distributions,
(b) compute the FlowVector, (c) score it under the fitted Isolation Forest,
and (d) -- when the input is *not* flagged as anomalous -- generate a response
and run the harm judge. The aggregate metrics ASR / FPR / AUROC are then
written to ``metrics.json``.

Output layout::

    $FLOWGUARD_RESULTS_DIR/<group>/<model>/<benchmark>/seed<seed>/
        responses.jsonl
        metrics.json

The script makes no assumptions about how the adversarial inputs were
constructed; the (image, query) pairs come straight from the benchmark loader
(MM-SafetyBench, VLSafe, VLSU, MOSSBench, etc.). To evaluate FlowGuard against
an externally generated attack set, simply load it as a custom dataset split
in ``flowguard.data``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from flowguard.data import load_split
from flowguard.detector import FlowGuardDetector
from flowguard.eval.harm_judge import judge_harm
from flowguard.eval.metrics import attack_success_rate, auroc, false_positive_rate
from flowguard.flowvectors import compute_flowvector
from flowguard.models import load as load_model
from flowguard.utils.io import dump_metrics, ensure_dir, save_jsonl
from flowguard.utils.seeding import set_global_seed


def _results_root() -> Path:
    root = os.environ.get("FLOWGUARD_RESULTS_DIR")
    if not root:
        raise RuntimeError("FLOWGUARD_RESULTS_DIR not set; source env.sh")
    return Path(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key (e.g. llava-1.5-7b)")
    parser.add_argument("--benchmark", required=True,
                        help="Dataset split, e.g. mmsb / vlsafe / vlsu_unsafe / mossbench / vqav2_val")
    parser.add_argument("--seed", type=int, default=248)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--detector-path", required=True,
                        help="Path to a fitted FlowGuard detector .pkl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip generation + harm judge (collect FlowVectors only)")
    parser.add_argument("--group", default="main",
                        help="Subfolder under FLOWGUARD_RESULTS_DIR")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out_root = _results_root() / args.group / args.model / args.benchmark / f"seed{args.seed}"
    ensure_dir(out_root)

    print(f"[eval] {args.model} | {args.benchmark} | detector={args.detector_path}",
          file=sys.stderr)
    detector = FlowGuardDetector.load(args.detector_path)
    model = load_model(args.model)
    split = load_split(args.benchmark)

    rows = []
    pos_scores: list[float] = []   # adversarial / unsafe scores
    neg_scores: list[float] = []   # benign / safe scores
    blocked: list[bool] = []
    blocked_benign: list[bool] = []
    harms: list[bool] = []
    n_done = 0
    t0 = time.perf_counter()

    for i, sample in tqdm(enumerate(split), total=args.n, file=sys.stderr):
        if args.n is not None and n_done >= args.n:
            break
        try:
            probe = model.probe(sample.image, sample.query)
            fv = compute_flowvector(probe.p_text, probe.p_vision, probe.p_multimodal)
            X = np.asarray([fv.to_array()])
            score = float(detector.score(X)[0])
            is_unsafe_flag = bool(detector.predict(X)[0])

            response = ""
            harmful = False
            if not is_unsafe_flag and not args.no_generate:
                response = model.generate(
                    sample.image, sample.query,
                    max_new_tokens=args.max_new_tokens, temperature=0.0,
                )
                if sample.label == "unsafe":
                    harmful = judge_harm(sample.query, response, ground_truth_label="unsafe")
        except Exception as e:  # noqa: BLE001
            print(f"[eval] sample {sample.sample_id} failed: {e}", file=sys.stderr)
            continue

        rows.append({
            "sample_id": sample.sample_id,
            "label": sample.label,
            "flowvector": fv.to_dict(),
            "score": score,
            "blocked": is_unsafe_flag,
            "harmful": harmful,
            "response": response[:1000],
            "meta": sample.meta,
        })
        if sample.label == "unsafe":
            pos_scores.append(score)
        elif sample.label in ("benign", "safe"):
            neg_scores.append(score)
            blocked_benign.append(is_unsafe_flag)
        blocked.append(is_unsafe_flag)
        harms.append(harmful)
        n_done += 1

    save_jsonl(out_root / "responses.jsonl", rows)
    asr = attack_success_rate(harms, blocked) if not args.no_generate else float("nan")
    fpr = false_positive_rate(blocked_benign) if blocked_benign else float("nan")
    au = auroc(pos_scores, neg_scores) if pos_scores and neg_scores else float("nan")

    dump_metrics(out_root / "metrics.json", {
        "model": args.model,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "n": n_done,
        "asr": asr,
        "fpr_benign": fpr,
        "auroc": au,
        "detector_path": args.detector_path,
        "elapsed_s": time.perf_counter() - t0,
    })
    print(f"[eval] ASR={asr} FPR={fpr} AUROC={au} (n={n_done})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
