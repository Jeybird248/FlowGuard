"""Extract FlowVectors over a dataset split and save them to JSONL.

Each row is::

    {
      "sample_id": "...",
      "label": "benign" | "unsafe" | "safe",
      "flowvector": {"u_v": ..., "u_t": ..., "r": ..., "s": ...},
      "meta": {...}
    }

Cached output paths follow ``flowguard.utils.io.features_path``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from flowguard.data import load_split
from flowguard.flowvectors import compute_flowvector
from flowguard.models import load as load_model
from flowguard.utils.io import dump_metrics, ensure_dir, features_path, save_jsonl
from flowguard.utils.seeding import set_global_seed


def _features_root() -> Path:
    root = os.environ.get("FLOWGUARD_FEATURES_DIR")
    if not root:
        raise RuntimeError("FLOWGUARD_FEATURES_DIR not set; source env.sh")
    return Path(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key (e.g. llava-1.5-7b)")
    parser.add_argument("--split", required=True, help="Dataset split key (e.g. vqav2_train)")
    parser.add_argument("--n", type=int, default=None, help="Cap the number of samples")
    parser.add_argument("--seed", type=int, default=248)
    parser.add_argument("--decoding-steps", type=int, default=1)
    parser.add_argument("--features-dir", default=None,
                        help="Override $FLOWGUARD_FEATURES_DIR")
    parser.add_argument("--out", default=None, help="Override the auto-derived output path")
    args = parser.parse_args()

    set_global_seed(args.seed)
    features_root = Path(args.features_dir) if args.features_dir else _features_root()
    out_path = Path(args.out) if args.out else features_path(
        features_root, args.model, args.split, args.seed, args.decoding_steps
    )
    ensure_dir(out_path.parent)
    metrics_path = out_path.with_suffix(".metrics.json")

    print(f"[extract] model={args.model} split={args.split} -> {out_path}", file=sys.stderr)
    model = load_model(args.model)
    split = load_split(args.split)

    n_done = 0
    n_skipped = 0
    rows = []
    t0 = time.perf_counter()
    for i, sample in tqdm(enumerate(split), total=args.n, file=sys.stderr):
        if args.n is not None and n_done >= args.n:
            break
        try:
            probe = model.probe(sample.image, sample.query, decoding_steps=args.decoding_steps)
            fv = compute_flowvector(probe.p_text, probe.p_vision, probe.p_multimodal)
        except Exception as e:  # noqa: BLE001
            n_skipped += 1
            continue

        rows.append({
            "sample_id": sample.sample_id,
            "label": sample.label,
            "flowvector": fv.to_dict(),
            "decoding_steps": args.decoding_steps,
            "meta": sample.meta,
            "probe_meta": probe.meta,
        })
        n_done += 1

    n_written = save_jsonl(out_path, rows)
    dump_metrics(metrics_path, {
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "decoding_steps": args.decoding_steps,
        "n_written": n_written,
        "n_skipped": n_skipped,
        "elapsed_s": time.perf_counter() - t0,
        "output_path": str(out_path),
    })
    print(f"[extract] wrote {n_written} rows ({n_skipped} skipped) -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
