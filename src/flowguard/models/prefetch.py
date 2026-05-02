"""Pre-download MLLM checkpoints into the shared HF cache.

Run before submitting compute jobs so the worker nodes don't need network
access. Honors ``HF_HUB_CACHE`` and the ``hf_transfer`` accelerator if it is
already installed in the calling environment.

Usage:
    python -m flowguard.models.prefetch \
        --models llava-1.5-7b qwen2.5-vl-7b gemma-3-4b
"""
from __future__ import annotations

import argparse
import sys

REPOS = {
    "llava-1.5-7b":     "llava-hf/llava-1.5-7b-hf",
    "qwen2.5-vl-7b":    "Qwen/Qwen2.5-VL-7B-Instruct",
    "gemma-3-4b":       "google/gemma-3-4b-it",
    "llama-3.1-70b":    "meta-llama/Llama-3.1-70B-Instruct",
    "clip-vit-l-14-336": "openai/clip-vit-large-patch14-336",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    for key in args.models:
        if key not in REPOS:
            print(f"[prefetch] unknown key: {key}", file=sys.stderr)
            return 2
        repo = REPOS[key]
        print(f"[prefetch] downloading {repo} ...")
        snapshot_download(repo_id=repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
