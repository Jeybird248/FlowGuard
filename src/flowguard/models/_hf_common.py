"""Shared utilities for HuggingFace-backed MLLM probes."""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
from PIL import Image


def load_dtype() -> torch.dtype:
    """Default to bfloat16 on Hopper / Ampere; fall back to float16 elsewhere."""
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def device_map_for(model_id: str) -> str | dict[str, Any]:
    """Single-card by default; larger models can override via ``device_map=auto``."""
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu >= 2:
        return "auto"
    return "cuda" if n_gpu == 1 else "cpu"


@torch.no_grad()
def first_token_distribution(model: Any,
                             inputs: dict[str, torch.Tensor],
                             decoding_steps: int = 1) -> np.ndarray:
    """Return the next-token distribution(s) at the first decoding step.

    For ``decoding_steps == 1`` returns shape ``(|V|,)``. For ``k > 1`` returns
    shape ``(k, |V|)`` by greedy-decoding ``k`` tokens and recording the
    pre-sampling softmax at every step.
    """
    model.eval()
    if decoding_steps == 1:
        out = model(**inputs)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)[0]
        return probs.detach().to("cpu").numpy()

    # Multi-step path. We use greedy continuation to keep the temporal probe
    # deterministic; the goal here is the *distribution* at each step, not the
    # generated text.
    distros: list[np.ndarray] = []
    cur_inputs = {k: v.clone() for k, v in inputs.items()}
    for _ in range(decoding_steps):
        out = model(**cur_inputs)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)[0]
        distros.append(probs.detach().to("cpu").numpy())
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        if "input_ids" in cur_inputs:
            cur_inputs["input_ids"] = torch.cat([cur_inputs["input_ids"], next_tok], dim=-1)
        if "attention_mask" in cur_inputs:
            cur_inputs["attention_mask"] = torch.cat(
                [cur_inputs["attention_mask"], torch.ones_like(next_tok)], dim=-1
            )
    return np.stack(distros, axis=0)


def timeit() -> float:
    return time.perf_counter()


def hf_cache_dir() -> str | None:
    return os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
