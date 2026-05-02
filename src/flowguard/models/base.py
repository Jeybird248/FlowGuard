"""Abstract base for MLLM probes.

A probe must implement three queries on the underlying MLLM:

    P_t  = P(y_1 | empty, Q)
    P_v  = P(y_1 | I, Q_neutral)
    P_mm = P(y_1 | I, Q)

These are returned as length-|V| numpy arrays (or top-k dicts for the API
backend). The MLLM is loaded eagerly in ``__init__``; ``probe`` is called once
per (image, text) input by ``flowguard.scripts.extract``.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

NEUTRAL_PROMPT = "Describe this image."


@dataclass
class ProbeOutputs:
    """Output of a single (image, text) probe call.

    Attributes
    ----------
    p_text:
        Distribution over the first generated token under text-only conditioning.
    p_vision:
        Distribution over the first token under vision-only conditioning
        (image present, query replaced by ``NEUTRAL_PROMPT``).
    p_multimodal:
        Distribution over the first token under joint multimodal conditioning.
    decoding_steps:
        Number of decoding steps captured. Default 1; passing ``k > 1`` returns
        ``(k, |V|)`` arrays for the temporal-efficiency ablation.
    meta:
        Per-call diagnostics (timings, vocab size, etc.).
    """

    p_text: np.ndarray
    p_vision: np.ndarray
    p_multimodal: np.ndarray
    decoding_steps: int = 1
    meta: dict[str, Any] = field(default_factory=dict)


class MLLMProbe(abc.ABC):
    """Probe interface implemented by every backend."""

    model_key: str
    vocab_size: int

    def __init__(self, model_key: str) -> None:
        self.model_key = model_key

    @abc.abstractmethod
    def probe(
        self,
        image: Image.Image | None,
        text: str,
        neutral_text: str = NEUTRAL_PROMPT,
        decoding_steps: int = 1,
    ) -> ProbeOutputs:
        """Run the three conditioning configurations and return distributions."""

    @abc.abstractmethod
    def generate(
        self,
        image: Image.Image | None,
        text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response under joint conditioning. Used for utility evals."""

    def hidden_state(
        self,
        image: Image.Image | None,
        text: str,
    ) -> np.ndarray | None:
        """Optional: return the final-layer hidden state for the Raw Embedding
        baseline. Backends that cannot expose hidden states return None and the
        Raw Embedding control will be skipped for them."""
        return None

    def close(self) -> None:
        """Release GPU memory (override in subclasses if needed)."""
        return None
