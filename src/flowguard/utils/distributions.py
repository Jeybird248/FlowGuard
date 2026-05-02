"""Numerically stable Shannon entropy, KL divergence, and Jensen--Shannon divergence
implemented over discrete probability distributions (numpy and torch).

All quantities are reported in **bits** (log base 2) so that JSD is bounded to
[0, 1], which lets us define ``R = 1 - JSD`` directly in the FlowVector.
"""
from __future__ import annotations

from typing import Union

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is required at runtime
    _TORCH_AVAILABLE = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]

_LN2 = np.log(2.0)
_EPS = 1e-12


def _to_numpy(p: ArrayLike) -> np.ndarray:
    if _TORCH_AVAILABLE and isinstance(p, torch.Tensor):
        return p.detach().to(dtype=torch.float64, device="cpu").numpy()
    return np.asarray(p, dtype=np.float64)


def normalize(p: ArrayLike, axis: int = -1) -> np.ndarray:
    """Renormalize ``p`` along ``axis`` to sum to 1.

    ``p`` may be unnormalized (e.g. softmax output that drifted slightly off,
    or top-k logprobs exponentiated and summed to <1). Negative values are
    clamped to zero.
    """
    arr = np.maximum(_to_numpy(p), 0.0)
    s = arr.sum(axis=axis, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return arr / s


def renormalize_topk(top_logprobs: dict[str, float] | list[tuple[str, float]],
                     vocab_size: int | None = None) -> np.ndarray:
    """Convert OpenAI-style top-k logprobs into a renormalized probability tail.

    The remaining mass (``1 - sum(exp(top_logprobs))``) is smeared uniformly
    over the unobserved tail when ``vocab_size`` is given; otherwise the
    distribution is left over only the observed tokens. This is the
    *truncated tail* approximation used for GPT-4.1-mini in
    Appendix A.4 of the paper.
    """
    if isinstance(top_logprobs, dict):
        items = list(top_logprobs.items())
    else:
        items = list(top_logprobs)

    tokens = [t for t, _ in items]
    probs = np.array([np.exp(lp) for _, lp in items], dtype=np.float64)
    observed_mass = float(probs.sum())

    if vocab_size is not None and vocab_size > len(tokens):
        unobs = max(0.0, 1.0 - observed_mass)
        per_token = unobs / float(vocab_size - len(tokens))
        full = np.full(vocab_size, per_token, dtype=np.float64)
        full[: len(probs)] = probs
        return full / full.sum()
    return probs / max(observed_mass, _EPS)


def entropy_bits(p: ArrayLike, axis: int = -1) -> np.ndarray | float:
    """Shannon entropy in bits, computed in float64 with eps clipping."""
    p = normalize(p, axis=axis)
    log_p = np.log(np.clip(p, _EPS, 1.0)) / _LN2
    out = -(p * log_p).sum(axis=axis)
    if out.ndim == 0:
        return float(out)
    return out


def kl_divergence_bits(p: ArrayLike, q: ArrayLike, axis: int = -1) -> np.ndarray | float:
    """KL(p || q) in bits.

    Uses log(p / q) under eps clipping. By convention 0 * log(0/q) = 0 and
    p * log(p/0) = +inf, but we clamp q to ``_EPS`` to keep the result finite.
    The fused-vs-unimodal divergences in the paper are bounded in practice
    because all three distributions come from the same softmax head.
    """
    p = normalize(p, axis=axis)
    q = normalize(q, axis=axis)
    p_safe = np.clip(p, _EPS, 1.0)
    q_safe = np.clip(q, _EPS, 1.0)
    log_ratio = np.log(p_safe / q_safe) / _LN2
    out = (p * log_ratio).sum(axis=axis)
    if out.ndim == 0:
        return float(out)
    return out


def js_divergence_bits(p: ArrayLike, q: ArrayLike, axis: int = -1) -> np.ndarray | float:
    """Jensen--Shannon divergence in bits, bounded to [0, 1].

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = 0.5 * (p + q).
    """
    p = normalize(p, axis=axis)
    q = normalize(q, axis=axis)
    m = 0.5 * (p + q)
    out = 0.5 * kl_divergence_bits(p, m, axis=axis) + 0.5 * kl_divergence_bits(q, m, axis=axis)
    # Clamp tiny negatives from float roundoff and the [0, 1] upper bound.
    return np.clip(out, 0.0, 1.0)
