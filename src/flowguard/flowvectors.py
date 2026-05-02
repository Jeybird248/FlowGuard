"""FlowVector extraction.

A FlowVector ``phi(x) = (U_v, U_t, R, S)`` summarizes how three first-token
predictive distributions relate to one another:

    P_t  = P(y_1 | empty,  Q)         # text-only prior
    P_v  = P(y_1 | I, Q_neutral)      # vision-only prior
    P_mm = P(y_1 | I, Q)              # joint multimodal posterior

The four scalar features instantiate PID-inspired notions
(visual / text uniqueness, redundancy, synergy):

    U_v = KL(P_mm || P_t)             # mass that fusion places where text-only didn't
    U_t = KL(P_mm || P_v)             # mass that fusion places where vision-only didn't
    R   = 1 - JSD(P_t || P_v)         # benign agreement between unimodal priors
    S   = 0.5 * (H(P_t) + H(P_v)) - H(P_mm)  # entropy reduction from fusion

All distributions are dense (full vocabulary) when the underlying model exposes
logits; for top-k logprob APIs, callers should pre-renormalize via
``flowguard.utils.distributions.renormalize_topk`` before calling
``compute_flowvector``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from flowguard.utils.distributions import (
    entropy_bits,
    js_divergence_bits,
    kl_divergence_bits,
    normalize,
)


@dataclass(frozen=True)
class FlowVector:
    """4-tuple of FlowGuard features for a single (image, text) input."""

    u_v: float
    u_t: float
    r: float
    s: float

    def to_array(self) -> np.ndarray:
        return np.array([self.u_v, self.u_t, self.r, self.s], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @staticmethod
    def from_array(arr: np.ndarray) -> "FlowVector":
        u_v, u_t, r, s = (float(x) for x in arr.reshape(-1)[:4])
        return FlowVector(u_v=u_v, u_t=u_t, r=r, s=s)


def compute_flowvector(p_text: np.ndarray,
                       p_vision: np.ndarray,
                       p_multimodal: np.ndarray) -> FlowVector:
    """Compute a FlowVector from three aligned probability distributions.

    All three inputs must be 1-D arrays of identical length representing
    distributions over the same vocabulary. They are renormalized internally
    so the caller need not normalize beforehand. Negative entries are clamped
    to zero (which can happen when callers exponentiate logits in float16).
    """
    p_t = normalize(p_text)
    p_v = normalize(p_vision)
    p_mm = normalize(p_multimodal)

    if not (p_t.shape == p_v.shape == p_mm.shape):
        raise ValueError(
            f"FlowVector inputs must share shape, got "
            f"{p_t.shape}, {p_v.shape}, {p_mm.shape}"
        )

    u_v = kl_divergence_bits(p_mm, p_t)
    u_t = kl_divergence_bits(p_mm, p_v)
    jsd = js_divergence_bits(p_t, p_v)
    r = 1.0 - float(jsd)
    h_mm = entropy_bits(p_mm)
    h_t = entropy_bits(p_t)
    h_v = entropy_bits(p_v)
    s = 0.5 * (float(h_t) + float(h_v)) - float(h_mm)

    return FlowVector(u_v=float(u_v), u_t=float(u_t), r=float(r), s=float(s))


def compute_flowvector_topk(p_text_topk: dict[str, float],
                            p_vision_topk: dict[str, float],
                            p_multimodal_topk: dict[str, float]) -> FlowVector:
    """FlowVector under partial-distribution access.

    Used for API-only models (e.g. GPT-4.1-mini) where the only signal is a
    top-k log-probability dictionary. We take the union of observed tokens,
    pad missing entries with the "unobserved tail" probability (uniform over
    the missing tokens), and renormalize before computing the FlowVector.
    """
    union = sorted(
        set(p_text_topk) | set(p_vision_topk) | set(p_multimodal_topk)
    )
    if not union:
        raise ValueError("All three top-k dicts are empty.")

    def _expand(dct: dict[str, float]) -> np.ndarray:
        observed_mass = sum(np.exp(v) for v in dct.values())
        unobs = max(0.0, 1.0 - float(observed_mass))
        unobs_tokens = [t for t in union if t not in dct]
        per_unobs = unobs / max(len(unobs_tokens), 1)
        out = np.zeros(len(union), dtype=np.float64)
        for i, tok in enumerate(union):
            if tok in dct:
                out[i] = float(np.exp(dct[tok]))
            else:
                out[i] = per_unobs
        s = out.sum()
        return out / s if s > 0 else out

    return compute_flowvector(
        _expand(p_text_topk),
        _expand(p_vision_topk),
        _expand(p_multimodal_topk),
    )


def compute_flowvector_multistep(p_text: np.ndarray,
                                 p_vision: np.ndarray,
                                 p_multimodal: np.ndarray) -> dict[str, FlowVector]:
    """Per-step FlowVectors for a (k, |V|) trio of distributions.

    Returns a dict keyed by ``step_{i}`` for i in 0..k-1, plus a ``mean``
    entry summarizing the average across steps (used for the k > 1 ablation
    in Section 5.5 / Appendix). Each input is shaped (k, |V|).
    """
    p_text = np.atleast_2d(p_text)
    p_vision = np.atleast_2d(p_vision)
    p_mm = np.atleast_2d(p_multimodal)
    k = p_text.shape[0]
    out: dict[str, FlowVector] = {}
    arrs = []
    for i in range(k):
        fv = compute_flowvector(p_text[i], p_vision[i], p_mm[i])
        out[f"step_{i}"] = fv
        arrs.append(fv.to_array())
    mean_arr = np.mean(np.stack(arrs, axis=0), axis=0)
    out["mean"] = FlowVector.from_array(mean_arr)
    return out


def feature_names() -> list[str]:
    return ["U_v", "U_t", "R", "S"]


def stack_flowvectors(records: list[dict[str, Any]]) -> np.ndarray:
    """Stack jsonl records into an (N, 4) numpy matrix.

    Each record must contain a ``flowvector`` field (the dict from
    ``FlowVector.to_dict``) or a flat ``u_v`` / ``u_t`` / ``r`` / ``s``.
    """
    rows = []
    for rec in records:
        if "flowvector" in rec:
            d = rec["flowvector"]
        else:
            d = rec
        rows.append([d["u_v"], d["u_t"], d["r"], d["s"]])
    return np.asarray(rows, dtype=np.float64)
