"""Numerical sanity tests for the distribution helpers.

These exercise the same code paths used during FlowVector extraction, so any
regression in KL/JSD/entropy will trip these immediately rather than weeks
later when an aggregation script returns nonsense.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from flowguard.utils.distributions import (
    entropy_bits,
    js_divergence_bits,
    kl_divergence_bits,
    normalize,
    renormalize_topk,
)


def test_entropy_uniform_is_log2_n():
    p = np.full(8, 1 / 8)
    assert entropy_bits(p) == pytest.approx(3.0, rel=1e-9)


def test_entropy_delta_is_zero():
    p = np.zeros(8); p[3] = 1.0
    assert entropy_bits(p) == pytest.approx(0.0, abs=1e-9)


def test_kl_to_self_is_zero():
    p = np.array([0.25, 0.25, 0.5])
    assert kl_divergence_bits(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_nonnegative_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(50))
        q = rng.dirichlet(np.ones(50))
        assert kl_divergence_bits(p, q) >= -1e-9


def test_jsd_symmetric_and_bounded():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.dirichlet(np.ones(50))
        q = rng.dirichlet(np.ones(50))
        a = float(js_divergence_bits(p, q))
        b = float(js_divergence_bits(q, p))
        assert a == pytest.approx(b, rel=1e-9)
        assert 0.0 - 1e-9 <= a <= 1.0 + 1e-9


def test_jsd_orthogonal_is_one():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert js_divergence_bits(p, q) == pytest.approx(1.0, abs=1e-9)


def test_normalize_clamps_and_renormalizes():
    p = normalize(np.array([-0.1, 0.5, 0.5]))
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


def test_renormalize_topk_smears_unobserved_mass():
    topk = {"a": math.log(0.4), "b": math.log(0.3)}  # 0.7 observed
    out = renormalize_topk(topk, vocab_size=10)
    assert out.sum() == pytest.approx(1.0)
    # Unobserved mass = 0.3, smeared over 8 tokens.
    assert out[2:].max() == pytest.approx(0.3 / 8.0, rel=1e-9)
