"""Tests for FlowVector construction and edge cases."""
from __future__ import annotations

import math

import numpy as np
import pytest

from flowguard.flowvectors import (
    FlowVector,
    compute_flowvector,
    compute_flowvector_multistep,
    compute_flowvector_topk,
)


def _delta(n: int, idx: int) -> np.ndarray:
    p = np.zeros(n)
    p[idx] = 1.0
    return p


def test_identical_distributions_yield_zero_uniqueness_and_perfect_redundancy():
    p = np.array([0.2, 0.3, 0.5])
    fv = compute_flowvector(p, p, p)
    assert fv.u_v == pytest.approx(0.0, abs=1e-9)
    assert fv.u_t == pytest.approx(0.0, abs=1e-9)
    assert fv.r == pytest.approx(1.0, abs=1e-9)
    # When P_t = P_v = P_mm, S = 0 by construction.
    assert fv.s == pytest.approx(0.0, abs=1e-9)


def test_orthogonal_unimodals_low_redundancy():
    # P_t and P_v fully disagree -> R = 1 - JSD = 0.
    p_t = _delta(4, 0)
    p_v = _delta(4, 3)
    p_mm = np.array([0.5, 0.0, 0.0, 0.5])
    fv = compute_flowvector(p_t, p_v, p_mm)
    assert fv.r == pytest.approx(0.0, abs=1e-9)


def test_flowvector_array_roundtrip():
    fv = FlowVector(u_v=0.1, u_t=0.2, r=0.3, s=0.4)
    arr = fv.to_array()
    assert arr.shape == (4,)
    assert FlowVector.from_array(arr) == fv


def test_compute_flowvector_topk_handles_disjoint_vocabs():
    a = {"yes": math.log(0.6), "no": math.log(0.4)}
    b = {"maybe": math.log(0.5), "no": math.log(0.5)}
    c = {"yes": math.log(0.45), "no": math.log(0.45), "maybe": math.log(0.10)}
    fv = compute_flowvector_topk(a, b, c)
    assert isinstance(fv, FlowVector)
    assert all(np.isfinite(fv.to_array()))


def test_multistep_returns_per_step_and_mean():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(8), size=3)
    out = compute_flowvector_multistep(p, p, p)
    assert "step_0" in out and "step_1" in out and "step_2" in out and "mean" in out
    # All identical inputs -> all zeros.
    arr = out["mean"].to_array()
    assert np.allclose(arr, [0, 0, 1, 0], atol=1e-9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_flowvector(np.array([0.5, 0.5]), np.array([1.0]), np.array([0.5, 0.5]))
