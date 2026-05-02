"""Tests for ASR / AUROC / FPR / F1."""
from __future__ import annotations

import numpy as np
import pytest

from flowguard.eval.metrics import (
    attack_success_rate,
    auroc,
    best_f1,
    f1_at_threshold,
    false_positive_rate,
)


def test_asr_counts_only_unblocked_harms():
    harm = [True, True, True, False]
    blocked = [False, True, False, False]
    assert attack_success_rate(harm, blocked) == pytest.approx(2 / 4)


def test_fpr_counts_blocked_benigns():
    blocked = [True, False, False, True, False]
    assert false_positive_rate(blocked) == pytest.approx(2 / 5)


def test_auroc_perfect_separation_is_one():
    pos = [1.0, 2.0, 3.0]
    neg = [-1.0, -2.0, -3.0]
    assert auroc(pos, neg) == pytest.approx(1.0)


def test_auroc_random_is_around_half():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=200)
    neg = rng.normal(size=200)
    assert 0.4 < auroc(pos, neg) < 0.6


def test_best_f1_returns_separating_threshold():
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = [False, False, False, True, True, True]
    t, f1 = best_f1(s, y)
    assert f1 == pytest.approx(1.0)
    assert 0.3 <= t <= 0.7
