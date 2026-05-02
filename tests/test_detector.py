"""End-to-end test for the FlowGuard one-class detector on synthetic clusters.

Generates a tight benign cluster around (0, 0, 1, 0) (the identity-distribution
fixed point of FlowVectors) and an adversarial cluster shifted by ~5 sigma.
The detector should cleanly separate them with AUROC > 0.95.
"""
from __future__ import annotations

import numpy as np

from flowguard.detector import DetectorConfig, FlowGuardDetector
from flowguard.eval.metrics import auroc


def _make_clusters(seed: int = 0, n: int = 800):
    rng = np.random.default_rng(seed)
    benign = rng.normal(loc=[0.05, 0.05, 0.95, 0.0], scale=0.05, size=(n, 4))
    adv = rng.normal(loc=[0.6, 0.5, 0.3, -0.4], scale=0.1, size=(n // 2, 4))
    return benign, adv


def test_detector_separates_synthetic_clusters():
    benign, adv = _make_clusters()
    det = FlowGuardDetector(DetectorConfig(random_state=0)).fit(benign)

    s_b = det.score(benign)
    s_a = det.score(adv)
    au = auroc(s_a, s_b)
    assert au > 0.95, f"AUROC dropped to {au}"


def test_detector_save_load_roundtrip(tmp_path):
    benign, adv = _make_clusters()
    det = FlowGuardDetector().fit(benign)
    pred_a = det.predict(adv)

    path = tmp_path / "det.pkl"
    det.save(path)
    det2 = FlowGuardDetector.load(path)
    np.testing.assert_array_equal(pred_a, det2.predict(adv))


def test_threshold_at_fpr_matches_target():
    benign, _ = _make_clusters()
    det = FlowGuardDetector().fit(benign)
    t = det.threshold_at(benign, fpr=0.05)
    realized = (det.score(benign) > t).mean()
    # Empirical FPR should be close to target on the *training* sample.
    assert abs(realized - 0.05) < 0.02
