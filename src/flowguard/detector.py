"""One-class FlowGuard detector.

The detector wraps a ``sklearn.ensemble.IsolationForest`` trained on benign
FlowVectors. The decision rule reported in the paper uses the standard
``contamination='auto'`` thresholding (decision boundary at normalized
score 0.5). At inference time the detector exposes both:

* ``predict`` (boolean: ``True`` = anomalous = block)
* ``score`` (continuous anomaly score, higher = more anomalous,
  the AUROC sweep uses this directly)
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class DetectorConfig:
    contamination: str | float = "auto"
    n_estimators: int = 200
    max_samples: int | str = "auto"
    random_state: int = 0
    n_jobs: int = -1


class FlowGuardDetector:
    """One-class FlowVector anomaly detector.

    Parameters
    ----------
    config:
        Hyperparameters; defaults match the paper's main configuration.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._model: IsolationForest | None = None
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FlowGuardDetector":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"Expected (N, 4) FlowVectors; got {X.shape}")
        self._train_mean = X.mean(axis=0)
        self._train_std = X.std(axis=0) + 1e-9
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self._model.fit(self._standardize(X))
        return self

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self._train_mean is None or self._train_std is None:
            raise RuntimeError("Detector not fitted")
        return (X - self._train_mean) / self._train_std

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a boolean mask: True = anomalous (block)."""
        if self._model is None:
            raise RuntimeError("Detector not fitted")
        Xs = self._standardize(np.asarray(X, dtype=np.float64))
        # IsolationForest.predict: +1 = inlier, -1 = outlier.
        return self._model.predict(Xs) == -1

    def score(self, X: np.ndarray) -> np.ndarray:
        """Continuous anomaly score (higher = more anomalous).

        The Isolation Forest ``score_samples`` returns higher values for
        inliers, so we negate to match the convention used elsewhere
        (AUROC sweeps, threshold tuning).
        """
        if self._model is None:
            raise RuntimeError("Detector not fitted")
        Xs = self._standardize(np.asarray(X, dtype=np.float64))
        return -self._model.score_samples(Xs)

    def threshold_at(self, X_benign: np.ndarray, fpr: float) -> float:
        """Pick the score threshold that yields ``fpr`` on benign data.

        Used for the FPR-controlled deployment configuration mentioned in
        Section 5.2 (``2.4% FPR under default threshold``).
        """
        scores = self.score(X_benign)
        return float(np.quantile(scores, 1.0 - fpr))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "model": self._model,
                    "train_mean": self._train_mean,
                    "train_std": self._train_std,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "FlowGuardDetector":
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        det = cls(config=blob["config"])
        det._model = blob["model"]
        det._train_mean = blob["train_mean"]
        det._train_std = blob["train_std"]
        return det

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        if self._model is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "n_estimators": self.config.n_estimators,
            "contamination": self.config.contamination,
            "train_mean": (self._train_mean.tolist() if self._train_mean is not None else None),
            "train_std": (self._train_std.tolist() if self._train_std is not None else None),
        }


def save_summary(detector: FlowGuardDetector, path: str | Path) -> None:
    Path(path).write_text(json.dumps(detector.summary(), indent=2), encoding="utf-8")
