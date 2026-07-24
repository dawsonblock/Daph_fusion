"""Surrogate Model Predictor (Phase 15-16).

Replaces the DummySurrogatePredictor (which ignored `y` in fit()) with
a real ensemble of tree-based regressors that actually learn from search
history.

Starts with simple, auditable models (Random Forest / Extra Trees /
Gradient Boosting) before neural surrogates, because the sample size is
initially small, tabular, irregular, and highly nonlinear.

The surrogate must actually use `y` in `fit()`. Track cross-validation
R², MAE, and ranking correlation. If the predictor cannot rank
candidates, don't use it for acquisition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SurrogateDiagnostics:
    """Diagnostics from surrogate cross-validation."""
    cv_r2: float = 0.0
    cv_mae: float = 0.0
    ranking_spearman: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    usable: bool = False
    reason: Optional[str] = None


class TreeSurrogatePredictor:
    """Real surrogate predictor using tree-based ensembles.

    Uses ExtraTreesRegressor by default (robust on small tabular datasets).
    The predictor MUST use `y` in fit(). Returns (mean, std) predictions
    for acquisition.

    Only marks itself as usable for acquisition if cross-validation shows
    positive ranking correlation (the predictor can rank candidates better
    than random).
    """

    def __init__(
        self,
        model_type: str = "extra_trees",
        n_estimators: int = 100,
        random_state: int = 42,
        min_samples_for_acquisition: int = 10,
    ) -> None:
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_samples_for_acquisition = min_samples_for_acquisition
        self.is_trained = False
        self._model: Optional[Any] = None
        self._diagnostics: Optional[SurrogateDiagnostics] = None

    def _make_model(self) -> Any:
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )

        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
        else:  # extra_trees (default)
            return ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> SurrogateDiagnostics:
        """Fit the surrogate on search history.

        Actually uses `y` (unlike the DummySurrogatePredictor).
        Computes cross-validation diagnostics.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.ravel()

        n_samples, n_features = X.shape
        self._diagnostics = SurrogateDiagnostics(
            n_samples=n_samples,
            n_features=n_features,
        )

        if n_samples < 3:
            self._diagnostics.reason = "insufficient_samples_for_cv"
            self.is_trained = False
            return self._diagnostics

        # Cross-validate
        from scipy.stats import spearmanr
        from sklearn.model_selection import cross_val_predict, KFold

        self._model = self._make_model()

        try:
            cv = KFold(n_splits=min(5, n_samples), shuffle=True,
                       random_state=self.random_state)
            y_pred = cross_val_predict(self._model, X, y, cv=cv)
            self._diagnostics.cv_mae = float(np.mean(np.abs(y_pred - y)))
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            self._diagnostics.cv_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
            rho, _ = spearmanr(y_pred, y)
            self._diagnostics.ranking_spearman = float(rho) if rho == rho else 0.0
        except Exception as e:
            self._diagnostics.reason = f"cv_failed: {e}"
            self.is_trained = False
            return self._diagnostics

        # Fit final model on all data
        self._model.fit(X, y)
        self.is_trained = True

        # Determine usability for acquisition
        if n_samples < self.min_samples_for_acquisition:
            self._diagnostics.usable = False
            self._diagnostics.reason = f"n_samples {n_samples} < min {self.min_samples_for_acquisition}"
        elif self._diagnostics.ranking_spearman <= 0:
            self._diagnostics.usable = False
            self._diagnostics.reason = "ranking_correlation_not_positive"
        else:
            self._diagnostics.usable = True

        return self._diagnostics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict (mean, std) for acquisition.

        Std is estimated from tree ensemble variance (for ExtraTrees/RF)
        or from prediction residuals (for GBM).
        """
        if not self.is_trained or self._model is None:
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # For tree ensembles with `estimators_`, use prediction variance
        if hasattr(self._model, "estimators_"):
            preds = np.array([est.predict(X) for est in self._model.estimators_])
            mean = preds.mean(axis=0).reshape(-1, 1)
            std = preds.std(axis=0).reshape(-1, 1) + 1e-4
        else:
            mean = self._model.predict(X).reshape(-1, 1)
            std = np.full_like(mean, self._diagnostics.cv_mae if self._diagnostics else 1.0)

        return mean, std

    @property
    def diagnostics(self) -> Optional[SurrogateDiagnostics]:
        return self._diagnostics


# Backward-compatible alias
DummySurrogatePredictor = TreeSurrogatePredictor


def constrained_expected_improvement(
    X_candidate: np.ndarray,
    surrogate: TreeSurrogatePredictor,
    best_y: float,
    feasibility_prob: Optional[np.ndarray] = None,
    xi: float = 0.01,
) -> np.ndarray:
    """Constrained Expected Improvement acquisition (Phase 16).

    a(c) = P(feasible | c) * EI(c)

    If feasibility_prob is None, assumes all candidates are feasible (P=1).
    """
    mean, std = surrogate.predict(X_candidate)
    mean = mean.ravel()
    std = std.ravel()

    # EI = (mu - best - xi) * Phi(z) + sigma * phi(z)
    # where z = (mu - best - xi) / sigma
    from scipy.stats import norm

    improvement = mean - best_y - xi
    z = improvement / np.clip(std, 1e-8, None)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    if feasibility_prob is not None:
        ei = ei * feasibility_prob.ravel()

    return ei
