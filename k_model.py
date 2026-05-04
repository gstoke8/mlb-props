from __future__ import annotations
#!/usr/bin/env python3
"""
K Model — Poisson GLM regression for pitcher strikeout props.

Predicts expected K count (lambda) per start using a Poisson GLM, then
converts lambda to over/under probability via the Poisson CDF.  Final
output blends model probability with market-implied probability.
"""

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "k-v4"
MODEL_PATH = Path.home() / "mlb-props" / "models" / "k_model.pkl"
MARKET_BLEND = 0.30
MIN_TRAIN_ROWS = 500

K_FEATURE_COLS = [
    # Core pitcher effectiveness
    "csw_rate_30d",
    "k_rate_30d",
    "k_rate_season",
    "whiff_rate_30d",
    # Pitch-type stuff metrics
    "swstr_rate_30d",
    "ff_whiff_rate_30d",
    "sl_whiff_rate_30d",
    "ch_whiff_rate_30d",
    # Opposing lineup — batter matchup (v4)
    "opp_k_rate_season",       # opposing lineup K/PA season
    "opp_lineup_whiff_factor", # pitcher_mix × lineup whiff rates / league avg
    "lineup_lhb_pct",          # fraction left-handed batters in lineup
    # Game context
    "umpire_k_factor",
    "park_k_factor",
    "is_home",
    "days_rest",
    # Role context
    "avg_ip_30d",
    "is_opener_risk",
]

# ---------------------------------------------------------------------------
# KModel
# ---------------------------------------------------------------------------


class KModel:
    """Poisson GLM for pitcher strikeout count estimation."""

    def __init__(self) -> None:
        self.model: Any = None
        self.is_trained: bool = False
        self.feature_cols: list[str] = K_FEATURE_COLS
        self.train_meta: dict[str, Any] = {}
        self._use_sklearn: bool = False
        self.feature_medians: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, training_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Fit Poisson GLM on labelled rows (each must have all K_FEATURE_COLS + ``actual_ks``).

        Returns dict with n_train, aic, coefficients, null_deviance, residual_deviance.
        Raises ValueError if fewer than MIN_TRAIN_ROWS rows or columns are missing.
        """
        if len(training_rows) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Training requires at least {MIN_TRAIN_ROWS} rows; "
                f"got {len(training_rows)}."
            )

        first = training_rows[0]
        missing_features = {col for col in self.feature_cols if col not in first}
        if missing_features:
            raise ValueError(f"Training rows missing columns: {missing_features}")
        if "actual_ks" not in first:
            raise ValueError("Training rows must include 'actual_ks' column.")

        X = np.array(
            [[row[col] for col in self.feature_cols] for row in training_rows],
            dtype=float,
        )

        # Compute per-feature medians and store for predict-time imputation
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        self.feature_medians = dict(zip(self.feature_cols, col_medians.tolist()))

        nan_count = int(np.sum(~np.isfinite(X)))
        if nan_count:
            logger.warning("train: %d NaN/inf values in X — imputing with feature medians", nan_count)
            for j, col in enumerate(self.feature_cols):
                bad = ~np.isfinite(X[:, j])
                X[bad, j] = self.feature_medians[col]

        y = np.array([int(row["actual_ks"]) for row in training_rows], dtype=float)

        meta = self._fit_statsmodels(X, y)
        if meta is None:
            meta = self._fit_sklearn(X, y)

        self.is_trained = True
        self.train_meta = {**meta, "model_version": MODEL_VERSION, "feature_medians": self.feature_medians}
        return {
            "n_train": meta["n_train"],
            "aic": meta["aic"],
            "coefficients": meta["coefficients"],
            "null_deviance": meta["null_deviance"],
            "residual_deviance": meta["residual_deviance"],
        }

    def _fit_statsmodels(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict[str, Any] | None:
        """Attempt statsmodels Poisson GLM; returns None on failure so caller falls back to sklearn."""
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.families import Poisson

            X_with_const = sm.add_constant(X, has_constant="add")
            glm = sm.GLM(y, X_with_const, family=Poisson())
            result = glm.fit()

            col_names = ["const"] + self.feature_cols
            coefficients = dict(zip(col_names, result.params.tolist()))

            self.model = result
            self._use_sklearn = False

            return {
                "n_train": len(y),
                "aic": float(result.aic),
                "coefficients": coefficients,
                "null_deviance": float(result.null_deviance),
                "residual_deviance": float(result.deviance),
            }

        except ImportError:
            logger.warning(
                "statsmodels not available; falling back to sklearn PoissonRegressor."
            )
            return None
        except Exception as exc:
            logger.warning(
                "statsmodels fit failed (%s); falling back to sklearn PoissonRegressor.",
                exc,
            )
            return None

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Fit sklearn PoissonRegressor as statsmodels fallback."""
        from sklearn.linear_model import PoissonRegressor

        reg = PoissonRegressor(alpha=0.0, max_iter=300)
        reg.fit(X, y)

        coefficients = dict(zip(self.feature_cols, reg.coef_.tolist()))
        y_pred = reg.predict(X)
        # Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))
        safe_ratio = np.where(y > 0, y / np.maximum(y_pred, 1e-10), 1.0)
        deviance = float(2.0 * np.sum(np.where(y > 0, y * np.log(safe_ratio), 0.0) - (y - y_pred)))
        null_mu = float(y.mean())
        null_ratio = np.where(y > 0, y / null_mu, 1.0)
        null_deviance = float(2.0 * np.sum(np.where(y > 0, y * np.log(null_ratio), 0.0) - (y - null_mu)))
        aic = deviance + 2.0 * (len(self.feature_cols) + 1)

        self.model = reg
        self._use_sklearn = True

        return {
            "n_train": len(y),
            "aic": aic,
            "coefficients": coefficients,
            "null_deviance": null_deviance,
            "residual_deviance": deviance,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_lambda(self, features_dict: dict[str, Any]) -> float:
        """Return expected K count (lambda) for a single start.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded.
        ValueError
            If required feature columns are missing.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError(
                "Model is not trained. Call train() or load() first."
            )

        missing = {col for col in self.feature_cols if col not in features_dict}
        if missing:
            raise ValueError(f"features_dict missing columns: {missing}")

        x = np.array(
            [[features_dict[col] for col in self.feature_cols]], dtype=float
        )
        for j, col in enumerate(self.feature_cols):
            if not np.isfinite(x[0, j]):
                x[0, j] = self.feature_medians.get(col, 0.0)

        if self._use_sklearn:
            lam: float = float(self.model.predict(x)[0])
        else:
            import statsmodels.api as sm

            x_const = sm.add_constant(x, has_constant="add")
            lam = float(self.model.predict(x_const)[0])

        return max(lam, 0.0)

    def k_over_probability(self, lambda_val: float, line: float) -> float:
        """Return P(actual_ks > line) using the Poisson CDF.

        Parameters
        ----------
        lambda_val:
            Expected K count from predict_lambda.
        line:
            The prop line (e.g. 5.5).

        Returns
        -------
        Probability of going OVER the line (0–1).
        """
        return float(1.0 - poisson.cdf(math.floor(line), lambda_val))

    def predict_with_blend(
        self,
        features_dict: dict[str, Any],
        market_line: float,
        market_implied_over: float,
    ) -> dict[str, Any]:
        """Blend model over-probability with market-implied probability.

        Parameters
        ----------
        features_dict:
            Feature dict for a single pitcher-game matchup.
        market_line:
            The strikeout prop line (e.g. 5.5).
        market_implied_over:
            Market-implied probability of going OVER (0–1).

        Returns
        -------
        dict with lambda, model_prob_over, final_prob_over, edge,
        market_implied, line.
        """
        lam = self.predict_lambda(features_dict)
        model_prob_over = self.k_over_probability(lam, market_line)
        final_prob_over = (model_prob_over * (1.0 - MARKET_BLEND)) + (
            market_implied_over * MARKET_BLEND
        )
        edge = final_prob_over - market_implied_over

        return {
            "lambda": lam,
            "model_prob_over": model_prob_over,
            "final_prob_over": final_prob_over,
            "edge": edge,
            "market_implied": market_implied_over,
            "line": market_line,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Serialize model state to disk using joblib (defaults to MODEL_PATH)."""
        dest = Path(path) if path is not None else MODEL_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        logger.info("KModel saved to %s", dest)

    def load(self, path: Path | None = None) -> None:
        """Deserialise model state from disk (defaults to MODEL_PATH).

        Raises FileNotFoundError if no model file exists at the given path.
        """
        src = Path(path) if path is not None else MODEL_PATH
        if not src.exists():
            raise FileNotFoundError(f"No model file found at {src}")

        loaded: KModel = joblib.load(src)
        self.model = loaded.model
        self.is_trained = True
        self._use_sklearn = getattr(loaded, "_use_sklearn", False)
        self.train_meta = getattr(loaded, "train_meta", {})
        self.feature_cols = getattr(loaded, "feature_cols", K_FEATURE_COLS)
        self.feature_medians = getattr(loaded, "feature_medians", {})
        logger.info("KModel loaded from %s", src)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_model_singleton: KModel | None = None


def get_model() -> KModel:
    """Singleton KModel — loads from MODEL_PATH if it exists, else returns untrained instance."""
    global _model_singleton  # noqa: PLW0603
    if _model_singleton is not None:
        return _model_singleton
    instance = KModel()
    if MODEL_PATH.exists():
        try:
            instance.load()
        except Exception:
            logger.exception(
                "Failed to load KModel from %s; using untrained instance.", MODEL_PATH
            )
    _model_singleton = instance
    return _model_singleton


def predict_game_ks(
    game_matchups: list[dict[str, Any]],
    db: Any | None = None,  # noqa: ARG001  (reserved for future logging)
) -> list[dict[str, Any]]:
    """Run K predictions for a list of pitcher-game matchups.

    Each dict in game_matchups must contain all K_FEATURE_COLS plus
    ``market_line`` and ``market_implied_over``.  Returns a list of
    result dicts merging the input matchup with prediction keys:
    lambda, model_prob_over, final_prob_over, edge, market_implied,
    line, model_version.
    """
    model = get_model()
    results: list[dict[str, Any]] = []
    for matchup in game_matchups:
        market_line = float(matchup.get("market_line", 0.0))
        market_implied_over = float(matchup.get("market_implied_over", 0.0))
        try:
            prediction = model.predict_with_blend(matchup, market_line, market_implied_over)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Skipping matchup due to prediction error: %s", exc)
            continue
        results.append({**matchup, **prediction, "model_version": MODEL_VERSION})
    return results
