#!/usr/bin/env python3
from __future__ import annotations
"""
Outs Model — Negative Binomial GLM for pitcher outs recorded props.

Predicts expected outs (lambda) per start using a NB GLM, then converts
lambda to over/under probability via the NB/Poisson CDF.  Final output
blends model probability with market-implied probability.

Key insight from research: pitch efficiency (P/IP) and manager/bullpen
context are the primary drivers of outs recorded — not raw stuff metrics
like whiff rate.  A pitcher who throws 19 P/IP with an active bullpen will
be pulled far earlier than one who throws 15 P/IP with a well-rested pen.
"""

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import nbinom, poisson

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "outs-v1"
MODEL_PATH = Path.home() / "mlb-props" / "models" / "outs_model.pkl"
MARKET_BLEND = 0.25
MIN_TRAIN_ROWS = 500

OUTS_FEATURE_COLS = [
    # Pitch efficiency — top predictors of how long a pitcher stays in
    "pitches_per_inning_30d",   # lower = more efficient = more outs
    "bb_rate_30d",              # high BB rate drives up pitch counts quickly
    # Workload context
    "avg_ip_last_3_starts",     # rolling usage trend
    "bullpen_usage_3d",         # bullpen fatigue → manager leaves SP in longer
    # Opposing lineup
    "opp_pa_per_k_30d",         # lineup PA/K — high PA/K = contact lineup = SP pulled earlier
    # Prior-start signals
    "prev_start_pitch_count",   # last start efficiency signal
    "manager_hook_factor",      # coach-specific tendency to use SP deep (>1) or short
    # Contact quality allowed
    "contact_quality_allowed_30d",  # xwOBA allowed — high contact = shorter starts
    # Game context
    "game_total",               # lower totals = pitching-favored matchup
    "k_rate_30d",               # K rate — high K = fewer baserunners = longer starts
    "days_rest",
    "is_home",
    "season_outs_per_start",    # season baseline depth
    "third_time_through_proximity",  # >0 when pitcher typically faces lineup 3x
]

# ---------------------------------------------------------------------------
# OutsModel
# ---------------------------------------------------------------------------


class OutsModel:
    """Negative Binomial GLM for pitcher outs recorded estimation."""

    def __init__(self) -> None:
        self.model: Any = None
        self.is_trained: bool = False
        self.feature_cols: list[str] = OUTS_FEATURE_COLS
        self.train_meta: dict[str, Any] = {}
        self._use_sklearn: bool = False
        self.feature_medians: dict[str, float] = {}
        self.nb_alpha: float = 0.0
        self._nb_active_mask: Any = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, training_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Fit NB GLM on labelled rows (each must have OUTS_FEATURE_COLS + ``actual_outs``).

        Returns dict with n_train, aic, coefficients, null_deviance, residual_deviance.
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
        if "actual_outs" not in first:
            raise ValueError("Training rows must include 'actual_outs' column.")

        X = np.array(
            [[row[col] for col in self.feature_cols] for row in training_rows],
            dtype=float,
        )

        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        self.feature_medians = dict(zip(self.feature_cols, col_medians.tolist()))

        nan_count = int(np.sum(~np.isfinite(X)))
        if nan_count:
            logger.warning("train: %d NaN/inf values in X — imputing with feature medians", nan_count)
            for j, col in enumerate(self.feature_cols):
                bad = ~np.isfinite(X[:, j])
                X[bad, j] = self.feature_medians[col]

        y = np.array([int(row["actual_outs"]) for row in training_rows], dtype=float)

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

    def _fit_statsmodels(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
        """Attempt statsmodels Negative Binomial GLM; falls back to sklearn on failure.

        NB is preferred over Poisson: outs per start are overdispersed relative to Poisson
        because bullpen decisions, manager tendencies, and game script all correlate the
        within-start outs distribution (dispersion ratio for 2024 MLB: ~1.12).
        """
        try:
            import statsmodels.api as sm
            from statsmodels.discrete.discrete_model import NegativeBinomial

            X_with_const = sm.add_constant(X, has_constant="add")
            col_names = ["const"] + self.feature_cols

            col_variances = np.var(X_with_const, axis=0)
            active_mask = col_variances > 1e-10
            active_mask[0] = True
            dropped = [n for n, m in zip(col_names, active_mask) if not m]
            if dropped:
                logger.info(
                    "_fit_statsmodels NB: dropping %d zero-variance column(s): %s",
                    len(dropped), dropped,
                )

            X_fit = X_with_const[:, active_mask]
            active_names = [n for n, m in zip(col_names, active_mask) if m]

            nb_model = NegativeBinomial(y, X_fit)
            result = nb_model.fit(method="bfgs", maxiter=300, disp=False)

            feature_params = result.params[:-1]
            alpha = float(result.params[-1])

            params_by_name = dict(zip(active_names, feature_params.tolist()))
            coefficients = {n: params_by_name.get(n, 0.0) for n in col_names}
            coefficients["nb_alpha"] = alpha

            self._nb_active_mask = active_mask
            self.model = result
            self._use_sklearn = False
            self.nb_alpha = alpha

            return {
                "n_train": len(y),
                "aic": float(result.aic),
                "coefficients": coefficients,
                "null_deviance": float(result.llnull * -2),
                "residual_deviance": float(result.llf * -2),
                "nb_alpha": alpha,
            }

        except ImportError:
            logger.warning("statsmodels not available; falling back to sklearn PoissonRegressor.")
            return None
        except Exception as exc:
            logger.warning(
                "statsmodels NB fit failed (%s); falling back to sklearn PoissonRegressor.", exc,
            )
            return None

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Fit sklearn PoissonRegressor as statsmodels fallback."""
        from sklearn.linear_model import PoissonRegressor

        reg = PoissonRegressor(alpha=0.0, max_iter=300)
        reg.fit(X, y)

        coefficients = dict(zip(self.feature_cols, reg.coef_.tolist()))
        y_pred = reg.predict(X)
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
        """Return expected outs count (lambda) for a single start."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() or load() first.")

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
            if self._nb_active_mask is not None:
                x_const = x_const[:, self._nb_active_mask]
            lam = float(self.model.predict(x_const)[0])

        return max(lam, 0.0)

    def outs_over_probability(self, lambda_val: float, line: float) -> float:
        """Return P(actual_outs > line) using NB CDF when alpha is available, else Poisson."""
        k = math.floor(line)
        if self.nb_alpha > 0 and not self._use_sklearn:
            n = 1.0 / self.nb_alpha
            p = 1.0 / (1.0 + self.nb_alpha * max(lambda_val, 1e-9))
            return float(1.0 - nbinom.cdf(k, n, p))
        return float(1.0 - poisson.cdf(k, lambda_val))

    def predict_with_blend(
        self,
        features_dict: dict[str, Any],
        market_line: float,
        market_implied_over: float,
    ) -> dict[str, Any]:
        """Blend model over-probability with market-implied probability."""
        lam = self.predict_lambda(features_dict)
        model_prob_over = self.outs_over_probability(lam, market_line)
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
        logger.info("OutsModel saved to %s", dest)

    def load(self, path: Path | None = None) -> None:
        """Deserialise model state from disk (defaults to MODEL_PATH)."""
        src = Path(path) if path is not None else MODEL_PATH
        if not src.exists():
            raise FileNotFoundError(f"No model file found at {src}")

        loaded: OutsModel = joblib.load(src)
        self.model = loaded.model
        self.is_trained = True
        self._use_sklearn = getattr(loaded, "_use_sklearn", False)
        self.train_meta = getattr(loaded, "train_meta", {})
        self.feature_cols = getattr(loaded, "feature_cols", OUTS_FEATURE_COLS)
        self.feature_medians = getattr(loaded, "feature_medians", {})
        self.nb_alpha = getattr(loaded, "nb_alpha", 0.0)
        self._nb_active_mask = getattr(loaded, "_nb_active_mask", None)
        logger.info("OutsModel loaded from %s", src)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_model_singleton: OutsModel | None = None


def get_model() -> OutsModel:
    """Singleton OutsModel — loads from MODEL_PATH if it exists, else returns untrained instance."""
    global _model_singleton  # noqa: PLW0603
    if _model_singleton is not None:
        return _model_singleton
    instance = OutsModel()
    if MODEL_PATH.exists():
        try:
            instance.load()
        except Exception:
            logger.exception(
                "Failed to load OutsModel from %s; using untrained instance.", MODEL_PATH
            )
    _model_singleton = instance
    return _model_singleton


def predict_game_outs(
    game_matchups: list[dict[str, Any]],
    db: Any | None = None,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Run outs predictions for a list of pitcher-game matchups.

    Each dict must contain all OUTS_FEATURE_COLS plus ``market_line`` and
    ``market_implied_over``.  Returns result dicts with lambda, model_prob_over,
    final_prob_over, edge, market_implied, line, model_version.
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
