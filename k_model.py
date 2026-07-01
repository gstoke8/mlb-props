from __future__ import annotations
#!/usr/bin/env python3
"""
K Model — XGBoost Poisson regression for pitcher strikeout props (k-v7).

Predicts expected K count (lambda) per start using XGBoost with a Poisson
objective, then converts lambda to over/under probability via the Poisson CDF.
Final output blends model probability with market-implied probability.

XGBoost replaces the NB GLM used in k-v6 to capture nonlinear feature
interactions (velocity × whiff rate, TTO × lineup, spin × CSW).
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

MODEL_VERSION = "k-v7"
MODEL_PATH = Path.home() / "mlb-props" / "models" / "k_model.pkl"
MARKET_BLEND = 0.30
MIN_TRAIN_ROWS = 500

K_FEATURE_COLS = [
    # Core K-generating ability
    "csw_rate_30d",           # Called Strike + Whiff %
    "k_rate_30d",             # actual K/BF (from bref)
    "k_rate_season",          # k_rate * 27 = K per 27 BF scale
    "k_rate_eb_30d",          # Empirical Bayes K rate (regressed toward league mean)
    "whiff_rate_30d",         # Whiff% = SwStr/Swings
    "swstr_rate_30d",         # SwStr% = SwStr/total pitches
    # Pitch quality by type
    "ff_whiff_rate_30d",      # 4-seam fastball whiff rate
    "sl_whiff_rate_30d",      # slider whiff rate
    "ch_whiff_rate_30d",      # changeup whiff rate
    "stuff_plus_30d",         # composite pitch quality proxy
    "ff_perceived_velo_30d",  # extension-adjusted effective fastball velocity
    "fb_spin_rate_30d",       # fastball (FF/SI) average spin rate (RPM)
    "breaking_spin_rate_30d", # breaking ball (SL/CU) average spin rate (RPM)
    "k_bb_ratio_30d",         # K/BB ratio — command quality signal
    # Workload / role context
    "avg_ip_30d",
    "is_opener_risk",
    "expected_tto3_pa_pct",   # fraction of projected PA in 3rd time-through-order
    # Recent performance momentum
    "k_prev_game",
    "k_prev3_weighted",
    # Opposing lineup
    "opp_k_rate_season",
    "opp_lineup_whiff_factor",
    "lineup_lhb_pct",
    "opp_lineup_o_swing_30d",
    # Game context
    "umpire_k_factor",
    "park_k_factor",
    "is_home",
    "days_rest",
]

# XGBoost hyperparameters — conservative to avoid overfitting on ~3k rows
_XGB_PARAMS: dict[str, Any] = {
    "objective":        "count:poisson",
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 8,
    "seed":             42,
    "verbosity":        0,
}
_XGB_ROUNDS = 300
_XGB_EARLY_STOP = 30

# ---------------------------------------------------------------------------
# KModel
# ---------------------------------------------------------------------------


class KModel:
    """XGBoost Poisson regression for pitcher strikeout count estimation."""

    def __init__(self) -> None:
        self.model: Any = None
        self.is_trained: bool = False
        self.feature_cols: list[str] = K_FEATURE_COLS
        self.train_meta: dict[str, Any] = {}
        self.feature_medians: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, training_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Fit XGBoost Poisson model on labelled rows.

        Each row must have all K_FEATURE_COLS plus ``actual_ks``.
        Returns dict with n_train, best_iteration, and feature importances.
        Raises ValueError if fewer than MIN_TRAIN_ROWS rows or columns are missing.
        """
        if len(training_rows) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Training requires at least {MIN_TRAIN_ROWS} rows; "
                f"got {len(training_rows)}."
            )

        first = training_rows[0]
        missing = {col for col in self.feature_cols if col not in first}
        if missing:
            raise ValueError(f"Training rows missing columns: {missing}")
        if "actual_ks" not in first:
            raise ValueError("Training rows must include 'actual_ks' column.")

        import xgboost as xgb

        X = np.array(
            [[row[col] for col in self.feature_cols] for row in training_rows],
            dtype=float,
        )

        # Compute per-feature medians for predict-time imputation
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        self.feature_medians = dict(zip(self.feature_cols, col_medians.tolist()))

        nan_count = int(np.sum(~np.isfinite(X)))
        if nan_count:
            logger.warning("train: %d NaN/inf values — imputing with feature medians", nan_count)
            for j, col in enumerate(self.feature_cols):
                bad = ~np.isfinite(X[:, j])
                X[bad, j] = self.feature_medians[col]

        y = np.array([max(int(row["actual_ks"]), 0) for row in training_rows], dtype=float)

        # 15% held-out val set for early stopping
        n = len(X)
        val_size = max(int(n * 0.15), 50)
        rng = np.random.default_rng(42)
        val_idx = rng.choice(n, val_size, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False

        dtrain = xgb.DMatrix(X[train_mask], label=y[train_mask], feature_names=self.feature_cols)
        dval   = xgb.DMatrix(X[~train_mask], label=y[~train_mask], feature_names=self.feature_cols)

        callbacks = [xgb.callback.EarlyStopping(rounds=_XGB_EARLY_STOP, metric_name="poisson-nloglik")]
        booster = xgb.train(
            _XGB_PARAMS,
            dtrain,
            num_boost_round=_XGB_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")],
            callbacks=callbacks,
            verbose_eval=False,
        )

        self.model = booster
        self.is_trained = True

        importances = booster.get_score(importance_type="gain")

        meta = {
            "n_train":           int(train_mask.sum()),
            "best_iteration":    booster.best_iteration,
            "model_version":     MODEL_VERSION,
            "feature_importances": importances,
            "feature_medians":   self.feature_medians,
        }
        self.train_meta = meta
        return {
            "n_train":           meta["n_train"],
            "best_iteration":    meta["best_iteration"],
            "coefficients":      importances,  # field name kept for _report() compat
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_lambda(self, features_dict: dict[str, Any]) -> float:
        """Return expected K count (lambda) for a single start."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() or load() first.")

        missing = {col for col in self.feature_cols if col not in features_dict}
        if missing:
            raise ValueError(f"features_dict missing columns: {missing}")

        import xgboost as xgb

        x = np.array(
            [[features_dict[col] for col in self.feature_cols]], dtype=float
        )
        for j, col in enumerate(self.feature_cols):
            if not np.isfinite(x[0, j]):
                x[0, j] = self.feature_medians.get(col, 0.0)

        dtest = xgb.DMatrix(x, feature_names=self.feature_cols)
        lam = float(self.model.predict(dtest)[0])
        return max(lam, 0.0)

    def k_over_probability(self, lambda_val: float, line: float) -> float:
        """Return P(actual_ks > line) using Poisson CDF."""
        k = math.floor(line)
        return float(1.0 - poisson.cdf(k, lambda_val))

    def predict_with_blend(
        self,
        features_dict: dict[str, Any],
        market_line: float,
        market_implied_over: float,
    ) -> dict[str, Any]:
        """Blend model over-probability with market-implied probability."""
        lam = self.predict_lambda(features_dict)
        model_prob_over = self.k_over_probability(lam, market_line)
        final_prob_over = (model_prob_over * (1.0 - MARKET_BLEND)) + (
            market_implied_over * MARKET_BLEND
        )
        edge = final_prob_over - market_implied_over

        return {
            "lambda":          lam,
            "model_prob_over": model_prob_over,
            "final_prob_over": final_prob_over,
            "edge":            edge,
            "market_implied":  market_implied_over,
            "line":            market_line,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        dest = Path(path) if path is not None else MODEL_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        logger.info("KModel saved to %s", dest)

    def load(self, path: Path | None = None) -> None:
        src = Path(path) if path is not None else MODEL_PATH
        if not src.exists():
            raise FileNotFoundError(f"No model file found at {src}")

        loaded: KModel = joblib.load(src)
        self.model = loaded.model
        self.is_trained = True
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
    db: Any | None = None,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Run K predictions for a list of pitcher-game matchups.

    Each dict must contain all K_FEATURE_COLS plus ``market_line`` and
    ``market_implied_over``. Returns results merging input with prediction keys.
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
