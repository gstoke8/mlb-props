from __future__ import annotations
#!/usr/bin/env python3
"""
HR Model — Binary classification for Home Run player props.

hr-v4: XGBoost + isotonic calibration, 21-feature set including batted-ball
quality, pitcher contact-quality allowed, weather, and empirical-Bayes HR rate.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "hr-v4"
MODEL_PATH = Path.home() / "mlb-props" / "models" / "hr_model.pkl"
MARKET_BLEND = 0.30   # weight on market implied prob
MIN_TRAIN_ROWS = 500

HR_FEATURE_COLS = [
    # Batter power (core)
    "barrel_rate_30d",
    "barrel_rate_60d",
    "hard_hit_rate_30d",
    "xiso_30d",
    "avg_launch_angle_30d",
    "hr_rate_season",
    "hr_rate_eb_30d",       # empirical-Bayes blend toward league mean
    "pull_pct_30d",
    # Batted ball quality (new in v4)
    "fly_ball_rate_30d",
    "sweet_spot_pct_30d",
    # Context
    "park_factor_h",
    "game_temp_f",
    "wind_hr_factor",       # hr_weather_multiplier from WeatherClient
    # Opposing pitcher
    "pitcher_hr_rate_season",
    "pitcher_gb_pct",
    "pitcher_barrel_rate_allowed",
    "pitcher_hard_hit_pct_allowed",
    # Handedness / matchup
    "batter_hand_vs_pitcher",
    "is_platoon_advantage",
    "pull_x_platoon",       # pull_pct_30d * is_platoon_advantage interaction
    "bvp_factor",
]

# ---------------------------------------------------------------------------
# HRModel
# ---------------------------------------------------------------------------

class HRModel:
    """XGBoost + isotonic calibration model for HR probability estimation."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.is_trained: bool = False
        self.feature_cols: list[str] = HR_FEATURE_COLS
        self.train_meta: dict[str, Any] = {}
        self.feature_medians: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, training_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Fit the model on labelled training rows.

        Parameters
        ----------
        training_rows:
            List of dicts, each containing all feature columns plus
            ``actual_hr`` (0 or 1).

        Returns
        -------
        dict with n_train, positive_rate, and feature_importances.
        """
        if len(training_rows) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Training requires at least {MIN_TRAIN_ROWS} rows; "
                f"got {len(training_rows)}."
            )

        missing = {col for col in self.feature_cols if col not in training_rows[0]}
        if missing:
            raise ValueError(f"Training rows missing columns: {missing}")

        if "actual_hr" not in training_rows[0]:
            raise ValueError("Training rows must include 'actual_hr' column.")

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

        # Binarize: P(at least one HR) — multi-HR games count as positive
        y = np.array([1 if int(row["actual_hr"]) >= 1 else 0 for row in training_rows], dtype=int)
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        positive_rate = float(n_pos / len(y))

        if _XGB_AVAILABLE:
            # scale_pos_weight compensates for heavy class imbalance in HR data
            spw = float(n_neg / max(n_pos, 1))
            base_model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        else:
            logger.warning("xgboost not available; falling back to LogisticRegression for hr-v4")
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
            base_model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")

        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        calibrated.fit(X, y)

        # Feature importances: average across isotonic calibration folds
        feature_importances: dict[str, float] = {col: 0.0 for col in self.feature_cols}
        if _XGB_AVAILABLE:
            try:
                fi_arrays = [
                    c.estimator.feature_importances_
                    for c in calibrated.calibrated_classifiers_
                ]
                mean_fi = np.mean(fi_arrays, axis=0)
                feature_importances = dict(zip(self.feature_cols, mean_fi.tolist()))
            except AttributeError:
                pass

        meta = {
            "n_train": len(training_rows),
            "positive_rate": positive_rate,
            "feature_importances": feature_importances,
            "model_version": MODEL_VERSION,
            "feature_medians": self.feature_medians,
        }

        self.model = calibrated
        self.is_trained = True
        self.train_meta = meta

        return {
            "n_train": meta["n_train"],
            "positive_rate": meta["positive_rate"],
            "feature_importances": meta["feature_importances"],
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, features_dict: dict[str, Any]) -> float:
        """Return raw model probability P(HR=1) for a single observation."""
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

        prob: float = float(self.model.predict_proba(x)[0, 1])
        return prob

    def predict_with_blend(
        self,
        features_dict: dict[str, Any],
        market_implied_prob: float,
    ) -> dict[str, float]:
        """Blend model probability with market-implied probability."""
        model_prob = self.predict_proba(features_dict)
        final_prob = (model_prob * (1 - MARKET_BLEND)) + (
            market_implied_prob * MARKET_BLEND
        )
        edge = final_prob - market_implied_prob

        return {
            "model_prob": model_prob,
            "final_prob": final_prob,
            "edge": edge,
            "market_implied": market_implied_prob,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        dest = Path(path) if path is not None else MODEL_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        logger.info("HRModel saved to %s", dest)

    def load(self, path: Path | None = None) -> None:
        src = Path(path) if path is not None else MODEL_PATH
        if not src.exists():
            raise FileNotFoundError(f"No model file found at {src}")

        loaded: HRModel = joblib.load(src)

        # Detect incompatible hr-v3 pickles and refuse to use them.
        loaded_version = getattr(loaded, "train_meta", {}).get("model_version", "")
        if loaded_version and loaded_version != MODEL_VERSION:
            logger.warning(
                "Loaded %s model; feature set incompatible with %s. "
                "Marking untrained — Poisson fallback active until retrain.",
                loaded_version, MODEL_VERSION,
            )
            self.is_trained = False
            return

        self.model = loaded.model
        self.is_trained = True
        self.train_meta = getattr(loaded, "train_meta", {})
        self.feature_cols = getattr(loaded, "feature_cols", HR_FEATURE_COLS)
        self.feature_medians = getattr(loaded, "feature_medians", {})
        logger.info("HRModel loaded from %s", src)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_model_singleton: HRModel | None = None


def get_model() -> HRModel:
    """Return the module-level singleton HRModel."""
    global _model_singleton  # noqa: PLW0603

    if _model_singleton is not None:
        return _model_singleton

    instance = HRModel()
    if MODEL_PATH.exists():
        try:
            instance.load()
        except Exception:
            logger.exception("Failed to load HRModel from %s; using untrained instance.", MODEL_PATH)

    _model_singleton = instance
    return _model_singleton


def predict_game_hrs(
    game_matchups: list[dict[str, Any]],
    db: Any | None = None,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Run HR probability predictions for a list of game matchups."""
    model = get_model()
    results: list[dict[str, Any]] = []

    for matchup in game_matchups:
        market_implied = float(matchup.get("market_implied_prob", 0.0))
        try:
            prediction = model.predict_with_blend(matchup, market_implied)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Skipping matchup due to prediction error: %s", exc)
            continue

        result = {
            **matchup,
            **prediction,
            "model_version": MODEL_VERSION,
        }
        results.append(result)

    return results
