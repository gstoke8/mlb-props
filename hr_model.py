from __future__ import annotations
#!/usr/bin/env python3
"""
HR Model — Binary classification for Home Run player props.

Predicts P(batter hits HR in this game) using Logistic Regression
with a market-implied probability blend for final output.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "hr-v2"
MODEL_PATH = Path.home() / "mlb-props" / "models" / "hr_model.pkl"
MARKET_BLEND = 0.30   # weight on market implied prob
MIN_TRAIN_ROWS = 500  # refuse to train on fewer rows

HR_FEATURE_COLS = [
    # Batter power metrics
    "barrel_rate_30d",
    "barrel_rate_60d",
    "hard_hit_rate_30d",
    "xiso_30d",
    "avg_launch_angle_30d",
    "hr_rate_season",
    "pull_pct_30d",
    # Context
    "park_factor_h",
    "weather_hr_multiplier",
    # Opposing pitcher
    "pitcher_hr_rate_season",
    "pitcher_gb_pct",
    "batter_hand_vs_pitcher",
    "is_platoon_advantage",
    "lineup_spot",
    "bvp_factor",
    # Lineup quality signal (v2)
    "opp_lineup_xwoba",
    # Market
    "market_implied_prob",
    "line_movement",
]

# ---------------------------------------------------------------------------
# HRModel
# ---------------------------------------------------------------------------

class HRModel:
    """Logistic Regression model for HR probability estimation."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.is_trained: bool = False
        self.feature_cols: list[str] = HR_FEATURE_COLS
        self.train_meta: dict[str, Any] = {}

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

        Raises
        ------
        ValueError
            If fewer than MIN_TRAIN_ROWS rows are provided or required
            columns are missing.
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
        nan_count = int(np.sum(~np.isfinite(X)))
        if nan_count:
            logger.warning("train: %d NaN/inf values in X — replacing with 0", nan_count)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array([int(row["actual_hr"]) for row in training_rows], dtype=int)

        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        lr.fit(X, y)

        feature_importances = dict(zip(self.feature_cols, lr.coef_[0].tolist()))
        positive_rate = float(y.sum() / len(y))

        meta = {
            "n_train": len(training_rows),
            "positive_rate": positive_rate,
            "feature_importances": feature_importances,
            "model_version": MODEL_VERSION,
        }

        self.model = lr
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
        """Return raw model probability P(HR=1) for a single observation.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded yet.
        ValueError
            If required feature columns are missing from features_dict.
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
        prob: float = float(self.model.predict_proba(x)[0, 1])
        return prob

    def predict_with_blend(
        self,
        features_dict: dict[str, Any],
        market_implied_prob: float,
    ) -> dict[str, float]:
        """Blend model probability with market-implied probability.

        Parameters
        ----------
        features_dict:
            Feature dict for a single batter-game matchup.
        market_implied_prob:
            Probability implied by current market odds (0–1).

        Returns
        -------
        dict with model_prob, final_prob, edge, and market_implied.
        """
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
        """Serialize the model to disk using joblib.

        Parameters
        ----------
        path:
            Destination path. Defaults to MODEL_PATH.
        """
        dest = Path(path) if path is not None else MODEL_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        logger.info("HRModel saved to %s", dest)

    def load(self, path: Path | None = None) -> None:
        """Deserialise model state from disk.

        Parameters
        ----------
        path:
            Source path. Defaults to MODEL_PATH.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist at the given path.
        """
        src = Path(path) if path is not None else MODEL_PATH
        if not src.exists():
            raise FileNotFoundError(f"No model file found at {src}")

        loaded: HRModel = joblib.load(src)
        self.model = loaded.model
        self.is_trained = True
        self.train_meta = getattr(loaded, "train_meta", {})
        self.feature_cols = getattr(loaded, "feature_cols", HR_FEATURE_COLS)
        logger.info("HRModel loaded from %s", src)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_model_singleton: HRModel | None = None


def get_model() -> HRModel:
    """Return the module-level singleton HRModel.

    Loads from MODEL_PATH if a saved model exists; otherwise returns
    an untrained HRModel instance.
    """
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
    db: Any | None = None,  # noqa: ARG001  (reserved for future logging)
) -> list[dict[str, Any]]:
    """Run HR probability predictions for a list of game matchups.

    Parameters
    ----------
    game_matchups:
        List of feature dicts, one per batter-game matchup (output of
        hr_features.compute_hr_features / build_training_features).
        Each dict must contain all HR_FEATURE_COLS including
        ``market_implied_prob``.
    db:
        Optional database handle (reserved for future run logging).

    Returns
    -------
    List of result dicts. Each result contains all keys from the input
    matchup merged with the prediction keys: model_prob, final_prob,
    edge, market_implied, model_version.
    """
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
