from __future__ import annotations
#!/usr/bin/env python3
"""
HR Model Calibration Module

Calibrates raw HR model probabilities using Platt scaling (isotonic regression),
generates calibration statistics, and produces reliability diagrams.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

from db import get_db
from hr_model import HR_FEATURE_COLS, HRModel, get_model

logger = logging.getLogger(__name__)

PLOTS_DIR = Path.home() / "mlb-props" / "plots"
DEFAULT_CALIBRATION_PLOT = PLOTS_DIR / "hr_calibration.png"


def _extract_features_and_labels(
    rows: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and label vector from a list of row dicts."""
    X = np.array([[row[col] for col in HR_FEATURE_COLS] for row in rows])
    y = np.array([int(row["actual_hr"]) for row in rows])
    return X, y


def run_platt_scaling(
    model: HRModel,
    holdout_rows: list[dict],
) -> tuple[CalibratedClassifierCV, dict[str, Any]]:
    """
    Fit isotonic calibration on held-out predictions.

    Parameters
    ----------
    model:
        Trained HRModel instance.
    holdout_rows:
        List of feature dicts each containing HR_FEATURE_COLS keys plus
        'actual_hr' (0 or 1).

    Returns
    -------
    calibrator:
        Fitted CalibratedClassifierCV (method='isotonic') wrapping the model.
    stats:
        Calibration statistics dict produced by calibration_stats().
    """
    if not holdout_rows:
        raise ValueError("holdout_rows must not be empty")

    X, y = _extract_features_and_labels(holdout_rows)

    raw_probs = model.predict_proba(X)[:, 1]

    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X, y)

    calibrated_probs = calibrator.predict_proba(X)[:, 1]

    stats = calibration_stats(y, calibrated_probs)
    logger.info(
        "Platt scaling complete — ECE=%.4f, holdout_n=%d",
        stats["ece"],
        len(holdout_rows),
    )
    return calibrator, stats


def calibration_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute per-bin calibration statistics and overall ECE.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_prob:
        Predicted probabilities (post-calibration or raw).
    n_bins:
        Number of equal-width probability buckets.

    Returns
    -------
    dict with keys:
        'bins'  – list of per-bin dicts
        'ece'   – Expected Calibration Error (float)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece_accumulator = 0.0
    total_n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        bin_center = float((lo + hi) / 2)

        if count == 0:
            bins.append(
                {
                    "bin_center": bin_center,
                    "predicted": bin_center,
                    "actual": 0.0,
                    "count": 0,
                }
            )
            continue

        predicted = float(y_prob[mask].mean())
        actual = float(y_true[mask].mean())
        bins.append(
            {
                "bin_center": bin_center,
                "predicted": predicted,
                "actual": actual,
                "count": count,
            }
        )
        ece_accumulator += (count / total_n) * abs(predicted - actual)

    return {"bins": bins, "ece": float(ece_accumulator)}


def plot_calibration(
    stats: dict[str, Any],
    save_path: str | Path | None = None,
) -> Path:
    """
    Render and save a reliability diagram.

    Parameters
    ----------
    stats:
        Output of calibration_stats().
    save_path:
        File path to save the PNG.  Defaults to
        ~/mlb-props/plots/hr_calibration.png.

    Returns
    -------
    Path to the saved file.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed — skipping calibration plot generation"
        )
        return Path(save_path or DEFAULT_CALIBRATION_PLOT)

    out_path = Path(save_path) if save_path else DEFAULT_CALIBRATION_PLOT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bins = [b for b in stats["bins"] if b["count"] > 0]
    predicted_vals = [b["predicted"] for b in bins]
    actual_vals = [b["actual"] for b in bins]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        predicted_vals,
        actual_vals,
        marker="o",
        color="steelblue",
        label=f"Model (ECE={stats['ece']:.4f})",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("HR Model — Reliability Diagram")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("Calibration plot saved to %s", out_path)
    return out_path


def evaluate_calibration(
    stats: dict[str, Any],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """
    Assess whether the model meets calibration quality thresholds.

    Passes when:
      - ECE < 0.05
      - No non-empty bin has |predicted - actual| > 0.10

    Parameters
    ----------
    stats:
        Output of calibration_stats().
    tolerance:
        Per-bin absolute tolerance for the summary (informational only;
        failing threshold is hard-coded at 0.10).

    Returns
    -------
    dict with keys: 'passes', 'failing_bins', 'ece', 'summary'
    """
    ece = stats["ece"]
    failing_bins: list[dict[str, Any]] = []

    for b in stats["bins"]:
        if b["count"] == 0:
            continue
        gap = abs(b["predicted"] - b["actual"])
        if gap > 0.10:
            failing_bins.append({**b, "gap": round(gap, 4)})

    passes = ece < 0.05 and len(failing_bins) == 0

    if passes:
        summary = f"PASS — ECE={ece:.4f}, all bins within 10% tolerance"
    else:
        reasons = []
        if ece >= 0.05:
            reasons.append(f"ECE={ece:.4f} >= 0.05")
        if failing_bins:
            centers = [b["bin_center"] for b in failing_bins]
            reasons.append(f"{len(failing_bins)} bin(s) exceed 10% gap: {centers}")
        summary = "FAIL — " + "; ".join(reasons)

    logger.info("Calibration evaluation: %s", summary)
    return {
        "passes": passes,
        "failing_bins": failing_bins,
        "ece": ece,
        "summary": summary,
    }


def run_full_calibration(
    training_rows: list[dict],
    holdout_fraction: float = 0.2,
) -> dict[str, Any]:
    """
    End-to-end calibration pipeline: split, train, calibrate, evaluate.

    Parameters
    ----------
    training_rows:
        All labelled rows (each must include HR_FEATURE_COLS + 'actual_hr').
    holdout_fraction:
        Fraction of rows reserved for calibration holdout.

    Returns
    -------
    Full report dict with keys:
        'calibrator', 'calibration_stats', 'evaluation',
        'plot_path', 'model_saved', 'holdout_n', 'train_n'
    """
    if not training_rows:
        raise ValueError("training_rows must not be empty")

    labels = [int(r["actual_hr"]) for r in training_rows]
    train_rows, holdout_rows = train_test_split(
        training_rows,
        test_size=holdout_fraction,
        stratify=labels,
        random_state=42,
    )

    logger.info(
        "Calibration split — train_n=%d, holdout_n=%d",
        len(train_rows),
        len(holdout_rows),
    )

    X_train, y_train = _extract_features_and_labels(train_rows)
    model = HRModel()
    model.fit(X_train, y_train)

    calibrator, cal_stats = run_platt_scaling(model, holdout_rows)
    evaluation = evaluate_calibration(cal_stats)
    plot_path = plot_calibration(cal_stats)

    model_saved = False
    if evaluation["passes"]:
        try:
            model.save()
            model_saved = True
            logger.info("Calibrated model saved successfully")
        except Exception:
            logger.exception("Failed to save calibrated model")

    report: dict[str, Any] = {
        "calibrator": calibrator,
        "calibration_stats": cal_stats,
        "evaluation": evaluation,
        "plot_path": str(plot_path),
        "model_saved": model_saved,
        "holdout_n": len(holdout_rows),
        "train_n": len(train_rows),
    }

    with get_db() as db:
        try:
            db.execute(
                """
                INSERT INTO model_runs (run_type, ece, passes, summary, plot_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "hr_calibration",
                    cal_stats["ece"],
                    int(evaluation["passes"]),
                    evaluation["summary"],
                    str(plot_path),
                ),
            )
        except Exception:
            logger.warning("Could not log calibration run to DB (table may not exist)")

    return report
