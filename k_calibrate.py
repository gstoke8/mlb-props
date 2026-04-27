from __future__ import annotations
#!/usr/bin/env python3
"""
K Model Calibration Module

Calibrates raw K model over/under probabilities (derived from Poisson output),
generates calibration statistics, and produces reliability diagrams.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from db import get_db
from k_model import K_FEATURE_COLS, KModel, get_model

logger = logging.getLogger(__name__)

PLOTS_DIR = Path.home() / "mlb-props" / "plots"
DEFAULT_CALIBRATION_PLOT = PLOTS_DIR / "k_calibration.png"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_over_labels(rows: list[dict]) -> np.ndarray:
    """Return binary array: 1 if actual_ks > market_line, else 0."""
    return np.array(
        [int(row["actual_ks"] > row["market_line"]) for row in rows],
        dtype=int,
    )


def _predicted_over_probs(model: KModel, rows: list[dict]) -> np.ndarray:
    """Get model's over-probability for each row via predict_with_blend."""
    probs: list[float] = []
    for row in rows:
        result = model.predict_with_blend(row)
        probs.append(float(result["over_prob"]))
    return np.array(probs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibration_stats(
    y_true_over: list[int] | np.ndarray,
    y_prob_over: list[float] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute per-bin calibration statistics and overall ECE.

    Parameters
    ----------
    y_true_over:
        Binary labels — 1 if pitcher went over the market line, else 0.
    y_prob_over:
        Model-predicted over-probability for each sample.
    n_bins:
        Number of equal-width probability buckets.

    Returns
    -------
    dict with keys:
        'bins'  – list of per-bin dicts (bin_center, predicted, actual, count)
        'ece'   – Expected Calibration Error (float)
    """
    y_true = np.asarray(y_true_over, dtype=float)
    y_prob = np.asarray(y_prob_over, dtype=float)

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
    Render and save a K model reliability diagram.

    Parameters
    ----------
    stats:
        Output of calibration_stats().
    save_path:
        File path to save the PNG. Defaults to
        ~/mlb-props/plots/k_calibration.png.

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

    ax.set_xlabel("Mean predicted over-probability")
    ax.set_ylabel("Fraction of overs (actual)")
    ax.set_title("K Model — Reliability Diagram")
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
    Assess whether the K model meets calibration quality thresholds.

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
        reasons: list[str] = []
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
    End-to-end K calibration pipeline: stratified split, train, evaluate.

    Parameters
    ----------
    training_rows:
        All labelled rows; each must contain K_FEATURE_COLS plus
        'actual_ks' and 'market_line'.
    holdout_fraction:
        Fraction of rows reserved for calibration holdout.

    Returns
    -------
    Full report dict with keys:
        'calibration_stats', 'evaluation', 'plot_path',
        'holdout_n', 'train_n'
    """
    if not training_rows:
        raise ValueError("training_rows must not be empty")

    over_labels = [int(r["actual_ks"] > r["market_line"]) for r in training_rows]
    train_rows, holdout_rows = train_test_split(
        training_rows,
        test_size=holdout_fraction,
        stratify=over_labels,
        random_state=42,
    )

    logger.info(
        "K calibration split — train_n=%d, holdout_n=%d",
        len(train_rows),
        len(holdout_rows),
    )

    model = KModel()
    model.train(train_rows)

    holdout_probs = _predicted_over_probs(model, holdout_rows)
    holdout_labels = _build_over_labels(holdout_rows)

    cal_stats = calibration_stats(holdout_labels, holdout_probs)
    evaluation = evaluate_calibration(cal_stats)
    plot_path = plot_calibration(cal_stats)

    report: dict[str, Any] = {
        "calibration_stats": cal_stats,
        "evaluation": evaluation,
        "plot_path": str(plot_path),
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
                    "k_calibration",
                    cal_stats["ece"],
                    int(evaluation["passes"]),
                    evaluation["summary"],
                    str(plot_path),
                ),
            )
        except Exception:
            logger.warning("Could not log calibration run to DB (table may not exist)")
    return report
