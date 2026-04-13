#!/usr/bin/env python3
"""
HR Backtest — Evaluates the HR model against historical data.

Computes CLV, hit rate by confidence tier, and ROI using both a simple
70/30 chronological split and walk-forward validation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from db import calculate_clv, calculate_pl, get_db
from hr_features import build_training_features
from hr_model import HRModel, get_model
from sizing import MIN_EDGE, classify_confidence, compute_units, edge_tier_units

log = logging.getLogger(__name__)

BACKTEST_DIR = Path.home() / "mlb-props" / "backtest"
BACKTEST_PATH = BACKTEST_DIR / "hr_backtest.json"
CLV_THRESHOLD = 1.5       # minimum avg CLV % to pass
HIT_RATE_THRESHOLD = 0.55  # minimum hit rate on MEDIUM+ to pass


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    n_bets: int
    n_wins: int
    hit_rate: float
    total_units: float
    roi_pct: float
    avg_edge: float
    avg_clv: float
    by_confidence: dict   # {'HIGH': {n, wins, roi}, 'MEDIUM': {...}, 'LOW': {...}}
    by_month: dict        # {'2024-04': {n, roi}, ...}
    calibration_ece: float
    passes_threshold: bool  # CLV >= +1.5% and hit rate >= 55% on MEDIUM+


# ---------------------------------------------------------------------------
# simulate_bet
# ---------------------------------------------------------------------------

def simulate_bet(
    features_row: dict[str, Any],
    model: HRModel,
    market_odds: int,
    actual_hr: int,
) -> Optional[dict[str, Any]]:
    """Simulate a single bet given a feature row and market odds.

    Parameters
    ----------
    features_row:
        Feature dict compatible with HRModel.predict_with_blend().
    model:
        Trained HRModel instance.
    market_odds:
        American-format opening odds.
    actual_hr:
        Ground truth (0 or 1).

    Returns
    -------
    dict with bet details, or None if edge is below MIN_EDGE.
    """
    market_implied = features_row.get("market_implied_prob", 0.0)

    try:
        prediction = model.predict_with_blend(features_row, float(market_implied))
    except (RuntimeError, ValueError) as exc:
        log.warning("simulate_bet: prediction failed — %s", exc)
        return None

    edge = prediction["edge"]
    confidence = classify_confidence(edge)

    if confidence is None:
        log.debug("simulate_bet: edge=%.4f below MIN_EDGE, skipping", edge)
        return None

    units = edge_tier_units(edge)
    outcome = "WIN" if actual_hr == 1 else "LOSS"
    pl = calculate_pl(outcome, market_odds, units)

    # CLV: we treat opening implied as our bet implied; closing implied
    # is stored on the row when available (else fall back to market_implied).
    close_implied = float(
        features_row.get("close_implied_prob", market_implied)
    )
    clv = calculate_clv(float(market_implied), close_implied)

    return {
        "edge": edge,
        "confidence": confidence,
        "units": units,
        "outcome": outcome,
        "pl": pl,
        "clv": clv,
        "model_prob": prediction["model_prob"],
        "final_prob": prediction["final_prob"],
        "market_implied": market_implied,
        "actual_hr": actual_hr,
        "game_date": features_row.get("game_date"),
    }


# ---------------------------------------------------------------------------
# _compute_backtest_result  (pure aggregation, reused by both backtest fns)
# ---------------------------------------------------------------------------

def _compute_backtest_result(bets: list[dict[str, Any]]) -> BacktestResult:
    """Aggregate a flat list of bet dicts into a BacktestResult."""
    if not bets:
        return BacktestResult(
            n_bets=0, n_wins=0, hit_rate=0.0, total_units=0.0,
            roi_pct=0.0, avg_edge=0.0, avg_clv=0.0,
            by_confidence={}, by_month={},
            calibration_ece=0.0, passes_threshold=False,
        )

    n_bets = len(bets)
    n_wins = sum(1 for b in bets if b["outcome"] == "WIN")
    hit_rate = n_wins / n_bets
    total_units = sum(b["units"] for b in bets)
    total_pl = sum(b["pl"] for b in bets)
    roi_pct = (total_pl / total_units * 100.0) if total_units > 0 else 0.0
    avg_edge = float(np.mean([b["edge"] for b in bets]))
    avg_clv = float(np.mean([b["clv"] for b in bets]))

    # --- by_confidence ---
    conf_buckets: dict[str, list[dict]] = {}
    for bet in bets:
        conf_buckets.setdefault(bet["confidence"], []).append(bet)

    by_confidence = {
        tier: {
            "n": len(bucket),
            "wins": sum(1 for b in bucket if b["outcome"] == "WIN"),
            "roi": (
                sum(b["pl"] for b in bucket)
                / sum(b["units"] for b in bucket)
                * 100.0
            ) if sum(b["units"] for b in bucket) > 0 else 0.0,
        }
        for tier, bucket in conf_buckets.items()
    }

    # --- by_month ---
    month_buckets: dict[str, list[dict]] = {}
    for bet in bets:
        gd = bet.get("game_date")
        month = str(gd)[:7] if gd else "unknown"
        month_buckets.setdefault(month, []).append(bet)

    by_month = {
        month: {
            "n": len(bucket),
            "roi": (
                sum(b["pl"] for b in bucket)
                / sum(b["units"] for b in bucket)
                * 100.0
            ) if sum(b["units"] for b in bucket) > 0 else 0.0,
        }
        for month, bucket in month_buckets.items()
    }

    # --- calibration ECE ---
    probs = np.array([b["final_prob"] for b in bets])
    actuals = np.array([b["actual_hr"] for b in bets], dtype=float)
    calibration_ece = _expected_calibration_error(probs, actuals)

    # --- passes_threshold ---
    medium_plus = [b for b in bets if b["confidence"] in ("HIGH", "MEDIUM")]
    mp_hit_rate = (
        sum(1 for b in medium_plus if b["outcome"] == "WIN") / len(medium_plus)
        if medium_plus else 0.0
    )
    passes_threshold = avg_clv >= CLV_THRESHOLD and mp_hit_rate >= HIT_RATE_THRESHOLD

    return BacktestResult(
        n_bets=n_bets,
        n_wins=n_wins,
        hit_rate=hit_rate,
        total_units=total_units,
        roi_pct=roi_pct,
        avg_edge=avg_edge,
        avg_clv=avg_clv,
        by_confidence=by_confidence,
        by_month=by_month,
        calibration_ece=calibration_ece,
        passes_threshold=passes_threshold,
    )


def _expected_calibration_error(
    probs: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE using equal-width probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(probs[mask].mean())
        bin_acc = float(actuals[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

def run_backtest(
    historical_rows: list[dict[str, Any]],
    model: Optional[HRModel] = None,
) -> BacktestResult:
    """Train on first 70% of rows (chronological), test on last 30%.

    Parameters
    ----------
    historical_rows:
        List of feature dicts from build_training_features(), each
        containing ``actual_hr``, ``market_odds``, and ``game_date``.
    model:
        Optional pre-trained HRModel. If None, a fresh model is trained.

    Returns
    -------
    BacktestResult aggregated over the test split.
    """
    sorted_rows = sorted(historical_rows, key=lambda r: str(r.get("game_date", "")))
    split = int(len(sorted_rows) * 0.70)
    train_rows = sorted_rows[:split]
    test_rows = sorted_rows[split:]

    log.info("run_backtest: %d train rows, %d test rows", len(train_rows), len(test_rows))

    if model is None:
        model = HRModel()
        model.train(train_rows)
        log.info("run_backtest: model trained on %d rows", len(train_rows))

    bets = _simulate_rows(test_rows, model)
    log.info("run_backtest: %d bets simulated from %d test rows", len(bets), len(test_rows))
    return _compute_backtest_result(bets)


# ---------------------------------------------------------------------------
# run_walk_forward_backtest
# ---------------------------------------------------------------------------

def run_walk_forward_backtest(
    historical_rows: list[dict[str, Any]],
    train_window_days: int = 365,
    step_days: int = 30,
) -> tuple[list[BacktestResult], BacktestResult]:
    """Walk-forward validation over historical data.

    Parameters
    ----------
    historical_rows:
        Full dataset with ``actual_hr``, ``market_odds``, ``game_date``.
    train_window_days:
        Number of days of history in each training window.
    step_days:
        Number of days to advance the cutoff per iteration.

    Returns
    -------
    Tuple of (per-window results list, combined BacktestResult).
    """
    sorted_rows = sorted(historical_rows, key=lambda r: str(r.get("game_date", "")))

    if not sorted_rows:
        empty = _compute_backtest_result([])
        return [], empty

    def _parse_date(row: dict) -> date:
        gd = row.get("game_date")
        if isinstance(gd, date):
            return gd
        return date.fromisoformat(str(gd)[:10])

    first_date = _parse_date(sorted_rows[0])
    last_date = _parse_date(sorted_rows[-1])

    cutoff = first_date + timedelta(days=train_window_days)
    window_results: list[BacktestResult] = []
    all_bets: list[dict[str, Any]] = []

    while cutoff < last_date:
        step_end = cutoff + timedelta(days=step_days)

        train_rows = [r for r in sorted_rows if _parse_date(r) < cutoff]
        test_rows = [
            r for r in sorted_rows
            if cutoff <= _parse_date(r) < step_end
        ]

        if len(train_rows) < 500 or not test_rows:
            log.debug(
                "walk_forward: skipping window ending %s — %d train, %d test",
                cutoff, len(train_rows), len(test_rows),
            )
            cutoff = step_end
            continue

        try:
            window_model = HRModel()
            window_model.train(train_rows)
        except ValueError as exc:
            log.warning("walk_forward: train failed for window %s — %s", cutoff, exc)
            cutoff = step_end
            continue

        bets = _simulate_rows(test_rows, window_model)
        all_bets.extend(bets)

        window_result = _compute_backtest_result(bets)
        window_results.append(window_result)
        log.info(
            "walk_forward: window %s→%s — %d bets, roi=%.1f%%",
            cutoff, step_end, window_result.n_bets, window_result.roi_pct,
        )

        cutoff = step_end

    combined = _compute_backtest_result(all_bets)
    log.info(
        "walk_forward: %d windows, combined %d bets, roi=%.1f%%",
        len(window_results), combined.n_bets, combined.roi_pct,
    )
    return window_results, combined


# ---------------------------------------------------------------------------
# _simulate_rows  (internal helper)
# ---------------------------------------------------------------------------

def _simulate_rows(
    rows: list[dict[str, Any]],
    model: HRModel,
) -> list[dict[str, Any]]:
    """Run simulate_bet on every row; collect non-None results."""
    bets: list[dict[str, Any]] = []
    for row in rows:
        market_odds = int(row.get("market_odds", -110))
        actual_hr = int(row.get("actual_hr", 0))
        result = simulate_bet(row, model, market_odds, actual_hr)
        if result is not None:
            bets.append(result)
    return bets


# ---------------------------------------------------------------------------
# print_backtest_report
# ---------------------------------------------------------------------------

def print_backtest_report(result: BacktestResult) -> str:
    """Format a BacktestResult as a readable text report.

    Returns the report string; does not print — callers should log as needed.
    """
    sep = "-" * 52
    lines = [
        sep,
        "HR MODEL BACKTEST REPORT",
        sep,
        f"  Bets          : {result.n_bets}",
        f"  Wins          : {result.n_wins}",
        f"  Hit Rate      : {result.hit_rate:.1%}",
        f"  Total Units   : {result.total_units:.2f}",
        f"  ROI           : {result.roi_pct:+.2f}%",
        f"  Avg Edge      : {result.avg_edge:+.3f}",
        f"  Avg CLV       : {result.avg_clv:+.2f}%",
        f"  Calibration   : ECE={result.calibration_ece:.4f}",
        f"  Passes Thresh : {'YES' if result.passes_threshold else 'NO'}",
        "",
        "By Confidence Tier:",
    ]

    for tier in ("HIGH", "MEDIUM", "LOW"):
        data = result.by_confidence.get(tier)
        if data:
            lines.append(
                f"  {tier:<8} n={data['n']:>4}  wins={data['wins']:>4}"
                f"  roi={data['roi']:+.2f}%"
            )

    lines.append("")
    lines.append("By Month:")
    for month in sorted(result.by_month):
        data = result.by_month[month]
        lines.append(
            f"  {month}  n={data['n']:>4}  roi={data['roi']:+.2f}%"
        )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# save_backtest_results
# ---------------------------------------------------------------------------

def save_backtest_results(
    result: BacktestResult,
    path: Optional[Path] = None,
) -> Path:
    """Serialise BacktestResult to JSON.

    Parameters
    ----------
    result:
        BacktestResult to persist.
    path:
        Destination path. Defaults to ~/mlb-props/backtest/hr_backtest.json.

    Returns
    -------
    Path where the file was written.
    """
    dest = Path(path) if path is not None else BACKTEST_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(result)
    dest.write_text(json.dumps(payload, indent=2, default=str))
    log.info("save_backtest_results: wrote %s", dest)
    return dest
