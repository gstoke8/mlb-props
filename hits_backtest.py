from __future__ import annotations
#!/usr/bin/env python3
"""
Hits Backtest — Evaluates the H1.5 model against historical data.

Computes CLV, hit rate by confidence tier, and ROI using both a simple
70/30 chronological split and walk-forward validation.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from math import ceil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from db import calculate_clv, calculate_pl, get_db
from hits_features import build_hits_training_features
from hits_model import HitsModel, get_model
from sizing import MIN_EDGE, classify_confidence, compute_units, edge_tier_units

log = logging.getLogger(__name__)

BACKTEST_DIR = Path.home() / "mlb-props" / "backtest"
BACKTEST_PATH = BACKTEST_DIR / "hits_backtest.json"
CLV_THRESHOLD = 1.5
HIT_RATE_THRESHOLD = 0.55
MIN_TRAIN_ROWS = 500


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
    by_confidence: dict
    by_month: dict
    calibration_ece: float
    passes_threshold: bool


# ---------------------------------------------------------------------------
# simulate_hits_bet
# ---------------------------------------------------------------------------

def simulate_hits_bet(
    features_row: dict[str, Any],
    model: HitsModel,
    market_odds: int,
    actual_hits: int,
) -> Optional[dict[str, Any]]:
    """Simulate a single H1.5 bet. Returns None if edge is below MIN_EDGE."""
    market_implied = features_row.get("market_implied_prob", 0.0)

    try:
        prediction = model.predict_with_blend(features_row, float(market_implied))
    except (RuntimeError, ValueError) as exc:
        log.warning("simulate_hits_bet: prediction failed — %s", exc)
        return None

    edge = prediction["edge"]
    confidence = classify_confidence(edge)
    if confidence is None:
        log.debug("simulate_hits_bet: edge=%.4f below MIN_EDGE, skipping", edge)
        return None

    market_line = float(features_row.get("market_line", 0.5))
    outcome = "WIN" if actual_hits >= int(ceil(market_line)) else "LOSS"
    units = edge_tier_units(edge)
    pl = calculate_pl(outcome, market_odds, units)

    close_implied = float(features_row.get("close_implied_prob", market_implied))
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
        "actual_hits": actual_hits,
        "game_date": features_row.get("game_date"),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _bucket_stats(bets: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for bet in bets:
        buckets.setdefault(bet[key], []).append(bet)
    return {
        k: {
            "n": len(v),
            "wins": sum(1 for b in v if b["outcome"] == "WIN"),
            "roi": (
                sum(b["pl"] for b in v) / sum(b["units"] for b in v) * 100.0
            ) if sum(b["units"] for b in v) > 0 else 0.0,
        }
        for k, v in buckets.items()
    }


def _expected_calibration_error(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(float(probs[mask].mean()) - float(actuals[mask].mean()))
    return float(ece)


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
    total_units = sum(b["units"] for b in bets)
    total_pl = sum(b["pl"] for b in bets)
    avg_clv = float(np.mean([b["clv"] for b in bets]))

    probs = np.array([b["final_prob"] for b in bets])
    actuals = np.array([1 if b["outcome"] == "WIN" else 0 for b in bets], dtype=float)
    calibration_ece = _expected_calibration_error(probs, actuals)

    medium_plus = [b for b in bets if b["confidence"] in ("HIGH", "MEDIUM")]
    mp_hit_rate = (
        sum(1 for b in medium_plus if b["outcome"] == "WIN") / len(medium_plus)
        if medium_plus else 0.0
    )

    # by_month keyed on YYYY-MM prefix of game_date
    month_bets = [
        {**b, "_month": str(b.get("game_date") or "unknown")[:7]}
        for b in bets
    ]

    return BacktestResult(
        n_bets=n_bets,
        n_wins=n_wins,
        hit_rate=n_wins / n_bets,
        total_units=total_units,
        roi_pct=(total_pl / total_units * 100.0) if total_units > 0 else 0.0,
        avg_edge=float(np.mean([b["edge"] for b in bets])),
        avg_clv=avg_clv,
        by_confidence=_bucket_stats(bets, "confidence"),
        by_month={
            k: {"n": v["n"], "roi": v["roi"]}
            for k, v in _bucket_stats(month_bets, "_month").items()
        },
        calibration_ece=calibration_ece,
        passes_threshold=avg_clv >= CLV_THRESHOLD and mp_hit_rate >= HIT_RATE_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

def run_backtest(
    historical_rows: list[dict[str, Any]],
    model: Optional[HitsModel] = None,
) -> BacktestResult:
    """Train on first 70 % of rows (chronological), test on last 30 %."""
    sorted_rows = sorted(historical_rows, key=lambda r: str(r.get("game_date", "")))
    split = int(len(sorted_rows) * 0.70)
    train_rows, test_rows = sorted_rows[:split], sorted_rows[split:]

    log.info("run_backtest: %d train, %d test", len(train_rows), len(test_rows))

    if model is None:
        model = HitsModel()
        model.train(train_rows)

    bets = _simulate_rows(test_rows, model)
    log.info("run_backtest: %d bets from %d test rows", len(bets), len(test_rows))
    return _compute_backtest_result(bets)


# ---------------------------------------------------------------------------
# run_walk_forward_backtest
# ---------------------------------------------------------------------------

def run_walk_forward_backtest(
    historical_rows: list[dict[str, Any]],
    train_window_days: int = 365,
    step_days: int = 30,
) -> tuple[list[BacktestResult], BacktestResult]:
    """Walk-forward validation. Returns (per-window results, combined result)."""
    sorted_rows = sorted(historical_rows, key=lambda r: str(r.get("game_date", "")))
    if not sorted_rows:
        return [], _compute_backtest_result([])

    def _parse_date(row: dict) -> date:
        gd = row.get("game_date")
        return gd if isinstance(gd, date) else date.fromisoformat(str(gd)[:10])

    first_date = _parse_date(sorted_rows[0])
    last_date = _parse_date(sorted_rows[-1])
    cutoff = first_date + timedelta(days=train_window_days)

    window_results: list[BacktestResult] = []
    all_bets: list[dict[str, Any]] = []

    while cutoff < last_date:
        step_end = cutoff + timedelta(days=step_days)
        train_rows = [r for r in sorted_rows if _parse_date(r) < cutoff]
        test_rows = [r for r in sorted_rows if cutoff <= _parse_date(r) < step_end]

        if len(train_rows) < MIN_TRAIN_ROWS or not test_rows:
            log.debug("walk_forward: skipping window %s (%d train, %d test)", cutoff, len(train_rows), len(test_rows))
            cutoff = step_end
            continue

        try:
            window_model = HitsModel()
            window_model.train(train_rows)
        except ValueError as exc:
            log.warning("walk_forward: train failed window %s — %s", cutoff, exc)
            cutoff = step_end
            continue

        bets = _simulate_rows(test_rows, window_model)
        all_bets.extend(bets)
        wr = _compute_backtest_result(bets)
        window_results.append(wr)
        log.info("walk_forward: %s→%s — %d bets roi=%.1f%%", cutoff, step_end, wr.n_bets, wr.roi_pct)
        cutoff = step_end

    combined = _compute_backtest_result(all_bets)
    log.info("walk_forward: %d windows, %d bets, roi=%.1f%%", len(window_results), combined.n_bets, combined.roi_pct)
    return window_results, combined


# ---------------------------------------------------------------------------
# _simulate_rows
# ---------------------------------------------------------------------------

def _simulate_rows(rows: list[dict[str, Any]], model: HitsModel) -> list[dict[str, Any]]:
    bets: list[dict[str, Any]] = []
    for row in rows:
        result = simulate_hits_bet(row, model, int(row.get("market_odds", -110)), int(row.get("actual_hits", 0)))
        if result is not None:
            bets.append(result)
    return bets


# ---------------------------------------------------------------------------
# save_backtest_results
# ---------------------------------------------------------------------------

def save_backtest_results(
    result: BacktestResult,
    path: Optional[Path] = None,
) -> Path:
    """Serialise BacktestResult to ~/mlb-props/backtest/hits_backtest.json (default)."""
    dest = Path(path) if path is not None else BACKTEST_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(asdict(result), indent=2, default=str))
    log.info("save_backtest_results: wrote %s", dest)
    return dest
