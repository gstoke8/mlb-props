#!/usr/bin/env python3
"""K Model Backtest — chronological and walk-forward validation."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from db import calculate_clv, calculate_pl, get_db
from k_features import build_k_training_features
from k_model import KModel, get_model
from sizing import classify_confidence, edge_tier_units

logger = logging.getLogger(__name__)

BACKTEST_DIR = Path.home() / "mlb-props" / "backtest"
DEFAULT_BACKTEST_PATH = BACKTEST_DIR / "k_backtest.json"


@dataclass
class BacktestResult:
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    total_pl_units: float = 0.0
    roi_pct: float = 0.0
    avg_edge: float = 0.0
    avg_odds: float = 0.0
    win_rate: float = 0.0
    clv_avg: float = 0.0
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)
    bet_log: list[dict[str, Any]] = field(default_factory=list)
    train_n: int = 0
    test_n: int = 0
    run_type: str = "standard"


def simulate_k_bet(
    features_row: dict[str, Any],
    model: KModel,
    market_line: float,
    market_odds_over: int,
    actual_ks: float,
) -> dict[str, Any] | None:
    """Simulate a single K over/under bet; returns None when edge is below threshold."""
    try:
        prediction = model.predict_with_blend(features_row)
    except (RuntimeError, ValueError) as exc:
        logger.warning("predict_with_blend failed: %s", exc)
        return None

    over_prob = prediction["over_prob"]
    market_implied = prediction["market_implied"]
    edge = prediction["edge"]

    confidence = classify_confidence(edge)
    if confidence is None:
        return None

    units = edge_tier_units(edge)
    if units == 0.0:
        return None

    outcome_str = "win" if actual_ks > market_line else "loss"
    pl = calculate_pl(outcome_str, market_odds_over, units)
    close_implied = features_row.get("closing_implied_prob", market_implied)
    clv = calculate_clv(over_prob, close_implied)

    return {
        "over_prob": round(over_prob, 4),
        "market_implied": round(market_implied, 4),
        "edge": round(edge, 4),
        "confidence": confidence,
        "units": units,
        "market_line": market_line,
        "market_odds_over": market_odds_over,
        "actual_ks": actual_ks,
        "outcome": outcome_str,
        "pl_units": round(pl, 4),
        "clv": round(clv, 4),
    }


def _aggregate_bets(
    bet_log: list[dict[str, Any]],
    train_n: int,
    test_n: int,
    run_type: str,
) -> BacktestResult:
    if not bet_log:
        return BacktestResult(train_n=train_n, test_n=test_n, run_type=run_type)

    total_bets = len(bet_log)
    winning_bets = sum(1 for b in bet_log if b["outcome"] == "win")
    total_pl = sum(b["pl_units"] for b in bet_log)
    total_wagered = sum(b["units"] for b in bet_log)
    roi_pct = (total_pl / total_wagered * 100) if total_wagered > 0 else 0.0

    confidence_counts: dict[str, dict[str, Any]] = {}
    for b in bet_log:
        conf = b["confidence"]
        entry = confidence_counts.get(conf, {"bets": 0, "pl": 0.0, "wins": 0})
        confidence_counts[conf] = {
            "bets": entry["bets"] + 1,
            "pl": round(entry["pl"] + b["pl_units"], 4),
            "wins": entry["wins"] + (1 if b["outcome"] == "win" else 0),
        }

    return BacktestResult(
        total_bets=total_bets,
        winning_bets=winning_bets,
        losing_bets=total_bets - winning_bets,
        total_pl_units=round(total_pl, 4),
        roi_pct=round(roi_pct, 2),
        avg_edge=round(float(np.mean([b["edge"] for b in bet_log])), 4),
        avg_odds=round(float(np.mean([b["market_odds_over"] for b in bet_log])), 2),
        win_rate=round(winning_bets / total_bets, 4),
        clv_avg=round(float(np.mean([b["clv"] for b in bet_log])), 4),
        confidence_breakdown=confidence_counts,
        bet_log=bet_log,
        train_n=train_n,
        test_n=test_n,
        run_type=run_type,
    )


def run_backtest(
    historical_rows: list[dict[str, Any]],
    model: KModel | None = None,
) -> BacktestResult:
    """Chronological 70/30 train/test backtest of the K model."""
    if not historical_rows:
        raise ValueError("historical_rows must not be empty")

    split_idx = int(len(historical_rows) * 0.70)
    train_rows = historical_rows[:split_idx]
    test_rows = historical_rows[split_idx:]

    logger.info("K backtest split — train_n=%d, test_n=%d", len(train_rows), len(test_rows))

    active_model = model
    if active_model is None:
        active_model = KModel()
        active_model.train(train_rows)

    bet_log = [
        bet
        for row in test_rows
        for bet in [
            simulate_k_bet(
                features_row=row,
                model=active_model,
                market_line=float(row["market_line"]),
                market_odds_over=int(row["market_odds_over"]),
                actual_ks=float(row["actual_ks"]),
            )
        ]
        if bet is not None
    ]

    logger.info("K backtest complete — %d bets from %d test rows", len(bet_log), len(test_rows))
    return _aggregate_bets(bet_log, len(train_rows), len(test_rows), "standard")


def run_walk_forward_backtest(
    historical_rows: list[dict[str, Any]],
    train_window_days: int = 365,
    step_days: int = 30,
) -> BacktestResult:
    """Walk-forward validation with a rolling training window."""
    if not historical_rows:
        raise ValueError("historical_rows must not be empty")

    all_bets: list[dict[str, Any]] = []
    total_train_n = 0
    total_test_n = 0

    dates = [datetime.fromisoformat(r["game_date"]).date() for r in historical_rows]
    min_date = min(dates)
    max_date = max(dates)
    step = timedelta(days=step_days)

    window_start = min_date
    window_end = window_start + timedelta(days=train_window_days)

    while window_end < max_date:
        test_end = window_end + step
        train_rows = [r for r, d in zip(historical_rows, dates) if window_start <= d < window_end]
        test_rows = [r for r, d in zip(historical_rows, dates) if window_end <= d < test_end]

        if len(train_rows) < 50 or not test_rows:
            window_start += step
            window_end += step
            continue

        try:
            fold_model = KModel()
            fold_model.train(train_rows)
        except ValueError as exc:
            logger.warning("Walk-forward fold skipped (train_n=%d): %s", len(train_rows), exc)
            window_start += step
            window_end += step
            continue

        total_train_n += len(train_rows)
        total_test_n += len(test_rows)

        for row in test_rows:
            bet = simulate_k_bet(
                features_row=row,
                model=fold_model,
                market_line=float(row["market_line"]),
                market_odds_over=int(row["market_odds_over"]),
                actual_ks=float(row["actual_ks"]),
            )
            if bet is not None:
                all_bets.append({**bet, "fold_start": window_start.isoformat()})

        logger.debug(
            "Fold %s–%s — %d train, %d test", window_start, window_end,
            len(train_rows), len(test_rows),
        )
        window_start += step
        window_end += step

    logger.info(
        "Walk-forward complete — %d total bets across %d test rows",
        len(all_bets), total_test_n,
    )
    return _aggregate_bets(all_bets, total_train_n, total_test_n, "walk_forward")


def print_backtest_report(result: BacktestResult) -> str:
    """Return a formatted backtest summary string."""
    lines = [
        "=" * 50,
        f"K Model Backtest Report  [{result.run_type}]",
        "=" * 50,
        f"Train rows   : {result.train_n}",
        f"Test rows    : {result.test_n}",
        f"Total bets   : {result.total_bets}",
        f"Win / Loss   : {result.winning_bets} / {result.losing_bets}",
        f"Win rate     : {result.win_rate:.1%}",
        f"Total P/L    : {result.total_pl_units:+.2f} units",
        f"ROI          : {result.roi_pct:+.2f}%",
        f"Avg edge     : {result.avg_edge:.4f}",
        f"Avg odds     : {result.avg_odds:.1f}",
        f"CLV avg      : {result.clv_avg:+.4f}",
        "",
        "Confidence breakdown:",
    ]
    for conf, stats in sorted(result.confidence_breakdown.items()):
        wr = stats["wins"] / stats["bets"] if stats["bets"] else 0.0
        lines.append(
            f"  {conf:6s}: {stats['bets']:3d} bets  WR={wr:.1%}  P/L={stats['pl']:+.2f}u"
        )
    lines.append("=" * 50)
    report = "\n".join(lines)
    logger.info("Backtest report:\n%s", report)
    return report


def save_backtest_results(
    result: BacktestResult,
    path: str | Path | None = None,
) -> Path:
    """Serialize BacktestResult to JSON at ~/mlb-props/backtest/k_backtest.json."""
    out_path = Path(path) if path else DEFAULT_BACKTEST_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text(json.dumps(asdict(result), indent=2, default=str))
        logger.info("K backtest results saved to %s", out_path)
    except OSError as exc:
        logger.error("Failed to save backtest results to %s: %s", out_path, exc)
        raise
    return out_path
