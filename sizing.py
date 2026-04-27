#!/usr/bin/env python3
"""
MLB Props Sizing Module

Fractional Kelly criterion sizing with bankroll management and per-bet unit
calculation for MLB player prop bets.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Tuple

from db import get_db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

KELLY_FRACTION = 0.25
MAX_BET_PCT = 0.02          # 2% of bankroll per bet
MIN_UNITS = 0.25
EDGE_TIERS: list[tuple[float, float]] = [
    (0.06, 1.50),   # HIGH:   edge > 6%    → 1.5u
    (0.03, 0.75),   # MEDIUM: edge 3–6%    → 0.75u
    (0.015, 0.25),  # LOW:    edge 1.5–3%  → 0.25u
]
MIN_EDGE = 0.015
PAPER_MIN_ODDS = -110       # paper mode: min American odds to qualify (filters juiced locks)
CORRELATION_CAP = 3         # max bets per game per day


# ---------------------------------------------------------------------------
# Core sizing functions
# ---------------------------------------------------------------------------

def kelly_fraction(edge: float, odds_american: int) -> float:
    """Compute fractional Kelly stake as a fraction of bankroll.

    Parameters
    ----------
    edge:
        Model edge over the market (e.g. 0.05 for 5%).
    odds_american:
        American-format odds (e.g. -110, +150).

    Returns
    -------
    float
        Fractional Kelly stake in [0, 1].  Zero when Kelly is negative.
    """
    if odds_american >= 0:
        b = (odds_american + 100) / 100 - 1
    else:
        b = 100 / abs(odds_american)

    implied_prob = 100 / (abs(odds_american) + 100) if odds_american < 0 else 100 / (odds_american + 100)
    p = edge + implied_prob
    q = 1.0 - p

    if b <= 0:
        log.warning("kelly_fraction: non-positive b=%.4f for odds=%d", b, odds_american)
        return 0.0

    full_kelly = (b * p - q) / b
    result = max(0.0, full_kelly * KELLY_FRACTION)
    log.debug(
        "kelly_fraction: edge=%.4f odds=%d b=%.4f p=%.4f full_kelly=%.4f result=%.4f",
        edge, odds_american, b, p, full_kelly, result,
    )
    return result


def edge_tier_units(edge: float) -> float:
    """Return the unit size for a given edge based on EDGE_TIERS.

    Parameters
    ----------
    edge:
        Model edge over the market.

    Returns
    -------
    float
        Units to bet (0.0 signals the bet should be skipped).
    """
    if edge < MIN_EDGE:
        log.debug("edge_tier_units: edge=%.4f below MIN_EDGE=%.4f, skip", edge, MIN_EDGE)
        return 0.0

    for threshold, units in EDGE_TIERS:
        if edge >= threshold:
            log.debug("edge_tier_units: edge=%.4f → %.2f units", edge, units)
            return units

    log.debug("edge_tier_units: edge=%.4f matched no tier, skip", edge)
    return 0.0


def compute_units(
    edge: float,
    odds_american: int,
    bankroll: float,
    method: str = "tier",
) -> float:
    """Compute the number of units to bet.

    Parameters
    ----------
    edge:
        Model edge over the market.
    odds_american:
        American-format odds.
    bankroll:
        Current bankroll in USD.
    method:
        ``'tier'`` uses ``edge_tier_units``; ``'kelly'`` uses fractional Kelly.

    Returns
    -------
    float
        Clamped unit count in [MIN_UNITS, max_units].  Returns 0.0 when the
        bet should be skipped entirely (edge below MIN_EDGE).
    """
    if edge < MIN_EDGE:
        return 0.0

    unit_size = bankroll / 100.0
    max_units = (bankroll * MAX_BET_PCT) / unit_size  # = MAX_BET_PCT * 100

    if method == "kelly":
        kelly_frac = kelly_fraction(edge, odds_american)
        raw_units = (kelly_frac * bankroll) / unit_size
    else:
        raw_units = edge_tier_units(edge)
        if raw_units == 0.0:
            return 0.0

    clamped = max(MIN_UNITS, min(raw_units, max_units))
    log.debug(
        "compute_units: method=%s edge=%.4f odds=%d raw=%.4f clamped=%.4f",
        method, edge, odds_american, raw_units, clamped,
    )
    return clamped


def classify_confidence(edge: float) -> Optional[str]:
    """Classify an edge value into a confidence label.

    Parameters
    ----------
    edge:
        Model edge over the market.

    Returns
    -------
    str or None
        ``'HIGH'``, ``'MEDIUM'``, ``'LOW'``, or ``None`` (skip).
    """
    labels = ["HIGH", "MEDIUM", "LOW"]
    for (threshold, _), label in zip(EDGE_TIERS, labels):
        if edge >= threshold:
            return label
    return None


# ---------------------------------------------------------------------------
# Bankroll-protection checks
# ---------------------------------------------------------------------------

def check_daily_exposure(db, game_pk: int) -> bool:
    """Check whether the correlation cap has been reached for a game today.

    Parameters
    ----------
    db:
        ``MLBPropsDB`` instance.
    game_pk:
        MLB Stats API game primary key.

    Returns
    -------
    bool
        ``True`` when a new bet is allowed; ``False`` when the cap is hit.
    """
    today = date.today().isoformat()
    try:
        bets = db.get_bets(date=today)
        count = sum(1 for b in bets if b.get("game_pk") == game_pk)
        if count >= CORRELATION_CAP:
            log.info(
                "check_daily_exposure: game_pk=%d already has %d bets today (cap=%d)",
                game_pk, count, CORRELATION_CAP,
            )
            return False
        return True
    except Exception as exc:
        log.error("check_daily_exposure: DB query failed: %s", exc)
        raise RuntimeError(f"Failed to query daily exposure for game_pk={game_pk}") from exc


def check_daily_stop_loss(db) -> bool:
    """Check whether the daily stop-loss threshold has been breached.

    Considers only resolved bets (those with a recorded ``pl_units``) placed
    today.

    Parameters
    ----------
    db:
        ``MLBPropsDB`` instance.

    Returns
    -------
    bool
        ``True`` when betting can continue; ``False`` when stop-loss is hit.
    """
    today = date.today().isoformat()
    try:
        bets = db.get_bets(date=today)
        daily_pl = sum(
            b["pl_units"]
            for b in bets
            if b.get("pl_units") is not None
        )
        if daily_pl <= DAILY_STOP_LOSS:
            log.warning(
                "check_daily_stop_loss: daily P/L=%.2f units hit stop-loss threshold=%.2f",
                daily_pl, DAILY_STOP_LOSS,
            )
            return False
        return True
    except Exception as exc:
        log.error("check_daily_stop_loss: DB query failed: %s", exc)
        raise RuntimeError("Failed to query daily P/L for stop-loss check") from exc


# ---------------------------------------------------------------------------
# Primary decision gate
# ---------------------------------------------------------------------------

def should_bet(
    edge: float,
    odds_american: int,
    game_pk: int,
    db=None,
) -> Tuple[bool, str]:
    """Determine whether a bet should be placed.

    Combines edge threshold, daily stop-loss, and per-game correlation cap.

    Parameters
    ----------
    edge:
        Model edge over the market.
    odds_american:
        American-format odds.
    game_pk:
        MLB Stats API game primary key.
    db:
        ``MLBPropsDB`` instance; obtained from ``get_db()`` when ``None``.

    Returns
    -------
    tuple[bool, str]
        ``(True, 'OK')`` when the bet should proceed, or
        ``(False, <reason>)`` when it should be skipped.
    """
    if edge < MIN_EDGE:
        reason = f"edge {edge:.4f} below MIN_EDGE {MIN_EDGE}"
        log.debug("should_bet: skip — %s", reason)
        return False, reason

    resolved_db = db if db is not None else get_db()

    try:
        exposure_ok = check_daily_exposure(resolved_db, game_pk)
    except RuntimeError as exc:
        return False, str(exc)

    if not exposure_ok:
        reason = f"correlation cap ({CORRELATION_CAP}) reached for game_pk={game_pk}"
        log.info("should_bet: skip — %s", reason)
        return False, reason

    log.debug(
        "should_bet: PASS edge=%.4f odds=%d game_pk=%d",
        edge, odds_american, game_pk,
    )
    return True, "OK"
