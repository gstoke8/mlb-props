#!/usr/bin/env python3
"""
Lineup Whiff Matchup Module

Computes a Synthetic Pitcher vs Lineup K-expectancy using pitch-type whiff
rates.  This is the strikeout analog to pitch_type_matchup.py for HR.

For each batter in the opposing lineup:
    For each pitch type in pitcher's mix:
        k_contribution += pitcher_mix[pitch] * batter_whiff_rate_vs[pitch]
    Expected_K_vs_batter = sum(k_contributions) * PA_WEIGHT_BY_SPOT[spot]
Total_expected_K = sum(Expected_K_vs_batter for batter in lineup)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from db import get_db
from pitch_type_matchup import get_pitcher_pitch_mix

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAGUE_AVG_WHIFF_BY_PITCH: Dict[str, float] = {
    "FF":    0.215,   # Four-seam fastball
    "SI":    0.155,   # Sinker
    "SL":    0.345,   # Slider
    "CU":    0.290,   # Curveball
    "CH":    0.340,   # Changeup
    "KC":    0.310,   # Knuckle-curve
    "FC":    0.220,   # Cutter
    "FS":    0.355,   # Splitter
    "OTHER": 0.240,
}

PA_WEIGHT_BY_SPOT: Dict[int, float] = {
    1: 3.20, 2: 3.10, 3: 3.00, 4: 2.90, 5: 2.80,
    6: 2.70, 7: 2.65, 8: 2.55, 9: 2.40,
}

LEAGUE_AVG_LINEUP_K_PER_6IP: float = 5.5

_MIN_PA_FOR_WHIFF = 10   # minimum PA to trust per-pitch-type whiff stats


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def get_batter_whiff_rates(
    batter_id: int,
    db,
    days: int = 60,  # noqa: ARG001 — reserved for future time-windowed queries
) -> Dict[str, float]:
    """Return the batter's whiff rate by pitch type.

    Queries ``pitch_type_performance`` for rows where player_type='batter'.
    Any pitch type not represented in the DB (or with insufficient PA) is
    filled with the corresponding ``LEAGUE_AVG_WHIFF_BY_PITCH`` value.

    Parameters
    ----------
    batter_id:
        MLB MLBAM player ID.
    db:
        An MLBPropsDB instance.
    days:
        Lookback window in calendar days (reserved; DB rows used as-is).

    Returns
    -------
    dict
        ``{pitch_type: whiff_rate}`` — rates between 0.0 and 1.0.
    """
    try:
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT pitch_type, whiff_rate, pa_or_pitches
                  FROM pitch_type_performance
                 WHERE player_id = ?
                   AND player_type = 'batter'
                """,
                (str(batter_id),),
            ).fetchall()
    except Exception as exc:
        log.warning(
            "DB query failed for batter whiff rates (batter_id=%s): %s",
            batter_id,
            exc,
        )
        rows = []

    db_rates: Dict[str, float] = {}
    for row in rows:
        pa = row["pa_or_pitches"] or 0
        if pa < _MIN_PA_FOR_WHIFF:
            continue
        if row["whiff_rate"] is not None:
            db_rates = {**db_rates, row["pitch_type"]: float(row["whiff_rate"])}

    # Fill any missing pitch types with league averages
    filled: Dict[str, float] = {
        pt: db_rates.get(pt, league_rate)
        for pt, league_rate in LEAGUE_AVG_WHIFF_BY_PITCH.items()
    }
    # Preserve any pitch types found in the DB that aren't in the constant map
    extra: Dict[str, float] = {
        pt: rate
        for pt, rate in db_rates.items()
        if pt not in filled
    }
    return {**filled, **extra}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_batter_k_expectancy(
    batter_id: int,
    lineup_spot: int,
    pitcher_mix: Dict[str, float],
    db,
    days: int = 60,
) -> Dict[str, Any]:
    """Compute expected K contribution for a single batter.

    Algorithm:
        whiff_rates = get_batter_whiff_rates(batter_id)
        expected_ks = sum(pitcher_mix[pt] * whiff_rates[pt] for pt in mix)
                      * PA_WEIGHT_BY_SPOT[lineup_spot]

    Parameters
    ----------
    batter_id:
        MLB MLBAM player ID.
    lineup_spot:
        Batting order position (1–9).
    pitcher_mix:
        ``{pitch_type: fraction}`` dict summing to 1.0.
    db:
        An MLBPropsDB instance.
    days:
        Lookback window for batter whiff-rate query.

    Returns
    -------
    dict
        ``{'batter_id': int, 'expected_ks': float, 'pa_weight': float}``
    """
    spot = max(1, min(9, lineup_spot))
    pa_weight = PA_WEIGHT_BY_SPOT[spot]

    whiff_rates = get_batter_whiff_rates(batter_id, db, days=days)

    raw_whiff_sum = sum(
        mix_frac * whiff_rates.get(pt, LEAGUE_AVG_WHIFF_BY_PITCH.get(pt, LEAGUE_AVG_WHIFF_BY_PITCH["OTHER"]))
        for pt, mix_frac in pitcher_mix.items()
        if mix_frac > 0.0
    )

    expected_ks = raw_whiff_sum * pa_weight

    return {
        "batter_id":   batter_id,
        "expected_ks": round(expected_ks, 4),
        "pa_weight":   pa_weight,
    }


def compute_lineup_k_matchup(
    pitcher_id: int,
    lineup: List[Dict[str, Any]],
    db,
    pitcher_days: int = 60,
    batter_days: int = 60,
) -> Dict[str, Any]:
    """Compute expected strikeouts vs a full lineup for a pitcher.

    Parameters
    ----------
    pitcher_id:
        MLB MLBAM player ID for the starting pitcher.
    lineup:
        List of dicts, each containing ``batter_id``, ``lineup_spot``, and
        ``batter_hand``.
    db:
        An MLBPropsDB instance.
    pitcher_days:
        Lookback window for pitcher pitch-mix query.
    batter_days:
        Lookback window for batter whiff-rate queries.

    Returns
    -------
    dict
        {
            'expected_lineup_ks': float,
            'matchup_factor':     float,   # expected / LEAGUE_AVG_LINEUP_K_PER_6IP
            'batter_details':     list,    # per-batter breakdown
            'data_quality':       str,     # 'full' | 'partial' | 'fallback'
            'pitcher_mix':        dict,
        }
    """
    pitcher_mix = get_pitcher_pitch_mix(pitcher_id, db, days=pitcher_days)
    has_pitcher_data = bool(pitcher_mix)

    if not has_pitcher_data:
        pitch_keys = [k for k in LEAGUE_AVG_WHIFF_BY_PITCH if k != "OTHER"]
        uniform = 1.0 / len(pitch_keys)
        pitcher_mix = {k: uniform for k in pitch_keys}
        log.debug(
            "No pitcher mix found for pitcher_id=%s; using uniform fallback",
            pitcher_id,
        )

    batter_details: List[Dict[str, Any]] = []
    fallback_batter_count = 0

    for slot in lineup:
        batter_id  = slot.get("batter_id")
        lineup_spot = slot.get("lineup_spot", 9)

        if batter_id is None:
            log.warning("Skipping lineup slot with missing batter_id: %s", slot)
            continue

        try:
            detail = compute_batter_k_expectancy(
                batter_id, lineup_spot, pitcher_mix, db, days=batter_days
            )
        except Exception as exc:
            log.error(
                "compute_batter_k_expectancy failed for batter=%s pitcher=%s: %s",
                batter_id,
                pitcher_id,
                exc,
            )
            spot = max(1, min(9, lineup_spot))
            detail = {
                "batter_id":   batter_id,
                "expected_ks": LEAGUE_AVG_LINEUP_K_PER_6IP / 9,
                "pa_weight":   PA_WEIGHT_BY_SPOT[spot],
            }
            fallback_batter_count += 1

        batter_details = [*batter_details, {**slot, **detail}]

    expected_lineup_ks = sum(d["expected_ks"] for d in batter_details)
    matchup_factor = (
        expected_lineup_ks / LEAGUE_AVG_LINEUP_K_PER_6IP
        if LEAGUE_AVG_LINEUP_K_PER_6IP > 0
        else 1.0
    )

    has_batter_data = fallback_batter_count < len(batter_details)
    if has_pitcher_data and has_batter_data and fallback_batter_count == 0:
        data_quality = "full"
    elif has_pitcher_data or has_batter_data:
        data_quality = "partial"
    else:
        data_quality = "fallback"

    return {
        "expected_lineup_ks": round(expected_lineup_ks, 4),
        "matchup_factor":     round(matchup_factor, 4),
        "batter_details":     batter_details,
        "data_quality":       data_quality,
        "pitcher_mix":        pitcher_mix,
    }


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def batch_compute_lineup_matchups(
    games: List[Dict[str, Any]],
    db=None,
) -> List[Dict[str, Any]]:
    """Compute lineup K matchups for a list of games.

    Parameters
    ----------
    games:
        List of dicts, each containing ``pitcher_id``, ``game_pk``, and
        ``lineup`` (a list of ``{batter_id, lineup_spot, batter_hand}`` dicts).
    db:
        An MLBPropsDB instance.  If None, one is created via ``get_db()``.

    Returns
    -------
    list[dict]
        Input game dicts augmented with the keys returned by
        ``compute_lineup_k_matchup``.
    """
    database = db if db is not None else get_db()
    results: List[Dict[str, Any]] = []
    fallback_count = 0

    for game in games:
        pitcher_id = game.get("pitcher_id")
        game_pk    = game.get("game_pk")
        lineup     = game.get("lineup", [])

        if pitcher_id is None:
            log.warning("Skipping game with missing pitcher_id: %s", game)
            continue

        try:
            matchup = compute_lineup_k_matchup(pitcher_id, lineup, database)
        except Exception as exc:
            log.error(
                "compute_lineup_k_matchup failed for pitcher=%s game=%s: %s",
                pitcher_id,
                game_pk,
                exc,
            )
            matchup = {
                "expected_lineup_ks": LEAGUE_AVG_LINEUP_K_PER_6IP,
                "matchup_factor":     1.0,
                "batter_details":     [],
                "data_quality":       "fallback",
                "pitcher_mix":        {},
            }

        if matchup["data_quality"] == "fallback":
            fallback_count += 1

        results = [*results, {**game, **matchup}]

    log.info(
        "batch_compute_lineup_matchups: %d games processed, %d used fallback data",
        len(results),
        fallback_count,
    )
    return results
